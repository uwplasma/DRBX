"""tau p_i stage 2: the short-leg wall implicit handoff, exercised on selected walls.

With ``polarization_variable="phi_plus_tau_pi"`` (``psi = phi + tau*n*Ti``) the local
backward-Euler handoff uses the force ``mu*tau*G(n*Ti)`` and the Jacobian entries
``[4, 2] = mu*tau*w_Ti*n`` and ``[4, 0] = mu*tau*w_n*Ti``; the electron-force wall diagnostic
subtracts ``mu*tau*G(n*Ti)`` on selected short walls; ``psi`` is threaded into the characteristic
wall data.  The existing short-leg tests use geometry with no short-leg wall selected.

Two layers:

* *kernel layer* (no geometry): :func:`parallel_short_wall_backward_euler` and friends on
  synthetic wall states, with the wall flags set by hand, for both selectors;
* *model layer*: ``LocalFciDrbEBRhs.apply_short_leg_implicit_material_step`` on the small
  shifted-torus fixture with a radial field-line tilt, so 24 backward and 24 forward legs end
  on the radial walls (48 of 96 cells selected; checked by every test that uses it).

Scope of the Jacobian checks (a pre-existing property, not a stage-2 regression): the
material Jacobian ``-A_plus/dx`` / ``+A_minus/dx`` is the *frozen* local linearization (face
matrix, eigensystem and wall state held fixed), so its independent reference is the explicit
characteristic split built here with NumPy.  The coupled ``[4, 0]``/``[4, 2]`` entries are the
frozen three-point-stencil centre derivative of ``mu*tau*G``; they are *not* the exact
derivative of the production support-core force (see the non-strict ``xfail`` test), so the exact
derivative (``jax.jvp`` of the model force) is compared only through the chain rule
``d(n*Ti) = Ti dn + n dTi``, which both the force and the claimed entries must obey.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.geometry.fci_geometry import (  # noqa: E402
    FciMaps3D,
    build_fci_maps_from_b_contravariant,
)
from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_parallel_production_flux import (  # noqa: E402
    parallel_characteristic_wall_data,
    parallel_production_principal_matrix,
    parallel_short_wall_backward_euler,
    parallel_short_wall_material_data,
)
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_local_fci_geometry,
    build_local_fci_geometries,
)
from fci_drb_eb_test_helpers import _build_rhs, _context_and_sharded_inputs  # noqa: E402
from shifted_torus_4field_mms_helpers import (  # noqa: E402
    build_shifted_torus_4field_geometry,
    c_phi,
    iota,
)

LEGACY = "phi_plus_tau_ti"
PI = "phi_plus_tau_pi"
SELECTORS = ("all-physical-walls", "cfl")
CFL_LIMIT = 2.785  # default ``parallel_short_leg_cfl_limit``

# ======================================================================================
# kernel layer: synthetic wall states
# ======================================================================================

TAU = 0.7
MU = 100.0
NROWS = 8


def _hand_matrix(state, tau, mu, psi):
    """Five-field parallel principal matrix, transcribed by hand from the equations."""
    n, Te, Ti, Vi, Ve = (float(x) for x in state)
    dV = Vi - Ve
    A = np.zeros((5, 5))
    A[0] = [Ve, 0.0, 0.0, 0.0, n]  # n_t + Ve n_l + n Ve_l
    A[1] = [-1.42 * Te * dV / (3 * n), Ve, 0.0, -1.42 * Te / 3, 3.42 * Te / 3]
    A[2] = [-2.0 * Ti * dV / (3 * n), 0.0, Vi, 0.0, 2.0 * Ti / 3]
    A[3] = [(Te + tau * Ti) / n, 1.0, tau, Vi, 0.0]  # ion momentum, p_i = n Ti
    # electron momentum: mu*G(psi) with psi = phi + tau*Ti (legacy) or phi + tau*n*Ti (p_i)
    A[4] = [mu * Te / n, 1.71 * mu, mu * tau, 0.0, Ve]
    if psi == PI:
        A[4, 0] += mu * tau * Ti  # dpsi/dn = tau*Ti
        A[4, 2] = mu * tau * n  # dpsi/dTi = tau*n
    return A


def _split(A, sign, min_abs=1e-3):
    """``A @ P`` for the projector onto positive (sign=+1) or negative (sign=-1) eigenvalues."""
    w, V = np.linalg.eig(A)
    assert np.max(np.abs(w.imag)) < 1e-9 * np.max(np.abs(w.real))
    assert np.min(np.abs(w.real)) > min_abs, "keep the states away from stationary modes"
    w = w.real
    V = V.real
    keep = (w * sign) > 0
    return V @ np.diag(np.where(keep, w, 0.0)) @ np.linalg.inv(V), np.max(np.abs(w))


def _wall_rows(seed=3):
    """Synthetic (center, minus, plus, dx_minus, dx_plus) and hand-set wall flags."""
    rng = np.random.default_rng(seed)

    def draw(scale=1.0):
        return np.stack(
            (
                rng.uniform(0.8, 2.0, NROWS) * scale,  # n
                rng.uniform(0.6, 1.5, NROWS) * scale,  # Te
                rng.uniform(0.5, 1.5, NROWS) * scale,  # Ti
                rng.uniform(-0.3, 0.3, NROWS),  # Vi
                rng.uniform(-0.3, 0.3, NROWS),  # Ve
            ),
            axis=-1,
        )

    center = draw()
    minus = center * (1 + 0.1 * rng.standard_normal(center.shape))
    plus = center * (1 + 0.1 * rng.standard_normal(center.shape))
    minus[:, 3:] = center[:, 3:] + 0.05 * rng.standard_normal((NROWS, 2))
    plus[:, 3:] = center[:, 3:] + 0.05 * rng.standard_normal((NROWS, 2))
    dxm = rng.uniform(0.3, 1.5, NROWS)
    dxp = rng.uniform(0.3, 1.5, NROWS)
    bw = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=bool)
    fw = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)  # rows 1 and 6 have both legs on walls
    return SimpleNamespace(center=center, minus=minus, plus=plus, dxm=dxm, dxp=dxp, bw=bw, fw=fw)


def _material_data(rows, psi, *, selection, selection_dt=1.0, tau=TAU):
    return parallel_short_wall_material_data(
        jnp.asarray(rows.center), jnp.asarray(rows.minus), jnp.asarray(rows.plus),
        jnp.asarray(rows.dxm), jnp.asarray(rows.dxp), tau, MU,
        selection_dt=selection_dt, parallel_short_leg_selection=selection,
        backward_wall=jnp.asarray(rows.bw), forward_wall=jnp.asarray(rows.fw),
        backward_wall_state=jnp.asarray(rows.minus), forward_wall_state=jnp.asarray(rows.plus),
        psi=psi,
    )


@pytest.mark.parametrize("psi", [LEGACY, PI])
def test_hand_principal_matrix_matches_library(psi):
    rows = _wall_rows()
    for r in range(NROWS):
        library = np.asarray(
            parallel_production_principal_matrix(*rows.center[r], TAU, MU, psi=psi)
        )
        np.testing.assert_allclose(library, _hand_matrix(rows.center[r], TAU, MU, psi), rtol=1e-14, atol=1e-13)
    # the two selectors differ exactly in A[4,0] and A[4,2]
    a_pi = np.stack([_hand_matrix(c, TAU, MU, PI) for c in rows.center])
    a_ti = np.stack([_hand_matrix(c, TAU, MU, LEGACY) for c in rows.center])
    diff = a_pi - a_ti
    assert np.max(np.abs(diff[:, 4, 0] - MU * TAU * rows.center[:, 2])) < 1e-12
    assert np.max(np.abs(diff[:, 4, 2] - MU * TAU * (rows.center[:, 0] - 1.0))) < 1e-12
    diff[:, 4, 0] = diff[:, 4, 2] = 0.0
    assert np.max(np.abs(diff)) == 0.0


@pytest.mark.parametrize("selection", SELECTORS)
@pytest.mark.parametrize("psi", [LEGACY, PI])
def test_wall_material_jacobian_is_the_independent_characteristic_split(psi, selection):
    """``J_b = -A_plus/dx_minus``, ``J_f = +A_minus/dx_plus`` with the psi matrix at the centre."""
    rows = _wall_rows()
    residual, jac, info = _material_data(rows, psi, selection=selection)
    selected = np.asarray(info["selected_wall"])
    assert selected.any(), "no short-leg wall selected"
    assert np.array_equal(selected, rows.bw | rows.fw)  # dt=1 -> every wall leg exceeds the CFL limit
    assert np.all(np.asarray(info["backward_valid"])[rows.bw])
    assert np.all(np.asarray(info["forward_valid"])[rows.fw])
    expected = np.zeros((NROWS, 5, 5))
    other_psi = PI if psi == LEGACY else LEGACY
    worst_wrong = 0.0
    for r in range(NROWS):
        A = _hand_matrix(rows.center[r], TAU, MU, psi)
        if rows.bw[r]:
            a_plus, _ = _split(A, +1)
            expected[r] += -a_plus / rows.dxm[r]
            wrong, _ = _split(_hand_matrix(rows.center[r], TAU, MU, other_psi), +1)
            worst_wrong = max(
                worst_wrong,
                np.max(np.abs(-wrong / rows.dxm[r] - expected[r])) / np.max(np.abs(expected[r])),
            )
        if rows.fw[r]:
            a_minus, _ = _split(A, -1)
            expected[r] += a_minus / rows.dxp[r]
    np.testing.assert_allclose(
        np.asarray(jac), expected, rtol=1e-8, atol=1e-8 * np.max(np.abs(expected))
    )
    assert worst_wrong > 1e-3  # the other convention gives a different split: the test can fail
    # the residual is the selected action: r = J (center - wall) with the same frozen matrix
    bwall = np.asarray(info["backward_action"])
    fwall = np.asarray(info["forward_action"])
    expected_r = (
        np.where(rows.bw[:, None], -bwall / rows.dxm[:, None], 0.0)
        + np.where(rows.fw[:, None], -fwall / rows.dxp[:, None], 0.0)
    )
    np.testing.assert_allclose(np.asarray(residual), expected_r, rtol=1e-12, atol=1e-12)


def test_wall_data_wrapper_threads_psi_like_the_material_data():
    rows = _wall_rows()
    results = {}
    for psi in (LEGACY, PI):
        data = parallel_characteristic_wall_data(
            jnp.asarray(rows.center), jnp.asarray(rows.minus), jnp.asarray(rows.plus),
            jnp.asarray(rows.dxm), jnp.asarray(rows.dxp), TAU, MU,
            selection_dt=1.0, parallel_short_leg_selection="all-physical-walls",
            backward_wall=jnp.asarray(rows.bw), forward_wall=jnp.asarray(rows.fw),
            backward_wall_state=jnp.asarray(rows.minus), forward_wall_state=jnp.asarray(rows.plus),
            psi=psi,
        )
        residual, jac, _ = _material_data(rows, psi, selection="all-physical-walls")
        np.testing.assert_array_equal(np.asarray(data["selected_jacobian"]), np.asarray(jac))
        np.testing.assert_array_equal(np.asarray(data["selected_residual"]), np.asarray(residual))
        results[psi] = np.asarray(jac)
    assert np.max(np.abs(results[PI] - results[LEGACY])) > 1e-3 * np.max(np.abs(results[LEGACY]))


def test_cfl_selector_selects_exactly_the_wall_legs_over_the_limit():
    rows = _wall_rows()
    # first pass: speeds alpha
    _, _, info = _material_data(rows, PI, selection="cfl", selection_dt=1.0)
    alpha_b = np.asarray(info["backward_alpha"])
    alpha_f = np.asarray(info["forward_alpha"])
    # per-leg dt: 1.5x the limit for rows 0, 1 (backward) and 2, 6 (forward); 0.5x otherwise
    over_b = np.isin(np.arange(NROWS), [0, 1])
    over_f = np.isin(np.arange(NROWS), [2, 6])
    # a single dt per row serves both legs: 1.5x the limit on the smaller requirement for the
    # rows named above, 0.5x of it elsewhere
    dt_row = np.where(
        over_b | over_f,
        np.minimum(1.5 * CFL_LIMIT * rows.dxm / alpha_b, 1.5 * CFL_LIMIT * rows.dxp / alpha_f),
        0.5 * np.minimum(CFL_LIMIT * rows.dxm / alpha_b, CFL_LIMIT * rows.dxp / alpha_f),
    )
    _, _, info = _material_data(rows, PI, selection="cfl", selection_dt=jnp.asarray(dt_row))
    cfl_b = dt_row * alpha_b / rows.dxm
    cfl_f = dt_row * alpha_f / rows.dxp
    np.testing.assert_allclose(np.asarray(info["backward_cfl"]), cfl_b, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(info["forward_cfl"]), cfl_f, rtol=1e-12)
    want_b = rows.bw & (cfl_b > CFL_LIMIT)
    want_f = rows.fw & (cfl_f > CFL_LIMIT)
    assert want_b.any() and want_f.any() and not (want_b | want_f).all()
    np.testing.assert_array_equal(np.asarray(info["selected_backward_wall"]), want_b)
    np.testing.assert_array_equal(np.asarray(info["selected_forward_wall"]), want_f)
    # tiny dt: nothing selected by CFL, everything on a wall by the other selector
    _, _, none = _material_data(rows, PI, selection="cfl", selection_dt=1e-9)
    assert not np.asarray(none["selected_wall"]).any()
    _, _, allw = _material_data(rows, PI, selection="all-physical-walls", selection_dt=1e-9)
    np.testing.assert_array_equal(np.asarray(allw["selected_wall"]), rows.bw | rows.fw)


def _coupled(rows, psi, tau, *, rng_seed=11):
    """The handoff residual/Jacobian exactly as ``apply_short_leg_implicit_material_step`` builds them."""
    rng = np.random.default_rng(rng_seed)
    w_ti = rng.uniform(-0.5, 0.5, NROWS)  # stencil centre weights (frozen)
    w_n = w_ti.copy()  # same geometry weights for n and Ti
    grad = rng.uniform(-0.2, 0.2, NROWS)  # G(q) at the row
    force = np.zeros((NROWS, 5))
    force[:, 4] = MU * tau * grad
    jac = np.zeros((NROWS, 5, 5))
    jac[:, 4, 2] = MU * tau * w_ti
    if psi == PI:
        jac[:, 4, 2] = jac[:, 4, 2] * rows.center[:, 0]
        jac[:, 4, 0] = MU * tau * w_n * rows.center[:, 2]
    return jnp.asarray(force), jnp.asarray(jac)


@pytest.mark.parametrize("selection", SELECTORS)
@pytest.mark.parametrize("psi", [LEGACY, PI])
def test_backward_euler_solves_its_own_residual_to_rounding(psi, selection):
    rows = _wall_rows()
    dt = 0.04
    force, cjac = _coupled(rows, psi, TAU)
    updated, delta, info = parallel_short_wall_backward_euler(
        jnp.asarray(rows.center), jnp.asarray(rows.minus), jnp.asarray(rows.plus),
        jnp.asarray(rows.dxm), jnp.asarray(rows.dxp), TAU, MU,
        selection_dt=1.0, solve_dt=dt, parallel_short_leg_selection=selection,
        backward_wall=jnp.asarray(rows.bw), forward_wall=jnp.asarray(rows.fw),
        backward_wall_state=jnp.asarray(rows.minus), forward_wall_state=jnp.asarray(rows.plus),
        coupled_residual=force, coupled_jacobian=cjac, psi=psi,
    )
    selected = np.asarray(info["selected_wall"])
    assert selected.sum() == (rows.bw | rows.fw).sum() > 0
    delta = np.asarray(delta)
    J = np.asarray(info["selected_complete_jacobian"])
    r = np.asarray(info["selected_complete_residual"])
    # complete = material + masked coupled
    np.testing.assert_array_equal(
        J, np.asarray(info["selected_material_jacobian"]) + np.where(selected[:, None, None], np.asarray(cjac), 0.0)
    )
    np.testing.assert_array_equal(
        r, np.asarray(info["selected_material_residual"]) + np.where(selected[:, None], np.asarray(force), 0.0)
    )
    # the backward-Euler equation delta = dt*(r + J delta), per row
    eq = delta - dt * (r + np.einsum("rij,rj->ri", J, delta))
    scale = np.max(np.abs(delta))
    assert scale > 1e-4
    assert np.max(np.abs(eq)) < 1e-12 * max(1.0, scale)
    # ... and it moves the row: the handoff force enters (differs from the material-only step)
    _, delta_m, _ = parallel_short_wall_backward_euler(
        jnp.asarray(rows.center), jnp.asarray(rows.minus), jnp.asarray(rows.plus),
        jnp.asarray(rows.dxm), jnp.asarray(rows.dxp), TAU, MU,
        selection_dt=1.0, solve_dt=dt, parallel_short_leg_selection=selection,
        backward_wall=jnp.asarray(rows.bw), forward_wall=jnp.asarray(rows.fw),
        backward_wall_state=jnp.asarray(rows.minus), forward_wall_state=jnp.asarray(rows.plus),
        psi=psi,
    )
    assert np.max(np.abs(delta - np.asarray(delta_m))) > 1e-6
    np.testing.assert_array_equal(np.asarray(updated), rows.center + delta)
    # unselected rows: exactly zero increment
    np.testing.assert_array_equal(delta[~selected], 0.0)


def test_backward_euler_tau_zero_pi_equals_legacy_bitwise():
    rows = _wall_rows()
    outs = {}
    for psi in (LEGACY, PI):
        force, cjac = _coupled(rows, psi, 0.0)
        outs[psi] = parallel_short_wall_backward_euler(
            jnp.asarray(rows.center), jnp.asarray(rows.minus), jnp.asarray(rows.plus),
            jnp.asarray(rows.dxm), jnp.asarray(rows.dxp), 0.0, MU,
            selection_dt=1.0, solve_dt=0.04, parallel_short_leg_selection="all-physical-walls",
            backward_wall=jnp.asarray(rows.bw), forward_wall=jnp.asarray(rows.fw),
            backward_wall_state=jnp.asarray(rows.minus), forward_wall_state=jnp.asarray(rows.plus),
            coupled_residual=force, coupled_jacobian=cjac, psi=psi,
        )
    np.testing.assert_array_equal(np.asarray(outs[PI][0]), np.asarray(outs[LEGACY][0]))
    np.testing.assert_array_equal(np.asarray(outs[PI][1]), np.asarray(outs[LEGACY][1]))
    np.testing.assert_array_equal(
        np.asarray(outs[PI][2]["selected_complete_jacobian"]),
        np.asarray(outs[LEGACY][2]["selected_complete_jacobian"]),
    )
    assert np.max(np.abs(np.asarray(outs[PI][1]))) > 1e-4  # a nontrivial step


def test_unit_density_electron_row_is_the_legacy_row_with_dti_to_dn_plus_dti():
    """About ``n = Ti = 1``: ``A_pi[4,:] = A_legacy[4,:] @ S`` with ``S: (dn, dTi) -> (dn, dn + dTi)``."""
    rng = np.random.default_rng(5)
    for _ in range(4):
        state = np.array([1.0, rng.uniform(0.6, 1.5), 1.0, rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)])
        a_pi = _hand_matrix(state, TAU, MU, PI)
        a_ti = _hand_matrix(state, TAU, MU, LEGACY)
        S = np.eye(5)
        S[2, 0] = 1.0  # dTi_legacy = dn + dTi_pi
        np.testing.assert_allclose(a_pi[4], a_ti[4] @ S, rtol=1e-14, atol=1e-12)
        np.testing.assert_array_equal(a_pi[:4], a_ti[:4])
        lib = np.asarray(parallel_production_principal_matrix(*state, TAU, MU, psi=PI))
        np.testing.assert_allclose(lib[4], a_ti[4] @ S, rtol=1e-14, atol=1e-12)


# ======================================================================================
# model layer: LocalFciDrbEBRhs on a small shifted torus with radial walls
# ======================================================================================

SOLVE_DT = 0.05
TILT = 0.2  # B^x / B^z: 24 backward + 24 forward legs end on the radial walls


@functools.lru_cache(maxsize=None)
def _case():
    context, mesh, local, partition, fields, cell_fields = _context_and_sharded_inputs()
    host = build_shifted_torus_4field_geometry(context.geometry.shape)
    J = np.asarray(host.cell_metric.J)
    B = jnp.stack((TILT * c_phi / J, iota * c_phi / J, c_phi / J), axis=-1)
    mf = build_fci_maps_from_b_contravariant(
        host.grid, B, host.cell_bfield.Bmag, periodic_axes=(False, True, True)
    )
    keys = (
        "forward_x", "forward_y", "backward_x", "backward_y", "forward_endpoint_x",
        "forward_endpoint_y", "forward_endpoint_z", "backward_endpoint_x",
        "backward_endpoint_y", "backward_endpoint_z", "forward_length", "backward_length",
        "forward_boundary", "backward_boundary",
    )
    geometry = replace(context.geometry, maps=FciMaps3D(**{k: mf[k] for k in keys}))
    sharded = build_local_fci_geometries(
        geometry, (1, 1, 1), halo_width=local.domain.layout.halo_width,
        periodic_axes=(False, True, True),
    )
    assert sharded.maps_valid
    sharding = NamedSharding(mesh, partition)
    return SimpleNamespace(
        context=context, mesh=mesh, local=local, partition=partition, sharded=sharded,
        sharding=sharding, cell_fields=cell_fields,
        map_fields=jax.device_put(sharded.map_fields, sharding),
        base_fields=[np.asarray(f) for f in fields],
        backward_boundary=np.asarray(mf["backward_boundary"]),
        forward_boundary=np.asarray(mf["forward_boundary"]),
        tau=float(context.parameters.tau),
        mu=float(context.parameters.mi_over_me),
    )


def _state(case, *, unit_density=False):
    """A positive state with ``n != 1`` and ``Ti != 1`` (so ``Ti*dn`` and ``n*dTi`` differ)."""
    n, phi, Te, Ti, Vi, Ve, w = (f.copy() for f in case.base_fields)
    n = np.ones_like(n) if unit_density else 1.5 * n
    return [n, phi, 1.2 * Te, 0.7 * Ti, 100.0 * Vi, 100.0 * Ve, w]


def _put(case, arrays):
    return [jax.device_put(jnp.asarray(a, dtype=jnp.float64), case.sharding) for a in arrays]


def _rhs(case, cells, maps, psi, selection, tau, treatment="local-backward-euler"):
    geometry = assemble_local_fci_geometry(case.sharded, cells, maps)
    parameters = replace(
        case.context.parameters,
        parallel_characteristic_wall_law="primitive-least-residual",
        polarization_variable=psi,
        tau=case.tau if tau is None else float(tau),
    )
    return replace(
        _build_rhs(replace(case.context, parameters=parameters), case.local, geometry),
        parallel_operator_scheme="fci",
        parallel_flux_pairing="support-core",
        parallel_boundary_pairing="characteristic-sat",
        parallel_material_scheme="production-path",
        parallel_short_leg_treatment=treatment,
        parallel_short_leg_selection=selection,
    )


_STEP_OUTPUTS = (
    "selected_wall", "selected_backward_wall", "selected_forward_wall", "backward_wall",
    "forward_wall", "selected_coupled_force", "selected_coupled_jacobian",
    "selected_material_jacobian", "selected_material_residual", "selected_complete_jacobian",
    "selected_complete_residual", "center", "backward_jacobian", "forward_jacobian",
    "backward_valid", "forward_valid", "implicit_finite",
)


@functools.lru_cache(maxsize=None)
def _step_kernel(psi, selection, tau=None):
    """One full handoff step: ``run(state, selection_dt, tangents) -> dict`` of increment/info.

    For ``selection="all-physical-walls"`` with the default ``tau`` the kernel also returns
    ``force_jvp``, the exact ``jax.jvp`` of the handoff force along ``tangents = (dn, dTi)``
    (one trace serves both, via ``has_aux``); other kernels return zeros there.
    """
    case = _case()
    part = case.partition

    def kernel(n, phi, Te, Ti, Vi, Ve, w, tn, tTi, selection_dt, cells, maps):
        rhs = _rhs(case, cells, maps, psi, selection, tau)

        def step(n_, Ti_):
            _, inc, info = rhs.apply_short_leg_implicit_material_step(
                FciDrbEBState(n_, phi, Te, Ti_, Vi, Ve, w),
                solve_dt=SOLVE_DT, selection_dt=selection_dt, phi_owned=phi,
                return_increment=True,
            )
            increment = jnp.stack((inc.density, inc.Te, inc.Ti, inc.Vi, inc.Ve), axis=-1)
            return info["selected_coupled_force"], (increment,) + tuple(info[k] for k in _STEP_OUTPUTS)

        if selection == "all-physical-walls" and tau is None:
            _, force_jvp, aux = jax.jvp(step, (n, Ti), (tn, tTi), has_aux=True)
        else:
            _, aux = step(n, Ti)
            force_jvp = jnp.zeros_like(n)
        return (force_jvp,) + aux

    jitted = jax.jit(jax.shard_map(
        kernel, mesh=case.mesh,
        in_specs=(part,) * 9 + (P(),) + (part, part),
        out_specs=(part,) * (2 + len(_STEP_OUTPUTS)), check_vma=False,
    ))

    def run(arrays, selection_dt=1.0, tangents=None):
        zero = np.zeros_like(arrays[0])
        tn, tTi = (zero, zero) if tangents is None else tangents
        out = jitted(
            *_put(case, arrays), *_put(case, (tn, tTi)),
            jnp.asarray(selection_dt, dtype=jnp.float64), case.cell_fields, case.map_fields,
        )
        result = {"force_jvp": np.asarray(out[0]), "increment": np.asarray(out[1])}
        result.update({k: np.asarray(v) for k, v in zip(_STEP_OUTPUTS, out[2:])})
        return result

    return run


def _selected_cells(mask, count=10):
    """A few selected cells, both backward and forward legs, spread over the grid."""
    cells = [tuple(c) for c in np.argwhere(mask)]
    return cells[:: max(1, len(cells) // count)]


@pytest.mark.parametrize("selection", SELECTORS)
def test_walls_are_actually_selected_and_fields_stay_finite(selection):
    case = _case()
    out = _step_kernel(PI, selection)(_state(case))
    sel = out["selected_wall"]
    assert sel.sum() == 48 == (case.backward_boundary | case.forward_boundary).sum()
    np.testing.assert_array_equal(out["backward_wall"], case.backward_boundary)
    np.testing.assert_array_equal(out["forward_wall"], case.forward_boundary)
    np.testing.assert_array_equal(out["selected_backward_wall"], case.backward_boundary)
    np.testing.assert_array_equal(out["selected_forward_wall"], case.forward_boundary)
    assert out["implicit_finite"].all()
    assert np.isfinite(out["increment"]).all()
    assert np.all(out["backward_valid"][case.backward_boundary])
    assert np.all(out["forward_valid"][case.forward_boundary])
    inc = out["increment"]
    assert np.max(np.abs(inc[sel])) > 1e-6
    np.testing.assert_array_equal(inc[~sel], 0.0)
    # cfl vs all-physical-walls differ only through ``selection_dt``
    kernel = _step_kernel(PI, selection)
    n_tiny = kernel(_state(case), selection_dt=0.0)["selected_wall"].sum()
    assert n_tiny == (48 if selection == "all-physical-walls" else 0)


@pytest.mark.parametrize("selection", SELECTORS)
def test_coupled_jacobian_entries_are_the_p_i_chain_rule_of_the_legacy_entry(selection):
    """``[4,2] = mu tau w n`` and ``[4,0] = mu tau w Ti``; the legacy row is ``mu tau w`` in ``[4,2]``."""
    case = _case()
    state = _state(case)
    pi = _step_kernel(PI, selection)(state)
    leg = _step_kernel(LEGACY, selection)(state)
    sel = pi["selected_wall"]
    np.testing.assert_array_equal(sel, leg["selected_wall"])
    cj_pi = pi["selected_coupled_jacobian"]
    cj_leg = leg["selected_coupled_jacobian"]
    # only the electron-momentum row is populated; legacy only in the Ti column
    for cj, cols in ((cj_pi, {(4, 0), (4, 2)}), (cj_leg, {(4, 2)})):
        mask = np.ones((5, 5), dtype=bool)
        for c in cols:
            mask[c] = False
        assert np.all(cj[..., mask] == 0.0)
    np.testing.assert_array_equal(cj_pi[~sel], 0.0)
    w = cj_leg[..., 4, 2] / (case.mu * case.tau)  # the Ti centre weight w_Ti
    assert np.max(np.abs(w[sel])) > 1e-2
    center = pi["center"]
    n_c, ti_c = center[..., 0], center[..., 2]
    assert np.max(np.abs(n_c - ti_c)[sel]) > 0.3  # the factors n and Ti are distinguishable
    np.testing.assert_allclose(cj_pi[..., 4, 2][sel], (cj_leg[..., 4, 2] * n_c)[sel], rtol=1e-12)
    np.testing.assert_allclose(cj_pi[..., 4, 0][sel], (cj_leg[..., 4, 2] * ti_c)[sel], rtol=1e-12)
    # the claimed ratio dF/dn : dF/dTi is Ti : n, exactly the chain rule of n*Ti
    np.testing.assert_allclose(
        (cj_pi[..., 4, 0] * n_c)[sel], (cj_pi[..., 4, 2] * ti_c)[sel], rtol=1e-12
    )


def test_coupled_force_obeys_the_p_i_chain_rule_in_the_model_derivative():
    """Exact ``jax.jvp`` of the model force: ``dF_pi/dn = Ti X``, ``dF_pi/dTi = n X``, ``X = dF_ti/dTi``.

    Holds when the wall traces do not depend on the centre (Dirichlet radial values), which
    is the fixture; it pins the product-of-traces construction of ``grad_pi``.
    """
    case = _case()
    state = _state(case)
    zero = np.zeros_like(state[0])
    sel = _step_kernel(PI, "all-physical-walls")(state)["selected_wall"]
    pi_run = _step_kernel(PI, "all-physical-walls")
    ti_run = _step_kernel(LEGACY, "all-physical-walls")
    n, ti = state[0], state[3]
    for cell in _selected_cells(sel):
        e = zero.copy()
        e[cell] = 1.0
        x = ti_run(state, tangents=(zero, e))["force_jvp"][cell]  # d(mu tau G(Ti))_i / dTi_i
        dn = pi_run(state, tangents=(e, zero))["force_jvp"][cell]
        dti = pi_run(state, tangents=(zero, e))["force_jvp"][cell]
        assert abs(x) > 1e-3
        np.testing.assert_allclose(dn, ti[cell] * x, rtol=1e-7)
        np.testing.assert_allclose(dti, n[cell] * x, rtol=1e-7)
        assert abs(dn - dti) > 1e-2 * abs(x)  # n != Ti: a swapped factor would be caught


@pytest.mark.xfail(
    strict=False,
    reason=(
        "pre-existing (legacy too): the coupled [4,2] entry mu*tau*w_c is the frozen "
        "three-point-stencil centre derivative, which differs from the exact diagonal "
        "derivative of the production support-core force by a factor 1-30 on wall rows"
    ),
)
@pytest.mark.parametrize("psi", [LEGACY, PI])
def test_coupled_jacobian_equals_exact_jvp_of_the_production_force(psi):
    case = _case()
    state = _state(case)
    zero = np.zeros_like(state[0])
    out = _step_kernel(psi, "all-physical-walls")(state)
    sel = out["selected_wall"]
    cj = out["selected_coupled_jacobian"]
    run = _step_kernel(psi, "all-physical-walls")
    for cell in _selected_cells(sel):
        e = zero.copy()
        e[cell] = 1.0
        exact = run(state, tangents=(zero, e))["force_jvp"][cell]
        np.testing.assert_allclose(cj[cell + (4, 2)], exact, rtol=1e-2)


@pytest.mark.parametrize("selection", SELECTORS)
@pytest.mark.parametrize("psi", [LEGACY, PI])
def test_model_material_jacobian_is_the_independent_split_with_the_psi_matrix(psi, selection):
    """Wall rows: ``-J_b dx_minus = A_plus``, ``J_f dx_plus = A_minus`` up to the (unknown) scalar ``1/dx``."""
    case = _case()
    out = _step_kernel(psi, selection)(_state(case))
    center = out["center"]
    other = PI if psi == LEGACY else LEGACY
    worst_right = 0.0
    worst_wrong = np.inf
    for key, flag, sign in (("backward_jacobian", "selected_backward_wall", +1),
                            ("forward_jacobian", "selected_forward_wall", -1)):
        cells = [tuple(c) for c in np.argwhere(out[flag])]
        assert cells
        for cell in cells:
            X = out[key][cell] * (-1.0 if sign > 0 else 1.0)  # = A_{+/-} / dx
            errs = []
            for conv in (psi, other):
                M, _ = _split(_hand_matrix(center[cell], case.tau, case.mu, conv), sign, 1e-6)
                s = np.sum(X * M) / np.sum(M * M)  # best scalar 1/dx
                errs.append((np.max(np.abs(X - s * M)) / np.max(np.abs(X)), s))
            assert errs[0][1] > 0.0
            assert 0.05 < errs[0][1] < 20.0  # a plausible 1/dx
            worst_right = max(worst_right, errs[0][0])
            worst_wrong = min(worst_wrong, errs[1][0])
    assert worst_right < 1e-9
    assert worst_wrong > 1e-4  # the other convention is distinguishable


@pytest.mark.parametrize("selection", SELECTORS)
@pytest.mark.parametrize("psi", [LEGACY, PI])
def test_model_step_solves_its_own_residual_to_rounding(psi, selection):
    case = _case()
    out = _step_kernel(psi, selection)(_state(case))
    sel = out["selected_wall"]
    assert sel.sum() == 48
    inc = out["increment"]
    J = out["selected_complete_jacobian"]
    r = out["selected_complete_residual"]
    np.testing.assert_allclose(
        J, out["selected_material_jacobian"] + out["selected_coupled_jacobian"], rtol=0, atol=1e-12
    )
    eq = inc - SOLVE_DT * (r + np.einsum("...ij,...j->...i", J, inc))
    scale = np.max(np.abs(inc))
    assert scale > 1e-6
    assert np.max(np.abs(eq[sel])) < 1e-11 * max(1.0, scale)
    # the handoff force is in the residual: [4] component of the coupled residual is the force
    np.testing.assert_allclose(
        out["selected_complete_residual"][..., 4] - out["selected_material_residual"][..., 4],
        out["selected_coupled_force"], rtol=0, atol=1e-12,
    )
    assert np.max(np.abs(out["selected_coupled_force"][sel])) > 1e-3


def test_tau_zero_pi_step_equals_legacy_step():
    selection = "all-physical-walls"
    case = _case()
    state = _state(case)
    pi = _step_kernel(PI, selection, 0.0)(state)
    leg = _step_kernel(LEGACY, selection, 0.0)(state)
    assert pi["selected_wall"].sum() == 48
    for key in ("increment",) + _STEP_OUTPUTS:
        np.testing.assert_allclose(pi[key], leg[key], rtol=1e-14, atol=1e-14, err_msg=key)
    assert np.all(pi["selected_coupled_force"] == 0.0)
    assert np.max(np.abs(pi["increment"])) > 1e-6


def test_unit_density_pi_force_is_the_legacy_force_and_jacobian_row_is_the_dti_to_dn_plus_dti_map():
    """``n = 1``: ``G(n Ti) = G(Ti)``; with ``Ti`` also 1 the Jacobian row is ``legacy @ S``."""
    case = _case()
    state = _state(case, unit_density=True)
    pi = _step_kernel(PI, "all-physical-walls")(state)
    leg = _step_kernel(LEGACY, "all-physical-walls")(state)
    sel = pi["selected_wall"]
    assert sel.sum() == 48
    scale = np.max(np.abs(leg["selected_coupled_force"]))
    assert scale > 1e-3
    np.testing.assert_allclose(
        pi["selected_coupled_force"], leg["selected_coupled_force"], rtol=0, atol=1e-13 * scale
    )
    # Jacobian: [4,2] equal, [4,0] = Ti * [4,2]_legacy
    cj_pi, cj_leg = pi["selected_coupled_jacobian"], leg["selected_coupled_jacobian"]
    ti_c = pi["center"][..., 2]
    np.testing.assert_allclose(cj_pi[..., 4, 2], cj_leg[..., 4, 2], rtol=1e-13, atol=0)
    np.testing.assert_allclose(cj_pi[..., 4, 0][sel], (cj_leg[..., 4, 2] * ti_c)[sel], rtol=1e-12)
    # with Ti = 1 as well, row 4 of the pi Jacobian is row 4 of the legacy one times S
    unit = _state(case, unit_density=True)
    unit[3] = np.ones_like(unit[3])
    pi1 = _step_kernel(PI, "all-physical-walls")(unit)["selected_coupled_jacobian"]
    leg1 = _step_kernel(LEGACY, "all-physical-walls")(unit)["selected_coupled_jacobian"]
    S = np.eye(5)
    S[2, 0] = 1.0
    row_pi = pi1[..., 4, :]
    row_leg = np.einsum("...j,jk->...k", leg1[..., 4, :], S)
    np.testing.assert_allclose(row_pi[sel], row_leg[sel], rtol=1e-12, atol=1e-10)
    assert np.max(np.abs(row_pi[sel][:, 0])) > 1.0


# --------------------------------------------------------------------------------------
# electron-force wall diagnostic
# --------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _diagnostic_kernel():
    """p_i force diagnostics of the BE-handoff model and of the otherwise identical explicit model."""
    case = _case()
    part = case.partition
    full = P(None, "x", "y", "z")

    def kernel(n, phi, Te, Ti, Vi, Ve, w, cells, maps):
        state = FciDrbEBState(n, phi, Te, Ti, Vi, Ve, w)
        be = _rhs(case, cells, maps, PI, "all-physical-walls", None)
        ex = _rhs(case, cells, maps, PI, "cfl", None, treatment="explicit")
        return (
            be.electron_parallel_force_diagnostics(state, phi_owned=phi)[0],
            ex.electron_parallel_force_diagnostics(state, phi_owned=phi)[0],
        )

    jitted = jax.jit(jax.shard_map(
        kernel, mesh=case.mesh, in_specs=(part,) * 9, out_specs=(full, full), check_vma=False,
    ))

    def run(arrays):
        return [np.asarray(x) for x in jitted(*_put(case, arrays), case.cell_fields, case.map_fields)]

    return run


def test_wall_diagnostic_explicit_plus_implicit_is_the_composite_force():
    """Invariant: diagnostic force + handoff force = the explicit-treatment composite force.

    On a selected short wall the BE stage carries ``mu*tau*G(n*Ti)``, so the explicit
    electrostatic force reported by the diagnostic is the composite ``mu*G(phi + tau*n*Ti)``
    minus exactly that piece (the ``selected_coupled_force`` of the implicit step); on every
    other row it is untouched.  The explicit-treatment model selects no wall (``cfl`` with
    ``selection_dt = 0``), so its diagnostic is the full composite.  The only other force
    term that may differ is the material residual, and only on the selected rows, where the
    BE stage owns the wall leg.
    """
    case = _case()
    state = _state(case)
    f_be, f_ex = _diagnostic_kernel()(state)
    step = _step_kernel(PI, "all-physical-walls")(state)
    sel, coupled = step["selected_wall"], step["selected_coupled_force"]
    assert sel.sum() == 48
    es_be, es_ex = f_be[2], f_ex[2]
    scale = np.max(np.abs(es_ex))
    assert scale > 1.0
    np.testing.assert_allclose(es_ex - es_be, coupled, rtol=0, atol=1e-13 * scale)
    np.testing.assert_array_equal(es_be[~sel], es_ex[~sel])
    assert np.max(np.abs(coupled[sel])) > 1e-3
    assert np.max(np.abs(es_ex - es_be)[sel]) > 1e-3
    np.testing.assert_array_equal(coupled[~sel], 0.0)
    # every other force term is bitwise unchanged, except the material residual (index 5) on
    # the selected rows (where the wall leg moved into the implicit stage)
    unchanged = [i for i in range(f_be.shape[0]) if i not in (2, 5)]
    np.testing.assert_array_equal(f_be[unchanged], f_ex[unchanged])
    np.testing.assert_array_equal(f_be[5][~sel], f_ex[5][~sel])
    assert np.max(np.abs(f_be[5] - f_ex[5])[sel]) > 1e-3


def test_wall_handoff_force_is_the_pi_force_not_the_ti_force():
    """For ``n != 1`` the p_i handoff is ``mu*tau*G(n*Ti)``, visibly different from ``mu*tau*G(Ti)``."""
    case = _case()
    state = _state(case)
    pi = _step_kernel(PI, "all-physical-walls")(state)
    ti = _step_kernel(LEGACY, "all-physical-walls")(state)
    sel = pi["selected_wall"]
    np.testing.assert_array_equal(sel, ti["selected_wall"])
    gap = np.max(np.abs(pi["selected_coupled_force"] - ti["selected_coupled_force"])[sel])
    assert gap > 1e-2 * np.max(np.abs(ti["selected_coupled_force"][sel]))
