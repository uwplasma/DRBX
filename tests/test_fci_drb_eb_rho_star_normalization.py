"""Single-length rho_star normalization of the production FCI EB RHS.

Model (``L_ref`` = the geometry's own length unit, ``t_ref = L_ref / c_s0``,
``rho_star = rho_s0 / L_ref``):

* E x B bracket on all six fields: ``-rho_star [phi, g] / B``;
* the whole curvature family (four rows, material and remainder parts and the
  directional component fields): ``rho_star * C(...)``;
* polarization ``Omega = rho_star**2 Lperp(phi + tau p_i)``, applied as
  ``Omega / rho_star**2`` on the right-hand side of the phi solve and as
  ``rho_star**2`` on omega-from-phi maps (the operator is never scaled);
* every parallel term, wall, source and diffusion term is unchanged.

Fixture: the small shifted-torus EB geometry used by the other EB tests, with
the radial Dirichlet wall data of ``test_mms_shifted_torus_EB_sharded`` (all
fields flat on the walls, ``phi = 0`` there, periodic ``theta`` and ``zeta``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from drbx.native import (
    assemble_local_fci_geometry,
    build_local_fci_geometries,
    make_shard_mesh,
)
from drbx.native.fci_drb_EB_rhs import (
    RHS_TERM_FIELD_NAMES,
    RHS_TERM_NAMES,
    FciDrbEBState,
)
from drbx.native.fci_operators import LocalPerpLaplacianInverseSolver

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fci_drb_eb_test_helpers import _build_rhs  # noqa: E402
from shifted_torus_eb_mms_data import (  # noqa: E402
    _mms_exact_state,
    build_shifted_torus_eb_mms_context,
)
from test_mms_shifted_torus_EB_sharded import (  # noqa: E402
    HALO_WIDTH,
    MMS_TIME,
    PERIODIC_AXES,
    _build_face_bcs,
)

WALL_LAWS = ("primitive-least-residual", "energy-absorbing")
FIELDS = ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")
PARTITION = P("x", "y", "z")
SPEC_FIELD = P("x", "y", "z")
SPEC_STACK = P(None, "x", "y", "z")
SPEC_TERMS = P(None, None, "x", "y", "z")

# (row, term name) pairs carrying the E x B bracket and the curvature family.
BRACKET_SLOTS = tuple(
    (row, RHS_TERM_NAMES[row].index("poisson_bracket"))
    for row in range(len(RHS_TERM_FIELD_NAMES))
)
CURVATURE_ROWS = {"density": 0, "Te": 1, "Ti": 2, "vorticity": 5}
CURVATURE_SLOTS = tuple(
    (row, RHS_TERM_NAMES[row].index("curvature"))
    for row in CURVATURE_ROWS.values()
)


@dataclass(frozen=True)
class _Fixture:
    context: object
    mesh: object
    local: object
    sharding: object
    cell_fields: object
    state: FciDrbEBState
    shape: tuple[int, int, int]


def _fixture(shape=(4, 6, 4), wall_law="primitive-least-residual") -> _Fixture:
    return _fixture_cached(tuple(shape), wall_law)


@lru_cache(maxsize=None)
def _fixture_cached(shape, wall_law) -> _Fixture:
    context = build_shifted_torus_eb_mms_context(shape)
    context = replace(
        context,
        parameters=replace(
            context.parameters, parallel_characteristic_wall_law=wall_law
        ),
    )
    state = _mms_exact_state(context, MMS_TIME)
    mesh = make_shard_mesh((1, 1, 1))
    local = build_local_fci_geometries(
        context.geometry,
        (1, 1, 1),
        halo_width=HALO_WIDTH,
        periodic_axes=PERIODIC_AXES,
    )
    sharding = NamedSharding(mesh, PARTITION)
    return _Fixture(
        context=context,
        mesh=mesh,
        local=local,
        sharding=sharding,
        cell_fields=jax.device_put(local.cell_fields, sharding),
        state=state,
        shape=shape,
    )


def _build_model(fx, cell_fields, rho_star, tau, regularization):
    geometry = assemble_local_fci_geometry(fx.local, cell_fields)
    parameters = replace(
        fx.context.parameters,
        tau=tau,
        rho_star=rho_star,
    )
    rhs = replace(_build_rhs(fx.context, fx.local, geometry), parameters=parameters)
    if regularization is not None:
        rhs = replace(
            rhs,
            gmres_config=replace(
                rhs.gmres_config, regularization_epsilon=regularization
            ),
        )
    return rhs


@lru_cache(maxsize=None)
def _full_kernel(wall_law, regularization, shape=(4, 6, 4)):
    """Jitted evaluation of every rho_star-dependent quantity.

    Returns ``(stage RHS (7 fields), RHS term fields, curvature component
    fields, phi from the solve, omega from phi, polarization residual,
    polarization balance terms)``.  ``rho_star`` and ``tau`` are runtime
    scalars; the wall law and regularization are static.
    """

    fx = _fixture(shape, wall_law)

    def kernel(rho_star, tau, density, phi, Te, Ti, Vi, Ve, vorticity, cell_fields):
        rhs = _build_model(fx, cell_fields, rho_star, tau, regularization)
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        result, terms, curvature = rhs.evaluate_stage(
            state,
            phi_owned=phi,
            return_rhs_term_fields=True,
            return_curvature_component_fields=True,
        )
        stage = jnp.stack(
            (
                result.density, result.phi, result.Te, result.Ti,
                result.Vi, result.Ve, result.vorticity,
            ),
            axis=0,
        )
        phi_solved = rhs.reconstruct_phi(state)
        face_bc = rhs._face_bcs(state)
        pressure, pressure_bc = rhs._polarization_pressure(state, face_bc)
        omega_from_phi = rhs._vorticity_from_polarization(
            phi, pressure, face_bc.phi, pressure_bc
        )
        residual = rhs.polarization_residual(state, phi_owned=phi)
        balance = rhs.polarization_balance_terms(state, phi_owned=phi)
        return (
            stage, terms, curvature, phi_solved, omega_from_phi, residual, balance
        )

    return jax.jit(
        jax.shard_map(
            kernel,
            mesh=fx.mesh,
            in_specs=(P(), P()) + (PARTITION,) * 8,
            out_specs=(
                SPEC_STACK, SPEC_TERMS, SPEC_TERMS, SPEC_FIELD, SPEC_FIELD,
                SPEC_FIELD, SPEC_STACK,
            ),
            check_vma=False,
        )
    )


@lru_cache(maxsize=None)
def _terms_kernel(shape):
    fx = _fixture(shape)

    def kernel(rho_star, tau, density, phi, Te, Ti, Vi, Ve, vorticity, cell_fields):
        rhs = _build_model(fx, cell_fields, rho_star, tau, None)
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        _, terms = rhs.evaluate_stage(
            state, phi_owned=phi, return_rhs_term_fields=True
        )
        return terms

    return jax.jit(
        jax.shard_map(
            kernel,
            mesh=fx.mesh,
            in_specs=(P(), P()) + (PARTITION,) * 8,
            out_specs=SPEC_TERMS,
            check_vma=False,
        )
    )


def _run(kernel, fx, rho_star, tau, fields):
    arrays = [
        jax.device_put(jnp.asarray(value, dtype=jnp.float64), fx.sharding)
        for value in fields
    ]
    outputs = kernel(
        jnp.asarray(rho_star, dtype=jnp.float64),
        jnp.asarray(tau, dtype=jnp.float64),
        *arrays,
        fx.cell_fields,
    )
    return jax.tree_util.tree_map(np.asarray, outputs)


def _state_fields(fx, **overrides):
    values = {name: np.asarray(getattr(fx.state, name)) for name in FIELDS}
    values.update({key: np.asarray(value) for key, value in overrides.items()})
    return [values[name] for name in FIELDS]


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _assert_scaled(actual, reference, factor, *, rtol=1.0e-13, label=""):
    scale = float(np.max(np.abs(reference)))
    assert scale > 0.0, f"{label}: degenerate reference"
    error = _max_abs(actual, factor * np.asarray(reference))
    assert error <= rtol * abs(factor) * scale, (
        f"{label}: max|actual - {factor}*reference| = {error:.3e} "
        f"(reference scale {scale:.3e})"
    )


# --------------------------------------------------------------------------
# homogeneity of the bracket / curvature family
# --------------------------------------------------------------------------


@pytest.mark.parametrize("wall_law", WALL_LAWS)
@pytest.mark.parametrize("rho_star", [0.05, 0.3])
def test_bracket_and_curvature_homogeneity(wall_law, rho_star):
    """Bracket and curvature term fields (and the directional curvature
    components) equal ``rho_star`` times their ``rho_star = 1`` values; every
    other term slot is rho_star-independent.  ``energy-absorbing`` selects the
    production FCI parallel path used by the HSX driver."""

    fx = _fixture((4, 6, 4), wall_law)
    fields = _state_fields(fx)
    kernel = _full_kernel(wall_law, None)
    base = _run(kernel, fx, 1.0, 1.0, fields)
    out = _run(kernel, fx, rho_star, 1.0, fields)
    base_terms, terms = base[1], out[1]
    assert np.all(np.isfinite(terms)) and np.max(np.abs(base_terms)) > 0.0

    for row, slot in BRACKET_SLOTS + CURVATURE_SLOTS:
        label = f"{RHS_TERM_FIELD_NAMES[row]}/{RHS_TERM_NAMES[row][slot]}"
        _assert_scaled(terms[row, slot], base_terms[row, slot], rho_star, label=label)

    # Every other term slot (parallel, diffusion, sources, upwind) is
    # untouched by rho_star.
    touched = set(BRACKET_SLOTS) | set(CURVATURE_SLOTS)
    for row in range(len(RHS_TERM_FIELD_NAMES)):
        for slot in range(len(RHS_TERM_NAMES[row])):
            if (row, slot) not in touched:
                assert _max_abs(terms[row, slot], base_terms[row, slot]) == 0.0, (
                    RHS_TERM_FIELD_NAMES[row], RHS_TERM_NAMES[row][slot]
                )

    # The directional curvature component fields carry rho_star as well and
    # still sum to the curvature term of each row.
    _assert_scaled(out[2], base[2], rho_star, label="curvature components")
    for component_row, term_row in enumerate(CURVATURE_ROWS.values()):
        slot = RHS_TERM_NAMES[term_row].index("curvature")
        np.testing.assert_allclose(
            np.sum(out[2][component_row], axis=0),
            terms[term_row, slot],
            rtol=0.0,
            atol=1.0e-12 * float(np.max(np.abs(terms[term_row, slot]))),
        )

    # The assembled stage RHS: parallel part + rho_star * perpendicular part.
    perpendicular = np.zeros_like(base[0])
    state_index = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    for row, slot in BRACKET_SLOTS + CURVATURE_SLOTS:
        perpendicular[state_index[row]] += base_terms[row, slot]
    expected = base[0] + (rho_star - 1.0) * perpendicular
    scale = float(np.max(np.abs(base[0])))
    assert _max_abs(out[0], expected) <= 1.0e-11 * scale


@pytest.mark.parametrize("rho_star", [0.05, 0.3])
def test_phi_solve_and_omega_maps_with_rho_star_squared(rho_star):
    """Omega scaled by rho_star**2 returns the same phi; omega-from-phi scales
    by rho_star**2; the balance terms stay in phi units.  A large fixed
    algebraic regularization checks that the operator itself is never scaled."""

    fx = _fixture()
    fields = _state_fields(fx)
    omega_1 = np.asarray(fx.state.vorticity)
    scaled = _state_fields(fx, vorticity=rho_star**2 * omega_1)
    kernel = _full_kernel("primitive-least-residual", 1.0e-3)

    base = _run(kernel, fx, 1.0, 1.0, fields)
    out = _run(kernel, fx, rho_star, 1.0, scaled)

    phi_scale = float(np.max(np.abs(base[3])))
    assert phi_scale > 0.0
    # Same phi from Omega = rho_star**2 * omega_1 (GMRES tolerance 1e-10).
    assert _max_abs(out[3], base[3]) <= 1.0e-8 * phi_scale
    # The solve is not trivially insensitive to the vorticity.
    other = _run(kernel, fx, 1.0, 1.0, _state_fields(fx, vorticity=2.0 * omega_1))
    assert _max_abs(other[3], base[3]) > 1.0e-3 * phi_scale
    # Without the rho_star**2 scaling of Omega the solve would differ.
    unscaled = _run(kernel, fx, rho_star, 1.0, fields)
    assert _max_abs(unscaled[3], base[3]) > 1.0e-3 * phi_scale

    # omega from phi (same phi): rho_star**2 * (rho_star = 1 value).
    _assert_scaled(out[4], base[4], rho_star**2, label="omega from phi")

    # Balance terms: (A phi, -tau A q) are phi-units and rho_star-free; the
    # vorticity entry is -Omega / rho_star**2.
    assert _max_abs(out[6][0], base[6][0]) == 0.0
    assert _max_abs(out[6][1], base[6][1]) == 0.0
    _assert_scaled(out[6][2], base[6][2], 1.0, rtol=1.0e-13, label="-Omega/rho*^2")


@pytest.mark.parametrize("rho_star", [0.05, 0.3])
def test_phi_omega_round_trip_without_regularization(rho_star):
    """phi -> Omega -> phi returns the original potential, and the polarization
    residual closes, at rho_star != 1."""

    fx = _fixture()
    fields = _state_fields(fx)
    kernel = _full_kernel("primitive-least-residual", 0.0)
    omega = _run(kernel, fx, rho_star, 1.0, fields)[4]
    closed = _run(kernel, fx, rho_star, 1.0, _state_fields(fx, vorticity=omega))
    phi_scale = float(np.max(np.abs(fx.state.phi)))
    assert _max_abs(closed[3], fx.state.phi) <= 1.0e-6 * phi_scale
    # The residual A phi + tau A q + Omega/rho_star**2 closes for
    # Omega = omega_from_phi(phi).
    balance_scale = float(np.max(np.abs(closed[6][0])))
    assert float(np.max(np.abs(closed[5]))) <= 1.0e-12 * balance_scale
    base_omega = _run(kernel, fx, 1.0, 1.0, fields)[4]
    _assert_scaled(omega, base_omega, rho_star**2, label="omega pairing")


# --------------------------------------------------------------------------
# augmented (simplified-gbs-mpe) multiplier and derived wall vorticity
# --------------------------------------------------------------------------


def test_augmented_multiplier_and_derived_wall_vorticity_scale(monkeypatch):
    """With an identity polarization action the derived wall omega is
    ``rho_star**2 * (-phi - tau q) - rho_star**2 * lambda`` and the recovered
    multiplier of ``Omega = rho_star**2 * omega_1`` equals the rho_star = 1 one
    (it keeps phi units)."""

    fx = _fixture()

    def fake_apply(
        self, values_owned, *, face_bc=None, control_volume_boundary_bc=None,
        project_mean_zero=False,
    ):
        del self, face_bc, control_volume_boundary_bc, project_mean_zero
        return jnp.asarray(values_owned, dtype=jnp.float64)

    monkeypatch.setattr(
        LocalPerpLaplacianInverseSolver, "apply_positive_operator", fake_apply
    )
    rho_values = (1.0, 0.3)

    def face_builder(state, geometry, domain, parameters, **kwargs):
        del kwargs
        return _build_face_bcs(state, geometry, domain, parameters)

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cell_fields):
        errors = []
        multipliers = []
        raw_values = {}
        for rho_star in rho_values:
            rhs = _build_model(fx, cell_fields, rho_star, 1.0, None)
            rhs = replace(
                rhs,
                physical_wall_model_name="simplified-gbs-mpe",
                polarization_operator_form="weighted-symmetric",
                face_bc_builder=face_builder,
            )
            state = FciDrbEBState(
                density, phi, Te, Ti, Vi, Ve, vorticity * rho_star**2
            )
            face_bc = rhs._face_bcs(state)
            multipliers.append(
                rhs.recover_polarization_multiplier(state, face_bc=face_bc)
            )
            zero = rhs._derived_vorticity_face_bc_from_polarization(
                state, state.phi, face_bc, polarization_multiplier=0.0
            )
            shifted = rhs._derived_vorticity_face_bc_from_polarization(
                state, state.phi, face_bc, polarization_multiplier=2.0
            )
            raw_values[rho_star] = zero.value_x
            for mask, z, sh in (
                (zero.mask_x, zero.value_x, shifted.value_x),
                (zero.mask_y, zero.value_y, shifted.value_y),
                (zero.mask_z, zero.value_z, shifted.value_z),
            ):
                errors.append(
                    jnp.max(jnp.abs(jnp.where(mask, sh - z + 2.0 * rho_star**2, 0.0)))
                )
        reference = raw_values[1.0]
        for rho_star in rho_values:
            errors.append(
                jnp.max(jnp.abs(raw_values[rho_star] - rho_star**2 * reference))
            )
        spread = jnp.max(jnp.abs(jnp.stack(multipliers) - multipliers[0]))
        scale = jnp.maximum(jnp.max(jnp.abs(jnp.stack(multipliers))), 1.0e-30)
        return (
            jax.lax.pmax(jnp.max(jnp.stack(errors)), axis_name=("x", "y", "z")),
            jax.lax.pmax(spread / scale, axis_name=("x", "y", "z")),
        )

    with jax.disable_jit(False):
        mapped = jax.jit(
            jax.shard_map(
                kernel,
                mesh=fx.mesh,
                in_specs=(PARTITION,) * 8,
                out_specs=(P(), P()),
                check_vma=False,
            )
        )
        arrays = [
            jax.device_put(jnp.asarray(value, dtype=jnp.float64), fx.sharding)
            for value in _state_fields(fx)
        ]
        wall_error, multiplier_spread = mapped(*arrays, fx.cell_fields)
    assert float(wall_error) <= 1.0e-13
    # lambda is a phi-unit quantity: identical for Omega = rho_star**2 omega_1.
    assert float(multiplier_spread) <= 1.0e-13


# --------------------------------------------------------------------------
# 3. particle conservation on a closed configuration
# --------------------------------------------------------------------------


def _conservation_integrals(fx, rho_star, fields):
    terms = _run(_terms_kernel(fx.shape), fx, rho_star, 0.0, fields)
    geometry = fx.context.geometry
    volume = np.asarray(
        geometry.cell_metric.J
        * geometry.spacing.dx
        * geometry.spacing.dy
        * geometry.spacing.dz
    )
    density_names = RHS_TERM_NAMES[0]
    bracket = float(np.sum(volume * terms[0, density_names.index("poisson_bracket")]))
    curvature = float(np.sum(volume * terms[0, density_names.index("curvature")]))
    return bracket, curvature


def test_perpendicular_particle_conservation_scales_with_rho_star():
    """Closed configuration: ``phi = 0`` (equipotential) and ``n = 1`` on the
    radial walls, periodic ``theta`` and ``zeta``, tau = 0, uniform Te = Ti = 1.

    The volume integral of the perpendicular density RHS is
    ``S_bracket + S_curvature`` where ``S_bracket = int n div(v_E)`` (advective
    bracket) and ``S_curvature ~ -int n div(v_E)`` (compression plus the
    divergence-free diamagnetic flux).  A rho_star-consistent model has all
    three proportional to ``rho_star``, so the relative conservation defect is
    rho_star-independent.  (The earlier bracket/rho_star with rho_star-free
    curvature left ``(1/rho_star - 1) * S_bracket`` uncancelled; that old
    arithmetic is evaluated below from the rho_star = 1 integrals only, to
    document the contrast.)  The fixture is the 16x24x16 shifted torus, where
    the discrete cancellation is ~3 percent (second order in h)."""

    shape = (16, 24, 16)
    fx = _fixture(shape)
    state = fx.state
    fields = _state_fields(
        fx,
        density=1.0 + 20.0 * (np.asarray(state.density) - 1.0),
        phi=100.0 * np.asarray(state.phi),
        Te=np.ones(shape),
        Ti=np.ones(shape),
        Vi=np.zeros(shape),
        Ve=np.zeros(shape),
        vorticity=np.zeros(shape),
    )
    b1, c1 = _conservation_integrals(fx, 1.0, fields)
    assert abs(b1) > 1.0e-3 and abs(c1) > 1.0e-3
    relative_one = abs(b1 + c1) / (abs(b1) + abs(c1))
    assert relative_one < 0.05

    for rho_star in (0.05, 0.3):
        b, c = _conservation_integrals(fx, rho_star, fields)
        assert b == pytest.approx(rho_star * b1, rel=1.0e-12)
        assert c == pytest.approx(rho_star * c1, rel=1.0e-12)
        assert (b + c) == pytest.approx(rho_star * (b1 + c1), rel=1.0e-10)
        relative = abs(b + c) / (abs(b) + abs(c))
        assert relative == pytest.approx(relative_one, rel=1.0e-10)
        # Documentation of the old placement (bracket / rho_star, curvature
        # unscaled), from the rho_star = 1 integrals: O(1) defect.
        old_b, old_c = b1 / rho_star, c1
        assert abs(old_b + old_c) / (abs(old_b) + abs(old_c)) > 10.0 * relative
