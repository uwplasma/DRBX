"""P05 / P05N JAX bracket operator on the synthetic perpendicular world (P08 step 2b, E5; fast).

The bounded closure of ``tests.perpendicular_synthetic`` (conditioned cells / common / side rows, Neumann
rows, a wall face with a missing upper side) is lowered into a plan; the centered part and the live jump of
``p05_terms`` / ``p05n_action`` are compared with the ``scripts/p_shared`` host functions (``apply.p05_*``)
plus the replay's own scatters (``_sparse_scatter`` / ``_sparse_scatter_signed``) on the host D/N
reconstructions of the same row objects. Eager == jit bitwise; the centered JVP equals the bilinear
derivative; the jump JVP matches a central finite difference (away from speed sign changes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native.fci_perpendicular_midpoint_bracket import pair_actions, point_bracket, project_raw_to_owners
from drbx.native.fci_perpendicular_p05_operator import (
    p05_action, p05_face_jump, p05_terms, p05_terms_from_state, p05n_action)
from drbx.native.fci_perpendicular_reconstruction_state import (
    boundary_data_from_callables, cell_state, face_state, zero_boundary_data)
from tests.perpendicular_synthetic import NF, Boundary, host_cells, host_faces, lower_world, make_world

OWNERS = [0, 1, 2, 40, 96, 99, 100, 114]
PAIRS = ((0, 1), (1, 2), (2, 0), (0, 2), (1, 1))
DKINDS = ("dirichlet",) * NF
# P05N roles: physical columns (repeats allowed) and per-role kinds; N pairs and their D counterparts
ROLE_COLUMNS = (0, 1, 2, 1, 0)
ROLE_KINDS = ("neumann", "neumann", "dirichlet", "dirichlet", "dirichlet")
N_PAIRS = ((0, 1), (1, 2), (0, 2))
D_PAIRS = ((4, 3), (3, 2), (4, 2))
TOL = 1e-12


@pytest.fixture(scope="module")
def world():
    return make_world(owners=OWNERS)


@pytest.fixture(scope="module")
def plan(world):
    return lower_world(world)


@pytest.fixture(scope="module")
def boundary():
    return Boundary()


@pytest.fixture(scope="module")
def fields(world):
    return np.random.default_rng(5).normal(size=(world.n_owners, NF))


@pytest.fixture(scope="module")
def bc(plan, boundary):
    return boundary_data_from_callables(plan, boundary.dirichlet, boundary.normal)


@pytest.fixture(scope="module")
def host(world, fields, boundary):
    cells = host_cells(world, fields, boundary)
    faces = host_faces(world, fields, boundary)
    return cells, faces


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _close(a, b, tol):
    assert _rel(a, b) <= tol, _rel(a, b)


def _dense(pair, n_owners):
    uniq, values = pair
    out = np.zeros((n_owners,) + values.shape[1:])
    out[uniq] = values
    return out


def _host_centered(world, gradients, pairs):
    from p_shared import apply as A
    from p_shared.replay_units import _sparse_scatter
    g = world.geometry
    raw_action, antisym = A.p05_pair_actions(g.p05_raw_h, np.abs(g.p05_raw_jacobian), gradients, pairs)
    owner_ids = world.grid.raw_to_owner[world.raw_ids]
    return _dense(_sparse_scatter(raw_action, world.raw_volume[world.raw_ids], owner_ids), world.n_owners), antisym


def _host_jump(world, common_grad, lower, upper, pairs, mask):
    """Per-face host jump on the masked faces + the replay's signed scatter (numerators)."""
    from p_shared import apply as A
    from p_shared.replay_units import _sparse_scatter_signed
    g, census = world.geometry, world.census
    rows = world.face_rows
    axis = census.axis[rows].astype(np.int64)
    sel = np.flatnonzero(mask)
    live = np.zeros((len(rows), len(pairs)))
    for i in sel:
        live[i] = np.asarray(A.p05_face_jump(common_grad[i][None], lower[i][None], upper[i][None], g.p05_face_h[i][None],
                                             g.p06_face_weight[i][None], np.array([axis[i]]), np.asarray(pairs)))[0]
    num = _dense(_sparse_scatter_signed(live[sel], census.owner_lo[rows][sel], census.owner_hi[rows][sel], 1.0, -1.0),
                 world.n_owners)
    return live, num


def _world_mask(world):
    return world.census.p07_id[world.face_rows] >= 0


# --------------------------------------------------------------------------
# Plain P05 vs the host
# --------------------------------------------------------------------------

def test_p05_centered_and_jump_match_host(world, plan, fields, bc, host):
    (cd, _cn), fh = host
    mask = _world_mask(world)
    assert (mask == plan.faces.p07_valid).all() and mask.any() and (~mask).any()
    # the domain contains a wall face with a missing upper side, and conditioned rows of every kind
    assert (mask & ~plan.faces.upper_present).any() and plan.cells.conditioned.any()
    assert plan.faces.common_conditioned.any() and (plan.faces.lower_conditioned | plan.faces.upper_conditioned).any()
    t = p05_terms(plan, fields, bc, DKINDS, PAIRS)
    ref_c, ref_antisym = _host_centered(world, cd[1], PAIRS)
    live, ref_j = _host_jump(world, fh["cgd"], fh["ld"], fh["ud"], PAIRS, mask)
    vol = world.owner_volume
    assert _rel(t.centered_numerator, ref_c) < TOL
    assert _rel(t.centered_owner, ref_c / vol[:, None]) < TOL
    assert _rel(t.jump_numerator, ref_j) < TOL
    assert _rel(t.jump_owner, ref_j / vol[:, None]) < TOL
    assert _rel(t.face_jump, live) < TOL and np.max(np.abs(live)) > 0
    assert np.all(np.asarray(t.face_jump)[~mask] == 0.0)
    assert float(t.antisymmetry) == pytest.approx(ref_antisym, abs=1e-12)
    centered, jump = p05_action(plan, fields, bc, DKINDS, PAIRS)
    np.testing.assert_array_equal(np.asarray(centered), np.asarray(t.centered_owner))
    np.testing.assert_array_equal(np.asarray(jump), np.asarray(t.jump_owner))
    np.testing.assert_array_equal(np.asarray(p05_face_jump(plan, fields, bc, DKINDS, PAIRS)), np.asarray(t.face_jump))
    # numerators recover the normalized terms bitwise and the E1 projection is the centered building block
    np.testing.assert_array_equal(np.asarray(t.centered_numerator) / vol[:, None], np.asarray(t.centered_owner))
    np.testing.assert_array_equal(np.asarray(t.jump_numerator) / vol[:, None], np.asarray(t.jump_owner))
    cs = cell_state(plan, fields, bc, DKINDS, values=False)
    c = plan.cells
    raw, _ = pair_actions(c.h, c.jac, cs.gradient, PAIRS)
    np.testing.assert_array_equal(np.asarray(project_raw_to_owners(raw, c.raw_volume, c.raw_owner, c.owner_volume)),
                                  np.asarray(t.centered_owner))
    # numbers, no need for the closure to be trivial
    assert np.max(np.abs(t.centered_owner)) > 1e-3 and np.max(np.abs(t.jump_owner)) > 1e-3


def test_p05_jump_mask_and_validation(world, plan, fields, bc, host):
    """A custom ``jump_mask`` is the host's face selection; masked-out faces contribute exactly nothing."""
    (_cd, _cn), fh = host
    mask = _world_mask(world) & (np.arange(len(world.face_rows)) % 2 == 0)
    live, ref_j = _host_jump(world, fh["cgd"], fh["ld"], fh["ud"], PAIRS, mask)
    _c, jump = p05_action(plan, fields, bc, DKINDS, PAIRS, jump_mask=mask)
    assert _rel(np.asarray(jump) * world.owner_volume[:, None], ref_j) < TOL
    with pytest.raises(ValueError):
        p05_action(plan, fields, bc, DKINDS, PAIRS, jump_mask=mask[:-1])
    with pytest.raises(ValueError):
        p05_action(plan, fields, bc, DKINDS, ())
    with pytest.raises(ValueError):
        p05_action(plan, fields, bc, DKINDS, ((0, NF),))
    with pytest.raises(ValueError):
        p05_action(lower_world(world, include=("p07",)), fields, bc, DKINDS, PAIRS)


# --------------------------------------------------------------------------
# P05N vs the host
# --------------------------------------------------------------------------

def _host_role_state(world, host):
    (cd, cn), fh = host
    is_n = np.asarray([k == "neumann" for k in ROLE_KINDS])
    cols = np.asarray(ROLE_COLUMNS)
    grad = np.where(is_n[None, None, :], cn[1][:, :, cols], cd[1][:, :, cols])
    cg = np.where(is_n[None, None, None, :], fh["cgn"][..., cols], fh["cgd"][..., cols])
    lower = np.where(is_n[None, None, :], fh["ln"][..., cols], fh["ld"][..., cols])
    upper = np.where(is_n[None, None, :], fh["un"][..., cols], fh["ud"][..., cols])
    return grad, cg, lower, upper


def test_p05n_matches_host(world, plan, fields, bc, host):
    from p_shared import apply as A
    from p_shared.replay_units import _sparse_scatter
    grad, cg, lower, upper = _host_role_state(world, host)
    g = world.geometry
    P = len(N_PAIRS)
    owner_ids = world.grid.raw_to_owner[world.raw_ids]
    rv = world.raw_volume[world.raw_ids]

    def raw(pairs):
        action = np.stack([A.p05_bracket(g.p05_raw_h, np.abs(g.p05_raw_jacobian), grad[:, :, a], grad[:, :, b])
                           for a, b in pairs], axis=1)
        return _dense(_sparse_scatter(action, rv, owner_ids), world.n_owners)

    all_faces = np.ones(len(world.face_rows), dtype=bool)
    live, jump_num = _host_jump(world, cg, lower, upper, N_PAIRS + D_PAIRS, all_faces)
    assert (~plan.faces.upper_present).any()                     # the missing-side (wall) face is in the domain
    t = p05n_action(plan, fields, bc, ROLE_KINDS, N_PAIRS, D_PAIRS, columns=ROLE_COLUMNS)
    vol = world.owner_volume[:, None]
    ref = dict(raw_N=raw(N_PAIRS), raw_D=raw(D_PAIRS), face_N=jump_num[:, :P], face_D=jump_num[:, P:])
    for name, num in ref.items():
        assert _rel(getattr(t, name + "_numerator"), num) < TOL, name
        assert _rel(getattr(t, name), num / vol) < TOL, name
        assert np.max(np.abs(num)) > 1e-3, name
    assert _rel(t.face_jump_N, live[:, :P]) < TOL and _rel(t.face_jump_D, live[:, P:]) < TOL
    # the Neumann roles genuinely differ from the Dirichlet ones
    assert np.max(np.abs(np.asarray(t.raw_N) - np.asarray(t.raw_D))) > 1e-3
    # role-column inputs (columns=None) give the same result; the missing side copies the other side's N value
    cols = np.asarray(ROLE_COLUMNS)
    t2 = p05n_action(plan, fields[:, cols], type(bc)(bc.dirichlet_value[:, cols], bc.dirichlet_tangential[..., cols],
                                                      bc.neumann_normal[:, cols]), ROLE_KINDS, N_PAIRS, D_PAIRS)
    for a, b in zip(t, t2):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # the concatenated-pair P05 entry is the same computation
    t3 = p05_terms(plan, fields[:, cols], type(bc)(bc.dirichlet_value[:, cols], bc.dirichlet_tangential[..., cols],
                                                   bc.neumann_normal[:, cols]), ROLE_KINDS, N_PAIRS + D_PAIRS,
                   jump_mask=all_faces)
    np.testing.assert_array_equal(np.asarray(t3.centered_owner)[:, :P], np.asarray(t.raw_N))
    np.testing.assert_array_equal(np.asarray(t3.jump_owner)[:, P:], np.asarray(t.face_D))
    with pytest.raises(ValueError):
        p05n_action(plan, fields, bc, ROLE_KINDS, N_PAIRS, D_PAIRS[:2], columns=ROLE_COLUMNS)
    with pytest.raises(ValueError):
        p05n_action(plan, fields, bc, ROLE_KINDS[:-1], N_PAIRS, D_PAIRS, columns=ROLE_COLUMNS)


def test_terms_from_state_is_the_same_computation(plan, fields, bc):
    t = p05_terms(plan, fields, bc, DKINDS, PAIRS)
    cs = cell_state(plan, fields, bc, DKINDS, values=False)
    fs = face_state(plan, fields, bc, DKINDS)
    t2 = p05_terms_from_state(plan, cs.gradient, fs.gradient, fs.lower, fs.upper, PAIRS)
    for a, b in zip(t, t2):        # not bitwise: XLA fuses the reconstruction into the operator in the one-jit path
        _close(a, b, 1e-14)


# --------------------------------------------------------------------------
# Eager/jit and derivatives
# --------------------------------------------------------------------------

def test_eager_equals_jit_bitwise(plan, fields, bc):
    eager = p05_terms(plan, fields, bc, DKINDS, PAIRS)
    jitted = jax.jit(lambda p, f, b: p05_terms(p, f, b, DKINDS, PAIRS))(plan, fields, bc)
    for a, b in zip(eager, jitted):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    eager_n = p05n_action(plan, fields, bc, ROLE_KINDS, N_PAIRS, D_PAIRS, columns=ROLE_COLUMNS)
    jit_n = jax.jit(lambda p, f, b: p05n_action(p, f, b, ROLE_KINDS, N_PAIRS, D_PAIRS, columns=ROLE_COLUMNS))(
        plan, fields, bc)
    for a, b in zip(eager_n, jit_n):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _jvp_and_fd(f, fields, seed=3, eps=1e-6):
    tangent = np.random.default_rng(seed).normal(size=fields.shape)
    _, jvp = jax.jvp(f, (jnp.asarray(fields),), (jnp.asarray(tangent),))
    fd = (f(jnp.asarray(fields + eps * tangent)) - f(jnp.asarray(fields - eps * tangent))) / (2 * eps)
    return np.asarray(jvp), np.asarray(fd), tangent


def test_jvp_centered_is_the_bilinear_derivative_and_matches_fd(plan, fields, bc):
    jvp, fd, tangent = _jvp_and_fd(lambda x: p05_action(plan, x, bc, DKINDS, PAIRS)[0], fields)
    c = plan.cells
    g = cell_state(plan, fields, bc, DKINDS, values=False).gradient
    dg = cell_state(plan, tangent, zero_boundary_data(plan, NF), DKINDS, values=False).gradient
    raw = np.stack([np.asarray(point_bracket(c.h, c.jac, dg[:, :, a], g[:, :, b])
                               + point_bracket(c.h, c.jac, g[:, :, a], dg[:, :, b])) for a, b in PAIRS], axis=1)
    exact = np.asarray(project_raw_to_owners(raw, c.raw_volume, c.raw_owner, c.owner_volume))
    assert _rel(jvp, exact) < 1e-12
    assert _rel(jvp, fd) < 1e-6


def test_jvp_jump_matches_finite_difference(plan, fields, bc):
    jvp, fd, _ = _jvp_and_fd(lambda x: p05_action(plan, x, bc, DKINDS, PAIRS)[1], fields)
    assert np.max(np.abs(jvp)) > 1e-3
    assert _rel(jvp, fd) < 1e-6


def test_jvp_p05n_matches_finite_difference(plan, fields, bc):
    def f(x):
        t = p05n_action(plan, x, bc, ROLE_KINDS, N_PAIRS, D_PAIRS, columns=ROLE_COLUMNS)
        return jnp.stack([t.raw_N, t.raw_D, t.face_N, t.face_D])   # (4, n_owners, P)

    jvp, fd, _ = _jvp_and_fd(f, fields)
    for index, name in enumerate(("raw_N", "raw_D", "face_N", "face_D")):
        assert np.max(np.abs(jvp[index])) > 1e-3, name
        assert _rel(jvp[index], fd[index]) < 1e-6, name
