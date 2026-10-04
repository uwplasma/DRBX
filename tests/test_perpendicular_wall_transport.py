"""Characteristic (upwind) wall closure of the P05 bracket, ``PerpendicularParams(wall_transport="characteristic")`` (fast).

At the outer wall (radial rings ``n - 2`` and ``n - 1``) the R1 cell rows of a Dirichlet-kind field are one-sided cubics
through the pinned wall value; where the E x B transport carries the field into the wall (``U^u < 0``, the bracket being
``dg/dt = U . grad g / (|J| rho_star)`` with ``U = -h x grad phi``) that stencil leans downwind. The closure replaces, at those
cells and for the Dirichlet-kind columns, the radial component of the cell gradient in the ``(phi, g)`` bracket by the
interior one-sided cubic on the cell's own radial column (rings ``n - 4 .. n - 1``, no wall value).

* the column stencils (``CellPlan.wall_cells`` / ``wall_donors`` / ``wall_weights``): weights from the Lagrange derivative,
  exact for cubics, ``None`` unless the wall columns are full rings, empty without wall cells;
* the default ``"dirichlet"`` is the unmodified RHS (bitwise, with or without the stencils in the plan);
* ``"characteristic"``: the Dirichlet-kind brackets at the outflow wall cells equal the bracket of a hand-computed gradient
  (an independent Vandermonde stencil, ``pair_actions``); inflow cells, non-wall cells, Neumann columns, the jump, P06 and P07
  are bitwise unchanged; the explicit ``raw_pairs`` keep the unmodified gradient; the RHS is continuous across
  ``U^u = 0`` and its JVP matches central finite differences away from it;
* the sharded plan localizes and pads the stencils; the sharded RHS equals the single-device one (multi-device cases run in a
  subprocess, ``tests/perpendicular_wall_transport_case.py``).

The synthetic world is the plane-structured one of ``tests.perpendicular_synthetic`` with full rings on the four outermost
rings (``tests.perpendicular_wall_transport_case``).
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drbx.native.fci_perpendicular_midpoint_bracket import pair_actions, project_raw_to_owners
from drbx.native.fci_perpendicular_p06_operator import bc_columns
from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables, cell_state
from drbx.native.fci_perpendicular_rhs import (
    FIELDS, PHI, WALL_TRANSPORTS, PerpendicularParams, _characteristic_wall_gradient, perpendicular_rhs)
from drbx.native.fci_perpendicular_sharding import (
    local_plan, make_plane_mesh, plane_major_permutation, shard_boundary_data, shard_perpendicular_plan,
    sharded_perpendicular_rhs, to_plane_major)
from drbx.stencils.operator_plan import (
    WALL_COLUMN, WALL_RINGS, CellPlan, lagrange_derivative_weights, wall_column_rows)
from tests import perpendicular_wall_transport_case as case
from tests.perpendicular_synthetic import lower_world, make_world

N = 8
KINDS = case.KINDS["mixed"]                                   # density N, Te N, Ti D, vorticity D, phi D
KINDS_TUPLE = tuple(KINDS[c] for c in (*FIELDS, PHI))
DIRICHLET_FIELDS = [i for i, f in enumerate(FIELDS) if KINDS[f] == "dirichlet"]      # columns Ti, vorticity
RHO = case.RHO
PHI_COL = len(FIELDS)
BRACKET = "poisson_bracket"
CASE = Path(case.__file__).resolve()


def _scale(a):
    return float(np.max(np.abs(np.asarray(a))))


def _same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built():
    world, raw_to_owner = case.make_wall_world(N)
    return world, raw_to_owner, lower_world(world)


@pytest.fixture(scope="module")
def world(built):
    return built[0]


@pytest.fixture(scope="module")
def raw_to_owner(built):
    return built[1]


@pytest.fixture(scope="module")
def plan(built):
    return built[2]


@pytest.fixture(scope="module")
def inputs(world, plan):
    return case.wall_inputs(world, plan)                      # state, phi, bc5, params


@pytest.fixture(scope="module")
def char(inputs):
    return dataclasses.replace(inputs[3], wall_transport="characteristic")


@pytest.fixture(scope="module")
def rhs_default(plan, inputs):
    state, phi, bc5, params = inputs
    return perpendicular_rhs(plan, state, phi, bc5, KINDS, params)


@pytest.fixture(scope="module")
def rhs_char(plan, inputs, char):
    state, phi, bc5, _ = inputs
    return perpendicular_rhs(plan, state, phi, bc5, KINDS, char)


@pytest.fixture(scope="module")
def stacked(inputs):
    state, phi, _bc, _params = inputs
    return np.stack([state[f] for f in FIELDS] + [phi], axis=1)


def _wall_view(plan, stacked_fields, bc5):
    """The wall cells with the plan's own cell gradient, ``U^u`` (``U = -h x grad phi``) and the outflow mask (host side)."""
    cells = plan.cells
    gradient = np.asarray(cell_state(plan, stacked_fields, bc5, KINDS_TUPLE, values=False, gradients=True).gradient)
    ring = np.asarray(cells.raw_ids) // N ** 2
    index = np.flatnonzero(ring >= N - WALL_RINGS)
    u_radial = -np.cross(np.asarray(cells.h)[index], gradient[index, :, PHI_COL])[:, 0]
    return SimpleNamespace(index=index, ring=ring[index], gradient=gradient, u_radial=u_radial, outflow=u_radial < 0)


@pytest.fixture(scope="module")
def wall(plan, stacked, inputs):
    return _wall_view(plan, stacked, inputs[2])


# --------------------------------------------------------------------------
# Column stencils
# --------------------------------------------------------------------------

def test_lagrange_derivative_weights_are_exact_for_polynomials():
    rng = np.random.default_rng(1)
    for count in (2, 3, 4, 5):
        x = np.sort(rng.uniform(-1.0, 2.0, size=count))
        coeff = rng.normal(size=count)                                           # a polynomial of degree count - 1
        value = lambda t: np.polynomial.polynomial.polyval(t, coeff)
        slope = lambda t: np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(coeff))
        for at in range(count):
            w = lagrange_derivative_weights(x, at)
            assert abs(w @ value(x) - slope(x[at])) <= 1e-11 * max(1.0, abs(slope(x[at])))
            assert abs(w.sum()) <= 1e-12 * np.abs(w).max()                       # the derivative of a constant is 0
    with pytest.raises(ValueError, match="distinct"):
        lagrange_derivative_weights([0.0, 1.0, 1.0], 0)
    with pytest.raises(ValueError, match="distinct"):
        lagrange_derivative_weights([0.0, 1.0], 2)


@pytest.mark.parametrize("n", (8, 12, 32))
def test_wall_column_rows_weights_reproduce_the_radial_derivative_of_cubics(n):
    raw_ids = np.arange(n ** 3)
    owners = np.random.default_rng(2).permutation(n ** 3)                        # one owner per raw cell, any numbering
    rows = wall_column_rows(raw_ids, owners, n)
    cells, donors, weights = rows["wall_cells"], rows["wall_donors"], rows["wall_weights"]
    assert (WALL_RINGS, WALL_COLUMN) == (2, 4)
    assert cells.dtype == donors.dtype == np.int32 and weights.dtype == np.float64
    np.testing.assert_array_equal(cells, np.flatnonzero(raw_ids // n ** 2 >= n - 2))
    assert len(cells) == 2 * n * n and donors.shape == weights.shape == (len(cells), 4)
    ring, j, k = cells // n ** 2, (cells // n) % n, cells % n
    for column in range(4):                                                      # the owners of the column on rings n-4 .. n-1
        np.testing.assert_array_equal(donors[:, column], owners[((n - 4 + column) * n + j) * n + k])
    # the literal stencils (1, -6, 3, 2) / (6 du) on ring n - 2 and (-2, 9, -18, 11) / (6 du) on ring n - 1, du = 1 / n
    np.testing.assert_allclose(weights[ring == n - 2], np.tile(np.array([1.0, -6.0, 3.0, 2.0]) * n / 6, (n * n, 1)),
                               rtol=1e-14, atol=0)
    np.testing.assert_allclose(weights[ring == n - 1], np.tile(np.array([-2.0, 9.0, -18.0, 11.0]) * n / 6, (n * n, 1)),
                               rtol=1e-14, atol=0)
    # exact d/du of any cubic at the cell's own radial centre u = (ring + 1/2) / n
    u = (np.arange(n) + 0.5) / n
    layers = np.arange(n - 4, n)
    rng = np.random.default_rng(3)
    for _ in range(5):
        a = rng.normal(size=4)
        value = lambda t: a[0] + a[1] * t + a[2] * t ** 2 + a[3] * t ** 3
        slope = lambda t: a[1] + 2 * a[2] * t + 3 * a[3] * t ** 2
        np.testing.assert_allclose(weights @ value(u[layers]), slope(u[ring]), rtol=0, atol=1e-11 * n)
    np.testing.assert_allclose(weights.sum(axis=1), 0.0, atol=1e-12 * n)           # constants have no derivative
    np.testing.assert_allclose(weights @ u[layers], 1.0, rtol=1e-12)               # d u / d u


def test_wall_column_rows_needs_full_rings():
    n = 8
    raw_ids = np.arange(n ** 3)
    full = np.arange(n ** 3)
    assert wall_column_rows(raw_ids, full, n) is not None
    for ring in range(n - 4, n):                        # one agglomerated owner anywhere in a column makes it unavailable
        pair = full.copy()
        pair[(ring * n + 3) * n + 5] = pair[(ring * n + 2) * n + 5]
        assert wall_column_rows(raw_ids, pair, n) is None, ring
    inside = full.copy()                                # the rings below the column are not read
    inside[(3 * n + 3) * n + 5] = inside[(3 * n + 2) * n + 5]
    assert wall_column_rows(raw_ids, inside, n) is not None
    ownerless = full.copy()
    ownerless[(5 * n + 1) * n + 1] = -1
    assert wall_column_rows(raw_ids, ownerless, n) is None
    # a bounded plan (raw_ids a subset): only the columns of the wall cells it holds count; the owner map is the global one
    pair = full.copy()
    pair[(5 * n + 2) * n + 6] = pair[(5 * n + 1) * n + 6]                         # agglomerates the column (j, k) = (2, 6)
    assert wall_column_rows(np.array([(6 * n + 2) * n + 6]), pair, n) is None
    assert wall_column_rows(np.array([(7 * n + 2) * n + 6, (6 * n + 4) * n + 6]), pair, n) is None
    elsewhere = wall_column_rows(np.array([(6 * n + 4) * n + 6, (7 * n + 4) * n + 6]), pair, n)
    assert elsewhere is not None and len(elsewhere["wall_cells"]) == 2
    # no wall cell among raw_ids: empty arrays (not None)
    empty = wall_column_rows(np.arange(3 * n * n), pair, n)
    assert empty["wall_cells"].shape == (0,) and empty["wall_donors"].shape == (0, 4) \
        and empty["wall_weights"].shape == (0, 4)
    with pytest.raises(ValueError, match="n\\*\\*3"):
        wall_column_rows(raw_ids, full[:-1], n)
    assert wall_column_rows(np.arange(3 ** 3), np.arange(3 ** 3), 3) is None        # fewer than four radial rings


def test_plan_holds_the_wall_columns(world, plan):
    cells = plan.cells
    expected = wall_column_rows(cells.raw_ids, world.grid.raw_to_owner, N)
    assert expected is not None and len(expected["wall_cells"]) == 2 * N * N
    for name, value in expected.items():
        np.testing.assert_array_equal(getattr(cells, name), value)
    assert (np.asarray(cells.raw_ids)[cells.wall_cells] // N ** 2 >= N - 2).all()
    # the plan pytree carries them as array leaves (a plan without them is the same pytree with None children)
    leaves, tree = jax.tree_util.tree_flatten(plan)
    assert all(hasattr(leaf, "shape") for leaf in leaves)
    assert len(jax.tree_util.tree_leaves(case.without_wall_columns(plan))) == len(leaves) - 3
    again = jax.tree_util.tree_unflatten(tree, leaves)
    np.testing.assert_array_equal(again.cells.wall_donors, cells.wall_donors)
    # a CellPlan constructed without them (an older plan) has None
    old = CellPlan(**{f.name: getattr(cells, f.name) for f in dataclasses.fields(CellPlan)
                      if not f.name.startswith("wall_")})
    assert old.wall_cells is old.wall_donors is old.wall_weights is None


def test_lowering_leaves_the_wall_columns_unset_for_agglomerated_rings():
    aggregated = make_world(owners=[0, 1, 2, 40, 96, 99, 100, 114])             # three raw cells per owner: no full rings
    plan = lower_world(aggregated)
    assert (np.asarray(plan.cells.raw_ids) // aggregated.n ** 2 >= aggregated.n - 2).any()
    assert plan.cells.wall_cells is plan.cells.wall_donors is plan.cells.wall_weights is None


def test_lowering_without_wall_cells_gives_empty_wall_columns():
    world, raw_to_owner = case.make_wall_world(N, owners=[0, 1, 2, 3, 20, 21])         # inner (agglomerated) owners only
    plan = lower_world(world)
    assert (np.asarray(plan.cells.raw_ids) // N ** 2 < N - 2).all()
    assert plan.cells.wall_cells.shape == (0,) and plan.cells.wall_donors.shape == (0, 4)
    assert plan.cells.wall_weights.shape == (0, 4)


# --------------------------------------------------------------------------
# The RHS option
# --------------------------------------------------------------------------

def test_default_wall_transport_is_the_unmodified_rhs_bitwise(plan, inputs, rhs_default):
    state, phi, bc5, params = inputs
    assert PerpendicularParams().wall_transport == "dirichlet" and WALL_TRANSPORTS == ("dirichlet", "characteristic")
    explicit = perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(params, wall_transport="dirichlet"))
    stripped = perpendicular_rhs(case.without_wall_columns(plan), state, phi, bc5, KINDS, params)
    assert case.same_bits(explicit, rhs_default) and case.same_bits(stripped, rhs_default)
    # a plan without the stencils is a valid plan for the other options too (here with explicit raw pairs)
    pairs = [("density", "Te"), ("phi", "Ti")]
    without = perpendicular_rhs(case.without_wall_columns(plan), state, phi, bc5, KINDS, params, raw_pairs=pairs)
    with_columns = perpendicular_rhs(plan, state, phi, bc5, KINDS, params, raw_pairs=pairs)
    assert case.same_bits(without, with_columns)
    assert all(_same(a, b) for a, b in zip(without.raw_pairs, with_columns.raw_pairs))


def test_wall_transport_is_validated_and_needs_the_wall_columns(plan, inputs):
    state, phi, bc5, params = inputs
    with pytest.raises(ValueError, match="wall_transport must be one of"):
        perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(params, wall_transport="upwind"))
    char = dataclasses.replace(params, wall_transport="characteristic")
    stripped = case.without_wall_columns(plan)
    with pytest.raises(ValueError, match="full rings"):
        perpendicular_rhs(stripped, state, phi, bc5, KINDS, char)
    with pytest.raises(ValueError, match="full rings"):
        perpendicular_rhs(stripped, state, phi, bc5, KINDS, char, terms=("bracket",))
    # the option acts on the bracket only: without it the stencils are not needed (and not an error)
    other = perpendicular_rhs(stripped, state, phi, bc5, KINDS, char, terms=("curvature", "diffusion"))
    ref = perpendicular_rhs(stripped, state, phi, bc5, KINDS, params, terms=("curvature", "diffusion"))
    assert _same(other.terms["Ti"]["curvature"], ref.terms["Ti"]["curvature"])
    assert _same(other.terms["Ti"]["perpendicular_diffusion"], ref.terms["Ti"]["perpendicular_diffusion"])
    # a plan lowered from agglomerated rings has no stencils, with the same clear message
    aggregated = make_world(owners=[0, 1, 2, 40, 96, 99, 100, 114])
    agg_plan = lower_world(aggregated)
    agg_fields = 1.0 + 0.4 * np.random.default_rng(5).uniform(size=(aggregated.n_owners, 5))
    boundary = case.PositiveBoundary(8)
    agg_bc = bc_columns(boundary_data_from_callables(agg_plan, boundary.dirichlet, boundary.normal), np.arange(5))
    agg_state = {f: agg_fields[:, i] for i, f in enumerate(FIELDS)}
    with pytest.raises(ValueError, match="wall column stencils"):
        perpendicular_rhs(agg_plan, agg_state, agg_fields[:, 4], agg_bc, KINDS, char, terms=("bracket",))
    default = perpendicular_rhs(agg_plan, agg_state, agg_fields[:, 4], agg_bc, KINDS, params, terms=("bracket",))
    assert np.isfinite(np.asarray(default.terms["Ti"][BRACKET])).all()


def test_transport_direction_convention_of_the_bracket():
    """``dg/dt = +U . grad g / (|J| rho_star)``, ``U = -h x grad phi``: for ``g = u`` (``grad g = e_u``) the bracket is
    ``U^u / |J|``, so the profile moves toward the wall (outflow, ``V^u = -U^u / (|J| rho_star) > 0``) where ``U^u < 0``."""
    rng = np.random.default_rng(7)
    h, jac, gphi = rng.normal(size=(6, 3)), rng.uniform(0.5, 2.0, size=6), rng.normal(size=(6, 3))
    grads = np.stack([gphi, np.tile([1.0, 0.0, 0.0], (6, 1))], axis=2)           # (points, 3, [phi, u])
    action, _ = pair_actions(h, jac, grads, [(0, 1)])
    u_radial = -np.cross(h, gphi)[:, 0]
    np.testing.assert_allclose(np.asarray(action)[:, 0], u_radial / jac, rtol=1e-14, atol=1e-15)


def _hand_gradient(plan, world, view, stacked_fields):
    """The plan's cell gradient with, at the outflow wall cells and the Dirichlet-kind field columns, the radial component
    replaced by an independent Vandermonde stencil on the owners of the cell's column (NumPy; no ``wall_*`` fields)."""
    ro = np.asarray(world.grid.raw_to_owner)
    raw = np.asarray(plan.cells.raw_ids)
    layers = np.arange(N - 4, N)
    vander = np.vander((layers + 0.5) / N, 4, increasing=True)                   # V[l, m] = u_l ** m
    gradient = view.gradient.copy()
    for cell, ring, outflow in zip(view.index, view.ring, view.outflow):
        if not outflow:
            continue
        u = (ring + 0.5) / N
        w = np.linalg.solve(vander.T, np.array([0.0, 1.0, 2 * u, 3 * u ** 2]))   # sum_l w_l u_l^m = d(u^m)/du
        j, k = (raw[cell] // N) % N, raw[cell] % N
        owners = ro[(layers * N + j) * N + k]
        for f in DIRICHLET_FIELDS:
            gradient[cell, 0, f] = w @ stacked_fields[owners, f]
    return gradient


def test_characteristic_bracket_equals_the_hand_computed_column_stencil_at_outflow_cells(
        plan, world, stacked, wall, rhs_default, rhs_char):
    cells = plan.cells
    # both transport directions occur at the wall, on both rings
    assert wall.outflow.any() and (~wall.outflow).any()
    for ring in (N - 2, N - 1):
        assert wall.outflow[wall.ring == ring].any() and (~wall.outflow[wall.ring == ring]).any()
    gradient = _hand_gradient(plan, world, wall, stacked)
    changed = np.flatnonzero((gradient != wall.gradient).any(axis=(1, 2)))
    np.testing.assert_array_equal(changed, wall.index[wall.outflow])
    assert (gradient[..., [0, 1, 4]] == wall.gradient[..., [0, 1, 4]]).all()      # Neumann columns and phi untouched
    assert (gradient[:, 1:] == wall.gradient[:, 1:]).all()                        # only the radial component changes
    action, _ = pair_actions(cells.h, cells.jac, gradient, [(PHI_COL, i) for i in range(len(FIELDS))])
    ones = np.ones_like(np.asarray(cells.owner_volume))
    centered = np.asarray(project_raw_to_owners(action, cells.raw_volume, cells.raw_owner, ones)) \
        / np.asarray(cells.owner_volume)[:, None]
    for i, f in enumerate(FIELDS):                      # roundoff of the weighted sums (observed: 1e-14 of the scale)
        got = np.asarray(rhs_char.detail[f]["bracket_centered"])
        assert np.max(np.abs(got - centered[:, i] / RHO)) <= 1e-12 * _scale(centered[:, i] / RHO), f
        assert _same(rhs_char.detail[f]["bracket_jump"], rhs_default.detail[f]["bracket_jump"]), f
        total = np.asarray(rhs_char.terms[f][BRACKET])
        expected = centered[:, i] / RHO + np.asarray(rhs_default.detail[f]["bracket_jump"])
        assert np.max(np.abs(total - expected)) <= 1e-12 * _scale(expected), f


def test_characteristic_changes_only_the_outflow_wall_owners_of_dirichlet_columns(plan, wall, rhs_default, rhs_char):
    owner = np.asarray(plan.cells.raw_owner)
    assert len(np.unique(owner[wall.index])) == len(wall.index)                  # one raw cell per wall owner (full rings)
    outflow_owners = np.sort(owner[wall.index[wall.outflow]])
    for f in FIELDS:
        differs = np.flatnonzero(np.asarray(rhs_default.terms[f][BRACKET]) != np.asarray(rhs_char.terms[f][BRACKET]))
        if KINDS[f] == "dirichlet":
            np.testing.assert_array_equal(differs, outflow_owners, err_msg=f)     # inflow and interior owners: bitwise
        else:
            assert len(differs) == 0, f                                          # Neumann columns: bitwise
        assert _same(rhs_default.detail[f]["bracket_jump"], rhs_char.detail[f]["bracket_jump"])
        for term in ("curvature", "perpendicular_diffusion"):
            assert _same(rhs_default.terms[f][term], rhs_char.terms[f][term]), (f, term)
        for key in ("curvature_material", "curvature_remainder", "curvature_q1", "curvature_correction"):
            assert _same(rhs_default.detail[f][key], rhs_char.detail[f][key]), (f, key)
    for key in ("spectral_fallback", "floor_hits", "wall_fallback"):
        assert _same(rhs_default.diagnostics[key], rhs_char.diagnostics[key])


def test_characteristic_with_a_field_subset_reads_the_right_columns(plan, inputs, char, rhs_char):
    state, phi, bc5, params = inputs                                             # (vorticity, phi): phi is column 1, not 4
    sub = perpendicular_rhs(plan, state, phi, bc_columns(bc5, [3, 4]), {"vorticity": "dirichlet"}, char,
                            fields=("vorticity",), terms=("bracket",))
    ref = rhs_char.terms["vorticity"][BRACKET]
    assert np.max(np.abs(np.asarray(sub.terms["vorticity"][BRACKET]) - np.asarray(ref))) <= 1e-13 * _scale(ref)
    default = perpendicular_rhs(plan, state, phi, bc_columns(bc5, [3, 4]), {"vorticity": "dirichlet"}, params,
                                fields=("vorticity",), terms=("bracket",))
    assert not _same(sub.terms["vorticity"][BRACKET], default.terms["vorticity"][BRACKET])
    # a Neumann-kind field alone has nothing to replace: the closure is the unmodified bracket (bitwise)
    neumann = perpendicular_rhs(plan, state, phi, bc_columns(bc5, [0, 4]), {"density": "neumann"}, char,
                                fields=("density",), terms=("bracket",))
    neumann_default = perpendicular_rhs(plan, state, phi, bc_columns(bc5, [0, 4]), {"density": "neumann"}, params,
                                        fields=("density",), terms=("bracket",))
    assert _same(neumann.terms["density"][BRACKET], neumann_default.terms["density"][BRACKET])


def test_characteristic_leaves_explicit_raw_pairs_on_the_unmodified_gradient(plan, inputs, char):
    state, phi, bc5, params = inputs
    pairs = [("density", "Te"), ("phi", "Ti"), ("vorticity", "density")]
    default = perpendicular_rhs(plan, state, phi, bc5, KINDS, params, terms=("bracket",), raw_pairs=pairs)
    other = perpendicular_rhs(plan, state, phi, bc5, KINDS, char, terms=("bracket",), raw_pairs=pairs)
    alone = perpendicular_rhs(plan, state, phi, bc5, KINDS, char, terms=("bracket",))
    for name in ("centered_owner", "jump_owner", "centered_numerator", "jump_numerator", "face_jump"):
        a, b = np.asarray(getattr(default.raw_pairs, name)), np.asarray(getattr(other.raw_pairs, name))
        assert a.shape == b.shape and np.max(np.abs(a - b)) <= 1e-13 * max(_scale(a), 1e-300), name
    # the (phi, Ti) raw pair is the plan's own bracket column while the (phi, Ti) bracket term is the closed one
    ti = other.terms["Ti"][BRACKET]
    raw_ti = np.asarray(other.raw_pairs.centered_owner)[:, 1] + np.asarray(other.raw_pairs.jump_owner)[:, 1]
    assert np.max(np.abs(raw_ti / RHO - np.asarray(default.terms["Ti"][BRACKET]))) <= 1e-13 * _scale(raw_ti / RHO)
    assert np.max(np.abs(raw_ti / RHO - np.asarray(ti))) > 1e-6 * _scale(raw_ti / RHO)
    # the bracket terms are those of the call without raw pairs
    for f in FIELDS:
        a, b = np.asarray(other.terms[f][BRACKET]), np.asarray(alone.terms[f][BRACKET])
        assert np.max(np.abs(a - b)) <= 1e-13 * _scale(b), f


def test_padded_wall_entries_do_nothing(plan, stacked, wall):
    """Padding entries (out-of-range cell index, any donor, weight 0) leave the gradient untouched, as in sharded plans."""
    cells = plan.cells
    n_cells = len(cells.raw_ids)
    columns = (2, 3)
    # the last cell is a wall cell (ring n - 1): make it an outflow cell (U^u = -(h x grad phi)_u = -1 < 0), so that a
    # padding entry that landed on it (a clipped index) would overwrite its gradient with the entry's zero stencil
    h = np.asarray(cells.h).copy()
    gradient = np.array(wall.gradient)
    h[-1] = (0.0, 1.0, 0.0)
    gradient[-1, :, PHI_COL] = (0.0, 0.0, 1.0)
    gradient, values = jnp.asarray(gradient), jnp.asarray(stacked)
    as_cells = lambda cell, donor, weight: SimpleNamespace(
        h=jnp.asarray(h), wall_cells=jnp.asarray(cell), wall_donors=jnp.asarray(donor), wall_weights=jnp.asarray(weight))
    base = _characteristic_wall_gradient(as_cells(cells.wall_cells, cells.wall_donors, cells.wall_weights), gradient,
                                         values, PHI_COL, columns)
    assert not _same(base, gradient) and not _same(base[-1], gradient[-1])           # the last cell is switched
    for pad_donor in (0, stacked.shape[0] - 1):
        padded = as_cells(np.concatenate([cells.wall_cells, np.full(5, n_cells, dtype=np.int32)]),
                          np.concatenate([cells.wall_donors, np.full((5, 4), pad_donor, dtype=np.int32)]),
                          np.concatenate([cells.wall_weights, np.zeros((5, 4))]))
        assert _same(_characteristic_wall_gradient(padded, gradient, values, PHI_COL, columns), base)
    # nothing to replace: the same array comes back; no wall entry at all: the gradient is unchanged
    assert _characteristic_wall_gradient(padded, gradient, values, PHI_COL, ()) is gradient
    none = as_cells(np.zeros(0, dtype=np.int32), np.zeros((0, 4), dtype=np.int32), np.zeros((0, 4)))
    assert _same(_characteristic_wall_gradient(none, gradient, values, PHI_COL, columns), gradient)


def test_characteristic_without_wall_cells_in_the_plan_is_the_unmodified_rhs():
    world, _ro = case.make_wall_world(N, owners=[0, 1, 2, 3, 20, 21])
    plan = lower_world(world)
    state, phi, bc5, params = case.wall_inputs(world, plan)
    char = dataclasses.replace(params, wall_transport="characteristic")
    a = perpendicular_rhs(plan, state, phi, bc5, KINDS, params, terms=("bracket",))
    b = perpendicular_rhs(plan, state, phi, bc5, KINDS, char, terms=("bracket",))
    assert case.same_bits(a, b, terms=(BRACKET,))


def _transport(plan, state, phi, bc5):
    """``U^u`` of the wall cells at ``(state, phi)`` (host side)."""
    view = _wall_view(plan, np.stack([state[f] for f in FIELDS] + [phi], axis=1), bc5)
    return view.u_radial


def test_characteristic_rhs_is_continuous_across_the_switch(plan, inputs, char, wall, stacked):
    """The hard switch only changes the radial component, which the bracket multiplies by ``U^u``: across ``U^u = 0`` of a
    wall cell the RHS is continuous (an ``|U^u|`` kink), although the two closures' gradients differ."""
    state, phi, bc5, _params = inputs
    cells = plan.cells
    index = int(np.argmin(np.abs(wall.u_radial)))
    rng = np.random.default_rng(13)
    direction = 0.3 * rng.normal(size=phi.shape)
    u0 = _transport(plan, state, phi, bc5)
    u1 = _transport(plan, state, phi + direction, bc5)
    crossing = -u0[index] / (u1[index] - u0[index])                              # U^u is affine in phi
    cell = wall.index[index]
    w = int(np.flatnonzero(np.asarray(cells.wall_cells) == cell)[0])
    interior = np.asarray(cells.wall_weights)[w] @ stacked[np.asarray(cells.wall_donors)[w], 3]
    assert abs(interior - wall.gradient[cell, 0, 3]) > 1e-3 * abs(wall.gradient[cell, 0, 3])     # the closures differ
    eps = 1e-9
    below, above = (perpendicular_rhs(plan, state, phi + (crossing + s * eps) * direction, bc5, KINDS, char,
                                      terms=("bracket",)) for s in (-1, 1))
    assert _transport(plan, state, phi + (crossing - eps) * direction, bc5)[index] * \
        _transport(plan, state, phi + (crossing + eps) * direction, bc5)[index] < 0              # the switch is crossed
    for f in ("Ti", "vorticity"):
        a, b = np.asarray(below.terms[f][BRACKET]), np.asarray(above.terms[f][BRACKET])
        assert np.max(np.abs(a - b)) <= 1e-6 * _scale(a), f


def test_characteristic_jvp_matches_central_finite_differences(plan, inputs, char):
    state, phi, bc5, _params = inputs
    rng = np.random.default_rng(9)
    x = jnp.stack([jnp.asarray(state[f]) for f in FIELDS] + [jnp.asarray(phi)], axis=1)
    tangent = jnp.asarray(rng.normal(size=x.shape))
    h = 1e-6

    def fun(z):
        out = perpendicular_rhs(plan, {f: z[:, i] for i, f in enumerate(FIELDS)}, z[:, 4], bc5, KINDS, char)
        return {f: dict(out.terms[f]) for f in FIELDS}

    def pattern(z):
        return _transport(plan, {f: np.asarray(z[:, i]) for i, f in enumerate(FIELDS)}, np.asarray(z[:, 4]), bc5) < 0

    assert np.array_equal(pattern(x + h * tangent), pattern(x - h * tangent))     # away from U^u = 0
    assert np.array_equal(pattern(x + h * tangent), pattern(x))
    _, jvp = jax.jvp(fun, (x,), (tangent,))
    plus, minus = fun(x + h * tangent), fun(x - h * tangent)
    errors = {}
    for f in FIELDS:
        for t in ("poisson_bracket", "curvature", "perpendicular_diffusion"):
            fd = (plus[f][t] - minus[f][t]) / (2 * h)
            errors[(f, t)] = float(jnp.max(jnp.abs(jvp[f][t] - fd)) / max(float(jnp.max(jnp.abs(fd))), 1e-300))
            assert float(jnp.max(jnp.abs(jvp[f][t]))) > 0
    assert max(errors.values()) <= 1e-6, errors
    # the closure is active in the tangent: the characteristic JVP of a Dirichlet-kind bracket differs from the default one
    default = dataclasses.replace(char, wall_transport="dirichlet")

    def fun_default(z):
        out = perpendicular_rhs(plan, {f: z[:, i] for i, f in enumerate(FIELDS)}, z[:, 4], bc5, KINDS, default,
                                terms=("bracket",))
        return out.terms["vorticity"][BRACKET]

    _, jvp_default = jax.jvp(fun_default, (x,), (tangent,))
    assert float(jnp.max(jnp.abs(jvp_default - jvp["vorticity"][BRACKET]))) > 1e-3 * float(jnp.max(jnp.abs(jvp_default)))


# --------------------------------------------------------------------------
# Sharded plan and RHS
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_shards, uneven", ((1, False), (2, False), (2, True)))
def test_sharded_plan_localizes_and_pads_the_wall_columns(raw_to_owner, plan, n_shards, uneven):
    cells = plan.cells
    if uneven:                                      # no wall entry for the planes of the first shard: it pads all of its own
        planes = np.asarray(cells.raw_ids)[cells.wall_cells] % N
        keep = planes >= N // n_shards
        cells = dataclasses.replace(cells, wall_cells=cells.wall_cells[keep], wall_donors=cells.wall_donors[keep],
                                    wall_weights=cells.wall_weights[keep])
        plan = dataclasses.replace(plan, cells=cells)
    sharded = shard_perpendicular_plan(plan, raw_to_owner, N, n_shards)
    _perm, inverse, m = plane_major_permutation(raw_to_owner, N)
    p, halo, trash = sharded.planes_per_shard, sharded.halo, sharded.local_rows - 1
    global_raw = np.asarray(cells.raw_ids)[cells.wall_cells]
    expected = np.array([((global_raw % N) // p == s).sum() for s in range(n_shards)])
    width = np.asarray(sharded.plan.cells.wall_cells).shape[1]
    assert width == expected.max() and np.asarray(sharded.plan.cells.wall_donors).shape == (n_shards, width, 4)
    assert (expected == 0).any() == uneven
    kept = []
    for s in range(n_shards):
        local = local_plan(sharded, s).cells
        n_local = len(local.raw_ids)
        wc = np.asarray(local.wall_cells)
        real = np.arange(width) < expected[s]                                       # the real entries come first
        assert (wc[real] < n_local).all() and (wc[~real] == n_local).all()          # padding: the out-of-range cell index
        assert (np.asarray(local.wall_donors)[~real] == trash).all()                # ... trash donors, zero weight
        assert (np.asarray(local.wall_weights)[~real] == 0).all()
        raw = np.asarray(local.raw_ids)[wc[real]]
        kept.append(raw)
        ring = raw // N ** 2
        assert (ring >= N - WALL_RINGS).all() and ((raw % N) // p == s).all()      # kept with its owner's eta plane
        donors = np.asarray(local.wall_donors)[real]
        own = donors[np.arange(len(raw)), ring - (N - WALL_COLUMN)]
        np.testing.assert_array_equal(own, np.asarray(local.raw_owner)[wc[real]])   # the donor of its own ring is the owner
        assert ((donors >= halo * m) & (donors < (halo + p) * m)).all()             # owned local rows (same eta plane)
        position = np.searchsorted(global_raw, raw)
        np.testing.assert_array_equal(inverse[s * p * m + donors - halo * m], np.asarray(cells.wall_donors)[position])
        np.testing.assert_array_equal(np.asarray(local.wall_weights)[real], np.asarray(cells.wall_weights)[position])
        assert wc.dtype == cells.wall_cells.dtype and np.asarray(local.wall_donors).dtype == cells.wall_donors.dtype
    np.testing.assert_array_equal(np.sort(np.concatenate(kept)), global_raw)         # every wall cell on exactly one shard
    stripped = shard_perpendicular_plan(case.without_wall_columns(plan), raw_to_owner, N, n_shards)
    assert stripped.plan.cells.wall_cells is stripped.plan.cells.wall_donors is stripped.plan.cells.wall_weights is None


def test_sharded_plan_asserts_that_the_wall_columns_stay_in_the_owning_shard(raw_to_owner, plan):
    perm, _inverse, m = plane_major_permutation(raw_to_owner, N)
    cells = plan.cells
    first_plane = int(np.flatnonzero(np.asarray(cells.raw_ids)[cells.wall_cells] % N == 0)[0])
    far = int(np.flatnonzero(perm // m == N // 2 + 1)[0])                           # an owner of the other shard's planes
    donors = np.asarray(cells.wall_donors).copy()
    donors[first_plane, 0] = far
    broken = dataclasses.replace(plan, cells=dataclasses.replace(cells, wall_donors=donors))
    with pytest.raises(ValueError, match="not owned by the shard"):
        shard_perpendicular_plan(broken, raw_to_owner, N, 2)


def test_sharded_rhs_needs_the_wall_columns_for_the_characteristic_closure(raw_to_owner, plan, inputs, char):
    state, phi, bc5, _params = inputs
    sharded = shard_perpendicular_plan(case.without_wall_columns(plan), raw_to_owner, N, 1)
    _perm, inverse, _m = plane_major_permutation(raw_to_owner, N)
    state_pm = {k: jnp.asarray(to_plane_major(v, inverse)) for k, v in state.items()}
    with pytest.raises(ValueError, match="wall column stencils"):
        sharded_perpendicular_rhs(sharded, state_pm, jnp.asarray(to_plane_major(phi, inverse)),
                                  shard_boundary_data(bc5, sharded), KINDS, char, make_plane_mesh(1), terms=("bracket",))
    with pytest.raises(ValueError, match="wall_transport must be one of"):
        sharded_perpendicular_rhs(sharded, state_pm, jnp.asarray(to_plane_major(phi, inverse)),
                                  shard_boundary_data(bc5, sharded), KINDS,
                                  dataclasses.replace(char, wall_transport="upwind"), make_plane_mesh(1))


def _subprocess(*args, devices: int = 4, timeout: int = 900) -> dict:
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={devices}".strip()
    env["JAX_PLATFORMS"] = "cpu"
    done = subprocess.run([sys.executable, str(CASE), *args], capture_output=True, text=True, env=env,
                          timeout=timeout, check=False)
    assert done.returncode == 0, f"subprocess failed ({done.returncode}):\n{done.stderr[-4000:]}"
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_sharded_characteristic_rhs_equals_single_device():
    result = _subprocess("sharded")
    assert result["devices"] == 4 and result["stripped_has_none"]
    assert result["wall_cells"] == result["expected_wall_cells"] > 0
    assert result["changed_owners"]["Ti"] > 0 and result["changed_owners"]["vorticity"] > 0
    assert result["changed_owners"]["density"] == result["changed_owners"]["Te"] == 0
    for sz in (1, 2, 4):
        entry = result[f"Sz{sz}"]
        assert sum(entry["wall_counts"]) == result["wall_cells"] and len(entry["wall_counts"]) == sz
        assert entry["max_rel_bracket"] <= 1e-14, (sz, entry)
        assert entry["char_differs_from_default"]                               # the closure acts in every sharded run
        assert entry["default_bitwise_without_wall_columns"], (sz, entry)       # the default RHS ignores the stencils
    everything = result["all_terms"]                                              # bracket, curvature and diffusion, 2 shards
    assert everything["max_rel_total"] <= 1e-13 and everything["unchanged_terms_bitwise"]
    assert everything["default_bitwise_without_wall_columns"]
    uneven = result["uneven"]                                                     # no wall entry on shard 0: padding is inert
    assert len(set(uneven["wall_counts"])) > 1 and uneven["padded_entries"] > 0
    assert uneven["max_rel_bracket"] <= 1e-14 and uneven["differs_from_full"]
