"""P06 / P06N / P06-legacy JAX curvature operator on synthetic rows (P08 step 2b, E4; fast).

The bounded closure of the shared synthetic world (``tests.perpendicular_synthetic``: real census, random
shaped rows incl. conditioned rows, Neumann rows, a wall face with a missing upper side) is made
*thermodynamically admissible* (positive convex value weights, positive fields and trace) so the curvature
matrix and the wall solve behave; then ``p06_action`` is checked against the host arithmetic of
``scripts/p_shared`` (``apply.p06_q1_terms`` / ``apply.p06_q3_correction`` on the host D/N reconstructions of
the same rows, the replay's evolution-weighted scatter and ``reduce_grid``'s single division); eager == jit
bitwise; JVP vs central finite differences.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native.fci_perpendicular_p06_operator import (
    FLOOR, TAU, bc_columns, p06_action, p06_q1_raw_numerators, p06_q1_state_numerators,
    p06_q3_face_numerators, p06n_layout)
from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables
from tests.perpendicular_synthetic import Boundary, host_cells, host_faces, lower_world, make_world

OWNERS = [0, 1, 2, 40, 96, 99, 100, 114]
NPHYS = 8
TOL = 1e-12
D, N = "dirichlet", "neumann"
#: variants over the shared physical columns (mixed D/N, a repeated (column, kind) across variants)
RECONSTRUCTIONS = (
    SimpleNamespace(columns=np.arange(5), field_kinds=(D,) * 5),
    SimpleNamespace(columns=np.arange(5), field_kinds=(N,) * 5),
    SimpleNamespace(columns=np.array([0, 5, 2, 6, 7]), field_kinds=(D, N, N, D, N)),
)


class PositiveBoundary(Boundary):
    """Trace ``1 + 0.3 sin(phase)`` (positive), physical-normal data from its gradient."""

    def dirichlet(self, points):
        v, g = super().dirichlet(points)
        return 1.0 + 0.3 * v, 0.3 * g


def _admissible(world):
    """Positive convex value weights (so reconstructed n, Te, Ti stay positive), small Neumann boundary weights."""
    for key, row in list(world.row_index.items()):
        if hasattr(row, "value") and getattr(row, "value").ndim == 2 and hasattr(row, "gradient"):
            a = np.abs(row.value)
            world.row_index[key] = dataclasses.replace(row, value=a / a.sum(axis=-1, keepdims=True))
    for key, row in list(world.neumann_index.items()):
        a = np.abs(row.value)
        world.neumann_index[key] = dataclasses.replace(row, value=a / a.sum(), boundary_value=0.02 * row.boundary_value)
    return world


@pytest.fixture(scope="module")
def world():
    return _admissible(make_world(owners=OWNERS))


@pytest.fixture(scope="module")
def plan(world):
    return lower_world(world)


@pytest.fixture(scope="module")
def boundary():
    return PositiveBoundary(NPHYS)


@pytest.fixture(scope="module")
def owner_fields(world):
    return 1.0 + 0.4 * np.random.default_rng(5).uniform(size=(world.n_owners, NPHYS))


@pytest.fixture(scope="module")
def layout():
    return p06n_layout(RECONSTRUCTIONS)


@pytest.fixture(scope="module")
def bc_all(plan, boundary):
    return boundary_data_from_callables(plan, boundary.dirichlet, boundary.normal)


@pytest.fixture(scope="module")
def host(world, plan, boundary, owner_fields):
    """Host q1 and q3 numerators/means per P06N variant from ``p_shared.apply`` on the same rows."""
    from p_shared import apply as A
    cells, faces = plan.cells, plan.faces
    prepared = SimpleNamespace(B=np.asarray(cells.B), K=np.asarray(cells.K), J=np.asarray(cells.J))
    w = np.asarray(cells.evolution_weight)
    raw_owner = np.asarray(cells.raw_owner)
    n_owners = world.n_owners
    (vd, gd), (vn, gn) = host_cells(world, owner_fields, boundary)
    fc = host_faces(world, owner_fields, boundary)
    keys = world.census.keys()[world.face_rows]
    is_wall = (keys[:, 0] == 0) & (keys[:, 1] == world.n)
    lower_owner, upper_owner = np.asarray(faces.lower_owner), np.asarray(faces.upper_owner)
    out = dict(material=[], remainder=[], total=[], mat_num=[], rem_num=[], corr_lo=[], corr_hi=[], corr_num=[],
               spectral=[])
    ev = np.zeros(n_owners)
    np.add.at(ev, raw_owner, w)
    out["ev"] = ev
    for rec in RECONSTRUCTIONS:
        cols, isn = np.asarray(rec.columns), np.array([k == N for k in rec.field_kinds])
        value = np.where(isn[None, :], vn[:, cols], vd[:, cols])
        grad = np.where(isn[None, None, :], gn[:, :, cols], gd[:, :, cols])
        mat, rem, tot, *_ = A.p06_q1_terms(value, grad, prepared)
        nums = []
        for arr in (mat, rem, tot):
            num = np.zeros((n_owners, 4)); np.add.at(num, raw_owner, w[:, None] * arr); nums.append(num)
        out["mat_num"].append(nums[0]); out["rem_num"].append(nums[1])
        evs = np.maximum(ev, 1e-300)[:, None]
        out["material"].append(nums[0] / evs); out["remainder"].append(nums[1] / evs); out["total"].append(nums[2] / evs)
        pick = lambda d, n_: np.where(isn[None, None, :], n_[:, :, cols], d[:, :, cols])
        cv, lv, uv = pick(fc["cvd"], fc["cvn"]), pick(fc["ld"], fc["ln"]), pick(fc["ud"], fc["un"])
        lo = np.zeros((len(world.face_rows), 4)); hi = np.zeros_like(lo); spectral = 0
        for f in range(len(world.face_rows)):
            axis = int(keys[f, 0])
            B = np.asarray(faces.B[f]); K = np.asarray(faces.K[f]); J = np.asarray(faces.J[f])
            normal = J * K[:, axis] / np.maximum(B * B, 1e-30)
            l, h, fb = A.p06_q3_correction(cv[f, :, :4], lv[f, :, :4], uv[f, :, :4], B, normal,
                                           np.asarray(faces.weight[f]), wall=bool(is_wall[f]))
            lo[f], hi[f] = l, h; spectral += int(fb)
        out["corr_lo"].append(lo); out["corr_hi"].append(hi); out["spectral"].append(spectral)
        num = np.zeros((n_owners, 4))
        for owner, term in ((lower_owner, lo), (upper_owner, hi)):
            ok = owner >= 0
            np.add.at(num, owner[ok], term[ok])
        out["corr_num"].append(num)
    out["is_wall"] = is_wall
    return out


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _action(plan, fields, bc, layout, **kw):
    columns, kinds, groups = layout
    return p06_action(plan, jnp.asarray(fields)[:, columns], bc_columns(bc, columns), kinds, groups, **kw)


def test_world_has_wall_missing_side_and_neumann_columns(plan, host):
    assert host["is_wall"].any() and np.array_equal(host["is_wall"], np.asarray(plan.faces.wall))
    assert plan.faces.has_missing_side and plan.cells.conditioned.any()
    columns, kinds, groups = p06n_layout(RECONSTRUCTIONS)
    assert groups.shape == (3, 5)
    # a (column, kind) pair shared by two variants is reconstructed once: (0,D)/(2,N) appear in two variants
    assert len(columns) == len(kinds) < sum(len(r.columns) for r in RECONSTRUCTIONS)
    for v, rec in enumerate(RECONSTRUCTIONS):
        assert np.array_equal(columns[groups[v]], rec.columns)
        assert tuple(kinds[i] for i in groups[v]) == tuple(rec.field_kinds)


def test_q1_and_q3_match_host(plan, host, bc_all, owner_fields, layout):
    out = _action(plan, owner_fields, bc_all, layout)
    errors = {}
    for name, key in (("material", "material"), ("remainder", "remainder"), ("total", "total"),
                      ("material_numerator", "mat_num"), ("remainder_numerator", "rem_num"),
                      ("correction_numerator", "corr_num")):
        errors[name] = max(_rel(np.asarray(getattr(out, name))[v], host[key][v]) for v in range(3))
    ev = np.maximum(host["ev"], 1e-300)[:, None]
    errors["correction"] = max(_rel(np.asarray(out.correction)[v], host["corr_num"][v] / ev) for v in range(3))
    errors["evolution_volume"] = _rel(out.evolution_volume, host["ev"])
    print("\nE4 fast synthetic rel errors vs host:", errors)
    assert max(errors.values()) <= TOL, errors
    # the world exercises the Neumann columns, the wall solve and non-trivial q3 (not all-zero) terms
    assert np.max(np.abs(host["corr_num"][1])) > 1e-6
    assert _rel(out.material[1], out.material[0]) > 1e-3
    assert _rel(out.correction_numerator[1], out.correction_numerator[0]) > 1e-3


def test_face_numerators_match_host_and_counters(plan, host, bc_all, owner_fields, layout):
    columns, kinds, groups = layout
    mult = np.where(np.arange(len(plan.faces.census_row)) % 5 == 0, 2.0, 1.0)
    faces = p06_q3_face_numerators(plan, owner_fields[:, columns], bc_columns(bc_all, columns), kinds, groups,
                                   face_multiplier=mult)
    for v in range(3):
        assert _rel(faces.lower[v], host["corr_lo"][v] * mult[:, None]) <= TOL
        assert _rel(faces.upper[v], host["corr_hi"][v] * mult[:, None]) <= TOL
    assert faces.lower.shape == (3, len(plan.faces.census_row), 4)
    assert faces.spectral_fallback.shape == faces.floor_hits.shape == faces.wall_fallback.shape == (3,)
    np.testing.assert_array_equal(np.asarray(faces.spectral_fallback), host["spectral"])



def test_legacy_single_state_and_face_multiplier(world, plan, host, bc_all, owner_fields):
    """``groups=None``: five Dirichlet columns, ``face_multiplier`` on the q3 numerators, outputs without a V axis."""
    cols = np.arange(5)
    mult = np.where(np.arange(len(plan.faces.census_row)) % 4 == 1, 2.0, 1.0)
    out = p06_action(plan, owner_fields[:, cols], bc_columns(bc_all, cols), D, face_multiplier=mult)
    assert out.material.shape == (world.n_owners, 4) and out.correction.shape == (world.n_owners, 4)
    # the host's D variant is the first RECONSTRUCTION; recompute its corrected numerators with the multiplier
    lower_owner, upper_owner = np.asarray(plan.faces.lower_owner), np.asarray(plan.faces.upper_owner)
    num = np.zeros((world.n_owners, 4))
    for owner, term in ((lower_owner, host["corr_lo"][0]), (upper_owner, host["corr_hi"][0])):
        ok = owner >= 0
        np.add.at(num, owner[ok], (term * mult[:, None])[ok])
    assert _rel(out.correction_numerator, num) <= TOL
    assert _rel(out.material, host["material"][0]) <= TOL and _rel(out.total, host["total"][0]) <= TOL
    default = p06_action(plan, owner_fields[:, cols], bc_columns(bc_all, cols), D)
    assert _rel(default.correction_numerator, host["corr_num"][0]) <= TOL         # plan.faces.face_multiplier is ones


def test_raw_numerators_are_the_weighted_terms(plan, host, bc_all, owner_fields, layout):
    columns, kinds, groups = layout
    raw = p06_q1_raw_numerators(plan, owner_fields[:, columns], bc_columns(bc_all, columns), kinds, groups)
    seg = np.zeros((groups.shape[0], plan.cells.evolution_volume.shape[0], 4))
    for v in range(groups.shape[0]):
        np.add.at(seg[v], np.asarray(plan.cells.raw_owner), np.asarray(raw.material)[v])
    assert _rel(seg, np.stack(host["mat_num"])) <= TOL
    np.testing.assert_array_equal(np.asarray(raw.total), np.asarray(raw.material) + np.asarray(raw.remainder))


def test_eager_equals_jit_bitwise(plan, bc_all, owner_fields):
    """Eager call vs the call inside an outer ``jax.jit`` (the P06-legacy layout, with a face multiplier), incl. the
    lower-level q3 face numerators; the internal-jit pattern of the P07 operator."""
    cols = np.arange(5)
    fields = jnp.asarray(owner_fields)[:, cols]
    bc = bc_columns(bc_all, cols)
    mult = np.where(np.arange(len(plan.faces.census_row)) % 3 == 0, 2.0, 1.0)
    eager = p06_action(plan, fields, bc, D, face_multiplier=mult)
    jitted = jax.jit(lambda p, f, b, m: p06_action(p, f, b, D, face_multiplier=m))(plan, fields, bc, jnp.asarray(mult))
    for a, b in zip(eager, jitted):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    q3 = p06_q3_face_numerators(plan, fields, bc, D, face_multiplier=mult)
    q3_jit = jax.jit(lambda p, f, b, m: p06_q3_face_numerators(p, f, b, D, face_multiplier=m))(
        plan, fields, bc, jnp.asarray(mult))
    for a, b in zip(q3, q3_jit):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert p06_q1_raw_numerators(plan, fields, bc, D).material.shape == (len(plan.cells.raw_ids), 4)


def _fd_jvp(fun, x, tangent, h):
    plus = fun(x + h * tangent); minus = fun(x - h * tangent)
    return jax.tree_util.tree_map(lambda a, b: (a - b) / (2 * h), plus, minus)


def test_jvp_matches_central_finite_difference(plan, bc_all, owner_fields, layout):
    """Full P06 (q1 and q3 numerators/means) JVP vs a central FD at random smooth, thermodynamically sane states."""
    columns, kinds, groups = layout
    bc = bc_columns(bc_all, columns)
    x = jnp.asarray(owner_fields)[:, columns]
    tangent = jnp.asarray(np.random.default_rng(9).normal(size=x.shape))

    def fun(f):
        o = p06_action(plan, f, bc, kinds, groups)
        return (o.material, o.remainder, o.correction_numerator, o.correction)

    _, jvp = jax.jvp(fun, (x,), (tangent,))
    fd = _fd_jvp(fun, x, tangent, 1e-6)
    errors = [_rel(a, b) for a, b in zip(jvp, fd)]
    print("\nE4 JVP vs central FD rel errors (material, remainder, correction_numerator, correction):", errors)
    assert max(errors) <= 1e-6, errors
    assert min(float(jnp.max(jnp.abs(a))) for a in jvp) > 0


def test_q1_numerators_are_linear_in_the_gradients(plan, bc_all, owner_fields, layout):
    """At fixed values the q1 numerators are linear in the gradients (2 g = 2 x term; g1 + g2 additive)."""
    columns, kinds, groups = layout
    R = len(plan.cells.raw_ids)
    rng = np.random.default_rng(2)
    value = 1.0 + 0.4 * rng.uniform(size=(2, R, 5))
    g1, g2 = rng.normal(size=(2, R, 3, 5)), rng.normal(size=(2, R, 3, 5))
    args = (plan.cells.B, plan.cells.K, plan.cells.evolution_weight, value)
    m1, r1 = p06_q1_state_numerators(*args, g1); m2, r2 = p06_q1_state_numerators(*args, g2)
    m12, r12 = p06_q1_state_numerators(*args, g1 + g2)
    m2x, r2x = p06_q1_state_numerators(*args, 2 * g1)
    assert _rel(m12, m1 + m2) <= 1e-13 and _rel(r12, r1 + r2) <= 1e-13
    assert _rel(m2x, 2 * m1) <= 1e-15 and _rel(r2x, 2 * r1) <= 1e-15
    _, jvp = jax.jvp(lambda g: p06_q1_state_numerators(*args, g)[0], (jnp.asarray(g1),), (jnp.asarray(g2),))
    assert _rel(jvp, m2) <= 1e-13


def test_constants_and_layout_errors(plan, bc_all, owner_fields):
    assert TAU == 1.0 and FLOOR == 1.0e-12
    with pytest.raises(ValueError):
        p06_action(plan, owner_fields[:, :4], bc_columns(bc_all, np.arange(4)), D)   # groups=None needs 5 columns
    with pytest.raises(ValueError):
        p06_action(plan, owner_fields[:, :5], bc_columns(bc_all, np.arange(5)), D, face_multiplier=np.ones(3))
