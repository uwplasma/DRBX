"""Fast, geometry-free tests for the P06N row/BC-selection machinery.

Mirrors ``tests/test_p05n_neumann_rows.py``'s synthetic fixture (a small,
partly-aggregated structured grid, module-scope, <1s to build) plus the
oblique physical-normal Neumann context from
``tests/test_fci_perpendicular_neumann_trace.py``. Covers, per the P06N
design (work/p_neumann_p05n_p06n_design_20260927/design.md section 3b):

- P06N's own new primitive, ``p06n_field_derived_global.rows.batched_face_common_value``
  (the row's *value*, both BC variants, batched over a field-column axis;
  P06's q3 term needs this, never a face-common gradient).
- Reuse (not reimplementation) of ``p05n_field_derived_global.core``'s
  catalogue-agnostic ``batched_cell_values``/``batched_side_values``:
  routing, interior BC-independence, linearity, constant reproduction, and
  the zero-jump property on the wall/radial-n-1/transverse-last-two-layers
  face kinds (both sides use the identical point-anchored row there).
- The wall characteristic solve's recovered-trace contract
  (``drbx.native.fci_operators._curvature_bc_characteristic_wall_states``
  called with ``boundary_trace = interior``): Delta = 0 by construction,
  independent of the (arbitrary, synthetic) state/normal/B fed in -- no HSX
  geometry or continuum reference is needed for this property, only the
  wall-solve's own contract, so it is checked directly here without loading
  any runtime input.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from perpendicular_structured.reconstruction import StructuredReconstruction  # noqa: E402
from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext  # noqa: E402
from drbx.native.fci_operators import _curvature_bc_characteristic_wall_states  # noqa: E402
from p07_combined_global import kernels as pk  # noqa: E402
from p05n_field_derived_global import core as p05n_core  # noqa: E402
from p06n_field_derived_global import rows as p06n_rows  # noqa: E402


@pytest.fixture(scope="module")
def geometry():
    n = 32
    faces = (np.linspace(0, 1, n + 1), np.linspace(0, 2 * np.pi, n + 1), np.linspace(0, 2 * np.pi, n + 1))
    centers = tuple((f[:-1] + f[1:]) / 2 for f in faces)
    ijk = np.array(np.unravel_index(np.arange(n ** 3), (n,) * 3)).T
    pts = np.column_stack([centers[a][ijk[:, a]] for a in range(3)])
    ro = np.empty((n, n, n), int); count = 0
    for i in range(n):
        block = [32, 8, 4, 2][i] if i < 4 else 1
        for j in range(0, n, block):
            for k in range(n):
                ro[i, j:j + block, k] = count; count += 1
    ro = ro.ravel()
    rv = 1 + .1 * np.cos(pts[:, 1])
    vol = np.bincount(ro, weights=rv)
    order = np.argsort(ro, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(ro))]
    xy = np.column_stack((pts[:, 0] * np.cos(pts[:, 1]), pts[:, 0] * np.sin(pts[:, 1])))
    centroid = np.column_stack([np.bincount(ro, weights=rv * xy[:, a]) / vol for a in range(2)])
    g = SimpleNamespace(dr=1 / n, dtheta=2 * np.pi / n, deta=2 * np.pi / n, eta_period=2 * np.pi,
                        owner_centroid_xy=centroid)
    t = SimpleNamespace(n=n, faces=faces, centers=centers, pts=pts, xy=xy, ro=ro, rv=rv, vol=vol,
                        order=order, starts=starts, g=g)
    S = StructuredReconstruction(t)
    context = PointRowContext.from_arrays(faces=faces, centers=centers, raw_to_owner=ro, raw_volume=rv,
                                           owner_volume=vol, owner_centroid_xy=centroid, eta_period=g.eta_period,
                                           dr=g.dr, dtheta=g.dtheta, deta=g.deta)
    return t, S, context


# Two synthetic fields packed as the (Q,2) field-column axis: field 0 mirrors
# test_p05n_neumann_rows.py's polynomial; field 1 is an independent one, to
# exercise the "batched over an arbitrary field axis" contract with F>1.
COEFF = np.array((np.sqrt(1.04), -.2 / np.sqrt(1.04), 0.))


def fields(q):
    q = np.asarray(q, dtype=float)
    u, th, eta = q[:, 0], q[:, 1], q[:, 2]
    v0 = 1 + .3 * u + .2 * np.sin(th) + .1 * eta
    g0 = np.stack((np.full_like(u, .3), .2 * np.cos(th), np.full_like(u, .1)), axis=1)
    v1 = 2 - .1 * u + .05 * np.cos(2 * th) - .2 * eta
    g1 = np.stack((np.full_like(u, -.1), -.1 * np.sin(2 * th), np.full_like(u, -.2)), axis=1)
    v = np.stack((v0, v1), axis=1)
    g = np.stack((g0, g1), axis=2)
    return v, g


def dirichlet_trace_fn(q):
    return fields(q)


def normal_data_fn(q):
    _, g = fields(q)
    return np.einsum("qda,d->qa", g, COEFF)


def normal_coefficients(q):
    return np.broadcast_to(COEFF, (len(q), 3))


def owner_values_matrix(t):
    v, _ = fields(t.pts)
    sums = np.stack([np.bincount(t.ro, weights=t.rv * v[:, f], minlength=len(t.vol)) for f in range(v.shape[1])], axis=1)
    return sums / t.vol[:, None]


@pytest.fixture(scope="module")
def owner_values(geometry):
    t, _, _ = geometry
    return owner_values_matrix(t)


def _face_anchor(t, key):
    axis = key[0]
    return np.array([[
        (t.faces[a][key[a + 1]] if a == axis else t.centers[a][key[a + 1]]) for a in range(3)
    ]])


@pytest.mark.parametrize("key", [(0, 32, 4, 5), (0, 31, 4, 5), (1, 31, 4, 5), (1, 30, 4, 5), (0, 16, 4, 5)])
def test_face_common_value_routing_and_bc_agreement(geometry, owner_values, key):
    t, S, context = geometry
    point = _face_anchor(t, key)
    axis, i = key[0], key[1]

    result = p06n_rows.batched_face_common_value(t, S, owner_values, key, point,
                                                   normal_coefficients=normal_coefficients, ctx=context,
                                                   patch_cache={}, dirichlet_trace_fn=dirichlet_trace_fn,
                                                   normal_data_fn=normal_data_fn)
    exact_v, _ = fields(point)
    kind = pk.family(t.n, key)
    boundary = kind in ("quartic_wall", "boundary_transverse")
    assert result["boundary"] == boundary
    np.testing.assert_allclose(result["dirichlet"], exact_v, atol=1e-8)
    np.testing.assert_allclose(result["neumann"], exact_v, atol=1e-8)
    if not boundary:
        # Interior/regular faces: both BC branches take the identical,
        # BC-independent accepted row -- not merely numerically close.
        np.testing.assert_array_equal(result["dirichlet"], result["neumann"])
    else:
        assert result["condition"].shape == (1,)
        assert np.all(result["condition"] < 1e3)
        assert np.all(result["residual"] < 1e-8)


def test_face_common_value_matches_face_common_gradient_row_family(geometry, owner_values):
    """Sanity: value and gradient routing agree on which faces are boundary-conditioned."""
    t, S, context = geometry
    for key in [(0, 32, 4, 5), (0, 31, 4, 5), (1, 31, 4, 5), (0, 16, 4, 5)]:
        point = _face_anchor(t, key)
        value_result = p06n_rows.batched_face_common_value(t, S, owner_values, key, point,
                                                             normal_coefficients=normal_coefficients, ctx=context,
                                                             patch_cache={}, dirichlet_trace_fn=dirichlet_trace_fn,
                                                             normal_data_fn=normal_data_fn)
        gradient_result = p05n_core.batched_face_common_gradient(t, S, owner_values, key, point,
                                                                  normal_coefficients=normal_coefficients, ctx=context,
                                                                  patch_cache={}, dirichlet_trace_fn=dirichlet_trace_fn,
                                                                  normal_data_fn=normal_data_fn)
        assert value_result["family"] == gradient_result["family"]
        assert value_result["boundary"] == gradient_result["boundary"]


def test_constant_field_reproduces_through_value_and_cell_kernels(geometry):
    t, S, context = geometry
    ones = np.ones((len(t.vol), 1))

    def zero_trace(q):
        q = np.asarray(q, dtype=float)
        return np.ones((len(q), 1)), np.zeros((len(q), 3, 1))

    def zero_normal(q):
        return np.zeros((len(q), 1))

    for key in [(0, 32, 4, 5), (0, 31, 4, 5), (1, 31, 4, 5)]:
        point = _face_anchor(t, key)
        result = p06n_rows.batched_face_common_value(t, S, ones, key, point, normal_coefficients=normal_coefficients,
                                                       ctx=context, patch_cache={}, dirichlet_trace_fn=zero_trace,
                                                       normal_data_fn=zero_normal)
        np.testing.assert_allclose(result["dirichlet"], 1.0, atol=1e-8)
        np.testing.assert_allclose(result["neumann"], 1.0, atol=1e-8)

    raw_ids = np.flatnonzero(t.ro == int(t.ro.reshape(t.n, t.n, t.n)[31, 4, 5]))
    keys = np.array(np.unravel_index(raw_ids, (t.n,) * 3)).T
    points = t.pts[raw_ids]
    cell_result = p05n_core.batched_cell_values(t, S, ones, keys, points, normal_coefficients=normal_coefficients,
                                                 ctx=context, patch_cache={}, dirichlet_trace_fn=zero_trace,
                                                 normal_data_fn=zero_normal, radial_degree=3)
    vd, gd = cell_result["dirichlet"]
    vn, gn = cell_result["neumann"]
    np.testing.assert_allclose(vd, 1.0, atol=1e-8)
    np.testing.assert_allclose(vn, 1.0, atol=1e-8)
    np.testing.assert_allclose(gd, 0.0, atol=1e-8)
    np.testing.assert_allclose(gn, 0.0, atol=1e-8)


@pytest.mark.parametrize("key", [(0, 32, 4, 5), (0, 31, 4, 5), (1, 31, 4, 5), (1, 30, 4, 5)])
def test_zero_jump_on_identical_row_faces(geometry, owner_values, key):
    """The wall/radial-n-1/transverse-last-two-layers property: both sides of
    ``batched_side_values`` use the identical point-anchored row there, for
    either BC variant -- so the q3 jump vanishes independent of BC choice.
    """
    t, S, context = geometry
    point = _face_anchor(t, key)
    sides = p05n_core.batched_side_values(t, S, owner_values, key, point, normal_coefficients=normal_coefficients,
                                           ctx=context, patch_cache={}, dirichlet_trace_fn=dirichlet_trace_fn,
                                           normal_data_fn=normal_data_fn)
    np.testing.assert_allclose(sides["lower"]["dirichlet"], sides["upper"]["dirichlet"], atol=1e-11)
    np.testing.assert_allclose(sides["lower"]["neumann"], sides["upper"]["neumann"], atol=1e-11)


def test_cell_values_linear_in_owner_values_and_g_N(geometry, owner_values):
    t, S, context = geometry
    raw_ids = np.flatnonzero(t.ro == int(t.ro.reshape(t.n, t.n, t.n)[31, 6, 19]))
    keys = np.array(np.unravel_index(raw_ids, (t.n,) * 3)).T
    points = t.pts[raw_ids]

    def zero_normal(q):
        return np.zeros((len(q), owner_values.shape[1]))

    both = p05n_core.batched_cell_values(t, S, owner_values, keys, points, normal_coefficients=normal_coefficients,
                                          ctx=context, patch_cache={}, dirichlet_trace_fn=dirichlet_trace_fn,
                                          normal_data_fn=normal_data_fn, radial_degree=3)
    data_only = p05n_core.batched_cell_values(t, S, owner_values, keys, points, normal_coefficients=normal_coefficients,
                                               ctx=context, patch_cache={}, dirichlet_trace_fn=dirichlet_trace_fn,
                                               normal_data_fn=zero_normal, radial_degree=3)
    bc_only = p05n_core.batched_cell_values(t, S, np.zeros_like(owner_values), keys, points,
                                             normal_coefficients=normal_coefficients, ctx=context, patch_cache={},
                                             dirichlet_trace_fn=dirichlet_trace_fn, normal_data_fn=normal_data_fn,
                                             radial_degree=3)
    v_both, g_both = both["neumann"]
    v_data, g_data = data_only["neumann"]
    v_bc, g_bc = bc_only["neumann"]
    np.testing.assert_allclose(v_both, v_data + v_bc, atol=1e-12)
    np.testing.assert_allclose(g_both, g_data + g_bc, atol=1e-11)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wall_recovered_trace_contract_zero_correction(seed):
    """The adopted contract (design.md section 3): the physical-wall
    characteristic solve receives ``boundary_trace = interior`` -- Delta = 0
    by construction, for *any* positive (n,Te,Ti,omega) state, B and normal
    (no HSX geometry needed; this is the wall-solve's own contract).
    """
    rng = np.random.default_rng(seed)
    Q = 9
    n = rng.uniform(0.5, 2.0, Q)
    te = rng.uniform(0.5, 2.0, Q)
    ti = rng.uniform(0.5, 2.0, Q)
    omega = rng.normal(size=Q)
    interior = np.stack((n, te, ti, omega), axis=1)
    B = rng.uniform(0.3, 1.5, Q)
    normal = rng.normal(size=Q)
    exterior, working, fallback = _curvature_bc_characteristic_wall_states(
        jnp.asarray(interior), jnp.asarray(interior), jnp.asarray(B), 1.0,
        jnp.asarray(normal), interior_on_right=False, positivity_floor=1e-12,
    )
    np.testing.assert_array_equal(np.asarray(exterior), interior)
    assert np.asarray(fallback).shape[0] == Q
