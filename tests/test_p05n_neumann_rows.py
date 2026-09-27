"""Fast, geometry-free tests for the P05N per-field-and-BC row selector.

No HSX runtime inputs are needed: this uses the same kind of small synthetic,
partly-aggregated structured grid as tests/test_perpendicular_structured_campaign.py
(module-scope fixture, <1s to build) combined with a fixed-support Neumann
elimination context (tests/test_fci_perpendicular_neumann_trace.py's oblique
normal-coefficient pattern). It covers the row selector's Neumann/Dirichlet
routing (rows.py), linearity in (owner values, g_N), and constant reproduction,
per the P05N design (work/p_neumann_p05n_p06n_design_20260927/design.md).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from perpendicular_structured.reconstruction import StructuredReconstruction  # noqa: E402
from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext  # noqa: E402
from p05n_field_derived_global import rows as p05n_rows  # noqa: E402


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


# Oblique physical-normal coefficient, the same family as the qualified
# tests/test_fci_perpendicular_neumann_trace.py::test_oblique_normal_trace_and_runtime_jvp.
COEFF = np.array((np.sqrt(1.04), -.2 / np.sqrt(1.04), 0.))


def field(q):
    q = np.asarray(q, dtype=float)
    u, th, eta = q[:, 0], q[:, 1], q[:, 2]
    v = 1 + .3 * u + .2 * np.sin(th) + .1 * eta
    g = np.stack((np.full_like(u, .3), .2 * np.cos(th), np.full_like(u, .1)), axis=1)
    return v, g


def dirichlet_trace(q):
    return field(q)


def normal_data(q):
    _, g = field(q)
    return g @ COEFF


def normal_coefficients(q):
    return np.broadcast_to(COEFF, (len(q), 3))


def zero_normal_data(q):
    return np.zeros(len(q))


def owner_column(t, values_at_raw):
    sums = np.bincount(t.ro, weights=t.rv * values_at_raw, minlength=len(t.vol))
    return sums / t.vol


@pytest.fixture(scope="module")
def live_column(geometry):
    t, _, _ = geometry
    return owner_column(t, field(t.pts)[0])


@pytest.fixture(scope="module")
def zero_column(geometry):
    t, _, _ = geometry
    return np.zeros(len(t.vol))


def _cell_keys_points(t, i, js, k=5):
    keys = np.array([(i, j, k) for j in js])
    points = np.column_stack([t.centers[a][keys[:, a]] for a in range(3)])
    return keys, points


@pytest.mark.parametrize("i", [31, 30])  # physical wall cell and its neighbor: both boundary-conditioned
def test_cell_rows_dirichlet_and_neumann_reproduce_polynomial_and_agree(geometry, live_column, i):
    t, S, context = geometry
    keys, points = _cell_keys_points(t, i, js=[3, 11, 22])
    ev, eg = field(points)

    vd, gd, diag_d = p05n_rows.cell_rows(t, S, live_column, keys, points, bc="dirichlet",
                                          dirichlet_trace=dirichlet_trace)
    np.testing.assert_allclose(vd, ev, atol=1e-11)
    np.testing.assert_allclose(gd, eg, atol=1e-9)
    assert all(d.boundary_conditioned and not d.neumann for d in diag_d)

    vn, gn, diag_n = p05n_rows.cell_rows(t, S, live_column, keys, points, bc="neumann",
                                          normal_coefficients=normal_coefficients, context=context,
                                          patch_cache={}, normal_data=normal_data)
    np.testing.assert_allclose(vn, ev, atol=1e-11)
    np.testing.assert_allclose(gn, eg, atol=1e-9)
    assert all(d.boundary_conditioned and d.neumann for d in diag_n)
    assert all(d.condition < 1e3 for d in diag_n)
    assert all(d.constraint_residual < 1e-9 for d in diag_n)

    # Both boundary conditions recover the identical exact field: they must
    # also agree with each other, not merely with the analytic answer.
    np.testing.assert_allclose(vd, vn, atol=1e-10)
    np.testing.assert_allclose(gd, gn, atol=1e-8)


def test_interior_rows_are_independent_of_the_requested_bc(geometry, live_column):
    """Interior rows (i < n-2) always take the accepted, BC-independent path."""
    t, S, context = geometry
    keys, points = _cell_keys_points(t, 10, js=[2, 17])
    vd, gd, diag_d = p05n_rows.cell_rows(t, S, live_column, keys, points, bc="dirichlet")
    vn, gn, diag_n = p05n_rows.cell_rows(t, S, live_column, keys, points, bc="neumann",
                                          normal_coefficients=normal_coefficients, context=context,
                                          patch_cache={}, normal_data=normal_data)
    np.testing.assert_array_equal(vd, vn)
    np.testing.assert_array_equal(gd, gn)
    assert not any(d.boundary_conditioned for d in diag_d + diag_n)
    ev, eg = field(points)
    np.testing.assert_allclose(vd, ev, atol=1e-10)
    np.testing.assert_allclose(gd, eg, atol=1e-8)


@pytest.mark.parametrize("key", [(0, 32, 4, 5), (0, 31, 4, 5), (1, 31, 4, 5), (1, 30, 4, 5)])
def test_face_and_side_rows_agree_across_bc_and_the_wall_jump_vanishes(geometry, live_column, key):
    t, S, context = geometry
    # Use the exact face anchor point (matches StructuredReconstruction.anchor).
    axis = key[0]
    point = np.array([[
        (t.faces[a][key[a + 1]] if a == axis else t.centers[a][key[a + 1]]) for a in range(3)
    ]])

    common_d, _ = p05n_rows.face_common_gradient(t, S, live_column, key, point, bc="dirichlet",
                                                  dirichlet_trace=dirichlet_trace)
    common_n, _ = p05n_rows.face_common_gradient(t, S, live_column, key, point, bc="neumann",
                                                  normal_coefficients=normal_coefficients, context=context,
                                                  patch_cache={}, normal_data=normal_data)
    _, exact_g = field(point)
    np.testing.assert_allclose(common_d, exact_g, atol=1e-8)
    np.testing.assert_allclose(common_n, exact_g, atol=1e-8)

    lv_d, rv_d, _ = p05n_rows.side_values(t, S, live_column, key, point, bc="dirichlet",
                                          dirichlet_trace=dirichlet_trace)
    lv_n, rv_n, _ = p05n_rows.side_values(t, S, live_column, key, point, bc="neumann",
                                          normal_coefficients=normal_coefficients, context=context,
                                          patch_cache={}, normal_data=normal_data)
    # The established zero-jump property: on the physical wall face, radial
    # face n-1, and transverse faces of the last two layers, both sides
    # reconstruct the identical value (same query point, same patch family).
    np.testing.assert_allclose(lv_d, rv_d, atol=1e-11)
    np.testing.assert_allclose(lv_n, rv_n, atol=1e-11)


def test_neumann_row_is_linear_in_owner_values_and_g_N(geometry, live_column, zero_column):
    t, S, context = geometry
    keys, points = _cell_keys_points(t, 31, js=[6, 19])
    v_both, g_both, _ = p05n_rows.cell_rows(t, S, live_column, keys, points, bc="neumann",
                                            normal_coefficients=normal_coefficients, context=context,
                                            patch_cache={}, normal_data=normal_data)
    v_data, g_data, _ = p05n_rows.cell_rows(t, S, live_column, keys, points, bc="neumann",
                                            normal_coefficients=normal_coefficients, context=context,
                                            patch_cache={}, normal_data=zero_normal_data)
    v_bc, g_bc, _ = p05n_rows.cell_rows(t, S, zero_column, keys, points, bc="neumann",
                                        normal_coefficients=normal_coefficients, context=context,
                                        patch_cache={}, normal_data=normal_data)
    np.testing.assert_allclose(v_both, v_data + v_bc, atol=1e-13)
    np.testing.assert_allclose(g_both, g_data + g_bc, atol=1e-12)


def test_constant_field_reproduces_to_tight_tolerance(geometry, zero_column):
    """A constant field (owner values == 1, g_N == 0) reproduces exactly."""
    t, S, context = geometry
    ones = np.ones(len(t.vol))
    keys, points = _cell_keys_points(t, 31, js=[1, 9, 25])
    for bc, kwargs in (("dirichlet", dict(dirichlet_trace=lambda q: (np.ones(len(q)), np.zeros((len(q), 3))))),
                       ("neumann", dict(normal_coefficients=normal_coefficients, context=context,
                                        patch_cache={}, normal_data=zero_normal_data))):
        v, g, diag = p05n_rows.cell_rows(t, S, ones, keys, points, bc=bc, **kwargs)
        np.testing.assert_allclose(v, 1.0, atol=1e-8)
        np.testing.assert_allclose(g, 0.0, atol=1e-8)


def test_unknown_boundary_condition_is_rejected(geometry, live_column):
    t, S, context = geometry
    keys, points = _cell_keys_points(t, 31, js=[4])
    with pytest.raises(ValueError, match="boundary condition"):
        p05n_rows.cell_rows(t, S, live_column, keys, points, bc="robin")
