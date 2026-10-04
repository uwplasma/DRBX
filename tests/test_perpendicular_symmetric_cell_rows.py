"""``StructuredReconstruction(cell_stencil="symmetric")``: fourth-order centred R1 cell rows ``1/2 (A + B)``.

A is the historic ("biased") cell row (radial layers ``i-1..i+2``, eta planes ``k-2..k+1``: third-order biased
cell-centre derivatives); B is its mirror (layers ``i-2..i+1``, eta planes reflected about ``k``).  Their average is the
fourth-order centred derivative ``(1, -8, 0, 8, -1) / 12 h``.  Only unconditioned singleton/ringwise cell rows at the cell
centre change; coupled, boundary and face rows (and the face-anchored R3 side rows) are untouched.

Synthetic and fast: a toy ``n = 12`` grid whose rings 0, 1, 2 hold 1, 4, 8 owners per eta plane (so cells ``i = 1, 2`` are
coupled, ``i = 3`` has a mirror that cannot be built, ``i = 4`` a mirror reaching an agglomerated ring) and whose outer
rings are one owner per raw cell.  The real-grid equivalence to the prototype rows is the one-off
``work/p09_symrows_20261003/adopt_check.py``.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]              # .../DRBX
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.geometry.fci_perpendicular_reconstruction import (  # noqa: E402
    CELL_STENCILS, PairedFactors, PointRowContext, StructuredReconstruction)

N = 12
RINGS = {0: 1, 1: 4, 2: 8}                  # owners per eta plane of the agglomerated rings (others: one per raw cell)


def _context(n=N, owners_per_ring=None):
    faces = (np.linspace(0, 1, n + 1), np.linspace(0, 2 * np.pi, n + 1), np.linspace(0, 2 * np.pi, n + 1))
    centers = tuple((x[:-1] + x[1:]) / 2 for x in faces)
    ro = np.zeros((n, n, n), dtype=np.int64)
    for i in range(n):
        m = (owners_per_ring or {}).get(i, n)
        ro[i] = ((np.arange(n) * m) // n)[:, None] + m * np.arange(n)[None, :] + 1000 * i
    ro = ro.ravel()
    _, first, inverse = np.unique(ro, return_index=True, return_inverse=True)
    ro = np.argsort(np.argsort(first))[inverse]         # owner ids ordered by first raw cell: raw id == owner id if unmerged
    ijk = np.array(np.unravel_index(np.arange(n ** 3), (n, n, n))).T
    pts = np.column_stack([centers[a][ijk[:, a]] for a in range(3)])
    xy = np.column_stack((pts[:, 0] * np.cos(pts[:, 1]), pts[:, 0] * np.sin(pts[:, 1])))
    count = np.bincount(ro)
    centroid = np.column_stack([np.bincount(ro, weights=xy[:, a]) / count for a in range(2)])
    return PointRowContext.from_arrays(
        faces=faces, centers=centers, raw_to_owner=ro, raw_volume=np.ones(n ** 3), owner_volume=count.astype(float),
        owner_centroid_xy=centroid, eta_period=2 * np.pi, dr=1 / n, dtheta=2 * np.pi / n, deta=2 * np.pi / n)


@pytest.fixture(scope="module")
def full():
    return _context()                                    # every ring full: all non-boundary cells are singletons


@pytest.fixture(scope="module")
def agglomerated():
    return _context(owners_per_ring=RINGS)


def _cell(context, key):
    return context.pts[np.ravel_multi_index(key, (context.n,) * 3)][None, :]


def _same_row(a, b):
    assert np.array_equal(a.donor_ids, b.donor_ids)
    assert np.array_equal(a.value, b.value) and np.array_equal(a.gradient, b.gradient)
    assert a.boundary_conditioned == b.boundary_conditioned
    assert np.array_equal(a.trace_donor_points, b.trace_donor_points)
    assert np.array_equal(a.trace_target_points, b.trace_target_points)
    assert a.diagnostics == b.diagnostics


def _face_points(builder, key):
    anchor = builder.anchor(key, "face")
    return np.stack((anchor, anchor + np.array([0.013, 0.07, -0.05])))


FACES = ((1, 5, 3, 4), (2, 6, 7, 2), (0, 5, 3, 4), (1, 3, 3, 4), (0, 2, 4, 4), (2, 3, 9, 10))
CELLS = ((1, 2, 4), (2, 2, 4), (3, 2, 4), (4, 2, 4), (5, 7, 11), (7, 0, 0), (10, 3, 4), (11, 9, 9))


def _weights(row, context, layers, j, k, column, scale):
    """The weights of ``row.gradient[:, column]`` on the (layer, j, k) raw cells (one owner per raw cell), times ``scale``."""
    n = context.n
    ids = np.array([(l * n + j) * n + k for l in layers]) if column == 0 else np.array([(layers * n + j) * n + kk for kk in k])
    out = np.zeros(len(ids))
    for q, oid in enumerate(ids):
        hit = np.flatnonzero(row.donor_ids == oid)
        out[q] = row.gradient[0, column, hit[0]] * scale if len(hit) else 0.0
    return out


# ---------------------------------------------------------------------------
# (a) option, default and bitwise "biased"
# ---------------------------------------------------------------------------
def test_option_choices_and_default():
    assert CELL_STENCILS == ("biased", "symmetric")
    parameter = inspect.signature(StructuredReconstruction.__init__).parameters["cell_stencil"]
    assert parameter.default == "biased" and parameter.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(ValueError):
        StructuredReconstruction(_context(8), cell_stencil="bogus")
    assert StructuredReconstruction(_context(8)).cell_stencil == "biased"


@pytest.mark.parametrize("inner_support", ("profile7", "fixed_radius"))
def test_default_is_bitwise_explicit_biased(agglomerated, inner_support):
    default = StructuredReconstruction(agglomerated, inner_support=inner_support)
    biased = StructuredReconstruction(agglomerated, inner_support=inner_support, cell_stencil="biased")
    for key in CELLS:
        p = _cell(agglomerated, key)
        _same_row(default.rows(key, p, "cell"), biased.rows(key, p, "cell"))
        a, fa = default.rows_with_factors(key, p, "cell")
        b, fb = biased.rows_with_factors(key, p, "cell")
        _same_row(a, b)
        assert (fa is None) == (fb is None)
    for key in FACES:
        p = _face_points(default, key)
        _same_row(default.rows(key, p, "face"), biased.rows(key, p, "face"))


# ---------------------------------------------------------------------------
# (b) the fourth-order centred weights of an interior singleton cell
# ---------------------------------------------------------------------------
def test_interior_singleton_cell_weights_are_fourth_order_centred(full):
    n = full.n
    i, j, k = 6, 3, 5
    key = (i, j, k)
    p = _cell(full, key)
    biased = StructuredReconstruction(full).rows(key, p, "cell")
    sym = StructuredReconstruction(full, cell_stencil="symmetric").rows(key, p, "cell")
    assert biased.diagnostics["family"] == "singleton"
    assert sym.diagnostics["family"] == "singleton" and sym.diagnostics["cell_stencil"] == "symmetric"
    assert not sym.boundary_conditioned and sym.value.shape == (1, len(sym.donor_ids)) and sym.gradient.shape == (1, 3, len(sym.donor_ids))
    # biased: third-order, layers i-1..i+2 and eta planes k-2..k+1
    np.testing.assert_allclose(_weights(biased, full, np.arange(i - 1, i + 3), j, k, 0, 6 / n), [-2, -3, 6, -1], atol=1e-12)
    np.testing.assert_allclose(_weights(biased, full, i, j, np.arange(k - 2, k + 2), 2, 6 * full.g.deta), [1, -6, 3, 2], atol=1e-12)
    # symmetric: (1, -8, 0, 8, -1) / 12 h on layers i-2..i+2 and planes k-2..k+2
    layers, planes = np.arange(i - 2, i + 3), np.arange(k - 2, k + 3)
    wu = _weights(sym, full, layers, j, k, 0, 12 / n)
    we = _weights(sym, full, i, j, planes, 2, 12 * full.g.deta)
    np.testing.assert_allclose(wu, [1, -8, 0, 8, -1], atol=1e-12)
    np.testing.assert_allclose(we, [1, -8, 0, 8, -1], atol=1e-12)
    # moments of the five-point rule: sum 0, first 12 (the derivative), second/third/fourth 0 (fourth-order centred)
    x = np.arange(-2, 3).astype(float)
    for w in (wu, we):
        assert abs(w.sum()) < 1e-12 and abs(w @ x - 12) < 1e-12 and abs(w @ x ** 2) < 1e-12
        assert abs(w @ x ** 3) < 1e-12 and abs(w @ x ** 4) < 1e-12
    # value weight 1 on the cell's own owner, nothing else
    own = np.ravel_multi_index(key, (n, n, n))
    np.testing.assert_allclose(sym.value[0, sym.donor_ids == own], 1.0, atol=1e-13)
    assert abs(np.sum(sym.value) - 1.0) < 1e-13 and np.sum(np.abs(sym.value) > 1e-13) == 1
    # a symmetric row is not one tensor factorization: its factors are those of its two parts (paired tensor sources)
    row, factors = StructuredReconstruction(full, cell_stencil="symmetric").rows_with_factors(key, p, "cell")
    assert isinstance(factors, PairedFactors) and factors.a.family == factors.b.family == "singleton"
    _same_row(row, sym)
    _, biased_factors = StructuredReconstruction(full).rows_with_factors(key, p, "cell")
    assert biased_factors is not None and biased_factors.family == "singleton"


# ---------------------------------------------------------------------------
# (c) what must not change
# ---------------------------------------------------------------------------
def test_boundary_coupled_unmirrorable_and_face_rows_unchanged(agglomerated):
    biased = StructuredReconstruction(agglomerated)
    sym = StructuredReconstruction(agglomerated, cell_stencil="symmetric")
    n = agglomerated.n
    kinds = {}
    for key in ((1, 2, 4), (2, 2, 4), (3, 2, 4), (n - 2, 3, 4), (n - 1, 9, 9)):
        p = _cell(agglomerated, key)
        a, b = biased.rows(key, p, "cell"), sym.rows(key, p, "cell")
        kinds[key[0]] = a.diagnostics["family"]
        _same_row(a, b)
        fa, fb = biased.rows_with_factors(key, p, "cell")[1], sym.rows_with_factors(key, p, "cell")[1]
        assert (fa is None) == (fb is None)
    # coupled (1, 2), ringwise cell 3 whose mirror reaches the 4-owner ring 1 (stays biased), boundary (n-2, n-1)
    assert kinds == {1: "coupled_quartic", 2: "coupled_quartic", 3: "ringwise", n - 2: "boundary_transverse",
                     n - 1: "boundary_transverse"}
    for key in FACES:
        p = _face_points(biased, key)
        _same_row(biased.rows(key, p, "face"), sym.rows(key, p, "face"))
    # R3 side rows are anchored on the face, not the cell centre: unchanged, factors kept
    for key in ((1, 6, 3, 4), (2, 5, 7, 2), (0, 6, 3, 4)):
        p = _face_points(biased, key)
        for (ra, fa), (rb, fb) in zip(biased.side_rows_with_factors(key, p), sym.side_rows_with_factors(key, p), strict=True):
            _same_row(ra, rb)
            assert (fa is None) == (fb is None)
    # fixed_radius: the coupled/unconditioned split follows the anchor radius, symmetric still only touches unconditioned rows
    fixed_b = StructuredReconstruction(agglomerated, inner_support="fixed_radius")
    fixed_s = StructuredReconstruction(agglomerated, inner_support="fixed_radius", cell_stencil="symmetric")
    key = (2, 2, 4)
    _same_row(fixed_b.rows(key, _cell(agglomerated, key), "cell"), fixed_s.rows(key, _cell(agglomerated, key), "cell"))


# ---------------------------------------------------------------------------
# (d) a mirror reaching an agglomerated ring (singleton -> ringwise fallback)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ((4, 2, 4), (5, 7, 11), (6, 0, 0)))
def test_mirror_reaching_agglomerated_ring_differentiates_a_cubic(agglomerated, key):
    biased = StructuredReconstruction(agglomerated)
    sym = StructuredReconstruction(agglomerated, cell_stencil="symmetric")
    p = _cell(agglomerated, key)
    a, s = biased.rows(key, p, "cell"), sym.rows(key, p, "cell")
    assert a.diagnostics["family"] == "singleton"                     # A: layers i-1..i+2 all full rings
    assert s.diagnostics["cell_stencil"] == "symmetric"
    assert s.diagnostics["mirror_family"] == ("ringwise" if key[0] == 4 else "singleton")   # ring 2 (8 owners) reached only from i = 4
    assert np.all(np.isfinite(s.value)) and np.all(np.isfinite(s.gradient))
    assert np.all(np.diff(s.donor_ids) > 0)
    # a field of u only, observed on the owners (volume-weighted raw midpoints): exact for any cubic in u
    u = agglomerated.pts[:, 0]
    owners = np.arange(len(agglomerated.vol))
    for poly, dpoly in ((lambda x: 1 + 2 * x - 3 * x ** 2 + 4 * x ** 3, lambda x: 2 - 6 * x + 12 * x ** 2),
                        (lambda x: x ** 3, lambda x: 3 * x ** 2), (lambda x: np.ones_like(x), lambda x: np.zeros_like(x))):
        observed = agglomerated.observe_owners(poly(u), owners)
        u0 = p[0, 0]
        np.testing.assert_allclose(s.value[0] @ observed[s.donor_ids], poly(u0), atol=1e-12, rtol=0)
        np.testing.assert_allclose(s.gradient[0, 0] @ observed[s.donor_ids], dpoly(u0), atol=1e-11, rtol=0)
        np.testing.assert_allclose(s.gradient[0, 1:] @ observed[s.donor_ids], 0.0, atol=1e-11, rtol=0)
        np.testing.assert_allclose(a.gradient[0, 0] @ observed[a.donor_ids], dpoly(u0), atol=1e-11, rtol=0)
    # exactly the average of A and its mirror: the donor union, half weights
    assert set(a.donor_ids) <= set(s.donor_ids)
    assert abs(s.value.sum() - 1.0) < 1e-13 and abs(s.gradient[0, 0].sum()) < 1e-11


def test_symmetric_row_is_half_of_a_plus_mirror(full):
    """``_average_rows`` on hand-made rows: union of donors, half weights, A's family kept."""
    from drbx.geometry.fci_perpendicular_reconstruction import PointRows
    p = np.zeros((2, 3))
    a = PointRows(np.array([1, 3]), np.array([[1., 2.], [3., 4.]]), np.arange(12.).reshape(2, 3, 2), False,
                  np.empty((0, 3)), p.copy(), {"family": "singleton", "max_residual": 1e-3})
    b = PointRows(np.array([3, 7]), np.array([[5., 6.], [7., 8.]]), -np.arange(12.).reshape(2, 3, 2), False,
                  np.empty((0, 3)), p.copy(), {"family": "ringwise", "max_residual": 2e-3, "min_rank": 7})
    m = StructuredReconstruction._average_rows(a, b, p)
    assert list(m.donor_ids) == [1, 3, 7]
    np.testing.assert_array_equal(m.value, [[.5, 3.5, 3.], [1.5, 5.5, 4.]])
    np.testing.assert_array_equal(m.gradient[:, :, 0], 0.5 * a.gradient[:, :, 0])
    np.testing.assert_array_equal(m.gradient[:, :, 1], 0.5 * (a.gradient[:, :, 1] + b.gradient[:, :, 0]))
    np.testing.assert_array_equal(m.gradient[:, :, 2], 0.5 * b.gradient[:, :, 1])
    assert m.diagnostics == {"family": "singleton", "max_residual": 2e-3, "min_rank": 7, "cell_stencil": "symmetric",
                             "mirror_family": "ringwise"}
    assert not m.boundary_conditioned and m.trace_donor_points.shape == (0, 3) and np.array_equal(m.trace_target_points, p)


# ---------------------------------------------------------------------------
# (e) the environment and the artifact identity
# ---------------------------------------------------------------------------
def test_cell_stencil_is_threaded_and_recorded_only_when_symmetric(monkeypatch):
    from p_shared import build_artifact as ba
    from p_shared import replay_support

    assert inspect.signature(replay_support.build_environment).parameters["cell_stencil"].default == "biased"
    assert replay_support.Environment.__dataclass_fields__["cell_stencil"].default == "biased"
    for fn in (ba.build_policy, ba.build_identity, ba.run_full_build, ba._init_worker):
        assert inspect.signature(fn).parameters["cell_stencil"].default == "biased", fn
    argv = ["--n", "32", "--input-root", ".", "--sidecar", "s", "--output", "o", "--workers", "1"]
    assert ba.parse_args(argv).cell_stencil == "biased"
    assert ba.parse_args(argv + ["--cell-stencil", "symmetric"]).cell_stencil == "symmetric"

    assert ba.build_policy("fd", "q3", "profile7") == ba.POLICY == ba.build_policy("fd", "q3", "profile7", cell_stencil="biased")
    sym = ba.build_policy("fd", "q3", "profile7", cell_stencil="symmetric")
    assert sym == {**ba.POLICY, "cell_stencil": "symmetric"}
    assert ba.build_policy("autodiff", "q2", "fixed_radius", cell_stencil="symmetric") == {
        **ba.build_policy("autodiff", "q2", "fixed_radius"), "cell_stencil": "symmetric"}
    assert "cell_stencil" not in ba.POLICY
    with pytest.raises(ValueError):
        ba.build_policy("fd", "q3", "profile7", cell_stencil="bogus")

    monkeypatch.setattr(ba, "_geometry_component_hashes", lambda root, n: {"geometry": "g"})
    monkeypatch.setattr(ba, "_sidecar_component_hashes", lambda path: {"sidecar": "s"})
    kw = dict(n=32, input_root=Path("."), sidecar_path=Path("."), curvature="fd", face_quadrature="q3", inner_support="profile7")
    biased_id = ba.build_identity(**kw)
    assert ba.build_identity(**kw, cell_stencil="biased") == biased_id and biased_id["policy"] == ba.POLICY
    sym_id = ba.build_identity(**kw, cell_stencil="symmetric")
    assert sym_id != biased_id and sym_id["policy"]["cell_stencil"] == "symmetric"
    assert sym_id["source_hashes"] == biased_id["source_hashes"]
    assert "src/drbx/geometry/fci_perpendicular_reconstruction.py" in sym_id["source_hashes"]
    with pytest.raises(ValueError):
        ba.build_identity(**kw, cell_stencil="bogus")
    with pytest.raises(ValueError):
        replay_support.build_environment(n=32, input_root=Path("."), sidecar_path=Path("."), cell_stencil="bogus")
