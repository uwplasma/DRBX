"""Fast, mostly geometry-free safeguards for the P06N global static campaign
(the accepted P06 curvature operator under physical-normal Neumann rows).

Covers: the frozen catalogue's freeze/hash and case-table partitioning, the
two P06N rich fields' analytic derivatives (finite differences, periodicity,
axis regularity, positivity), the accepted-vs-deduplicated face census
(``owner_all_faces``'s ``dedupe`` flag), the q3 "add both sides
independently" scatter sign convention (not P05N/P07N's antisymmetric
jump), campaign identity-digest stability, resume/valid-chunk logic, and the
no-local-``work/``-literal portability guard.

Uses the small synthetic, partly-aggregated structured grid
``tests/test_p06n_neumann_rows.py`` already builds (no HSX runtime inputs)
for the census/scatter tests; the rich-field and catalogue tests need no
grid at all.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from p06n_field_derived_global import fields as p06n_fields  # noqa: E402
from p06n_field_derived_global import core  # noqa: E402
from p06n_field_derived_global import campaign  # noqa: E402
from p06n_field_derived_global import operator as p06n_operator  # noqa: E402

PKG = REPO / "scripts/p06n_field_derived_global"
CONFIG = json.loads((PKG / "configuration.json").read_text())
CATALOGUE = json.loads((PKG / CONFIG["catalogue_reference"]).read_text())
PERIOD = 1.7
RNG = np.random.default_rng(20260928)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Catalogue freeze / hash / partitioning.
# ---------------------------------------------------------------------------
def test_configuration_schema_and_resolutions():
    assert CONFIG["schema"] == "drbx.p06n-field-derived-static-global-v1"
    assert CONFIG["resolutions"] == [32, 48, 64]


def test_catalogue_file_hash_matches_configuration():
    assert _sha(PKG / "p06n_catalogue.json") == CONFIG["catalogue_sha256"]


def test_catalogue_state_order_and_case_count():
    assert CATALOGUE["state_order"] == ["n", "Te", "Ti", "omega", "phi"]
    assert set(CATALOGUE["cases"]) == set(core.CASE_NAMES)
    assert len(core.CASE_NAMES) == 7


def test_case_tables_partition_correctly():
    assert set(core.MAIN_CASES) | set(core.HELDOUT_CASES) | set(core.CONTROL_CASES) | {"dirichlet_rich"} == set(core.CASE_NAMES)
    assert not (set(core.MAIN_CASES) & set(core.HELDOUT_CASES))
    assert set(core.GATED_CASES) == set(core.NONCONSTANT_CASES)
    assert set(core.GATED_CASES) == set(core.MAIN_CASES) | set(core.HELDOUT_CASES) | {"dirichlet_rich"}


def test_heldout_rich_h_only_appears_in_heldout_cases():
    for name in set(core.GATED_CASES) - set(core.HELDOUT_CASES):
        fields_used = {f for f, _bc in core.CASES[name]}
        assert "heldout_rich_h" not in fields_used, f"{name} (gated) must never use the frozen held-out field"
    heldout_fields = {f for name in core.HELDOUT_CASES for f, _bc in core.CASES[name]}
    assert "heldout_rich_h" in heldout_fields


def test_matched_dirichlet_variant_is_all_dirichlet_same_fields():
    for name in core.CASE_NAMES:
        spec = core.CASES[name]
        d_spec = core.matched_dirichlet_spec(spec)
        assert tuple(f for f, _bc in d_spec) == tuple(f for f, _bc in spec)
        assert all(bc == "dirichlet" for _f, bc in d_spec)


def test_variant_names_are_case_names_plus_d_suffix():
    assert core.VARIANT_NAMES == tuple(core.CASE_NAMES) + tuple(f"{c}:D" for c in core.CASE_NAMES)
    assert len(core.VARIANT_NAMES) == 14


def test_configuration_case_lists_match_core():
    assert CONFIG["cases"] == list(core.CASE_NAMES)
    assert CONFIG["gated_cases"] == list(core.GATED_CASES)
    assert CONFIG["heldout_cases"] == list(core.HELDOUT_CASES)
    assert CONFIG["control_cases"] == list(core.CONTROL_CASES)
    assert CONFIG["physical_fields"] == list(core.NAMES)


# ---------------------------------------------------------------------------
# Rich fields: derivatives vs finite differences, periodicity, axis
# regularity, positivity (both live within the catalogue's documented
# 0.9-1.1 band, per p06n_catalogue.json's "coordinates" note).
# ---------------------------------------------------------------------------
def _random_points(n, u_range=(1e-3, 1.0)):
    u = RNG.uniform(*u_range, n); th = RNG.uniform(-15, 15, n); eta = RNG.uniform(-15, 15, n)
    return np.column_stack([u, th, eta])


def _fd4(func, q, axis, h):
    e = np.zeros(3); e[axis] = h
    return (-func(q + 2 * e) + 8 * func(q + e) - 8 * func(q - e) + func(q - 2 * e)) / (12 * h)


@pytest.mark.parametrize("name", ["rich_c", "heldout_rich_h"])
def test_p06n_rich_field_gradient_and_hessian_match_finite_differences(name):
    q = _random_points(40, u_range=(0.05, 1.0))
    v, g, H = p06n_fields.evaluate(None, q, name, PERIOD)

    def value(qq):
        return p06n_fields.evaluate(None, qq, name, PERIOD)[0]

    def grad_component(qq, i):
        return p06n_fields.evaluate(None, qq, name, PERIOD)[1][:, i]

    for axis in range(3):
        np.testing.assert_allclose(g[:, axis], _fd4(value, q, axis, 1e-3), atol=1e-7)
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(H[:, i, j], _fd4(lambda qq, i=i: grad_component(qq, i), q, j, 1e-3), atol=1e-6)


@pytest.mark.parametrize("name", ["rich_c", "heldout_rich_h"])
def test_p06n_rich_field_periodicity(name):
    q = _random_points(30)
    v0, g0, _ = p06n_fields.evaluate(None, q, name, PERIOD)
    shifted = q.copy(); shifted[:, 1] += 2 * np.pi
    v1, g1, _ = p06n_fields.evaluate(None, shifted, name, PERIOD)
    np.testing.assert_allclose(v1, v0, atol=1e-10); np.testing.assert_allclose(g1, g0, atol=1e-8)
    shifted = q.copy(); shifted[:, 2] += PERIOD
    v2, g2, _ = p06n_fields.evaluate(None, shifted, name, PERIOD)
    np.testing.assert_allclose(v2, v0, atol=1e-10); np.testing.assert_allclose(g2, g0, atol=1e-8)


@pytest.mark.parametrize("name", ["rich_c", "heldout_rich_h"])
def test_p06n_rich_field_axis_regularity(name):
    """At u -> 0 every term is a smooth function of (u cos th, u sin th)
    times eta harmonics, so the value must be independent of theta."""
    ring = np.column_stack([np.full(24, 1e-7), np.linspace(0, 2 * np.pi, 24, endpoint=False), np.full(24, 0.3)])
    v = p06n_fields.evaluate(None, ring, name, PERIOD)[0]
    assert np.ptp(v) < 1e-6


@pytest.mark.parametrize("name", ["rich_c", "heldout_rich_h"])
def test_p06n_rich_field_positivity_band(name):
    """Catalogue's own documented range: n, Te, Ti stay within about 0.9-1.1."""
    q = _random_points(200, u_range=(0.0, 1.0))
    v = p06n_fields.evaluate(None, q, name, PERIOD)[0]
    assert np.all(v > 0.85) and np.all(v < 1.15)


def test_shifted_variants_are_shift_invariant_in_gradient_and_normal_data():
    q = _random_points(20, u_range=(0.5, 1.0))
    for shifted, base in (("rich_a_minus1", "rich_a"), ("rich_f_minus1", "rich_f"),
                          ("field_b1_minus1", "field_b1")):
        vs, gs, Hs = p06n_fields.evaluate(None, q, shifted, PERIOD)
        vb, gb, Hb = p06n_fields.evaluate(None, q, base, PERIOD)
        np.testing.assert_allclose(vs, vb - 1.0, atol=1e-12)
        np.testing.assert_allclose(gs, gb, atol=1e-12)
        np.testing.assert_allclose(Hs, Hb, atol=1e-12)


# ---------------------------------------------------------------------------
# Accepted-vs-deduplicated face census (owner_all_faces's dedupe flag).
# ---------------------------------------------------------------------------
def test_owner_all_faces_dedupe_drops_exactly_the_periodic_duplicate():
    """A synthetic small periodic grid: an owner whose raw-cell traversal
    reaches theta face-slot n (i.e. one with a member at theta index n-1)
    has exactly one more face in accepted-census mode than deduplicated."""
    n = 8
    ro = np.arange(n ** 3, dtype=np.int64)  # trivial: every raw cell its own owner
    faces = np.linspace(0.0, 1.0, n + 1)
    t = type("T", (), {})()
    t.n = n; t.ro = ro
    owner = int(ro.reshape(n, n, n)[3, n - 1, 2])  # a raw cell at the last theta index
    faces_accepted = p06n_operator.owner_all_faces(t, owner, dedupe=False)
    faces_dedup = p06n_operator.owner_all_faces(t, owner, dedupe=True)
    assert len(faces_accepted) == len(faces_dedup) + 1
    dedup_keys = {key for key, _lo, _hi in faces_dedup}
    extra = [key for key, _lo, _hi in faces_accepted if key not in dedup_keys]
    assert len(extra) == 1
    axis, i, j, k = extra[0]
    assert axis == 1 and j == n  # the theta periodic-duplicate slot


def test_periodic_duplicate_face_predicate_matches_p06_structured_global():
    import p06_structured_global.numerics as p06numerics
    n = 16
    keys = p06numerics.base._face_keys(n, np.arange(p06numerics._face_count(n)))
    expected = p06numerics._periodic_duplicate_face(n, keys)
    mine = ((keys[:, 0] == 1) & (keys[:, 2] == n)) | ((keys[:, 0] == 2) & (keys[:, 3] == n))
    np.testing.assert_array_equal(expected, mine)


# ---------------------------------------------------------------------------
# q3 scatter sign convention: "add both sides independently" (accepted P06's
# own convention, NOT the antisymmetric +jump/-jump convention P05N/P07N use
# for their genuinely antisymmetric bracket jump).
# ---------------------------------------------------------------------------
def test_reduce_faces_scatter_adds_both_sides_independently(tmp_path, monkeypatch):
    """An internal aggregation-seam face (lo == hi == owner) must contribute
    BOTH its lower and upper correction to that one owner (sum, not
    difference) -- exactly what an antisymmetric-jump reducer would get wrong."""
    n = 4
    owners = 3
    V = len(core.VARIANT_NAMES)

    class FakeT:
        pass
    t = FakeT(); t.vol = np.ones(owners)

    monkeypatch.setattr(core, "load_context", lambda n_, root: t)
    monkeypatch.setattr(campaign, "ids_for", lambda n_, stage: np.array([0], dtype=np.int64))

    unit = {"stage": "faces", "n": n, "start": 0, "stop": 1}
    path = campaign.unit_path(tmp_path, unit)
    correction_lo = np.zeros((V, 1, 4)); correction_lo[:, 0] = 1.0
    correction_hi = np.zeros((V, 1, 4)); correction_hi[:, 0] = 2.0
    campaign.save(path, ids=np.array([0]), keys=np.array([[1, 0, 0, 0]]),
                  lo=np.array([0]), hi=np.array([0]),  # internal seam: lo == hi == owner 0
                  correction_lo=correction_lo, correction_hi=correction_hi,
                  condition_max=0.0, residual_max=0.0, wall_exterior_defect_max=0.0, wall_faces_seen=0,
                  zero_physical_wall=0.0, zero_radial_n_minus_1=0.0, zero_transverse_last_two_layers=0.0)
    record = {"identity": "id", "unit": unit, "sha256": campaign.sha(path), "seconds": 0.1,
              "peak_rss_gib": 0.1, "pid": 0, "jax_backend": "cpu"}
    campaign.write(path.with_suffix(".json"), record)

    result = campaign.reduce_faces(None, tmp_path, n, [unit], "id")
    with np.load(tmp_path / f"N{n}.faces.npz", allow_pickle=False) as z:
        correction = z["correction"]
    # Owner 0 gets BOTH the lower (1.0) and upper (2.0) contribution: sum == 3.0.
    np.testing.assert_allclose(correction[:, 0], 3.0)
    np.testing.assert_allclose(correction[:, 1:], 0.0)


# ---------------------------------------------------------------------------
# Campaign identity digest stability.
# ---------------------------------------------------------------------------
def test_config_matches_frozen_schema_and_gate_thresholds():
    cfg = campaign.config()
    assert cfg["global_l2_order_minimum"] == 1.8
    assert cfg["constant_action_absolute_maximum"] == 1e-8


def test_digest_is_stable_and_sensitive():
    obj = {"a": 1, "b": [1, 2, 3], "c": {"d": "e"}}
    d1 = campaign.digest(obj); d2 = campaign.digest(dict(obj))
    assert d1 == d2
    changed = dict(obj); changed["a"] = 2
    assert campaign.digest(changed) != d1


def test_source_hashes_are_stable_across_calls():
    h1 = campaign.source_hashes(); h2 = campaign.source_hashes()
    assert h1 == h2
    assert all(len(v) == 64 for v in h1.values())


# ---------------------------------------------------------------------------
# Resume / valid-chunk logic.
# ---------------------------------------------------------------------------
def test_valid_unit_accepts_a_well_formed_chunk_and_rejects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ids_for", lambda n_, stage: np.arange(10, dtype=np.int64))
    unit = {"stage": "observations", "n": 32, "start": 0, "stop": 5}
    identity = "deadbeef"
    path = campaign.unit_path(tmp_path, unit)
    campaign.save(path, ids=np.arange(5, dtype=np.int64), owner_ids=np.zeros(5, dtype=np.int64),
                  numerator=np.zeros((5, len(core.NAMES))))
    record = {"identity": identity, "unit": unit, "sha256": campaign.sha(path), "seconds": 0.1,
              "peak_rss_gib": 0.1, "pid": 0, "jax_backend": "cpu"}
    campaign.write(path.with_suffix(".json"), record)
    assert campaign.valid_unit(tmp_path, unit, identity) is True

    with pytest.raises(ValueError, match="stale checkpoint"):
        campaign.valid_unit(tmp_path, unit, "other-identity")

    campaign.save(path, ids=np.arange(5, dtype=np.int64), owner_ids=np.ones(5, dtype=np.int64),
                  numerator=np.zeros((5, len(core.NAMES))))
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        campaign.valid_unit(tmp_path, unit, identity)


def test_valid_unit_false_when_missing():
    unit = {"stage": "observations", "n": 32, "start": 100, "stop": 105}
    assert campaign.valid_unit(Path("/tmp/does-not-exist-p06n-campaign-test"), unit, "x") is False


def test_ids_for_faces_excludes_collapsed_r0():
    ids = campaign.ids_for(16, "faces")
    keys = core.face_keys_for_ids(16, ids)
    assert not np.any((keys[:, 0] == 0) & (keys[:, 1] == 0))


def test_ids_for_raw_is_full_raw_cell_range():
    ids = campaign.ids_for(16, "raw")
    np.testing.assert_array_equal(ids, np.arange(16 ** 3, dtype=np.int64))


# ---------------------------------------------------------------------------
# Portability: no package source may reference a local scratch `work/`
# directory as a runtime path, except preflight_fixtures/extract.py (the one
# sanctioned local-only tool) and campaign.py's one functional localized-
# sidecar literal (shared verbatim with p05n/p07n's own campaign.py; the
# provenance is under --input-root, not a scratch path). A handful of
# pre-existing frozen files (fields.py, operator.py -- built before this task
# and read-only per the task's scope) cite historical `work/...` paths in
# *documentation only* (docstrings/comments), never in code that resolves a
# path at runtime; those specific citations are allow-listed by exact
# substring rather than edited, since this task does not otherwise touch
# those files' prose.
# ---------------------------------------------------------------------------
_ALLOWED_WORK_LITERALS = (
    "DRBX/work/perpendicular_second_order_hsx_p01_p03",  # campaign.py: functional, under --input-root
    "work/p_neumann_p05n_p06n_design_20260927",           # fields.py: docstring provenance citation
    "work/p06n_exact_input_screen_20260927",              # operator.py: docstring provenance citation
)


def _allowed_spans(line):
    spans = []
    for literal in _ALLOWED_WORK_LITERALS:
        start = 0
        while True:
            idx = line.find(literal, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(literal)))
            start = idx + 1
    return spans


def test_package_sources_never_reference_a_local_work_directory():
    exempt = {PKG / "preflight_fixtures" / "extract.py"}
    offenders = []
    for path in sorted(PKG.rglob("*.py")):
        if "__pycache__" in path.parts or path in exempt:
            continue
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            allowed = _allowed_spans(line)
            start = 0
            while True:
                idx = line.find("work/", start)
                if idx < 0:
                    break
                covered = any(lo <= idx and idx + len("work/") <= hi for lo, hi in allowed)
                if not covered:
                    offenders.append(f"{path.relative_to(PKG)}:{lineno}: {line.strip()}")
                start = idx + 1
    assert not offenders, "package sources reference a local work/ directory:\n" + "\n".join(offenders)


def test_input_manifest_matches_p07n_byte_identically():
    p07n = REPO / "scripts/p07n_field_derived_global/input_manifest.json"
    assert (PKG / "input_manifest.json").read_bytes() == p07n.read_bytes()


def test_preflight_fixtures_are_small_and_present_for_every_resolution():
    fixtures = PKG / "preflight_fixtures"
    manifest = json.loads((fixtures / "fixtures_manifest.json").read_text())
    total_bytes = 0
    for n in (32, 48, 64):
        assert str(n) in manifest["resolutions"]
        for name in (f"N{n}.selection.json", f"N{n}.replay.npz"):
            path = fixtures / name
            assert path.is_file(), name
            total_bytes += path.stat().st_size
    assert total_bytes < 5 * 1024 * 1024, f"preflight_fixtures total {total_bytes} bytes exceeds the 5 MB target"
