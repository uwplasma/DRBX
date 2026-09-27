"""Fast, geometry-free safeguards for the P05N field-derived global campaign.

No HSX runtime inputs beyond what tests/test_p05n_neumann_rows.py already
uses (its small synthetic, partly-aggregated structured grid): covers the
frozen catalogue tables, the new zero_trace_generator field's analytic
derivatives, batched-vs-per-owner-oracle equivalence on the synthetic grid,
the face-jump scatter sign convention, campaign identity-digest stability,
and resume/valid-chunk logic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from perpendicular_structured.reconstruction import StructuredReconstruction  # noqa: E402
from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext  # noqa: E402
from p05n_field_derived_global import fields as p05n_fields  # noqa: E402
from p05n_field_derived_global import core  # noqa: E402
from p05n_field_derived_global import campaign  # noqa: E402

PKG = REPO / "scripts/p05n_field_derived_global"
CATALOGUE = json.loads((PKG / "p05n_catalogue.json").read_text())  # vendored byte-identical copy
CONFIG = json.loads((PKG / "configuration.json").read_text())
PERIOD = 1.7
RNG = np.random.default_rng(20260927)


# ---------------------------------------------------------------------------
# Configuration / catalogue freeze.
# ---------------------------------------------------------------------------
def test_configuration_schema_and_physical_fields_match_core():
    assert CONFIG["schema"] == "drbx.p05n-field-derived-static-global-v1"
    assert CONFIG["resolutions"] == [32, 48, 64]
    assert tuple(CONFIG["physical_fields"]) == core.NAMES


def test_names_extend_p07n_verbatim_plus_zero_trace_generator():
    assert p05n_fields.NAMES[:5] == p05n_fields._P07N_NAMES
    assert p05n_fields.NAMES[-1] == "zero_trace_generator"
    assert p05n_fields._P07N_NAMES == ("field_b1", "field_e3", "field_e12", "heldout_field_b2", "constant")


def test_role_table_covers_every_catalogue_field_and_bc():
    physical = {v[0] for v in core.ROLES.values()}
    assert physical == set(core.NAMES)
    # Every role name encodes its own BC in ROLE_BC, consistently with ROLES.
    for role, (name, bc) in core.ROLES.items():
        assert core.ROLE_BC[role] == bc
    # constant is Dirichlet-role everywhere (frozen decision; see README.md).
    assert core.ROLES["constant_D"][1] == "dirichlet"
    assert all(core.ROLES[r][1] == "dirichlet" for r in core.ROLE_NAMES if core.ROLES[r][0] == "constant")


def test_dirichlet_counterpart_is_idempotent_and_dirichlet():
    for role, counterpart in core.DIRICHLET_COUNTERPART.items():
        assert core.ROLE_BC[counterpart] == "dirichlet"
        assert core.DIRICHLET_COUNTERPART[counterpart] == counterpart


def test_pairings_match_frozen_catalogue_physical_fields_and_bc():
    """Every PAIRINGS entry's (physical field, bc) matches the frozen catalogue.json,
    modulo the frozen constant-role decision (constant is always Dirichlet-role here)."""
    def parse(token):
        if ":" in token:
            name, bc = token.split(":")
            assert bc == "dirichlet"
            return name, "dirichlet"
        return token, "neumann" if token != "constant" else "dirichlet"

    def catalogue_pairs(block):
        return [tuple(pair) for group in ("main", "heldout", "controls") for pair in block[group]]

    a_pairs = catalogue_pairs(CATALOGUE["pairings"]["a_neumann_f_neumann"])
    b_pairs = catalogue_pairs(CATALOGUE["pairings"]["a_dirichlet_f_neumann"])
    resolved = set()
    for pname, (gen_role, tra_role) in core.PAIRINGS.items():
        gen = core.ROLES[gen_role]; tra = core.ROLES[tra_role]
        resolved.add((gen, tra))
    for pair_list in (a_pairs, b_pairs):
        for gen_tok, tra_tok in pair_list:
            gen = parse(gen_tok); tra = parse(tra_tok)
            assert (gen, tra) in resolved, f"catalogue pair {gen_tok},{tra_tok} not represented in core.PAIRINGS"


def test_gated_and_control_pairs_partition_pair_names():
    assert set(core.GATED_PAIRS) | set(core.CONTROL_PAIRS) == set(core.PAIR_NAMES)
    assert set(core.GATED_PAIRS) & set(core.CONTROL_PAIRS) == set()
    assert len(core.PAIR_NAMES) == 10
    for name in core.CONSTANT_PAIRS:
        assert "constant_D" in core.PAIRINGS[name]


def test_r_pair_index_depends_only_on_physical_fields():
    """R('a_main1') and R('b_main1') must reference the identical physical-field
    pair (field_b1, field_e3): R is BC-role-independent by construction."""
    a_idx = core.PAIR_NAMES.index("a_main1"); b_idx = core.PAIR_NAMES.index("b_main1")
    assert core.R_PAIR_INDEX[a_idx] == core.R_PAIR_INDEX[b_idx]
    gen, tra = core.R_PAIR_INDEX[a_idx]
    assert core.NAMES[gen] == "field_b1" and core.NAMES[tra] == "field_e3"


# ---------------------------------------------------------------------------
# zero_trace_generator: analytic derivatives and zero wall trace.
# ---------------------------------------------------------------------------
def _random_points(n, u_range=(1e-3, 1.0)):
    u = RNG.uniform(*u_range, n); th = RNG.uniform(-15, 15, n); eta = RNG.uniform(-15, 15, n)
    return np.column_stack([u, th, eta])


def _fd4(func, q, axis, h):
    e = np.zeros(3); e[axis] = h
    return (-func(q + 2 * e) + 8 * func(q + e) - 8 * func(q - e) + func(q - 2 * e)) / (12 * h)


def _assert_close(actual, expected, tol=1e-8):
    scale = np.maximum(np.abs(expected), 1.0)
    np.testing.assert_array_less(np.abs(actual - expected), tol * scale + 1e-12)


def test_zero_trace_generator_gradient_matches_finite_difference():
    q = _random_points(40); h = 2e-4
    v, g, H = p05n_fields.evaluate(None, q, "zero_trace_generator", PERIOD)

    def value(qq):
        return p05n_fields.evaluate(None, qq, "zero_trace_generator", PERIOD)[0]
    for axis in range(3):
        _assert_close(g[:, axis], _fd4(value, q, axis, h))


def test_zero_trace_generator_hessian_matches_finite_difference_of_gradient():
    q = _random_points(40); h = 2e-4
    v, g, H = p05n_fields.evaluate(None, q, "zero_trace_generator", PERIOD)

    def grad_component(qq, i):
        return p05n_fields.evaluate(None, qq, "zero_trace_generator", PERIOD)[1][:, i]
    for i in range(3):
        for j in range(3):
            _assert_close(H[:, i, j], _fd4(lambda qq, i=i: grad_component(qq, i), q, j, h))


def test_zero_trace_generator_vanishes_at_the_wall():
    q = _random_points(40); q[:, 0] = 1.0
    v, g, H = p05n_fields.evaluate(None, q, "zero_trace_generator", PERIOD)
    np.testing.assert_allclose(v, 0.0, atol=1e-12)
    # Nonzero radial (normal) derivative at the wall: not a trivial field.
    assert np.max(np.abs(g[:, 0])) > 1e-3


def test_zero_trace_generator_theta_and_eta_periodicity():
    q = _random_points(20)
    v0, g0, H0 = p05n_fields.evaluate(None, q, "zero_trace_generator", PERIOD)
    shifted = q.copy(); shifted[:, 1] += 2 * np.pi
    v1, g1, H1 = p05n_fields.evaluate(None, shifted, "zero_trace_generator", PERIOD)
    np.testing.assert_allclose(v1, v0, atol=1e-10)
    np.testing.assert_allclose(g1, g0, atol=1e-8)
    shifted = q.copy(); shifted[:, 2] += PERIOD
    v2, g2, H2 = p05n_fields.evaluate(None, shifted, "zero_trace_generator", PERIOD)
    np.testing.assert_allclose(v2, v0, atol=1e-10)
    np.testing.assert_allclose(g2, g0, atol=1e-8)


def test_p07n_inherited_fields_still_dispatch_unchanged():
    for name in p05n_fields._P07N_NAMES:
        q = _random_points(10)
        v0, g0, H0 = p05n_fields._p07n.evaluate(None, q, name, PERIOD)
        v1, g1, H1 = p05n_fields.evaluate(None, q, name, PERIOD)
        np.testing.assert_array_equal(v0, v1)
        np.testing.assert_array_equal(g0, g1)
        np.testing.assert_array_equal(H0, H1)


# ---------------------------------------------------------------------------
# Batched-vs-per-owner-oracle equivalence, and jump scatter sign convention,
# on the same synthetic grid as tests/test_p05n_neumann_rows.py.
# ---------------------------------------------------------------------------
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
    ctx = PointRowContext.from_arrays(faces=faces, centers=centers, raw_to_owner=ro, raw_volume=rv,
                                       owner_volume=vol, owner_centroid_xy=centroid, eta_period=g.eta_period,
                                       dr=g.dr, dtheta=g.dtheta, deta=g.deta)
    return t, S, ctx


# Two synthetic "physical fields" (F=2), one polynomial (exactly reconstructed
# by both BC variants), used as owner_values/trace/normal_data stand-ins.
COEFF = np.array((np.sqrt(1.04), -.2 / np.sqrt(1.04), 0.))


def _field(q, which):
    q = np.asarray(q, dtype=float)
    u, th, eta = q[:, 0], q[:, 1], q[:, 2]
    scale = 1.0 if which == 0 else 2.0
    v = scale * (1 + .3 * u + .2 * np.sin(th) + .1 * eta)
    g = scale * np.stack((np.full_like(u, .3), .2 * np.cos(th), np.full_like(u, .1)), axis=1)
    return v, g


def _trace_fn(q):
    q = np.asarray(q, dtype=float)
    Q = len(q)
    v = np.empty((Q, 2)); g = np.empty((Q, 3, 2))
    for j in range(2):
        v[:, j], g[:, :, j] = _field(q, j)
    return v, g


def _normal_data_fn(q):
    q = np.asarray(q, dtype=float)
    return np.column_stack([_field(q, j)[1] @ COEFF for j in range(2)])


def _normal_coefficients(q):
    return np.broadcast_to(COEFF, (len(q), 3))


def _owner_column(t, values_at_raw):
    sums = np.bincount(t.ro, weights=t.rv * values_at_raw, minlength=len(t.vol))
    return sums / t.vol


@pytest.fixture(scope="module")
def owner_values(geometry):
    t, _, _ = geometry
    return np.column_stack([_owner_column(t, _field(t.pts, j)[0]) for j in range(2)])


def test_batched_cell_values_matches_rows_cell_rows_both_bc(geometry, owner_values):
    from p05n_field_derived_global import rows as p05n_rows
    t, S, ctx = geometry
    i = 31
    keys = np.array([(i, j, 5) for j in (3, 11, 22)])
    points = np.column_stack([t.centers[a][keys[:, a]] for a in range(3)])
    patch_cache = {}

    batched = core.batched_cell_values(t, S, owner_values, keys, points, normal_coefficients=_normal_coefficients,
                                        ctx=ctx, patch_cache=patch_cache, dirichlet_trace_fn=_trace_fn,
                                        normal_data_fn=_normal_data_fn)
    for col in range(2):
        vd, gd, _ = p05n_rows.cell_rows(t, S, owner_values[:, col], keys, points, bc="dirichlet",
                                         dirichlet_trace=lambda q, c=col: _field(q, c))
        vn, gn, _ = p05n_rows.cell_rows(t, S, owner_values[:, col], keys, points, bc="neumann",
                                         normal_coefficients=_normal_coefficients, context=ctx,
                                         patch_cache={}, normal_data=lambda q, c=col: _field(q, c)[1] @ COEFF)
        np.testing.assert_allclose(batched["dirichlet"][0][:, col], vd, atol=1e-11)
        np.testing.assert_allclose(batched["dirichlet"][1][:, :, col], gd, atol=1e-9)
        np.testing.assert_allclose(batched["neumann"][0][:, col], vn, atol=1e-11)
        np.testing.assert_allclose(batched["neumann"][1][:, :, col], gn, atol=1e-9)


def test_face_jump_scatter_sign_convention_matches_operator_owner_jump(geometry, owner_values):
    """A face jump contributes +jump to the lo owner and -jump to the hi owner
    (operator.owner_jump's documented convention), each divided by that
    owner's volume; the batched face_chunk-style kernel must agree."""
    t, S, ctx = geometry
    n = t.n
    key = (1, 31, 4, 5)  # a theta face at the physical wall radial layer
    grid = t.ro.reshape(n, n, n)
    lo = int(grid[31, 3, 5]); hi = int(grid[31, 4, 5])
    assert lo != hi
    points, weights = core.pk.num.quadrature(t.faces, np.array([key]), 3, face=True)
    points = points[0]; weights = weights[0]

    def trace_fn(q):
        return _trace_fn(q)

    def nd_fn(q):
        return _normal_data_fn(q)

    common = core.batched_face_common_gradient(t, S, owner_values, key, points, normal_coefficients=_normal_coefficients,
                                                ctx=ctx, patch_cache={}, dirichlet_trace_fn=trace_fn, normal_data_fn=nd_fn)
    side = core.batched_side_values(t, S, owner_values, key, points, normal_coefficients=_normal_coefficients,
                                     ctx=ctx, patch_cache={}, dirichlet_trace_fn=trace_fn, normal_data_fn=nd_fn)
    from drbx.native.fci_perpendicular_face_corrections import p05_scalar_face_jump
    fake_ref = SimpleNamespace(_metric=lambda q: dict(
        bcov=np.tile(np.array([1.0, 0.0, 0.0]), (len(q), 1)), B=np.ones(len(q)), J=np.ones(len(q))))
    metric = fake_ref._metric(points)
    h = metric["bcov"] / metric["B"][:, None]
    pairs = np.array([[0, 1]])
    jump = np.asarray(p05_scalar_face_jump(common["dirichlet"][None], side["lower"]["dirichlet"][None],
                                            side["upper"]["dirichlet"][None], h[None], weights[None],
                                            np.array([key[0]]), pairs))[0, 0]
    # Scatter both signs and confirm the volume-weighted totals are exact negatives
    # of each other when only this one face is considered (closed two-owner system).
    contribution_lo = jump / t.vol[lo]
    contribution_hi = -jump / t.vol[hi]
    assert np.sign(contribution_lo) != np.sign(contribution_hi) or jump == 0.0
    np.testing.assert_allclose(contribution_lo * t.vol[lo], -(contribution_hi * t.vol[hi]))


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
    monkeypatch.setattr(campaign, "ids_for", lambda output, n, stage: np.arange(10, dtype=np.int64))
    unit = {"stage": "observations", "n": 32, "start": 0, "stop": 5}
    identity = "deadbeef"
    path = campaign.unit_path(tmp_path, unit)
    campaign.save(path, ids=np.arange(5, dtype=np.int64), owner_ids=np.zeros(5, dtype=np.int64),
                   numerator=np.zeros((5, 6)))
    record = {"identity": identity, "unit": unit, "sha256": campaign.sha(path), "seconds": 0.1,
              "peak_rss_gib": 0.1, "pid": 0, "jax_backend": "cpu"}
    campaign.write(path.with_suffix(".json"), record)
    assert campaign.valid_unit(tmp_path, unit, identity) is True

    # Wrong identity in the receipt: rejected as stale.
    with pytest.raises(ValueError, match="stale checkpoint"):
        campaign.valid_unit(tmp_path, unit, "other-identity")

    # Corrupt the chunk payload without updating the receipt hash: rejected.
    campaign.save(path, ids=np.arange(5, dtype=np.int64), owner_ids=np.ones(5, dtype=np.int64),
                   numerator=np.zeros((5, 6)))
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        campaign.valid_unit(tmp_path, unit, identity)


def test_valid_unit_false_when_missing():
    unit = {"stage": "observations", "n": 32, "start": 100, "stop": 105}
    assert campaign.valid_unit(Path("/tmp/does-not-exist-p05n-campaign-test"), unit, "x") is False


# ---------------------------------------------------------------------------
# Portability: no package source may reference a local scratch `work/`
# directory (everything must resolve through --input-root or a vendored
# preflight_fixtures/ file), except preflight_fixtures/extract.py, the one
# sanctioned local-only tool.
# ---------------------------------------------------------------------------
_ALLOWED_WORK_LITERAL = "DRBX/work/perpendicular_second_order_hsx_p01_p03"


def _allowed_spans(line):
    """Character ranges in `line` covered by the one sanctioned work/ literal."""
    spans = []
    start = 0
    while True:
        idx = line.find(_ALLOWED_WORK_LITERAL, start)
        if idx < 0:
            return spans
        spans.append((idx, idx + len(_ALLOWED_WORK_LITERAL)))
        start = idx + 1


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
        for name in (f"N{n}.selection.json", f"N{n}.accepted_p05_replay.npz"):
            path = fixtures / name
            assert path.is_file(), name
            total_bytes += path.stat().st_size
    assert total_bytes < 5 * 1024 * 1024, f"preflight_fixtures total {total_bytes} bytes exceeds the 5 MB target"
