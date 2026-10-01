"""Fast tests of the P08 step-5 P07 export campaign (``scripts/p08_step5_export``) on the synthetic world.

The environment, artifact loading, option check and lowering are monkeypatched so that ``run`` exports the
synthetic ``lower_world(world, include=("p07",))`` plan; no real geometry or row artifact is needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
sys.path.insert(0, str(REPO / "scripts"))

from drbx.native.fci_perpendicular_p07_operator import p07_action                  # noqa: E402
from drbx.native.fci_perpendicular_p07_sparse import (                             # noqa: E402
    KINDS, apply_p07_sparse, export_p07_sparse, load_p07_sparse)
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData        # noqa: E402
from p08_step5_export import campaign                                              # noqa: E402
from tests.perpendicular_synthetic import lower_world, make_world                  # noqa: E402

FINAL = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius"}
OWNERS = [0, 1, 2, 40, 96, 99, 100, 114]


@pytest.fixture(scope="module")
def plan():
    return lower_world(make_world(owners=OWNERS), include=("p07",))


def _step4(tmp_path, grids=(32,), **overrides):
    s4 = tmp_path / "step4" / "campaign"
    (s4 / "artifact").mkdir(parents=True)
    (s4 / "localized_sidecar.json").write_text("{}")
    (s4 / "campaign_manifest.json").write_text(json.dumps({"identity": "step4id"}))
    val = {"identity": "step4id", "operator_options": FINAL,
           "grids": {str(n): {"smoke_pass": True, "preflight_pass": True} for n in grids}}
    val.update(overrides)
    (s4 / "validation.json").write_text(json.dumps(val))
    for n in grids:
        g = s4 / "artifact" / f"N{n}"
        g.mkdir()
        (g / "manifest.json").write_text(json.dumps({"schema": "x"}))
        (g / "build_identity.json").write_text(json.dumps({"policy": {"n": n}}))
    return s4


@pytest.fixture
def patched(plan, monkeypatch):
    calls = {"env": [], "lower": 0, "options": []}
    n_owners = len(plan.p07.owner_volume)
    rng = np.random.default_rng(5)

    def fake_env(*, n, input_root, sidecar_path, **options):
        calls["env"].append((n, Path(sidecar_path).name, options))
        return SimpleNamespace(t=SimpleNamespace(vol=np.asarray(plan.p07.owner_volume), ro=np.arange(7),
                                                 centers=[rng.normal(size=n_owners) for _ in range(3)]))

    def fake_load(root, n, **kw):
        g = Path(root) / f"N{n}"
        return {"grid_dir": g, "identity": json.loads((g / "build_identity.json").read_text())}

    def fake_lower(env, artifact, root, n):
        calls["lower"] += 1
        return plan

    monkeypatch.setattr(campaign, "build_environment", fake_env)
    monkeypatch.setattr(campaign.replay, "load_artifact", fake_load)
    monkeypatch.setattr(campaign, "check_artifact_options", lambda identity, options: calls["options"].append(options))
    monkeypatch.setattr(campaign, "lower_p07_plan", fake_lower)
    monkeypatch.setattr(campaign.replay, "jax_info", lambda: {"backend": "cpu", "x64": True})
    return calls


def _argv(command, s4, tmp_path, *extra):
    return [command, "--step4-campaign", str(s4), "--input-root", str(tmp_path), "--output", str(tmp_path / "out"),
            *extra]


def test_config_pins_options_and_refuses_drift(monkeypatch):
    cfg = campaign.config()
    assert cfg["schema"] == "drbx.p08-step5-p07-export-v1"
    assert cfg["operator_options"] == FINAL == campaign.OPERATOR_OPTIONS == campaign.operator_options()
    assert cfg["resolutions"] == [32, 48, 64] and cfg["kinds"] == list(KINDS)
    assert cfg["gate_relative_tolerance"] == 1e-11 and cfg["gate_columns"] == 2 and cfg["gate_seed"] == 0
    monkeypatch.setattr(campaign, "OPERATOR_OPTIONS", {**FINAL, "face_quadrature": "q3"})
    with pytest.raises(ValueError, match="differ from the pinned final bundle"):
        campaign.config()


def test_sources_exist_and_artifact_options_check():
    for rel, h in campaign.source_hashes().items():
        assert (REPO / rel).is_file() and len(h) == 64
    campaign.check_artifact_options({"policy": {"curvature": "autodiff", "quadrature": {"face": "q2"},
                                                "inner_support": "fixed_radius"}}, FINAL)
    with pytest.raises(ValueError):
        campaign.check_artifact_options({"policy": {}}, FINAL)


def test_cli_parsing(tmp_path):
    a = campaign.parse(["run", "--step4-campaign", "s", "--input-root", "i", "--output", "o"])
    assert a.command == "run" and a.grids == [32, 48, 64] and a.step4_campaign == Path("s")
    a = campaign.parse(["validate", "--step4-campaign", "s", "--input-root", "i", "--output", "o",
                        "--grids", "32", "64"])
    assert a.grids == [32, 64]
    with pytest.raises(SystemExit):
        campaign.parse(["run", "--step4-campaign", "s", "--input-root", "i", "--output", "o", "--grids", "40"])
    with pytest.raises(SystemExit):
        campaign.parse(["bogus", "--step4-campaign", "s", "--input-root", "i", "--output", "o"])


def test_lower_p07_plan_passes_only_p07(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(campaign, "GeometryArrays", SimpleNamespace(load=lambda p: "geometry"))
    monkeypatch.setattr(campaign, "LoaderGrid", SimpleNamespace(from_arrays=lambda **kw: "grid"))
    monkeypatch.setattr(campaign, "lower_perpendicular_plan_from_artifact",
                        lambda root, n, **kw: seen.update(root=root, n=n, **kw) or "plan")
    env = SimpleNamespace(census="census", t=SimpleNamespace(ro=1, rv=2, vol=3, centers=[0, 1, 2]))
    out = campaign.lower_p07_plan(env, {"grid_dir": tmp_path, "identity": {"a": 1}}, tmp_path, 32)
    assert out == "plan" and seen["include"] == ("p07",) and seen["n"] == 32 and seen["identity"] == {"a": 1}


def test_verify_inputs_refuses_failed_grid_and_non_cpu(tmp_path, patched, monkeypatch):
    s4 = _step4(tmp_path, grids=(32, 48))
    val = json.loads((s4 / "validation.json").read_text())
    val["grids"]["48"]["preflight_pass"] = False
    (s4 / "validation.json").write_text(json.dumps(val))
    with pytest.raises(ValueError, match="N48 did not pass"):
        campaign.main(_argv("verify-inputs", s4, tmp_path))
    with pytest.raises(ValueError, match="no grid N64"):
        campaign.main(_argv("verify-inputs", s4, tmp_path, "--grids", "64"))
    campaign.main(_argv("verify-inputs", s4, tmp_path, "--grids", "32"))          # the passing grid alone is fine
    assert (tmp_path / "out" / "provenance" / "inputs.json").is_file()
    monkeypatch.setattr(campaign.replay, "jax_info", lambda: {"backend": "gpu", "x64": True})
    with pytest.raises(ValueError, match="CPU backend"):
        campaign.main(_argv("verify-inputs", s4, tmp_path, "--grids", "32"))
    monkeypatch.setattr(campaign.replay, "jax_info", lambda: {"backend": "cpu", "x64": False})
    with pytest.raises(ValueError, match="x64"):
        campaign.main(_argv("verify-inputs", s4, tmp_path, "--grids", "32"))


def test_verify_inputs_records_provenance_and_missing_files(tmp_path, patched):
    s4 = _step4(tmp_path)
    summary = campaign.main(_argv("verify-inputs", s4, tmp_path, "--grids", "32"))
    rec = json.loads((tmp_path / "out" / "provenance" / "inputs.json").read_text())
    assert rec["identity"] == summary["identity"] and rec["step4_identity"] == "step4id"
    assert len(rec["grids"]["32"]["artifact_identity_sha256"]) == 64
    assert (tmp_path / "out" / "last_exit.json").is_file() and list((tmp_path / "out" / "invocations").glob("*"))
    (s4 / "artifact" / "N32" / "build_identity.json").unlink()
    with pytest.raises(ValueError, match="missing step-4 artifact file"):
        campaign.main(_argv("verify-inputs", s4, tmp_path, "--grids", "32"))
    (s4 / "artifact" / "N32" / "build_identity.json").write_text("{}")
    (s4 / "localized_sidecar.json").unlink()
    with pytest.raises(ValueError, match="localized_sidecar"):
        campaign.main(_argv("verify-inputs", s4, tmp_path, "--grids", "32"))


def test_run_exports_gates_and_resumes(tmp_path, patched, plan, capsys):
    s4 = _step4(tmp_path, grids=(32, 48))
    summary = campaign.main(_argv("run", s4, tmp_path, "--grids", "32", "48"))
    assert summary["all_pass"] and set(summary["grids"]) == {"32", "48"}
    assert [c[0] for c in patched["env"]] == [32, 48] and patched["lower"] == 2
    assert all(c[1] == "localized_sidecar.json" and c[2] == FINAL for c in patched["env"])
    out = tmp_path / "out"
    identity = summary["identity"]
    for n in (32, 48):
        rec = json.loads((out / f"N{n}" / "export_receipt.json").read_text())
        assert rec["pass"] and rec["identity"] == identity and rec["n_owners"] == len(plan.p07.owner_volume)
        for kind in KINDS:
            assert rec["gates"][kind]["pass"] and rec["gates"][kind]["full"]["max_abs_action"] > 0
            assert set(rec["nnz"][kind]) == set(campaign.BLOCKS) and rec["nnz"][kind]["matrix"] > 0
            assert rec["files"][kind]["bytes"] > 0 and len(rec["files"][kind]["sha256"]) == 64
        assert rec["nnz"]["neumann"]["neumann_normal"] > 0 and rec["nnz"]["dirichlet"]["neumann_normal"] == 0
        assert {"environment", "lower_p07", "export_dirichlet", "gate_neumann"} <= set(rec["seconds"])
        omap = np.load(out / f"N{n}" / "owner_map.npz")
        assert set(omap.files) == {"owner_volume", "raw_to_owner", "centers_u", "centers_theta", "centers_eta"}
        assert np.array_equal(omap["owner_volume"], np.asarray(plan.p07.owner_volume))
    # exported files load back with the exact identity and reproduce the action
    rng = np.random.default_rng(3)
    u = rng.normal(size=(len(plan.p07.owner_volume), 2))
    qd, qn = len(plan.dirichlet_points), len(plan.neumann_points)
    bc = BoundaryData(rng.normal(size=(qd, 2)), rng.normal(size=(qd, 2, 2)), rng.normal(size=(qn, 2)))
    art_sha = json.loads((out / "N32" / "export_receipt.json").read_text())["artifact_identity_sha256"]
    for kind in KINDS:
        ident = {"campaign": identity, "n": 32, "kind": kind, "artifact_identity_sha256": art_sha,
                 "operator_options": FINAL}
        op = load_p07_sparse(out / "N32" / f"p07_{kind}.npz", ident)
        ref = np.asarray(p07_action(plan, u, bc, kind))
        assert np.max(np.abs(apply_p07_sparse(op, u, bc) - ref)) <= 1e-11 * np.max(np.abs(ref))
    # validation outputs
    val = json.loads((out / "validation.json").read_text())
    assert val["all_pass"] and set(val["grids"]) == {"32", "48"}
    sm = json.loads((out / "summary" / "step5_export_summary.json").read_text())
    assert sm["grids"]["32"]["nnz"]["neumann"]["matrix"] > 0 and sm["grids"]["32"]["bytes"]["dirichlet"] > 0
    # resumable: a second run skips every grid
    again = campaign.main(_argv("run", s4, tmp_path, "--grids", "32", "48"))
    assert {r["status"] for r in again["grids"].values()} == {"skipped"}
    assert patched["lower"] == 2 and len(patched["env"]) == 2


def test_failed_gate_writes_receipt_and_exits_nonzero(tmp_path, patched, monkeypatch):
    s4 = _step4(tmp_path, grids=(32, 48))
    real = campaign.export_p07_sparse

    def corrupt(plan, kind):
        op = real(plan, kind)
        op.matrix.data[0] += 1.0
        return op

    monkeypatch.setattr(campaign, "export_p07_sparse", corrupt)
    with pytest.raises(SystemExit) as exc:
        campaign.main(_argv("run", s4, tmp_path, "--grids", "32", "48"))
    assert exc.value.code == 1
    out = tmp_path / "out"
    rec = json.loads((out / "N32" / "export_receipt.json").read_text())
    assert rec["pass"] is False and not rec["gates"]["dirichlet"]["pass"]
    assert not (out / "N48").exists()                                  # stopped after the failing grid
    assert json.loads((out / "last_exit.json").read_text())["status"] == "failed_gates"
    # a failed receipt is not skipped: with a working exporter the grid is recomputed and passes
    monkeypatch.setattr(campaign, "export_p07_sparse", real)
    summary = campaign.main(_argv("run", s4, tmp_path, "--grids", "32"))
    assert summary["grids"]["32"]["status"] == "complete" and summary["all_pass"]


def test_validate_detects_sha_mismatch_and_missing_receipt(tmp_path, patched):
    s4 = _step4(tmp_path, grids=(32, 48))
    campaign.main(_argv("run", s4, tmp_path, "--grids", "32"))
    out = tmp_path / "out"
    assert campaign.main(_argv("validate", s4, tmp_path, "--grids", "32"))["all_pass"]
    with pytest.raises(SystemExit):                                    # N48 was never run
        campaign.main(_argv("validate", s4, tmp_path, "--grids", "32", "48"))
    val = json.loads((out / "validation.json").read_text())
    assert not val["all_pass"] and "no export_receipt.json" in val["grids"]["48"]["errors"]
    path = out / "N32" / "p07_neumann.npz"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(SystemExit):
        campaign.main(_argv("validate", s4, tmp_path, "--grids", "32"))
    val = json.loads((out / "validation.json").read_text())
    assert not val["all_pass"] and any("sha256" in e for e in val["grids"]["32"]["errors"])


def test_identity_change_requires_new_output(tmp_path, patched):
    s4 = _step4(tmp_path)
    campaign.main(_argv("verify-inputs", s4, tmp_path, "--grids", "32"))
    (s4 / "campaign_manifest.json").write_text(json.dumps({"identity": "other"}))
    (s4 / "validation.json").write_text(json.dumps({"identity": "other", "operator_options": FINAL,
                                                   "grids": {"32": {"smoke_pass": True, "preflight_pass": True}}}))
    with pytest.raises(ValueError, match="identity changed"):
        campaign.main(_argv("verify-inputs", s4, tmp_path, "--grids", "32"))


def test_p07_only_artifact_lowering_exports_the_in_memory_action(tmp_path):
    """The real lowering path: rows saved as an artifact, read back with ``include=("p07",)`` (only the P07 and
    P07-stage Neumann chunks), exported, and compared with ``p07_action`` of the in-memory full plan."""
    from drbx.stencils import artifact as art
    from drbx.stencils.operator_plan import lower_perpendicular_plan_from_artifact, pack_owner_rows

    world = make_world(owners=OWNERS)
    full = lower_world(world)
    packed = pack_owner_rows(world.row_index, world.neumann_index, raw_ids=world.raw_ids, face_rows=world.face_rows,
                             p07_ids=world.census.p07_id[world.p07_rows])
    root = tmp_path / "artifact"
    art.save_row_artifact(root, world.n, identity={"synthetic": True}, cells=list(packed.cells),
                          faces=list(packed.faces), p07=list(packed.p07), neumann=list(packed.neumann))
    p07_only = lower_perpendicular_plan_from_artifact(
        root, world.n, grid=world.grid, census=world.census, geometry=world.geometry, raw_volume=world.raw_volume,
        owner_volume=world.owner_volume, identity={"synthetic": True}, include=("p07",), raw_ids=world.raw_ids,
        face_rows=world.face_rows, p07_rows=world.p07_rows)
    assert p07_only.cells is None and p07_only.faces is None
    rng = np.random.default_rng(3)
    u = rng.normal(size=(len(full.p07.owner_volume), 2))
    bc_full = BoundaryData(rng.normal(size=(len(full.dirichlet_points), 2)),
                           rng.normal(size=(len(full.dirichlet_points), 2, 2)),
                           rng.normal(size=(len(full.neumann_points), 2)))
    # the P07-only plan's point tables are a subset of the full plan's; map its rows onto the full tables
    def rows_of(sub, table):
        index = {tuple(map(float, p)): i for i, p in enumerate(np.asarray(table))}
        return np.asarray([index[tuple(map(float, p))] for p in np.asarray(sub)], dtype=np.int64)
    di = rows_of(p07_only.dirichlet_points, full.dirichlet_points)
    ni = rows_of(p07_only.neumann_points, full.neumann_points)
    bc_sub = BoundaryData(bc_full.dirichlet_value[di], bc_full.dirichlet_tangential[di], bc_full.neumann_normal[ni])
    for kind in KINDS:
        op = export_p07_sparse(p07_only, kind)
        ref = np.asarray(p07_action(full, u, bc_full, kind))
        np.testing.assert_allclose(apply_p07_sparse(op, u, bc_sub), ref, rtol=0, atol=1e-12 * np.abs(ref).max())
