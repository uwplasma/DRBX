"""Fast tests of the P08 step-5.3 campaign (``scripts/p08_step5_combined``): the pinned contract in the configuration, the
identity and ``verify-inputs`` (synthetic step-4 folder), the CLI, the reference stage (monkeypatched ``reference_rhs``;
shape / merge / resume / sha), the psi references, the JAX stage with a synthetic world (both arms fed the right phi, the
solver gates computed and enforced, checkpoint resume), the preflight verdict, the reduction math (relative L2,
regions, degenerate-reference rule, orders), the headline criterion, ``validate`` and ``analyze``.

Fully synthetic / ``tmp_path`` based: no real geometry, row artifact or oracle data (the real bounded check is
``tests/test_p08_step5_combined_campaign_real.py``, slow).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
sys.path.insert(0, str(REPO / "scripts"))

from drbx.native.fci_perpendicular_p07_sparse import P07SparseOperator            # noqa: E402
from drbx.native.fci_perpendicular_phi_solver import PHI_RTOL_DEFAULT            # noqa: E402

from p_shared import runner                                                       # noqa: E402
from p_shared import perpendicular_reference_rhs as prr                          # noqa: E402
from p08_step5_combined import analysis, campaign, checkpoints, jaxstage, reduction, references  # noqa: E402

FINAL = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius"}
FIELDS = references.FIELDS
STEP4_IDENTITY = "step4-identity"


# ---------------------------------------------------------------------------
# Configuration and catalogue
# ---------------------------------------------------------------------------
def test_config_pins_the_scientific_contract():
    cfg = campaign.config()
    assert cfg["schema"] == "drbx.p08-step5-combined-dirichlet-v1" == campaign.SCHEMA
    assert cfg["resolutions"] == [32, 48, 64]
    assert cfg["operator_options"] == FINAL == campaign.operator_options() == campaign.OPERATOR_OPTIONS
    assert cfg["params"] == {"rho_star": 0.05, "tau": 1.0,
                             "diffusion": {"density": 0.01, "Te": 0.01, "Ti": 0.01, "vorticity": 0.01}}
    assert tuple(cfg["variants"]) == ("main_phi_dirichlet", "heldout_phi_dirichlet", "dirichlet_rich",
                                      "control_constant_dirichlet")
    assert cfg["variants"]["control_constant_dirichlet"]["constant"] is True
    assert not any(v["constant"] for k, v in cfg["variants"].items() if k != "control_constant_dirichlet")
    # the phi solve: Dirichlet, eta-plane block Jacobi, float32 factors, MEASUREMENT tolerance 1e-11 (not the default)
    assert cfg["phi_solve_boundary"] == "dirichlet" and cfg["phi_solve_preconditioner"] == "block_jacobi_eta_plane"
    assert (cfg["phi_solve_factor_dtype"], cfg["phi_solve_restart"], cfg["phi_solve_max_restarts"]) == ("float32", 50, 40)
    assert cfg["phi_solve_rtol"] == 1e-11 and cfg["phi_solve_rtol_production_default"] == PHI_RTOL_DEFAULT == 1e-8
    assert cfg["consistency_tolerance"] == 1e-8
    # deferred items are recorded
    assert "main_phi_neumann" in cfg["deferred"]["neumann_phi_variants"]
    assert "step-6" in cfg["deferred"]["transverse_wave_controls"]
    assert cfg["order_criterion"]["informational"] is True and cfg["order_criterion"]["min_order"] == 1.8


@pytest.mark.parametrize("name,value", [("OPERATOR_OPTIONS", {**FINAL, "face_quadrature": "q3"}),
                                        ("PARAMS", {"rho_star": 0.1, "tau": 1.0, "diffusion": {}}),
                                        ("VARIANTS", ("main_phi_dirichlet",)),
                                        ("PHI_SOLVE", {**campaign.PHI_SOLVE, "phi_solve_rtol": 1e-8})])
def test_config_refuses_drift_from_the_pinned_literals(monkeypatch, name, value):
    monkeypatch.setattr(campaign, name, value)
    with pytest.raises(ValueError, match="differ from the pinned|differs from the pinned"):
        campaign.config()


def test_config_pins_the_variants_to_the_frozen_catalogue():
    cfg = campaign.config()
    names = references.check_catalogue(cfg)                      # real frozen P06N catalogue
    assert set(names) == {"main", "heldout", "constant"}
    assert cfg["variants"]["main_phi_dirichlet"]["field_set"] == cfg["variants"]["dirichlet_rich"]["field_set"] == "main"
    assert all(len(f) == 5 for f in names.values()) and names["main"] != names["heldout"]
    bad = json.loads(json.dumps(cfg))
    bad["variants"]["main_phi_dirichlet"]["kinds"][0] = "dirichlet"
    with pytest.raises(ValueError, match="differ from the pinned"):
        references.check_catalogue(bad)
    bad = json.loads(json.dumps(cfg))
    bad["variants"]["dirichlet_rich"]["field_set"] = "heldout"
    with pytest.raises(ValueError, match="disagree"):
        references.check_catalogue(bad)
    bad = json.loads(json.dumps(cfg))
    bad["variants"]["not_a_variant"] = bad["variants"]["main_phi_dirichlet"]
    with pytest.raises(ValueError, match="not in the frozen P06N catalogue"):
        references.check_catalogue(bad)
    bad = json.loads(json.dumps(cfg))
    bad["field_sets"]["main"]["representative"] = "heldout_phi_dirichlet"
    with pytest.raises(ValueError, match="representative"):
        references.check_catalogue(bad)


def test_source_hashes_cover_the_package_and_the_p_path_sources():
    hashes = campaign.source_hashes()
    assert len(hashes) == len(campaign.SOURCE_FILES) == len(set(campaign.SOURCE_FILES))
    for rel, digest in hashes.items():
        assert (REPO / rel).is_file(), rel
        assert len(digest) == 64
    for needed in ("scripts/p08_step5_combined/campaign.py", "scripts/p08_step5_combined/configuration.json",
                   "scripts/p08_step5_combined/jaxstage.py", "scripts/p08_step5_combined/references.py",
                   "src/drbx/native/fci_perpendicular_rhs.py", "src/drbx/native/fci_perpendicular_p05_operator.py",
                   "src/drbx/native/fci_perpendicular_p06_operator.py", "src/drbx/native/fci_perpendicular_p07_operator.py",
                   "src/drbx/native/fci_perpendicular_reconstruction_state.py", "src/drbx/stencils/operator_plan.py",
                   "src/drbx/native/fci_perpendicular_p07_sparse.py", "src/drbx/native/fci_perpendicular_p07_solve.py",
                   "src/drbx/native/fci_perpendicular_phi_solver.py",
                   "src/drbx/native/fci_perpendicular_plane_preconditioner.py",
                   "scripts/p_shared/perpendicular_reference_rhs.py", "scripts/p_shared/curvature_reference.py",
                   "scripts/p_shared/campaign_fields.py", "scripts/p_shared/replay_support.py"):
        assert needed in hashes


# ---------------------------------------------------------------------------
# verify-inputs / identity (synthetic step-4 folder and oracle tree)
# ---------------------------------------------------------------------------
def _fake_world(tmp_path, monkeypatch, *, grids=(32, 48, 64)):
    monkeypatch.setattr(campaign.step1, "_input_manifest", lambda: {"files": []})
    workspace = tmp_path / "workspace"
    (workspace / "work" / "p06n").mkdir(parents=True)
    step4 = tmp_path / "step4" / "campaign"
    step4.mkdir(parents=True)
    oracle_entries = {}
    for n in (32, 48, 64):
        path = workspace / "work" / "p06n" / f"N{n}.owner_values.npz"
        np.savez(path, values=np.arange(4.0) + n)
        raw = workspace / "work" / "p06n" / f"N{n}.raw.npz"
        np.savez(raw, x=np.arange(3.0))
        oracle_entries[str(n)] = {"files": [
            {"path": f"work/p06n/N{n}.owner_values.npz", "bytes": path.stat().st_size, "sha256": runner.sha256_file(path)},
            {"path": f"work/p06n/N{n}.raw.npz", "bytes": raw.stat().st_size, "sha256": "unchecked"}], "missing": []}
    (step4 / "oracle_manifest.json").write_text(json.dumps({"schema": "x", "workspace_root": "",
                                                           "campaigns": {"p06n": oracle_entries}}))
    (step4 / "campaign_manifest.json").write_text(json.dumps({"identity": STEP4_IDENTITY}))
    (step4 / "localized_sidecar.json").write_text(json.dumps({"sidecar": 1}))
    (step4 / "validation.json").write_text(json.dumps({
        "identity": STEP4_IDENTITY, "operator_options": FINAL,
        "grids": {str(n): {"smoke_pass": True, "preflight_pass": True} for n in (32, 48, 64)}}))
    for n in (32, 48, 64):
        gdir = step4 / "artifact" / f"N{n}"
        gdir.mkdir(parents=True)
        (gdir / "manifest.json").write_text(json.dumps({"n": n}))
        (gdir / "build_identity.json").write_text(json.dumps({"n": n, "curvature": "autodiff"}))
    return workspace, step4


def _verify(workspace, step4, output, grids=(32, 48, 64), oracle_root=None):
    return campaign.verify_inputs(step4=step4, input_root=workspace, oracle_root=oracle_root, output=output,
                                  grids=list(grids))


def test_verify_inputs_records_provenance_and_the_identity_is_stable(tmp_path, monkeypatch):
    workspace, step4 = _fake_world(tmp_path, monkeypatch)
    identity, record = _verify(workspace, step4, tmp_path / "out")
    assert _verify(workspace, step4, tmp_path / "out")[0] == identity
    saved = json.loads((tmp_path / "out" / "provenance" / "inputs.json").read_text())
    assert saved["identity"] == identity and saved["campaign"] == "p08_step5_combined"
    assert saved["operator_options"] == FINAL and saved["configuration"]["phi_solve_rtol"] == 1e-11
    assert saved["step4_identity"] == STEP4_IDENTITY and set(saved["grids"]) == {"32", "48", "64"}
    assert saved["jax"]["backend"] == "cpu" and saved["jax"]["x64"] is True
    assert set(saved["oracle_p06n_owner_values"]) == {"32", "48", "64"}
    assert checkpoints.artifact_sha(record, 32) == saved["grids"]["32"]["artifact_identity_sha256"]


def test_identity_depends_on_configuration_step4_identity_and_sidecar(tmp_path, monkeypatch):
    workspace, step4 = _fake_world(tmp_path, monkeypatch)
    one = _verify(workspace, step4, tmp_path / "one")[0]
    real_config = campaign.config
    monkeypatch.setattr(campaign, "config", lambda: {**real_config(), "consistency_tolerance": 1e-3})
    assert _verify(workspace, step4, tmp_path / "two")[0] != one
    with pytest.raises(ValueError, match="campaign identity changed"):
        _verify(workspace, step4, tmp_path / "one")
    monkeypatch.setattr(campaign, "config", real_config)
    (step4 / "campaign_manifest.json").write_text(json.dumps({"identity": "other"}))
    (step4 / "validation.json").write_text(json.dumps({
        "identity": "other", "operator_options": FINAL,
        "grids": {str(n): {"smoke_pass": True, "preflight_pass": True} for n in (32, 48, 64)}}))
    three = _verify(workspace, step4, tmp_path / "three")[0]
    assert three != one
    (step4 / "localized_sidecar.json").write_text(json.dumps({"sidecar": 2}))
    four = _verify(workspace, step4, tmp_path / "four")[0]
    assert four not in (one, three)
    monkeypatch.setattr(campaign, "SOURCE_FILES", campaign.SOURCE_FILES[:-1])
    assert _verify(workspace, step4, tmp_path / "five")[0] not in (one, three, four)


def test_verify_inputs_refuses_bad_step4_inputs(tmp_path, monkeypatch):
    workspace, step4 = _fake_world(tmp_path, monkeypatch)
    validation = json.loads((step4 / "validation.json").read_text())
    bad = json.loads(json.dumps(validation))
    bad["grids"]["48"]["smoke_pass"] = False
    (step4 / "validation.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="did not pass"):
        _verify(workspace, step4, tmp_path / "a", grids=(32, 48))
    _verify(workspace, step4, tmp_path / "b", grids=(32,))                  # the failing grid is not requested
    bad = {**validation, "operator_options": {**FINAL, "curvature": "fd"}}
    (step4 / "validation.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="differ from"):
        _verify(workspace, step4, tmp_path / "c")
    (step4 / "validation.json").write_text(json.dumps({**validation, "identity": "mismatch"}))
    with pytest.raises(ValueError, match="different campaign identities"):
        _verify(workspace, step4, tmp_path / "d")
    (step4 / "validation.json").write_text(json.dumps(validation))
    (step4 / "artifact" / "N64" / "build_identity.json").unlink()
    with pytest.raises(ValueError, match="missing step-4 artifact file"):
        _verify(workspace, step4, tmp_path / "e")
    (step4 / "artifact" / "N64" / "build_identity.json").write_text("{}")
    (step4 / "localized_sidecar.json").unlink()
    with pytest.raises(ValueError, match="missing input file"):
        _verify(workspace, step4, tmp_path / "f")
    (step4 / "localized_sidecar.json").write_text("{}")
    manifest = json.loads((step4 / "oracle_manifest.json").read_text())
    (step4 / "oracle_manifest.json").write_text(json.dumps({**manifest, "campaigns": {}}))
    with pytest.raises(ValueError, match="no p06n entry"):
        _verify(workspace, step4, tmp_path / "g")
    (step4 / "oracle_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not a directory"):
        _verify(tmp_path / "nowhere", step4, tmp_path / "h")


def test_verify_inputs_checks_oracle_files_and_oracle_root_and_backend(tmp_path, monkeypatch):
    workspace, step4 = _fake_world(tmp_path, monkeypatch)
    _verify(workspace, step4, tmp_path / "out")
    with pytest.raises(ValueError, match="oracle verification failed"):          # no oracle files under that root
        _verify(workspace, step4, tmp_path / "o2", oracle_root=tmp_path / "elsewhere")
    shutil.copytree(workspace / "work", tmp_path / "elsewhere" / "work")
    with pytest.raises(ValueError, match="oracle root changed"):                  # intact copy, but another root
        _verify(workspace, step4, tmp_path / "out", oracle_root=tmp_path / "elsewhere")
    np.savez(workspace / "work" / "p06n" / "N48.owner_values.npz", values=np.zeros(4))      # corrupted oracle file
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify(workspace, step4, tmp_path / "o3")
    _verify(workspace, step4, tmp_path / "o4", grids=(32,))                                # N48 not requested
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with pytest.raises(ValueError, match="CPU backend required"):
        _verify(workspace, step4, tmp_path / "o5", grids=(32,))


def test_cli_parsing():
    base = ["--step4-campaign", "/s", "--input-root", "/x", "--output", "/y"]
    a = campaign.parse(["run", *base])
    assert a.resolutions == [32, 48, 64] and a.workers == 4 and a.oracle_root is None and a.stage is None
    a = campaign.parse(["preflight", *base, "--workers", "96", "--oracle-root", "/o", "--memory-budget-gib", "400",
                        "--worker-memory-gib", "3.5", "--memory-reserve-gib", "8", "--max-tasks-per-worker", "50",
                        "--resolutions", "32"])
    assert a.workers == 96 and a.oracle_root == Path("/o") and a.resolutions == [32] and a.step4_campaign == Path("/s")
    a = campaign.parse(["run-stage", *base, "--stage", "jax", "--n", "48"])
    assert (a.stage, a.n) == ("jax", 48)
    for command in ("verify-inputs", "validate", "analyze"):
        assert campaign.parse([command, *base]).command == command
    for bad in (["run", "--resolutions", "16"], ["frobnicate"], ["run-stage", "--stage", "cells"],
                ["run", "--max-units", "2"]):
        with pytest.raises(SystemExit):
            campaign.parse([*bad, *base])
    with pytest.raises(SystemExit):
        campaign.parse(["run", "--input-root", "/x", "--output", "/y"])          # --step4-campaign is required


def test_main_arguments_are_checked_and_verify_inputs_locks_and_records(tmp_path, monkeypatch, capsys):
    workspace, step4 = _fake_world(tmp_path, monkeypatch)
    argv = ["--step4-campaign", str(step4), "--input-root", str(workspace)]
    with pytest.raises(ValueError, match="--stage and --n are required"):
        campaign.main(["run-stage", *argv, "--output", str(tmp_path / "o1")])
    with pytest.raises(ValueError, match="unique and ascending"):
        campaign.main(["verify-inputs", *argv, "--output", str(tmp_path / "o1"), "--resolutions", "48", "32"])
    with pytest.raises(ValueError, match="not among the verified"):
        campaign.main(["run-stage", *argv, "--output", str(tmp_path / "o1"), "--stage", "jax", "--n", "64",
                       "--resolutions", "32"])
    output = tmp_path / "out"
    campaign.main(["verify-inputs", *argv, "--output", str(output)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "verify-inputs" and payload["status"] == "complete"
    assert (output / ".campaign.lock").exists() and list((output / "invocations").glob("*_verify-inputs.json"))
    assert json.loads((output / "last_exit.json").read_text())["status"] == "complete"
    assert (output / "provenance" / "inputs.json").is_file()


# ---------------------------------------------------------------------------
# Reference stage
# ---------------------------------------------------------------------------
class _State:
    """A synthetic five-slot state: values, gradients and Hessians linear in the points."""

    def values_gradients(self, points):
        pts = np.asarray(points, dtype=float)
        v = np.stack([(k + 1.0) + pts[:, 0] * (k + 1.0) for k in range(5)])
        g = np.stack([np.tile(np.array([0.1, 0.2, 0.3]) * (k + 1.0), (len(pts), 1)) for k in range(5)])
        return v, g

    def values_gradients_hessians(self, points):
        v, g = self.values_gradients(points)
        q = len(points)
        h = np.stack([np.tile(np.arange(9.0).reshape(3, 3) * (k + 1.0), (q, 1, 1)) for k in range(5)])
        return v, g, h


def test_psi_gradient_is_phi_plus_tau_ti():
    state = _State()
    grads = references.psi_gradient_function(state, 0.7)(np.zeros((3, 3)))
    _v, g = state.values_gradients(np.zeros((3, 3)))
    assert grads.shape == (3, 1, 3)
    np.testing.assert_allclose(grads[:, 0, :], g[4] + 0.7 * g[2])


def test_psi_positive_operator_uses_the_positive_convention(monkeypatch):
    seen = {}

    def fake_diffusion(env, support, exact_gradients, *, positive_operator=False):
        seen["positive"] = positive_operator
        seen["grad"] = exact_gradients(np.zeros((2, 3)))
        return np.array([[1.0], [2.0]])

    monkeypatch.setattr(prr, "diffusion_reference", fake_diffusion)
    out = references.psi_positive_operator(None, None, _State(), 1.0)
    assert seen["positive"] is True and seen["grad"].shape == (2, 1, 3)
    np.testing.assert_array_equal(out, [1.0, 2.0])


def test_psi_midpoint_reference_formula_and_owner_projection(monkeypatch):
    from p_shared import curvature_reference

    points = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])
    tensor = np.tile(np.eye(3), (4, 1, 1)) * np.array([1.0, 2.0, 3.0, 4.0])[:, None, None]
    divergence = np.arange(12.0).reshape(4, 3) / 10.0
    calls = {}

    def fake_geometry(reference, pts, method="fd"):
        calls["method"] = method
        return tensor, divergence

    monkeypatch.setattr(curvature_reference, "perpendicular_geometry", fake_geometry)
    jac = np.array([-2.0, 2.0, 4.0, -4.0])
    support = SimpleNamespace(points=points, raw_volume=np.array([1.0, 1.0, 2.0, 2.0]),
                              raw_owner=np.array([0, 0, 1, 1]), owners=np.array([0, 1]),
                              owner_volume=np.array([2.0, 4.0]), raw_geometry=lambda env: {"J": jac})
    tau = 0.5
    got = references.psi_midpoint_reference(SimpleNamespace(ref=None), support, _State(), tau)
    assert calls["method"] == "autodiff"                      # the autodiff divergence, not the finite-difference one
    _v, g, h = _State().values_gradients_hessians(points)
    gp, hp = g[4] + tau * g[2], h[4] + tau * h[2]
    pointwise = (np.einsum("qj,qj->q", divergence, gp) + np.einsum("qij,qij->q", tensor, hp)) / np.abs(jac)
    expected = np.array([-(pointwise[0] + pointwise[1]) / 2.0, -(2 * pointwise[2] + 2 * pointwise[3]) / 4.0])
    np.testing.assert_allclose(got, expected, rtol=1e-14)


def _fake_reference_chunk(env, owners, *, states, params):
    owners = np.asarray(owners, dtype=np.int64)
    out = {"owners": owners}
    for i, fs in enumerate(states):
        for j, field in enumerate(FIELDS):
            for k, term in enumerate(references.REF_TERMS):
                out[references.ref_key(fs, field, term)] = owners * (1.0 + i + 0.1 * j + 0.01 * k) + 0.5
        out[references.psi_key(fs, "O")] = owners * 2.0 + i
        out[references.psi_key(fs, "R_mid")] = owners * 2.0 + i + 0.25
    return out


def test_reference_chunk_collects_every_field_set_and_term(monkeypatch):
    seen = {}

    class FakeParams:
        def __init__(self, **kwargs):
            seen["params"] = kwargs

    def fake_reference_rhs(env, owners, state, params, fields, *, support):
        seen.setdefault("states", []).append(state)
        n = len(owners)
        return {f: {t: np.full(n, 1.0 + i + 0.1 * j) for j, t in enumerate(references.REF_TERMS)}
                for i, f in enumerate(fields)}

    monkeypatch.setattr(prr, "owner_support", lambda env, owners: SimpleNamespace(owners=owners))
    monkeypatch.setattr(prr, "ReferenceParams", FakeParams)
    monkeypatch.setattr(prr, "reference_rhs", fake_reference_rhs)
    monkeypatch.setattr(references, "psi_positive_operator", lambda env, support, state, tau: np.full(3, 5.0))
    monkeypatch.setattr(references, "psi_midpoint_reference", lambda env, support, state, tau: np.full(3, 6.0))
    states = {"main": "S1", "heldout": "S2"}
    params = campaign.config()["params"]
    out = references.reference_chunk(None, [4, 5, 6], states=states, params=params)
    assert seen["params"]["rho_star"] == 0.05 and seen["params"]["tau"] == 1.0
    assert seen["params"]["diffusion"] == {f: 0.01 for f in FIELDS} and seen["states"] == ["S1", "S2"]
    assert set(out) == {"owners"} | {references.ref_key(fs, f, t) for fs in states for f in FIELDS
                                     for t in references.REF_TERMS} | {references.psi_key(fs, n) for fs in states
                                                                       for n in ("O", "R_mid")}
    assert all(v.shape == (3,) for v in out.values())
    np.testing.assert_array_equal(out["owners"], [4, 5, 6])


def test_reference_chunk_rejects_a_wrongly_shaped_array(monkeypatch):
    monkeypatch.setattr(prr, "owner_support", lambda env, owners: SimpleNamespace(owners=owners))
    monkeypatch.setattr(prr, "reference_rhs", lambda env, owners, state, params, fields, *, support:
                        {f: {t: np.zeros(2) for t in references.REF_TERMS} for f in fields})
    monkeypatch.setattr(references, "psi_positive_operator", lambda *a: np.zeros(3))
    monkeypatch.setattr(references, "psi_midpoint_reference", lambda *a: np.zeros(3))
    with pytest.raises(ValueError, match="expected"):
        references.reference_chunk(None, [0, 1, 2], states={"main": None}, params=campaign.config()["params"])


def _serial_run_stage(calls):
    """A stand-in for ``runner.run_stage`` honouring its resume semantics (the real one spawns processes)."""
    def run_stage(output, stage, units, identity, *, compute, initializer, initargs, workers, parts=("chunk",),
                  max_tasks_per_worker=None, max_units=None):
        todo = [u for u in units if not runner.valid_unit(output, u, identity, parts=parts)]
        if todo:
            initializer(*initargs)
        for u in todo:
            calls.append(u["start"])
            compute(u)
        return {"stage": stage, "executed_units": len(todo), "resumed_units": len(units) - len(todo),
                "total_units": len(units), "workers": workers}
    return run_stage


def _stage_world(monkeypatch, n_owners=10):
    volume = 1.0 + 0.1 * np.arange(n_owners)
    regions = {"physical_wall": np.arange(n_owners) >= n_owners - 3, "interior": np.arange(n_owners) < n_owners - 3}
    monkeypatch.setattr(references, "load_owner_context", lambda n, input_root: (volume, regions))

    def init(settings):
        references._WORKER.clear()
        references._WORKER.update(env=None, states={fs: None for fs in settings["cfg"]["field_sets"]},
                                  params=settings["cfg"]["params"], work=settings["work"],
                                  identity=settings["identity"])

    monkeypatch.setattr(references, "init_worker", init)
    monkeypatch.setattr(references, "reference_chunk", _fake_reference_chunk)
    return volume, regions


def test_references_stage_merges_chunks_writes_context_and_manifest(tmp_path, monkeypatch):
    cfg = {**campaign.config(), "owner_chunk_size": 4}
    volume, regions = _stage_world(monkeypatch)
    calls = []
    monkeypatch.setattr(runner, "run_stage", _serial_run_stage(calls))
    kwargs = dict(n=32, cfg=cfg, options=FINAL, input_root=tmp_path, sidecar_path=tmp_path / "s.json",
                  output=tmp_path / "out", identity="ID", workers=3)
    manifest = references.references_stage(**kwargs)
    assert calls == [0, 4, 8] and manifest["skipped"] is False and manifest["n_owners"] == 10 and manifest["units"] == 3
    merged = references.load_references(tmp_path / "out", 32, "ID")
    assert len(merged) == 3 * (4 * len(references.REF_TERMS) + 2) and "owners" not in merged
    key = references.ref_key("main", "Te", "curvature")
    np.testing.assert_allclose(merged[key], np.arange(10) * (1.0 + 0.1 * 1 + 0.01 * 1) + 0.5)
    assert all(v.shape == (10,) for v in merged.values())
    assert manifest["sha256"] == runner.sha256_file(references.references_dir(tmp_path / "out", 32) / references.MERGED)
    assert manifest["field_sets"] == {"main": "main_phi_dirichlet", "heldout": "heldout_phi_dirichlet",
                                      "constant": "control_constant_dirichlet"}
    vol, reg = references.load_context_file(tmp_path / "out", 32)
    np.testing.assert_array_equal(vol, volume)
    assert set(reg) == set(regions) and all(np.array_equal(reg[k], regions[k]) for k in regions)


def test_references_stage_skips_resumes_and_refuses_other_identities(tmp_path, monkeypatch):
    cfg = {**campaign.config(), "owner_chunk_size": 4}
    _stage_world(monkeypatch)
    calls = []
    monkeypatch.setattr(runner, "run_stage", _serial_run_stage(calls))
    out = tmp_path / "out"
    kwargs = dict(n=32, cfg=cfg, options=FINAL, input_root=tmp_path, sidecar_path=tmp_path / "s.json", output=out,
                  identity="ID", workers=1)
    references.references_stage(**kwargs)
    assert calls == [0, 4, 8]
    again = references.references_stage(**kwargs)
    assert again["skipped"] is True and calls == [0, 4, 8]                     # merged result valid: nothing recomputed
    # a lost merged file with intact chunks: merged again from the checkpoints, no chunk recomputed
    (references.references_dir(out, 32) / references.MERGED).unlink()
    assert references.references_stage(**kwargs)["skipped"] is False and calls == [0, 4, 8]
    # one lost chunk: only that chunk is recomputed
    (references.references_dir(out, 32) / references.MERGED).unlink()
    work = references.work_dir(out)
    unit = runner.chunk_units("references", 32, 10, 4)[1]
    runner.receipt_path(work, unit).unlink()
    references.references_stage(**kwargs)
    assert calls == [0, 4, 8, 4]
    # a corrupted merged file is not trusted
    path = references.references_dir(out, 32) / references.MERGED
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="no valid merged references"):
        references.load_references(out, 32, "ID")
    # another identity cannot read nor reuse the chunks
    with pytest.raises(ValueError, match="no valid merged references"):
        references.load_references(out, 32, "OTHER")
    with pytest.raises(ValueError, match="stale checkpoint"):
        references.references_stage(**{**kwargs, "identity": "OTHER"})


def test_compute_unit_writes_a_valid_checkpoint_and_merge_checks_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(references, "reference_chunk", _fake_reference_chunk)
    work = tmp_path / "work"
    references._WORKER.clear()
    references._WORKER.update(env=None, states={"main": None}, params={}, work=str(work), identity="ID")
    units = runner.chunk_units("references", 32, 6, 4)
    for unit in units:
        result = references.compute_unit(unit)
        assert result["seconds"] >= 0 and "peak_rss_gib" in result
        assert runner.valid_unit(work, unit, "ID")
    merged = references.merge_chunks(work, units, "ID", 6)
    np.testing.assert_array_equal(merged["owners"], np.arange(6))
    with pytest.raises(ValueError, match="cover the owners"):
        references.merge_chunks(work, units, "ID", 7)
    with pytest.raises(ValueError, match="missing or invalid"):
        references.merge_chunks(work, [*units, {"stage": "references", "n": 32, "start": 6, "stop": 7}], "ID", 7)


# ---------------------------------------------------------------------------
# JAX stage: a synthetic world (6 owners, 3 Dirichlet points, a real sparse P07 operator, monkeypatched RHS and solve)
# ---------------------------------------------------------------------------
N_OWN, Q_D, N_COL = 6, 3, 7
TAU = 1.0


class _Exact:
    def values_gradients(self, points):
        pts = np.asarray(points, dtype=float)
        v = np.stack([(k + 1.0) + 0.5 * pts[:, 0] + 0.1 * pts[:, 1] * k for k in range(5)])
        g = np.stack([np.tile(np.array([0.3, 0.1 * (k + 1), 0.2 - 0.05 * k]), (len(pts), 1)) for k in range(5)])
        return v, g


def _operator():
    rng = np.random.default_rng(3)
    matrix = sp.diags([-1.0 * np.ones(N_OWN - 1), 2.5 * np.ones(N_OWN), -1.0 * np.ones(N_OWN - 1)], [-1, 0, 1],
                      format="csr")
    dv = sp.csr_matrix(rng.normal(size=(N_OWN, Q_D)) * (rng.random((N_OWN, Q_D)) < 0.5))
    dt = sp.csr_matrix(rng.normal(size=(N_OWN, 2 * Q_D)) * (rng.random((N_OWN, 2 * Q_D)) < 0.5))
    return P07SparseOperator(kind="dirichlet", matrix=matrix, dirichlet_value=dv, dirichlet_tangential=dt,
                             neumann_normal=sp.csr_matrix((N_OWN, 0)), owner_volume=np.linspace(1.0, 2.0, N_OWN),
                             dirichlet_points=np.zeros((Q_D, 3)), neumann_points=np.zeros((0, 3)))


def _prep(tmp_path=None, variant="main_phi_dirichlet", solve_error=0.0, converged=True):
    cfg = campaign.config()
    rng = np.random.default_rng(5)
    values = 1.0 + rng.random((N_OWN, N_COL))
    columns = np.array([3, 1, 2, 0, 4])
    kinds = tuple(cfg["variants"][variant]["kinds"])
    adapter = SimpleNamespace(owner_values=values, reconstructions={variant: SimpleNamespace(columns=columns,
                                                                                          field_kinds=kinds)})
    op = _operator()
    plan = SimpleNamespace(dirichlet_points=np.zeros((Q_D, 3)), neumann_points=np.zeros((0, 3)))
    solver = SimpleNamespace(op=op, setup_seconds={"export": 0.0})
    solves = []

    def fake_solve(solver_, rhs, bc=None, *, x0=None):
        from drbx.native.fci_perpendicular_phi_solver import phi_boundary_term
        rhs_eff = np.asarray(rhs) - phi_boundary_term(solver_, bc)
        x = spla.splu(sp.csc_matrix(op.matrix)).solve(rhs_eff)
        solves.append({"rhs": np.asarray(rhs).copy(), "x0": x0})
        return x * (1.0 + solve_error), {"iterations": 7, "converged": converged, "residual_norm": 1e-14,
                                         "relative_residual": 1e-13, "rhs_norm": 1.0, "seconds": 0.01}

    prep = SimpleNamespace(
        n=32, env=SimpleNamespace(normal_coefficients=lambda q: np.ones((len(q), 3))), plan=plan, solver=solver,
        adapter=adapter, owner_volume=op.owner_volume, exact_state=lambda v: _Exact(), boundary_batch=8192,
        artifact_identity_sha256="art", plan_summary={"cells": 1}, setup_seconds={})
    prep.solves, prep.fake_solve, prep.columns, prep.values = solves, fake_solve, columns, values
    return cfg, prep


class _Recorder:
    """A stand-in for ``perpendicular_rhs`` that is linear in phi and records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, plan, state, phi, bc, field_kinds, params, **kwargs):
        self.calls.append({"phi": np.asarray(phi).copy(), "state": {k: np.asarray(v) for k, v in state.items()},
                           "bc": bc, "kinds": dict(field_kinds), "params": params})
        phi = np.asarray(phi)
        terms, total, detail = {}, {}, {}
        for i, f in enumerate(FIELDS):
            s = np.asarray(state[f])
            terms[f] = {"poisson_bracket": (i + 1.0) * phi, "curvature": 2.0 * phi + s,
                        "perpendicular_diffusion": -0.5 * phi}
            total[f] = sum(terms[f].values())
            detail[f] = {"curvature_material": 1.5 * phi, "curvature_remainder": 0.5 * phi + s}
        return SimpleNamespace(terms=terms, total=total, detail=detail)


def _run(monkeypatch, *, omega_rhs="exact", **prep_kwargs):
    cfg, prep = _prep(**prep_kwargs)
    recorder = _Recorder()
    monkeypatch.setattr(jaxstage, "perpendicular_rhs", recorder)
    monkeypatch.setattr(jaxstage, "solve_phi", prep.fake_solve)
    variant = prep_kwargs.get("variant", "main_phi_dirichlet")
    rhs = None
    if omega_rhs == "exact":                    # O(psi) equal to the discrete operator on psi_bar (N - O = 0)
        bc = jaxstage.psi_boundary_data(jaxstage.variant_boundary_data(prep, variant), TAU)
        psi_bar = prep.values[:, prep.columns[4]] + TAU * prep.values[:, prep.columns[2]]
        from drbx.native.fci_perpendicular_phi_solver import phi_boundary_term
        rhs = prep.solver.op.matrix @ psi_bar + phi_boundary_term(prep.solver, bc)
    elif omega_rhs is not None:
        rhs = omega_rhs(prep)
    arrays, info = jaxstage.run_variant(prep, variant, cfg, omega_rhs=rhs, log=lambda m: None)
    return cfg, prep, recorder, arrays, info, rhs


def test_boundary_data_of_the_five_columns_and_the_psi_trace():
    cfg, prep = _prep()
    bc = jaxstage.variant_boundary_data(prep, "main_phi_dirichlet")
    v, g = _Exact().values_gradients(np.zeros((Q_D, 3)))
    assert np.asarray(bc.dirichlet_value).shape == (Q_D, 5) and np.asarray(bc.dirichlet_tangential).shape == (Q_D, 2, 5)
    np.testing.assert_allclose(bc.dirichlet_value, v.T)
    np.testing.assert_allclose(bc.dirichlet_tangential, np.moveaxis(g, 0, -1)[:, 1:, :])
    assert bc.neumann_normal is None                                  # the synthetic plan has no Neumann points
    psi = jaxstage.psi_boundary_data(bc, 0.7)
    np.testing.assert_allclose(psi.dirichlet_value[:, 0], v[4] + 0.7 * v[2])
    np.testing.assert_allclose(psi.dirichlet_tangential[:, :, 0], (g[4] + 0.7 * g[2])[:, 1:])
    assert psi.neumann_normal is None and np.asarray(psi.dirichlet_value).shape == (Q_D, 1)


def test_both_arms_are_fed_the_right_phi_and_the_gates_pass(monkeypatch):
    cfg, prep, recorder, arrays, info, rhs = _run(monkeypatch)
    phi_bar = prep.values[:, prep.columns[4]]
    ti_bar = prep.values[:, prep.columns[2]]
    assert len(recorder.calls) == 2
    presc, solved = recorder.calls
    np.testing.assert_array_equal(presc["phi"], phi_bar)                  # prescribed: the frozen owner average
    # solved: phi_h = psi_h - tau * Ti_bar, with psi_h the solve of the (exact) psi problem = psi_bar here
    np.testing.assert_allclose(solved["phi"], phi_bar, atol=1e-13)
    np.testing.assert_allclose(arrays["psi_h"], phi_bar + TAU * ti_bar, atol=1e-13)
    np.testing.assert_allclose(arrays["phi_h"], solved["phi"], atol=0)
    for call in (presc, solved):
        for name, c in zip(FIELDS, prep.columns[:4]):
            np.testing.assert_array_equal(call["state"][name], prep.values[:, c])
        assert call["kinds"] == dict(zip((*FIELDS, "phi"), cfg["variants"]["main_phi_dirichlet"]["kinds"]))
        assert float(call["params"].rho_star) == 0.05 and float(call["params"].tau) == 1.0
        assert {f: float(d) for f, d in call["params"].diffusion.items()} == {f: 0.01 for f in FIELDS}
        assert np.asarray(call["bc"].dirichlet_value).shape == (Q_D, 5)
    # one cold consistency solve and one cold psi solve
    assert len(prep.solves) == 2 and all(s["x0"] is None for s in prep.solves)
    np.testing.assert_allclose(prep.solves[1]["rhs"], rhs)
    gates = info["gates"]
    assert gates == {"consistency": True, "converged": True, "finite": True} and info["gates_pass"] is True
    assert info["consistency"]["relative_error_M"] < 1e-12 and info["rhs_source"] == "reference_O"
    assert info["n_minus_o_psi"]["relative"] < 1e-12
    assert info["solved_solve"]["iterations"] == 7 and info["consistency"]["solve"]["iterations"] == 7
    assert info["kinds"] == list(cfg["variants"]["main_phi_dirichlet"]["kinds"])
    assert info["columns"] == [int(c) for c in prep.columns]
    # every array of both arms is stored per field and term
    for arm in ("presc", "solved"):
        for f in FIELDS:
            for term in references.REF_TERMS:
                assert checkpoints.arm_key(arm, f, term) in arrays
    np.testing.assert_allclose(arrays[checkpoints.arm_key("presc", "Te", "poisson_bracket")], 2.0 * phi_bar)
    np.testing.assert_allclose(arrays[checkpoints.arm_key("presc", "Ti", "total")],
                               3.0 * phi_bar + 2.0 * phi_bar + prep.values[:, prep.columns[2]] - 0.5 * phi_bar)


def test_solved_arm_follows_the_psi_solve_and_n_minus_o_reports_the_rhs_mismatch(monkeypatch):
    def perturbed(prep):
        bc = jaxstage.psi_boundary_data(jaxstage.variant_boundary_data(prep, "main_phi_dirichlet"), TAU)
        psi_bar = prep.values[:, prep.columns[4]] + TAU * prep.values[:, prep.columns[2]]
        from drbx.native.fci_perpendicular_phi_solver import phi_boundary_term
        n_psi = prep.solver.op.matrix @ psi_bar + phi_boundary_term(prep.solver, bc)
        prep.n_psi = n_psi
        return n_psi * 1.01                                             # O(psi) 1% above the discrete operator

    cfg, prep, recorder, arrays, info, rhs = _run(monkeypatch, omega_rhs=perturbed)
    presc, solved = recorder.calls
    assert info["n_minus_o_psi"]["relative"] == pytest.approx(0.01 / 1.01, rel=1e-9)
    assert info["gates"]["consistency"] is True                        # the consistency gate does not see O
    delta = spla.splu(sp.csc_matrix(prep.solver.op.matrix)).solve(0.01 * prep.n_psi)
    np.testing.assert_allclose(solved["phi"] - presc["phi"], delta, rtol=1e-9)    # phi_h - phi_bar = A^-1 (O - N)
    np.testing.assert_allclose(arrays["phi_h"] - arrays["phi_bar"], delta, rtol=1e-9)
    assert not np.allclose(arrays[checkpoints.arm_key("solved", "Te", "total")],
                           arrays[checkpoints.arm_key("presc", "Te", "total")])


def test_a_failing_consistency_gate_or_an_unconverged_solve_fails_the_variant(monkeypatch):
    info = _run(monkeypatch, solve_error=1e-6)[4]
    assert info["gates"]["consistency"] is False and info["gates_pass"] is False
    assert info["consistency"]["relative_error_M"] == pytest.approx(1e-6, rel=1e-6)
    assert info["gates"]["converged"] is True
    info = _run(monkeypatch, converged=False)[4]
    assert info["gates"]["converged"] is False and info["gates"]["consistency"] is True and info["gates_pass"] is False
    info = _run(monkeypatch, solve_error=5e-9)[4]                  # below the 1e-8 threshold: still passes
    assert info["gates"]["consistency"] is True


def test_preflight_style_run_without_the_reference_uses_the_discrete_operator(monkeypatch):
    cfg, prep, recorder, arrays, info, rhs = _run(monkeypatch, omega_rhs=None)
    presc, solved = recorder.calls
    assert info["rhs_source"] == "discrete_operator" and info["n_minus_o_psi"]["relative"] < 1e-12
    np.testing.assert_allclose(solved["phi"], presc["phi"], atol=1e-13)
    np.testing.assert_allclose(arrays[checkpoints.arm_key("solved", "Te", "total")],
                               arrays[checkpoints.arm_key("presc", "Te", "total")], atol=1e-12)


def test_run_variant_refuses_other_kinds_wrong_rhs_shape_and_flags_nonfinite(monkeypatch):
    cfg, prep = _prep()
    monkeypatch.setattr(jaxstage, "perpendicular_rhs", _Recorder())
    monkeypatch.setattr(jaxstage, "solve_phi", prep.fake_solve)
    bad = json.loads(json.dumps(cfg))
    bad["variants"]["main_phi_dirichlet"]["kinds"][4] = "neumann"
    with pytest.raises(ValueError, match="differ from the pinned"):
        jaxstage.run_variant(prep, "main_phi_dirichlet", bad, omega_rhs=None, log=lambda m: None)
    with pytest.raises(ValueError, match="right-hand side has shape"):
        jaxstage.run_variant(prep, "main_phi_dirichlet", cfg, omega_rhs=np.zeros(N_OWN + 1), log=lambda m: None)
    prep.adapter.owner_values[0, prep.columns[0]] = np.nan                    # a NaN in the density column
    _a, info = jaxstage.run_variant(prep, "main_phi_dirichlet", cfg, omega_rhs=None, log=lambda m: None)
    assert info["gates"]["finite"] is False and info["gates_pass"] is False


def _stage_inputs():
    return {"grids": {"32": {"artifact_identity_sha256": "art32"}}}


def _stub_jax_stage(monkeypatch, tmp_path, *, solve_error=0.0, prepare_calls=None):
    """``jax_stage`` with a synthetic world: references / context on disk, ``prepare`` replaced by the synthetic prep."""
    cfg = campaign.config()
    out = tmp_path / "out"
    volume = np.linspace(1.0, 2.0, N_OWN)
    references.save_context(out, 32, volume, {"physical_wall": np.arange(N_OWN) >= 4})
    refs = {references.psi_key(fs, "O"): np.linspace(1.0, 2.0, N_OWN) for fs in cfg["field_sets"]}
    monkeypatch.setattr(references, "load_references", lambda output, n, identity: refs)
    recorder = _Recorder()
    monkeypatch.setattr(jaxstage, "perpendicular_rhs", recorder)

    def prepare(**kwargs):
        if prepare_calls is not None:
            prepare_calls.append(1)
        _cfg, prep = _prep(solve_error=solve_error)
        # variants of the other catalogue entries read their own reconstruction
        for variant, spec in cfg["variants"].items():
            prep.adapter.reconstructions[variant] = SimpleNamespace(columns=prep.columns,
                                                                    field_kinds=tuple(spec["kinds"]))
        monkeypatch.setattr(jaxstage, "solve_phi", prep.fake_solve)
        prep.owner_volume = volume
        return prep

    monkeypatch.setattr(jaxstage, "prepare", prepare)
    kwargs = dict(n=32, cfg=cfg, options=FINAL, step4=tmp_path / "s4", input_root=tmp_path, output=out, identity="ID",
                  inputs=_stage_inputs(), paths={}, log=lambda m: None)
    return cfg, out, recorder, kwargs


def test_jax_stage_checkpoints_every_variant_and_resumes(tmp_path, monkeypatch):
    prepare_calls = []
    cfg, out, recorder, kwargs = _stub_jax_stage(monkeypatch, tmp_path, prepare_calls=prepare_calls)
    summary = jaxstage.jax_stage(**kwargs)
    assert summary["computed"] == list(cfg["variants"]) and summary["resumed"] == [] and len(prepare_calls) == 1
    assert len(recorder.calls) == 2 * len(cfg["variants"])
    jid = checkpoints.jax_identity("ID", 32, "art32")
    for variant in cfg["variants"]:
        arrays, info = checkpoints.load_variant(out, 32, variant, jid)
        assert info["variant"] == variant and info["gates_pass"] is True and "phi_h" in arrays
        assert info["constant"] is cfg["variants"][variant]["constant"]
    assert json.loads((out / "N32" / "jax_stage.json").read_text())["computed"] == list(cfg["variants"])
    # everything valid: nothing recomputed, no environment / plan built
    again = jaxstage.jax_stage(**kwargs)
    assert again["computed"] == [] and again["resumed"] == list(cfg["variants"]) and len(prepare_calls) == 1
    assert len(recorder.calls) == 2 * len(cfg["variants"])
    # a lost variant checkpoint: only that variant is recomputed
    runner.receipt_path(references.work_dir(out), checkpoints.variant_unit(32, "dirichlet_rich")).unlink()
    third = jaxstage.jax_stage(**kwargs)
    assert third["computed"] == ["dirichlet_rich"] and len(prepare_calls) == 2
    assert len(recorder.calls) == 2 * len(cfg["variants"]) + 2
    # another identity cannot read the checkpoints
    with pytest.raises(ValueError, match="stale checkpoint"):
        checkpoints.load_variant(out, 32, "dirichlet_rich", checkpoints.jax_identity("OTHER", 32, "art32"))
    with pytest.raises(ValueError, match="no valid JAX checkpoint"):
        checkpoints.load_variant(out, 48, "dirichlet_rich", jid)


def test_jax_stage_records_a_failing_gate_and_the_reduction_fails_the_grid(tmp_path, monkeypatch):
    cfg, out, recorder, kwargs = _stub_jax_stage(monkeypatch, tmp_path, solve_error=1e-5)
    jaxstage.jax_stage(**kwargs)
    jid = checkpoints.jax_identity("ID", 32, "art32")
    _arrays, info = checkpoints.load_variant(out, 32, "main_phi_dirichlet", jid)
    assert info["gates_pass"] is False and info["gates"]["consistency"] is False      # checkpointed, not raised
    refs_ = _synthetic_references(cfg, N_OWN)
    monkeypatch.setattr(references, "load_references", lambda output, n, identity: refs_)
    summary = reduction.reduce_grid(n=32, cfg=cfg, output=out, identity="ID", inputs=_stage_inputs())
    assert summary["gates"]["solver_gates_pass"] is False and summary["gates"]["grid_pass"] is False
    assert not any(summary["gates"]["solver_gates"].values())


def test_jax_stage_refuses_owner_volumes_that_differ_from_the_references(tmp_path, monkeypatch):
    cfg, out, recorder, kwargs = _stub_jax_stage(monkeypatch, tmp_path)
    references.save_context(out, 32, np.linspace(1.0, 3.0, N_OWN), {})
    with pytest.raises(ValueError, match="owner volumes"):
        jaxstage.jax_stage(**kwargs)


# ---------------------------------------------------------------------------
# Reduction math
# ---------------------------------------------------------------------------
def test_term_metrics_relative_l2_max_and_regions():
    volume = np.array([1.0, 3.0, 1.0, 3.0])
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    diff = 0.01 * ref
    m = reduction.term_metrics(diff, ref, volume)
    l2_ref = np.sqrt(np.sum(volume * ref ** 2) / volume.sum())
    assert m["ref_l2"] == pytest.approx(l2_ref) and m["l2"] == pytest.approx(0.01 * l2_ref)
    assert m["rel_l2"] == pytest.approx(0.01) and m["rel_max"] == pytest.approx(0.01)
    assert m["max_abs"] == pytest.approx(0.04) and m["ref_max"] == 4.0 and m["count"] == 4
    assert m["degenerate"] is False and m["finite"] is True
    mask = np.array([True, True, False, False])
    r = reduction.term_metrics(diff, ref, volume, mask)
    l2_r = np.sqrt((1 * 1 + 3 * 4) / 4.0)
    assert r["ref_l2"] == pytest.approx(l2_r) and r["rel_l2"] == pytest.approx(0.01) and r["count"] == 2
    assert reduction.term_metrics(diff, ref, volume, np.zeros(4, bool)) is None           # empty region
    nan = reduction.term_metrics(np.array([np.nan, 0, 0, 0.0]), ref, volume)
    assert nan["finite"] is False and not np.isfinite(nan["l2"])


def test_degenerate_reference_uses_the_largest_term_as_scale():
    volume = np.ones(4)
    ref = np.array([1e-15, -1e-15, 1e-15, 1e-15])
    diff = np.array([1e-4, 0.0, 0.0, 0.0])
    m = reduction.term_metrics(diff, ref, volume, fallback_l2=2.0, fallback_max=5.0, degenerate_fraction=1e-8)
    assert m["degenerate"] is True
    assert m["rel_l2"] == pytest.approx(0.5e-4 / 2.0) and m["rel_max"] == pytest.approx(1e-4 / 5.0)
    plain = reduction.term_metrics(diff, ref, volume, fallback_l2=1e-9, fallback_max=5.0, degenerate_fraction=1e-8)
    assert plain["degenerate"] is False and plain["rel_l2"] > 1e3               # relative to its own tiny scale
    zero = reduction.term_metrics(np.full(4, 1e-16), np.zeros(4), volume, fallback_l2=0.0, fallback_max=0.0)
    assert zero["degenerate"] is False and np.isnan(zero["rel_l2"]) and zero["l2"] > 0   # the constant control


def test_observed_order_and_orders_across_grids():
    assert analysis.observed_order(1e-2, 1e-2 * (32 / 48) ** 2, 32, 48) == pytest.approx(2.0)
    assert analysis.observed_order(None, 1.0, 32, 48) is None and analysis.observed_order(0.0, 1.0, 32, 48) is None
    assert analysis.observed_order(1.0, float("nan"), 32, 48) is None
    rows = {n: {"v|presc_vs_ref|Te|total|global": {"rel_l2": 1e-2 * (32 / n) ** 2, "l2": 3e-2 * (32 / n) ** 3,
                                                   "max_abs": 1.0, "degenerate": False, "finite": True}}
            for n in (32, 48, 64)}
    o = analysis.orders(rows)["v|presc_vs_ref|Te|total|global"]
    assert o["n"] == [32, 48, 64]
    assert o["order_rel_l2"] == pytest.approx([2.0, 2.0]) and o["order_l2"] == pytest.approx([3.0, 3.0])
    assert o["order_max_abs"] == pytest.approx([0.0, 0.0])


def _synthetic_references(cfg, n_owners):
    refs = {}
    k = np.arange(n_owners)
    for i, fs in enumerate(cfg["field_sets"]):
        for j, field in enumerate(FIELDS):
            if fs == "constant":
                for term in references.REF_TERMS:
                    refs[references.ref_key(fs, field, term)] = np.zeros(n_owners)
                continue
            prod = {t: (1.0 + j + 0.5 * m) * np.sin(0.37 * k * (1 + i) + j + 1.0) + 0.2
                    for m, t in enumerate(reduction.PRODUCTION_TERMS)}
            if field == "Te" and fs == "heldout":
                prod["poisson_bracket"] = np.full(n_owners, 1e-16)             # a degenerate reference
            for t, a in prod.items():
                refs[references.ref_key(fs, field, t)] = a
            refs[references.ref_key(fs, field, "total")] = sum(prod.values())
            refs[references.ref_key(fs, field, "curvature_material")] = 0.6 * prod["curvature"]
            refs[references.ref_key(fs, field, "curvature_remainder")] = 0.4 * prod["curvature"]
        refs[references.psi_key(fs, "O")] = (2.0 + np.cos(0.21 * k * (1 + i))) if fs != "constant" else np.zeros(n_owners)
        refs[references.psi_key(fs, "R_mid")] = refs[references.psi_key(fs, "O")] * 1.002
    return refs


def _write_grid(output, cfg, n, *, err_presc, err_solved, err_psi=1e-3, gates_ok=True, n_owners=40,
                inputs=None, identity="ID"):
    """A complete synthetic grid: context, merged references, per-variant checkpoints with errors proportional to the
    reference (so every relative L2 equals the given error)."""
    inputs = inputs or {"grids": {str(g): {"artifact_identity_sha256": f"art{g}"} for g in (32, 48, 64)}}
    volume = 1.0 + 0.1 * np.arange(n_owners)
    regions = {"physical_wall": np.arange(n_owners) >= n_owners - 8, "interior": np.arange(n_owners) < n_owners - 8}
    references.save_context(output, n, volume, regions)
    refs = _synthetic_references(cfg, n_owners)
    folder = references.references_dir(output, n)
    runner.save_npz(folder / references.MERGED, **refs)
    runner.write_json(folder / references.MANIFEST, {"identity": identity, "sha256": runner.sha256_file(folder / references.MERGED),
                                                       "n_owners": n_owners})
    jid = checkpoints.jax_identity(identity, n, checkpoints.artifact_sha(inputs, n))
    rng = np.random.default_rng(n)
    for variant, spec in cfg["variants"].items():
        fs = spec["field_set"]
        arrays = {}
        for field in FIELDS:
            for term in references.REF_TERMS:
                ref = refs[references.ref_key(fs, field, term)]
                noise = 1e-15 * rng.normal(size=n_owners) if spec["constant"] else 0.0
                arrays[checkpoints.arm_key("presc", field, term)] = ref * (1.0 + err_presc) + noise
                arrays[checkpoints.arm_key("solved", field, term)] = ref * (1.0 + err_solved) + noise
        phi_bar = 0.3 + 0.05 * np.sin(0.2 * np.arange(n_owners))
        arrays.update(phi_bar=phi_bar, ti_bar=np.full(n_owners, 1.0), psi_bar=phi_bar + 1.0,
                      phi_h=phi_bar * (1.0 + err_solved), psi_h=phi_bar * (1.0 + err_solved) + 1.0,
                      psi_N=refs[references.psi_key(fs, "O")] * (1.0 + err_psi))
        info = {"variant": variant, "field_set": fs, "constant": spec["constant"],
                "consistency": {"relative_error_M": 1e-12 if gates_ok else 1e-3, "tolerance": 1e-8, "pass": gates_ok,
                                "solve": {"iterations": 30, "converged": True}},
                "n_minus_o_psi": {"relative": err_psi, "l2": 1.0, "rhs_source": "reference_O"},
                "solved_solve": {"iterations": 55, "converged": True, "seconds": 2.5, "relative_residual": 1e-12,
                                 "residual_norm": 1e-12, "rhs_norm": 1.0},
                "gates": {"consistency": gates_ok, "converged": True, "finite": True}, "gates_pass": gates_ok}
        runner.write_unit(references.work_dir(output), checkpoints.variant_unit(n, variant), jid,
                          chunks={"chunk": arrays}, started=0.0, extra={"info": info})
    return inputs


def test_reduce_variant_errors_regions_and_the_degenerate_rule(tmp_path):
    cfg = campaign.config()
    n_owners = 40
    inputs = _write_grid(tmp_path, cfg, 32, err_presc=2e-3, err_solved=5e-3, n_owners=n_owners)
    summary = reduction.reduce_grid(n=32, cfg=cfg, output=tmp_path, identity="ID", inputs=inputs)
    results = reduction.load_results(tmp_path / "N32" / "results.npz")
    assert summary["regions"] == ["global", "interior", "physical_wall"] and summary["n_owners"] == n_owners
    assert summary["region_owner_counts"] == {"physical_wall": 8, "interior": 32}
    assert summary["rows"] == len(results) == 4 * 4 * 6 * 3 * 3
    k = reduction.row_key
    for region in ("global", "interior", "physical_wall"):
        assert results[k("main_phi_dirichlet", "presc_vs_ref", "density", "total", region)]["rel_l2"] == pytest.approx(2e-3)
        assert results[k("main_phi_dirichlet", "solved_vs_ref", "Ti", "curvature", region)]["rel_l2"] == pytest.approx(5e-3)
        assert results[k("main_phi_dirichlet", "solved_vs_presc", "Te", "perpendicular_diffusion", region)]["rel_l2"] == \
            pytest.approx(3e-3)
        assert results[k("main_phi_dirichlet", "solved_vs_ref", "density", "curvature_material", region)]["rel_l2"] == \
            pytest.approx(5e-3)
    assert results[k("main_phi_dirichlet", "presc_vs_ref", "density", "total", "global")]["max_abs"] > 0
    # the degenerate bracket of heldout Te is judged against the largest production term of that field
    degenerate = results[k("heldout_phi_dirichlet", "presc_vs_ref", "Te", "poisson_bracket", "global")]
    assert degenerate["degenerate"] is True and degenerate["rel_l2"] < 1e-12 + 2e-3 * 1e-10
    assert results[k("main_phi_dirichlet", "presc_vs_ref", "Te", "poisson_bracket", "global")]["degenerate"] is False
    # the constant control has an identically zero reference: relative errors are undefined, absolute ones are noise
    control = results[k("control_constant_dirichlet", "solved_vs_ref", "density", "total", "global")]
    assert np.isnan(control["rel_l2"]) and control["ref_l2"] == 0.0 and control["l2"] < 1e-13
    # phi error and the psi diffusion triple
    main = summary["variants"]["main_phi_dirichlet"]
    assert main["phi_error"]["global"]["rel_l2"] == pytest.approx(5e-3) and set(main["phi_error"]) == \
        {"global", "interior", "physical_wall"}
    psi = main["psi_diffusion"]
    assert psi["O_minus_R"]["global"]["rel_l2"] == pytest.approx(0.002 / 1.002, rel=1e-6)
    assert psi["N_minus_O"]["global"]["rel_l2"] == pytest.approx(1e-3)
    assert psi["N_minus_R"]["global"]["l2"] > 0
    assert summary["gates"] == {"solver_gates_pass": True, "solver_gates": {v: True for v in cfg["variants"]},
                                "all_finite": True, "grid_pass": True}
    assert summary["variants"]["control_constant_dirichlet"]["constant"] is True
    assert summary["variants"]["main_phi_dirichlet"]["global_rows"]["presc_vs_ref"]["density"]["total"]["rel_l2"] == \
        pytest.approx(2e-3)
    assert summary["variants"]["control_constant_dirichlet"]["global_rows"]["solved_vs_ref"]["density"]["total"][
        "rel_l2"] is None                                                       # NaN is stored as null in the JSON
    reloaded, results2 = reduction.load_grid(tmp_path, 32, "ID")
    assert reloaded["results_sha256"] == summary["results_sha256"] and len(results2) == len(results)


def test_reduction_refuses_another_identity_and_a_changed_results_file(tmp_path):
    cfg = campaign.config()
    inputs = _write_grid(tmp_path, cfg, 32, err_presc=1e-3, err_solved=2e-3)
    reduction.reduce_grid(n=32, cfg=cfg, output=tmp_path, identity="ID", inputs=inputs)
    with pytest.raises(ValueError, match="another campaign identity"):
        reduction.load_grid(tmp_path, 32, "OTHER")
    path = tmp_path / "N32" / "results.npz"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="does not match its summary"):
        reduction.load_grid(tmp_path, 32, "ID")
    with pytest.raises(ValueError, match="no reduction for N48"):
        reduction.load_grid(tmp_path, 48, "ID")
    other = tmp_path / "other"
    with pytest.raises(ValueError, match="no valid merged references"):
        reduction.reduce_grid(n=32, cfg=cfg, output=other, identity="ID", inputs=inputs)


def _three_grids(tmp_path, *, exponent=2.0, gates_ok=True, solved_exponent=None):
    cfg = campaign.config()
    inputs = None
    for n in (32, 48, 64):
        e = 1e-2 * (32.0 / n) ** exponent
        es = 2e-2 * (32.0 / n) ** (exponent if solved_exponent is None else solved_exponent)
        inputs = _write_grid(tmp_path, cfg, n, err_presc=e, err_solved=es, err_psi=1e-3 * (32.0 / n) ** 2,
                             gates_ok=gates_ok)
        reduction.reduce_grid(n=n, cfg=cfg, output=tmp_path, identity="ID", inputs=inputs)
    return cfg


def test_headline_criterion_passes_fails_and_is_undefined(tmp_path):
    cfg = _three_grids(tmp_path, exponent=2.0)
    results = {n: reduction.load_grid(tmp_path, n, "ID")[1] for n in (32, 48, 64)}
    h = analysis.headline_criterion(results, cfg)
    assert h["pass"] is True and h["informational"] is True and h["user_decides"] is True
    assert h["rows_failed"] == 0 and h["rows_undefined"] == 0
    assert len(h["rows"]) == 3 * 2 * 4                                   # non-constant variants x 2 arms x 4 fields
    assert all(r["orders"] == pytest.approx([2.0, 2.0]) for r in h["rows"])
    assert {r["variant"] for r in h["rows"]} == {"main_phi_dirichlet", "heldout_phi_dirichlet", "dirichlet_rich"}
    assert {r["cmp"] for r in h["rows"]} == {"presc_vs_ref", "solved_vs_ref"}
    # a solved arm that converges at order 1.5 fails the criterion (and is reported, not raised)
    cfg2 = _three_grids(tmp_path / "slow", exponent=2.0, solved_exponent=1.5)
    results2 = {n: reduction.load_grid(tmp_path / "slow", n, "ID")[1] for n in (32, 48, 64)}
    h2 = analysis.headline_criterion(results2, cfg2)
    assert h2["pass"] is False and h2["rows_failed"] == 12
    assert {r["cmp"] for r in h2["rows"] if r["pass"] is False} == {"solved_vs_ref"}
    # one grid only: the orders are undefined
    h3 = analysis.headline_criterion({32: results[32]}, cfg)
    assert h3["pass"] is None and h3["rows_undefined"] == 24
    # two grids: one interval
    h4 = analysis.headline_criterion({32: results[32], 48: results[48]}, cfg)
    assert h4["pass"] is True and all(len(r["orders"]) == 1 for r in h4["rows"])


def _args(output, grids=(32, 48, 64)):
    return SimpleNamespace(output=Path(output), resolutions=list(grids))


def _passing_preflight(output):
    runner.write_json(Path(output) / "preflight.json", {"identity": "ID", "all_pass": True,
                                                         "cases": {"32": {"all_pass": True}}})


def test_validate_requires_the_preflight_and_writes_the_gates(tmp_path):
    _three_grids(tmp_path)
    inputs = {}
    with pytest.raises(ValueError, match="requires a matching preflight"):
        campaign.validate(args=_args(tmp_path), identity="ID", inputs=inputs)
    runner.write_json(tmp_path / "preflight.json", {"identity": "ID", "all_pass": False,
                                                     "cases": {"32": {"all_pass": False}}})
    with pytest.raises(ValueError, match="preflight did not pass"):
        campaign.validate(args=_args(tmp_path), identity="ID", inputs=inputs)
    _passing_preflight(tmp_path)
    with pytest.raises(ValueError, match="requires a matching preflight"):
        campaign.validate(args=_args(tmp_path), identity="OTHER", inputs=inputs)
    payload = campaign.validate(args=_args(tmp_path), identity="ID", inputs=inputs)
    saved = json.loads((tmp_path / "validation.json").read_text())
    assert saved == json.loads(json.dumps(checkpoints.jsonable(payload)))
    assert saved["identity"] == "ID" and saved["acceptance_gate"] is None and saved["user_decides_acceptance"] is True
    assert saved["solver_gates_pass"] is True and saved["all_finite"] is True and saved["grids_pass"] is True
    assert saved["headline_order_criterion"]["pass"] is True and saved["headline_order_criterion"]["informational"] is True
    assert saved["operator_options"] == FINAL and set(saved["grids"]) == {"32", "48", "64"}
    assert saved["grids"]["48"]["solver"]["main_phi_dirichlet"]["iterations"] == 55
    assert saved["grids"]["64"]["grid_pass"] is True


def test_validate_records_failing_solver_gates_and_a_failing_order_without_raising(tmp_path):
    _three_grids(tmp_path / "gates", gates_ok=False)
    _passing_preflight(tmp_path / "gates")
    saved = campaign.validate(args=_args(tmp_path / "gates"), identity="ID", inputs={})
    assert saved["solver_gates_pass"] is False and saved["grids_pass"] is False and saved["all_finite"] is True
    _three_grids(tmp_path / "order", exponent=1.0)
    _passing_preflight(tmp_path / "order")
    saved = campaign.validate(args=_args(tmp_path / "order"), identity="ID", inputs={})
    assert saved["grids_pass"] is True and saved["headline_order_criterion"]["pass"] is False   # informational only


def test_analyze_writes_the_report_and_the_summary(tmp_path):
    cfg = _three_grids(tmp_path)
    payload = analysis.analyze(output=tmp_path, identity="ID", grids=[32, 48, 64], cfg=cfg)
    report = (tmp_path / "summary" / "step5_combined_report.md").read_text()
    saved = json.loads((tmp_path / "summary" / "step5_combined_summary.json").read_text())
    assert saved["identity"] == "ID" and saved["grids"] == [32, 48, 64] and saved["acceptance_gate"] is None
    assert saved["phi_solve"]["rtol"] == 1e-11 and saved["phi_solve"]["rtol_production_default"] == 1e-8
    assert saved["headline_order_criterion"]["pass"] is True and len(saved["solver"]) == 3 * 4
    key = reduction.row_key("main_phi_dirichlet", "presc_vs_ref", "Te", "total", "global")
    assert saved["orders"][key]["order_rel_l2"] == pytest.approx([2.0, 2.0])
    assert saved["orders"][key]["rel_l2"] == pytest.approx([1e-2, 1e-2 * (32 / 48) ** 2, 1e-2 * 0.25])
    assert key.replace("global", "physical_wall") in saved["orders"]                 # all regions are kept
    assert saved["regions"] == ["global", "interior", "physical_wall"]
    assert set(saved["phi_error"]) == {"32", "48", "64"} and "psi_diffusion" in saved
    for needle in ("# P08 step 5.3", "## Gates", "## Headline order criterion", "main_phi_dirichlet",
                   "control_constant_dirichlet", "solved_vs_presc / Te / total", "1e-11", "2.00 / 2.00",
                   "Potential error", "psi diffusion", "user decides", "(degenerate ref.)"):
        assert needle in report, needle
    assert payload["headline_order_criterion"]["pass"] is True
    # two grids only: the report still builds, with one order per row
    two = analysis.analyze(output=tmp_path, identity="ID", grids=[32, 48], cfg=cfg)
    assert two["grids"] == [32, 48]


# ---------------------------------------------------------------------------
# Preflight verdict
# ---------------------------------------------------------------------------
def _preflight_world(cfg, *, sign=1.0, presc_error=1e-3, arm_difference=1e-8, gates_pass=True):
    fs = "main"
    owners = np.array([2, 5, 7, 11])
    n_owners = 20
    refs = _synthetic_references(cfg, n_owners)
    ref = {k: v[owners] for k, v in refs.items() if f"__{fs}__" in k}
    ref["owners"] = owners
    arrays = {"psi_N": np.zeros(n_owners)}
    arrays["psi_N"][owners] = sign * ref[references.psi_key(fs, "O")] * 1.001
    for field in FIELDS:
        for arm, err in (("presc", presc_error), ("solved", presc_error + arm_difference)):
            full = np.zeros(n_owners)
            full[owners] = ref[references.ref_key(fs, field, "total")] * (1.0 + err)
            arrays[checkpoints.arm_key(arm, field, "total")] = full
    info = {"gates": {"finite": True, "consistency": gates_pass, "converged": True}, "gates_pass": gates_pass}
    return dict(ref=ref, arrays=arrays, info=info, owners=owners, volume=np.linspace(1.0, 2.0, n_owners), fs=fs, cfg=cfg)


def test_preflight_checks_pass_and_catch_sign_solver_and_sanity_failures():
    cfg = campaign.config()
    good = campaign.preflight_checks(**_preflight_world(cfg))
    assert good["all_pass"] is True and all(good["checks"].values())
    assert good["n_minus_o_psi_subset"] == pytest.approx(0.001, rel=1e-6) and good["n_owners_subset"] == 4
    assert good["prescribed_sanity_worst"] < 0.5 and good["arm_difference_worst"] < 1e-5
    # a sign error of O (the positive-operator convention) is caught by the N - O check
    bad = campaign.preflight_checks(**_preflight_world(cfg, sign=-1.0))
    assert bad["checks"]["n_minus_o_psi"] is False and bad["all_pass"] is False
    # an arm that is far from the reference fails the sanity check; a solver gate failure fails the verdict
    assert campaign.preflight_checks(**_preflight_world(cfg, presc_error=2.0))["checks"]["prescribed_sanity"] is False
    assert campaign.preflight_checks(**_preflight_world(cfg, gates_pass=False))["checks"]["solver_gates"] is False
    assert campaign.preflight_checks(**_preflight_world(cfg, arm_difference=1e-2))["checks"]["arm_difference"] is False
    world = _preflight_world(cfg)
    world["ref"][references.psi_key("main", "O")][0] = np.nan
    assert campaign.preflight_checks(**world)["checks"]["finite"] is False


def test_preflight_command_records_and_gates_run(tmp_path, monkeypatch, capsys):
    workspace, step4 = _fake_world(tmp_path, monkeypatch)
    argv = ["--step4-campaign", str(step4), "--input-root", str(workspace)]
    out = tmp_path / "out"
    calls = []

    def fake_preflight(*, n, args, identity, inputs):
        calls.append(n)
        return {"n": n, "all_pass": len(calls) > 1, "checks": {"finite": True}}

    monkeypatch.setattr(campaign, "preflight_grid", fake_preflight)
    with pytest.raises(SystemExit):                                         # a failing preflight exits nonzero
        campaign.main(["preflight", *argv, "--output", str(out)])
    assert calls == [32]
    last = json.loads((out / "last_exit.json").read_text())
    assert last["status"] == "failed_gates" and last["summary"]["all_pass"] is False
    assert json.loads((out / "preflight.json").read_text())["all_pass"] is False
    with pytest.raises(ValueError, match="preflight did not pass"):          # run refuses after a failed preflight
        campaign.main(["run", *argv, "--output", str(out), "--resolutions", "32"])
    capsys.readouterr()
    campaign.main(["preflight", *argv, "--output", str(out)])                # not resumed (failed): recomputed, passes
    assert calls == [32, 32] and json.loads((out / "preflight.json").read_text())["all_pass"] is True
    campaign.main(["preflight", *argv, "--output", str(out)])                # passed: not redone
    assert calls == [32, 32]


def test_run_flow_stops_after_a_failing_grid_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    workspace, step4 = _fake_world(tmp_path, monkeypatch)
    argv = ["--step4-campaign", str(step4), "--input-root", str(workspace)]
    out = tmp_path / "out"
    _, _ = _verify(workspace, step4, out)
    identity = json.loads((out / "provenance" / "inputs.json").read_text())["identity"]
    runner.write_json(out / "preflight.json", {"identity": identity, "all_pass": True, "cases": {"32": {"all_pass": True}}})
    ran = []

    def fake_run_grid(*, n, args, identity, inputs):
        ran.append(n)
        return {"n": n, "status": "complete", "grid_pass": n != 48, "solver_gates_pass": n != 48, "all_finite": True}

    monkeypatch.setattr(campaign, "run_grid", fake_run_grid)
    monkeypatch.setattr(campaign, "validate", lambda **kw: {"solver_gates_pass": False, "all_finite": True})
    monkeypatch.setattr(campaign, "analyze", lambda **kw: {})
    with pytest.raises(SystemExit):
        campaign.main(["run", *argv, "--output", str(out)])
    assert ran == [32, 48]                                                  # N64 is not run after N48 failed
    last = json.loads((out / "last_exit.json").read_text())
    assert last["status"] == "failed_gates" and last["summary"]["operational_complete"] is False
    assert [g["n"] for g in last["summary"]["grids"]] == [32, 48]


def test_run_stage_dispatch_and_run_grid_order(tmp_path, monkeypatch, capsys):
    workspace, step4 = _fake_world(tmp_path, monkeypatch)
    argv = ["--step4-campaign", str(step4), "--input-root", str(workspace)]
    out = tmp_path / "out"
    order = []
    monkeypatch.setattr(campaign, "references_stage", lambda *, n, args, identity:
                        order.append(("references", n)) or {"n_owners": 5, "skipped": False})
    monkeypatch.setattr(campaign, "jax_stage", lambda *, n, args, identity, inputs:
                        order.append(("jax", n)) or {"computed": ["a"], "resumed": []})
    gates = {"grid_pass": True, "solver_gates_pass": True, "all_finite": True}
    monkeypatch.setattr(campaign, "reduce_stage", lambda *, n, args, identity, inputs:
                        order.append(("reduce", n)) or {"gates": gates})
    for stage in ("references", "jax", "reduce"):
        campaign.main(["run-stage", *argv, "--output", str(out), "--stage", stage, "--n", "48"])
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["stage"] == stage
    assert order == [("references", 48), ("jax", 48), ("reduce", 48)]
    order.clear()
    args = SimpleNamespace(output=out)
    grid = campaign.run_grid(n=64, args=args, identity="ID", inputs={})
    assert order == [("references", 64), ("jax", 64), ("reduce", 64)] and grid["grid_pass"] is True
    assert grid["variants_computed"] == ["a"] and grid["references_skipped"] is False
