"""Step-6 transverse-wave campaign on real HSX data, bounded (slow): the campaign's ``preflight_grid`` at N32 with the
four pinned operator options: the ``compact_c3`` evaluator is verified, the owner averages of the new fields are bitwise
the one-shot P06N routine and reproduce the committed P06N ``owner_values`` file when applied to the catalogue, the
references of both field sets on the 12 closure owners are finite and non-zero and agree with the independent production
formulas, the Hessians match finite differences, the axis is smooth and the u-bands partition the owners. Needs no
artifact rows.

Needs the HSX N32 geometry / sidecar and the P06N N32 oracle file (skipped otherwise). About half a minute.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N = 32
FINAL = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius",
         "bfield_toroidal": "compact_c3"}

pytestmark = pytest.mark.slow

_have = (GEOMETRY / f"{N}x{N}x{N}" / "base_geometry.npz").is_file() and SIDECAR.is_file()
needs_inputs = pytest.mark.skipif(not _have, reason="HSX N32 geometry/sidecar inputs are unavailable")


@needs_inputs
def test_campaign_preflight_passes_at_n32(tmp_path):
    from p_shared.replay_support import DEFAULT_PATHS
    from p08_step6_global import campaign

    paths = dict(DEFAULT_PATHS)
    oracle = Path(paths["p05n_p06n_upwind"]) / "p06n" / f"N{N}.owner_values.npz"
    if not oracle.is_file():
        pytest.skip("the P06N N32 owner_values oracle file is unavailable")
    case = campaign.preflight_grid(n=N, input_root=WORKSPACE, sidecar=SIDECAR, output=tmp_path, paths=paths)
    assert case["all_pass"] is True, {k: v for k, v in case["checks"].items() if not v}
    assert case["operator_options"] == FINAL and case["jax"]["backend"] == "cpu" and case["jax"]["x64"] is True
    assert case["owner_average"]["chunked_vs_oneshot_max_abs"] == 0.0
    assert case["owner_average"]["p06n_reproduction_max_abs"] <= 1e-13
    assert case["references"]["owners"] == 12 and case["u_bands"]["owner_counts"]["uband_0.12-0.21"] > 0
    assert (tmp_path / "preflight" / f"N{N}_transverse.json").is_file()
    report = json.loads((tmp_path / "preflight" / f"N{N}_transverse.json").read_text())
    assert report["derivatives"]["worst_hessian_fd_rel"] <= 1e-6
    print("\nstep-6 preflight N32:", {k: case[k] for k in ("peak_rss_gib", "seconds")})


SUBSTITUTE_WORKER = Path(__file__).resolve().with_name("p08_step6_sharding_substitute_worker.py")
SUBSTITUTE_EXPORT = WORKSPACE / "work/p08-step5-export-2ff50718-20261001T165909Z-ebcbd2"


@needs_inputs
def test_sharding_stage_machinery_on_substitute_n32_data(tmp_path, monkeypatch):
    """``sharding.run`` end to end with a REAL subprocess (8 forced host devices, log capture, result parsing, gates,
    record) whose worker (``tests/p08_step6_sharding_substitute_worker.py``) runs the production ``run_checks`` on substitute
    data: the bounded N32 closure plan (targets on every eta plane, P06N variants ``main_phi_dirichlet`` /
    ``dirichlet_rich``) for the RHS part and the local N32 P07 Dirichlet export (manufactured solution, rtol 1e-10) for the
    potential part. The campaign's own artifacts are not needed. Shard counts 2, 4 and 8."""
    from types import SimpleNamespace

    from p_shared.replay_support import DEFAULT_PATHS
    from p08_step6_global import campaign, sharding

    if not (SUBSTITUTE_EXPORT / f"N{N}" / "p07_dirichlet.npz").is_file():
        pytest.skip("the local N32 P07 export is unavailable")
    if not (Path(dict(DEFAULT_PATHS)["p05n_p06n_upwind"]) / "p06n" / f"N{N}.owner_values.npz").is_file():
        pytest.skip("the P06N N32 owner_values oracle file is unavailable")
    monkeypatch.setattr(sharding, "worker_command", lambda spec, result: [
        sys.executable, str(SUBSTITUTE_WORKER), "--worker", str(spec), "--result", str(result)])
    monkeypatch.setenv("P08_STEP6_SHARDING_SUBSTITUTE_EXPORT", str(SUBSTITUTE_EXPORT))
    args = SimpleNamespace(output=tmp_path, input_root=WORKSPACE, oracle_root=None, sharding_max_shards=None)
    record = sharding.run(n=N, args=args, identity="ID", inputs={}, cfg=campaign.config(), log=print)
    log = (tmp_path / "logs" / f"sharding_N{N}.log").read_text()
    assert record["devices"] == 8 and record["source"] == "substitute" and record["shard_counts"] == [2, 4, 8]
    assert record["status"] == "pass", record["failures"]
    assert record["gates"]["rhs"] and record["gates"]["phi"] and record["gates"]["complete"]
    for variant, entries in record["rhs"].items():
        for sz in (2, 4, 8):
            e = entries[f"Sz{sz}"]
            assert e["finite"] and e["worst_rel"] <= 1e-12 and len(e["fields"]) == 16, (variant, sz)
            assert all(m["scale"] > 0 for m in e["fields"].values()), (variant, sz)
            assert e["lowering_seconds"] > 0 and e["first_seconds"] > 0 and e["seconds"] > 0
    for sz in (2, 4, 8):
        e = record["phi"]["manufactured"][f"Sz{sz}"]
        single = record["phi"]["manufactured"]["single"]
        assert e["pass"] and e["converged"] and abs(e["iterations"] - single["iterations"]) <= 1
        assert e["dpsi_m"] <= 10 * 1e-10 * e["psi_m"]
    assert record["peak_rss_gib"]["final"] > 0
    assert "Sz8 RHS main_phi_dirichlet" in log and "Sz8 solve manufactured" in log
    print(f"\nsharding substitute N32: worst rhs rel "
          f"{max(e['worst_rel'] for v in record['rhs'].values() for k, e in v.items() if k.startswith('Sz')):.2e}, "
          f"bitwise {record['gates']['rhs_bitwise']}, peak RSS {record['peak_rss_gib']}, "
          f"wall {record['wall_seconds']:.0f} s")
    assert sharding.stored_record(tmp_path, N, "ID") is None                                  # the campaign wrapper stores
