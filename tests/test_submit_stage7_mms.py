"""Static and dry-run checks for the remote Stage-7 MMS launcher."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "submit_stage7_mms.sbatch"


def _dry_run(tmp_path: Path, campaign: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "vessel.txt"
    makegrid.write_bytes(b"dry-run fixture\n")
    vessel.write_text("dry-run fixture\n", encoding="utf-8")
    output_root = tmp_path / "output"
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "1",
            "CAMPAIGN_KIND": campaign,
            "DRBX_ROOT": str(ROOT),
            "MAKEGRID": str(makegrid),
            "VESSEL": str(vessel),
            "OUTPUT_ROOT": str(output_root),
            # A successful dry run proves the launcher never executes Python.
            "PYTHON_BIN": "/definitely/not/a/python/interpreter",
        }
    )
    completed = subprocess.run(
        ("bash", str(LAUNCHER)),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, output_root


def _real_launcher_with_fake_python(
    tmp_path: Path,
    campaign: str,
    *,
    gate_status: int,
) -> tuple[subprocess.CompletedProcess[str], Path, list[str]]:
    output_root = tmp_path / "output"
    fake_python = tmp_path / "fake-python"
    invocation_log = tmp_path / "fake-python.log"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_PYTHON_LOG\"\n"
        "if [[ \"$*\" == *analyze_hsx_mms.py* ]]; then\n"
        "  exit \"$FAKE_GATE_STATUS\"\n"
        "fi\n"
        "exit 91\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "0",
            "CAMPAIGN_KIND": campaign,
            "DRBX_ROOT": str(ROOT),
            "MAKEGRID": str(tmp_path / "mgrid.nc"),
            "VESSEL": str(tmp_path / "vessel.txt"),
            "OUTPUT_ROOT": str(output_root),
            "PYTHON_BIN": str(fake_python),
            "SLURM_JOB_ID": "test-job",
            "FAKE_PYTHON_LOG": str(invocation_log),
            "FAKE_GATE_STATUS": str(gate_status),
        }
    )
    completed = subprocess.run(
        ("bash", str(LAUNCHER)),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    invocations = (
        invocation_log.read_text(encoding="utf-8").splitlines()
        if invocation_log.exists()
        else []
    )
    return completed, output_root, invocations


def test_launcher_leaves_scheduler_allocation_to_operator():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH" not in source
    assert "must run inside a Slurm allocation" not in source
    assert "--shard-counts 1 1 4" in source
    assert "scheduler_allocation=operator-managed" in source
    assert "-type f -name '*.py' -print0" in source
    assert "read -r -d '' source_path" in source
    assert '"${hash_inputs[@]}"' in source


@pytest.mark.parametrize(
    ("campaign", "name", "resolutions", "final_time", "dt", "save_every"),
    (
        ("frozen", "frozen_32_48_64", "32,48,64", "1e-6", "1e-6", "1"),
        ("evolved", "evolved_32_48_64_t2e-5_dt1e-6", "32,48,64", "2e-5", "1e-6", "1"),
        ("temporal_5e-7", "evolved_N64_t2e-5_dt5e-7", "64", "2e-5", "5e-7", "2"),
        ("temporal_2p5e-7", "evolved_N64_t2e-5_dt2p5e-7", "64", "2e-5", "2.5e-7", "4"),
    ),
)
def test_dry_run_builds_manifest_without_launching(
    tmp_path: Path,
    campaign: str,
    name: str,
    resolutions: str,
    final_time: str,
    dt: str,
    save_every: str,
):
    completed, output_root = _dry_run(tmp_path, campaign)
    assert completed.returncode == 0, completed.stderr
    assert "no GPU/Python command was launched" in completed.stdout
    campaign_dir = output_root / name
    manifest = (campaign_dir / "run_manifest.txt").read_text(encoding="utf-8")
    assert f"campaign_kind={campaign}" in manifest
    assert f"resolutions={resolutions}" in manifest
    assert f"final_time={final_time}" in manifest
    assert f"dt={dt}" in manifest
    assert f"save_every={save_every}" in manifest
    assert "shard_counts=1,1,4" in manifest
    assert "command_shell=" in manifest
    assert "simulate_hsx_mms.py" in manifest
    assert "scheduler_allocation=operator-managed" in manifest
    assert "sha256_begin" in manifest and "sha256_end" in manifest
    assert "simulate_hsx_mms.py" in manifest
    assert "hsx_mms_continuum_reference.py" in manifest
    hash_block = manifest.split("sha256_begin\n", 1)[1].split(
        "sha256_end\n", 1
    )[0]
    hash_paths = [line.split("  ", 1)[1] for line in hash_block.splitlines()]
    expected_hash_paths = sorted(
        {
            "scripts/submit_stage7_mms.sbatch",
            "simulate_hsx_mms.py",
            "hsx_mms_continuum_reference.py",
            "simulate_hsx_blob.py",
            "scripts/analyze_hsx_mms.py",
            *(
                str(path.relative_to(ROOT))
                for path in (ROOT / "src" / "drbx").rglob("*.py")
            ),
        }
    )
    assert hash_paths == expected_hash_paths
    assert "git_status_begin" in manifest and "git_status_end" in manifest
    if campaign == "frozen":
        assert "prerequisite_gate_mode=none" in manifest
        assert "prerequisite_count=0" in manifest
        assert "prerequisite_gate_command_shell=none" in manifest
        assert "prerequisite-gate=none" in completed.stdout
    elif campaign == "evolved":
        assert "prerequisite_gate_mode=spatial" in manifest
        assert "prerequisite_count=1" in manifest
        assert "--require-spatial" in manifest
        assert "prerequisite_aggregate_sha256_0=missing" in manifest
        assert "--require-spatial" in completed.stdout
    else:
        assert "prerequisite_gate_mode=evolved" in manifest
        assert "prerequisite_count=2" in manifest
        assert "--require-evolved" in manifest
        assert "prerequisite_aggregate_sha256_0=missing" in manifest
        assert "prerequisite_aggregate_sha256_1=missing" in manifest
        assert "--require-evolved" in completed.stdout
    assert (campaign_dir / "run_exit_status.txt").read_text().strip() == "0"
    assert not (campaign_dir / f"{name}.npz").exists()


def test_dry_run_rejects_unknown_campaign_before_launch(tmp_path: Path):
    completed, _ = _dry_run(tmp_path, "not-a-campaign")
    assert completed.returncode == 2
    assert "unsupported CAMPAIGN_KIND" in completed.stderr


def test_existing_aggregate_refusal_preserves_successful_campaign_evidence(
    tmp_path: Path,
):
    campaign_dir = tmp_path / "output" / "frozen_32_48_64"
    campaign_dir.mkdir(parents=True)
    sentinels = {
        campaign_dir / "frozen_32_48_64.npz": b"successful aggregate\x00\x01\n",
        campaign_dir / "run_manifest.txt": b"successful manifest must survive\n",
        campaign_dir / "run_exit_status.txt": b"0\n",
        campaign_dir / "end_time_utc.txt": b"2026-09-01T12:34:56Z\n",
    }
    for path, content in sentinels.items():
        path.write_bytes(content)

    completed, _, invocations = _real_launcher_with_fake_python(
        tmp_path, "frozen", gate_status=0
    )

    assert completed.returncode == 6
    assert "aggregate already exists" in completed.stderr
    assert not invocations
    for path, content in sentinels.items():
        assert path.read_bytes() == content
    assert not (campaign_dir / "start_time_utc.txt").exists()


def test_failed_spatial_prerequisite_gate_prevents_gpu_preflight_and_srun(
    tmp_path: Path,
):
    frozen_dry_run, output_root = _dry_run(tmp_path, "frozen")
    assert frozen_dry_run.returncode == 0
    frozen_dir = output_root / "frozen_32_48_64"
    frozen_aggregate = frozen_dir / "frozen_32_48_64.npz"
    frozen_aggregate.write_bytes(b"artifact content is parsed by the fake gate\n")

    completed, _, invocations = _real_launcher_with_fake_python(
        tmp_path, "evolved", gate_status=23
    )
    assert completed.returncode == 23
    assert len(invocations) == 1
    assert "analyze_hsx_mms.py" in invocations[0]
    assert "--require-spatial" in invocations[0]
    assert "import jax" not in invocations[0]
    manifest = (
        output_root
        / "evolved_32_48_64_t2e-5_dt1e-6"
        / "run_manifest.txt"
    ).read_text(encoding="utf-8")
    expected_digest = hashlib.sha256(frozen_aggregate.read_bytes()).hexdigest()
    assert f"prerequisite_aggregate_sha256_0={expected_digest}" in manifest


def test_successful_artifact_gate_runs_before_gpu_preflight(tmp_path: Path):
    frozen_dry_run, output_root = _dry_run(tmp_path, "frozen")
    assert frozen_dry_run.returncode == 0
    frozen_dir = output_root / "frozen_32_48_64"
    (frozen_dir / "frozen_32_48_64.npz").write_bytes(b"fake aggregate\n")

    completed, _, invocations = _real_launcher_with_fake_python(
        tmp_path, "evolved", gate_status=0
    )
    assert completed.returncode == 91
    assert len(invocations) == 2
    assert "analyze_hsx_mms.py" in invocations[0]
    assert "--require-spatial" in invocations[0]
    assert "import jax" in invocations[1]
    assert all("simulate_hsx_mms.py" not in line for line in invocations)


def test_source_hash_mismatch_prevents_gate_and_gpu_preflight(tmp_path: Path):
    frozen_dry_run, output_root = _dry_run(tmp_path, "frozen")
    assert frozen_dry_run.returncode == 0
    frozen_dir = output_root / "frozen_32_48_64"
    (frozen_dir / "frozen_32_48_64.npz").write_bytes(b"fake aggregate\n")
    manifest_path = frozen_dir / "run_manifest.txt"
    manifest = manifest_path.read_text(encoding="utf-8")
    before, hash_block = manifest.split("sha256_begin\n", 1)
    first_hash, remainder = hash_block.split("\n", 1)
    digest, filename = first_hash.split("  ", 1)
    changed = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest_path.write_text(
        before + "sha256_begin\n" + changed + "  " + filename + "\n" + remainder,
        encoding="utf-8",
    )

    completed, _, invocations = _real_launcher_with_fake_python(
        tmp_path, "evolved", gate_status=0
    )
    assert completed.returncode == 7
    assert not invocations
    assert "source-hash blocks differ" in completed.stderr
