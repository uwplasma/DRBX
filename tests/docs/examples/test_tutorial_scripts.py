"""Smoke test: every tutorial script runs end to end with reduced constants.

Mechanism: each script keeps its inputs as top-level ``NAME = value`` lines in
the "Input parameters" block. The test reads the script, replaces the listed
assignments (one exact top-level line each, asserted), sends figures and
outputs to a temporary directory, and executes the patched source in a fresh
interpreter whose working directory is a temporary folder containing a link to
``examples/`` (the scripts use repository-relative paths). The scripts
themselves are not modified and read no environment variables.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TUTORIAL = REPO_ROOT / "examples" / "tutorial"
HSX_BUNDLE = REPO_ROOT / "artifacts" / "geometry" / "hsx_fci_32x32x32"

# script stem -> {constant: reduced value (Python source)}
REDUCED = {
    "01_first_run": {"NOUT": "4"},
    "02_drift_wave_turbulence": {"N": "16", "STEPS_PER_BLOCK": "20", "BLOCKS": "3"},
    "03_linear_vs_nonlinear": {"KY_SCAN": "np.linspace(0.05, 3.0, 8)", "N": "16", "MODE": "(0, 2)",
                               "STEPS_PER_BLOCK": "20", "BLOCKS": "3"},
    "04_open_sol_sheath": {"NZ": "40", "STEPS": "400", "REPORT_EVERY": "200"},
    "05_neutrals_and_detachment": {"NZ": "40", "UPSTREAM_SCAN_A": "(1.0, 4.0)", "STEPS_A": "400",
                                   "REPORT_EVERY_A": "200", "NY_B": "40", "UPSTREAM_SCAN_B": "(2.0e19, 3.0e19)"},
    "06_geometries": {"SHAPE": "(6, 8, 4)", "STEPS": "4", "REPORT_EVERY": "2",
                      "CONNECTION_GRID": "(12, 8)", "MAX_TRANSITS": "4"},
    "07_differentiable": {"N": "12", "STEPS": "10", "ITERATIONS": "3", "NY": "40"},
    "08_hsx_fci_braginskii": {"NUM_STEPS": "1", "FINAL_TIME": "7.5e-4"},
}


def patched_source(path: Path, overrides: dict[str, str], tmp: Path) -> str:
    source = path.read_text(encoding="utf-8")
    overrides = dict(overrides)
    overrides["FIGURE"] = f"Path({str(tmp / (path.stem + '.png'))!r})"
    for name in ("OUTPUT_DIR", "OUTPUT"):
        if re.search(rf"^{name} = ", source, flags=re.M):
            overrides[name] = f"Path({str(tmp / 'out' / path.stem)!r})" + (" / 'history.npz'" if name == "OUTPUT" else "")
    if path.stem == "08_hsx_fci_braginskii":
        overrides["GEOMETRY"] = f"Path({str(HSX_BUNDLE)!r})"
    for name, value in overrides.items():
        source, count = re.subn(rf"^{name}(?: *: *[^=\n]+)? *=.*$", lambda _: f"{name} = {value}", source,
                                count=1, flags=re.M)
        assert count == 1, f"{path.name}: no top-level assignment of {name}"
    return source


@pytest.mark.parametrize("stem", sorted(REDUCED))
def test_tutorial_script_runs_with_reduced_constants(stem: str, tmp_path: Path) -> None:
    script = TUTORIAL / f"{stem}.py"
    assert script.is_file()
    if stem == "08_hsx_fci_braginskii" and not (HSX_BUNDLE / "manifest.json").is_file():
        pytest.skip("HSX geometry bundle (artifacts/geometry/hsx_fci_32x32x32) not available")
    patched = tmp_path / script.name
    patched.write_text(patched_source(script, REDUCED[stem], tmp_path), encoding="utf-8")
    (tmp_path / "examples").symlink_to(REPO_ROOT / "examples")
    environment = dict(os.environ, MPLBACKEND="Agg")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT / "src"), environment.get("PYTHONPATH")]))
    completed = subprocess.run([sys.executable, str(patched)], cwd=tmp_path, env=environment, text=True,
                               capture_output=True, timeout=600)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    assert (tmp_path / f"{stem}.png").is_file(), completed.stdout[-2000:]
