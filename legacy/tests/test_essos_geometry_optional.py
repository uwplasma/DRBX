from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxdrb.geometry import TabulatedGeometry
from jaxdrb.geometry.essos import (
    biotsavart_fieldline_to_tabulated,
    near_axis_fieldline_to_tabulated,
    vmec_fieldline_to_tabulated,
)


def _essos_root() -> Path:
    essos = pytest.importorskip("essos")  # namespace package in editable installs
    roots = list(getattr(essos, "__path__", []))
    if not roots:
        pytest.skip("Could not locate ESSOS package path.")
    # ESSOS can be installed such that __path__ points at either:
    #   - <repo>/essos          (namespace root)
    #   - <repo>/essos/essos    (package dir)
    # The example input files live at <repo>/essos/examples/input_files.
    for p in roots:
        cand = Path(p).resolve()
        for up in [cand, cand.parent, cand.parent.parent]:
            if (up / "examples" / "input_files").exists():
                return up
    pytest.skip("Could not locate ESSOS repo root containing examples/input_files.")


def test_essos_near_axis_tabulated_geometry_smoke(tmp_path: Path) -> None:
    # A small near-axis configuration and a short field-line trace: this should run quickly
    # and produce a valid TabulatedGeometry file with a uniform l-grid.
    rc = np.array([1.0, 0.045])
    zs = np.array([0.0, -0.045])
    out = tmp_path / "near_axis_geom.npz"

    res = near_axis_fieldline_to_tabulated(
        rc=rc,
        zs=zs,
        etabar=-0.9,
        nfp=3,
        r=0.1,
        alpha=0.0,
        nphi=41,
        out_path=out,
    )
    assert res.path.exists()
    geom = TabulatedGeometry.from_npz(res.path)
    assert geom.l.ndim == 1 and geom.l.size == 41
    assert np.all(np.isfinite(np.asarray(geom.kperp2(0.0, 0.3))))
    assert np.all(np.isfinite(np.asarray(geom.B())))


@pytest.mark.parametrize("s", [0.9])
def test_essos_vmec_tabulated_geometry_smoke(tmp_path: Path, s: float) -> None:
    root = _essos_root()
    wout = root / "examples" / "input_files" / "wout_QH_simple_scaled.nc"
    if not wout.exists():
        pytest.skip("ESSOS VMEC example file not found.")

    out = tmp_path / "vmec_geom.npz"
    res = vmec_fieldline_to_tabulated(
        wout_file=wout, s=s, alpha=0.0, nphi=32, nfield_periods=1, out_path=out
    )
    assert res.path.exists()
    geom = TabulatedGeometry.from_npz(res.path)
    k2 = np.asarray(geom.kperp2(0.0, 0.3))
    assert np.all(np.isfinite(k2))
    assert np.all(k2 > 0.0)


def test_essos_biotsavart_tabulated_geometry_smoke(tmp_path: Path) -> None:
    root = _essos_root()
    coils = root / "examples" / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json"
    if not coils.exists():
        pytest.skip("ESSOS Biot-Savart example coils JSON not found.")

    out = tmp_path / "biotsavart_geom.npz"
    res = biotsavart_fieldline_to_tabulated(
        coils_json=coils,
        R0=1.4,
        Z0=0.0,
        phi0=0.0,
        nsteps=200,
        nout=64,
        maxtime=200.0,
        out_path=out,
    )
    assert res.path.exists()
    geom = TabulatedGeometry.from_npz(res.path)
    assert geom.l.size == 64
    assert np.all(np.isfinite(np.asarray(geom.B())))
