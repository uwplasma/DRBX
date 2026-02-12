from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxdrb.fci.builder import (
    EssosToroidalFCIConfig,
    build_fci_maps_essos_toroidal_planes,
)
from jaxdrb.fci.io import load_fci_maps_npz, save_fci_maps_npz


class _FakeEssosField:
    """Simple analytic B field in Cartesian coordinates.

    Cylindrical components are:
      B_R = 0.08
      B_phi = 1.0
      B_Z = 0.15
    """

    def B(self, point_xyz: np.ndarray) -> np.ndarray:
        x, y, _ = point_xyz
        phi = np.arctan2(y, x)
        BR = 0.08
        Bphi = 1.0
        BZ = 0.15
        Bx = BR * np.cos(phi) - Bphi * np.sin(phi)
        By = BR * np.sin(phi) + Bphi * np.cos(phi)
        return np.asarray([Bx, By, BZ], dtype=float)


def test_fci_essos_toroidal_builder_hits_and_metadata() -> None:
    cfg = EssosToroidalFCIConfig(
        R0=1.0,
        Z0=-0.2,
        dR=0.05,
        dZ=0.1,
        nR=6,
        nZ=5,
        phi0=0.0,
        dphi=0.25,
        nphi=5,
        open_field_line=True,
        periodic_R=False,
        periodic_Z=False,
        R_min=1.0,
        R_max=1.20,
        Z_min=-0.2,
        Z_max=0.2,
    )
    map_fwd, map_bwd, meta = build_fci_maps_essos_toroidal_planes(
        cfg,
        field=_FakeEssosField(),
        nsub=12,
    )

    assert map_fwd.ix.shape == (cfg.nphi, cfg.nR, cfg.nZ, 4)
    assert map_fwd.w.shape == (cfg.nphi, cfg.nR, cfg.nZ, 4)
    assert map_fwd.hit is not None
    assert map_fwd.dl_hit is not None
    assert map_fwd.hit_R is not None
    assert map_fwd.hit_Z is not None
    assert map_fwd.hit_phi is not None
    assert map_fwd.hit_target is not None
    assert int(jnp.count_nonzero(map_fwd.hit)) > 0
    assert int(jnp.count_nonzero(map_bwd.hit)) > 0

    hit_mask = np.asarray(map_fwd.hit)
    dl = np.asarray(map_fwd.dl)
    dl_hit = np.asarray(map_fwd.dl_hit)
    assert np.all(dl_hit[hit_mask] <= dl[hit_mask] + 1e-12)
    assert np.all(np.isfinite(np.asarray(map_fwd.hit_R)[hit_mask]))
    assert np.all(np.isfinite(np.asarray(map_fwd.hit_Z)[hit_mask]))
    assert np.all(np.isfinite(np.asarray(map_fwd.hit_phi)[hit_mask]))
    assert np.all(np.asarray(map_fwd.hit_target)[hit_mask] == 1)

    assert meta["builder"] == "essos_toroidal_planes"
    assert meta["n_hit_fwd"] > 0.0
    assert meta["n_hit_bwd"] > 0.0


def test_fci_map_io_roundtrip_with_target_metadata(tmp_path: Path) -> None:
    cfg = EssosToroidalFCIConfig(
        R0=1.0,
        Z0=-0.1,
        dR=0.05,
        dZ=0.1,
        nR=5,
        nZ=4,
        phi0=0.0,
        dphi=0.2,
        nphi=4,
        open_field_line=True,
        R_min=1.0,
        R_max=1.14,
        Z_min=-0.1,
        Z_max=0.2,
    )
    map_fwd, map_bwd, meta = build_fci_maps_essos_toroidal_planes(
        cfg,
        field=_FakeEssosField(),
        nsub=8,
    )

    path = tmp_path / "fci_essos_map.npz"
    save_fci_maps_npz(path, map_fwd=map_fwd, map_bwd=map_bwd, meta=meta)
    fwd2, bwd2, meta2 = load_fci_maps_npz(path)

    assert meta2["builder"] == "essos_toroidal_planes"
    np.testing.assert_array_equal(np.asarray(fwd2.hit), np.asarray(map_fwd.hit))
    np.testing.assert_allclose(np.asarray(fwd2.hit_R), np.asarray(map_fwd.hit_R), equal_nan=True)
    np.testing.assert_allclose(np.asarray(fwd2.hit_Z), np.asarray(map_fwd.hit_Z), equal_nan=True)
    np.testing.assert_allclose(
        np.asarray(fwd2.hit_phi), np.asarray(map_fwd.hit_phi), equal_nan=True
    )
    np.testing.assert_array_equal(np.asarray(fwd2.hit_target), np.asarray(map_fwd.hit_target))
    np.testing.assert_array_equal(np.asarray(bwd2.hit), np.asarray(map_bwd.hit))


def _essos_root() -> Path:
    essos = pytest.importorskip("essos")
    roots = list(getattr(essos, "__path__", []))
    if not roots:
        pytest.skip("Could not locate ESSOS package path.")
    p = Path(roots[0]).resolve()
    if (p / "examples" / "input_files").exists():
        return p
    if (p.parent / "examples" / "input_files").exists():
        return p.parent
    pytest.skip("Could not locate ESSOS repo root containing examples/input_files.")


def test_fci_essos_biotsavart_builder_smoke() -> None:
    pytest.importorskip("essos")
    from essos.coils import Coils_from_json
    from essos.fields import BiotSavart

    root = _essos_root()
    coils_file = root / "examples" / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json"
    if not coils_file.exists():
        pytest.skip("ESSOS Biot-Savart coils file not found.")

    field = BiotSavart(Coils_from_json(str(coils_file)))
    cfg = EssosToroidalFCIConfig(
        R0=1.18,
        Z0=-0.08,
        dR=0.03,
        dZ=0.04,
        nR=4,
        nZ=4,
        phi0=0.0,
        dphi=0.10,
        nphi=3,
        open_field_line=True,
        R_min=1.15,
        R_max=1.30,
        Z_min=-0.12,
        Z_max=0.12,
    )
    map_fwd, map_bwd, _ = build_fci_maps_essos_toroidal_planes(
        cfg,
        field=field,
        nsub=6,
    )
    assert map_fwd.ix.shape == (cfg.nphi, cfg.nR, cfg.nZ, 4)
    assert map_bwd.ix.shape == (cfg.nphi, cfg.nR, cfg.nZ, 4)
