from pathlib import Path
import sys

import numpy as np
from scipy.io import netcdf_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vmec_logical_surfaces import (  # noqa: E402
    VmecSurfaceModel,
    compute_vmec_flux_label,
    load_or_compute_vmec_flux_label,
)


def _write_circular_wout(path: Path) -> Path:
    s = np.linspace(0.0, 1.0, 33)
    with netcdf_file(path, "w") as dataset:
        dataset.createDimension("ns", s.size)
        dataset.createDimension("mn_mode", 2)
        dataset.createVariable("nfp", "i", ())[...] = 1
        dataset.createVariable("lasym__logical__", "i", ())[...] = 0
        dataset.createVariable("xm", "d", ("mn_mode",))[:] = [0.0, 1.0]
        dataset.createVariable("xn", "d", ("mn_mode",))[:] = [0.0, 0.0]
        dataset.createVariable("phi", "d", ("ns",))[:] = s
        rmnc = dataset.createVariable("rmnc", "d", ("ns", "mn_mode"))
        zmns = dataset.createVariable("zmns", "d", ("ns", "mn_mode"))
        rmnc[:] = np.column_stack((np.full_like(s, 1.4), 0.2 * np.sqrt(s)))
        zmns[:] = np.column_stack((np.zeros_like(s), 0.2 * np.sqrt(s)))
    return path


def test_vmec_inverse_recovers_normalized_flux(tmp_path: Path) -> None:
    wout_path = _write_circular_wout(tmp_path / "wout_circle.nc")
    model = VmecSurfaceModel.from_wout(wout_path)
    true_flux = np.array([0.05, 0.2, 0.55, 0.9])
    theta = np.array([0.2, 1.4, 3.3, 5.2])
    phi = np.array([0.0, 0.3, 1.1, 5.0])
    major_radius, vertical = model.evaluate(true_flux, theta, phi)
    fitted_flux, _fitted_theta, residual = model.invert_rz(
        major_radius, vertical, phi
    )
    np.testing.assert_allclose(fitted_flux, true_flux, atol=2.0e-12)
    assert np.max(residual) < 1.0e-12


def test_mapping_masks_points_outside_vmec_lcfs_and_caches(tmp_path: Path) -> None:
    wout_path = _write_circular_wout(tmp_path / "wout_circle.nc")
    positions = np.array(
        [
            [[1.4, 0.0, 0.0], [1.5, 0.0, 0.0]],
            [[1.4, 0.0, 0.1], [1.7, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    flux, residual = compute_vmec_flux_label(positions, wout_path, batch_size=2)
    np.testing.assert_allclose(flux[0, 0], 0.0, atol=2.0e-6)
    np.testing.assert_allclose(flux[0, 1], 0.25, atol=2.0e-3)
    np.testing.assert_allclose(flux[1, 0], 0.25, atol=2.0e-3)
    assert np.isnan(flux[1, 1])
    assert residual[1, 1] > 0.05

    first, first_metadata = load_or_compute_vmec_flux_label(
        positions, wout_path, tmp_path / "cache"
    )
    second, second_metadata = load_or_compute_vmec_flux_label(
        positions, wout_path, tmp_path / "cache"
    )
    np.testing.assert_equal(first, second)
    assert not first_metadata["cache_hit"]
    assert second_metadata["cache_hit"]
