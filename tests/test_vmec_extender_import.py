from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

from jax_drb.geometry.vmec_extender_import import (
    build_vmec_extender_fci_maps,
    interpolate_vmec_extender_B_cyl,
    load_vmec_extender_grid_netcdf,
    vmec_extender_absB,
    vmec_extender_fieldline_rhs_RZ_phi,
)


def _analytic_B(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    R = points[..., 0]
    phi = points[..., 1]
    Z = points[..., 2]
    return np.stack((R + 2.0 * phi + 3.0 * Z, 2.0 + R, R - phi + Z), axis=-1)


def _write_synthetic_vmec_extender_grid(
    path: Path,
    *,
    nfp: int = 5,
    missing_convention: bool = False,
    zeta_convention: bool = False,
    nonmonotone_R: bool = False,
    inconsistent_shape: bool = False,
    inconsistent_absB: bool = False,
    tiny_Bphi: bool = False,
) -> Path:
    phi_period = 2.0 * np.pi / float(nfp)
    R = np.asarray([1.0, 1.35, 1.8], dtype=np.float64)
    if nonmonotone_R:
        R = np.asarray([1.0, 0.9, 1.8], dtype=np.float64)
    phi = np.linspace(0.0, phi_period, 4, endpoint=False, dtype=np.float64)
    Z = np.asarray([-0.25, 0.2, 0.85], dtype=np.float64)
    RR, PP, ZZ = np.meshgrid(R, phi, Z, indexing="ij")
    points = np.stack((RR, PP, ZZ), axis=-1)
    B = _analytic_B(points)
    if tiny_Bphi:
        B[..., 1] = 0.0
    absB = np.sqrt(np.sum(B * B, axis=-1))
    if inconsistent_absB:
        absB = absB + 1.0

    with Dataset(path, "w") as dataset:
        dataset.createDimension("nR", R.size)
        dataset.createDimension("nphi", phi.size)
        dataset.createDimension("nZ", Z.size)
        dataset.createVariable("R", "f8", ("nR",))[:] = R
        dataset.createVariable("phi", "f8", ("nphi",))[:] = phi
        dataset.createVariable("Z", "f8", ("nZ",))[:] = Z
        if inconsistent_shape:
            dataset.createVariable("BR", "f8", ("nR", "nphi"))[:] = B[..., 0].mean(axis=-1)
        else:
            dataset.createVariable("BR", "f8", ("nR", "nphi", "nZ"))[:] = B[..., 0]
        dataset.createVariable("Bphi", "f8", ("nR", "nphi", "nZ"))[:] = B[..., 1]
        dataset.createVariable("BZ", "f8", ("nR", "nphi", "nZ"))[:] = B[..., 2]
        dataset.createVariable("absB", "f8", ("nR", "nphi", "nZ"))[:] = absB
        dataset.setncattr("format", "extended_field")
        if not missing_convention:
            convention = "(R, zeta, Z)" if zeta_convention else "physical cylindrical (R, phi, Z)"
            dataset.setncattr("coordinate_convention", convention)
        dataset.setncattr("field_components", "BR,Bphi,BZ")
        dataset.setncattr("nfp", int(nfp))
        dataset.setncattr("stellsym", 1)
        dataset.setncattr("source", "synthetic_vmec_extender_test")
        dataset.setncattr("src_nphi", 8)
        dataset.setncattr("src_ntheta", 8)
        dataset.setncattr("digits", 8)
        dataset.setncattr("branch", "internal")
        dataset.setncattr("units", "SI")
    return path


def test_vmec_extender_import_interpolates_grid_nodes_and_midpoints(tmp_path: Path) -> None:
    grid = load_vmec_extender_grid_netcdf(_write_synthetic_vmec_extender_grid(tmp_path / "field.nc"))
    RR, PP, ZZ = np.meshgrid(np.asarray(grid.R), np.asarray(grid.phi), np.asarray(grid.Z), indexing="ij")
    node_points = np.stack((RR, PP, ZZ), axis=-1)

    np.testing.assert_allclose(
        np.asarray(interpolate_vmec_extender_B_cyl(grid, node_points)),
        _analytic_B(node_points),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    midpoint = np.asarray(
        [
            0.5 * (float(grid.R[0]) + float(grid.R[1])),
            0.5 * (float(grid.phi[1]) + float(grid.phi[2])),
            0.5 * (float(grid.Z[0]) + float(grid.Z[1])),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        np.asarray(interpolate_vmec_extender_B_cyl(grid, midpoint)),
        _analytic_B(midpoint),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_vmec_extender_import_wraps_physical_phi_and_handles_shapes(tmp_path: Path) -> None:
    grid = load_vmec_extender_grid_netcdf(_write_synthetic_vmec_extender_grid(tmp_path / "field.nc", nfp=5))
    point = jnp.asarray([1.35, float(grid.phi[1]), 0.2], dtype=jnp.float64)
    wrapped = point.at[1].add(grid.phi_period)

    np.testing.assert_allclose(
        np.asarray(interpolate_vmec_extender_B_cyl(grid, point)),
        np.asarray(interpolate_vmec_extender_B_cyl(grid, wrapped)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    batched = jnp.stack((point, point.at[0].set(1.5)), axis=0)
    higher_rank = jnp.reshape(jnp.stack((batched, batched), axis=0), (2, 2, 3))
    assert interpolate_vmec_extender_B_cyl(grid, point).shape == (3,)
    assert interpolate_vmec_extender_B_cyl(grid, batched).shape == (2, 3)
    assert interpolate_vmec_extender_B_cyl(grid, higher_rank).shape == (2, 2, 3)


def test_vmec_extender_interpolation_is_jittable_and_locally_differentiable(tmp_path: Path) -> None:
    grid = load_vmec_extender_grid_netcdf(_write_synthetic_vmec_extender_grid(tmp_path / "field.nc"))
    point = jnp.asarray([1.2, 0.2, 0.0], dtype=jnp.float64)

    compiled = jax.jit(lambda imported_grid, target: interpolate_vmec_extender_B_cyl(imported_grid, target))
    np.testing.assert_allclose(np.asarray(compiled(grid, point)), _analytic_B(np.asarray(point)), rtol=1.0e-12, atol=1.0e-12)

    jacobian = jax.jacfwd(lambda target: interpolate_vmec_extender_B_cyl(grid, target))(point)
    expected = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0],
            [1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.asarray(jacobian), expected, rtol=1.0e-12, atol=1.0e-12)


def test_vmec_extender_absB_and_fieldline_rhs_match_definitions(tmp_path: Path) -> None:
    grid = load_vmec_extender_grid_netcdf(_write_synthetic_vmec_extender_grid(tmp_path / "field.nc"))
    point = jnp.asarray([1.35, float(grid.phi[1]), 0.2], dtype=jnp.float64)
    B = _analytic_B(np.asarray(point))
    rhs = vmec_extender_fieldline_rhs_RZ_phi(grid, point)

    assert vmec_extender_absB(grid, point) == pytest.approx(float(np.sqrt(np.sum(B * B))))
    np.testing.assert_allclose(
        np.asarray(rhs),
        np.asarray([point[0] * B[0] / B[1], point[0] * B[2] / B[1]], dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_vmec_extender_fieldline_rhs_bounds_tiny_Bphi(tmp_path: Path) -> None:
    grid = load_vmec_extender_grid_netcdf(_write_synthetic_vmec_extender_grid(tmp_path / "field.nc", tiny_Bphi=True))
    point = jnp.asarray([1.35, float(grid.phi[1]), 0.2], dtype=jnp.float64)
    rhs = vmec_extender_fieldline_rhs_RZ_phi(grid, point, min_abs_Bphi=0.5)

    assert np.all(np.isfinite(np.asarray(rhs)))
    B = _analytic_B(np.asarray(point))
    np.testing.assert_allclose(np.asarray(rhs), np.asarray([point[0] * B[0] / 0.5, point[0] * B[2] / 0.5]))


def test_vmec_extender_import_rejects_bad_metadata_and_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="coordinate_convention"):
        load_vmec_extender_grid_netcdf(_write_synthetic_vmec_extender_grid(tmp_path / "missing.nc", missing_convention=True))
    with pytest.raises(ValueError, match="zeta"):
        load_vmec_extender_grid_netcdf(_write_synthetic_vmec_extender_grid(tmp_path / "zeta.nc", zeta_convention=True))
    with pytest.raises(ValueError, match="strictly increasing"):
        load_vmec_extender_grid_netcdf(_write_synthetic_vmec_extender_grid(tmp_path / "axis.nc", nonmonotone_R=True))
    with pytest.raises(ValueError, match="shape"):
        load_vmec_extender_grid_netcdf(_write_synthetic_vmec_extender_grid(tmp_path / "shape.nc", inconsistent_shape=True))
    with pytest.raises(ValueError, match="absB"):
        load_vmec_extender_grid_netcdf(_write_synthetic_vmec_extender_grid(tmp_path / "absB.nc", inconsistent_absB=True))


def test_vmec_extender_import_allows_unsafe_missing_convention_when_requested(tmp_path: Path) -> None:
    grid = load_vmec_extender_grid_netcdf(
        _write_synthetic_vmec_extender_grid(tmp_path / "field.nc", missing_convention=True),
        strict_metadata=False,
    )

    assert grid.nfp == 5
    assert grid.phi_period == pytest.approx(2.0 * np.pi / 5.0)


def test_vmec_extender_fci_maps_reduce_to_identity_for_toroidal_field(tmp_path: Path) -> None:
    grid = load_vmec_extender_grid_netcdf(_write_synthetic_vmec_extender_grid(tmp_path / "field.nc", tiny_Bphi=True))
    toroidal_grid = grid.__class__(
        R=grid.R,
        phi=grid.phi,
        Z=grid.Z,
        BR=jnp.zeros_like(grid.BR),
        Bphi=jnp.ones_like(grid.Bphi),
        BZ=jnp.zeros_like(grid.BZ),
        absB=jnp.ones_like(grid.absB),
        nfp=grid.nfp,
        phi_period=grid.phi_period,
        metadata=grid.metadata,
    )

    maps = build_vmec_extender_fci_maps(toroidal_grid)
    expected_R = np.arange(toroidal_grid.R.size, dtype=np.float64)[:, None, None]
    expected_Z = np.arange(toroidal_grid.Z.size, dtype=np.float64)[None, None, :]
    np.testing.assert_allclose(np.asarray(maps.forward_x), np.broadcast_to(expected_R, maps.shape))
    np.testing.assert_allclose(np.asarray(maps.forward_z), np.broadcast_to(expected_Z, maps.shape))
    np.testing.assert_allclose(np.asarray(maps.backward_x), np.broadcast_to(expected_R, maps.shape))
    np.testing.assert_allclose(np.asarray(maps.backward_z), np.broadcast_to(expected_Z, maps.shape))
    assert not np.any(np.asarray(maps.forward_boundary))
    assert not np.any(np.asarray(maps.backward_boundary))
