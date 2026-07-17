from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset


@dataclass(frozen=True)
class FciMaps:
    """Forward/backward traced-field-line maps in logical index coordinates."""

    forward_x: jnp.ndarray
    forward_z: jnp.ndarray
    backward_x: jnp.ndarray
    backward_z: jnp.ndarray
    forward_boundary: jnp.ndarray
    backward_boundary: jnp.ndarray
    dphi: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.forward_x.shape)


def identity_fci_maps(*, nx: int, ny: int, nz: int, dphi: float = 1.0) -> FciMaps:
    """Build maps whose field lines connect to the same `(x,z)` point."""

    x = jnp.arange(nx, dtype=jnp.float64)[:, None, None]
    z = jnp.arange(nz, dtype=jnp.float64)[None, None, :]
    forward_x = jnp.broadcast_to(x, (nx, ny, nz))
    forward_z = jnp.broadcast_to(z, (nx, ny, nz))
    boundary = jnp.zeros((nx, ny, nz), dtype=bool)
    return FciMaps(
        forward_x=forward_x,
        forward_z=forward_z,
        backward_x=forward_x,
        backward_z=forward_z,
        forward_boundary=boundary,
        backward_boundary=boundary,
        dphi=float(dphi),
    )


def load_fci_maps_netcdf(path: str | Path) -> FciMaps:
    """Load a compact FCI map bundle from a NetCDF grid file."""

    resolved = Path(path)
    with Dataset(resolved) as dataset:
        forward_x = _read_variable(dataset, ("forward_xt_prime", "forward_x", "forward_R_index"))
        forward_z = _read_variable(dataset, ("forward_zt_prime", "forward_z", "forward_Z_index"))
        backward_x = _read_variable(dataset, ("backward_xt_prime", "backward_x", "backward_R_index"))
        backward_z = _read_variable(dataset, ("backward_zt_prime", "backward_z", "backward_Z_index"))
        dphi = _infer_dphi(dataset, forward_x.shape[1])

    forward_boundary = (forward_x < 0.0) | (forward_x > float(forward_x.shape[0] - 1))
    backward_boundary = (backward_x < 0.0) | (backward_x > float(backward_x.shape[0] - 1))
    return FciMaps(
        forward_x=jnp.asarray(forward_x, dtype=jnp.float64),
        forward_z=jnp.asarray(forward_z, dtype=jnp.float64),
        backward_x=jnp.asarray(backward_x, dtype=jnp.float64),
        backward_z=jnp.asarray(backward_z, dtype=jnp.float64),
        forward_boundary=jnp.asarray(forward_boundary),
        backward_boundary=jnp.asarray(backward_boundary),
        dphi=float(dphi),
    )


def _read_variable(dataset: Dataset, candidates: tuple[str, ...]) -> np.ndarray:
    for name in candidates:
        if name in dataset.variables:
            return np.asarray(dataset.variables[name][:], dtype=np.float64)
    raise KeyError(f"None of the FCI map variables were found: {', '.join(candidates)}")


def _infer_dphi(dataset: Dataset, ny: int) -> float:
    if "dy" in dataset.variables:
        dy = np.asarray(dataset.variables["dy"][:], dtype=np.float64)
        if dy.size:
            return float(np.nanmean(dy))
    if "yperiod" in dataset.ncattrs():
        return float(dataset.getncattr("yperiod")) / float(ny)
    return float(2.0 * np.pi / max(ny, 1))
