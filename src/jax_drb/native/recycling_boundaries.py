from __future__ import annotations

import numpy as np

from .mesh import StructuredMesh
from .open_field import apply_noflow_scalar_guards


def apply_neutral_target_density_guards(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    lower_y: bool,
    upper_y: bool,
) -> np.ndarray:
    """Reconstruct neutral target guard cells from active-domain values.

    The neutral density guard rule is a one-sided linear extrapolation clamped
    at zero. This matches the guarded neutral target treatment used in the
    recycling backbone and is tested directly because it affects both parity and
    compare-window diagnostics.
    """

    result = np.array(field, dtype=np.float64, copy=True)
    if mesh.myg <= 0:
        return result
    if lower_y and mesh.ystart + 1 <= mesh.yend:
        result[:, mesh.ystart - 1, :] = np.maximum(
            2.0 * result[:, mesh.ystart, :] - result[:, mesh.ystart + 1, :],
            0.0,
        )
    if upper_y and mesh.yend - 1 >= mesh.ystart:
        result[:, mesh.yend + 1, :] = np.maximum(
            2.0 * result[:, mesh.yend, :] - result[:, mesh.yend - 1, :],
            0.0,
        )
    return result


def apply_open_field_neumann_scalar_guards(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    lower_y: bool,
    upper_y: bool,
) -> np.ndarray:
    """Apply zero-gradient scalar guards on the open-field target boundaries."""

    return np.asarray(
        apply_noflow_scalar_guards(field, mesh=mesh, lower_y=lower_y, upper_y=upper_y),
        dtype=np.float64,
    )


def apply_open_field_dirichlet_scalar_guards(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    lower_y: bool,
    upper_y: bool,
) -> np.ndarray:
    """Apply odd Dirichlet scalar guards on the open-field target boundaries."""

    result = np.asarray(field, dtype=np.float64, copy=True)
    if mesh.myg <= 0:
        return result
    if lower_y:
        result[:, mesh.ystart - 1, :] = -result[:, mesh.ystart, :]
    if upper_y:
        result[:, mesh.yend + 1, :] = -result[:, mesh.yend, :]
    return result
