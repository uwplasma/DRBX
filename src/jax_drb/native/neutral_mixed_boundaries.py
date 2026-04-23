from __future__ import annotations

import numpy as np

from .mesh import StructuredMesh


def apply_neumann_x_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    for offset in range(1, mesh.mxg + 1):
        result[mesh.xstart - offset, y_slice, :] = result[mesh.xstart - 1 + offset, y_slice, :]
        result[mesh.xend + offset, y_slice, :] = result[mesh.xend + 1 - offset, y_slice, :]
    return result


def apply_dirichlet_x_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    result[mesh.xstart - 1, y_slice, :] = -result[mesh.xstart, y_slice, :]
    result[mesh.xend + 1, y_slice, :] = -result[mesh.xend, y_slice, :]
    for offset in range(2, mesh.mxg + 1):
        result[mesh.xstart - offset, y_slice, :] = 0.0
        result[mesh.xend + offset, y_slice, :] = 0.0
    return result


def apply_density_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = apply_neumann_x_boundaries(field, mesh)
    return apply_density_y_boundaries(result, mesh)


def apply_pressure_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = apply_neumann_x_boundaries(field, mesh)
    return apply_zero_gradient_y_boundaries(result, mesh)


def apply_temperature_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = apply_neumann_x_boundaries(field, mesh)
    return apply_zero_gradient_y_boundaries(result, mesh)


def apply_diffusion_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = apply_dirichlet_x_boundaries(field, mesh)
    return apply_antisymmetric_y_boundaries(result, mesh)


def apply_momentum_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = apply_dirichlet_x_boundaries(field, mesh)
    return apply_antisymmetric_y_boundaries(result, mesh)


def apply_velocity_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = apply_dirichlet_x_boundaries(field, mesh)
    return apply_antisymmetric_y_boundaries(result, mesh)


def apply_zero_gradient_y_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    for offset in range(1, mesh.myg + 1):
        result[:, mesh.ystart - offset, :] = result[:, mesh.ystart - 1 + offset, :]
        result[:, mesh.yend + offset, :] = result[:, mesh.yend + 1 - offset, :]
    return result


def apply_density_y_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    for offset in range(1, mesh.myg + 1):
        lower_wall = np.maximum(
            0.5 * (3.0 * result[:, mesh.ystart, :] - result[:, mesh.ystart + 1, :]),
            0.0,
        )
        upper_wall = np.maximum(
            0.5 * (3.0 * result[:, mesh.yend, :] - result[:, mesh.yend - 1, :]),
            0.0,
        )
        result[:, mesh.ystart - offset, :] = 2.0 * lower_wall - result[:, mesh.ystart - offset + 1, :]
        result[:, mesh.yend + offset, :] = 2.0 * upper_wall - result[:, mesh.yend + offset - 1, :]
    return result


def soft_floor(field: np.ndarray, minimum: float) -> np.ndarray:
    if minimum <= 0.0:
        raise ValueError("soft floor minimum must be positive")
    values = np.maximum(np.asarray(field, dtype=np.float64), 0.0)
    return values + float(minimum) * np.exp(-values / float(minimum))


def apply_antisymmetric_y_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    for offset in range(1, mesh.myg + 1):
        result[:, mesh.ystart - offset, :] = -result[:, mesh.ystart - 1 + offset, :]
        result[:, mesh.yend + offset, :] = -result[:, mesh.yend + 1 - offset, :]
    return result
