from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..config.boutinp import BoutConfig
from ..runtime import runtime_jax_dtype
from ..runtime.run_config import RunConfiguration


@dataclass(frozen=True)
class StructuredMesh:
    nx: int
    ny: int
    nz: int
    mxg: int
    myg: int
    symmetric_global_x: bool
    symmetric_global_y: bool
    jyseps1_1: int
    jyseps2_1: int
    jyseps1_2: int
    jyseps2_2: int
    ny_inner: int
    has_lower_y_target: bool
    has_upper_y_target: bool
    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray

    @property
    def xstart(self) -> int:
        return self.mxg

    @property
    def xend(self) -> int:
        return self.nx - self.mxg - 1

    @property
    def ystart(self) -> int:
        return self.myg

    @property
    def yend(self) -> int:
        return self.myg + self.ny - 1

    @property
    def local_ny(self) -> int:
        return self.ny + 2 * self.myg

    def expression_context(self, *, time: float = 0.0) -> dict[str, jnp.ndarray]:
        dtype = runtime_jax_dtype()
        x3 = self.x[:, None, None]
        y3 = self.y[None, :, None]
        z3 = self.z[None, None, :]
        return {
            "x": x3,
            "y": 2.0 * jnp.pi * y3,
            "z": 2.0 * jnp.pi * z3,
            "t": jnp.array(time, dtype=dtype),
        }


def build_structured_mesh(config: BoutConfig, run_config: RunConfiguration) -> StructuredMesh:
    if run_config.mesh.nx is None or run_config.mesh.ny is None or run_config.mesh.nz is None:
        raise ValueError("Structured native execution requires explicit mesh nx, ny, and nz.")

    nx = run_config.mesh.nx
    ny = run_config.mesh.ny
    nz = run_config.mesh.nz
    mxg = run_config.mesh.mxg
    myg = run_config.mesh.myg
    symmetric_global_x = _mesh_bool(config, "symmetricGlobalX", default=True)
    symmetric_global_y = _mesh_bool(config, "symmetricGlobalY", default=True)
    jyseps1_1 = _mesh_int(config, "jyseps1_1", default=-1)
    jyseps2_1 = _mesh_int(config, "jyseps2_1", default=ny // 2)
    jyseps1_2 = _mesh_int(config, "jyseps1_2", default=ny // 2)
    jyseps2_2 = _mesh_int(config, "jyseps2_2", default=ny - 1)
    ny_inner = _mesh_int(config, "ny_inner", default=jyseps2_1)

    x = _global_x_coordinates(nx=nx, mxg=mxg, symmetric=symmetric_global_x)
    y = _global_y_coordinates(
        ny=ny,
        myg=myg,
        symmetric=symmetric_global_y,
        jyseps1_1=jyseps1_1,
        jyseps2_1=jyseps2_1,
        jyseps1_2=jyseps1_2,
        jyseps2_2=jyseps2_2,
        ny_inner=ny_inner,
    )
    dtype = runtime_jax_dtype()
    z = jnp.arange(nz, dtype=dtype) / float(nz)
    return StructuredMesh(
        nx=nx,
        ny=ny,
        nz=nz,
        mxg=mxg,
        myg=myg,
        symmetric_global_x=symmetric_global_x,
        symmetric_global_y=symmetric_global_y,
        jyseps1_1=jyseps1_1,
        jyseps2_1=jyseps2_1,
        jyseps1_2=jyseps1_2,
        jyseps2_2=jyseps2_2,
        ny_inner=ny_inner,
        has_lower_y_target=False,
        has_upper_y_target=True,
        x=x,
        y=y,
        z=z,
    )


def broadcast_to_field_shape(value: jnp.ndarray, mesh: StructuredMesh) -> jnp.ndarray:
    dtype = runtime_jax_dtype()
    array = jnp.asarray(value, dtype=dtype)
    target_shape = (mesh.nx, mesh.local_ny, mesh.nz)
    if array.shape == target_shape:
        return array
    return jnp.broadcast_to(array, target_shape)


def apply_zero_dirichlet_x_guards(field: jnp.ndarray, mesh: StructuredMesh) -> jnp.ndarray:
    dtype = runtime_jax_dtype()
    result = jnp.asarray(field, dtype=dtype)
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    if mesh.mxg <= 0:
        return result

    result = result.at[mesh.xstart - 1, y_slice, :].set(-result[mesh.xstart, y_slice, :])
    result = result.at[mesh.xend + 1, y_slice, :].set(-result[mesh.xend, y_slice, :])
    for offset in range(2, mesh.mxg + 1):
        result = result.at[mesh.xstart - offset, y_slice, :].set(0.0)
        result = result.at[mesh.xend + offset, y_slice, :].set(0.0)
    return result


def apply_neumann_x_guards(field: jnp.ndarray, mesh: StructuredMesh) -> jnp.ndarray:
    dtype = runtime_jax_dtype()
    result = jnp.asarray(field, dtype=dtype)
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    if mesh.mxg <= 0:
        return result

    for offset in range(1, mesh.mxg + 1):
        result = result.at[mesh.xstart - offset, y_slice, :].set(result[mesh.xstart - 1 + offset, y_slice, :])
        result = result.at[mesh.xend + offset, y_slice, :].set(result[mesh.xend + 1 - offset, y_slice, :])
    return result


def communicate_y_guards(field: jnp.ndarray, mesh: StructuredMesh) -> jnp.ndarray:
    dtype = runtime_jax_dtype()
    result = jnp.asarray(field, dtype=dtype)
    for offset in range(mesh.myg):
        result = result.at[:, mesh.ystart - 1 - offset, :].set(result[:, mesh.ystart + offset, :])
        result = result.at[:, mesh.yend + 1 + offset, :].set(result[:, mesh.yend - offset, :])
    return result


def project_nonnegative_x_boundaries(field: jnp.ndarray, mesh: StructuredMesh) -> jnp.ndarray:
    dtype = runtime_jax_dtype()
    result = jnp.asarray(field, dtype=dtype)
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    if mesh.mxg > 0:
        left = result[:mesh.xstart, y_slice, :]
        right = result[mesh.xend + 1 :, y_slice, :]
        result = result.at[:mesh.xstart, y_slice, :].set(jnp.maximum(left, 0.0))
        result = result.at[mesh.xend + 1 :, y_slice, :].set(jnp.maximum(right, 0.0))
    return result


def apply_x_boundary(field: jnp.ndarray, mesh: StructuredMesh, boundary_kind: str) -> jnp.ndarray:
    normalized = boundary_kind.strip().lower()
    if normalized == "neumann":
        return apply_neumann_x_guards(field, mesh)
    if normalized in {"dirichlet", "dirichlet_zero", "zero"}:
        return apply_zero_dirichlet_x_guards(field, mesh)
    raise NotImplementedError(f"Unsupported X boundary kind {boundary_kind!r}")


def apply_field_boundaries(field: jnp.ndarray, mesh: StructuredMesh, *, x_boundary: str) -> jnp.ndarray:
    result = apply_x_boundary(field, mesh, x_boundary)
    result = communicate_y_guards(result, mesh)
    if x_boundary.strip().lower() in {"dirichlet", "dirichlet_zero", "zero"}:
        result = project_nonnegative_x_boundaries(result, mesh)
    return result


def _global_x_coordinates(*, nx: int, mxg: int, symmetric: bool) -> jnp.ndarray:
    dtype = runtime_jax_dtype()
    mx = nx - 2 * mxg
    global_indices = jnp.arange(nx, dtype=dtype)
    if symmetric:
        return (0.5 + global_indices - (nx - mx) * 0.5) / float(mx)
    return global_indices / float(mx)


def _global_y_coordinates(
    *,
    ny: int,
    myg: int,
    symmetric: bool,
    jyseps1_1: int,
    jyseps2_1: int,
    jyseps1_2: int,
    jyseps2_2: int,
    ny_inner: int,
) -> jnp.ndarray:
    dtype = runtime_jax_dtype()
    local_indices = jnp.arange(ny + 2 * myg, dtype=dtype) - float(myg)
    nycore = float((jyseps2_1 - jyseps1_1) + (jyseps2_2 - jyseps1_2))
    if symmetric:
        before = (local_indices - (jyseps1_1 + 0.5)) / nycore
        after = (local_indices - (jyseps1_1 + 0.5 + (jyseps1_2 - jyseps2_1))) / nycore
        return jnp.where(local_indices < float(ny_inner), before, after)

    core_indices = local_indices
    lower = core_indices - (jyseps1_1 + 1.0)
    upper = core_indices - (jyseps1_1 + 1.0 + (jyseps1_2 - jyseps2_1))
    return jnp.where(core_indices <= float(jyseps2_1), lower / nycore, upper / nycore)


def _mesh_bool(config: BoutConfig, key: str, *, default: bool) -> bool:
    if not config.has_option("mesh", key):
        return default
    return bool(config.parsed("mesh", key))


def _mesh_int(config: BoutConfig, key: str, *, default: int) -> int:
    if not config.has_option("mesh", key):
        return default
    value = config.parsed("mesh", key)
    return int(round(float(value)))
