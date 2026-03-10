from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.operators.fd2d import (
    biharmonic as biharmonic_fd,
    ddx as ddx_fd,
    ddy as ddy_fd,
    laplacian as laplacian_fd,
)
from jaxdrb.operators.spectral2d import (
    biharmonic as biharmonic_spec,
    ddx as ddx_spec,
    ddy as ddy_spec,
    laplacian as laplacian_spec,
)


def grid_of(geom: GeometryAdapter):
    grid = getattr(geom, "grid", None)
    if grid is None:
        return None
    if getattr(geom, "ndim", None) == 2:
        return grid
    return None


def is_2d(geom: GeometryAdapter) -> bool:
    grid = grid_of(geom)
    return grid is not None and getattr(geom, "ndim", None) == 2


def is_periodic_bc(bc: BC2D, geom: GeometryAdapter | None = None) -> bool:
    if bc.kind_x != 0 or bc.kind_y != 0:
        return False
    if geom is None:
        return True
    grid = grid_of(geom)
    if grid is None:
        return True
    return grid.bc.kind_x == 0 and grid.bc.kind_y == 0


def ddx(params: DRBSystemParams, geom: GeometryAdapter, f: jnp.ndarray, bc: BC2D) -> jnp.ndarray:
    grid = grid_of(geom)
    if grid is None:
        return geom.ddx(f)
    if is_periodic_bc(bc, geom) and params.poisson == "spectral":
        return ddx_spec(f, grid.kx)
    return ddx_fd(f, grid.dx, bc)


def ddy(params: DRBSystemParams, geom: GeometryAdapter, f: jnp.ndarray, bc: BC2D) -> jnp.ndarray:
    grid = grid_of(geom)
    if grid is None:
        return geom.ddy(f)
    if is_periodic_bc(bc, geom) and params.poisson == "spectral":
        return ddy_spec(f, grid.ky)
    return ddy_fd(f, grid.dy, bc)


def laplacian(
    params: DRBSystemParams, geom: GeometryAdapter, f: jnp.ndarray, bc: BC2D
) -> jnp.ndarray:
    grid = grid_of(geom)
    if grid is None:
        return geom.laplacian(f)
    if is_periodic_bc(bc, geom) and params.poisson == "spectral":
        return laplacian_spec(f, grid.k2)
    return laplacian_fd(f, grid.dx, grid.dy, bc)


def biharmonic(
    params: DRBSystemParams, geom: GeometryAdapter, f: jnp.ndarray, bc: BC2D
) -> jnp.ndarray:
    grid = grid_of(geom)
    if grid is None:
        return geom.biharmonic(f)
    if is_periodic_bc(bc, geom) and params.poisson == "spectral":
        return biharmonic_spec(f, grid.k2)
    return biharmonic_fd(f, grid.dx, grid.dy, bc)


def _broadcast_mask(mask: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
    if mask.shape == shape:
        return mask
    if mask.ndim == 1:
        if len(shape) == 3 and mask.shape[0] == shape[0]:
            mask = mask[:, None, None]
        elif len(shape) == 2:
            if mask.shape[0] == shape[0]:
                mask = mask[:, None]
            elif mask.shape[0] == shape[1]:
                mask = mask[None, :]
    elif mask.ndim == 2 and len(shape) == 3 and mask.shape == shape[1:]:
        mask = mask[None, :, :]
    return jnp.broadcast_to(mask, shape)


def region_mask(geom: GeometryAdapter, name: str, shape: tuple[int, ...]) -> jnp.ndarray | None:
    grid = getattr(geom, "grid", None)
    masks = None
    if grid is not None:
        masks = getattr(grid, "region_masks", None)
    if masks is None:
        masks = getattr(geom, "region_masks", None)
    if not masks or name not in masks:
        return None
    mask = jnp.asarray(masks[name], dtype=jnp.float64)
    return _broadcast_mask(mask, shape)
