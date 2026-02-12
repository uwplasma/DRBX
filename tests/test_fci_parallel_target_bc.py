from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdrb.bc import BC1D
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.fci.parallel import parallel_derivative_target_aware_3d


def _rel_l2(a: jnp.ndarray, b: jnp.ndarray) -> float:
    err = jnp.sqrt(jnp.mean((a - b) ** 2))
    ref = jnp.maximum(jnp.sqrt(jnp.mean(b**2)), 1e-14)
    return float(err / ref)


def _target_aware_error(*, nz: int) -> float:
    nx = 40
    ny = 44
    Lx = 2 * math.pi
    Ly = 2 * math.pi
    Lz = 5.0

    # Straight field lines: mapping is trivial, isolating the plate handling accuracy.
    grid = FCISlabGrid.make(
        nx=nx,
        ny=ny,
        nz=nz,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )
    bc = BC1D.dirichlet(left=0.0, right=0.0, nu=0.0)

    xs = grid.x0 + grid.dx * jnp.arange(grid.nx)
    ys = grid.y0 + grid.dy * jnp.arange(grid.ny)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")

    # Exact solution satisfying Dirichlet plates at z=±Lz/2.
    kx = 2.0
    ky = 3.0
    z = grid.l
    z_left = -0.5 * float(Lz)
    phase_xy = kx * X + ky * Y
    sin_xy = jnp.sin(phase_xy)
    sin_z = jnp.sin(jnp.pi * (z - z_left) / float(Lz))
    cos_z = jnp.cos(jnp.pi * (z - z_left) / float(Lz))
    f = sin_xy[None, :, :] * sin_z[:, None, None]

    dpar_num = parallel_derivative_target_aware_3d(
        f,
        map_fwd=grid.map_fwd,
        map_bwd=grid.map_bwd,
        open_field_line=True,
        bc=bc,
    )
    dpar_exact = sin_xy[None, :, :] * (jnp.pi / float(Lz)) * cos_z[:, None, None]
    return _rel_l2(dpar_num, dpar_exact)


def test_fci_parallel_derivative_target_bc_converges() -> None:
    err_coarse = _target_aware_error(nz=24)
    err_fine = _target_aware_error(nz=48)
    assert err_fine < 0.35 * err_coarse


def test_fci_parallel_derivative_target_bc_is_differentiable() -> None:
    nx = 16
    ny = 18
    nz = 20
    grid = FCISlabGrid.make(
        nx=nx,
        ny=ny,
        nz=nz,
        Lx=2 * math.pi,
        Ly=2 * math.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )
    bc = BC1D.dirichlet(left=0.0, right=0.0, nu=0.0)

    key = jax.random.key(0)
    f0 = jax.random.normal(key, (nz, nx, ny))

    def loss(a: float) -> jnp.ndarray:
        f = a * f0
        dpar = parallel_derivative_target_aware_3d(
            f,
            map_fwd=grid.map_fwd,
            map_bwd=grid.map_bwd,
            open_field_line=True,
            bc=bc,
        )
        return jnp.mean(dpar**2)

    g = jax.grad(loss)(1.0)
    assert bool(jnp.isfinite(g))
