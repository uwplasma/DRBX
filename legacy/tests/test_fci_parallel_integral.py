from __future__ import annotations

import math

import jax.numpy as jnp

from jaxdrb.fci.integrate import line_integral_mapped
from jaxdrb.fci.map import SlabFCIConfig, make_slab_fci_map


def _integral_error(*, nx: int, ny: int, nz: int, dz: float) -> float:
    Lx = 2 * math.pi
    Ly = 2 * math.pi
    dx = Lx / nx
    dy = Ly / ny

    cfg = SlabFCIConfig(
        x0=0.0,
        y0=0.0,
        dx=dx,
        dy=dy,
        nx=nx,
        ny=ny,
        dz=dz,
        Bx=0.4,
        By=0.2,
        Bz=1.0,
    )
    fwd, _ = make_slab_fci_map(cfg)

    xs = cfg.x0 + cfg.dx * jnp.arange(cfg.nx)
    ys = cfg.y0 + cfg.dy * jnp.arange(cfg.ny)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")

    kx = 2.0
    ky = 3.0
    kz = -1.0

    z = dz * jnp.arange(nz)
    phase0 = kx * X + ky * Y
    f_planes = jnp.sin(phase0[None, :, :] + kz * z[:, None, None])

    dl0 = float(fwd.dl[0, 0])
    L = dl0 * float(nz - 1)

    B = jnp.array([cfg.Bx, cfg.By, cfg.Bz])
    b = B / jnp.linalg.norm(B)
    alpha = b[0] * kx + b[1] * ky + b[2] * kz
    alpha_safe = jnp.where(jnp.abs(alpha) < 1e-8, 1.0, alpha)
    integral_exact = (jnp.cos(phase0) - jnp.cos(phase0 + alpha * L)) / alpha_safe
    integral_exact = jnp.where(jnp.abs(alpha) < 1e-8, L * jnp.sin(phase0), integral_exact)

    integral_num = line_integral_mapped(f_planes, map_fwd=fwd, dl=fwd.dl, periodic=False)
    rel = jnp.sqrt(jnp.mean((integral_num - integral_exact) ** 2)) / jnp.maximum(
        jnp.sqrt(jnp.mean(integral_exact**2)), 1e-12
    )
    return float(rel)


def test_fci_line_integral_converges_with_refinement() -> None:
    err_coarse = _integral_error(nx=48, ny=48, nz=16, dz=0.4)
    err_fine = _integral_error(nx=96, ny=96, nz=32, dz=0.2)
    assert err_fine < 0.6 * err_coarse
