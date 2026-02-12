from __future__ import annotations

import math

import jax.numpy as jnp

from jaxdrb.fci.builder import ZPlaneFCIConfig, build_fci_maps_zplanes
from jaxdrb.fci.map import SlabFCIConfig, make_slab_fci_map


def test_fci_zplane_builder_matches_slab_constant_B() -> None:
    nx = 64
    ny = 56
    nz = 12
    Lx = 2 * math.pi
    Ly = 2 * math.pi
    dx = Lx / nx
    dy = Ly / ny
    dz = 0.25

    Bx = 0.35
    By = -0.2
    Bz = 1.0

    slab_cfg = SlabFCIConfig(
        x0=0.0,
        y0=0.0,
        dx=dx,
        dy=dy,
        nx=nx,
        ny=ny,
        dz=dz,
        Bx=Bx,
        By=By,
        Bz=Bz,
    )
    fwd_ref, bwd_ref = make_slab_fci_map(slab_cfg)

    z_cfg = ZPlaneFCIConfig(
        x0=0.0,
        y0=0.0,
        dx=dx,
        dy=dy,
        nx=nx,
        ny=ny,
        z0=0.0,
        dz=dz,
        nz=nz,
        periodic_z=False,
        open_field_line=False,
        cell_centered=False,
    )

    def B(points: jnp.ndarray) -> jnp.ndarray:
        return jnp.broadcast_to(jnp.array([Bx, By, Bz]), points.shape)

    fwd, bwd = build_fci_maps_zplanes(z_cfg, B=B, nsub=4)

    xs = slab_cfg.x0 + slab_cfg.dx * jnp.arange(slab_cfg.nx)
    ys = slab_cfg.y0 + slab_cfg.dy * jnp.arange(slab_cfg.ny)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    f = jnp.sin(2.0 * X) + 0.3 * jnp.cos(3.0 * Y) + 0.2 * jnp.sin(X + 2.0 * Y)

    f_stack = jnp.broadcast_to(f, (nz, nx, ny))
    mapped_fwd = fwd.apply(f_stack)
    mapped_bwd = bwd.apply(f_stack)

    err_fwd = jnp.sqrt(jnp.mean((mapped_fwd[0] - fwd_ref.apply(f)) ** 2))
    err_bwd = jnp.sqrt(jnp.mean((mapped_bwd[0] - bwd_ref.apply(f)) ** 2))
    assert float(err_fwd) < 2e-3
    assert float(err_bwd) < 2e-3

    dl_ref = float(fwd_ref.dl[0, 0])
    assert float(jnp.max(jnp.abs(fwd.dl - dl_ref))) < 5e-7
    assert float(jnp.max(jnp.abs(bwd.dl - dl_ref))) < 5e-7
