from __future__ import annotations

import math

import jax.numpy as jnp

from jaxdrb.fci.map import SlabFCIConfig, make_slab_fci_map_variable_B


def _curved_map_error(nx: int, ny: int, dz: float, shear: float) -> float:
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
        Bx=0.0,
        By=0.0,
        Bz=1.0,
    )

    xs = cfg.x0 + cfg.dx * jnp.arange(cfg.nx)
    ys = cfg.y0 + cfg.dy * jnp.arange(cfg.ny)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")

    # Curved/sheared field: Bx varies with y, By=0, Bz=1.
    Bx = shear * (Y - 0.5 * Ly)
    By = jnp.zeros_like(Bx)

    fwd, _ = make_slab_fci_map_variable_B(cfg, Bx=Bx, By=By, Bz=1.0)

    f = jnp.sin(2.0 * X) + 0.3 * jnp.cos(3.0 * Y) + 0.2 * jnp.sin(X + 2.0 * Y)

    shift_x = (Bx / 1.0) * dz
    shift_y = (By / 1.0) * dz
    Xs = jnp.mod(X + shift_x, Lx)
    Ys = jnp.mod(Y + shift_y, Ly)
    f_ref = jnp.sin(2.0 * Xs) + 0.3 * jnp.cos(3.0 * Ys) + 0.2 * jnp.sin(Xs + 2.0 * Ys)
    f_bilin = fwd.apply(f)

    err = jnp.sqrt(jnp.mean((f_bilin - f_ref) ** 2))
    return float(err)


def test_fci_curved_map_refinement_regression() -> None:
    shear = 0.15
    err_coarse = _curved_map_error(48, 48, dz=0.2, shear=shear)
    err_fine = _curved_map_error(96, 96, dz=0.1, shear=shear)
    assert err_fine < 0.45 * err_coarse
