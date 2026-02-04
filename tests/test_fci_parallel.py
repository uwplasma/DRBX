from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdrb.fci.map import SlabFCIConfig, make_slab_fci_map
from jaxdrb.fci.parallel import parallel_derivative_centered


def test_fci_bilinear_map_matches_fft_shift() -> None:
    """Bilinear periodic interpolation should reproduce a smooth continuous shift accurately.

    Note: for rough/random fields with significant power near Nyquist, bilinear interpolation
    can have O(1) errors. Here we use a smooth low-mode field as a meaningful verification.
    """

    nx = 96
    ny = 80
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
        dz=0.2,
        Bx=0.7,
        By=-0.3,
        Bz=1.0,
    )
    fwd, _ = make_slab_fci_map(cfg)

    xs = cfg.x0 + cfg.dx * jnp.arange(cfg.nx)
    ys = cfg.y0 + cfg.dy * jnp.arange(cfg.ny)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    f = jnp.sin(2.0 * X) + 0.3 * jnp.cos(3.0 * Y) + 0.2 * jnp.sin(X + 2.0 * Y)

    shift_x = (cfg.Bx / cfg.Bz) * cfg.dz
    shift_y = (cfg.By / cfg.Bz) * cfg.dz
    # Reference: evaluate the smooth function at shifted coordinates.
    Xs = jnp.mod(X + shift_x, Lx)
    Ys = jnp.mod(Y + shift_y, Ly)
    f_ref = jnp.sin(2.0 * Xs) + 0.3 * jnp.cos(3.0 * Ys) + 0.2 * jnp.sin(Xs + 2.0 * Ys)
    f_bilin = fwd.apply(f)

    err = jnp.sqrt(jnp.mean((f_bilin - f_ref) ** 2))
    # Bilinear is 2nd order; on this grid the error should be small.
    assert float(err) < 2e-3


def test_fci_parallel_derivative_matches_analytic_constant_B() -> None:
    """FCI centered derivative matches b·∇ for a smooth periodic function (constant B)."""

    nx = 96
    ny = 96
    Lx = 2 * math.pi
    Ly = 2 * math.pi
    dx = Lx / nx
    dy = Ly / ny

    # Use a non-grid-aligned shift.
    cfg = SlabFCIConfig(
        x0=0.0,
        y0=0.0,
        dx=dx,
        dy=dy,
        nx=nx,
        ny=ny,
        dz=0.15,
        Bx=0.4,
        By=0.2,
        Bz=1.0,
    )
    fwd, bwd = make_slab_fci_map(cfg)

    xs = cfg.x0 + cfg.dx * jnp.arange(cfg.nx)
    ys = cfg.y0 + cfg.dy * jnp.arange(cfg.ny)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")

    kx = 2.0
    ky = 3.0
    kz = -1.0

    def f(x, y, z):
        return jnp.sin(kx * x + ky * y + kz * z)

    f_k = f(X, Y, 0.0)
    f_kp1 = f(X, Y, cfg.dz)
    f_km1 = f(X, Y, -cfg.dz)

    dpar_num = parallel_derivative_centered(
        f_k,
        f_kp1=f_kp1,
        f_km1=f_km1,
        map_fwd=fwd,
        map_bwd=bwd,
    )

    # Analytic b·∇f at z=0.
    B = jnp.array([cfg.Bx, cfg.By, cfg.Bz])
    b = B / jnp.linalg.norm(B)
    phase = kx * X + ky * Y
    dpar_exact = (b[0] * kx + b[1] * ky + b[2] * kz) * jnp.cos(phase)

    rel = jnp.sqrt(jnp.mean((dpar_num - dpar_exact) ** 2)) / jnp.maximum(
        jnp.sqrt(jnp.mean(dpar_exact**2)), 1e-12
    )
    assert float(rel) < 2e-2


def test_fci_operator_is_differentiable_wrt_field_values() -> None:
    """Gradients should flow through the mapping and centered difference."""

    nx = 32
    ny = 32
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
        dz=0.3,
        Bx=0.25,
        By=-0.1,
        Bz=1.0,
    )
    fwd, bwd = make_slab_fci_map(cfg)

    key = jax.random.key(1)
    f0 = jax.random.normal(key, (nx, ny))

    def loss(a: float) -> jnp.ndarray:
        f_k = a * f0
        f_kp1 = 1.1 * a * f0
        f_km1 = 0.9 * a * f0
        dpar = parallel_derivative_centered(f_k, f_kp1=f_kp1, f_km1=f_km1, map_fwd=fwd, map_bwd=bwd)
        return jnp.mean(dpar**2)

    g = jax.grad(loss)(1.0)
    assert bool(jnp.isfinite(g))
