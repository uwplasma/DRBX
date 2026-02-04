from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class Grid2D(eqx.Module):
    """Uniform 2D periodic grid with FFT wavenumbers and dealias mask."""

    nx: int
    ny: int
    Lx: float
    Ly: float

    dx: float = eqx.field(static=True)
    dy: float = eqx.field(static=True)

    x: jnp.ndarray
    y: jnp.ndarray

    kx: jnp.ndarray
    ky: jnp.ndarray
    k2: jnp.ndarray
    dealias_mask: jnp.ndarray

    @classmethod
    def make(cls, *, nx: int, ny: int, Lx: float, Ly: float, dealias: bool = True) -> "Grid2D":
        dx = Lx / nx
        dy = Ly / ny

        x = jnp.linspace(0.0, Lx, nx, endpoint=False)
        y = jnp.linspace(0.0, Ly, ny, endpoint=False)

        kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
        ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
        kx, ky = jnp.meshgrid(kx_1d, ky_1d, indexing="ij")
        k2 = kx**2 + ky**2

        if dealias:
            # 2/3-rule dealiasing mask (Orszag 1971).
            kx_cut = (2.0 / 3.0) * jnp.max(jnp.abs(kx_1d))
            ky_cut = (2.0 / 3.0) * jnp.max(jnp.abs(ky_1d))
            mask = (jnp.abs(kx) <= kx_cut) & (jnp.abs(ky) <= ky_cut)
            dealias_mask = mask.astype(jnp.float32)
        else:
            dealias_mask = jnp.ones((nx, ny), dtype=jnp.float32)

        return cls(
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
            dx=float(dx),
            dy=float(dy),
            x=x,
            y=y,
            kx=kx,
            ky=ky,
            k2=k2,
            dealias_mask=dealias_mask,
        )
