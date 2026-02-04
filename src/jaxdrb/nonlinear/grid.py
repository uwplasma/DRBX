from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.bc import BC2D, bc2d_from_strings


class Grid2D(eqx.Module):
    """Uniform 2D grid with optional periodic/Dirichlet/Neumann BCs.

    Notes
    -----
    - For periodic directions, the grid is cell-centered with `endpoint=False` and spacing `L/n`.
    - For non-periodic directions, the grid includes both boundaries with `endpoint=True` and
      spacing `L/(n-1)`. This makes finite-difference boundary stencils consistent with the
      physical domain endpoints.
    - FFT wavenumbers and dealias masks are provided for convenience; they are only valid when
      both directions are periodic.
    """

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
    bc: BC2D = eqx.field(static=True)

    @classmethod
    def make(
        cls,
        *,
        nx: int,
        ny: int,
        Lx: float,
        Ly: float,
        dealias: bool = True,
        bc_x: str = "periodic",
        bc_y: str = "periodic",
        bc_value_x: float = 0.0,
        bc_value_y: float = 0.0,
        bc_grad_x: float = 0.0,
        bc_grad_y: float = 0.0,
    ) -> "Grid2D":
        bc = bc2d_from_strings(
            bc_x=bc_x,  # type: ignore[arg-type]
            bc_y=bc_y,  # type: ignore[arg-type]
            value_x=bc_value_x,
            value_y=bc_value_y,
            grad_x=bc_grad_x,
            grad_y=bc_grad_y,
        )

        dx = Lx / nx if bc.kind_x == 0 else Lx / (nx - 1)
        dy = Ly / ny if bc.kind_y == 0 else Ly / (ny - 1)

        x = jnp.linspace(0.0, Lx, nx, endpoint=(bc.kind_x != 0))
        y = jnp.linspace(0.0, Ly, ny, endpoint=(bc.kind_y != 0))

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
            bc=bc,
        )
