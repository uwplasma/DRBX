from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .map import SlabFCIConfig, make_slab_fci_map


@dataclass(frozen=True)
class FCISlabGrid:
    """Structured slab grid for FCI tests (periodic in x/y, open or periodic in z)."""

    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int

    z0: float
    dz: float
    nz: int

    Bx: float
    By: float
    Bz: float

    l: jnp.ndarray
    map_fwd: object
    map_bwd: object
    sheath_mask: jnp.ndarray
    sheath_sign: jnp.ndarray
    open_field_line: bool

    @classmethod
    def make(
        cls,
        *,
        nx: int,
        ny: int,
        nz: int,
        Lx: float,
        Ly: float,
        Lz: float,
        Bx: float,
        By: float,
        Bz: float,
        open_field_line: bool = True,
    ) -> "FCISlabGrid":
        dx = float(Lx) / float(nx)
        dy = float(Ly) / float(ny)
        x0 = 0.0
        y0 = 0.0

        if open_field_line:
            l = jnp.linspace(-0.5 * Lz, 0.5 * Lz, nz)
            dz = float(l[1] - l[0]) if nz > 1 else float(Lz)
        else:
            l = jnp.linspace(0.0, Lz, nz, endpoint=False)
            dz = float(Lz) / float(nz)

        cfg = SlabFCIConfig(
            x0=x0,
            y0=y0,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
            dz=dz,
            Bx=float(Bx),
            By=float(By),
            Bz=float(Bz),
        )
        map_fwd, map_bwd = make_slab_fci_map(cfg)

        sheath_mask = jnp.zeros((nz, 1, 1))
        sheath_sign = jnp.zeros((nz, 1, 1))
        if open_field_line and nz >= 2:
            sheath_mask = sheath_mask.at[0].set(1.0)
            sheath_mask = sheath_mask.at[-1].set(1.0)
            sheath_sign = sheath_sign.at[0].set(-1.0)
            sheath_sign = sheath_sign.at[-1].set(1.0)

        return cls(
            x0=x0,
            y0=y0,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
            z0=float(l[0]),
            dz=dz,
            nz=nz,
            Bx=float(Bx),
            By=float(By),
            Bz=float(Bz),
            l=l,
            map_fwd=map_fwd,
            map_bwd=map_bwd,
            sheath_mask=sheath_mask,
            sheath_sign=sheath_sign,
            open_field_line=open_field_line,
        )
