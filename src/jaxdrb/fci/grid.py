from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .map import FCIBilinearMap, SlabFCIConfig, make_slab_fci_map


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
    cell_centered: bool

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
        cell_centered: bool = False,
    ) -> "FCISlabGrid":
        dx = float(Lx) / float(nx)
        dy = float(Ly) / float(ny)
        x0 = 0.0
        y0 = 0.0

        if open_field_line:
            if cell_centered:
                dz = float(Lz) / float(nz)
                l = (-0.5 * float(Lz)) + dz * (0.5 + jnp.arange(nz))
            else:
                l = jnp.linspace(-0.5 * Lz, 0.5 * Lz, nz)
                dz = float(l[1] - l[0]) if nz > 1 else float(Lz)
        else:
            l = jnp.linspace(0.0, Lz, nz, endpoint=False)
            dz = float(Lz) / float(nz)
            cell_centered = False

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

        if open_field_line and cell_centered and nz >= 2:
            # Encode target-hit metadata for the first/last planes:
            # - backward map at k=0 hits the left plate after dl/2,
            # - forward map at k=nz-1 hits the right plate after dl/2.
            hit_fwd = jnp.zeros((nz, nx, ny), dtype=bool)
            hit_bwd = jnp.zeros((nz, nx, ny), dtype=bool)
            hit_fwd = hit_fwd.at[-1].set(True)
            hit_bwd = hit_bwd.at[0].set(True)

            dl_step = map_fwd.dl
            if dl_step.ndim != 2:
                raise ValueError("Slab map dl expected to have shape (nx,ny).")
            dl_half = 0.5 * jnp.broadcast_to(dl_step, (nz, nx, ny))
            dl_hit_fwd = dl_half
            dl_hit_bwd = dl_half
            map_fwd = FCIBilinearMap(
                ix=map_fwd.ix,
                iy=map_fwd.iy,
                w=map_fwd.w,
                dl=map_fwd.dl,
                hit=hit_fwd,
                dl_hit=dl_hit_fwd,
            )
            map_bwd = FCIBilinearMap(
                ix=map_bwd.ix,
                iy=map_bwd.iy,
                w=map_bwd.w,
                dl=map_bwd.dl,
                hit=hit_bwd,
                dl_hit=dl_hit_bwd,
            )

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
            cell_centered=cell_centered,
        )
