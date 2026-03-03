from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

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
    def from_maps(
        cls,
        *,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        nx: int,
        ny: int,
        l: jnp.ndarray,
        map_fwd,
        map_bwd,
        open_field_line: bool,
        cell_centered: bool,
        Bx: float = 0.0,
        By: float = 0.0,
        Bz: float = 1.0,
        sheath_mask: jnp.ndarray | None = None,
        sheath_sign: jnp.ndarray | None = None,
    ) -> "FCISlabGrid":
        """Build an FCI slab-grid container from externally computed maps."""

        l_arr = jnp.asarray(l, dtype=jnp.float64)
        if l_arr.ndim != 1:
            raise ValueError("FCISlabGrid.from_maps requires 1D l coordinate.")
        nz = int(l_arr.size)
        dz = float(jnp.mean(jnp.diff(l_arr))) if nz > 1 else 1.0
        shape = (nz, int(nx), int(ny))

        if sheath_mask is None or sheath_sign is None:
            hit_fwd = getattr(map_fwd, "hit", None)
            hit_bwd = getattr(map_bwd, "hit", None)
            if hit_fwd is not None and hit_bwd is not None:
                hf = jnp.asarray(hit_fwd, dtype=jnp.float64)
                hb = jnp.asarray(hit_bwd, dtype=jnp.float64)
                if hf.ndim == 2:
                    hf = hf[None, ...]
                if hb.ndim == 2:
                    hb = hb[None, ...]
                hf = jnp.broadcast_to(hf, shape)
                hb = jnp.broadcast_to(hb, shape)
                sheath_mask = jnp.clip(hf + hb, 0.0, 1.0)
                sheath_sign = hf - hb
            else:
                sheath_mask = jnp.zeros(shape, dtype=jnp.float64)
                sheath_sign = jnp.zeros(shape, dtype=jnp.float64)
                if open_field_line and nz >= 2:
                    sheath_mask = sheath_mask.at[0].set(1.0)
                    sheath_mask = sheath_mask.at[-1].set(1.0)
                    sheath_sign = sheath_sign.at[0].set(-1.0)
                    sheath_sign = sheath_sign.at[-1].set(1.0)
        else:
            sheath_mask = jnp.asarray(sheath_mask, dtype=jnp.float64)
            sheath_sign = jnp.asarray(sheath_sign, dtype=jnp.float64)
            sheath_mask = jnp.broadcast_to(sheath_mask, shape)
            sheath_sign = jnp.broadcast_to(sheath_sign, shape)

        return cls(
            x0=float(x0),
            y0=float(y0),
            dx=float(dx),
            dy=float(dy),
            nx=int(nx),
            ny=int(ny),
            z0=float(l_arr[0]),
            dz=dz,
            nz=nz,
            Bx=float(Bx),
            By=float(By),
            Bz=float(Bz),
            l=l_arr,
            map_fwd=map_fwd,
            map_bwd=map_bwd,
            sheath_mask=sheath_mask,
            sheath_sign=sheath_sign,
            open_field_line=bool(open_field_line),
            cell_centered=bool(cell_centered),
        )

    @classmethod
    def from_npz(
        cls,
        *,
        path: str,
        open_field_line: bool | None = None,
        cell_centered: bool | None = None,
        Bx: float | None = None,
        By: float | None = None,
        Bz: float | None = None,
    ) -> "FCISlabGrid":
        data = np.load(path)

        def get_required(name: str):
            if name not in data:
                raise KeyError(f"Missing '{name}' in FCI map file.")
            return np.asarray(data[name])

        def get_optional(name: str, default=None):
            if name in data:
                return np.asarray(data[name])
            return default

        def to_scalar(arr, name: str) -> float:
            arr = np.asarray(arr)
            if arr.ndim == 0:
                return float(arr)
            if arr.size == 1:
                return float(arr.ravel()[0])
            raise ValueError(f"Expected scalar for {name}, got shape {arr.shape}.")

        x0 = to_scalar(get_required("x0"), "x0")
        y0 = to_scalar(get_required("y0"), "y0")
        dx = to_scalar(get_required("dx"), "dx")
        dy = to_scalar(get_required("dy"), "dy")
        nx = int(to_scalar(get_required("nx"), "nx"))
        ny = int(to_scalar(get_required("ny"), "ny"))

        l = get_optional("l", None)
        if l is None:
            l = get_optional("z", None)
        if l is None:
            raise KeyError("Missing 'l' or 'z' coordinate in FCI map file.")
        l = np.asarray(l)

        open_field_line = (
            bool(open_field_line)
            if open_field_line is not None
            else bool(to_scalar(get_optional("open_field_line", 0.0), "open_field_line"))
        )
        cell_centered = (
            bool(cell_centered)
            if cell_centered is not None
            else bool(to_scalar(get_optional("cell_centered", 0.0), "cell_centered"))
        )

        if Bx is None:
            Bx = float(to_scalar(get_optional("Bx", 0.0), "Bx"))
        if By is None:
            By = float(to_scalar(get_optional("By", 0.0), "By"))
        if Bz is None:
            Bz = float(to_scalar(get_optional("Bz", 1.0), "Bz"))

        def map_from_prefix(prefix: str) -> FCIBilinearMap:
            ix = get_required(f"{prefix}_ix").astype(np.int32)
            iy = get_required(f"{prefix}_iy").astype(np.int32)
            w = get_required(f"{prefix}_w")
            dl = get_required(f"{prefix}_dl")
            hit = get_optional(f"{prefix}_hit", None)
            dl_hit = get_optional(f"{prefix}_dl_hit", None)
            hit_R = get_optional(f"{prefix}_hit_R", None)
            hit_Z = get_optional(f"{prefix}_hit_Z", None)
            hit_phi = get_optional(f"{prefix}_hit_phi", None)
            hit_target = get_optional(f"{prefix}_hit_target", None)
            return FCIBilinearMap(
                ix=jnp.asarray(ix),
                iy=jnp.asarray(iy),
                w=jnp.asarray(w),
                dl=jnp.asarray(dl),
                hit=None if hit is None else jnp.asarray(hit),
                dl_hit=None if dl_hit is None else jnp.asarray(dl_hit),
                hit_R=None if hit_R is None else jnp.asarray(hit_R),
                hit_Z=None if hit_Z is None else jnp.asarray(hit_Z),
                hit_phi=None if hit_phi is None else jnp.asarray(hit_phi),
                hit_target=None if hit_target is None else jnp.asarray(hit_target),
            )

        map_fwd = map_from_prefix("map_fwd")
        map_bwd = map_from_prefix("map_bwd")

        sheath_mask = get_optional("sheath_mask", None)
        sheath_sign = get_optional("sheath_sign", None)

        return cls.from_maps(
            x0=x0,
            y0=y0,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
            l=jnp.asarray(l),
            map_fwd=map_fwd,
            map_bwd=map_bwd,
            open_field_line=open_field_line,
            cell_centered=cell_centered,
            Bx=Bx,
            By=By,
            Bz=Bz,
            sheath_mask=None if sheath_mask is None else jnp.asarray(sheath_mask),
            sheath_sign=None if sheath_sign is None else jnp.asarray(sheath_sign),
        )

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
