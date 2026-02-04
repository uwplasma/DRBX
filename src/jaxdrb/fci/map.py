from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp


class FCIBilinearMap(eqx.Module):
    """Bilinear FCI map for a single step to the next/previous plane.

    The map is defined on a *structured* perpendicular grid with periodic indexing:

      f(x+, y+) ≈ Σ w_mn f[i_m, j_n],

    where (i_m, j_n) are the four cell corners surrounding the mapped footpoint.

    Attributes
    ----------
    ix, iy:
        Integer corner indices with shape (..., 4) where the last axis enumerates the
        four bilinear corners in a fixed order.
    w:
        Corresponding bilinear weights with shape (..., 4).
    dl:
        Distance along the field line between the two planes (can be uniform).
    """

    ix: jnp.ndarray
    iy: jnp.ndarray
    w: jnp.ndarray
    dl: jnp.ndarray

    def apply(self, f_plane: jnp.ndarray) -> jnp.ndarray:
        """Interpolate a scalar field defined on a single perpendicular plane."""

        # f_plane: (nx, ny)
        # ix/iy: (..., 4)
        vals = f_plane[self.ix, self.iy]
        return jnp.sum(self.w * vals, axis=-1)


def _bilinear_weights_periodic(
    *,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return periodic bilinear stencil indices and weights for points (x,y)."""

    # Convert to cell coordinates.
    fx = (x - x0) / dx
    fy = (y - y0) / dy

    i0 = jnp.floor(fx).astype(jnp.int32)
    j0 = jnp.floor(fy).astype(jnp.int32)

    tx = fx - i0
    ty = fy - j0

    i1 = i0 + 1
    j1 = j0 + 1

    # Periodic wrap.
    i0 = jnp.mod(i0, nx)
    i1 = jnp.mod(i1, nx)
    j0 = jnp.mod(j0, ny)
    j1 = jnp.mod(j1, ny)

    # Corner order: (i0,j0), (i1,j0), (i0,j1), (i1,j1)
    ix = jnp.stack([i0, i1, i0, i1], axis=-1)
    iy = jnp.stack([j0, j0, j1, j1], axis=-1)
    w = jnp.stack(
        [
            (1 - tx) * (1 - ty),
            tx * (1 - ty),
            (1 - tx) * ty,
            tx * ty,
        ],
        axis=-1,
    )
    return ix, iy, w


@dataclass(frozen=True)
class SlabFCIConfig:
    """Analytic slab configuration for generating an FCI map."""

    # Plane geometry.
    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int

    # Parallel step size (between planes).
    dz: float

    # Constant magnetic-field direction B = (Bx, By, Bz).
    Bx: float
    By: float
    Bz: float


def make_slab_fci_map(cfg: SlabFCIConfig) -> tuple[FCIBilinearMap, FCIBilinearMap]:
    """Make (forward, backward) slab FCI maps for constant B on periodic planes.

    For constant B, field lines are straight:

      x(z+dz) = x + (Bx/Bz) dz,
      y(z+dz) = y + (By/Bz) dz,

    so the map is exact up to the interpolation/FD approximation.
    """

    if cfg.Bz == 0.0:
        raise ValueError("FCI slab map requires Bz != 0.")

    xs = cfg.x0 + cfg.dx * jnp.arange(cfg.nx)
    ys = cfg.y0 + cfg.dy * jnp.arange(cfg.ny)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")

    shift_x = (cfg.Bx / cfg.Bz) * cfg.dz
    shift_y = (cfg.By / cfg.Bz) * cfg.dz

    Xp = X + shift_x
    Yp = Y + shift_y
    Xm = X - shift_x
    Ym = Y - shift_y

    ixp, iyp, wp = _bilinear_weights_periodic(
        x=Xp, y=Yp, x0=cfg.x0, y0=cfg.y0, dx=cfg.dx, dy=cfg.dy, nx=cfg.nx, ny=cfg.ny
    )
    ixm, iym, wm = _bilinear_weights_periodic(
        x=Xm, y=Ym, x0=cfg.x0, y0=cfg.y0, dx=cfg.dx, dy=cfg.dy, nx=cfg.nx, ny=cfg.ny
    )

    dl = jnp.asarray(
        jnp.abs(cfg.dz) * jnp.sqrt(cfg.Bx**2 + cfg.By**2 + cfg.Bz**2) / jnp.abs(cfg.Bz)
    )
    dl_arr = jnp.broadcast_to(dl, (cfg.nx, cfg.ny))
    fwd = FCIBilinearMap(ix=ixp, iy=iyp, w=wp, dl=dl_arr)
    bwd = FCIBilinearMap(ix=ixm, iy=iym, w=wm, dl=dl_arr)
    return fwd, bwd


def apply_periodic_plane_shift(
    f: jnp.ndarray, *, shift_x: float, shift_y: float, dx: float, dy: float
) -> jnp.ndarray:
    """Reference helper: apply a continuous shift on a periodic grid via FFT phase factors.

    This is useful for verification on periodic planes.
    """

    nx, ny = f.shape
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    phase = jnp.exp(1j * (KX * shift_x + KY * shift_y))
    return jnp.fft.ifft2(jnp.fft.fft2(f) * phase).real
