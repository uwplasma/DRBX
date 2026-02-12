from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .map import FCIBilinearMap, _bilinear_weights_periodic


@dataclass(frozen=True)
class ZPlaneFCIConfig:
    """Configuration for building an FCI map between equally-spaced z-planes.

    This is a pragmatic, early-stage map builder that is useful for:
      - analytic MMS tests,
      - regression tests on curved maps in a periodic box,
      - ESSOS BiotSavart field maps in a local Cartesian patch.

    Notes
    -----
    - Perpendicular planes are periodic rectangles in (x,y).
    - Planes are equally spaced in z by ``dz`` with ``nz`` planes.
    - The map is constructed by integrating the field-line ODE using z as the
      independent variable:

        dx/dz = Bx/Bz,  dy/dz = By/Bz,

      with periodic wrapping in (x,y).

    - For open field lines with ``cell_centered=True``, we encode plate hits at
      the first/last planes via ``map.hit`` and ``map.dl_hit`` (half-step plates).
    """

    # Perpendicular grid (periodic).
    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int

    # Parallel plane stack.
    z0: float
    dz: float
    nz: int
    periodic_z: bool = False

    # Open-field-line metadata (simple slab plates at z boundaries).
    open_field_line: bool = False
    cell_centered: bool = False


BFieldFn = Callable[[jnp.ndarray], jnp.ndarray]


def build_fci_maps_zplanes(
    cfg: ZPlaneFCIConfig,
    *,
    B: BFieldFn,
    nsub: int = 8,
    bz_min: float = 1e-8,
) -> tuple[FCIBilinearMap, FCIBilinearMap]:
    """Build (forward, backward) FCI bilinear maps for a periodic (x,y) box.

    Parameters
    ----------
    cfg:
        Z-plane map configuration.
    B:
        Magnetic field callback. Must accept an array of shape (..., 3) and return
        an array of shape (..., 3) with components (Bx, By, Bz). The function should
        be JAX-traceable (jit/vmap compatible).
    nsub:
        Number of substeps used in the z-ODE integration between planes.
    bz_min:
        Safety floor for |Bz| in the dx/dz, dy/dz ratios. Points with |Bz| below this
        threshold are regularized (this is primarily a guardrail for early-stage use).
    """

    if cfg.nz <= 1:
        raise ValueError("FCI map builder requires nz > 1.")
    if nsub <= 0:
        raise ValueError("nsub must be > 0.")

    Lx = cfg.dx * cfg.nx
    Ly = cfg.dy * cfg.ny

    xs = cfg.x0 + cfg.dx * jnp.arange(cfg.nx)
    ys = cfg.y0 + cfg.dy * jnp.arange(cfg.ny)
    X0, Y0 = jnp.meshgrid(xs, ys, indexing="ij")

    z_planes = cfg.z0 + cfg.dz * jnp.arange(cfg.nz)
    dz_sub = float(cfg.dz) / float(nsub)
    i_sub = jnp.arange(nsub, dtype=jnp.int32)

    def _wrap_xy(x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = cfg.x0 + jnp.mod(x - cfg.x0, Lx)
        y = cfg.y0 + jnp.mod(y - cfg.y0, Ly)
        return x, y

    def _b_components(points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        Bxyz = B(points)
        return Bxyz[..., 0], Bxyz[..., 1], Bxyz[..., 2]

    def _integrate_one_step(
        z_start: jnp.ndarray, *, sign: float
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Integrate from z_start to z_start + sign*dz using nsub midpoint steps."""

        def step(carry, i):
            x, y, dl = carry
            z = z_start + sign * (jnp.asarray(i, dtype=x.dtype) + 0.5) * dz_sub
            if cfg.periodic_z:
                Lz = cfg.dz * cfg.nz
                z = cfg.z0 + jnp.mod(z - cfg.z0, Lz)

            pts = jnp.stack([x, y, jnp.broadcast_to(z, x.shape)], axis=-1).reshape((-1, 3))
            Bx, By, Bz = _b_components(pts)
            Bx = Bx.reshape(x.shape)
            By = By.reshape(y.shape)
            Bz = Bz.reshape(y.shape)

            Bz_safe = jnp.where(
                jnp.abs(Bz) > bz_min, Bz, jnp.sign(Bz) * bz_min + (Bz == 0) * bz_min
            )
            dx_dz = Bx / Bz_safe
            dy_dz = By / Bz_safe

            x = x + sign * dx_dz * dz_sub
            y = y + sign * dy_dz * dz_sub
            x, y = _wrap_xy(x, y)

            dl = dl + jnp.sqrt(1.0 + dx_dz**2 + dy_dz**2) * dz_sub
            return (x, y, dl), None

        (x1, y1, dl), _ = jax.lax.scan(step, (X0, Y0, jnp.zeros_like(X0)), i_sub)
        return x1, y1, dl

    # Vectorize over planes: build footpoints for each plane independently.
    x_fwd, y_fwd, dl_fwd = jax.vmap(lambda z0: _integrate_one_step(z0, sign=1.0))(z_planes)
    x_bwd, y_bwd, dl_bwd = jax.vmap(lambda z0: _integrate_one_step(z0, sign=-1.0))(z_planes)

    # Compute periodic bilinear weights on the (x,y) plane grid.
    ix_fwd, iy_fwd, w_fwd = _bilinear_weights_periodic(
        x=x_fwd, y=y_fwd, x0=cfg.x0, y0=cfg.y0, dx=cfg.dx, dy=cfg.dy, nx=cfg.nx, ny=cfg.ny
    )
    ix_bwd, iy_bwd, w_bwd = _bilinear_weights_periodic(
        x=x_bwd, y=y_bwd, x0=cfg.x0, y0=cfg.y0, dx=cfg.dx, dy=cfg.dy, nx=cfg.nx, ny=cfg.ny
    )

    hit_fwd = None
    hit_bwd = None
    dl_hit_fwd = None
    dl_hit_bwd = None

    if cfg.open_field_line and cfg.cell_centered and cfg.nz >= 2:
        hit_fwd = jnp.zeros((cfg.nz, cfg.nx, cfg.ny), dtype=bool).at[-1].set(True)
        hit_bwd = jnp.zeros((cfg.nz, cfg.nx, cfg.ny), dtype=bool).at[0].set(True)
        dl_hit_fwd = 0.5 * dl_fwd
        dl_hit_bwd = 0.5 * dl_bwd

    map_fwd = FCIBilinearMap(
        ix=ix_fwd, iy=iy_fwd, w=w_fwd, dl=dl_fwd, hit=hit_fwd, dl_hit=dl_hit_fwd
    )
    map_bwd = FCIBilinearMap(
        ix=ix_bwd, iy=iy_bwd, w=w_bwd, dl=dl_bwd, hit=hit_bwd, dl_hit=dl_hit_bwd
    )
    return map_fwd, map_bwd
