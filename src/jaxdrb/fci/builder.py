from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

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


@dataclass(frozen=True)
class EssosToroidalFCIConfig:
    """FCI map builder config for ESSOS fields on toroidal planes.

    Coordinates and tracing model
    -----------------------------
    - Perpendicular plane coordinates are cylindrical ``(R, Z)`` at fixed toroidal angle ``phi``.
    - We build maps between successive toroidal planes separated by ``dphi``.
    - Field-line tracing integrates:

      ``dR/dphi = R * B_R / B_phi``
      ``dZ/dphi = R * B_Z / B_phi``

      using midpoint substeps.

    Target metadata
    ---------------
    A rectangular target/window can be specified with ``R_min, R_max, Z_min, Z_max``.
    If a traced segment exits this box, we mark ``hit=True`` and record:
      - ``dl_hit``: distance to first intersection
      - ``hit_R, hit_Z, hit_phi``: intersection coordinates
      - ``hit_target``: integer id (1 = wall intersection)
    """

    R0: float
    Z0: float
    dR: float
    dZ: float
    nR: int
    nZ: int

    phi0: float
    dphi: float
    nphi: int

    periodic_R: bool = False
    periodic_Z: bool = False
    periodic_phi: bool = True

    open_field_line: bool = True
    cell_centered: bool = True

    R_min: float | None = None
    R_max: float | None = None
    Z_min: float | None = None
    Z_max: float | None = None


def _bilinear_weights_clipped(
    *,
    R: np.ndarray,
    Z: np.ndarray,
    R0: float,
    Z0: float,
    dR: float,
    dZ: float,
    nR: int,
    nZ: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear interpolation indices/weights on a bounded R-Z grid."""

    fR = (R - R0) / dR
    fZ = (Z - Z0) / dZ

    i0 = np.floor(fR).astype(np.int32)
    j0 = np.floor(fZ).astype(np.int32)

    i0 = np.clip(i0, 0, nR - 2)
    j0 = np.clip(j0, 0, nZ - 2)
    i1 = i0 + 1
    j1 = j0 + 1

    tR = np.clip(fR - i0, 0.0, 1.0)
    tZ = np.clip(fZ - j0, 0.0, 1.0)

    ix = np.stack([i0, i1, i0, i1], axis=-1)
    iy = np.stack([j0, j0, j1, j1], axis=-1)
    w = np.stack(
        [
            (1.0 - tR) * (1.0 - tZ),
            tR * (1.0 - tZ),
            (1.0 - tR) * tZ,
            tR * tZ,
        ],
        axis=-1,
    )
    return ix, iy, w


def _segment_rect_intersection(
    R0: float,
    Z0: float,
    R1: float,
    Z1: float,
    *,
    R_min: float,
    R_max: float,
    Z_min: float,
    Z_max: float,
) -> tuple[float, float, float]:
    """Return first intersection (fraction, R, Z) for segment exiting a rectangle."""

    dR = R1 - R0
    dZ = Z1 - Z0
    candidates: list[tuple[float, float, float]] = []
    eps = 1e-14

    if abs(dR) > eps:
        for Rb in (R_min, R_max):
            t = (Rb - R0) / dR
            if 0.0 <= t <= 1.0:
                Zb = Z0 + t * dZ
                if Z_min - 1e-12 <= Zb <= Z_max + 1e-12:
                    candidates.append((t, Rb, Zb))
    if abs(dZ) > eps:
        for Zb in (Z_min, Z_max):
            t = (Zb - Z0) / dZ
            if 0.0 <= t <= 1.0:
                Rb = R0 + t * dR
                if R_min - 1e-12 <= Rb <= R_max + 1e-12:
                    candidates.append((t, Rb, Zb))

    if not candidates:
        return 1.0, R1, Z1
    t_hit, R_hit, Z_hit = min(candidates, key=lambda a: a[0])
    return float(t_hit), float(R_hit), float(Z_hit)


def _outside_rect(
    R: float,
    Z: float,
    *,
    R_min: float,
    R_max: float,
    Z_min: float,
    Z_max: float,
) -> bool:
    return (R < R_min) or (R > R_max) or (Z < Z_min) or (Z > Z_max)


def build_fci_maps_essos_toroidal_planes(
    cfg: EssosToroidalFCIConfig,
    *,
    field,
    nsub: int = 8,
    bphi_min: float = 1e-10,
    dl_min: float = 1e-6,
) -> tuple[FCIBilinearMap, FCIBilinearMap, dict[str, float]]:
    """Build FCI maps by tracing ESSOS field lines between toroidal planes.

    Parameters
    ----------
    cfg:
        Toroidal-plane tracing/grid config.
    field:
        ESSOS field object exposing ``B(point_xyz) -> (Bx, By, Bz)``.
    nsub:
        Number of midpoint substeps in toroidal-angle integration.
    bphi_min:
        Floor for ``|B_phi|`` in cylindrical-fieldline ratios.
    dl_min:
        Minimum positive map distance used for ``dl`` and ``dl_hit`` to avoid zero-distance
        degeneracy at immediate target intersections.
    """

    if cfg.nphi <= 1:
        raise ValueError("build_fci_maps_essos_toroidal_planes requires nphi > 1.")
    if nsub <= 0:
        raise ValueError("nsub must be > 0.")

    R_min = float(cfg.R_min if cfg.R_min is not None else cfg.R0)
    R_max = float(cfg.R_max if cfg.R_max is not None else cfg.R0 + cfg.dR * (cfg.nR - 1))
    Z_min = float(cfg.Z_min if cfg.Z_min is not None else cfg.Z0)
    Z_max = float(cfg.Z_max if cfg.Z_max is not None else cfg.Z0 + cfg.dZ * (cfg.nZ - 1))

    R_axis = cfg.R0 + cfg.dR * np.arange(cfg.nR, dtype=float)
    Z_axis = cfg.Z0 + cfg.dZ * np.arange(cfg.nZ, dtype=float)
    RR0, ZZ0 = np.meshgrid(R_axis, Z_axis, indexing="ij")

    phi_planes = cfg.phi0 + cfg.dphi * np.arange(cfg.nphi, dtype=float)
    dphi_sub = float(cfg.dphi) / float(nsub)

    def _wrap_periodic(R: float, Z: float) -> tuple[float, float]:
        if cfg.periodic_R:
            LR = cfg.dR * cfg.nR
            R = cfg.R0 + np.mod(R - cfg.R0, LR)
        if cfg.periodic_Z:
            LZ = cfg.dZ * cfg.nZ
            Z = cfg.Z0 + np.mod(Z - cfg.Z0, LZ)
        return R, Z

    def _trace_one(plane_phi: float, *, sign: float) -> tuple[np.ndarray, ...]:
        R_out = np.empty((cfg.nR, cfg.nZ), dtype=float)
        Z_out = np.empty((cfg.nR, cfg.nZ), dtype=float)
        dl_out = np.empty((cfg.nR, cfg.nZ), dtype=float)
        hit_out = np.zeros((cfg.nR, cfg.nZ), dtype=bool)
        dl_hit_out = np.full((cfg.nR, cfg.nZ), np.nan, dtype=float)
        Rhit_out = np.full((cfg.nR, cfg.nZ), np.nan, dtype=float)
        Zhit_out = np.full((cfg.nR, cfg.nZ), np.nan, dtype=float)
        phit_out = np.full((cfg.nR, cfg.nZ), np.nan, dtype=float)
        target_out = np.zeros((cfg.nR, cfg.nZ), dtype=np.int32)

        for i in range(cfg.nR):
            for j in range(cfg.nZ):
                R = float(RR0[i, j])
                Z = float(ZZ0[i, j])
                dl = 0.0
                hit = False
                Rhit = np.nan
                Zhit = np.nan
                phit = np.nan
                dl_hit = np.nan

                for m in range(nsub):
                    if hit:
                        break
                    phi_mid = plane_phi + sign * (m + 0.5) * dphi_sub
                    if cfg.periodic_phi:
                        period = cfg.dphi * cfg.nphi
                        phi_mid = cfg.phi0 + np.mod(phi_mid - cfg.phi0, period)

                    xyz = np.asarray(
                        [R * np.cos(phi_mid), R * np.sin(phi_mid), Z],
                        dtype=float,
                    )
                    Bx, By, Bz = np.asarray(field.B(xyz), dtype=float)
                    BR = Bx * np.cos(phi_mid) + By * np.sin(phi_mid)
                    Bphi = -Bx * np.sin(phi_mid) + By * np.cos(phi_mid)
                    if abs(Bphi) < bphi_min:
                        Bphi = np.sign(Bphi) * bphi_min if Bphi != 0.0 else bphi_min

                    dR_dphi = R * BR / Bphi
                    dZ_dphi = R * Bz / Bphi
                    dl_dphi = abs(R) * np.sqrt(BR * BR + Bphi * Bphi + Bz * Bz) / abs(Bphi)

                    R_next = R + sign * dR_dphi * dphi_sub
                    Z_next = Z + sign * dZ_dphi * dphi_sub
                    R_next, Z_next = _wrap_periodic(R_next, Z_next)

                    dl_step = abs(dl_dphi * dphi_sub)

                    if cfg.open_field_line and _outside_rect(
                        R_next, Z_next, R_min=R_min, R_max=R_max, Z_min=Z_min, Z_max=Z_max
                    ):
                        frac, Rint, Zint = _segment_rect_intersection(
                            R,
                            Z,
                            R_next,
                            Z_next,
                            R_min=R_min,
                            R_max=R_max,
                            Z_min=Z_min,
                            Z_max=Z_max,
                        )
                        dl += frac * dl_step
                        hit = True
                        Rhit = Rint
                        Zhit = Zint
                        phit = plane_phi + sign * (m + frac) * dphi_sub
                        dl_hit = dl
                        R = Rint
                        Z = Zint
                    else:
                        dl += dl_step
                        R = R_next
                        Z = Z_next

                R_out[i, j] = R
                Z_out[i, j] = Z
                dl_out[i, j] = dl
                hit_out[i, j] = hit
                dl_hit_out[i, j] = dl_hit
                Rhit_out[i, j] = Rhit
                Zhit_out[i, j] = Zhit
                phit_out[i, j] = phit
                target_out[i, j] = 1 if hit else 0

        return R_out, Z_out, dl_out, hit_out, dl_hit_out, Rhit_out, Zhit_out, phit_out, target_out

    arrays_fwd = [_trace_one(phi, sign=1.0) for phi in phi_planes]
    arrays_bwd = [_trace_one(phi, sign=-1.0) for phi in phi_planes]

    def _stack(arrs, idx):
        return np.stack([a[idx] for a in arrs], axis=0)

    R_fwd = _stack(arrays_fwd, 0)
    Z_fwd = _stack(arrays_fwd, 1)
    dl_fwd = _stack(arrays_fwd, 2)
    hit_fwd = _stack(arrays_fwd, 3)
    dl_hit_fwd = _stack(arrays_fwd, 4)
    Rhit_fwd = _stack(arrays_fwd, 5)
    Zhit_fwd = _stack(arrays_fwd, 6)
    phit_fwd = _stack(arrays_fwd, 7)
    tgt_fwd = _stack(arrays_fwd, 8)

    R_bwd = _stack(arrays_bwd, 0)
    Z_bwd = _stack(arrays_bwd, 1)
    dl_bwd = _stack(arrays_bwd, 2)
    hit_bwd = _stack(arrays_bwd, 3)
    dl_hit_bwd = _stack(arrays_bwd, 4)
    Rhit_bwd = _stack(arrays_bwd, 5)
    Zhit_bwd = _stack(arrays_bwd, 6)
    phit_bwd = _stack(arrays_bwd, 7)
    tgt_bwd = _stack(arrays_bwd, 8)

    dl_floor = float(dl_min)
    dl_fwd = np.maximum(dl_fwd, dl_floor)
    dl_bwd = np.maximum(dl_bwd, dl_floor)
    dl_hit_fwd = np.where(np.isfinite(dl_hit_fwd), np.maximum(dl_hit_fwd, dl_floor), dl_hit_fwd)
    dl_hit_bwd = np.where(np.isfinite(dl_hit_bwd), np.maximum(dl_hit_bwd, dl_floor), dl_hit_bwd)

    if cfg.periodic_R and cfg.periodic_Z:
        ix_fwd, iy_fwd, w_fwd = _bilinear_weights_periodic(
            x=jnp.asarray(R_fwd),
            y=jnp.asarray(Z_fwd),
            x0=cfg.R0,
            y0=cfg.Z0,
            dx=cfg.dR,
            dy=cfg.dZ,
            nx=cfg.nR,
            ny=cfg.nZ,
        )
        ix_bwd, iy_bwd, w_bwd = _bilinear_weights_periodic(
            x=jnp.asarray(R_bwd),
            y=jnp.asarray(Z_bwd),
            x0=cfg.R0,
            y0=cfg.Z0,
            dx=cfg.dR,
            dy=cfg.dZ,
            nx=cfg.nR,
            ny=cfg.nZ,
        )
        ix_fwd = np.asarray(ix_fwd)
        iy_fwd = np.asarray(iy_fwd)
        w_fwd = np.asarray(w_fwd)
        ix_bwd = np.asarray(ix_bwd)
        iy_bwd = np.asarray(iy_bwd)
        w_bwd = np.asarray(w_bwd)
    else:
        ix_fwd, iy_fwd, w_fwd = _bilinear_weights_clipped(
            R=R_fwd,
            Z=Z_fwd,
            R0=cfg.R0,
            Z0=cfg.Z0,
            dR=cfg.dR,
            dZ=cfg.dZ,
            nR=cfg.nR,
            nZ=cfg.nZ,
        )
        ix_bwd, iy_bwd, w_bwd = _bilinear_weights_clipped(
            R=R_bwd,
            Z=Z_bwd,
            R0=cfg.R0,
            Z0=cfg.Z0,
            dR=cfg.dR,
            dZ=cfg.dZ,
            nR=cfg.nR,
            nZ=cfg.nZ,
        )

    map_fwd = FCIBilinearMap(
        ix=jnp.asarray(ix_fwd, dtype=jnp.int32),
        iy=jnp.asarray(iy_fwd, dtype=jnp.int32),
        w=jnp.asarray(w_fwd, dtype=jnp.float64),
        dl=jnp.asarray(dl_fwd, dtype=jnp.float64),
        hit=jnp.asarray(hit_fwd),
        dl_hit=jnp.asarray(dl_hit_fwd, dtype=jnp.float64),
        hit_R=jnp.asarray(Rhit_fwd, dtype=jnp.float64),
        hit_Z=jnp.asarray(Zhit_fwd, dtype=jnp.float64),
        hit_phi=jnp.asarray(phit_fwd, dtype=jnp.float64),
        hit_target=jnp.asarray(tgt_fwd, dtype=jnp.int32),
    )
    map_bwd = FCIBilinearMap(
        ix=jnp.asarray(ix_bwd, dtype=jnp.int32),
        iy=jnp.asarray(iy_bwd, dtype=jnp.int32),
        w=jnp.asarray(w_bwd, dtype=jnp.float64),
        dl=jnp.asarray(dl_bwd, dtype=jnp.float64),
        hit=jnp.asarray(hit_bwd),
        dl_hit=jnp.asarray(dl_hit_bwd, dtype=jnp.float64),
        hit_R=jnp.asarray(Rhit_bwd, dtype=jnp.float64),
        hit_Z=jnp.asarray(Zhit_bwd, dtype=jnp.float64),
        hit_phi=jnp.asarray(phit_bwd, dtype=jnp.float64),
        hit_target=jnp.asarray(tgt_bwd, dtype=jnp.int32),
    )

    meta = {
        "builder": "essos_toroidal_planes",
        "R_min": R_min,
        "R_max": R_max,
        "Z_min": Z_min,
        "Z_max": Z_max,
        "n_hit_fwd": float(np.count_nonzero(hit_fwd)),
        "n_hit_bwd": float(np.count_nonzero(hit_bwd)),
        "nphi": float(cfg.nphi),
        "nR": float(cfg.nR),
        "nZ": float(cfg.nZ),
    }
    return map_fwd, map_bwd, meta
