from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.fci.map import FCIBilinearMap


@dataclass(frozen=True)
class XPointPsi76Config:
    R_min: float
    R_max: float
    Z_min: float
    Z_max: float
    nx: int
    ny: int
    nz: int
    Lz: float
    dphi: float
    B0: float
    R0: float
    I0: float
    sigma0: float
    R1: float
    Z1: float
    Z2: float
    rho_s0: float
    open_field_line: bool = True
    cell_centered: bool = False
    nsteps: int = 16


def _psi76_grad(R, Z, cfg: XPointPsi76Config):
    r1 = (R - cfg.R1) ** 2 + (Z - cfg.Z1) ** 2
    r2 = (R - cfg.R1) ** 2 + (Z - cfg.Z2) ** 2
    r1 = jnp.maximum(r1, 1e-12)
    r2 = jnp.maximum(r2, 1e-12)
    exp1 = jnp.exp(r1 / (cfg.sigma0**2))
    dpsi_dR = cfg.I0 * (R - cfg.R1) * (1.0 / r1 + exp1 / r1 + 1.0 / r2)
    dpsi_dZ = cfg.I0 * ((Z - cfg.Z1) * (1.0 / r1 + exp1 / r1) + (Z - cfg.Z2) * (1.0 / r2))
    return dpsi_dR, dpsi_dZ


def _field_drdz_dphi(R, Z, cfg: XPointPsi76Config):
    dpsi_dR, dpsi_dZ = _psi76_grad(R, Z, cfg)
    BR = dpsi_dZ / jnp.maximum(R, 1e-8)
    BZ = -dpsi_dR / jnp.maximum(R, 1e-8)
    Bphi = cfg.B0 * cfg.R0 / jnp.maximum(R, 1e-8)
    dR_dphi = R * BR / jnp.maximum(Bphi, 1e-8)
    dZ_dphi = R * BZ / jnp.maximum(Bphi, 1e-8)
    ds_dphi = jnp.sqrt(dR_dphi**2 + dZ_dphi**2 + R**2)
    return dR_dphi, dZ_dphi, ds_dphi


def _trace_map(R, Z, cfg: XPointPsi76Config, *, dphi: float):
    step = float(dphi) / float(cfg.nsteps)
    Rn = R
    Zn = Z
    hit = jnp.zeros_like(R, dtype=bool)
    dl_acc = jnp.zeros_like(R)
    dl_hit = jnp.zeros_like(R)

    for _ in range(cfg.nsteps):
        k1_R, k1_Z, k1_s = _field_drdz_dphi(Rn, Zn, cfg)
        k2_R, k2_Z, k2_s = _field_drdz_dphi(Rn + 0.5 * step * k1_R, Zn + 0.5 * step * k1_Z, cfg)
        k3_R, k3_Z, k3_s = _field_drdz_dphi(Rn + 0.5 * step * k2_R, Zn + 0.5 * step * k2_Z, cfg)
        k4_R, k4_Z, k4_s = _field_drdz_dphi(Rn + step * k3_R, Zn + step * k3_Z, cfg)

        R_next = Rn + (step / 6.0) * (k1_R + 2.0 * k2_R + 2.0 * k3_R + k4_R)
        Z_next = Zn + (step / 6.0) * (k1_Z + 2.0 * k2_Z + 2.0 * k3_Z + k4_Z)
        ds = (step / 6.0) * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s)

        in_domain = (
            (R_next >= cfg.R_min)
            & (R_next <= cfg.R_max)
            & (Z_next >= cfg.Z_min)
            & (Z_next <= cfg.Z_max)
        )
        newly_hit = (~hit) & (~in_domain)
        dl_hit = jnp.where(newly_hit, dl_acc + ds, dl_hit)
        hit = hit | newly_hit

        Rn = jnp.where(hit, Rn, R_next)
        Zn = jnp.where(hit, Zn, Z_next)
        dl_acc = jnp.where(hit, dl_acc, dl_acc + ds)

    return Rn, Zn, dl_acc, hit, dl_hit


def _bilinear_weights_clamped(Rp, Zp, cfg: XPointPsi76Config):
    dx = (cfg.R_max - cfg.R_min) / cfg.nx
    dy = (cfg.Z_max - cfg.Z_min) / cfg.ny
    fx = (Rp - cfg.R_min) / dx
    fy = (Zp - cfg.Z_min) / dy
    i0 = jnp.floor(fx).astype(jnp.int32)
    j0 = jnp.floor(fy).astype(jnp.int32)
    i0 = jnp.clip(i0, 0, cfg.nx - 2)
    j0 = jnp.clip(j0, 0, cfg.ny - 2)
    tx = jnp.clip(fx - i0, 0.0, 1.0)
    ty = jnp.clip(fy - j0, 0.0, 1.0)
    i1 = i0 + 1
    j1 = j0 + 1
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


def build_xpoint_psi76_fci_grid(cfg: XPointPsi76Config) -> FCISlabGrid:
    xs = jnp.linspace(cfg.R_min, cfg.R_max, cfg.nx, endpoint=False)
    ys = jnp.linspace(cfg.Z_min, cfg.Z_max, cfg.ny, endpoint=False)
    R, Z = jnp.meshgrid(xs, ys, indexing="ij")

    Rf, Zf, dlf, hit_fwd, dl_hit_fwd = _trace_map(R, Z, cfg, dphi=cfg.dphi)
    Rb, Zb, dlb, hit_bwd, dl_hit_bwd = _trace_map(R, Z, cfg, dphi=-cfg.dphi)

    ix_f, iy_f, w_f = _bilinear_weights_clamped(Rf, Zf, cfg)
    ix_b, iy_b, w_b = _bilinear_weights_clamped(Rb, Zb, cfg)

    map_fwd = FCIBilinearMap(
        ix=ix_f,
        iy=iy_f,
        w=w_f,
        dl=dlf,
        hit=hit_fwd,
        dl_hit=dl_hit_fwd,
    )
    map_bwd = FCIBilinearMap(
        ix=ix_b,
        iy=iy_b,
        w=w_b,
        dl=dlb,
        hit=hit_bwd,
        dl_hit=dl_hit_bwd,
    )

    dx = (cfg.R_max - cfg.R_min) / cfg.nx
    dy = (cfg.Z_max - cfg.Z_min) / cfg.ny
    l = jnp.linspace(0.0, cfg.Lz, cfg.nz, endpoint=not cfg.open_field_line)
    return FCISlabGrid.from_maps(
        x0=cfg.R_min,
        y0=cfg.Z_min,
        dx=dx,
        dy=dy,
        nx=cfg.nx,
        ny=cfg.ny,
        l=l,
        map_fwd=map_fwd,
        map_bwd=map_bwd,
        open_field_line=cfg.open_field_line,
        cell_centered=cfg.cell_centered,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
    )
