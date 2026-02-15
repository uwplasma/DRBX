"""GRILLIX-style limiter snapshot (Stegmeir 2018 Fig. 16 proxy).

This example implements the analytic poloidal flux psi(R,Z) from Stegmeir et al. (2018),
builds FCI maps using the z-plane builder with a toroidal-angle coordinate, and produces
an annular snapshot of density with a poloidal limiter mask (bottom or HFS).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

from jaxdrb.fci.builder import EssosToroidalFCIConfig, build_fci_maps_essos_toroidal_planes
from jaxdrb.fci.drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def _make_forcing_sequence(
    key: jax.Array, *, nsteps: int, shape: tuple[int, int, int], dt: float, tau: float
) -> np.ndarray:
    """Ornstein-Uhlenbeck forcing sequence for vorticity."""
    alpha = float(np.exp(-dt / max(tau, 1e-12)))
    seq = np.zeros((nsteps,) + shape, dtype=np.float64)
    prev = np.zeros(shape, dtype=np.float64)
    for i in range(nsteps):
        key, sub = jax.random.split(key)
        noise = np.asarray(jax.random.normal(sub, shape))
        prev = alpha * prev + np.sqrt(1.0 - alpha**2) * noise
        seq[i] = prev
    return seq


def _psi_stegmeir(R: np.ndarray, Z: np.ndarray, *, R0: float) -> np.ndarray:
    """Analytic poloidal flux (eq. 50) in Stegmeir 2018 PPCF 60 035005."""
    R_hat = R / R0
    Z_hat = Z / R0
    J1 = sp.j1
    Y1 = sp.y1
    psi = (
        0.0159
        - 0.0363 * R_hat
        - 0.00262 * R_hat * J1(5.836 * R_hat)
        - 0.0117 * R_hat * (1.769 * Z_hat - 0.231) * J1(5.836 * R_hat)
        - 0.0665 * R_hat * Y1(5.836 * R_hat)
        - 0.0461 * R_hat * J1(4.669 * R_hat) * np.cos(3.502 * Z_hat - 0.457)
        + 0.0360 * R_hat * J1(3.502 * R_hat) * np.cos(4.669 * Z_hat - 0.610)
        + 0.0218 * R_hat * J1(0.584 * R_hat) * np.cos(5.807 * Z_hat - 0.758)
        - 0.0383 * R_hat * J1(6.825 * R_hat) * np.cosh(3.537 * Z_hat - 0.462)
        + 0.0238 * R_hat * J1(4.669 * R_hat) * np.sin(3.502 * Z_hat - 0.457)
        - 0.00926 * np.sin(5.836 * Z_hat - 0.762)
    )
    return psi


def _build_bfield_arrays(
    *,
    R: np.ndarray,
    Z: np.ndarray,
    R0: float,
    bphi_mode: str,
    q0: float,
    q_shear: float,
    q_r0: float,
    bphi_floor: float,
    bphi_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute BR, BZ, Bphi from psi(R,Z) on a mesh."""
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    psi = _psi_stegmeir(RR, ZZ, R0=R0)
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    dpsi_dR = np.gradient(psi, dR, axis=0, edge_order=2)
    dpsi_dZ = np.gradient(psi, dZ, axis=1, edge_order=2)

    R_safe = np.maximum(RR, 1e-6)
    BR = (1.0 / R_safe) * dpsi_dZ
    BZ = -(1.0 / R_safe) * dpsi_dR
    if bphi_mode == "q_profile":
        r2 = ((RR / R0) - 1.0) ** 2 + (ZZ / R0) ** 2
        r_minor = np.sqrt(np.maximum(r2, 1e-8))
        q = q0 + q_shear * (r_minor - q_r0)
        Bp = np.sqrt(BR**2 + BZ**2)
        Bphi = (q * Bp * R0 / r_minor) * float(bphi_scale)
        Bphi = np.maximum(Bphi, float(bphi_floor))
    else:
        Bphi = (1.0 / R_safe) * float(bphi_scale)
    return BR, BZ, Bphi


def _bilinear_interp_clipped(
    x: float, y: float, *, x0: float, y0: float, dx: float, dy: float, arr: np.ndarray
) -> float:
    nx, ny = arr.shape
    fx = (x - x0) / dx
    fy = (y - y0) / dy
    ix = int(np.floor(fx))
    iy = int(np.floor(fy))
    ix = int(np.clip(ix, 0, nx - 2))
    iy = int(np.clip(iy, 0, ny - 2))
    tx = fx - ix
    ty = fy - iy
    v00 = arr[ix, iy]
    v10 = arr[ix + 1, iy]
    v01 = arr[ix, iy + 1]
    v11 = arr[ix + 1, iy + 1]
    return float(
        (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11
    )


class _StegmeirField:
    def __init__(
        self,
        *,
        R_axis: np.ndarray,
        Z_axis: np.ndarray,
        BR: np.ndarray,
        BZ: np.ndarray,
        Bphi: np.ndarray,
    ) -> None:
        self.R0 = float(R_axis[0])
        self.Z0 = float(Z_axis[0])
        self.dR = float(R_axis[1] - R_axis[0])
        self.dZ = float(Z_axis[1] - Z_axis[0])
        self.BR = BR
        self.BZ = BZ
        self.Bphi = Bphi

    def _interp(self, R: float, Z: float, arr: np.ndarray) -> float:
        return _bilinear_interp_clipped(
            R, Z, x0=self.R0, y0=self.Z0, dx=self.dR, dy=self.dZ, arr=arr
        )

    def B(self, xyz: np.ndarray) -> np.ndarray:
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        R = float(np.hypot(x, y))
        phi = float(np.arctan2(y, x))
        Z = z
        BR = self._interp(R, Z, self.BR)
        BZ = self._interp(R, Z, self.BZ)
        Bphi = self._interp(R, Z, self.Bphi)
        Bx = BR * np.cos(phi) - Bphi * np.sin(phi)
        By = BR * np.sin(phi) + Bphi * np.cos(phi)
        return np.asarray([Bx, By, BZ], dtype=float)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        type=str,
        default="quick",
        choices=("quick", "turbulence"),
        help="Parameter preset for quick checks vs longer turbulence tuning.",
    )
    parser.add_argument("--out", type=str, default="out_grillix_fig16_bottom")
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.0002)
    parser.add_argument("--tmax", type=float, default=1.2)
    parser.add_argument("--save-stride", type=int, default=10)
    parser.add_argument(
        "--save-last-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save only the final state to reduce memory use.",
    )
    parser.add_argument("--solver", type=str, default="dopri5")
    parser.add_argument("--R0", type=float, default=1.0)
    parser.add_argument("--rmin", type=float, default=0.97)
    parser.add_argument("--rmax", type=float, default=1.03)
    parser.add_argument("--zmin", type=float, default=-0.03)
    parser.add_argument("--zmax", type=float, default=0.03)
    parser.add_argument("--lcfs", type=float, default=1.0)
    parser.add_argument("--lcfs-width", type=float, default=0.004)
    parser.add_argument("--pure-sol", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limiter-mode", type=str, default="bottom", choices=("bottom", "hfs"))
    parser.add_argument("--limiter-width", type=float, default=0.4)
    parser.add_argument("--omega-n", type=float, default=8.0)
    parser.add_argument("--omega-Te", type=float, default=5.0)
    parser.add_argument("--kappa", type=float, default=4.0)
    parser.add_argument(
        "--kappa-profile", type=str, default="cosine", choices=("constant", "cosine")
    )
    parser.add_argument("--kappa-theta0", type=float, default=0.0)
    parser.add_argument("--Dn", type=float, default=1.5e-4)
    parser.add_argument("--DOmega", type=float, default=1.5e-4)
    parser.add_argument("--Dvpar", type=float, default=4e-4)
    parser.add_argument("--DTe", type=float, default=1.5e-4)
    parser.add_argument("--chi-par", type=float, default=3e-4)
    parser.add_argument("--bphi-mode", type=str, default="q_profile", choices=("unit", "q_profile"))
    parser.add_argument("--bphi-floor", type=float, default=0.15)
    parser.add_argument("--bphi-scale", type=float, default=1.0)
    parser.add_argument("--q0", type=float, default=2.5)
    parser.add_argument("--q-shear", type=float, default=30.0)
    parser.add_argument("--q-r0", type=float, default=0.024)
    parser.add_argument("--sheath-on", action="store_true")
    parser.add_argument("--sheath-nu-particle", type=float, default=0.06)
    parser.add_argument("--sheath-nu-energy", type=float, default=0.04)
    parser.add_argument("--sheath-nu-mom", type=float, default=0.15)
    parser.add_argument(
        "--perp-operator",
        type=str,
        default="fd",
        choices=("spectral", "fd", "fv"),
    )
    parser.add_argument("--forcing-amp", type=float, default=3e-3)
    parser.add_argument("--forcing-tau", type=float, default=0.08)
    parser.add_argument("--noise-amp", type=float, default=3e-3)
    parser.add_argument("--source-tau", type=float, default=0.12)
    parser.add_argument("--source-amp", type=float, default=1.2)
    parser.add_argument("--source-Te-amp", type=float, default=0.9)
    parser.add_argument("--source-r0", type=float, default=0.020)
    parser.add_argument("--source-width", type=float, default=0.004)
    parser.add_argument("--sink-nu", type=float, default=0.4)
    parser.add_argument("--sink-Te-nu", type=float, default=0.4)
    parser.add_argument("--sink-floor", type=float, default=0.2)
    parser.add_argument("--limiter-sink-nu", type=float, default=0.6)
    parser.add_argument("--limiter-sink-Te-nu", type=float, default=0.6)
    parser.add_argument("--buffer-r0", type=float, default=0.028)
    parser.add_argument("--buffer-width", type=float, default=0.002)
    parser.add_argument("--r-minor-min", type=float, default=0.018)
    parser.add_argument("--r-minor-max", type=float, default=0.030)
    parser.add_argument("--buffer-nu", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    defaults = {
        "dt": 0.0002,
        "tmax": 1.2,
        "save_stride": 10,
        "Dn": 1.5e-4,
        "DOmega": 1.5e-4,
        "Dvpar": 4e-4,
        "DTe": 1.5e-4,
        "chi_par": 3e-4,
        "forcing_amp": 3e-3,
        "noise_amp": 3e-3,
        "sink_nu": 0.4,
        "sink_Te_nu": 0.4,
        "limiter_sink_nu": 0.6,
        "limiter_sink_Te_nu": 0.6,
        "buffer_nu": 2e-3,
        "sheath_on": False,
    }
    if args.preset == "turbulence":
        tuning = {
            "dt": 1.5e-4,
            "tmax": 3.0,
            "save_stride": 20,
            "Dn": 8e-5,
            "DOmega": 8e-5,
            "Dvpar": 2.5e-4,
            "DTe": 8e-5,
            "chi_par": 2e-4,
            "forcing_amp": 2e-3,
            "noise_amp": 2e-3,
            "sink_nu": 0.25,
            "sink_Te_nu": 0.25,
            "limiter_sink_nu": 0.8,
            "limiter_sink_Te_nu": 0.8,
            "buffer_nu": 1e-3,
            "sheath_on": True,
        }
        for name, value in tuning.items():
            if getattr(args, name) == defaults[name]:
                setattr(args, name, value)
        print("[grillix-fig16] using turbulence preset (override with explicit CLI args).")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    R = np.linspace(float(args.rmin), float(args.rmax), int(args.nx))
    Z = np.linspace(float(args.zmin), float(args.zmax), int(args.ny))
    dx = float(R[1] - R[0])
    dy = float(Z[1] - Z[0])

    BR, BZ, Bphi = _build_bfield_arrays(
        R=R,
        Z=Z,
        R0=float(args.R0),
        bphi_mode=str(args.bphi_mode),
        q0=float(args.q0),
        q_shear=float(args.q_shear),
        q_r0=float(args.q_r0),
        bphi_floor=float(args.bphi_floor),
        bphi_scale=float(args.bphi_scale),
    )
    field = _StegmeirField(R_axis=R, Z_axis=Z, BR=BR, BZ=BZ, Bphi=Bphi)
    print(
        f"[grillix-fig16] Bphi min/max: {np.min(Bphi):.3e}/{np.max(Bphi):.3e} "
        f"(mode={args.bphi_mode})"
    )

    cfg = EssosToroidalFCIConfig(
        R0=float(args.rmin),
        Z0=float(args.zmin),
        dR=dx,
        dZ=dy,
        nR=int(args.nx),
        nZ=int(args.ny),
        phi0=0.0,
        dphi=(2.0 * np.pi) / float(args.nz),
        nphi=int(args.nz),
        periodic_R=False,
        periodic_Z=False,
        periodic_phi=True,
        open_field_line=False,
        cell_centered=False,
        R_min=float(args.rmin),
        R_max=float(args.rmax),
        Z_min=float(args.zmin),
        Z_max=float(args.zmax),
    )

    map_fwd, map_bwd, _meta = build_fci_maps_essos_toroidal_planes(cfg, field=field, nsub=8)
    dl_min = float(np.min(map_fwd.dl))
    dl_max = float(np.max(map_fwd.dl))
    print(f"[grillix-fig16] map dl min/max: {dl_min:.3e}/{dl_max:.3e}")

    l = jnp.linspace(0.0, 2.0 * np.pi, int(args.nz), endpoint=False)
    grid = FCISlabGrid.from_maps(
        x0=cfg.R0,
        y0=cfg.Z0,
        dx=cfg.dR,
        dy=cfg.dZ,
        nx=cfg.nR,
        ny=cfg.nZ,
        l=l,
        map_fwd=map_fwd,
        map_bwd=map_bwd,
        open_field_line=False,
        cell_centered=False,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
    )

    x = grid.x0 + grid.dx * (jnp.arange(grid.nx) + 0.5)
    y = grid.y0 + grid.dy * (jnp.arange(grid.ny) + 0.5)
    Rg, Zg = jnp.meshgrid(x, y, indexing="ij")
    theta = jnp.arctan2(Zg, Rg - float(args.R0))
    r2 = ((Rg / float(args.R0)) - 1.0) ** 2 + (Zg / float(args.R0)) ** 2
    r_minor = jnp.sqrt(jnp.maximum(r2, 1e-8))

    sol_mask_radial = (
        jnp.ones_like(r2)
        if bool(args.pure_sol)
        else 0.5 * (1.0 + jnp.tanh((Rg - float(args.lcfs)) / max(float(args.lcfs_width), 1e-6)))
    )
    if args.limiter_mode == "bottom":
        theta0 = -0.5 * jnp.pi
    else:
        theta0 = jnp.pi
    dtheta = jnp.arctan2(jnp.sin(theta - theta0), jnp.cos(theta - theta0))
    dtheta_op = jnp.arctan2(jnp.sin(theta - (theta0 + jnp.pi)), jnp.cos(theta - (theta0 + jnp.pi)))
    width = max(float(args.limiter_width), 1e-3)
    w1 = jnp.exp(-0.5 * (dtheta / width) ** 2)
    w2 = jnp.exp(-0.5 * (dtheta_op / width) ** 2)
    limiter_mask = jnp.clip(w1 + w2, 0.0, 1.0)
    sheath_mask = sol_mask_radial * limiter_mask
    sol_mask_radial = jnp.broadcast_to(sol_mask_radial[None, ...], (grid.nz, grid.nx, grid.ny))
    sheath_mask = jnp.broadcast_to(sheath_mask[None, ...], (grid.nz, grid.nx, grid.ny))
    sheath_sign = jnp.where(w1 >= w2, 1.0, -1.0) * sheath_mask[0]
    sheath_sign = jnp.broadcast_to(sheath_sign[None, ...], (grid.nz, grid.nx, grid.ny))

    source_mask = jnp.exp(
        -((r_minor - float(args.source_r0)) ** 2) / max(float(args.source_width), 1e-6) ** 2
    )
    source_mask = jnp.broadcast_to(source_mask[None, ...], (grid.nz, grid.nx, grid.ny))
    buffer_mask = 0.5 * (
        1.0 + jnp.tanh((r_minor - float(args.buffer_r0)) / max(float(args.buffer_width), 1e-6))
    )
    buffer_mask = jnp.broadcast_to(buffer_mask[None, ...], (grid.nz, grid.nx, grid.ny))
    grid = FCISlabGrid.from_maps(
        x0=grid.x0,
        y0=grid.y0,
        dx=grid.dx,
        dy=grid.dy,
        nx=grid.nx,
        ny=grid.ny,
        l=l,
        map_fwd=map_fwd,
        map_bwd=map_bwd,
        open_field_line=False,
        cell_centered=False,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        sheath_mask=sheath_mask,
        sheath_sign=sheath_sign,
    )

    params = FCIDRB3DFullParams(
        omega_n=float(args.omega_n),
        omega_Te=float(args.omega_Te),
        kappa=float(args.kappa),
        kappa_profile=str(args.kappa_profile),
        kappa_theta0=float(args.kappa_theta0),
        alpha=0.35,
        eta_par=0.04,
        Dn=float(args.Dn),
        DOmega=float(args.DOmega),
        Dvpar=float(args.Dvpar),
        DTe=float(args.DTe),
        chi_par=float(args.chi_par),
        sheath_on=bool(args.sheath_on),
        sheath_bc_model="loizu_linear",
        sheath_nu_particle=float(args.sheath_nu_particle),
        sheath_nu_energy=float(args.sheath_nu_energy),
        sheath_nu_mom=float(args.sheath_nu_mom),
        sheath_gamma_e=3.2,
        sheath_gamma_i=3.0,
        perp_operator=str(args.perp_operator),
        bracket="arakawa",
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)

    n_target = 1.0 + float(args.source_amp)
    Te_target = 1.0 + float(args.source_Te_amp)
    n0 = n_target * source_mask + (1.0 - source_mask)
    Te0 = Te_target * source_mask + (1.0 - source_mask)
    noise = jax.random.normal(jax.random.key(int(args.seed)), (grid.nz, grid.nx, grid.ny))
    y0 = FCIDRB3DFullState(
        n=n0 + float(args.noise_amp) * noise,
        omega=float(args.noise_amp) * noise,
        vpar_e=jnp.zeros_like(n0),
        vpar_i=jnp.zeros_like(n0),
        Te=Te0,
    )

    dt = float(args.dt)
    nsteps = int(round(float(args.tmax) / dt))
    save_stride = int(args.save_stride)
    if bool(args.save_last_only):
        save_stride = max(save_stride, nsteps)
    forcing_seq = None
    if float(args.forcing_amp) > 0.0:
        forcing_seq = float(args.forcing_amp) * _make_forcing_sequence(
            jax.random.key(int(args.seed) + 1),
            nsteps=nsteps + 1,
            shape=(grid.nz, grid.nx, grid.ny),
            dt=dt,
            tau=float(args.forcing_tau),
        )

    def rhs_with_source(t, y):
        dy = model.rhs(t, y)
        dn = 0.0
        dTe = 0.0
        if float(args.source_tau) > 0.0:
            tau = float(args.source_tau)
            nbar = jnp.mean(y.n, axis=-1, keepdims=True)
            Tebar = jnp.mean(y.Te, axis=-1, keepdims=True)
            dn = dn + source_mask * (n_target - nbar) / tau
            dTe = dTe + source_mask * (Te_target - Tebar) / tau
        if float(args.sink_nu) > 0.0:
            n_floor = float(args.sink_floor)
            n_pos = jnp.maximum(y.n, n_floor)
            dn = dn - float(args.sink_nu) * buffer_mask * (n_pos - n_floor)
        if float(args.sink_Te_nu) > 0.0:
            Te_floor = float(args.sink_floor)
            Te_pos = jnp.maximum(y.Te, Te_floor)
            dTe = dTe - float(args.sink_Te_nu) * buffer_mask * (Te_pos - Te_floor)
        if float(args.limiter_sink_nu) > 0.0:
            n_floor = float(args.sink_floor)
            n_pos = jnp.maximum(y.n, n_floor)
            dn = dn - float(args.limiter_sink_nu) * sheath_mask * (n_pos - n_floor)
        if float(args.limiter_sink_Te_nu) > 0.0:
            Te_floor = float(args.sink_floor)
            Te_pos = jnp.maximum(y.Te, Te_floor)
            dTe = dTe - float(args.limiter_sink_Te_nu) * sheath_mask * (Te_pos - Te_floor)
        if float(args.buffer_nu) > 0.0:
            nu_buf = float(args.buffer_nu)
            dn = dn + nu_buf * buffer_mask * model._lap(y.n)
            dTe = dTe + nu_buf * buffer_mask * model._lap(y.Te)
            domega = nu_buf * buffer_mask * model._lap(y.omega)
        else:
            domega = jnp.zeros_like(y.omega)
        if forcing_seq is not None:
            idx = jnp.clip(jnp.floor(t / dt).astype(jnp.int32), 0, forcing_seq.shape[0] - 1)
            domega = domega + jnp.asarray(forcing_seq)[idx]
        return FCIDRB3DFullState(
            n=dy.n + dn,
            omega=dy.omega + domega,
            vpar_e=dy.vpar_e,
            vpar_i=dy.vpar_i,
            Te=dy.Te + dTe,
        )

    dy0 = rhs_with_source(0.0, y0)
    phi0 = model._phi_from_omega(y0.omega, y0.n)
    for name, arr in (
        ("n0", y0.n),
        ("omega0", y0.omega),
        ("Te0", y0.Te),
        ("phi0", phi0),
        ("rhs_n0", dy0.n),
        ("rhs_omega0", dy0.omega),
        ("rhs_Te0", dy0.Te),
    ):
        arr_np = np.asarray(jax.device_get(arr))
        print(
            f"[grillix-fig16] {name} min/max: " f"{np.nanmin(arr_np):.3e}/{np.nanmax(arr_np):.3e}"
        )
        if not np.all(np.isfinite(arr_np)):
            print(f"[grillix-fig16] warning: non-finite values in {name}.")

    start = time.time()
    ys, _ = diffeqsolve_fixed_steps(
        rhs_with_source,
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        save_every=save_stride,
        solver=str(args.solver),
        progress=True,
    )
    wall = time.time() - start
    print(f"[grillix-fig16] runtime {wall:.2f}s")

    n_series = np.asarray(jax.device_get(ys.n))
    n0_saved = n_series[0]
    print(
        f"[grillix-fig16] saved n0 min/max: " f"{np.nanmin(n0_saved):.3e}/{np.nanmax(n0_saved):.3e}"
    )
    nan_steps = np.where(~np.isfinite(n_series.reshape(n_series.shape[0], -1)).all(axis=1))[0]
    if nan_steps.size:
        print(f"[grillix-fig16] first NaN step index: {int(nan_steps[0])}")
    n0_plane = n_series[-1, 0]
    if not np.all(np.isfinite(n0_plane)):
        print("[grillix-fig16] warning: non-finite values in snapshot.")
    print(
        f"[grillix-fig16] snapshot n min/max: {np.nanmin(n0_plane):.3e}/{np.nanmax(n0_plane):.3e}"
    )
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    r_minor_np = np.sqrt(
        np.maximum(((RR / float(args.R0)) - 1.0) ** 2 + (ZZ / float(args.R0)) ** 2, 1e-12)
    )
    annulus_mask = (r_minor_np >= float(args.r_minor_min)) & (r_minor_np <= float(args.r_minor_max))
    n_plot = np.where(annulus_mask, n0_plane, np.nan)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    im = ax.pcolormesh(RR, ZZ, n_plot, cmap="turbo", shading="auto")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, pad=0.02)
    snap_path = out_dir / "snapshot_fig16.png"
    fig.savefig(snap_path, dpi=180)
    plt.close(fig)
    print(f"[grillix-fig16] wrote {snap_path}")


if __name__ == "__main__":
    main()
