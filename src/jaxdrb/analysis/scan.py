from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

from jaxdrb.linear.arnoldi import arnoldi_eigs
from jaxdrb.linear.growthrate import GrowthRateResult, estimate_growth_rate
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State, equilibrium
from jaxdrb.models.params import DRBParams


@dataclass
class Scan1DResult:
    ky: np.ndarray
    gamma_eigs: np.ndarray
    omega_eigs: np.ndarray
    eigs: np.ndarray
    gamma_iv: np.ndarray | None = None
    omega_iv: np.ndarray | None = None


def _ensure_out_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def scan_ky(
    params: DRBParams,
    geom,
    *,
    ky: np.ndarray,
    kx: float = 0.0,
    nl: int | None = None,
    arnoldi_m: int = 40,
    arnoldi_tol: float = 1e-3,
    arnoldi_max_m: int | None = None,
    nev: int = 6,
    seed: int = 0,
    do_initial_value: bool = True,
    tmax: float = 30.0,
    dt0: float = 0.01,
    nsave: int = 200,
) -> Scan1DResult:
    """Scan leading eigenvalues (and optionally initial-value growth rates) over a ky grid."""

    ky = np.asarray(ky, dtype=float)
    if ky.ndim != 1:
        raise ValueError("ky must be a 1D array.")

    if nl is None:
        nl = int(getattr(geom, "l").size)

    y_eq = equilibrium(nl)
    key = jax.random.PRNGKey(seed)

    gamma_eigs = np.zeros((ky.size,), dtype=float)
    omega_eigs = np.zeros((ky.size,), dtype=float)
    eigs = np.zeros((ky.size, nev), dtype=np.complex128)
    gamma_iv = np.zeros((ky.size,), dtype=float) if do_initial_value else None
    omega_iv = np.zeros((ky.size,), dtype=float) if do_initial_value else None

    max_m = arnoldi_max_m
    if max_m is None:
        max_m = 5 * nl
    max_m = min(max_m, 5 * nl)

    for i, ky_i in enumerate(ky):
        key, subkey = jax.random.split(key)
        v0 = State.random(subkey, nl, amplitude=1e-3)

        matvec = linear_matvec(y_eq, params, geom, kx=float(kx), ky=float(ky_i))

        m = min(int(arnoldi_m), max_m)
        arn = arnoldi_eigs(matvec, v0, m=m, nev=nev, seed=seed)
        lead_idx = int(np.argmax(np.real(arn.eigenvalues)))
        lead = arn.eigenvalues[lead_idx]
        rel_resid = float(arn.residual_norms[lead_idx] / (abs(lead) + 1.0))
        while rel_resid > arnoldi_tol and m < max_m:
            m = min(int(np.ceil(m * 2.0)), max_m)
            arn = arnoldi_eigs(matvec, v0, m=m, nev=nev, seed=seed)
            lead_idx = int(np.argmax(np.real(arn.eigenvalues)))
            lead = arn.eigenvalues[lead_idx]
            rel_resid = float(arn.residual_norms[lead_idx] / (abs(lead) + 1.0))

        eigs[i, : len(arn.eigenvalues)] = arn.eigenvalues
        gamma_eigs[i] = float(np.real(lead))
        omega_eigs[i] = float(np.imag(lead))

        if do_initial_value:
            gr: GrowthRateResult = estimate_growth_rate(
                matvec, v0, tmax=tmax, dt0=dt0, nsave=nsave, fit_window=0.5
            )
            assert gamma_iv is not None
            assert omega_iv is not None
            gamma_iv[i] = gr.gamma
            omega_iv[i] = gr.omega

    return Scan1DResult(
        ky=ky,
        gamma_eigs=gamma_eigs,
        omega_eigs=omega_eigs,
        eigs=eigs,
        gamma_iv=gamma_iv,
        omega_iv=omega_iv,
    )


@dataclass
class Scan2DResult:
    kx: np.ndarray
    ky: np.ndarray
    gamma_eigs: np.ndarray
    omega_eigs: np.ndarray


def scan_kx_ky(
    params: DRBParams,
    geom,
    *,
    kx: np.ndarray,
    ky: np.ndarray,
    nl: int | None = None,
    arnoldi_m: int = 40,
    arnoldi_tol: float = 1e-3,
    arnoldi_max_m: int | None = None,
    nev: int = 6,
    seed: int = 0,
) -> Scan2DResult:
    """2D scan of leading eigenvalues over (kx, ky).

    This routine focuses on eigenvalues only, since doing initial-value integrations
    at every grid point is typically too expensive.
    """

    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    if kx.ndim != 1 or ky.ndim != 1:
        raise ValueError("kx and ky must be 1D arrays.")

    if nl is None:
        nl = int(getattr(geom, "l").size)

    y_eq = equilibrium(nl)
    key = jax.random.PRNGKey(seed)
    key, subkey0 = jax.random.split(key)
    v0 = State.random(subkey0, nl, amplitude=1e-3)

    gamma_eigs = np.zeros((kx.size, ky.size), dtype=float)
    omega_eigs = np.zeros((kx.size, ky.size), dtype=float)

    max_m = arnoldi_max_m
    if max_m is None:
        max_m = 5 * nl
    max_m = min(max_m, 5 * nl)

    for ix, kx_i in enumerate(kx):
        for iy, ky_i in enumerate(ky):
            matvec = linear_matvec(y_eq, params, geom, kx=float(kx_i), ky=float(ky_i))

            m = min(int(arnoldi_m), max_m)
            arn = arnoldi_eigs(matvec, v0, m=m, nev=nev, seed=seed)
            lead_idx = int(np.argmax(np.real(arn.eigenvalues)))
            lead = arn.eigenvalues[lead_idx]
            rel_resid = float(arn.residual_norms[lead_idx] / (abs(lead) + 1.0))
            while rel_resid > arnoldi_tol and m < max_m:
                m = min(int(np.ceil(m * 2.0)), max_m)
                arn = arnoldi_eigs(matvec, v0, m=m, nev=nev, seed=seed)
                lead_idx = int(np.argmax(np.real(arn.eigenvalues)))
                lead = arn.eigenvalues[lead_idx]
                rel_resid = float(arn.residual_norms[lead_idx] / (abs(lead) + 1.0))

            gamma_eigs[ix, iy] = float(np.real(lead))
            omega_eigs[ix, iy] = float(np.imag(lead))

    return Scan2DResult(kx=kx, ky=ky, gamma_eigs=gamma_eigs, omega_eigs=omega_eigs)
