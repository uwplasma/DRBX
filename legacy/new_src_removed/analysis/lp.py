from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import numpy as np

from jaxdrb.analysis.scan import scan_ky
from jaxdrb.models.params import DRBParams
from jaxdrb.models.registry import DEFAULT_MODEL, ModelSpec


@dataclass
class LpFixedPointResult:
    Lp: float
    ky_star: float
    gamma_star: float
    gamma_over_ky_star: float
    history: np.ndarray


def solve_lp_fixed_point(
    params: DRBParams,
    geom,
    *,
    q: float,
    ky: np.ndarray,
    Lp0: float,
    omega_n_scale: float = 1.0,
    model: ModelSpec = DEFAULT_MODEL,
    max_iter: int = 50,
    tol: float = 1e-3,
    relax: float = 0.7,
    kx: float = 0.0,
    arnoldi_m: int = 40,
    arnoldi_tol: float = 1e-3,
    arnoldi_max_m: int | None = None,
    nev: int = 6,
    seed: int = 0,
    verbose: bool = False,
) -> LpFixedPointResult:
    """Self-consistent SOL width estimate using the Halpern 2013 fixed-point rule.

    Implements the iteration described around Eq. (20) of Halpern et al. (Phys. Plasmas 20, 052306),
    solving

        (gamma/ky)_max(Lp) = Lp / q

    by fixed-point iteration:

        Lp_{n+1} = q * (gamma/ky)_max(Lp_n).

    Mapping to `jaxdrb` parameters:
      - We interpret `omega_n = omega_n_scale / Lp` (i.e. a background gradient drive scaling like 1/Lp).
      - Temperature-gradient drive is left untouched in `params` for now.
    """

    if q <= 0:
        raise ValueError("q must be > 0.")
    ky = np.asarray(ky, dtype=float)
    if ky.ndim != 1:
        raise ValueError("ky must be a 1D array.")
    if np.any(ky <= 0):
        raise ValueError("ky must be strictly positive for gamma/ky maximization.")
    if Lp0 <= 0:
        raise ValueError("Lp0 must be > 0.")
    if not (0.0 < relax <= 1.0):
        raise ValueError("relax must be in (0, 1].")

    Lp = float(Lp0)
    hist = []

    for _it in range(max_iter):
        params_lp: DRBParams = eqx.tree_at(lambda p: p.omega_n, params, float(omega_n_scale / Lp))

        scan = scan_ky(
            params_lp,
            geom,
            ky=ky,
            kx=kx,
            model=model,
            arnoldi_m=arnoldi_m,
            arnoldi_tol=arnoldi_tol,
            arnoldi_max_m=arnoldi_max_m,
            nev=nev,
            seed=seed,
            do_initial_value=False,
            verbose=False,
        )

        gamma = scan.gamma_eigs
        ratio = np.maximum(gamma, 0.0) / ky
        idx = int(np.argmax(ratio))
        ky_star = float(ky[idx])
        gamma_star = float(gamma[idx])
        ratio_star = float(ratio[idx])
        Lp_target = float(q * ratio_star)

        hist.append((Lp, ky_star, gamma_star, ratio_star, Lp_target))

        if verbose:
            print(
                f"[Lp fixed-point] Lp={Lp:10.4f}  ky*={ky_star:8.4f}  "
                f"(gamma/ky)*={ratio_star:10.4e}  target={Lp_target:10.4f}",
                flush=True,
            )

        if not np.isfinite(Lp_target) or Lp_target <= 0:
            raise RuntimeError(
                "Failed to find a positive Lp_target. The scan may be stable "
                "(max(gamma,0)/ky == 0) or the parameters/ky range may be inconsistent."
            )

        Lp_next = (1.0 - relax) * Lp + relax * Lp_target
        if abs(Lp_next - Lp) / Lp < tol:
            Lp = float(Lp_next)
            break
        Lp = float(Lp_next)

    history = np.asarray(hist, dtype=float)
    return LpFixedPointResult(
        Lp=Lp,
        ky_star=float(history[-1, 1]),
        gamma_star=float(history[-1, 2]),
        gamma_over_ky_star=float(history[-1, 3]),
        history=history,
    )
