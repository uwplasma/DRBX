from __future__ import annotations

from ..config.normalization import ELEMENTARY_CHARGE, PROTON_MASS
from ..runtime.run_config import RunConfiguration


def resolved_dataset_scalars(run_config: RunConfiguration) -> dict[str, float]:
    if run_config.normalization is not None:
        normalization = run_config.normalization
        return {
            "Nnorm": normalization.Nnorm,
            "Tnorm": normalization.Tnorm,
            "Bnorm": normalization.Bnorm,
            "Cs0": normalization.Cs0,
            "Omega_ci": normalization.Omega_ci,
            "rho_s0": normalization.rho_s0,
        }

    Nnorm = float(run_config.model_scalars.get("Nnorm", 1.0e19))
    Tnorm = float(run_config.model_scalars.get("Tnorm", 100.0))
    Bnorm = float(run_config.model_scalars.get("Bnorm", 1.0))
    Cs0 = float((ELEMENTARY_CHARGE * Tnorm / PROTON_MASS) ** 0.5)
    Omega_ci = float(ELEMENTARY_CHARGE * Bnorm / PROTON_MASS)
    rho_s0 = float(Cs0 / Omega_ci)
    return {
        "Nnorm": Nnorm,
        "Tnorm": Tnorm,
        "Bnorm": Bnorm,
        "Cs0": Cs0,
        "Omega_ci": Omega_ci,
        "rho_s0": rho_s0,
    }
