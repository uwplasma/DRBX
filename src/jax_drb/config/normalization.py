from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .boutinp import BoutConfig, NumericResolver

ELEMENTARY_CHARGE = 1.602176634e-19
PROTON_MASS = 1.67262192369e-27


@dataclass(frozen=True)
class MetricPolicy:
    normalise_metric: bool
    recalculate_metric: bool


@dataclass(frozen=True)
class HermesNormalization:
    Nnorm: float
    Tnorm: float
    Bnorm: float
    Cs0: float
    Omega_ci: float
    rho_s0: float
    units: Mapping[str, float]
    metric_policy: MetricPolicy

    @classmethod
    def from_config(
        cls,
        config: BoutConfig,
        *,
        external_values: Mapping[str, float] | None = None,
    ) -> "HermesNormalization":
        resolver = NumericResolver(config, external_values=external_values)
        Nnorm = resolver.resolve("hermes", "Nnorm")
        Tnorm = resolver.resolve("hermes", "Tnorm")
        Bnorm = resolver.resolve("hermes", "Bnorm")
        Cs0 = (ELEMENTARY_CHARGE * Tnorm / PROTON_MASS) ** 0.5
        Omega_ci = ELEMENTARY_CHARGE * Bnorm / PROTON_MASS
        rho_s0 = Cs0 / Omega_ci
        normalise_metric = bool(config.parsed("hermes", "normalise_metric")) if config.has_option("hermes", "normalise_metric") else False
        recalculate_metric = bool(config.parsed("hermes", "recalculate_metric")) if config.has_option("hermes", "recalculate_metric") else False
        units = {
            "inv_meters_cubed": Nnorm,
            "eV": Tnorm,
            "Tesla": Bnorm,
            "seconds": 1.0 / Omega_ci,
            "meters": rho_s0,
        }
        return cls(
            Nnorm=Nnorm,
            Tnorm=Tnorm,
            Bnorm=Bnorm,
            Cs0=Cs0,
            Omega_ci=Omega_ci,
            rho_s0=rho_s0,
            units=units,
            metric_policy=MetricPolicy(
                normalise_metric=normalise_metric,
                recalculate_metric=recalculate_metric,
            ),
        )
