from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .boutinp import BoutConfig, NumericResolver
from .model import locate_model_section

ELEMENTARY_CHARGE = 1.602176634e-19
PROTON_MASS = 1.67262192369e-27


@dataclass(frozen=True)
class MetricPolicy:
    normalise_metric: bool
    recalculate_metric: bool


@dataclass(frozen=True)
class ModelNormalization:
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
    ) -> "ModelNormalization":
        resolver = NumericResolver(config, external_values=external_values)
        model_section = locate_model_section(config)
        Nnorm = resolver.resolve(model_section, "Nnorm")
        Tnorm = resolver.resolve(model_section, "Tnorm")
        Bnorm = resolver.resolve(model_section, "Bnorm")
        Cs0 = (ELEMENTARY_CHARGE * Tnorm / PROTON_MASS) ** 0.5
        Omega_ci = ELEMENTARY_CHARGE * Bnorm / PROTON_MASS
        rho_s0 = Cs0 / Omega_ci
        normalise_metric = bool(config.parsed(model_section, "normalise_metric")) if config.has_option(model_section, "normalise_metric") else False
        recalculate_metric = bool(config.parsed(model_section, "recalculate_metric")) if config.has_option(model_section, "recalculate_metric") else False
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
