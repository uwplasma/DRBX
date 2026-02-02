from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from jaxdrb.models.cold_ion_drb import Equilibrium, default_equilibrium
from jaxdrb.models.cold_ion_drb import equilibrium as cold_equilibrium
from jaxdrb.models.cold_ion_drb import rhs_nonlinear as cold_rhs
from jaxdrb.models.em_drb import equilibrium as em_equilibrium
from jaxdrb.models.em_drb import rhs_nonlinear as em_rhs
from jaxdrb.models.hot_ion_drb import equilibrium as hot_ion_equilibrium
from jaxdrb.models.hot_ion_drb import rhs_nonlinear as hot_ion_rhs


@dataclass(frozen=True)
class ModelSpec:
    """Minimal model interface for scans/CLI."""

    name: str
    rhs: Callable[..., Any]
    equilibrium: Callable[[int], Any]
    random_state: Callable[..., Any]
    default_eq: Callable[[int], Equilibrium] | None = None


def _cold_model() -> ModelSpec:
    from jaxdrb.models.cold_ion_drb import State as ColdState

    return ModelSpec(
        name="cold-ion-es",
        rhs=cold_rhs,
        equilibrium=cold_equilibrium,
        random_state=ColdState.random,
        default_eq=lambda nl: default_equilibrium(nl, n0=1.0),
    )


def _em_model() -> ModelSpec:
    from jaxdrb.models.em_drb import State as EMState

    return ModelSpec(
        name="em",
        rhs=em_rhs,
        equilibrium=em_equilibrium,
        random_state=EMState.random,
        default_eq=lambda nl: default_equilibrium(nl, n0=1.0),
    )


def _hot_ion_model() -> ModelSpec:
    from jaxdrb.models.hot_ion_drb import State as HotState

    return ModelSpec(
        name="hot-ion-es",
        rhs=hot_ion_rhs,
        equilibrium=hot_ion_equilibrium,
        random_state=HotState.random,
        default_eq=lambda nl: default_equilibrium(nl, n0=1.0),
    )


MODELS: dict[str, ModelSpec] = {
    "cold-ion-es": _cold_model(),
    "em": _em_model(),
    "hot-ion-es": _hot_ion_model(),
}


def get_model(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError as e:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}") from e


DEFAULT_MODEL: ModelSpec = MODELS["cold-ion-es"]
