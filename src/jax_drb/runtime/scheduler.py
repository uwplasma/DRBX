from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol

from ..config.boutinp import BoutConfig
from ..config.model import locate_model_section

_COMPONENT_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ComponentRequest:
    label: str
    section: str
    implementation: str
    source_component: str
    typed: bool


class SupportsSchedulerHooks(Protocol):
    name: str

    def transform(self, state: Any, time: float) -> Any:
        ...

    def finalize(self, state: Any, time: float) -> Any:
        ...


@dataclass(frozen=True)
class Scheduler:
    components: tuple[SupportsSchedulerHooks, ...]

    def execute_cycle(self, state: Any, time: float) -> Any:
        current = state
        for component in self.components:
            current = component.transform(current, time)
        for component in self.components:
            current = component.finalize(current, time)
        return current


def expand_component_requests(config: BoutConfig) -> tuple[ComponentRequest, ...]:
    model_section = locate_model_section(config)
    component_value = config.get(model_section, "components")
    component_names = _as_tuple(component_value.parsed)
    requests: list[ComponentRequest] = []

    for name in component_names:
        if config.has_section(name) and config.has_option(name, "type") and _is_species_component(config, name):
            for implementation in _as_tuple(config.parsed(name, "type")):
                requests.append(
                    ComponentRequest(
                        label=f"{name}:{implementation}",
                        section=name,
                        implementation=implementation,
                        source_component=name,
                        typed=True,
                    )
                )
            continue

        requests.append(
            ComponentRequest(
                label=name,
                section=name,
                implementation=name,
                source_component=name,
                typed=False,
            )
        )

    return tuple(requests)


def _as_tuple(value: bool | int | float | str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return value
    return (str(value),)


def _is_species_component(config: BoutConfig, section: str) -> bool:
    type_value = config.parsed(section, "type")
    return all(_COMPONENT_TYPE_PATTERN.fullmatch(item) for item in _as_tuple(type_value))
