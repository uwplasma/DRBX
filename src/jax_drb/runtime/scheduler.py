from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..config.boutinp import BoutConfig


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
    component_value = config.get("hermes", "components")
    component_names = _as_tuple(component_value.parsed)
    requests: list[ComponentRequest] = []

    for name in component_names:
        if config.has_section(name) and config.has_option(name, "type"):
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
