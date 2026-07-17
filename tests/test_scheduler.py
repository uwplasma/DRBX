from __future__ import annotations

from dataclasses import dataclass

from drbx.config.boutinp import parse_bout_input
from drbx.runtime.scheduler import Scheduler, expand_component_requests


def test_expand_component_requests_splits_typed_species() -> None:
    config = parse_bout_input(
        """
        [model]
        components = e, reactions, vorticity, sheath_boundary

        [e]
        type = evolve_density, evolve_pressure
        AA = 1/1836

        [reactions]
        type = a + b -> c, c + e -> d
        """
    )

    requests = expand_component_requests(config)
    assert tuple(request.label for request in requests) == (
        "e:evolve_density",
        "e:evolve_pressure",
        "reactions",
        "vorticity",
        "sheath_boundary",
    )


@dataclass
class _RecordingComponent:
    name: str
    events: list[str]

    def transform(self, state: list[str], time: float) -> list[str]:
        self.events.append(f"transform:{self.name}:{time}")
        return [*state, f"{self.name}:transform"]

    def finalize(self, state: list[str], time: float) -> list[str]:
        self.events.append(f"finalize:{self.name}:{time}")
        return [*state, f"{self.name}:finalize"]


def test_scheduler_runs_all_transforms_before_all_finalizers() -> None:
    events: list[str] = []
    scheduler = Scheduler(
        components=(
            _RecordingComponent(name="a", events=events),
            _RecordingComponent(name="b", events=events),
        )
    )

    result = scheduler.execute_cycle([], time=3.0)

    assert events == [
        "transform:a:3.0",
        "transform:b:3.0",
        "finalize:a:3.0",
        "finalize:b:3.0",
    ]
    assert result == ["a:transform", "b:transform", "a:finalize", "b:finalize"]
