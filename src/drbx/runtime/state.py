from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SimulationState:
    time: float = 0.0
    fields: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_field(self, name: str, value: Any) -> "SimulationState":
        updated_fields = dict(self.fields)
        updated_fields[name] = value
        return SimulationState(
            time=self.time,
            fields=updated_fields,
            diagnostics=self.diagnostics,
            metadata=self.metadata,
        )

    def with_diagnostic(self, name: str, value: Any) -> "SimulationState":
        updated_diagnostics = dict(self.diagnostics)
        updated_diagnostics[name] = value
        return SimulationState(
            time=self.time,
            fields=self.fields,
            diagnostics=updated_diagnostics,
            metadata=self.metadata,
        )

    def advance_time(self, dt: float) -> "SimulationState":
        return SimulationState(
            time=self.time + dt,
            fields=self.fields,
            diagnostics=self.diagnostics,
            metadata=self.metadata,
        )
