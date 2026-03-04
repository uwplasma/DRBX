from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class DRBConfig:
    """Container for unified DRB configuration parsed from TOML."""

    data: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"Section '{name}' must be a table in the TOML config.")
        return value


_VALID_ENGINES = {"unified", "drb_fv", "fv_drb", "drb-fv"}


def load_config(path: str | Path) -> DRBConfig:
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    engine = str(data.get("engine", "unified")).strip().lower()
    if engine not in _VALID_ENGINES:
        raise ValueError(f"Invalid engine '{engine}'. Valid options: {sorted(_VALID_ENGINES)}")
    data["engine"] = "drb_fv" if engine in {"fv_drb", "drb-fv"} else engine
    return DRBConfig(data=data)
