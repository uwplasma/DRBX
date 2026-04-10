from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_reference_root() -> Path | None:
    env_value = os.environ.get("JAX_DRB_REFERENCE_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    for sibling_name in ("reference-suite", "external-benchmarks"):
        sibling = repo_root().parent / sibling_name
        if sibling.exists():
            return sibling.resolve()
    parent = repo_root().parent
    for sibling in parent.iterdir():
        if not sibling.is_dir():
            continue
        if (sibling / "tests" / "integrated").exists() and (sibling / "examples" / "tokamak-2D").exists():
            return sibling.resolve()
    return None


def require_reference_root() -> Path:
    resolved = default_reference_root()
    if resolved is None:
        raise FileNotFoundError(
            "Set JAX_DRB_REFERENCE_ROOT to a checkout containing the external benchmark decks."
        )
    return resolved
