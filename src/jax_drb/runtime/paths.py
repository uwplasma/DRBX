from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Repository root (the directory containing ``src/``)."""

    return Path(__file__).resolve().parents[3]
