"""Literal engine term registry.

This module is the ownership boundary for the strict `hermes_literal` engine.
The current Stage 1 runtime still shares most lower-level term implementations
with the unified code path, but the engine now imports its schedule and
dispatch table from here rather than from `core.terms.registry` directly.
"""

from __future__ import annotations

from jaxdrb.core.terms.registry import (
    DEFAULT_TERM_SCHEDULE as CORE_DEFAULT_TERM_SCHEDULE,
    STIFF_TERM_SCHEDULE as CORE_STIFF_TERM_SCHEDULE,
    TERM_REGISTRY as CORE_TERM_REGISTRY,
    TermSpec,
)

DEFAULT_TERM_SCHEDULE: tuple[str, ...] = tuple(CORE_DEFAULT_TERM_SCHEDULE)
STIFF_TERM_SCHEDULE: tuple[str, ...] = tuple(CORE_STIFF_TERM_SCHEDULE)
TERM_REGISTRY: dict[str, TermSpec] = dict(CORE_TERM_REGISTRY)
