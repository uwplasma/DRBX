from __future__ import annotations

from jaxdrb.core.terms.registry import (
    DEFAULT_TERM_SCHEDULE as CORE_DEFAULT_TERM_SCHEDULE,
    STIFF_TERM_SCHEDULE as CORE_STIFF_TERM_SCHEDULE,
    TERM_REGISTRY as CORE_TERM_REGISTRY,
)
from jaxdrb.hermes_literal.registry import (
    DEFAULT_TERM_SCHEDULE,
    STIFF_TERM_SCHEDULE,
    TERM_REGISTRY,
)


def test_literal_registry_matches_current_stage1_contract() -> None:
    assert DEFAULT_TERM_SCHEDULE == CORE_DEFAULT_TERM_SCHEDULE
    assert STIFF_TERM_SCHEDULE == CORE_STIFF_TERM_SCHEDULE
    assert tuple(TERM_REGISTRY) == tuple(CORE_TERM_REGISTRY)
