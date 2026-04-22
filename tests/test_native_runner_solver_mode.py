from __future__ import annotations

import pytest

from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input, parse_bout_input
from jax_drb.native.runner_solver_mode import (
    configured_recycling_transient_solver_mode,
    select_integrated_2d_transient_solver_mode,
    select_recycling_transient_solver_mode,
)


_ONE_ION_INPUT = """
[mesh]
nx = 1
ny = 4
nz = 1

[model]
components = e, d+

[e]
type = quasineutral
charge = -1

[d+]
type = evolve_density, evolve_pressure, evolve_momentum
charge = 1
AA = 2
"""


_TWO_ION_INPUT = """
[mesh]
nx = 1
ny = 4
nz = 1

[model]
components = e, d+, t+

[e]
type = quasineutral
charge = -1

[d+]
type = evolve_density, evolve_pressure, evolve_momentum
charge = 1
AA = 2

[t+]
type = evolve_density, evolve_pressure, evolve_momentum
charge = 1
AA = 3
"""


def test_configured_recycling_transient_solver_mode_reads_runtime_override() -> None:
    config = apply_bout_overrides(parse_bout_input(_ONE_ION_INPUT), ("runtime:recycling_transient_solver_mode=adaptive_be",))
    assert configured_recycling_transient_solver_mode(config) == "adaptive_be"


def test_configured_recycling_transient_solver_mode_rejects_unknown_mode() -> None:
    config = apply_bout_overrides(parse_bout_input(_ONE_ION_INPUT), ("runtime:recycling_transient_solver_mode=bad_mode",))
    with pytest.raises(ValueError):
        configured_recycling_transient_solver_mode(config)


def test_select_recycling_transient_solver_mode_defaults_by_parity_and_ion_count() -> None:
    one_ion = parse_bout_input(_ONE_ION_INPUT)
    two_ion = parse_bout_input(_TWO_ION_INPUT)

    assert select_recycling_transient_solver_mode(one_ion, parity_mode="short_window") == "continuation"
    assert select_recycling_transient_solver_mode(one_ion, parity_mode="one_step") == "continuation"
    assert select_recycling_transient_solver_mode(two_ion, parity_mode="one_step") == "bdf"


def test_select_integrated_2d_transient_solver_mode_prefers_bdf_for_promoted_cases() -> None:
    config = load_bout_input("/Users/rogerio/local/hermes-3/tests/integrated/2D-production/data/BOUT.inp")
    assert select_integrated_2d_transient_solver_mode(
        "integrated_2d_production_one_step",
        config=config,
        parity_mode="one_step",
    ) == "bdf"
    assert select_integrated_2d_transient_solver_mode(
        "tokamak_recycling_dthe_one_step",
        config=config,
        parity_mode="one_step",
    ) == "bdf"
    assert select_integrated_2d_transient_solver_mode(
        "integrated_2d_recycling_one_step",
        config=config,
        parity_mode="one_step",
    ) == select_recycling_transient_solver_mode(config, parity_mode="one_step")
