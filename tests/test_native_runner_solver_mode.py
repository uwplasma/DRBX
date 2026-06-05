from __future__ import annotations

import pytest

from jax_drb.config.boutinp import apply_bout_overrides, parse_bout_input
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
    config = apply_bout_overrides(
        parse_bout_input(_ONE_ION_INPUT),
        ("runtime:recycling_transient_solver_mode=adaptive_be",),
    )
    assert configured_recycling_transient_solver_mode(config) == "adaptive_be"


@pytest.mark.parametrize(
    "mode",
    (
        "bdf_fixed_full_field_jvp",
        "fixed_bdf2_jax_linearized",
        "fixed_bdf2_jax_linearized_lineax",
    ),
)
def test_configured_recycling_transient_solver_mode_accepts_jax_bdf_opt_ins(
    mode: str,
) -> None:
    config = apply_bout_overrides(
        parse_bout_input(_TWO_ION_INPUT),
        (f"runtime:recycling_transient_solver_mode={mode}",),
    )

    assert configured_recycling_transient_solver_mode(config) == mode
    assert (
        select_recycling_transient_solver_mode(config, parity_mode="one_step") == mode
    )


def test_configured_recycling_transient_solver_mode_reads_legacy_jax_drb_section() -> (
    None
):
    config = apply_bout_overrides(
        parse_bout_input(_ONE_ION_INPUT),
        ("jax_drb:recycling_transient_solver_mode=adaptive_bdf",),
    )
    assert configured_recycling_transient_solver_mode(config) == "adaptive_bdf"


def test_configured_recycling_transient_solver_mode_rejects_unknown_mode() -> None:
    config = apply_bout_overrides(
        parse_bout_input(_ONE_ION_INPUT),
        ("runtime:recycling_transient_solver_mode=bad_mode",),
    )
    with pytest.raises(ValueError):
        configured_recycling_transient_solver_mode(config)


def test_select_recycling_transient_solver_mode_defaults_by_parity_and_ion_count() -> (
    None
):
    one_ion = parse_bout_input(_ONE_ION_INPUT)
    two_ion = parse_bout_input(_TWO_ION_INPUT)

    assert (
        select_recycling_transient_solver_mode(one_ion, parity_mode="short_window")
        == "continuation"
    )
    assert (
        select_recycling_transient_solver_mode(one_ion, parity_mode="one_step")
        == "continuation"
    )
    assert (
        select_recycling_transient_solver_mode(two_ion, parity_mode="one_step") == "bdf"
    )


def test_select_recycling_transient_solver_mode_honors_configured_override() -> None:
    config = apply_bout_overrides(
        parse_bout_input(_TWO_ION_INPUT),
        ("runtime:recycling_transient_solver_mode=adaptive_bdf",),
    )

    assert (
        select_recycling_transient_solver_mode(config, parity_mode="one_step")
        == "adaptive_bdf"
    )


def test_select_recycling_transient_solver_mode_ignores_unresolvable_charge_entries() -> (
    None
):
    config = apply_bout_overrides(
        parse_bout_input(_ONE_ION_INPUT),
        ("d+:charge=missing:charge",),
    )

    assert (
        select_recycling_transient_solver_mode(config, parity_mode="one_step")
        == "continuation"
    )


def test_select_integrated_2d_transient_solver_mode_prefers_bdf_for_promoted_cases() -> (
    None
):
    config = parse_bout_input(_TWO_ION_INPUT)
    assert (
        select_integrated_2d_transient_solver_mode(
            "integrated_2d_production_one_step",
            config=config,
            parity_mode="one_step",
        )
        == "bdf"
    )
    assert (
        select_integrated_2d_transient_solver_mode(
            "tokamak_recycling_dthe_one_step",
            config=config,
            parity_mode="one_step",
        )
        == "bdf"
    )
    assert select_integrated_2d_transient_solver_mode(
        "integrated_2d_recycling_one_step",
        config=config,
        parity_mode="one_step",
    ) == select_recycling_transient_solver_mode(config, parity_mode="one_step")


def test_select_integrated_2d_transient_solver_mode_honors_configured_override() -> (
    None
):
    config = apply_bout_overrides(
        parse_bout_input(_ONE_ION_INPUT),
        ("runtime:recycling_transient_solver_mode=adaptive_be",),
    )

    assert (
        select_integrated_2d_transient_solver_mode(
            "tokamak_recycling_dthe_one_step",
            config=config,
            parity_mode="one_step",
        )
        == "adaptive_be"
    )
