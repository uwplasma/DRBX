from __future__ import annotations

import numpy as np

from jax_drb.native import neutral_mixed
from jax_drb.native.neutral_mixed_state import (
    NeutralMixedHistoryResult,
    NeutralMixedImplicitStepInfo,
    NeutralMixedRhsResult,
    NeutralMixedState,
    PreparedNeutralMixedState,
)


def test_neutral_mixed_reexports_state_dataclasses() -> None:
    assert neutral_mixed.NeutralMixedState is NeutralMixedState
    assert neutral_mixed.NeutralMixedRhsResult is NeutralMixedRhsResult
    assert neutral_mixed.NeutralMixedHistoryResult is NeutralMixedHistoryResult
    assert neutral_mixed.NeutralMixedImplicitStepInfo is NeutralMixedImplicitStepInfo


def test_prepared_neutral_mixed_state_layout_is_explicit() -> None:
    field = np.ones((1, 1, 1), dtype=np.float64)
    state = PreparedNeutralMixedState(
        density=field,
        pressure=field,
        momentum=field,
        density_limited=field,
        pressure_limited=field,
        temperature=field,
        temperature_limited=field,
        velocity=field,
        diffusion=field,
        diffusion_density=field,
        diffusion_pressure=field,
        diffusion_momentum=field,
        conductivity=field,
        viscosity=field,
        log_pressure=field,
        sound_speed=field,
    )

    assert state.sound_speed.shape == (1, 1, 1)
