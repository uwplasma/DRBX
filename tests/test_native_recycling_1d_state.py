from __future__ import annotations

import numpy as np

from jax_drb.native import recycling_1d
from jax_drb.native.recycling_1d_state import (
    DensityFeedbackTerms,
    ElectronBoundaryResult,
    FullSheathSettings,
    IonBoundaryResult,
    Recycling1DHistoryResult,
    Recycling1DImplicitStepInfo,
    Recycling1DRhsResult,
    SimpleSheathSettings,
)


def test_recycling_1d_reexports_state_and_boundary_dataclasses() -> None:
    assert recycling_1d.Recycling1DRhsResult is Recycling1DRhsResult
    assert recycling_1d.Recycling1DHistoryResult is Recycling1DHistoryResult
    assert recycling_1d.Recycling1DImplicitStepInfo is Recycling1DImplicitStepInfo
    assert recycling_1d._SimpleSheathSettings is SimpleSheathSettings
    assert recycling_1d._FullSheathSettings is FullSheathSettings
    assert recycling_1d._DensityFeedbackTerms is DensityFeedbackTerms
    assert recycling_1d._IonBoundaryResult is IonBoundaryResult
    assert recycling_1d._ElectronBoundaryResult is ElectronBoundaryResult


def test_recycling_1d_result_defaults_feedback_rhs_to_empty_dict() -> None:
    result = Recycling1DRhsResult(variables={"Ne": np.ones((1, 1, 1), dtype=np.float64)})
    assert result.feedback_integral_rhs == {}


def test_recycling_boundary_result_layouts_are_explicit() -> None:
    field = np.ones((1, 1, 1), dtype=np.float64)
    ion_boundary = IonBoundaryResult(
        density={"d+": field},
        pressure={"d+": field},
        temperature={"d+": field},
        velocity={"d+": field},
        momentum={"d+": field},
        energy_source={"d+": field},
    )
    electron_boundary = ElectronBoundaryResult(
        density=field,
        temperature=field,
        pressure=field,
        velocity=field,
        momentum=field,
        energy_source=field,
    )

    assert ion_boundary.density["d+"].shape == (1, 1, 1)
    assert electron_boundary.pressure.shape == (1, 1, 1)
