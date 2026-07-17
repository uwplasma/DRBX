from __future__ import annotations

import math

from drbx.config.boutinp import parse_bout_input
from drbx.config.normalization import ELEMENTARY_CHARGE, PROTON_MASS, ModelNormalization


def test_model_normalization_matches_reference_formulas() -> None:
    config = parse_bout_input(
        """
        [mesh]
        Bxy = 0.35

        [model]
        Nnorm = 2e18
        Tnorm = 5
        Bnorm = mesh:Bxy
        normalise_metric = true
        recalculate_metric = false
        """
    )

    normalization = ModelNormalization.from_config(config)
    expected_cs0 = math.sqrt(ELEMENTARY_CHARGE * 5.0 / PROTON_MASS)
    expected_omega_ci = ELEMENTARY_CHARGE * 0.35 / PROTON_MASS

    assert math.isclose(normalization.Nnorm, 2e18)
    assert math.isclose(normalization.Tnorm, 5.0)
    assert math.isclose(normalization.Bnorm, 0.35)
    assert math.isclose(normalization.Cs0, expected_cs0)
    assert math.isclose(normalization.Omega_ci, expected_omega_ci)
    assert math.isclose(normalization.rho_s0, expected_cs0 / expected_omega_ci)
    assert normalization.metric_policy.normalise_metric is True
    assert normalization.metric_policy.recalculate_metric is False
    assert math.isclose(normalization.units["seconds"], 1.0 / expected_omega_ci)
    assert math.isclose(normalization.units["meters"], expected_cs0 / expected_omega_ci)
