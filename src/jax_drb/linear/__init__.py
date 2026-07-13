"""Linear stability and dispersion analysis for the DRB models.

This subpackage linearizes a reduced drift-reduced Braginskii model about an
equilibrium and returns the growth rates and frequencies of its eigenmodes. It
is both a user-facing tool ("the linear solver of the DRB equations") and the
engine behind the drift-wave and shear-Alfven dispersion benchmarks (B2, B3).
"""

from __future__ import annotations

from .dispersion import (
    drift_wave_adiabatic_frequency,
    resistive_drift_wave_operator,
    shear_alfven_frequency,
    shear_alfven_operator,
)
from .eigen import (
    LinearModes,
    dominant_mode,
    eigenmodes,
    jacobian_operator,
)

__all__ = [
    "LinearModes",
    "jacobian_operator",
    "eigenmodes",
    "dominant_mode",
    "shear_alfven_operator",
    "shear_alfven_frequency",
    "resistive_drift_wave_operator",
    "drift_wave_adiabatic_frequency",
]
