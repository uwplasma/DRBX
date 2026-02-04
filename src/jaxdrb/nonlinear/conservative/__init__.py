"""Conservative nonlinear building blocks.

This subpackage contains small, reusable utilities for **conservative** nonlinear evolution,
intended to support the transition from the current HW2D milestone to conservative nonlinear
drift-reduced Braginskii (DRB) formulations.

See `docs/nonlinear/conservative-drb.md` and `docs/roadmap.md`.
"""

from .checks import energy_drift, energy_time_series

__all__ = ["energy_drift", "energy_time_series"]
