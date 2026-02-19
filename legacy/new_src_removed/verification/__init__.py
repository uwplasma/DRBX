"""Verification models and utilities.

This package contains small, literature-anchored verification problems used to validate
numerical operators and solver workflows (without requiring a full SOL nonlinear run).
"""

from .gdb2018 import saw_linear_matrix, saw_phase_speed_sq

__all__ = ["saw_linear_matrix", "saw_phase_speed_sq"]
