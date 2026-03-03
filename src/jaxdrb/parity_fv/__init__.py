"""Hermes-parity finite-volume core (new rewrite path).

This package hosts the strict parity-first implementation used to align
jax_drb against Hermes before broader model extensions are reintroduced.
"""

from .params import ParityFVParams
from .state import ParityFVState
from .rhs import ParityFVRHS
from .flux_parallel import div_parallel_fv

__all__ = ["ParityFVParams", "ParityFVState", "ParityFVRHS", "div_parallel_fv"]
