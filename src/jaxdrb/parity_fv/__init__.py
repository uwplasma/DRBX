"""Hermes-parity finite-volume core (new rewrite path).

This package hosts the strict parity-first implementation used to align
jax_drb against Hermes before broader model extensions are reintroduced.
"""

from .params import ParityFVParams
from .state import ParityFVState
from .rhs import ParityFVRHS

__all__ = ["ParityFVParams", "ParityFVState", "ParityFVRHS"]
