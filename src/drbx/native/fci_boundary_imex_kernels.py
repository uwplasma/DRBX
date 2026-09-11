"""Reusable isolated kernels for the coupled-boundary residual."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from .fci_boundary_imex_model import CoupledBoundaryStageContext


def build_coupled_residual_kernel(
    model: Any, *, residual_fn: Callable | None = None
) -> Callable:
    """Build a JIT residual kernel with only model geometry static.

    The returned function has signature ``(vector, base_state, solve_dt,
    source)``.  ``base_state``, ``solve_dt``, and ``source`` remain dynamic, so
    the compiled executable can be reused across stages and timesteps.  The
    optional ``residual_fn`` is a small-test hook; production callers use the
    coupled context residual directly.
    """
    if residual_fn is None:
        def residual_fn(vector, base_state, solve_dt, source):
            context = CoupledStageContext(model, base_state, solve_dt, source)
            return context.residual(vector)

    # Close over only the immutable model and function.  Do not capture a
    # timestep, source, or base state here.
    # The coupled residual contains large static concatenations from the
    # material/owner packing path.  Disable only XLA's constant-folding pass
    # for this isolated executable; global JAX/XLA policy is untouched.
    return jax.jit(residual_fn, compiler_options={"xla_disable_hlo_passes": "constant_folding"})


# Local alias keeps the production closure above readable and avoids exposing
# the adapter's implementation detail as part of this module's public API.
CoupledStageContext = CoupledBoundaryStageContext

__all__ = ["build_coupled_residual_kernel"]
