from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..geometry import FciMaps, MetricTensor3D
from .fci import conservative_perp_diffusion_xz
from .fci_neutral import compute_fci_neutral_reaction_diffusion
from .fci_sheath_recycling import fci_sheath_recycling_field_rhs
from .fci_vorticity import solve_fci_vorticity_potential_cg


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FciDrbState:
    ion_density: jax.Array
    electron_density: jax.Array
    neutral_density: jax.Array
    ion_pressure: jax.Array
    electron_pressure: jax.Array
    neutral_pressure: jax.Array
    ion_momentum: jax.Array
    neutral_momentum: jax.Array
    vorticity: jax.Array

    def tree_flatten(self):
        return (
            (
                self.ion_density,
                self.electron_density,
                self.neutral_density,
                self.ion_pressure,
                self.electron_pressure,
                self.neutral_pressure,
                self.ion_momentum,
                self.neutral_momentum,
                self.vorticity,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@dataclass(frozen=True)
class FciDrbRhsParameters:
    recycling_fraction: float = 0.98
    recycled_neutral_energy: float = 0.03
    vorticity_diffusivity: float = 2.0e-4
    potential_iterations: int = 40


@dataclass(frozen=True)
class FciDrbRhsResult:
    rhs: FciDrbState
    potential: jax.Array
    potential_residual_l2: jax.Array


def compute_fci_drb_rhs(
    state: FciDrbState,
    *,
    maps: FciMaps,
    metric: MetricTensor3D,
    parameters: FciDrbRhsParameters = FciDrbRhsParameters(),
) -> FciDrbRhsResult:
    """Assemble the first transformable non-axisymmetric DRB component RHS."""

    fields = {
        "Ni": state.ion_density,
        "Ne": state.electron_density,
        "Nn": state.neutral_density,
        "Pi": state.ion_pressure,
        "Pe": state.electron_pressure,
        "Pn": state.neutral_pressure,
    }
    sheath = fci_sheath_recycling_field_rhs(
        fields,
        maps,
        recycling_fraction=parameters.recycling_fraction,
        recycled_neutral_energy=parameters.recycled_neutral_energy,
    )
    neutral = compute_fci_neutral_reaction_diffusion(
        neutral_density=state.neutral_density,
        neutral_pressure=state.neutral_pressure,
        neutral_momentum=state.neutral_momentum,
        ion_density=state.ion_density,
        ion_pressure=state.ion_pressure,
        ion_momentum=state.ion_momentum,
        electron_density=state.electron_density,
        electron_pressure=state.electron_pressure,
        maps=maps,
        metric=metric,
    )
    potential_solve = solve_fci_vorticity_potential_cg(
        state.vorticity,
        state.ion_density,
        metric,
        iterations=parameters.potential_iterations,
    )
    vorticity_rhs = conservative_perp_diffusion_xz(
        state.vorticity,
        jnp.ones_like(state.vorticity, dtype=jnp.float64) * parameters.vorticity_diffusivity,
        metric,
    )
    rhs = FciDrbState(
        ion_density=sheath.get("Ni", 0.0) + neutral.ion_density_source,
        electron_density=sheath.get("Ne", 0.0) + neutral.electron_density_source,
        neutral_density=sheath.get("Nn", 0.0) + neutral.neutral_density_source,
        ion_pressure=sheath.get("Pi", 0.0) + neutral.ion_pressure_source,
        electron_pressure=sheath.get("Pe", 0.0) + neutral.electron_pressure_source,
        neutral_pressure=sheath.get("Pn", 0.0) + neutral.neutral_pressure_source,
        ion_momentum=neutral.ion_momentum_source,
        neutral_momentum=neutral.neutral_momentum_source,
        vorticity=vorticity_rhs,
    )
    return FciDrbRhsResult(
        rhs=rhs,
        potential=potential_solve.potential,
        potential_residual_l2=potential_solve.residual_l2,
    )
