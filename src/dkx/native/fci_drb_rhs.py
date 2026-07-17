from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..geometry import FciMaps, MetricTensor3D
from .fci import conservative_perp_diffusion_xz, logical_exb_bracket_xz
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
    potential_boussinesq: bool = True
    plasma_exb_advection_strength: float = 0.0
    potential_iterations: int = 40
    potential_regularization: float = 1.0e-9
    potential_preconditioner: str | None = None


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
        boussinesq=parameters.potential_boussinesq,
        regularization=parameters.potential_regularization,
        preconditioner=parameters.potential_preconditioner,
    )
    vorticity_rhs = conservative_perp_diffusion_xz(
        state.vorticity,
        jnp.ones_like(state.vorticity, dtype=jnp.float64) * parameters.vorticity_diffusivity,
        metric,
    )
    plasma_exb = _plasma_exb_advection_terms(
        state,
        potential_solve.potential,
        metric,
        strength=parameters.plasma_exb_advection_strength,
    )
    rhs = FciDrbState(
        ion_density=sheath.get("Ni", 0.0) + neutral.ion_density_source + plasma_exb.ion_density,
        electron_density=sheath.get("Ne", 0.0) + neutral.electron_density_source + plasma_exb.electron_density,
        neutral_density=sheath.get("Nn", 0.0) + neutral.neutral_density_source,
        ion_pressure=sheath.get("Pi", 0.0) + neutral.ion_pressure_source + plasma_exb.ion_pressure,
        electron_pressure=sheath.get("Pe", 0.0) + neutral.electron_pressure_source + plasma_exb.electron_pressure,
        neutral_pressure=sheath.get("Pn", 0.0) + neutral.neutral_pressure_source,
        ion_momentum=neutral.ion_momentum_source + plasma_exb.ion_momentum,
        neutral_momentum=neutral.neutral_momentum_source,
        vorticity=vorticity_rhs + plasma_exb.vorticity,
    )
    return FciDrbRhsResult(
        rhs=rhs,
        potential=potential_solve.potential,
        potential_residual_l2=potential_solve.residual_l2,
    )


def _plasma_exb_advection_terms(
    state: FciDrbState,
    potential: jax.Array,
    metric: MetricTensor3D,
    *,
    strength: float,
) -> FciDrbState:
    """Return compact ExB advection terms driven by the solved potential.

    The compact FCI DRB lane treats ExB advection as a plasma perpendicular
    drift, so it acts on charged-fluid density, pressure, parallel momentum,
    and vorticity. Neutral gas density, pressure, and momentum remain governed
    by the neutral reaction/diffusion closures rather than by ExB drift.
    """

    coefficient = float(strength)
    zero = jnp.zeros_like(state.neutral_density, dtype=jnp.float64)
    if coefficient == 0.0:
        return FciDrbState(
            ion_density=jnp.zeros_like(state.ion_density, dtype=jnp.float64),
            electron_density=jnp.zeros_like(state.electron_density, dtype=jnp.float64),
            neutral_density=zero,
            ion_pressure=jnp.zeros_like(state.ion_pressure, dtype=jnp.float64),
            electron_pressure=jnp.zeros_like(state.electron_pressure, dtype=jnp.float64),
            neutral_pressure=jnp.zeros_like(state.neutral_pressure, dtype=jnp.float64),
            ion_momentum=jnp.zeros_like(state.ion_momentum, dtype=jnp.float64),
            neutral_momentum=jnp.zeros_like(state.neutral_momentum, dtype=jnp.float64),
            vorticity=jnp.zeros_like(state.vorticity, dtype=jnp.float64),
        )

    def advect(field: jax.Array) -> jax.Array:
        return -coefficient * logical_exb_bracket_xz(
            potential,
            field,
            metric,
            periodic_x=False,
            periodic_z=True,
        )

    return FciDrbState(
        ion_density=advect(state.ion_density),
        electron_density=advect(state.electron_density),
        neutral_density=zero,
        ion_pressure=advect(state.ion_pressure),
        electron_pressure=advect(state.electron_pressure),
        neutral_pressure=jnp.zeros_like(state.neutral_pressure, dtype=jnp.float64),
        ion_momentum=advect(state.ion_momentum),
        neutral_momentum=jnp.zeros_like(state.neutral_momentum, dtype=jnp.float64),
        vorticity=advect(state.vorticity),
    )
