"""Neutral / recycling physics for the 1D scrape-off-layer model.

Self-contained JAX implementations of the hydrogenic atomic reaction rates
(AMJUEL ionization / recombination fits, the Janev/AMJUEL charge-exchange
polynomial) used to couple a plasma fluid to a neutral fluid in the
scrape-off layer. All coefficients ship with the package
(``drbx.data.atomic_rates``); there is no external-database dependency.
"""

from .atomic_rates import (
    charge_exchange_rate_coefficient,
    energy_loss_coefficient,
    eval_amjuel_fit,
    load_amjuel_coefficients,
    rate_coefficient,
)
from .reactions import (
    HydrogenReactionSources,
    PlasmaNormalization,
    compute_hydrogen_reaction_sources,
)
from .detachment_sol_model import (
    DetachmentSolParameters,
    DetachmentSolState,
    detachment_diagnostics,
    detachment_sol_run,
    detachment_sol_step,
)
from .recycling_sol_model import (
    SolRecyclingParameters,
    SolRecyclingState,
    linear_target_temperature_profile,
    sol_recycling_run,
    sol_recycling_step,
    target_ion_flux,
)

__all__ = [
    "rate_coefficient",
    "energy_loss_coefficient",
    "charge_exchange_rate_coefficient",
    "eval_amjuel_fit",
    "load_amjuel_coefficients",
    "PlasmaNormalization",
    "HydrogenReactionSources",
    "compute_hydrogen_reaction_sources",
    "SolRecyclingParameters",
    "SolRecyclingState",
    "linear_target_temperature_profile",
    "sol_recycling_run",
    "sol_recycling_step",
    "target_ion_flux",
    "DetachmentSolParameters",
    "DetachmentSolState",
    "detachment_diagnostics",
    "detachment_sol_run",
    "detachment_sol_step",
]
