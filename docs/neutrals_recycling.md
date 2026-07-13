# Neutrals and Recycling

`jax_drb` couples a hydrogenic plasma to a recycled neutral gas through the
hermes-3 atomic reaction model, implemented natively in JAX (differentiable,
self-contained). The physics is exercised on a 1D scrape-off-layer flux tube and,
as source terms, on the 3D flux-coordinate-independent (FCI) geometries -- closed
and open field lines.

![Coupled recycling SOL](https://github.com/uwplasma/jax_drb/releases/download/media-v2.0.0-dev/recycling_sol.png)

## Atomic reactions ([`jax_drb.native.neutrals`](../src/jax_drb/native/neutrals/atomic_rates.py))

Ionization and recombination rate coefficients `<sigma v>(Te, ne)` come from the
packaged AMJUEL double-polynomial fits; charge exchange uses the AMJUEL H.2 3.1.8
polynomial `<sigma v>(Teff)`. The coefficient tables ship with the package
(`jax_drb.data.atomic_rates`), so there is no external-database dependency. The
rates are physically correct -- ionization rises steeply through 3-30 eV,
**recombination rises as the plasma cools** (the detachment driver), and charge
exchange grows with the collision energy -- and every routine is
`jit`/`grad`/`vmap` transparent.

[`compute_hydrogen_reaction_sources`](../src/jax_drb/native/neutrals/reactions.py)
assembles the plasma <-> neutral source channels following the hermes-3 closure:
Galilean-invariant particle and momentum transfer (each transfer carries the
source species' `m V` and `1.5 T`), a charge-exchange frictional heating
`0.5 m R dV^2`, and the electron ionization-cost / recombination-radiation channel
from the AMJUEL energy-loss fits. The ion and neutral particle and momentum
sources cancel exactly.

## 1D recycling SOL ([`recycling_sol_model`](../src/jax_drb/native/neutrals/recycling_sol_model.py))

The coupled 1D model transports the plasma to the target on a prescribed
hot-upstream / cold-target temperature profile (the imposed-temperature closure,
which sidesteps the stiff self-consistent conduction/radiation balance), with a
neutral that is recycled from the Bohm ion flux at the target, transported by
parallel diffusion (solved implicitly with a solvax tridiagonal solve so the
parabolic term is unconditionally stable), and ionized/recombined back into the
plasma (operator-split, per-cell implicit against the stiff ionization source).
The charge-exchange + recombination momentum friction drags the flow toward the
neutrals.

The figure shows the result: neutrals recycled at the target build a **cushion**
there, ionization feeds the plasma, and -- crucially -- as the upstream density
rises the friction **chokes the parallel flow**, so the target Mach number falls
toward and below 1 (right panel): the onset of detachment. The gate
[`tests/test_native_recycling_sol.py`](../tests/test_native_recycling_sol.py)
pins stability, the ionization/recombination spatial structure, the neutral
cushion, the detachment-onset Mach trend, and differentiability.

## Neutrals on 3D FCI field lines

Because the reactions act on any field shape and the FCI sheath closure acts on
the traced target endpoints, the coupling carries over to the 3D FCI geometries.
[`tests/test_fci_neutrals_3d.py`](../tests/test_fci_neutrals_3d.py) checks that on
the genuinely non-axisymmetric **closed** rotating ellipse the reaction sources
conserve particles and momentum cell-by-cell and integrate to zero over the
metric-Jacobian-weighted volume, and that on the **open** slab the Bohm-sheath
recycling turns the target ion flux into a neutral source matching the recycled
accounting (residuals ~1e-16), landing only on the two open target planes.

## Scope

Self-consistent detachment -- an *evolved* temperature with Spitzer parallel
conduction and radiative rollover -- is the further extension. Its stiff energy
balance (the radiative loss must self-limit as the temperature drops) needs an
implicit energy solve; the model here uses a prescribed temperature and captures
the recycling, atomic coupling, and detachment *onset*.

## Reproduce

```bash
PYTHONPATH=src python examples/sol/recycling_sol_demo.py
pytest -q tests/test_native_atomic_rates.py tests/test_native_reactions.py \
          tests/test_native_recycling_sol.py tests/test_fci_neutrals_3d.py
```
