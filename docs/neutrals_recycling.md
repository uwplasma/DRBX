# Neutrals and Recycling

`drbx` couples a hydrogenic plasma to a recycled neutral gas through the
hermes-3 atomic reaction model, implemented natively in JAX (differentiable,
self-contained). The physics is exercised on a 1D scrape-off-layer flux tube and,
as source terms, on the 3D flux-coordinate-independent (FCI) geometries -- closed
and open field lines.

![Coupled recycling SOL](media/recycling_sol.png)

## Atomic reactions ([`drbx.native.neutrals`](../src/drbx/native/neutrals/atomic_rates.py))

Ionization and recombination rate coefficients `<sigma v>(Te, ne)` come from the
packaged AMJUEL double-polynomial fits; charge exchange uses the AMJUEL H.2 3.1.8
polynomial `<sigma v>(Teff)`. The coefficient tables ship with the package
(`drbx.data.atomic_rates`), so there is no external-database dependency. The
rates are physically correct -- ionization rises steeply through 3-30 eV,
**recombination rises as the plasma cools** (the detachment driver), and charge
exchange grows with the collision energy -- and every routine is
`jit`/`grad`/`vmap` transparent.

[`compute_hydrogen_reaction_sources`](../src/drbx/native/neutrals/reactions.py)
assembles the plasma <-> neutral source channels following the hermes-3 closure:
Galilean-invariant particle and momentum transfer (each transfer carries the
source species' `m V` and `1.5 T`), a charge-exchange frictional heating
`0.5 m R dV^2`, and the electron ionization-cost / recombination-radiation channel
from the AMJUEL energy-loss fits. The ion and neutral particle and momentum
sources cancel exactly.

## 1D recycling SOL ([`recycling_sol_model`](../src/drbx/native/neutrals/recycling_sol_model.py))

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

## Self-consistent detachment (B6)

[`detachment_sol_model`](../src/drbx/native/neutrals/detachment_sol_model.py)
evolves the plasma pressure as well, so the target temperature responds
self-consistently to the plasma conditions -- the ingredient a detachment study
needs. It adds **Spitzer parallel conduction** `kappa ~ T^{5/2}` solved
implicitly (a solvax tridiagonal, so the stiff parabolic heat transport is
unconditionally stable) and a **self-limiting radiative / ionization energy
loss** (applied semi-implicitly, `P <- P / (1 + dt * loss_rate)`, so the loss
cannot drive the pressure negative and switches off as the plasma cools), on top
of the Bohm sheath heat sink and the recycling neutral coupling.

![B6 detachment rollover](media/b6_detachment.png)

Scanning the upstream density at fixed upstream power reproduces the classic
SD1D detachment picture (Dudson et al., *PPCF* 61, 065008 (2019)): the target
cools from an attached hot target through a sharp thermal collapse into the
recombining regime **below 1 eV**, and the target ion flux rises then **rolls
over** — on the shipped scan the rollover sits at upstream density
`n_up = 8` with a **23% flux reduction** in the deepest detached point. The gate
[`tests/test_native_detachment_sol.py`](../tests/test_native_detachment_sol.py)
pins the monotonic cooling, the attached/detached temperatures, the flux
rollover, and differentiability; the example
[`examples/benchmarks/b6_detachment_rollover.py`](../examples/benchmarks/b6_detachment_rollover.py)
draws the figure. The whole solve is differentiable, so the detachment front
responds to `jax.grad` -- the basis for gradient-based detachment control.

## Gradient-based detachment control

Because the detaching solve is differentiable, the exhaust-control problem
becomes a gradient computation: find the upstream density that places the
target exactly at the 1 eV detachment threshold. The sensitivity
`dTe_target/dn_up` comes from forward-mode autodiff through the entire
20,000-step stiff solve, and a trust-region Newton iteration (contracting when
the residual changes sign, since the detachment cliff is steeper than any local
derivative) converges onto the threshold in ~11 solves:

![Detachment control](media/detachment_control.png)

The gate [`tests/test_detachment_control.py`](../tests/test_detachment_control.py)
verifies the autodiff sensitivity against a central finite difference (to 1e-4)
in the attached regime and the sign of the physics (raising the upstream
density cools the target). Reproduce with
`examples/autodiff/detachment_control.py`.

## Reproduce

Both examples are flat scripts: the physically meaningful knobs —
`PlasmaNormalization(Tnorm=...)`, the connection length, the conduction and
sheath-transmission coefficients, recycling fraction, and the density scans —
are top-of-file constants. They print stage-by-stage progress (setup block,
per-chunk relaxation lines with target Mach/flux/temperature, per-density
convergence lines, and the final rollover summary).

```bash
PYTHONPATH=src python examples/sol/recycling_sol.py
PYTHONPATH=src python examples/benchmarks/b6_detachment_rollover.py
pytest -q tests/test_native_atomic_rates.py tests/test_native_reactions.py \
          tests/test_native_recycling_sol.py tests/test_fci_neutrals_3d.py \
          tests/test_native_detachment_sol.py
```
