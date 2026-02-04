# Neutral interactions (minimal model)

Nonlinear SOL simulations often require neutral physics (ionization, recombination, charge-exchange, recycling). As a first step, `jaxdrb` includes a **minimal neutral density** model that can be toggled on/off.

## Additional field

When enabled, the state becomes:

$$
y = (n, \omega, N),
$$

where $N(x,y,t)$ is a neutral particle density.

## Neutral equation

The neutral model is:

$$
\partial_t N + [\phi, N] = D_N \nabla_\perp^2 N + S_0 - \nu_s N - S_{\text{ion}} + S_{\text{rec}}.
$$

Ionization and recombination are modeled as:

$$
S_{\text{ion}} = \nu_{\text{ion}}\,n_{\text{abs}}\,N_{\text{abs}},\qquad
S_{\text{rec}} = \nu_{\text{rec}}\,n_{\text{abs}},
$$

where, in the HW2D milestone, $n$ is treated as a fluctuation about a constant background
$n_{\text{abs}} = n_0 + n$ (with small floors applied to keep the rates physical).

## Coupling to the plasma density

Particle exchange between neutrals and plasma enters the density equation as:

$$
\partial_t n \;\;\leftarrow\;\; \partial_t n + S_{\text{ion}} - S_{\text{rec}}.
$$

With $S_0=0$, $D_N=0$, and $\nu_s=0$, this choice conserves the domain-mean total particle content:

$$
\frac{d}{dt}\langle n + N \rangle = 0.
$$

## What this is (and is not)

- This minimal model is meant to be **physically motivated** and **testable**, not complete.
- It provides clean hooks for upcoming additions:
  - charge-exchange momentum sinks,
  - energy loss terms due to ionization/radiation,
  - recycling sources tied to sheath fluxes and geometry,
  - kinetic neutral closures (or coupling to external neutral solvers).

## References (SOL context)

Plasma–neutral coupling is a core ingredient of quantitative SOL modeling. For broader context and
state-of-the-art nonlinear SOL simulations that include neutrals and sheath physics, see:

- P. Ricci et al., *Simulation of plasma turbulence in scrape-off layer conditions: the GBS code*,
  Plasma Phys. Control. Fusion 54, 124047 (2012).
- F. D. Halpern et al., *Model extensions for SOL simulations (neutrals, sheath, etc.)* (2016) (PDF in `drb_literature/`).
- G. Bufferand et al., *Review of SOL turbulence modeling*, Nucl. Fusion 61, 116052 (2021).
