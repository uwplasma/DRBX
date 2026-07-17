# Drift-wave turbulence (tokamak, closed field lines)

The closed-field-line tokamak flagship is a JAX-native two-field
Hasegawa-Wakatani drift-wave turbulence model
(`drbx.native.hasegawa_wakatani`), a pseudo-spectral solver for the
perpendicular plane of a periodic flux tube:

```
d/dt zeta = -{phi, zeta} + alpha (phi - n) - nu * lap^2 zeta - mu * zeta
d/dt n    = -{phi, n} - kappa d/dy phi + alpha (phi - n) - nu * lap^2 n - mu * n
```

with vorticity `zeta = lap phi`, adiabaticity `alpha`, density-gradient drive
`kappa`, hyperviscosity `nu`, and an optional scale-independent friction `mu`
(`HasegawaWakataniParameters.friction`, default 0) representing large-scale
sheath/neutral drag: it absorbs the 2-D inverse cascade so a fixed-step run
reaches a statistically steady saturated state. The `E x B` Poisson bracket is
evaluated pseudo-spectrally with 2/3-rule dealiasing. The whole right-hand side
is JAX, so a run is `jit`-compiled, GPU-portable, and differentiable.

One requirement worth stating explicitly: spectral initial conditions must be
Hermitian (the FFT of a *real* field, e.g. `jnp.fft.fft2(real_noise)`). A
non-Hermitian complex seed evolves the unphysical complexified system, which
violates the real system's energy balance and blows up at finite amplitude
independently of the timestep.

## Linear phase is benchmark-verified

A single Fourier mode carries zero self-bracket, so it evolves purely linearly.
Its growth rate reproduces the eigenvalue of
`drbx.linear.resistive_drift_wave_operator` to machine precision -- the same
operator used for the B2 dispersion benchmark. This ties the nonlinear flagship
directly to the [linear dispersion benchmarks](linear_dispersion_benchmark.md).

## Instability growth and transport

From small noise the model grows through the linear drift-wave instability and
develops an outward radial `E x B` particle flux `<n v_x> > 0` -- density
transported down the background gradient. With a Hermitian seed and a small
friction, the shipped fixed-step example runs the full life cycle: four decades
of exponential growth through the linear instability, then nonlinear saturation
with a statistically steady particle-flux plateau. Gates in
`tests/test_hasegawa_wakatani.py` cover the linear cross-check, the ideal
energy invariant (no drive/coupling/dissipation), the transport direction, and
end-to-end differentiability (the final fluctuation energy has a finite,
finite-difference-verified gradient with respect to the adiabaticity).

## Differentiable inverse design

Because the whole run is JAX, the gradient of any diagnostic with respect to any
model parameter is available by autodiff -- through the entire nonlinear time
evolution. `examples/tokamak/drift_wave_inverse_design.py` uses this to
recover the density-gradient drive that produced a target fluctuation-energy
level by gradient descent, and reports transport sensitivities
(`d(energy)/d(kappa)`) that match finite differences exactly. To our knowledge
no other drift-reduced Braginskii edge code can optimize through turbulence this
way. The gate is `test_inverse_design_recovers_a_parameter_through_turbulence`.

## Turbulence optimization: the hydrodynamic-to-adiabatic transition

`examples/tokamak/hasegawa_wakatani_optimization.py` turns the classic
adiabaticity dependence of Hasegawa-Wakatani transport (Camargo, Biskamp &
Scott, *Phys. Plasmas* 2, 48 (1995)) into a gradient-based design problem: find
the adiabaticity at which the saturated particle flux drops to a quarter of its
hydrodynamic-regime value. A safeguarded damped Newton iteration on
`ln(alpha)` uses forward-mode gradients (`jax.jvp` -- one tangent for one
scalar parameter) of a windowed saturated-flux objective through the turbulence
rollout; the differentiated window is kept short because gradients through long
chaotic horizons are Lyapunov-inflated. The iteration converges in 7 solves
(`alpha`: 0.3 -> 0.73) and an independent long verification run measures a
3.96x flux reduction against the 4x target. The example writes a two-column
initial-vs-optimized figure (density snapshot + flux trace, shared scales) to
`output/hasegawa_wakatani_optimization/`.

## Reproduce

```bash
PYTHONPATH=src python examples/tokamak/drift_wave_turbulence.py
PYTHONPATH=src python examples/tokamak/drift_wave_inverse_design.py
PYTHONPATH=src python examples/tokamak/hasegawa_wakatani_optimization.py
```

writes `output/drift_wave_turbulence/drift_wave_turbulence.png` (vorticity
field, fluctuation-energy growth, particle-flux history) and a JSON
time series. References: Hasegawa & Wakatani, *Phys. Rev. Lett.* 50, 682
(1983); Numata et al., *Phys. Plasmas* 14, 102312 (2007).
