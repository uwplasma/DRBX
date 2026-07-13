# Drift-wave turbulence (tokamak, closed field lines)

The closed-field-line tokamak flagship is a JAX-native two-field
Hasegawa-Wakatani drift-wave turbulence model
(`jax_drb.native.hasegawa_wakatani`), a pseudo-spectral solver for the
perpendicular plane of a periodic flux tube:

```
d/dt zeta = -{phi, zeta} + alpha (phi - n) - nu * lap^2 zeta
d/dt n    = -{phi, n} - kappa d/dy phi + alpha (phi - n) - nu * lap^2 n
```

with vorticity `zeta = lap phi`, adiabaticity `alpha`, density-gradient drive
`kappa`, and hyperviscosity `nu`. The `E x B` Poisson bracket is evaluated
pseudo-spectrally with 2/3-rule dealiasing. The whole right-hand side is JAX, so
a run is `jit`-compiled, GPU-portable, and differentiable.

## Linear phase is benchmark-verified

A single Fourier mode carries zero self-bracket, so it evolves purely linearly.
Its growth rate reproduces the eigenvalue of
`jax_drb.linear.resistive_drift_wave_operator` to machine precision -- the same
operator used for the B2 dispersion benchmark. This ties the nonlinear flagship
directly to the [linear dispersion benchmarks](linear_dispersion_benchmark.md).

## Instability growth and transport

From small noise the model grows through the linear drift-wave instability and
develops an outward radial `E x B` particle flux `<n v_x> > 0` -- density
transported down the background gradient. Reaching a deep, statistically
stationary saturated state needs CFL-adaptive time stepping (a Phase 7
performance item); the shipped example runs a bounded fixed-step window. Gates in
`tests/test_hasegawa_wakatani.py` cover the linear cross-check, the ideal
energy invariant (no drive/coupling/dissipation), the transport direction, and
end-to-end differentiability (the final fluctuation energy has a finite,
finite-difference-verified gradient with respect to the adiabaticity).

## Reproduce

```bash
PYTHONPATH=src python examples/tokamak/drift_wave_turbulence_demo.py
```

writes `output/drift_wave_turbulence/drift_wave_turbulence.png` (vorticity
field, fluctuation-energy growth, particle-flux history) and a JSON
time series. References: Hasegawa & Wakatani, *Phys. Rev. Lett.* 50, 682
(1983); Numata et al., *Phys. Plasmas* 14, 102312 (2007).
