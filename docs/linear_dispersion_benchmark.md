# Linear dispersion benchmarks (B2, B3)

`jax_drb.linear` linearizes a reduced drift-reduced Braginskii model about an
equilibrium and returns the growth rates and frequencies of its eigenmodes.
Writing a perturbation as `delta ~ exp(lambda t)`, the eigenvalues
`lambda = gamma + i*Omega` of the linear operator give the growth rate
`gamma = Re(lambda)` and the oscillation frequency `Omega = Im(lambda)`.

This is both a user-facing tool — the linear solver of the DRB equations — and
the engine behind two literature-anchored benchmark rungs. Each dispersion
operator is assembled directly from the model equations, so that diagonalizing
it *reproduces* the analytic dispersion relation rather than having that
relation wired in.

## B3 — shear-Alfven wave with electron inertia

The reduced electromagnetic two-field model in `(phi, psi)` (potential,
parallel flux) has

```
omega = k_par * v_A / sqrt(1 + (k_perp * d_e)^2)
```

with Alfven speed `v_A = B / sqrt(mu_0 n m_i)` and electron skin depth
`d_e = c / omega_pe`. `jax_drb.linear.shear_alfven_operator` reproduces this to
machine precision across a `(k_par, k_perp)` scan, including the
electron-inertia reduction of the phase velocity at finite `k_perp d_e`.
Reference: Stegmeir et al., *Phys. Plasmas* 26, 052517 (2019).

## B2 — resistive drift wave (Hasegawa-Wakatani)

The two-field `(phi, n)` model with adiabaticity `alpha` and density-gradient
drive `kappa` gives the drift-wave dispersion. In the adiabatic limit
`alpha -> infinity` the frequency tends to the diamagnetic drift frequency
`omega_star = kappa k_y / (1 + k_perp^2)`; at finite `alpha` the parallel
resistivity destabilizes the mode, and the growth rate rises as `alpha`
decreases toward the hydrodynamic regime. References: Dudson et al.,
*Comput. Phys. Commun.* 180, 1467 (2009); Hasegawa & Wakatani,
*Phys. Rev. Lett.* 50, 682 (1983).

## Reproduce

```bash
PYTHONPATH=src python examples/benchmarks/linear_dispersion_demo.py
```

writes `output/linear_dispersion/linear_dispersion.png` (the B3 dispersion
curve and the B2 growth/frequency-vs-adiabaticity curves) and a JSON of the
scan. The gates are in `tests/test_linear_dispersion.py`.
