# Nonlinear roadmap

`jaxdrb` started as a linear, field-line (flux-tube) drift-reduced Braginskii (DRB) solver. The next major capability is solving **nonlinear** systems efficiently in JAX while keeping the code modular and physics-aligned.

This section introduces the first nonlinear milestone included in the codebase:

- A **2D periodic nonlinear drift-wave testbed** (Hasegawa–Wakatani-like), used to validate and benchmark:
  - Poisson bracket implementations (conservative finite-difference and pseudo-spectral),
  - FFT-based polarization (Poisson) solves,
  - dealiasing,
  - time stepping with JAX + Diffrax,
  - optional coupling to a **neutral density** field.
  - operator verification via the **method of manufactured solutions** (MMS).

The purpose is not to replace SOL-specific DRB models. Rather, it is a fast and controlled environment to:

- test numerical kernels (operators and time stepping),
- validate invariants (when applicable),
- prepare the code structure for the eventual nonlinear DRB system (including open-field-line boundary conditions, sources/sinks, and additional closures).

In addition, `jaxdrb` now includes a **periodic cold-ion DRB conservative gate** on the actual field-line
branch (energy/mass/charge/current/momentum invariants), with both finite-time and operator-level checks:

- test: `tests/test_drb_nonlinear_conservative_gate.py`
- test: `tests/test_drb_operator_rates.py`
- example: `examples/10_verification/drb_cold_ion_conservative_gate.py`
- example: `examples/10_verification/drb_cold_ion_operator_gate.py`
- CI benchmark: `benchmarks/check_drb_conservative_gate.py`

![Cold-ion DRB strict conservative operator gate](../assets/images/drb_cold_ion_operator_gate.png)

![Cold-ion DRB operator split diagnostics](../assets/images/drb_operator_split_diagnostics.png)

## Run the nonlinear examples

From the repository root:

```bash
python examples/08_nonlinear_hw2d/hw2d_driftwave_turbulence.py
python examples/08_nonlinear_hw2d/hw2d_neutrals_effect.py
python examples/08_nonlinear_hw2d/hw2d_movie.py
python examples/08_nonlinear_drb2d/drb2d_movie.py
python examples/08_nonlinear_drb2d/drb2d_linear_phase_benchmark.py

python examples/08_nonlinear_drb2d/drb2d_conservative_gate.py
python examples/04_closures_transport/nonlinear_flux_tube_toggles.py
```

Both examples write results to small `out_*` folders with plots and `.npz` data.
