# Nonlinear DRB2D milestone

This folder contains the 2D nonlinear drift-reduced Braginskii (DRB2D) workflows used
to validate conservative operators, curvature drive, non-Boussinesq polarization, and
SOL-style closed→open setups.

Core scripts:

- `drb2d_movie.py`: baseline DRB2D turbulence run (curvature on, spectral Poisson, Arakawa bracket).
- `drb2d_sol_movie.py`: SOL-style closed→open radial setup with LCFS at `x=x_s` and sheath-like sinks.
- `drb2d_nonbouss_movie.py`: non-Boussinesq polarization movie (variable-coefficient SPD solve).
- `drb2d_energy_budget.py`: term-by-term energy budget diagnostics.

Validation/benchmarks:

- `drb2d_conservative_gate.py`: strict conservative energy drift gate.
- `drb2d_nonbouss_gate.py`: non-Boussinesq stability/diagnostic gate.
- `drb2d_curvature_benchmarks.py`: curvature-drive comparisons (FD/FV).
- `drb2d_linear_phase_benchmark.py`: nonlinear → linear phase benchmark.
- `drb2d_linear_phase_benchmark_em_hot_ion.py`: EM/hot-ion linear-phase benchmarks.

Tips:

- Most scripts accept `--out` and `--seed` flags.
- Use `--fixed-step` and larger `--save-stride` to speed up movies.
