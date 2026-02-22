# Hermes vs jax_drb benchmark (s-alpha)

This benchmark compares a minimal, isothermal s-alpha case between Hermes-3 and `jax_drb`
using the aligned axisymmetric coefficients derived from the Hermes grid. The goal is to
validate qualitative behavior (growth/decay trends, spectra, snapshots) while keeping the
physics subset consistent and fully differentiable in `jax_drb`.

## Configuration

Hermes-3 (linear/nonlinear):
- `/Users/rogerio/local/jax_drb/benchmarks/cases/hermes_salpha_linear/BOUT.inp`
- `/Users/rogerio/local/jax_drb/benchmarks/cases/hermes_salpha_nonlinear/BOUT.inp`
- Grids: `salpha.nc` with injected `bxcv` curvature vectors to activate vorticity drive.

jax_drb:
- `/Users/rogerio/local/jax_drb/benchmarks/cases/jaxdrb/salpha_linear.toml`
- `/Users/rogerio/local/jax_drb/benchmarks/cases/jaxdrb/salpha_nonlinear.toml`
- Alignment preset: `/Users/rogerio/local/jax_drb/benchmarks/cases/jaxdrb/benchmark_alignment_hermes_salpha.toml`
- Alignment (nonlinear): `/Users/rogerio/local/jax_drb/benchmarks/cases/jaxdrb/benchmark_alignment_hermes_salpha_nonlinear.toml`
- Geometry: axisymmetric coefficients from `hermes_salpha_coeffs.npz`.
- Hermes-mode perturbation preset: `mixmode_terms = ["z", "4z-x"]` with `mixmode_mode = "jax"`.
- `diag_mode = "full"` to ensure RMS(phi) and spectra are reported.
- `trace_stats = true` to record per‑frame mean/max |field| for blowup tracing.
- `bc_enforce_nu_phi` is applied directly to **phi** (operator-split), not via vorticity damping.
- `bc_physical.phi_boundary_timescale` matches Hermes `phi_boundary_timescale` and is normalized
  using `t_ref = L_ref / c_s`.
- Alignment normalization uses `length_unit = "rho_s"` so that `numerics.poisson_scale = 1`
  (see normalization audit below). This matches Hermes' normalized potential units and
  avoids an overly stiff phi response when lengths are scaled by `Lref`.

Axis mapping in analysis:
- Hermes file axes are `(x, y, z)`. For comparisons with `jax_drb` (parallel `z`), the report
  uses `--hermes-axes xzy` so Hermes `y` (parallel) maps to `z`.
- The benchmark report derives `k_y` using Hermes grid metrics (`gyy_ballooning`) and `ShiftAngle`
  via `--hermes-ky-scale metric_shift`, which brings ky peaks into alignment across codes.

## Normalization audit (Poisson / omega / phi)

- `jax_drb` normalization uses `phi_norm = e * phi_phys / Te0` and `t_norm = t_phys / (Lref / c_s)`
  when `length_unit = "lref"`. With `length_unit = "lref"`, the Poisson relation becomes
  ```math
  \Omega = \left(\frac{\rho_s}{L_\mathrm{ref}}\right)^2 \nabla_\perp^2 \phi
  ```
  so `numerics.poisson_scale = (rho_s / Lref)^2` when `poisson_scale` is not overridden.
- Hermes stores `phi` in normalized units with `conversion = Tnorm` (see Hermes output attributes).
  In the Hermes s‑alpha benchmark, the natural normalized length is **rho_s**, so we set
  `length_unit = "rho_s"` and obtain `numerics.poisson_scale = 1`. This avoids an artificially
  stiff phi response (and large parallel acceleration) that appears when `Lref` normalization
  is used with the same grid.
- If you must use `length_unit = "lref"`, ensure `numerics.poisson_scale` matches the
  intended omega–phi normalization and confirm stability with the blowup tracer.
- `sol_source_n0` and `sol_source_Te0` are scaled in `normalization.py` as
  `(t_ref / n0)` and `(t_ref / Te0)` respectively.

## Results

Panels (linear/nonlinear):
- `/Users/rogerio/local/jax_drb/benchmarks/analysis/panels/hermes_jaxdrb_linear_panel.png`
- `/Users/rogerio/local/jax_drb/benchmarks/analysis/panels/hermes_jaxdrb_nonlinear_panel.png`

Summary tables:
- `/Users/rogerio/local/jax_drb/benchmarks/analysis/panels/hermes_jaxdrb_summary.md`
- `/Users/rogerio/local/jax_drb/benchmarks/analysis/panels/hermes_jaxdrb_summary.csv`
  - Includes runtime (s) and max RSS (MB) from `/usr/bin/time -l` logs when provided
    to `benchmark_report.py` via `--hermes-time` and `--jaxdrb-time`.

Alignment outputs:
- Linear alignment run (rho_s normalization):  
  `/Users/rogerio/local/jax_drb/benchmarks/cases/jaxdrb/benchmark_alignment_hermes_salpha_rhos_t20.npz`

## Known mismatches (to resolve)

1. **Ky peak mismatch**  
   Hermes and `jax_drb` use different effective poloidal wavenumbers even after axis mapping.
   Ensure poloidal length scales (Ly) are consistent and verify mapping from Hermes grid to
   `jax_drb` axisymmetric coefficients.

2. **Nonlinear stability**  
   The nonlinear alignment still needs a fully matched physics subset (closures/dissipation)
   and consistent source terms before we can compare saturation levels.

3. **Growth/decay rate mismatch**  
   Hermes shows positive phi growth while `jax_drb` shows weak or negative growth in `n`.
   Remaining differences likely from term selection (Braginskii closures, resistive
   damping, parallel losses) and boundary handling.

These issues are tracked so the benchmark can be tightened to publication quality.
