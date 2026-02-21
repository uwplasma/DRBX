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
- `bc_enforce_nu_phi` is applied directly to **phi** (operator-split), not via vorticity damping.
- `bc_physical.phi_boundary_timescale` matches Hermes `phi_boundary_timescale` and is normalized
  using `t_ref = L_ref / c_s`.
- `numerics.poisson_scale` is set from the Hermes omega/phi audit (see below).

Axis mapping in analysis:
- Hermes file axes are `(x, y, z)`. For comparisons with `jax_drb` (parallel `z`), the report
  uses `--hermes-axes xzy` so Hermes `y` (parallel) maps to `z`.
- The benchmark report derives `k_y` using Hermes grid metrics (`gyy_ballooning`) and `ShiftAngle`
  via `--hermes-ky-scale metric_shift`, which brings ky peaks into alignment across codes.

## Normalization audit (Poisson / omega / phi)

- `jax_drb` normalization uses `phi_norm = e * phi_phys / Te0` and `t_norm = t_phys / (Lref / c_s)`.
- With lengths normalized by `Lref`, the Poisson relation becomes:
  - `omega = (rho_s / Lref)^2 * ∇⊥^2 phi`
  - so `numerics.poisson_scale = (rho_s / Lref)^2` when `poisson_scale` is not overridden.
- For benchmarking, Hermes `phi` is stored in normalized units with `conversion = Tnorm`
  (see Hermes output attributes). This should match `jax_drb` normalization if `Te0_eV`
  and `Tnorm` are aligned.
- Metric Laplacian estimate (using `gxx_ballooning/gxy_ballooning/gyy_ballooning` from `salpha.nc`):
  - Using physical spacings (`dx = dr/(nx-2*mxg)`, `dy = hthe*dy`) yields `poisson_scale ≈ 2.08e-8`,
    which matches `(rho_s / Lref)^2` for `Te0_eV = 1`, `B0 = 1`, `m_i = 2 amu`, `Lref = 1 m`.
  - This supports using the metric-consistent Poisson operator with the normalization-based scale.
- `sol_source_n0` and `sol_source_Te0` are now scaled in `normalization.py` as
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

## Known mismatches (to resolve)

1. **Linear alignment stability vs. scaling**  
   The alignment presets now use the metric-consistent Poisson operator with
   `poisson_scale = (rho_s / Lref)^2 ≈ 2.1e-8` (set via normalization).
   We need to re-run the linear case and confirm stability/growth rates under this
   scaling. If instabilities persist, adjust `dt`/diffusion while keeping the
   metric Poisson scaling intact.

2. **Ky peak mismatch**  
   Hermes and `jax_drb` use different effective poloidal wavenumbers even after axis mapping.
   Ensure poloidal length scales (Ly) are consistent and verify mapping from Hermes grid to
   `jax_drb` axisymmetric coefficients.

3. **Nonlinear stability**  
   With `poisson_scale = 1e-2`, strong diffusion, and smaller initial perturbations, the
   nonlinear run now survives to `t ≈ 12.7` before NaNs appear. Further stabilization likely
   requires either smaller `dt`, even stronger diffusion/hyperdiffusion, or additional damping
   terms that match Hermes closures.

3. **Growth/decay rate mismatch**  
   Hermes shows positive phi growth while `jax_drb` shows weak or negative growth in `n`.
   Remaining differences likely from term selection (Braginskii closures, resistive
   damping, parallel losses) and boundary handling.

These issues are tracked so the benchmark can be tightened to publication quality.
