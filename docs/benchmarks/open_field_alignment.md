# Open-Field Hermes Alignment (Staged)

This workflow aligns `jax_drb` and Hermes on the same open-field tokamak case,
using the same normalization convention and fluctuation diagnostics.

## One-Command Workflow

```bash
cd <repo>
PYTHONPATH=src python tools/run_tokamak_hermes_benchmark.py \
  --jax-config examples/open_field_line/input_tokamak_bxcv_benchmark_es_cold.toml \
  --hermes-data runs/hermes_open_field_short/data \
  --out-dir runs/tokamak_benchmark_latest \
  --fig-dir docs/figures \
  --t-end-short 0.1 \
  --t-end-visual 0.12 \
  --field n
```

Strict start-from-Hermes-state variant:

```bash
cd <repo>
PYTHONPATH=src python tools/run_tokamak_hermes_benchmark.py \
  --jax-config examples/open_field_line/input_tokamak_bxcv_benchmark_hermes_strict.toml \
  --hermes-data runs/hermes_open_field_short/data \
  --out-dir runs/tokamak_benchmark_strict_latest \
  --fig-dir runs/tokamak_benchmark_strict_latest/figures \
  --t-end-short 0.1 \
  --t-end-visual 0.12 \
  --field n \
  --use-hermes-init-state \
  --hermes-init-index 0
```

Outputs:
- `runs/tokamak_benchmark_latest/jax_short.npz`
- `runs/tokamak_benchmark_latest/bundle_jax_short.npz`
- `runs/tokamak_benchmark_latest/bundle_hermes_short.npz`
- `docs/figures/tokamak_sol_benchmark_panel.png`
- `docs/figures/tokamak_sol_poloidal_fluct.png`
- `docs/figures/tokamak_sol_movie.gif`
- `docs/figures/tokamak_sol_3d_movie.gif`
- When `--use-hermes-init-state` is enabled:
  - `<out-dir>/hermes_init_state_t<idx>.npz`

The benchmark panel uses the **poloidal (`x-z`) plane** with tokamak
`Rxy/Zxy` geometry mapping. If the coefficient file includes `mask_open`,
the open/closed boundary is overlaid on the snapshot row.

Base alignment config:
- `examples/open_field_line/input_tokamak_bxcv_benchmark_alignment.toml`
- Calibrated short-window config:
  `examples/open_field_line/input_tokamak_bxcv_benchmark_alignment_calibrated.toml`
- Hermes-like initial-perturbation variant:
  `examples/open_field_line/input_tokamak_bxcv_benchmark_alignment_hermes_init.toml`
- Hermes-equivalent exact IC variant (`n` mixmode + pressure-consistent `Te`):
  `examples/open_field_line/input_tokamak_bxcv_benchmark_alignment_hermes_exact_ic.toml`
- Strict benchmark baseline used by CI:
  `examples/open_field_line/input_tokamak_bxcv_benchmark_hermes_strict.toml`
  (flux-form ExB, `hermes_xppm`, shifted parallel transform, boundary-flux sheath mode).

## 1) Run staged windows with finite-run gating

```bash
cd <repo>
PYTHONPATH=src python tools/run_staged_benchmark.py \
  --config examples/open_field_line/input_tokamak_bxcv_benchmark_alignment.toml \
  --stages short:0.5,onset:1.0,saturated:3.0 \
  --max-growth-factor 200 \
  --max-rms-abs 20 \
  --out-dir runs/staged_open_field
```

Outputs per stage:
- `runs/staged_open_field/jax_<stage>.npz`
- `runs/staged_open_field/bundle_jax_<stage>.npz`

### Poisson/Vorticity strict audit (before any turbulence window)

First, prepare a short Hermes run with dense diagnostics (including
`vorticity:diagnose_terms=true`):

```bash
cd <repo>
PYTHONPATH=src python tools/prepare_hermes_dense_run.py \
  --base-run-dir runs/hermes_open_field_terms_t01 \
  --out-run-dir runs/hermes_open_field_terms_t01_vortterms \
  --hermes-bin <path-to-hermes-3> \
  --nout 10 \
  --timestep 0.01
cd runs/hermes_open_field_terms_t01_vortterms
mpirun -n 6 <path-to-hermes-3> -d data
```

Run the strict operator audit first:

```bash
cd <repo>
PYTHONPATH=src python tools/audit_term_alignment.py \
  --jax-config examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml \
  --hermes-data-dir runs/hermes_open_field_terms_t01_vortterms/data \
  --hermes-input runs/hermes_open_field_terms_t01_vortterms/data/BOUT.inp \
  --hermes-grid runs/hermes_open_field_terms_t01_vortterms/tokamak.nc \
  --out-dir runs/alignment_t01_strict_m1_nocurv_vortterms_v7 \
  --nsteps 3 \
  --start-index 1 \
  --match-hermes-dt \
  --use-hermes-state \
  --use-hermes-phi-in-terms \
  --hermes-parallel-axis y \
  --strict-axis
```

`poisson_alignment.csv` now includes both full-domain and core-cropped metrics:
- `*_corr`, `*_scale`: full domain
- `*_corr_core`, `*_scale_core`: interior (`x[2:-2]`) used for INVERT_SET alignment

For this strict run, the core Poisson forward alignment target is:
- `omega_from_phi_corr_core ≈ 1`
- `omega_from_phi_scale_core ≈ 1`

Early-time alignment-tuned knobs in
`examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml`:
- `parallel_pressure_model = "hermes_vgradp"`
- `parallel_pressure_flux_coeff = 5/3`
- `parallel_pressure_work_coeff = 2/3`
- `parallel_limiter = "none"`
- `parallel_flux_scheme = "rusanov"`
- `exb_flux_scheme = "hermes_xppm"` (Hermes/BOUT XPPM-like MC-limited X-Z transport)
- `exb_poloidal_flows = true`
- `exb_poloidal_scale = 1.0`
- `exb_poloidal_y_scale = 1.24` (strict early-time operator alignment in the
  poloidal Y-flux branch)
- `exb_poloidal_ddy_scheme = "c2"` (DDY-like centered stencil in the X-flux branch)
- `neumann_boundary_average_y = true` (BOUT/Hermes `neumann_boundary_average_z`)
- `parallel_sheath_flux_mode = "boundary_flux"` for `jpar` divergence
- `sheath_energy_flux_scale = 0.95` (early-time calibration of Hermes-flux
  sheath heat transmission in strict state audits)
- `m_i_amu = 1.0` with `me_hat = 1/1836` (time-unit alignment with Hermes dump `Omega_ci`)
- standalone `curvature_on = false` (Hermes-equivalent vorticity curvature is carried by `diamagnetic_current_on`)

`parallel_limiter = "none"` now uses un-limited second-order Fromm slopes
(`s_i = 0.5(f_{i+1}-f_{i-1})`) instead of piecewise-constant reconstruction.
This closes the main early-time drift in strict Hermes-state audits for
parallel channels (`n`, `Pe`, `Te`).

`exb_poloidal_flows` now routes through the metric-coupled X/Y finite-volume
transport path in `FieldAlignedGeometryAdapter.exb_flux_divergence()`:

\[
\nabla\cdot\Gamma_{E\times B}
= \frac{1}{J}\partial_x\left(J v_x f\right)
+ \frac{1}{J}\partial_{\parallel}\left(J v_y f\right),
\quad
v_x \propto \frac{g^{xx} g_{23}}{B^2}\partial_{\parallel}\phi,\;
v_y \propto -\frac{g^{xx} g_{23}}{B^2}\partial_x\phi
\]

with field-aligned shifted-metric handling on the parallel branch. This closes
the previous structural gap where `exb_poloidal_flows` existed in config but
was not applied in the active geometry adapter.

The radial boundary reconstruction in this branch now uses two Neumann ghost
layers for inflow faces (matching BOUT Neumann guard-cell behavior), which
reduced the leading `n advection exb` mismatch in strict early-time audits.

Axisymmetric coefficient files now carry `metric_dx`, `metric_dy`, and
`metric_dz` (from Hermes dump/grid `dx`, `dy`, `dz`), and the metric-coupled
ExB FV path consumes these local cell sizes in the X-Z and X-Y branches.
In strict early-time Hermes-state audits this reduced the dominant
`n advection exb` mismatch from about `1.91` to about `0.014` at the first
audited step (`t=0.01`), while keeping the 3-step window finite and stable.
(`rel_diff = |rms_jax-rms_hermes| / (0.1*rms_hermes)`), with the matching
pressure-channel advection term `Pe exb` reduced to about `0.09`.

With the strict Hermes-state audit (`start_index=1`, `nsteps=3`), the dominant
RHS alignment channels are:
- `omega total RHS vs ddt(Vort)`: rel-diff `~0.84 .. 1.13` (about 8–11%)
- `omega parallel (jax vs term_Vort_jpar)`: rel-diff `~0.01 .. 0.03`
- `omega diamagnetic current (jax vs term_Vort_divJdia)`: rel-diff `~0.42`
  (about 4% RMS gap)
- `n total RHS vs ddt(Ne)`: rel-diff `~0.92 .. 1.10` (about 9–11%)
- `Te total RHS vs ddt(Te)`: rel-diff `~1.85 .. 2.53` (about 18–25%)

The `DivJdia` channel now applies mass weighting by default when
`poisson_b_weighted=true` and `poisson_b_weighted_mode="hermes"`:

\[
\partial_t \omega \supset \bar{A}\,\nabla\cdot\mathbf{J}_{\mathrm{dia}}
\]

controlled by `physics.diamagnetic_current_mass_weighted=true`. This closes the
previous structural scale gap between Hermes vorticity normalization and the
JAX diamagnetic current term.

After enabling true Fromm behavior for `parallel_limiter="none"` the strict
term-level projection error in parallel channels dropped significantly in the
same (`start_index=1`, `nsteps=3`) audit window:
- `n parallel`: weighted-rel `~0.017..0.085` -> `~0.002..0.014`
- `Pe parallel`: weighted-rel `~0.022..0.101` -> `~0.001..0.006`
- `Te parallel`: weighted-rel `~0.030..0.127` -> `~0.008..0.009`

`first_failing_terms.csv` now ranks by `weighted_rel = rel_diff * frac_of_field_rhs`
so tiny terms do not dominate fail-fast triage.

## 2) Build Hermes bundle (same normalization metadata)

```bash
cd <repo>
PYTHONPATH=src python tools/build_benchmark_bundle.py \
  --code hermes \
  --input <hermes-run>/data \
  --output runs/staged_open_field/bundle_hermes_short.npz \
  --geometry tokamak_open_field
```

## 3) Canonical side-by-side panel (shared axes/colormap)

```bash
cd <repo>
PYTHONPATH=src python tools/plot_benchmark_panel.py \
  --hermes runs/staged_open_field/bundle_hermes_short.npz \
  --jax runs/staged_open_field/bundle_jax_short.npz \
  --out docs/figures/tokamak_sol_benchmark_panel.png \
  --summary-csv docs/figures/tokamak_sol_benchmark_panel.csv
```

The panel includes:
- side-by-side fluctuation snapshots (shared colormap range)
- fluctuation RMS overlays (`n, Te, omega, phi`)
- `k_y` spectrum, frequency spectrum
- PDFs, cross-coherence/phase, radial particle flux profile

## 4) Constrained `poisson_scale` scan before longer runs

```bash
cd <repo>
PYTHONPATH=src python tools/scan_poisson_scale.py \
  --config examples/open_field_line/input_tokamak_bxcv_benchmark_alignment.toml \
  --scales "1e-5,3e-5,1e-4,3e-4,1e-3" \
  --target-rms <hermes-rms>.npz \
  --dt 5e-5 \
  --nsteps 300 \
  --max-growth-factor 200 \
  --max-rms-abs 20 \
  --out-csv runs/staged_open_field/poisson_scale_scan.csv
```

Only finite, non-spiking candidates should be used for `t > 1.0` runs.

Latest short-loop scan (calibrated config) selected:
- `poisson_scale = 2e-4`
- score `1.348` (fluctuation RMS mismatch score)
- finite gate: passed (`growth=2.01`, `peak=0.295`)

## 5) Multi-parameter alignment loop (rtol target)

Use the calibration loop for staged, finite-gated scans and an explicit
`rtol` target on fluctuation RMS mismatch:

```bash
cd <repo>
PYTHONPATH=src python tools/calibrate_alignment_loop.py \
  --config examples/open_field_line/input_tokamak_bxcv_benchmark_alignment_hermes_exact_ic.toml \
  --hermes-rms <hermes-rms>.npz \
  --stages 0.1,0.5,1.0 \
  --grid-short 24,32,24 \
  --omega-mults 1.0,1.1 \
  --source-mults 1.0,1.2 \
  --dn-mults 1.0 \
  --domega-mults 0.8 \
  --poisson-scales 2e-4 \
  --phi-dissipation-on 0,1 \
  --phi-sheath-dissipation-on 0,1 \
  --core-vorticity-damping-on 0,1 \
  --promote-top-k 8 \
  --rtol-target 1e-1 \
  --out-csv runs/staged_open_field/alignment_scan_t01.csv
```

Recommended staged workflow:
- run `t_end=0.1` on reduced grid (`24x32x24`) to reject unstable candidates
- keep only finite candidates with lowest score
- rerun selected candidates at `t_end=0.5` on full grid
- extend to `t_end=1.0` only after `t_end=0.5` is finite and non-spiking

Current status for the best finite full-grid candidate (`t_end=0.5`):
- `Te` and `phi` fluctuation RMS are near the `rtol=1e-1` target
- `n` and `omega` fluctuation RMS remain under-predicted and require further
  term-level alignment

## Notes on Physics Alignment

- Open-field + sheath (`bohm_current`) enabled in the benchmark config.
- Curvature is read from the `bxcv` tokamak grid (not a proxy field).
- Parallel transport uses conservative + limiter options (`parallel_flux_conservative=true`,
  `parallel_limiter="mc"`).
- Fluctuation diagnostics are computed against equilibrium (`t0`) in both code paths.
- Initialization supports deterministic Hermes-style density perturbations
  (`n_mixmode_amp`, `n_mixmode_terms`) in addition to stochastic seeds.
- Short-loop calibration that reduced mismatch used:
  - radial BC: `bc_x = neumann` (geometry + perpendicular BC policy)
  - normalization-enabled physical inputs for drives/sources
  - reduced transport (`Dn=1e-3`, `DOmega=1e-4`, `DTe=1e-3`)

## Related docs

- `docs/diagnostics.md`
- `docs/normalization.md`
- `docs/validation.md`
