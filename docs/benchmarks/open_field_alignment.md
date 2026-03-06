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

## Hermes Mirror Rewrite Status

On 2026-03-06 the Stage 1 parity strategy switched from patching the old
Hermes-like path in place to building a temporary `hermes_mirror` translation
layer under `src/jaxdrb/hermes_mirror`.

The first landed mirror functions are boundary primitives only:

- `limit_free`
- `mc_limiter`
- `apply_neumann_boundary_average_z`
- `set_boundary_to_midpoint`

These do not change the strict audit yet, because the mirror engine and mirror
ExB/parallel operators are not wired. Their purpose is to create a tested,
source-cited, differentiable boundary foundation before the remaining operator
parity work resumes.

The first dump-backed mirror fixture is now checked in at
`tests/fixtures/hermes_mirror_ne_local_rank0_t1.npz`, built from
`runs/hermes_open_field_terms_t01_vortterms/data/BOUT.dmp.0.nc`
(`Ne`, local rank 0, `t=0.01`). This fixture backs the `neumann_boundary_average_z`
regression in `tests/hermes_mirror/test_primitives.py`.

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
- `exb_advection_simplified = false` (Hermes full vorticity ExB form rather than
  the simplified advect-`Vort` form)
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

In the 2026-03-05 strict Hermes-state audit refresh, the `gpar`-aware
boundary current divergence now uses the boundary-cell metric on the sheath
face for `Div_par(jpar)` (the `wave=None` path), matching the Hermes/BOUT
boundary-face coefficient more closely. In the same
`start_index=1`, `nsteps=3` audit window this reduced
`omega parallel (jax vs term_Vort_jpar)` at the first audited step
(`t=0.01`) from `rel_diff ~ 0.0123` to `~ 0.0020`, and moved the fail-fast
leader to the much smaller-contribution `omega advection exb` term
(`weighted_rel ~ 0.0070`).

In the follow-on 2026-03-05 strict Hermes-state audit, the next fail-fast
leader (`omega advection exb`) turned out not to be a boundary-ghost issue:
the Hermes dense run was using `exb_advection_simplified = false`, so
`term_Vort_exb` came from the full polarization-current form in
`vorticity.cxx` rather than the simplified advect-`Vort` branch. The unified
JAX path now exposes the same switch and, for the full branch, adds:

\[
-\nabla\cdot\left(\mathbf{v}_E \, 0.5\,\omega\right)
- \nabla_\perp\cdot\left(\frac{0.5\,\bar{A}}{B^2}\,\mathbf{v}_E\cdot\nabla\Pi\right)
- \nabla\cdot\left[\mathbf{v}_E(\phi + \hat{\Pi})\,
\frac{0.5\,\bar{A}}{B^2}\nabla_\perp^2 \phi\right]
\]

with the `\nabla_\perp^2 \phi` auxiliary field evaluated through the metric
operator and a zero-Dirichlet radial boundary when `INVERT_SET`-style Poisson
alignment is active. In the same `start_index=1`, `nsteps=3` audit window this
reduced `omega advection exb` at `t=0.01` from `weighted_rel ~ 0.00703` to
`~ 0.000701`, and moved the fail-fast leader to `Pe parallel/par_total`
(`weighted_rel ~ 0.00622`). Reproducible artifact:
`runs/audit_takeover_full_vort_exb_fix`.

In the next 2026-03-05 strict Hermes-state pass, the remaining `Pe parallel`
gap traced back to a numerics-stack mismatch rather than a sheath-flux
coefficient: Hermes was built with `HERMES_SLOPE_LIMITER=MC` for the
finite-wave `FV::Div_par_mod` channels, while `term_Vort_jpar` still came from
plain `Div_par(jpar)`. The unified JAX path now splits those choices with
`parallel_limiter = "mc"` for the finite-wave density/pressure fluxes and
`parallel_current_limiter = "none"` for the open-field `wave=None`
current-divergence path. In the strict `start_index=1`, `nsteps=3` audit window
this reduced `Pe parallel/par_total` at `t=0.01` from `weighted_rel ~ 0.00622`
to `~ 0.00258` while keeping `omega parallel/jpar` at `~ 0.001995`; the
fail-fast leader moved to `Pe advection/exb` (`weighted_rel ~ 0.00476`), with
`n parallel/par` next at `~ 0.00298`. Reproducible artifact:
`runs/audit_pe_parallel_split_limiter_3step`.

In a follow-up 2026-03-05 strict pass, the shifted parallel transform was
tightened toward Hermes `toFieldAligned(..., "RGN_NOX")` semantics by leaving
non-periodic x-boundary cells unshifted in the unified open-field parallel FV
channel and the poloidal ExB Y-flux branch. This had no visible effect on the
remaining `Pe advection/exb` leader, but it did reduce `n parallel/par` at
`t=0.01` from `weighted_rel ~ 0.00298` to `~ 0.00296`
(`runs/audit_pe_parallel_split_limiter_3step` ->
`runs/audit_shift_nox_fix_3step`). The next structural target remains the
radial-boundary semantics of the poloidal ExB X/Y transport path, since
`Pe advection/exb` is still concentrated at the first radial cells and is
unchanged by the `RGN_NOX` shift fix. Reproducible artifact:
`runs/audit_shift_nox_fix_3step`.

In the next 2026-03-05 strict cycle, the poloidal ExB X-face boundary velocity
was tightened to use Hermes-style ghost/cell metric averaging at the nonperiodic
radial faces, while leaving the Y-face boundary branch unchanged. This reduced
`Pe advection/exb` across the full 3-step strict window:
`t=0.01` `weighted_rel 0.00476 -> 0.00360`,
`t=0.02` `0.00833 -> 0.00714`,
`t=0.03` `0.01381 -> 0.01261`
(`runs/audit_shift_nox_fix_3step` -> `runs/audit_pe_exb_xface_avg_3step`).
The same change moved `n advection/exb` closer to the fail-fast band at
`t=0.01` (`0.00140 -> 0.00304`), but still below the remaining parallel and
pressure leaders. A follow-up attempt to apply Hermes-style boundary-face metric
averaging to the finite-wave parallel sheath flux was rejected because it
regressed `n parallel/par` and `Pe parallel/par_total` badly at the first
strict step. The next structural target remains the open-field density/pressure
sheath-target state construction in the parallel channel. Reproducible artifacts:
`runs/audit_pe_exb_xface_avg_3step`, `runs/audit_xface_and_parbnd_3step`.

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
