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

The next landed mirror slice adds precomputed shifted-transform weights plus
reference/fused implementations of `to_field_aligned_nox` and
`from_field_aligned_nobndry`, validated against the existing JAX shifted metric
path where the semantics overlap. This still does not change the strict Hermes
audit, because the mirror transform is not yet wired into the active RHS terms.

That transform work now also has a source-true FFT path matching Hermes
`ShiftedMetric`, plus a stitched global fixture at
`tests/fixtures/hermes_mirror_shiftedmetric_global_t1.npz` built from the
Hermes dump set in
`runs/hermes_open_field_terms_t01_vortterms/data`.

Phase 3 has now started with the first mirrored ExB transport slice in
`src/jaxdrb/hermes_mirror/exb.py`: `div_n_bxgrad_f_b_xppm_xz` and its
reference twin. This is only the X-Z branch of the Hermes
`Div_n_bxGrad_f_B_XPPM` operator. It is validated against the current unified
`hermes_xppm` X-Z path with `exb_poloidal_flows = false`, plus fused/reference
and autodiff tests in `tests/hermes_mirror/test_exb.py`.

This still does not change the strict Hermes audit yet, because the Y-flux
branch and the mirror runtime wiring are not landed.

A follow-up source check on 2026-03-06 added the missing `RGN_ALL` shifted
transform variants to the mirror layer and switched the active poloidal Y-flux
branch to use full-region shifted transforms, matching the Hermes source
signature more closely. The 1-step strict audit at
`runs/audit_phase3_yregion_probe` showed no measurable change:

- `Pe advection/exb`: `0.0036033052`
- `n advection/exb`: `0.0030432257`
- `n parallel/par`: `0.0029585638`

That rules out transform-region selection as the dominant remaining
`Pe advection/exb` cause. The next target is the guard-aware
`DDX(phi) -> communicate -> applyBoundary("neumann")` preparation chain before
the Y-flux transform.

The first part of that next target is now landed in the mirror layer only:

- `src/jaxdrb/hermes_mirror/derivs.py::ddx_centered_guarded`
- dump-backed fixture `tests/fixtures/hermes_mirror_phi_metric_local_rank0_t1.npz`

This step is intentionally diagnostic. A direct production-path attempt to
replace the current guardless Y-flux `DDX(phi)` boundary handling with a simple
ghost-centred derivative was rejected after the 1-step strict audit at
`runs/audit_phase3_ddxghost_probe` regressed badly:

- `n advection/exb`: `0.0030432257 -> 0.1658992790`
- `Pe advection/exb`: `0.0036033052 -> 0.1089011668`

That rejection means the remaining mismatch is in the full preparation chain,
not just the final boundary derivative formula. The next implementation target
is a literal local mirror of:

1. `DDX(phi)`
2. `mesh->communicate(dfdx)`
3. `dfdx.applyBoundary("neumann")`
4. `toFieldAligned(dfdx)`

That local mirror prep step is now partly landed. The new helper
`prepare_poloidal_y_dfdx_local_ref` in
`src/jaxdrb/hermes_mirror/species.py` operates on a dump-backed local
field-aligned fixture
`tests/fixtures/hermes_mirror_phi_field_aligned_local_rank0_t1.npz`
in `(npar, nx, nbinorm) = (y, x, z)` layout.

On that fixture, the literal local prep path differs from a guardless
approximation by a whole-field relative RMS of about `0.5269`. This is not yet
directly a strict-audit number, because the helper is not wired into the active
runtime path, but it is the clearest structural evidence so far that the
remaining `Pe advection/exb` mismatch lives in the local preparation chain
upstream of the final Y-flux formula.

The next mirror-only slice now adds the local field-aligned Y-flux operator in
`src/jaxdrb/hermes_mirror/exb.py`:
`div_n_bxgrad_f_b_xppm_xy_y_local_ref`,
`div_n_bxgrad_f_b_xppm_xy_y_local`,
and their `*_from_fields` wrappers. The dump-backed combined fixture is
`tests/fixtures/hermes_mirror_exb_local_rank0_t1.npz`, built from the same
Hermes rank-0 dump at `t=0.01` with local `phi`, `Ne`, `Pe`, and metric
coefficients. The current deterministic local RMS values are:

- `Ne`: `0.0052662453`
- `Pe`: `0.0050677802`

Fused and reference implementations now match exactly on this fixture, but the
strict Hermes audit is still unchanged because the runtime ExB term has not yet
been switched over to the assembled mirror operator.

The next two mirror-only Phase 3 slices are now landed on top of that same
fixture. First, `src/jaxdrb/hermes_mirror/species.py` now has the local
`DDY(phi) -> applyBoundary("neumann")` preparation path for the Hermes
poloidal X-flux branch, and `src/jaxdrb/hermes_mirror/exb.py` now has the
matching local X-flux operator:
`div_n_bxgrad_f_b_xppm_xy_x_local_ref` and friends. The current deterministic
local RMS values are:

- `Ne`: `0.0053911873`
- `Pe`: `0.0052891376`

Second, the same module now has the first assembled local full mirror ExB
operator: `div_n_bxgrad_f_b_xppm_local_ref` and `div_n_bxgrad_f_b_xppm_local`.
Those now combine the X-Z slice, the local X-flux branch, and the
field-aligned Y-flux branch plus `fromFieldAligned(...)` in one function. The
current deterministic local RMS values are:

- `Ne`: `0.0072043576`
- `Pe`: `0.0070714532`

That assembled local operator now also has a direct Hermes-term regression on a
new fixture,
`tests/fixtures/hermes_mirror_exb_term_local_rank0_t1.npz`, which includes the
raw `term_Ne_exb` and `term_Pe_exb` arrays from the same dump. On the physical
interior cells, the mirror operator is now essentially at parity:

- `Ne` interior diff RMS: `2.8867991448834276e-05`
- `Pe` interior diff RMS: `1.2432835191026055e-05`
- `Ne` interior correlation: `0.9998132247422601`
- `Pe` interior correlation: `0.9999591421467119`

That lower-open-boundary guard mismatch is now closed in the dump-backed mirror
operator path. The structural fix was to complete the Hermes
`DDY(f) -> communicate -> applyBoundary("neumann")` chain for the X-flux
preparation field in `src/jaxdrb/hermes_mirror/species.py`, including the
lower-open parallel Neumann copy that the earlier mirror helper was missing.

With that fix landed, the assembled local mirror ExB operator now matches the
Hermes dump term across all cells, not only the interior:

- `Ne` all-cell diff RMS: `3.072901445531812e-05`
- `Pe` all-cell diff RMS: `1.3376334360587529e-05`
- `Ne` all-cell correlation: `0.9999820919602114`
- `Pe` all-cell correlation: `0.9999963535995172`
- `Ne` lower-left corner diff RMS: `6.171447934311131e-08`
- `Pe` lower-left corner diff RMS: `7.577202115196461e-09`

So the remaining Milestone A ExB work is no longer operator parity inside the
mirror slice. It is runtime promotion: route the same mirrored preparation and
operator ordering through the strict Hermes audit path and then re-run the
1-step and 3-step gates.

That first runtime promotion slice is now landed as an opt-in scheme:
`exb_flux_scheme = "hermes_mirror"`. The runtime wrapper reconstructs a local
guard-inclusive Hermes/BOUT view from global `(nz, nx, ny)` JAX arrays and then
calls the validated local mirror ExB operator. The dump-backed wrapper
regression lives in `tests/hermes_mirror/test_exb_runtime.py` and uses only the
physical interior cells as input.

On the local-rank fixture interior, the runtime wrapper gives:

- `Ne` diff RMS: `2.488462499110523e-04`
- `Pe` diff RMS: `2.6183313968993464e-04`

That is close enough to keep the wrapper as the runtime promotion vehicle, but
it is not yet strict-audit quality.

The next landed runtime slice is a stitched global Hermes fixture plus a
hybrid open-boundary wrapper for `hermes_mirror`. The new fixture builder is
`tools/build_hermes_mirror_runtime_fixture.py`, the checked-in runtime fixture
is `tests/fixtures/hermes_mirror_exb_global_t1.npz`, and the new regression is
`tests/hermes_mirror/test_exb_runtime_global.py`.

That regression shows that the residual is concentrated in the first and last
open parallel subdomains. Re-evaluating only those edge blocks with the local
guard-inclusive mirror operator, via
`hermes_mirror_parallel_edge_block = 8`, improves the actual global term arrays
substantially:

- `Ne` runtime-wrapper RMS: `9.281612304656274e-04 -> 2.7785371223075885e-04`
- `Pe` runtime-wrapper RMS: `9.436398753984853e-04 -> 2.9023628701603716e-04`
- `Ne` correlation: `0.9507164518528228 -> 0.99676569423027`
- `Pe` correlation: `0.9452534023907078 -> 0.9964150807456237`

The first live 3-step Hermes-state audit for the opt-in runtime scheme is
recorded in `runs/audit_hermes_mirror_runtime_3step_v2`. After correcting the
shifted-transform FFT length to use `metric_dz * nbinorm`, the runtime mirror
path still regresses the early ExB channels relative to the current strict
baseline:

- `omega advection/exb`: `0.06804918916596805`
- `n advection/exb`: `0.04636472581495929`
- `Pe advection/exb`: `0.038900114007649214`

while the parallel channels stay identical to the current best strict run:

- `n parallel/par`: `0.0029585637833904267`
- `Pe parallel/par_total`: `0.0025796150980648175`
- `omega parallel/jpar`: `0.001995419920917737`

With the new edge-block wrapper, the smallest strict gate is
`runs/audit_hermes_mirror_edge_block_1step`. The current scalar fail-fast
metric only moves slightly:

- `omega advection/exb`: `0.06804918916596805 -> 0.06712108791244092`
- `Pe advection/exb`: `0.038900114007649214 -> 0.03873682407548267`

while the `n` scalar row becomes worse in `term_mismatch.csv` even though the
direct built-system term arrays improve strongly. That is because the current
`term_mismatch.csv` ranking compares only term RMS magnitudes, not array
differences. The built-system direct mirror calls on the same Hermes snapshot
now give:

- `n` term vs Hermes `term_Ne_exb` RMS: `2.7785371223075885e-04`
- `Pe` term vs Hermes `term_Pe_exb` RMS: `2.9023628701559654e-04`

So the next Milestone A target is narrower now: match the global Hermes
guard/communication contract at the open-end parallel blocks, and tighten the
strict audit reporting so operator-array improvements are visible alongside the
existing RMS-magnitude gate, before switching the strict configs to
`hermes_mirror`.

The first Phase 4 species state-preparation helpers are now also landed in
`src/jaxdrb/hermes_mirror/species.py`:
`density_transform_impl` and `pressure_transform_impl`. These reconstruct the
Hermes `neumann_boundary_average_z` x-guard states and the pressure/temperature
consistency step on the same local dump-backed fixture. The current
deterministic values are:

- density RMS: `1.7785461475`
- density interior RMS: `1.8245458655`
- temperature RMS: `0.5928697471`
- temperature interior RMS: `0.6081834468`

This is still not a strict-runtime number, but it closes the next structural
gap between the mirror ExB operator and the species states Hermes feeds into
that operator. The next remaining step is to mirror the `finally` ordering and
route the prepared states into the strict runtime path.

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
- `exb_flux_scheme = "hermes_mirror"` (runtime wrapper around the literal
  Hermes/BOUT ExB transport stack)
- `exb_advection_simplified = false` (Hermes full vorticity ExB form rather than
  the simplified advect-`Vort` form)
- `exb_poloidal_flows = true`
- `exb_poloidal_scale = 1.0`
- `exb_poloidal_y_scale = 1.0` (the earlier `1.24` tuning knob is no longer
  needed once the promoted path uses the literal mirror Y-flux operator)
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

As of the mirror-runtime promotion cycle, `tools/audit_term_alignment.py` now
also writes true term-array mismatch columns:
`array_diff_rms`, `array_rel_diff`, `array_corr`, and `weighted_array_rel`.
The fail-fast ranking now defaults to the array-based metric via
`--term-ranking-metric=array`, so triage follows actual Hermes operator-array
parity instead of only comparing term RMS magnitudes. The old scalar
RMS-magnitude columns (`rel_diff`, `weighted_rel`) are still written for
continuity.

With that updated audit, the strict early config
`examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml` is
now promoted to:

- `exb_flux_scheme = "hermes_mirror"`
- `hermes_mirror_parallel_edge_block = 8`

The first promoted 1-step audit is
`runs/audit_strict_early_mirror_promoted_1step`. Relative to
`runs/audit_current_arraymetric_1step`, the dominant array-weighted ExB
channels improve materially:

- `n advection/exb`: `weighted_array_rel 0.6415487257460786 -> 0.30603226941513645`
- `Pe advection/exb`: `weighted_array_rel 0.43066567430657776 -> 0.20417452847516265`
- `n advection/exb` correlation: `0.7888450540689848 -> 0.9947894182550701`
- `Pe advection/exb` correlation: `0.7699527249045816 -> 0.9952771323120512`

The already-closed parallel channels stay unchanged:

- `omega parallel/jpar`: `weighted_array_rel 0.2107103945115671`
- `n parallel/par`: `weighted_array_rel 0.16847301041461074`
- `Pe parallel/par_total`: `weighted_array_rel 0.15454019751690204`

The promoted runtime mirror path still leaves a smaller but real vorticity
advection regression:

- `omega advection/exb`: `weighted_array_rel 0.007979974955211428 -> 0.09741634145346564`

so Milestone A is not closed yet. The next structural target after this
promotion is therefore the mirrored vorticity ExB composition, while keeping
the improved density/pressure ExB transport as the new strict baseline.

In the 2026-03-08 strict refresh, the remaining density/pressure ExB overshoot
turned out to be one stale alignment knob, not another operator bug. The
promoted mirror path was still inheriting `exb_poloidal_y_scale = 1.24` from
the older pre-mirror geometry path. On the dump-backed global mirror fixture,
that exact multiplier reproduces the live promoted overshoot:

- `Ne` scale `1.051399020263945 -> 1.287035478033308`
- `Pe` scale `1.0611891505987026 -> 1.3028369266421438`

Resetting the strict mirror config to `exb_poloidal_y_scale = 1.0` moves the
1-step Hermes-state audit (`runs/audit_poloidal_y_scale_1p0_1step`) to:

- `n advection/exb`: `weighted_array_rel 0.30603226941513645 -> 0.09608755774957915`
- `Pe advection/exb`: `weighted_array_rel 0.20417452847516265 -> 0.06745309373399326`
- `omega advection/exb`: remains small at `0.004178515908061414`
- `omega parallel/jpar`: unchanged at `0.2107106038839909`

That makes `omega parallel/jpar` the next real fail-fast leader on the
promoted strict baseline.

In the next 2026-03-08 strict cycle, the `jpar` path itself was moved closer to
the Hermes source. Hermes vorticity uses `Div_par(jpar)` on the current field,
not the FV transport operator used by density and pressure. The promoted JAX
path was still routing `wave=None` through the finite-volume reconstruction,
which left a small but persistent structural gap in `term_Vort_jpar`. The
current path now uses a centered `Div_par`-style divergence for the `jpar`
channel, while still applying the explicit sheath-face current values in
`boundary_flux` mode.

On the strict 1-step Hermes-state audit
`runs/audit_jpar_centered_1step`, this reduces:

- `omega parallel/jpar`: `weighted_array_rel 0.2107106038839909 -> 0.11715792736854537`
- `omega parallel/jpar` correlation: `0.9997735493310655 -> 0.9999301165207451`
- `omega parallel/jpar` array diff RMS:
  `0.0005454509757019604 -> 0.0003032780724674658`

On the 3-step strict window `runs/audit_jpar_centered_3step`, the same channel
stays lower than the previous promoted baseline at all three early steps:

- `t=0.01`: `0.11715792736854537`
- `t=0.02`: `0.2543975649539223`
- `t=0.03`: `0.4021854262820713`

The next fail-fast terms on the promoted mirror baseline are now the density
and pressure parallel transport channels (`n parallel/par`, `Te parallel/par_total`,
`Pe parallel/par_total`), while the audit-level sheath residual remains a
separate follow-up for boundary-energy bookkeeping.

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

The latest literal-refactor slice adds the missing mirror boundary and
finite-volume building blocks for the vorticity path:

- `src/jaxdrb/hermes_mirror/boundary.py::apply_free_o2_field3d`
- `src/jaxdrb/hermes_mirror/fv.py::div_a_grad_perp_local`
- `src/jaxdrb/hermes_mirror/fv.py::div_a_grad_perp`
- `src/jaxdrb/hermes_mirror/vorticity.py::full_omega_exb_advection`

This lands with synthetic/autodiff regressions plus a stitched dump-backed
fixture at `tests/fixtures/hermes_mirror_vorticity_global_t1.npz`, built by
`tools/build_hermes_mirror_vorticity_fixture.py`.

The vorticity blocker is now structurally reduced. The mirror path adds a
dedicated literal perpendicular Laplacian in
`src/jaxdrb/hermes_mirror/delp2.py`, extends
`src/jaxdrb/core/geometry_field_aligned.py` to ingest Hermes `G1`, `G3`, and
optional `d1_dx`, updates
`tools/convert_hermes_dump_axisymmetric.py` to emit `G1`/`G3`, and checks those
coefficients into the strict mesh bundle
`examples/open_field_line/axisym_tokamak_bxcv_hermes_norm_parcurv_g22.npz`.

The stitched vorticity fixture now also carries the raw Hermes `G1`, `G3`,
`g11`, `g13`, `g33`, `dx`, `dz`, `Bxy`, and `zShift` planes via
`tools/build_hermes_mirror_vorticity_fixture.py`, so the literal `Delp2(phi)`
can be validated directly against the saved BOUT guard-cell state.

That validation shows the operator is now source-true:

- local rank-0 `Delp2(phi)` vs raw BOUT single-index formula:
  correlation `0.9999999979364631`, diff RMS `6.903925415803028e-07`
- stitched global `Delp2(phi)` vs rank-stitched raw BOUT evaluation:
  correlation `0.9999988050053542`, diff RMS `3.9164034002630735e-05`
- stitched global interior diff RMS: `1.1612800441803346e-07`

The remaining error after the first promotion attempt turned out not to be
`Delp2(phi)` itself but the boundary contract used to transport
`DelpPhi_2B2`. The wrong behavior was passing `poisson_invert_set` through the
runtime ExB transport wrapper for the `phi`/`phi + Pi_hat` transport field. In
Hermes that auxiliary Dirichlet override belongs only in the `Delp2(phi)` path,
not in the subsequent `Div_n_bxGrad_f_B_XPPM(...)` transport of the resulting
field. After removing that transport-side override and routing the omega path
through the same global mirror wrapper used by the density/pressure channels,
the dump-backed full omega mirror term moved to:

- full `term_Vort_exb` mirror correlation: `0.9286922397070627`
- full diff RMS: `9.242617198253543e-06`

The promoted strict 1-step Hermes-state audit is
`runs/audit_mirror_omega_transport_bc_fix_1step`. Relative to the previous
promoted strict baseline
`runs/audit_strict_early_mirror_promoted_1step`, the blocker term collapses:

- `omega advection/exb`: `weighted_array_rel 0.09741634145346564 -> 0.0035704721275969927`
- `omega advection/exb`: correlation `-0.6627029835778587 -> 0.9286922397070773`
- `omega advection/exb`: array diff RMS
  `0.00025217448726198274 -> 9.24261745643642e-06`

With that term reduced below the remaining density/pressure and parallel
leaders, the new strict fail-fast order is:

- `n advection/exb`: `0.30603226941513645`
- `omega parallel/jpar`: `0.2107103945115671`
- `Pe advection/exb`: `0.20417452847516265`
- `n parallel/par`: `0.16847301041461074`
- `Pe parallel/par_total`: `0.15454019751690204`

In the next 2026-03-09 strict parallel cycle, the open-field finite-volume
density and pressure channels were tightened to use the sheath ghost states in
the boundary-adjacent limited reconstruction, not only in the explicit sheath
face flux. This mirrors the Hermes `FV::Div_par_mod` stencil more closely at
the first and last physical parallel cells while leaving the `wave=None`
centered `Div_par(jpar)` path unchanged.

The confirm audit
`runs/audit_parallel_ghost_stencil_confirm_1step` shows a small but consistent
improvement in the remaining parallel transport leaders at `t=0.01`:

- `n parallel/par`: `0.15689932456328756 -> 0.15650650752322878`
- `Te parallel/par_total`: `0.15587502102513381 -> 0.1556861680908554`
- `Pe parallel/par_total`: `0.15453748447303708 -> 0.154109603265596`
- `omega parallel/jpar`: unchanged at `0.11715792736854537`

The same cycle also tested and rejected a simpler sheath-energy hypothesis:
replacing the current electron-sheath energy closure with a constant
Hermes-like `gamma_e = 3.5` contract made the audit-level boundary-energy rows
blow up instead of converge
(`runs/audit_parallel_and_sheath_fix_1step`):

- `Te sheath/source_residual_boundary`: `0.26493593205386157 -> 8.4765`
- `Pe sheath/source_residual_boundary`: `0.11172993716659292 -> 3.6046`

So the remaining `Te/Pe sheath source_residual_boundary` gap is not a missing
single gamma coefficient. The next structural target is the boundary-energy
bookkeeping contract itself, using explicit dumped term arrays rather than a
formula swap.

In the next 2026-03-09 strict parallel cycle, the finite-wave sheath-face
metric factor was brought back into line with Hermes `FV::Div_par_mod` for the
density and pressure channels. Hermes uses the boundary-cell metric on the
sheath face even for the finite-wave transport path; the JAX path had still
been using the first interior-face factor there.

The promoted 1-step audit
`runs/audit_parallel_boundary_metric_retry_1step` reduces the remaining
parallel leaders at `t=0.01` to:

- `n parallel/par`: `0.15650650752322878 -> 0.13432807982024225`
- `Te parallel/par_total`: `0.1556861680908554 -> 0.14849268403368665`
- `Pe parallel/par_total`: `0.154109603265596 -> 0.11330115527226602`
- `omega parallel/jpar`: unchanged at `0.11715751270556365`

The 3-step confirm window
`runs/audit_parallel_boundary_metric_retry_3step` keeps that improvement in
place while preserving very high correlations (`> 0.9990`) for the parallel
channels through `t=0.03`:

- `t=0.02`: `n/Te/Pe/omega parallel = 0.26710 / 0.26923 / 0.24906 / 0.25440`
- `t=0.03`: `n/Te/Pe/omega parallel = 0.42417 / 0.41920 / 0.40253 / 0.40219`

The audit-level `Te/Pe sheath source_residual_boundary` rows are unchanged by
this fix, which supports the earlier diagnosis that those rows are a residual
bookkeeping issue rather than the same transport bug.

The follow-up 2026-03-09 audit change addresses that bookkeeping ambiguity
directly. `tools/audit_term_alignment.py` now reconstructs the Hermes electron
sheath pressure source as a synthetic `term_Pe_sheath` from the raw BOUT dumps,
mirroring `sheath_boundary.cxx`, and the `Pe/Te sheath` mismatch rows now
prefer that direct term before falling back to the mixed
`source_residual_boundary` bucket.

On the strict 1-step audit
`runs/audit_direct_sheath_mapping_1step`, the direct sheath comparison is now
substantially cleaner:

- `Pe sheath/sheath`: `weighted_array_rel 0.022641938293208385`, correlation `1.0`
- `Te sheath/sheath`: `weighted_array_rel 0.08160536527344078`, correlation `1.0`
- `n sheath/source_residual_boundary`: unchanged bookkeeping row at
  `weighted_array_rel 0.006210587208797745`

This confirms the earlier diagnosis: the old `Te/Pe sheath/source_residual_boundary`
rows were not measuring the direct sheath term cleanly. They were comparing a
pure JAX sheath channel against a mixed Hermes residual-source bucket.

The next parallel refactor promoted a literal Hermes-mirror `FV::Div_par_mod`
and centered `Div_par(jpar)` path into the strict config, including the local
parallel `dy` metric in the face factors. The important result from that cycle
is operator closure, not yet live strict closure:

- the local dump-backed mirror parallel operator now matches the raw Hermes
  `term_Ne_par`, `term_Pe_par`, and `term_Vort_jpar` arrays in
  `/Users/rogerio/local/jax_drb/tests/hermes_mirror/test_parallel_dump.py`
- the promoted strict audits
  `/Users/rogerio/local/jax_drb/runs/audit_parallel_mirror_with_dy_1step` and
  `/Users/rogerio/local/jax_drb/runs/audit_parallel_mirror_with_dy_3step`
  are numerically unchanged relative to the previous promoted baseline

That isolates the remaining live `n/Pe/jpar` parallel gap to the runtime
boundary-conditioned state that feeds the operator, especially the open-end
sheath guard reconstruction and transform ordering, not the FV stencil itself.

The next mirror slice closes the sheath-state side of that contract directly.
A new literal guard builder in `src/jaxdrb/hermes_mirror/sheath.py`
transliterates the open-end updates in
`/Users/rogerio/local/hermes-3/src/sheath_boundary.cxx` for:

- `Ne`, `Te`, `Pe` via `limitFree(...)`
- `phi` guard extrapolation
- midpoint `vesheath` / `visheath`
- guard `Ve`, `NVe`, `Vi`, `NVi`, and `jpar`

Two new dump-backed fixtures pin both end ranks:

- `tests/fixtures/hermes_mirror_parallel_local_rank0_t1.npz`
- `tests/fixtures/hermes_mirror_parallel_local_rank5_t1.npz`

and `tests/hermes_mirror/test_sheath.py` now checks both the guard values and
the resulting mirrored local operator terms without reading dumped guards
directly into the operator. The open-end guard RMS is:

- lower end rank:
  - `Ne 7.63e-4`
  - `Te 2.60e-4`
  - `Pe 8.33e-7`
  - `Ve 6.06e-3`
  - `NVe 2.77e-6`
  - `NVd+ 1.26e-3`
- upper end rank:
  - `Ne 2.59e-3`
  - `Te 8.52e-4`
  - `Pe 2.10e-6`
  - `Ve 1.96e-2`
  - `NVe 7.80e-6`
  - `NVd+ 3.76e-3`

and the reconstructed-guard operator parity remains within:

- `term_Ne_par` RMS `< 4e-4`
- `term_Pe_par` RMS `< 3e-4`
- `term_Vort_jpar` RMS `< 2e-5`

The first promoted live audit with that builder was still unchanged. That
identified the missing piece: in the shifted-transform runtime path, the code
was moving the cell-centered fields and explicit boundary fluxes into
field-aligned coordinates, but not the sheath ghost planes themselves. The
next runtime fix applies the same boundary-plane shifted transform to
`ghost_low/high_f` and `ghost_low/high_v` before the mirror operator runs.

That produces a small but real strict-gate improvement in
`/Users/rogerio/local/jax_drb/runs/audit_sheath_shifted_ghosts_1step`:

- `n parallel/par`: `0.13448644700674087 -> 0.1338459414001929`
- `omega parallel/jpar`: `0.1169915003671119 -> 0.11697747997572151`
- `Pe parallel/par_total`: `0.11335202275260099 -> 0.11330118219042988`

The gain is modest, but it closes the last known missing shifted-ghost wiring
in the promoted mirror parallel stack. The 3-step confirm window
`/Users/rogerio/local/jax_drb/runs/audit_sheath_shifted_ghosts_3step` keeps
the same direction of change through `t=0.03`, for example:

- `n parallel/par`: `0.2672274769250367 -> 0.26673799080966265` at `t=0.02`
- `n parallel/par`: `0.4242715402774202 -> 0.4238822347442656` at `t=0.03`
- `Pe parallel/par_total`: `0.24908302291800197 -> 0.24905629487447392` at `t=0.02`

The next full-refactor slice promotes the remaining density/pressure
state-preparation stubs into the active mirror runtime. The new global no-guard
helpers in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/species.py`

- `density_final_global`
- `pressure_final_global`
- `prepare_reduced_species_state_global`

now mirror the reduced Hermes density/pressure ordering, and
`/Users/rogerio/local/jax_drb/src/jaxdrb/core/terms/context.py` constructs the
prepared mirror species state once before the strict ExB and parallel channels
run. The active transport code in

- `/Users/rogerio/local/jax_drb/src/jaxdrb/core/terms/advection.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/core/terms/parallel.py`

now consumes the prepared fields directly instead of rebuilding transformed
states locally:

- `ctx.n_prepared`
- `ctx.pe_prepared`
- `ctx.Te_prepared`
- `ctx.pi_prepared`
- `ctx.Ti_prepared`

The architectural layer is covered by
`/Users/rogerio/local/jax_drb/tests/hermes_mirror/test_species.py`, and the
promoted audits

- `/Users/rogerio/local/jax_drb/runs/audit_full_species_prep_1step`
- `/Users/rogerio/local/jax_drb/runs/audit_full_species_prep_3step`

are unchanged relative to the shifted-ghost baseline up to roundoff. That is
still a useful Milestone A result: the remaining live parity gap is now even
more clearly below the species-state layer, in the lower-level
operator/communication contract.

The next strict-runtime slice promotes the density/pressure `finally()`
assembly itself into the mirror term path. The new module
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/rhs.py`
contains:

- `density_rhs_terms`
- `pressure_rhs_terms`
- `build_reduced_mirror_term_cache`

and `/Users/rogerio/local/jax_drb/src/jaxdrb/core/terms/registry.py` now uses
that cache for the live `advection` and `parallel` term groups whenever the
mirror flux schemes are active. The regression
`/Users/rogerio/local/jax_drb/tests/hermes_mirror/test_rhs.py` checks the
pressure-space identities directly. On the promoted audit
`/Users/rogerio/local/jax_drb/runs/audit_mirror_rhs_cache_1step`, the strict
numbers are unchanged relative to
`/Users/rogerio/local/jax_drb/runs/audit_full_species_prep_1step_rerun` down
to roundoff. That closes the runtime density/pressure `finally()` assembly
question without changing the live parity leaders.

The next lower contract fix lands in
`/Users/rogerio/local/jax_drb/src/jaxdrb/core/terms/parallel.py`:
`_shift_boundary_flux_to_field_aligned` now uses the same Hermes-mirror
shifted-metric transform implementation as interior fields, including the
spectral path requested by the strict config. The regression
`/Users/rogerio/local/jax_drb/tests/test_parallel_shifted_boundary_flux.py`
now covers both linear and spectral boundary-plane shifts.

On the promoted audit
`/Users/rogerio/local/jax_drb/runs/audit_parallel_boundary_spectral_shift_1step`,
the effect is small but real at the intended layer:

- `n parallel/par`: `0.1338459414001856 -> 0.13383127252151306`
- `omega parallel/jpar`: `0.11697747997572525 -> 0.11697795624618619`
- `Pe parallel/par_total`: `0.11330118219042934 -> 0.1133024567583403`

So this source-true boundary-plane transform fix is numerically neutral to
slightly positive, and the remaining Milestone A blocker is still the full
runtime sheath / guard / transform contract feeding the mirror parallel
operator.

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

### 2026-03-09 literal runtime promotion

The active Hermes-specific runtime imports now point at the fresh
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal` modules for:

- reduced species preparation
- shifted-metric transforms
- ExB transport and vorticity ExB
- parallel FV transport
- mirror RHS cache assembly

The strict 1-step audit was rerun at:

- `/Users/rogerio/local/jax_drb/runs/audit_literal_runtime_promotion_1step`

This promotion preserved the previous parity baseline. The current top
array-ranked mismatches remain:

- `n parallel/par`: `0.13383127252151306`
- `omega parallel/jpar`: `0.11697795624618619`
- `Pe parallel/par_total`: `0.1133024567583403`
- `n advection/exb`: `0.09623829491706752`
- `Pe advection/exb`: `0.0676385260919583`

### 2026-03-09 literal engine execution

The strict early config
`/Users/rogerio/local/jax_drb/examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml`
now explicitly sets:

- `engine = "hermes_literal"`

and the smallest strict audit now executes through the literal engine itself:

- `/Users/rogerio/local/jax_drb/runs/audit_literal_engine_smoke`

This is the first point where the parity tool is no longer evaluating a hybrid
runtime. The top 1-step array-ranked mismatches from that literal-engine audit
are:

- `Te parallel`: `0.1474904091090806`
- `n parallel`: `0.13383127252151306`
- `omega parallel`: `0.11697795624618619`
- `Pe parallel`: `0.1133024567583403`
- `n advection`: `0.09623829491706752`
- `Pe advection`: `0.0676385260919583`

The ranking did not yet improve, but the architectural milestone matters more
than the numbers in this slice: strict Milestone A parity work is now running
through `engine="hermes_literal"` rather than the old unified/hybrid path.
Subsequent parity fixes should therefore land inside the literal engine and its
supporting runtime contract, not as more hybrid term patches.

### 2026-03-09 literal parallel runtime rehome

The reduced density/pressure cache now gets its live parallel transport state
from:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/parallel.py`

instead of routing that contract through the unified helper layer in
`core/terms/parallel.py`. This rehome includes:

- sheath-state reconstruction
- shifted boundary-plane transforms
- literal `Div_par(jpar)` / `FV::Div_par_mod` dispatch
- fastest-wave and pressure transport coefficients

The 1-step strict audit at
`/Users/rogerio/local/jax_drb/runs/audit_literal_parallel_runtime_rehome`
preserves the literal-engine baseline and nudges the leading parallel rows
slightly downward:

- `Te parallel`: `0.1474904091090806 -> 0.14748382093236653`
- `n parallel`: `0.13383127252151306 -> 0.1338298917677307`
- `Pe parallel`: `0.1133024567583403 -> 0.11330241103262646`

This slice is architectural, not a parity jump, but it matters because the
remaining dominant parallel mismatch is now owned by the literal runtime layer
itself rather than a copied call back into the unified core.

### 2026-03-09 literal advection runtime rehome

The reduced density/pressure cache now gets its live ExB advection group from:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/advection.py`

instead of importing that term group from `core/terms/advection.py`. The
literal module preserves the runtime switches that the strict engine still
needs during validation:

- `exb_advection_simplified`
- `exb_advect_conservative`

and is covered by:

- `/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_advection_runtime.py`
- `/Users/rogerio/local/jax_drb/tests/test_vorticity_alignment_switches.py`

The 1-step strict audit at
`/Users/rogerio/local/jax_drb/runs/audit_literal_advection_parallel_rehome_1step_after_fix`
preserves the literal-engine baseline exactly for the leading rows:

- `n advection/exb`: `0.09623829491706752`
- `Pe advection/exb`: `0.0676385260919583`
- `n parallel/par`: `0.1338298917677307`
- `Te parallel/par_total`: `0.14748382093236653`

This closes another hybrid dependency: the strict literal cache now gets its
dominant advection and parallel rows from literal runtime modules rather than
calling back into the unified core for those term groups.

### 2026-03-09 literal context rehome

The literal engine now builds its runtime context through:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/context.py`

instead of importing `core.terms.build_context` directly. This keeps the
strict engine’s prepared density/pressure state, `phi` solve staging, and SOL
mask assembly inside the literal package while still reusing the shared
low-level helper implementations for boundary and field transforms.

The new regression:

- `/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_context.py`

checks that the literal context reproduces the previous strict runtime
contract for `n_phys`, `Te_phys`, `phi`, and the prepared density/pressure
fields. This slice is again architectural rather than a parity jump, but it
removes another direct dependency from the literal engine onto `core/terms`.

### 2026-03-09 literal registry rehome

The literal engine now imports its schedule and dispatch table through:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/registry.py`

instead of importing `core.terms.registry` directly from `engine.py`. The new
regression:

- `/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_registry.py`

locks the current Stage 1 contract by checking that the literal registry still
matches the active schedule and term names used by the strict engine. This is
again architectural, but it narrows the remaining hybrid surface area inside
the literal runtime itself.

### 2026-03-09 literal ExB subdomain runtime

The strict literal config now sets:

- `hermes_mirror_parallel_subdomain_size = 8`

and the literal ExB runtime wrapper in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/exb.py`
evaluates the local Hermes operator over consecutive `MYSUB`-sized parallel
chunks instead of a single global transform. This is the first literal runtime
change in this refactor that materially moves the remaining live ExB rows.

The new stitched-global regression in
`/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_exb_runtime.py`
shows that the blockwise local runtime improves the raw Hermes dump-backed
global ExB term relative error from about `0.097/0.107` to about
`0.061/0.066` for `Ne/Pe`.

The strict 1-step audit at
`/Users/rogerio/local/jax_drb/runs/audit_literal_subdomain_parallel_1step`
moves the live transport rows in the same direction:

- `n advection/exb`: `0.09623829491706752 -> 0.06021497597645309`
- `Te advection/exb`: `0.03993444992422992 -> 0.03175328243530484`
- `Pe advection/exb`: `0.0676385260919583 -> 0.0417892594173691`

The leading parallel rows do not move in this slice:

- `n parallel/par`: `0.1338298917677307`
- `omega parallel/jpar`: `0.11697795624618619`
- `Te parallel/par_total`: `0.14748382093236653`
- `Pe parallel/par_total`: `0.11330241103262646`

This is still not strict parity closure. It does, however, confirm that the
remaining ExB gap is in the processor-local subdomain contract rather than in
the already-copied local operator body alone.

### 2026-03-09 literal communication layer rehome

The next literal slice makes the local parallel-slab assembly explicit in:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/communicate.py`

and rehomes the literal BC/operator-selection helpers to:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/bcs.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/ops.py`

The promoted ExB runtime in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/exb.py`
now uses that communication helper to assemble processor-local slabs before it
calls the validated local operator. For the active strict baseline, the helper
preserves the previously validated internal seam policy, so the stitched-global
Hermes regression remains:

- `Ne` raw relative error: `0.3242119312307518 -> 0.06090172693816785`
- `Pe` raw relative error: `0.3455716236178938 -> 0.06601079736963186`

The strict 1-step audit at
`/Users/rogerio/local/jax_drb/runs/audit_literal_comm_layer_1step`
is unchanged relative to
`/Users/rogerio/local/jax_drb/runs/audit_literal_subdomain_parallel_1step` on
the live leaders:

- `n advection/exb`: `0.06021497597645309`
- `Pe advection/exb`: `0.0417892594173691`
- `n parallel/par`: `0.1338298917677307`
- `Te parallel/par_total`: `0.14748382093236653`

So this slice is architectural rather than a new parity jump: it keeps the
current promoted literal baseline intact while removing more hidden runtime
logic from the ExB wrapper and shrinking the remaining `core.terms`
dependencies.

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
