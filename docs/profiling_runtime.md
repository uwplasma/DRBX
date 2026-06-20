# Profiling And Runtime Workflow

!!! note "Plan authority"
    This page documents profiling commands and runtime evidence. The active
    execution plan is
    [Research-Grade Execution Plan](research_grade_execution_plan.md). If this
    page conflicts with that plan, follow the execution plan and update this
    page afterward.

This page records the supported profiling workflow for `jax_drb` so runtime
work is reproducible instead of based on one-off local shell snippets.

The current public entry point is:

- [scripts/profile_curated_case.py](../scripts/profile_curated_case.py)

It is meant for the real worst offenders and live validation lanes, not only
for compact microbenchmarks.

The script now requires one of:

- `--reference-root /path/to/hermes-3`
- `JAX_DRB_REFERENCE_ROOT=/path/to/hermes-3`

## Recommended CPU Cases

The current highest-value local CPU cases are:

- `neutral_mixed_one_step`
- `recycling_1d_one_step`
- `recycling_dthe_one_step`
- `integrated_2d_recycling_one_step`
- `tokamak_recycling_one_step`

These are the cases where the current same-machine Hermès rerun matrix still
shows either the largest runtime ratios, the largest fidelity gaps, or both.

## Current Measured CPU Results

The latest local profiling pass gives the following reviewer-usable numbers on
this machine:

- `neutral_mixed_one_step`
  - timed local mean dropped from about `1.15 s` to about `0.63 s` after
    vectorizing `_gradient_magnitude`
  - fresh live Hermès rerun ratio is about `4.19x`
  - the visible `NVh` one-step mismatch is now small (`4.47e-6` max absolute
    error); the remaining neutral work is target-adjacent `Nh`/`Ph`
    state-history drift feeding otherwise closed momentum operators
- `recycling_dthe_one_step`
  - timed local mean dropped from about `75.3 s` to about `54.1 s` after the
    reaction/source allocation cleanup, and then to about `52.76 s` after
    caching target-boundary geometry in the recycling runtime model
  - the latest post-metric-selector one-run cProfile pass on this machine
    measured `74.39 s`; the separate RSS run measured `49.25 s` with peak
    process-tree RSS about `231.2 MiB`
  - fresh live Hermès rerun ratio is about `7.17x`
  - the fidelity band stayed essentially unchanged at about `4.9e-3` relative
    RMS on `NVd`
  - the isolated target-recycling operator now also shows a direct kernel-level
    improvement of about `1.17x` from the same cached-geometry path on the CPU
    NumPy RHS

The next heavy CPU optimization target is no longer generic reaction
allocation. The refreshed cProfile still shows the dominant remaining work in:

- the SciPy BDF history path itself
- finite-difference Jacobian assembly
- neutral parallel diffusion
- collision closure
- target recycling / target boundary-source assembly
- prepared-state and boundary setup on the open-field lane

On the latest `recycling_dthe_one_step` cProfile pass, after AMJUEL log-input
reuse and BDF Jacobian-plan reuse, the cumulative top costs were: SciPy
`solve_ivp`/BDF at about `66.4 s`, packed RHS calls at about `64.0 s`, species
RHS assembly at about `61.2 s`, sparse finite-difference Jacobian construction
at about `45.7 s`, reaction sources at about `13.7 s`, AMJUEL fit evaluation
at about `11.4 s`, neutral parallel advection at about `7.6 s`,
collision closure at about `6.3 s`, state preparation at about `6.1 s`, and
target recycling at about `6.1 s`. That split confirms that the next runtime
fix has to attack both the Jacobian/RHS call count and repeated source/closure
work; local threading alone is not the right primary fix for this path.

A follow-up `recycling_dthe_one_step` pass after wiring the diagnostics-free
packed RHS through `fixed_layout_dthe_reaction_sources` and reusing D/T AMJUEL
fits measured `64.45 s` under cProfile and `50.00 s` on the separate RSS run,
with a sampled peak process-tree RSS of about `232.7 MiB`. The source-level
split moved in the intended direction: fixed-layout D/T/He reaction sources
dropped to about `9.64 s`, neutral-ionisation collision-rate assembly dropped
to about `2.72 s`, and AMJUEL polynomial evaluations dropped to `117380` calls
with about `7.81 s` cumulative time. The full solve did not show a defensible
end-to-end speedup because the sparse finite-difference Jacobian still consumed
about `43.3 s` and the packed RHS was still called `11738` times. Treat this
as a validated source-kernel cleanup, not as the final performance result.

The current BDF callback now removes one avoidable source of call inflation:
when SciPy asks for `rhs(t, y)` and then for `jac(t, y)` at the same state, the
Jacobian callback reuses the cached base RHS for that state before applying the
colored sparse finite-difference perturbations. Perturbed Jacobian residuals now
bypass the mutable RHS cache directly, so setting
`JAX_DRB_FD_JACOBIAN_THREADS=<N>` can parallelize the BDF color groups without
thread races in that cache. This is a call-count and execution-policy cleanup,
not yet a full runtime solution. A post-change unprofiled
`recycling_dthe_one_step` timing on this MacBook measured `61.38 s`, which is
within local noise and not a defensible end-to-end speedup over the earlier
`~53 s` unprofiled RSS run. Future heavy reports should therefore include the
new BDF diagnostics counters, `bdf_rhs_callback_seconds`,
`bdf_rhs_evaluation_seconds`, `bdf_rhs_object_evaluation_seconds`,
`bdf_rhs_numpy_conversion_seconds`, `bdf_jacobian_mode`,
`bdf_jvp_batch_size`, and `bdf_jacobian_parallel_workers` in addition to wall
time. The RHS counters are deliberately split so heavy profiles can distinguish
fixed-layout residual work from host conversion and SciPy callback overhead.

A direct timing-only check on the same local machine confirms that this is an
opt-in capability rather than a universal default. With the latest source
cleanup, the serial RSS run measured about `50.00 s`; setting
`JAX_DRB_FD_JACOBIAN_THREADS=2` measured `49.81 s`, while `4` threads measured
`54.57 s`. The BDF residual is still dominated by Python/NumPy host work, so
per-solve color-group threading is not strong-scaling evidence for this lane.
For laptop users, the current robust recommendation remains ensemble-level
parallelism across independent heavy solves, with per-solve BDF threading used
only after a local timing check.

A later JAX-native residual refactor exposed an important profiling lesson:
concrete `StructuredMetrics` arrays are stored as JAX arrays even when the
dynamic state is NumPy. Backend selectors in the hot open-field operators
therefore must be driven by dynamic state/rate arrays, not by static metric
arrays. Treating metric arrays as dynamic accidentally routed the production
packed RHS through eager JAX and slowed one D/T/He RHS call to about
`8e-2 s`. After correcting the selectors, the same initial packed RHS warms at
about `3.7e-3` to `4.2e-3 s`, and a bounded current-code
`recycling_dthe_one_step --skip-cprofile` timing completed in `44.60 s` on
this MacBook. The env-enabled promoted parity gate completed in `44.66 s`.

The refreshed full cProfile/RSS bundle after the active-array startup-step and
neutral ladder-diagnostic pass was:

- command: `run_research_campaign_bundle.py --campaign heavy-recycling-profile --reference-root /path/to/reference/root --output-root /tmp/jax_drb_research_campaigns --timeout-seconds 420`
- wrapped profiler command: `profile_curated_case.py recycling_dthe_one_step --warm-runs 0 --timed-runs 1 --cprofile-top 35 --rss-profile`
- cProfile run: `68.20 s` wall, `1.67e8` Python function calls
- separate unprofiled RSS run: `49.65 s`
- peak sampled process-tree RSS: `227.6 MiB`
- BDF callback counts visible in the profile: `11838` packed RHS evaluations,
  `86` Jacobian callbacks, and `8428` finite-difference color-group
  perturbation residuals
- unprofiled BDF phase counters: `49.60 s` solve time, `47.16 s` fixed-layout
  RHS object evaluation time, `34.18 s` Jacobian callback time, and only about
  `2e-3 s` in RHS NumPy conversion
- top cumulative costs: sparse finite-difference Jacobian construction
  `46.8 s`, packed RHS `64.2 s`, species RHS assembly `61.6 s`, reaction
  sources `10.4 s`, fixed-layout D/T/He reaction sources `9.1 s`, collision
  closure `8.6 s`, open-field state preparation `7.1 s`, and the remaining
  backend dispatch/type-detection helpers inside the repeated host-side RHS
  loop

The backend-selector cleanup reduced `use_jax_backend`/`is_jax_array` overhead
from a visible cProfile hotspot to a smaller residual cost: `use_jax_backend`
now appears at about `5.15 s` cumulative instead of the previous `13 s`-class
cost. That confirms the selector was worth simplifying, but the remaining
dominant terms are still finite-difference Jacobian construction and repeated
host RHS assembly.

The absolute cProfile wall time is intentionally not compared directly with the
unprofiled timing because profiling overhead is large on this Python-heavy
path. The useful conclusion is the split: the full run is still dominated by
sparse finite-difference Jacobian assembly plus repeated host-side RHS
assembly, so the next fix remains fixed-layout JAX residual kernels and
JVP/Jacobian-action solves, not another local threading sweep.

The first real transformable recycling gate now has its own profile artifact:
`docs/data/runtime_profile_artifacts/recycling_1d_jax_linearized_gate/`. It
profiles the hydrogen `1D-recycling` fixed-layout backward-Euler residual
through `solver_mode="jax_linearized"` rather than the host-backed adaptive BDF
runner. The local run used cProfile, a process-tree RSS sampler, a JAX
Perfetto trace, a device-memory profile, persistent compilation cache, and an
XLA text dump. The cProfile+trace run completed the physical solve in about
`2.06 s` with residual `2.49e-12`; the separate RSS run completed in about
`0.83 s`, with sampled process-tree peak RSS about `2.85 GiB` and sub-MiB
incremental RSS during the timed gate. Solver diagnostics show one
JAX linearization refresh, one residual evaluation, no line search, and no
fallback. This is the right evidence for transformability of the fixed-layout
hydrogen BE residual; it is not yet the heavy D/T/He adaptive-BDF result.

The shared sparse Newton backend now records per-step diagnostics for:

- residual evaluation count and wall time
- sparse finite-difference Jacobian refresh count and wall time
- sparse/direct or Krylov linear-solve wall time
- line-search wall time
- fallback use

Those diagnostics are intentionally attached to the solver step info rather
than hidden inside one profiler script. They can now be surfaced by recycling,
neutral, and future tokamak campaign packages when the paper needs
phase-resolved runtime evidence. The sparse finite-difference Jacobian path
also precomputes the CSC row/column extraction plan once per solve, so each
Newton refresh no longer rebuilds the same color-group indexing metadata.

For residuals that are already JAX-transformable, the solver package also has a
grouped sparse-JVP Jacobian builder. It uses `jax.linearize` and one pushed
direction per color group, avoiding finite-difference residual perturbations
altogether. The heavy recycling RHS is not yet in that category, so the
remaining runtime work is to migrate the dominant recycling residual kernels to
JAX-native array code before making the JVP path a production backend.
The sparse-JVP builder now batches those color-group pushes with `jax.vmap`.
That makes the derivative path closer to the JAX autodiff cookbook model:
linearize once, then push a matrix of tangent directions through the same
linearized residual. The public `batch_size` parameter should be used during
profiling to separate memory pressure from dispatch overhead.

The SciPy BDF compatibility path can now exercise that same derivative
interface with:

```bash
JAX_DRB_RECYCLING_BDF_JACOBIAN_MODE=jvp \
PYTHONPATH=src python scripts/profile_curated_case.py recycling_dthe_one_step \
  --reference-root /path/to/reference/root \
  --output-dir tmp/profiles/recycling_dthe_one_step_jvp_bdf \
  --warm-runs 0 \
  --timed-runs 1 \
  --rss-profile
```

This is intentionally an opt-in profiling lane, not the default solver mode.
It is only meaningful when the callback residual is transformable enough for
JAX to see the dynamic state. If it falls back to host callbacks or forces
large host-device copies, the finite-difference BDF callback remains the
validated compatibility path and the result should be treated as diagnostic
evidence for the next residual-porting step.

The next BDF migration gate keeps the same SciPy BDF output-window timestepper
but switches both the RHS seam and Jacobian callback through a named runtime
mode:

```bash
PYTHONPATH=src python scripts/profile_curated_case.py recycling_dthe_one_step \
  --reference-root /path/to/reference/root \
  --override runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp \
  --output-dir tmp/profiles/recycling_dthe_one_step_fixed_full_field_jvp_bdf \
  --warm-runs 0 \
  --timed-runs 1 \
  --rss-profile \
  --jax-trace
```

The resulting history diagnostics should report
`bdf_rhs_backend="fixed_full_field_array"` and `bdf_jacobian_mode="jvp"`.
Compare its runtime, RSS, callback counts, and reference errors against the
default `bdf` profile before promoting it.

The compact parity gate has already been run on `recycling_1d_one_step`:
`compare_recycling_transient_modes.py --mode bdf --mode
bdf_fixed_full_field_jvp --field Pe --require-fixed-jvp-diagnostics
--require-bdf-pairwise-max 1e-5` passes with `Pe` pairwise delta `6.28e-6`.
It is not a speedup: the fixed-JVP route takes about `59.9 s` versus about
`8.2 s` for the default BDF route. The JVP callback subphase counters show
about `36.8 s` in repeated `jax.linearize`, about `20.0 s` in batched tangent
pushes, and negligible time in tangent construction or sparse assembly. Heavy
D/T/He fixed-JVP profiling should therefore be repeated only after the JVP
materialization path is improved or replaced by a native matrix-free solve.

The related JAX Newton path is matrix-free: JAX GMRES receives the linearized
Jacobian action as a callable rather than a materialized sparse matrix. This is
the preferred algorithmic target for future differentiable recycling kernels,
but it is intentionally not claimed as a production speedup until the residual
itself stops crossing the host/SciPy boundary.
Use `scripts/profile_recycling_jax_linearized_gate.py --jit-residual` to test
the opt-in pre-JIT residual seam on these gates. Local comparison runs showed
that this flag is diagnostic rather than a default candidate: it can reduce
reported linear-solve subphase time on a nontrivial D/T/He fixed-layout gate,
but current closure rebuilding and compilation overhead still keep total
wall-clock time from improving consistently.

The promoted active-source backend is now selectable in the same profiler with
`--rhs-backend promoted_active_sources`. For local campaign runs, use:

```bash
PYTHONPATH=src python scripts/run_research_campaign_bundle.py \
  --campaign dthe-promoted-active-sources-profile-gate \
  --reference-root /path/to/reference/root
```

A bounded `ny=100`, `dt=1e-4` local run against the lightweight D/T/He fixture
now requires a real JAX-linearized Newton/JVP update. It passed with
`solver_mode=promoted_active_sources_jax_linearized`,
`rhs_backend=promoted_active_sources`, active size `1900`, state size `1979`,
residual infinity norm `1.74e-12`, one nonlinear iteration, one JAX-GMRES
linear solve, five matrix-free operator calls, two residual evaluations, warm
jitted profiled solve time `8.29 s`, and sampled RSS run time `8.00 s` with
peak RSS delta `362 MiB`. Matched
`active_array` and `fixed_full_field_array` runs closed to the same residual in
`7.81 s` and `7.80 s`, respectively. This is nontrivial profiling evidence for
the source-kernel migration seam; it is still not a long-window or
default-promotion claim. Easier `ny=200` and `ny=400` promoted-source sweeps
passed as size/RSS sanity checks, but they performed zero nonlinear or linear
solves and should not be used as speedup evidence.

The campaign wrapper keeps cProfile enabled but now performs one warmup run and
requires the jitted JVP linear operator before the profiled solve. This avoids
using cold XLA compilation as the primary performance signal. In the local
post-warmup profile, the one-update solve spent `5.08 s` in JAX-GMRES/JVP,
`2.80 s` in residual evaluation and linearization, and `0.31 s` in line search.
Those numbers identify the next performance target as residual/JVP kernel cost
and Krylov operator count, not Python dictionary assembly.

The profiler also exposes `--line-search-mode {backtracking,full_step}` as a
first-class equivalent of
`runtime:recycling_jax_linear_line_search_mode=<mode>`. A matched warm
`full_step` D/T/He promoted-source probe preserved the residual
(`1.74e-12`) and removed the measurable line-search time, but the profiled
solve was effectively unchanged (`8.36 s` versus `8.29 s` for default
backtracking) with the same five JVP operator calls. Treat `full_step` as a
diagnostic lower-bound probe for accepted Newton updates, not as a default or
publication speedup until long-window fixed-BDF2 and full-output profiles
improve under the same mode.

A deliberately aggressive `dt=1.0` one-update probe failed on
`promoted_active_sources`, `active_array`, and `fixed_full_field_array` with
residuals near `2.73e2`. Treat that as a shared nonlinear/Krylov robustness
limit for the current one-update profile gate, not as a promoted-source
parity failure.

Solver-health counters are now guarded against overstating stalled runs. If a
JAX-linearized solve exits early after a rejected or stagnant line-search
sequence, `nonlinear_iterations` reports the actual attempted updates rather
than the requested maximum budget. In the hard `dt=1.0`, `mesh:ny=100`
promoted-source probe, scale-1 backtracking reduced the residual to
`1.41e2` and correctly reported two nonlinear attempts, two linear solves, ten
operator calls, five residual evaluations, and three line-search trials; a true
full-step probe diverged to `7.47e23`.

An opt-in minimum backtracking floor is available through
`runtime:recycling_jax_linear_line_search_min_step_scale` and the profiling
flag `--line-search-min-step-scale`. The default remains `1/64`. On the same
hard `dt=1.0`, `mesh:ny=100` promoted-source probe, lowering the floor to
`1e-4` did not improve the final residual; the solve still stopped at
`1.41e2` after two nonlinear attempts, ten operator calls, five residual
evaluations, and three line-search trials. The next large-step robustness work
should therefore target nonlinear globalization and nonfinite update handling,
not just the line-search floor.

The profiling gate can now require finite JVP update health with
`--require-linear-operator-finite`. This check uses
`diagnostics.linear_operator_finite`, which is populated when
`runtime:recycling_jax_linear_diagnose_update_residual=true` evaluates
`J(u) \delta u + R(u)` for the Krylov update. On the hard `dt=1.0`,
`mesh:ny=100` promoted-source probe, this diagnostic originally failed because
the linearized action touched clipped characteristic and collision
relative-speed square roots. The fixed-JVP square-root helper now selects the
inactive-branch subgradient at the clipping point, so the same probe passes
`--require-linear-operator-finite`. This does not make the large-step solve
converged: the residual remains a nonlinear globalization and
preconditioning problem, not a nonfinite operator-action problem.
The follow-up 5-iteration probe exposed one more zero-tangent singularity at
the accepted 4-step state. The new `diagnostics.linear_update_finite` and
`diagnostics.linear_update_inf_norm` fields showed that the Krylov update vector
was finite, while `J(u) \delta u + R(u)` was not. Term localization mapped the
bad residual entry to the electron pressure field at the upper target-adjacent
cell. After routing the full zero-current sheath sonic square roots through the
same finite-JVP helper, the hard D/T/He gate remains finite for longer Newton
runs and a 14-iteration request converges in 13 iterations to residual
`1.75e-11`. The cost is still high, about `101.90 s` locally with 65
matrix-free operator calls, so this is correctness evidence rather than a
performance promotion.

Performance gates should also require nonlinear convergence when the artifact
is used for speed or scaling claims. The profiling command supports
`--require-converged`, which fails unless the report contains
`diagnostics.converged=true`. This became necessary after the finite-JVP fixes:
`runtime:recycling_jax_linear_jit_residual=true` reduced a bounded four-step
D/T/He profile from `58.4 s` to `48.0 s`, but on the full hard gate it stalled
at residual `3.93` after 11 nonlinear iterations. A smaller line-search floor
kept the run finite but still did not converge (`0.106` after 20 iterations).
Those artifacts are useful profiling diagnostics, but they are not valid
runtime speedup evidence.

The current post-finite-JVP CPU sweep rules out several simple promotion knobs
for this gate. Disabling update diagnostics does not materially change runtime
(`101.3 s` to `100.8 s`), direct operator counting is slower (`107.6 s`),
starting line search at `0.5` fails to converge within the retained budget, and
looser inner tolerances either increase work or stall. A restart-5 GMRES run
also failed to improve runtime or convergence. The next performance lane should
therefore reduce residual/JVP kernel cost or introduce a genuinely spectral
preconditioner; it should not promote these simple toggles.

The current GPU evidence for the heavier fixed-layout seam lives in:

- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_gpu_current/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_gpu/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_gpu_warm/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_ny100_dt1e4_cpu/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_ny100_dt1e4_gpu/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_ny200_dt1e4_cpu/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_ny200_dt1e4_gpu/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_batched_jvp_gate_cpu/profile_summary.json`

Those summaries show equal residual closure between CPU and GPU. The
same-fidelity `dt=1.0` D/T/He gate now passes on GPU but is much slower
(`109.49 s` versus `8.92 s`) and uses more sampled process-tree RSS
(`12.34 GiB` versus `2.86 GiB`) than the CPU artifact. The correct profiling
conclusion is that the current full-field residual seam is
accelerator-executable but not GPU-efficient; reviewer-facing speedup claims
need active-array residuals, smaller compiled kernels, or a batched heavy
ensemble.

The larger real-kernel D/T/He GMRES gates use `mesh:ny=100` and `mesh:ny=200`
with `timestep=1e-4`, which forces a real JAX-linearized Newton/GMRES update
rather than the near-trivial one-residual gate. The matched local CPU runs
closed to `1.74e-12` and `7.47e-11` in about `7.28 s` and `7.32 s`,
respectively. The matched office-GPU runs closed to the same residuals but
took about `30.19 s` and `30.76 s` after large shape-specific compilation
warmups. GPU sampled RSS deltas were lower, roughly `341-344 MiB` versus
`585-694 MiB` locally, but the current JAX GMRES path is not a speedup on this
problem family. For the release, GPU acceleration should therefore remain a
measured development lane, not a promoted production claim.

The current production split should be read narrowly. The fixed-layout bridge
is now the state contract for the implicit recycling steppers, so future
term-level ports no longer have to rediscover packing, active slices, or
controller-scalar handling. The legacy SciPy BDF history mode still calls a
host RHS many times and still builds finite-difference sparse Jacobians, so its
profile remains the evidence for the next refactor rather than evidence that
the JVP path is already the default heavy-solve backend.

The full D/T/He active-array output-window sparse-JVP profile remains a
negative runtime gate on the local CPU. A June 19, 2026 bounded run of
`dthe-active-array-output-jvp-profile` reached the sparse-JVP Jacobian
construction path and timed out before writing an artifact. The interrupted
trace showed the cost inside repeated `jax.linearize` calls from the
SciPy-BDF Jacobian hook. The fixed-layout adapter now avoids redundant
`jnp.asarray(..., dtype=float64)` casts for already-float64 JAX arrays and
tracers, but this is trace hygiene rather than a full performance solution.
Promotion still requires moving this output-window route away from repeated
host-driven sparse-JVP materialization or replacing it with the matrix-free
fixed-BDF2/JAX-GMRES path that already has strict hydrogen health gates.

The bounded D/T/He replacement gate for that direction is:

```bash
PYTHONPATH=src python scripts/run_research_campaign_bundle.py \
  --campaign dthe-fixed-bdf2-active-array-gate \
  --reference-root tests/fixtures/reference-root
```

The retained local artifact is
`docs/data/runtime_profile_artifacts/recycling_dthe_fixed_bdf2_active_array_direct_counting_cpu/profile_summary.json`.
It runs `recycling_dthe_one_step` at `dt=1e-4` for two output steps through the
active-array fixed-BDF2/JAX-GMRES route, reports one startup step and one BDF2
corrector, two JAX-GMRES solves, four residual evaluations, zero failed or
unconverged subsolves, maximum residual `4.10e-14`, and `11.27 s` mode elapsed
time. This is solver-health evidence for the matrix-free D/T/He output-window
route; it is not yet a GPU or default-backend promotion claim.

Before broadening claims beyond the compact two-step gate, run the longer
D/T/He matrix-free gate:

```bash
PYTHONPATH=src python scripts/run_research_campaign_bundle.py \
  --campaign dthe-fixed-bdf2-active-array-long-window-gate \
  --reference-root tests/fixtures/reference-root
```

The retained artifact is
`docs/data/runtime_profile_artifacts/recycling_dthe_fixed_bdf2_active_array_long_window_cpu/profile_summary.json`.
It runs the same case for eight output steps at `dt=1e-4`, reports one startup
step, seven BDF2 correctors, eight JAX-GMRES solves, sixteen residual
evaluations, zero failed or unconverged subsolves, maximum residual
`4.10e-14`, `32.40 s` total linear-solve time, `10.86 s` total residual
evaluation time, and `44.92 s` mode elapsed time. This is the strongest local
D/T/He evidence for the matrix-free fixed-BDF2 route so far, but it remains a
bounded diagnostics-only gate rather than a default-solver or GPU-speedup
claim.

For physical-output parity against the stable BDF route, run:

```bash
PYTHONPATH=src python scripts/run_research_campaign_bundle.py \
  --campaign dthe-fixed-bdf2-active-array-physical-parity-gate \
  --reference-root tests/fixtures/reference-root
```

The retained artifact is
`docs/data/runtime_profile_artifacts/recycling_dthe_fixed_bdf2_active_array_physical_parity_cpu/profile_summary.json`.
It runs both `bdf` and `fixed_bdf2_active_array_jax_linearized` for the same
eight-step D/T/He window at `dt=1e-4` and gates the active-mesh field delta
with `--require-fixed-bdf2-pairwise-max=2.5e-7`. The measured worst delta is
`1.745e-7` on `NVd+`; `Pd+` is next at `5.58e-8`, and all density/pressure
neutral-field deltas are near `1e-12` to `1e-9`. This is physical-output
parity evidence for the bounded matrix-free route. It still does not promote
the route as the default production output-window solver until larger physical
windows and CPU/GPU profiles pass with the same route.

The next retained ramp is:

```bash
PYTHONPATH=src python scripts/run_research_campaign_bundle.py \
  --campaign dthe-fixed-bdf2-active-array-parity-ramp-gate \
  --reference-root tests/fixtures/reference-root
```

The retained artifact is
`docs/data/runtime_profile_artifacts/recycling_dthe_fixed_bdf2_active_array_parity_ramp_cpu/profile_summary.json`.
It increases the same eight-step D/T/He comparison to `dt=1e-3` and gates the
active-mesh field delta with `--require-fixed-bdf2-pairwise-max=2.5e-5`. The
measured worst delta is `1.761e-5` on `NVd+`, followed by `5.27e-6` on `Pd+`
and `1.28e-7` on `Pe`. Fixed-BDF2 remained solver-clean with eight
JAX-GMRES solves, sixteen residual evaluations, zero failed or unconverged
subsolves, maximum residual `4.05e-11`, `32.34 s` in linear solves, and
`10.91 s` in residual evaluations. A scratch `dt=1e-2`, two-step probe also
converged with residual `3.51e-8`, but the worst `NVd+` delta grew to
`1.55e-3`; do not promote the next timestep decade until either accuracy is
improved or the accepted tolerance is justified by a physics observable rather
than raw field max-norm.

The first retained `dt=1e-2` observable screen is deliberately narrower:

```bash
PYTHONPATH=src python scripts/run_research_campaign_bundle.py \
  --campaign dthe-fixed-bdf2-active-array-scalar-observable-gate \
  --reference-root tests/fixtures/reference-root
```

The retained artifact is
`docs/data/runtime_profile_artifacts/recycling_dthe_fixed_bdf2_active_array_scalar_observable_cpu/profile_summary.json`.
It keeps the same two-step D/T/He comparison but gates only scalar density and
pressure observables (`Nd+`, `Pd+`, `Nd`, `Pd`, `Pe`) with
`--require-fixed-bdf2-pairwise-l2-rel-max=5e-5` and
`--require-fixed-bdf2-pairwise-inventory-rel-max=1e-5`. The measured worst
scalar relative L2 error is `3.969e-5` on `Pd+`, and the worst unweighted
active-inventory relative error is `4.721e-6` on `Pd+`. Fixed-BDF2 remains
solver-clean with residual `3.51e-8`, two JAX-GMRES solves, four residual
evaluations, zero failed or unconverged subsolves, and `10.74 s` elapsed time.
This is not a speedup claim and not full momentum parity: the all-field probe
still reports `NVd+` as the pointwise offender and near-zero `NVd` inventory
makes relative momentum inventory ratios non-diagnostic.

The retained full-field `dt=1e-2` screen adds explicit internal substepping:

```bash
PYTHONPATH=src python scripts/run_research_campaign_bundle.py \
  --campaign dthe-fixed-bdf2-active-array-substepped-full-field-gate \
  --reference-root tests/fixtures/reference-root
```

The retained artifact is
`docs/data/runtime_profile_artifacts/recycling_dthe_fixed_bdf2_active_array_substepped_full_field_cpu/profile_summary.json`.
It uses `runtime:recycling_fixed_bdf2_max_internal_timestep=2.5e-3`, so each
output window is split into four implicit fixed-BDF2 substeps. This closes the
same `dt=1e-2` full-field max-norm gate with worst active-mesh pointwise
delta `1.099e-4` on `NVd+`, below the `1.25e-4` threshold. The fixed route
remains solver-clean with maximum residual `6.17e-10`, eight internal
substeps, eight JAX-GMRES solves, sixteen residual evaluations, zero failed or
unconverged subsolves, `32.10 s` in linear solves, `10.70 s` in residual
evaluations, and `44.43 s` elapsed time after warm compilation. The conclusion
is explicit: substepping fixes the momentum/full-field accuracy target, but
runtime and GPU scaling still require a cheaper preconditioned or compiled
linear solve before default promotion.

For output-window fixed-BDF2 solver-health checks, use the strict active-array
linearized residual gate:

```bash
PYTHONPATH=src python scripts/run_research_campaign_bundle.py \
  --campaign fixed-bdf2-linear-update-residual-gate \
  --reference-root tests/fixtures/reference-root
```

The retained local hydrogen fixture artifact is
`docs/data/runtime_profile_artifacts/recycling_1d_fixed_bdf2_active_array_linear_update_residual_cpu/profile_summary.json`.
It records zero failed linear solves,
`fixed_bdf2_max_linear_update_residual_inf_norm=1.58e-8`,
`fixed_bdf2_max_linear_update_relative_residual=1.02e-5`, and `23`
post-GMRES residual-action checks. Those checks cost about `4.03 s`, so this
gate should be interpreted as strict solver-health evidence paired with the
cheaper direct-counting gate, not as a speedup result.

The batched residual/JVP gate is the current fixed-layout differentiability
and parallel-throughput test:

```bash
PYTHONPATH=src python scripts/profile_recycling_batched_jvp_gate.py \
  --case dthe \
  --rhs-backend fixed_full_field_array \
  --override mesh:ny=100 \
  --batch-sizes 1,4,16,64 \
  --timed-runs 3 \
  --disable-pmap \
  --output-dir docs/data/runtime_profile_artifacts/recycling_dthe_batched_jvp_gate_cpu
```

When neither `--reference-root` nor `--input-path` is supplied, this direct
profiler uses the lightweight fixture decks committed under
`tests/fixtures/reference-root`. Pass `--reference-root /path/to/reference/root`
for full reference-suite decks, or `--input-path /path/to/BOUT.inp` for a
single staged deck. The default `fixed_full_field_array` backend is the
release-facing JAX-native evidence path; `--rhs-backend active_array` is the
opt-in migration seam that routes the same D/T/He residual through
`build_fixed_array_rhs` before term-specific kernels are promoted, and
`--rhs-backend host_bridge` is retained only for comparisons against the older
host bridge.

Each refreshed `profile_summary.json` now records per-batch
`*_states_per_second` fields and a top-level `throughput_summary`. That summary
identifies the batch sizes swept, the largest batch, the best residual/JVP
speedups against serial same-kernel calls, and the best batched or pmap JVP
throughput in `states_per_second`. This keeps the local profiling contract
reviewer-usable without requiring a separate notebook to interpret raw sample
lists. Each profile directory also receives `profile_progress.jsonl`, written
incrementally during problem construction, base residual/JVP warmup, derivative
checks, deterministic direction construction, batched residual warmup, batched
JVP warmup, serial warmup, and each batch. This file is intentionally small and
is the first place to inspect when a long CPU/GPU run is killed before
`profile_summary.json` is complete. The direct profiler also accepts
`--residual-partition-size` and `--jvp-partition-size` to split a large batch
into several same-kernel chunks. That is a compile-size and memory-pressure
control for GPU readiness, not a speedup claim. Partition sizes that divide the
requested batch sizes avoid compiling an extra remainder shape.

For a bounded local smoke refresh that does not spend CI minutes or require a
private reference checkout:

```bash
PYTHONPATH=src python scripts/profile_recycling_batched_jvp_gate.py \
  --case dthe \
  --batch-sizes 1,2 \
  --timed-runs 1 \
  --disable-pmap \
  --skip-objective-grad-check \
  --output-dir tmp/profiles/recycling_dthe_batched_jvp_gate_smoke
```

The retained local CPU fixed-full-field artifact now sweeps batches through
`64` states and shows about `2.28x` residual throughput speedup and `1.96x`
JVP throughput speedup over serial same-kernel calls, with batched/serial
residual and JVP mismatch at roundoff. Its best residual throughput is about
`3.13e4` states/s and its best JVP throughput is about `8.56e3` states/s. The
residual JVP agrees with centered finite difference to about `5.97e-9`, the
objective directional derivative agrees to about `1.34e-7`, and the reusable
linearized-action diagnostic agrees with direct JVPs to about `3.47e-18`.

The active-array migration seam has a separate retained CPU artifact:

```bash
PYTHONPATH=src python scripts/profile_recycling_batched_jvp_gate.py \
  --case dthe \
  --rhs-backend active_array \
  --override mesh:ny=100 \
  --batch-sizes 1,4,16,64 \
  --timed-runs 3 \
  --disable-pmap \
  --output-dir docs/data/runtime_profile_artifacts/recycling_dthe_active_array_batched_jvp_gate_cpu
```

That artifact reaches about `2.55x` residual throughput speedup and `2.02x`
JVP throughput speedup through batch `64`, with best residual and JVP
throughputs of about `3.14e4` and `9.03e3` states/s. It retains the same
finite-difference derivative checks and linearized-action diagnostic as the
fixed-full-field artifact. This is the current best local evidence that the
transformable active-array residual can be batched and differentiated without
falling back to Python residual loops.

The same direct profiler can also run one opt-in matrix-free Newton-update
health check on the fixed-layout residual. This is not part of the throughput
timing loop and does not change the production BDF default; it is a bounded
gate for preconditioner and JAX-native implicit-solver work:

```bash
PYTHONPATH=src python scripts/profile_recycling_batched_jvp_gate.py \
  --case dthe \
  --rhs-backend active_array \
  --override mesh:ny=16 \
  --batch-sizes 1 \
  --timed-runs 1 \
  --disable-pmap \
  --skip-objective-grad-check \
  --check-linearized-update \
  --linearized-update-tolerance 1e-8 \
  --linearized-update-restart 8 \
  --linearized-update-maxiter 8 \
  --linearized-update-jit-operator \
  --linearized-update-preconditioner none \
  --output-dir docs/data/runtime_profile_artifacts/recycling_dthe_active_array_linearized_update_cpu
```

The retained local CPU artifact uses the in-tree D/T/He fixture at `ny=16`.
It records GMRES solver status `0`, a successful jitted matrix-free operator,
linear-update relative residual `3.26e-16`, and post-update nonlinear residual
`2.11e-11` on the active-array fixed-layout residual. The update check took
about `4.33 s` inside a `14.6 s` campaign run. The update solve reuses the
same `jax.linearize` action already built for the serial/batched JVP
diagnostic, so the retained artifact reports `linearization_reused=true` and
zero solve-only Python action callbacks when the jitted linear operator is
enabled. Treat this as solver-health
evidence for future preconditioner work, not a speedup claim.

The strict gate keeps the post-GMRES linearized residual diagnostic enabled,
which applies one additional matrix-free residual action to report
`linear_update_residual_inf_norm` and `linear_update_relative_residual`.
After the strict gate has passed, production-style throughput probes can add
`--skip-linearized-update-residual-diagnostic` to avoid that extra action while
still recording solver status, update size, candidate nonlinear residual, and
whether the residual diagnostic was skipped.
The same command is available through
`scripts/run_research_campaign_bundle.py --campaign
dthe-active-array-linearized-update-throughput-probe`, which writes to a
separate retained-artifact directory and leaves the strict gate unchanged.
The retained local fixture artifact is
`docs/data/runtime_profile_artifacts/recycling_dthe_active_array_linearized_update_throughput_cpu/profile_summary.json`;
it reports solver status `0`, candidate nonlinear residual `2.11e-11`,
`linear_update_residual_checked=false`, and update-check wall time about
`3.89 s`. This is a timing probe, not a replacement for the strict residual
health gate above.

The companion JVP-diagonal preconditioner screen is:

```bash
PYTHONPATH=src python scripts/run_research_campaign_bundle.py \
  --campaign dthe-active-array-linearized-update-jvp-diag-gate \
  --reference-root /path/to/reference/root
```

On the same `ny=16` D/T/He fixture, `jvp_diag` built a `304`-entry diagonal in
about `0.61 s` and preserved solver health: GMRES status `0`, linear-update
relative residual `2.94e-15`, and post-update nonlinear residual `2.11e-11`.
It did not improve this residual family. The diagonal entries lie between
`1.000000007` and `1.000043761` in absolute value, so the preconditioner is
nearly the identity and the update check took about `4.99 s`, compared with
`4.33 s` without preconditioning. Keep it as an
auditable diagnostic and move performance effort toward transport/block
preconditioners or cheaper residual/JVP kernels.

For larger GPU or multi-device evidence, use the research-campaign wrapper
rather than hand-editing decks. These campaigns enable repeated timings,
persistent compilation cache, optional JAX traces, device-memory profiles, and
pmap parity metadata where applicable:

```bash
REFERENCE_ROOT=/path/to/reference/root
test -f "$REFERENCE_ROOT/tests/integrated/1D-recycling-dthe/data/BOUT.inp"

JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=src python scripts/run_research_campaign_bundle.py \
  --campaign all-gpu \
  --reference-root "$REFERENCE_ROOT" \
  --timeout-seconds 7200
```

The `gpu-dthe-jax-linearized-gate` command is a large fixed-layout residual
trace/memory run. The `gpu-dthe-batched-jvp-gate` command is the fixed-full-field
batched residual/JVP throughput run. The
`gpu-dthe-active-array-batched-jvp-gate` command measures the same family after
the residual is routed through the active-array backend, currently with pmap
disabled and with residual/JVP batch partitions of `16` so the first GPU
artifact is a single-device compiler/memory health probe rather than a
multi-device speedup claim. Neither command promotes the full output-window BDF
solve as GPU-accelerated; they are evidence-gathering gates for the residual
and derivative kernels that must become production-safe first.
The wrapper intentionally rejects reference roots that do not contain
`tests/integrated/1D-recycling-dthe/data/BOUT.inp`; that failure means the
reference prerequisite is missing, not that the GPU gate has failed.

For a reduced self-contained GPU readiness probe, use the direct profiler. It
defaults to the in-tree fixture decks when no input is supplied; pass a single
staged `BOUT.inp` with `--input-path` only when testing a nonstandard deck:

```bash
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=src python scripts/profile_recycling_batched_jvp_gate.py \
  --case dthe \
  --rhs-backend active_array \
  --override mesh:ny=100 \
  --batch-sizes 2,4,8,16 \
  --timed-runs 3 \
  --disable-pmap \
  --skip-objective-grad-check \
  --residual-partition-size 16 \
  --jvp-partition-size 16 \
  --jax-trace \
  --device-memory-profile \
  --compilation-cache-dir tmp/jax_cache/recycling_dthe_active_array_batched_jvp_gate_gpu_readiness \
  --output-dir tmp/profiles/recycling_dthe_active_array_batched_jvp_gate_gpu_readiness
```

For a nonstandard staged deck, add:

```bash
--input-path /path/to/1D-recycling-dthe/data/BOUT.inp
```

The current `office` GPU evidence for active-array batched JVPs should be read
narrowly. A tiny `ny=16`, batch `1,2`, single-device CUDA probe on one RTX
A4000 completed with JVP/finite-difference relative error `3.95e-10`, batch-2
JVP throughput about `1.45e3` states/s, and batch-2 residual throughput about
`2.97e3` states/s, proving that the reduced active-array residual can execute
on the GPU. Its progress log recorded base residual warmup `2.93 s`, base JVP
warmup `5.19 s`, batch-2 residual warmup `3.92 s`, and batch-2 JVP warmup
`6.77 s`; the next GPU blocker is therefore the compiled batched JVP transform,
not a missing CUDA code path. A follow-up partitioned probe is retained at
[recycling_dthe_active_array_batched_jvp_partition_probe_gpu/profile_summary.json](data/runtime_profile_artifacts/recycling_dthe_active_array_batched_jvp_partition_probe_gpu/profile_summary.json).
It uses the same `ny=16` active-array residual on one RTX A4000, batch `4`,
and residual/JVP partition size `2`. It reports backend `gpu`, partition count
`2` for residual and JVP calls, JVP/finite-difference relative error
`3.95e-10`, batch-4 residual throughput about `2.06e3` states/s, and batch-4
JVP throughput about `1.15e3` states/s. This proves the partitioned CUDA path
executes and records complete progress metadata; it is not speedup evidence.
Larger `ny=100` active-array pmap and
single-device probes did not complete within the practical profiling window;
both were host/compiler or memory bound, allocated roughly `12 GiB` of GPU
memory, showed near-zero GPU utilization, and wrote no JSON summary. This is
negative evidence for claiming GPU speedup today, and it points the next
implementation step toward reducing compiled active-array JVP size before
attempting multi-GPU promotion.

The current GPU speedup evidence instead comes from the source-term throughput
gate:

```bash
PYTHONPATH=src python scripts/profile_atomic_rate_throughput_gate.py \
  --output-dir docs/data/runtime_profile_artifacts/atomic_rate_throughput_gate_cpu
```

The matched office-GPU artifact lives in
`docs/data/runtime_profile_artifacts/atomic_rate_throughput_gate_gpu/profile_summary.json`.
At `4,194,304` temperature points the GPU is about `2.5x` faster than the
local CPU for the batched rate surface and about `2.0x` faster for the
autodiff derivative. The same report checks a scalar mean-rate sensitivity to
a log-temperature shift; autodiff and centered finite difference agree at
about `1e-10` relative error on CPU and GPU. This is the correct release
claim: dense JAX-native source kernels accelerate on GPU today; full heavy
recycling output-window GPU speedup is still blocked by host/SciPy residual
structure.

The source-throughput profiler also has an opt-in `--enable-pmap` flag. It is
not enabled in the committed office-GPU artifact, so that artifact remains a
single-device GPU result. When enabled, the profiler now runs a device-level
identity-map sanity check before constructing the real source-kernel pmap
timings. If that runtime sanity check or the subsequent real-kernel parity
check fails, the JSON records the failure and leaves pmap speedups unset.
Multi-device source speedup should therefore not be claimed until the
device-level sanity gate, the real-kernel parity gate, and the matching
committed summary all pass.

## Basic Usage

From the repo root:

```bash
PYTHONPATH=src python3 scripts/profile_curated_case.py neutral_mixed_one_step \
  --reference-root /path/to/hermes-3 \
  --output-dir tmp/profiles/neutral_mixed_one_step \
  --warm-runs 1 \
  --timed-runs 2 \
  --rss-profile
```

The script writes:

- `profile_summary.json`
- `cprofile_top.txt`
- process-tree peak RSS fields in the summary when `--rss-profile` is enabled

and, when requested:

- `jax_trace/` for TensorBoard / Perfetto-compatible traces
- `device_memory_profile.prof` for JAX device-memory snapshots

When `--rss-profile` and cProfile are both enabled, the script collects RSS on
a separate unprofiled run so the sampler thread does not contaminate the
cProfile table.

## JAX Trace And Perfetto

To capture a JAX trace that can be opened in TensorBoard/XProf or uploaded to
Perfetto:

```bash
PYTHONPATH=src python3 scripts/profile_curated_case.py recycling_1d_one_step \
  --reference-root /path/to/hermes-3 \
  --output-dir tmp/profiles/recycling_1d_one_step \
  --jax-trace
```

This uses the official JAX tracing path described in:

- [JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html)
- [`jax.profiler.start_trace`](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.start_trace.html)

## Device Memory Profiles

On GPU-capable systems, the same script can also snapshot device memory:

```bash
PYTHONPATH=src python3 scripts/profile_curated_case.py tokamak_turbulence_one_step \
  --reference-root /path/to/hermes-3 \
  --output-dir tmp/profiles/tokamak_turbulence_one_step \
  --jax-trace \
  --device-memory-profile
```

This follows the official JAX guidance in:

- [JAX device memory profiling](https://docs.jax.dev/en/latest/device_memory_profiling.html)
- [JAX GPU memory allocation notes](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)

## Compilation Cache And XLA Dumps

The script can also set up two useful runtime diagnostics before importing JAX:

- persistent compilation cache
- XLA dump directory

Example:

```bash
PYTHONPATH=src python3 scripts/profile_curated_case.py tokamak_recycling_one_step \
  --reference-root /path/to/hermes-3 \
  --output-dir tmp/profiles/tokamak_recycling_one_step \
  --compilation-cache-dir tmp/jax_cache \
  --xla-dump-dir tmp/xla_dump \
  --jax-trace
```

The supporting JAX references are:

- [persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
- [JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html)

## GPU Workflow On Self-Hosted Machines

The self-hosted GPU readiness audit on 2026-06-02 found that the reachable
`office` machine exposes:

- two `NVIDIA RTX A4000` GPUs
- no valid wrapper reference root at
  `tests/integrated/1D-recycling-dthe/data/BOUT.inp`

Do not cite `docs/data/jax_native_profile_audit_artifacts/data/jax_native_profile_audit.json`
as GPU-backed until that artifact is regenerated on a CUDA-visible backend.
The committed native-profile audit currently records the `cpu` backend and one
`TFRT_CPU_0` device. GPU-backed release evidence is limited to the committed
profile summaries listed above, especially the fixed-layout D/T/He gate
summaries and the dense atomic-rate throughput gate.

Before running the wrapper on `office` or any other self-hosted GPU node, prove
both prerequisites in the same shell:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
REFERENCE_ROOT=/path/to/reference/root
test -f "$REFERENCE_ROOT/tests/integrated/1D-recycling-dthe/data/BOUT.inp"
PYTHONPATH=src python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

That is the correct runtime split for the current codebase:

- compact native JAX lanes are ready for CPU/GPU audit when regenerated on the
  intended backend
- heavy recycling lanes are still primarily CPU/host-side optimization targets

## Current Interpretation Standard

Profiler output should not be read in isolation.

- `cProfile` tells us where Python and host-side time is spent.
- JAX traces tell us where time is spent in compiled dispatches, kernels, and
  host/device synchronization.
- memory profiles tell us whether runtime problems are actually memory-pressure
  problems.
- the live Hermès rerun matrix tells us whether a faster case is still solving
  the right problem.

For `jax_drb`, those have to be read together. A faster run that worsens the
Hermès compare surface is not an improvement.
