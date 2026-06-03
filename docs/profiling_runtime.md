# Profiling And Runtime Workflow

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
  - fresh live Hermès rerun ratio is about `3.18x`
  - the dominant mismatch remains the boundary-localized `NVh` field
- `recycling_dthe_one_step`
  - timed local mean dropped from about `75.3 s` to about `54.1 s` after the
    reaction/source allocation cleanup, and then to about `52.76 s` after
    caching target-boundary geometry in the recycling runtime model
  - the latest post-metric-selector one-run cProfile pass on this machine
    measured `74.39 s`; the separate RSS run measured `49.25 s` with peak
    process-tree RSS about `231.2 MiB`
  - fresh live Hermès rerun ratio is about `7.82x`
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

The refreshed full cProfile/RSS bundle after adding RHS phase counters to the
production BDF callback was:

- command: `profile_curated_case.py recycling_dthe_one_step --warm-runs 0 --timed-runs 1 --cprofile-top 50 --rss-profile`
- cProfile run: `68.64 s` wall, `1.67e8` Python function calls
- separate unprofiled RSS run: `48.82 s`
- peak sampled process-tree RSS: `228.9 MiB`
- BDF callback counts visible in the profile: `11838` packed RHS evaluations,
  `86` Jacobian callbacks, and `8428` finite-difference color-group
  perturbation residuals
- unprofiled BDF phase counters: `48.77 s` solve time, `46.41 s` fixed-layout
  RHS object evaluation time, `33.60 s` Jacobian callback time, and only about
  `2e-3 s` in RHS NumPy conversion
- top cumulative costs: sparse finite-difference Jacobian construction
  `47.0 s`, packed RHS `64.5 s`, species RHS assembly `62.2 s`, reaction
  sources `10.3 s`, fixed-layout D/T/He reaction sources `9.1 s`, collision
  closure `8.7 s`, open-field state preparation `7.2 s`, and the remaining
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

The current GPU evidence for the heavier fixed-layout seam lives in:

- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_gpu/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_gpu_warm/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_ny100_dt1e4_cpu/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_ny100_dt1e4_gpu/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_ny200_dt1e4_cpu/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_jax_linearized_gate_ny200_dt1e4_gpu/profile_summary.json`
- `docs/data/runtime_profile_artifacts/recycling_dthe_batched_jvp_gate_cpu/profile_summary.json`

Those summaries show equal residual closure between CPU and GPU, lower sampled
peak RSS on GPU for the small D/T/He fixed-layout gate, and slower warm GPU
wall time on the retained problem size. The correct profiling conclusion is
that the residual seam is accelerator-executable, while the reviewer-facing
speedup claim still requires a larger transformed residual or a batched heavy
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

The batched residual/JVP gate is the current fixed-layout differentiability
and parallel-throughput test:

```bash
PYTHONPATH=src python scripts/profile_recycling_batched_jvp_gate.py \
  --reference-root /path/to/reference/root \
  --case dthe \
  --override mesh:ny=100 \
  --batch-sizes 1,4,16,64 \
  --timed-runs 7 \
  --output-dir docs/data/runtime_profile_artifacts/recycling_dthe_batched_jvp_gate_cpu
```

The retained local CPU artifact now sweeps batches through 256 states and
shows about `2.8x` residual throughput speedup and `2.2x` JVP throughput
speedup over serial same-kernel calls, with batched/serial residual and JVP
mismatch at roundoff. The residual JVP agrees with centered finite difference
to about `6e-9`. A remote GPU run on the same heavy residual was not retained
as a release speedup because the JVP compile latency was still too high for a
bounded validation gate.

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
trace/memory run. The `gpu-dthe-batched-jvp-gate` command is the multi-device
batched residual/JVP throughput run. Neither command promotes the full
output-window BDF solve as GPU-accelerated; they are evidence-gathering gates
for the residual and derivative kernels that must become production-safe first.
The wrapper intentionally rejects reference roots that do not contain
`tests/integrated/1D-recycling-dthe/data/BOUT.inp`; that failure means the
reference prerequisite is missing, not that the GPU gate has failed.

If a self-hosted machine has only a staged D/T/He `BOUT.inp` outside that
reference-root layout, use the direct profiler until a full reference root is
installed. This is the safest reduced multi-GPU readiness command:

```bash
INPUT_PATH=/path/to/1D-recycling-dthe/data/BOUT.inp

JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=src python scripts/profile_recycling_batched_jvp_gate.py \
  --input-path "$INPUT_PATH" \
  --case dthe \
  --override mesh:ny=100 \
  --batch-sizes 2,4,8,16 \
  --timed-runs 3 \
  --skip-objective-grad-check \
  --jax-trace \
  --device-memory-profile \
  --compilation-cache-dir tmp/jax_cache/recycling_dthe_batched_jvp_gate_gpu_readiness \
  --output-dir tmp/profiles/recycling_dthe_batched_jvp_gate_gpu_readiness
```

The current GPU speedup evidence instead comes from the source-term throughput
gate:

```bash
PYTHONPATH=src python scripts/profile_atomic_rate_throughput_gate.py \
  --output-dir docs/data/runtime_profile_artifacts/atomic_rate_throughput_gate_cpu
```

The matched office-GPU artifact lives in
`docs/data/runtime_profile_artifacts/atomic_rate_throughput_gate_gpu/profile_summary.json`.
At `4,194,304` temperature points the GPU is about `2.5x` faster than the
local CPU for the batched rate surface and about `2.1x` faster for the
autodiff derivative. The same report checks a scalar mean-rate sensitivity to
a log-temperature shift; autodiff and centered finite difference agree at
about `1e-10` relative error on CPU and GPU. This is the correct release
claim: dense JAX-native source kernels accelerate on GPU today; full heavy
recycling output-window GPU speedup is still blocked by host/SciPy residual
structure.

The source-throughput profiler also has an opt-in `--enable-pmap` flag. It is
not enabled in the committed office-GPU artifact, so that artifact remains a
single-device GPU result. A 2026-06-02 self-hosted smoke check did pass a basic
two-device `pmap` operation, but no real source-kernel or recycling-kernel
multi-device artifact has been regenerated. Multi-device source speedup should
therefore not be claimed until the device-level real-kernel parity gate passes
and the matching summary is committed.

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
