# Neutral-Mixed Accepted-Step Reference Trace

This note records the reference-side diagnostic needed to close the remaining
neutral-mixed `NVh` accepted-step parity lane. It is not a JAXDRB production
dependency; it is a reproducible audit hook for generating the JSONL consumed
by:

```bash
PYTHONPATH=src jax-drb trace-neutral-mixed-reference-accepted-steps \
  --reference-root /path/to/reference-root \
  --hermes-binary /path/to/hermes-3 \
  --workdir /tmp/ref_trace \
  --trace-out /tmp/ref_trace/accepted_steps.jsonl \
  --species h \
  --cvode-max-order 2
```

The direct source-term patch in
[hermes_neutral_mixed_pressure_gradient_diagnostic.patch](hermes_neutral_mixed_pressure_gradient_diagnostic.patch)
must be present first, so the reference output can write
`SNVh_pressure_gradient`, `SNVh_parallel_viscosity`, and
`SNVh_perpendicular_viscosity` in addition to `ddt(NVh)`.

The accepted-step monitor itself is captured as a reproducible patch artifact:
[hermes_neutral_mixed_accepted_step_trace_monitor.patch](hermes_neutral_mixed_accepted_step_trace_monitor.patch).
Apply it to a clean, disposable reference-code checkout rather than to a
working tree with unrelated local edits. The patch has two responsibilities:
refreshing the CVODE accepted-step state and RHS before timestep monitors run,
exporting the CVODE method order used by each accepted internal step, and
adding a gated JSONL monitor in the reference application. The monitor
registers its input options during initialization so BOUT++ input validation
does not reject trace-enabled decks before the first accepted step. Both
reference patches are line-numbered `git apply` patches. Apply the direct
source-term diagnostic patch first, then the accepted-step monitor patch.

## Required Reference Changes

1. In the CVODE accepted-step loop, refresh the state and RHS before timestep
   monitors run. The target location is the `CV_ONE_STEP` branch in
   `external/BOUT-dev/src/solver/impls/cvode/cvode.cxx`, near the existing
   `call_timestep_monitors(internal_time, internal_time - last_time)` call.
   The monitor should see the accepted state and the corresponding RHS, not the
   previous output state.

```cpp
load_vars(N_VGetArrayPointer(uvec));
run_rhs(internal_time);
call_timestep_monitors(internal_time, internal_time - last_time);
```

2. Add a gated `timestepMonitor(BoutReal simtime, BoutReal dt)` implementation
   in the reference application. The monitor should be enabled only when the
   deck contains:

```ini
[solver]
monitor_timestep = true
cvode_max_order = 2

[hermes]
neutral_mixed_accepted_step_trace = true
neutral_mixed_accepted_step_trace_file = /tmp/ref_trace/accepted_steps.jsonl
neutral_mixed_accepted_step_trace_species = h
```

3. On rank zero, append one JSON object per accepted internal step. Each record
   must contain `time`, `dt`, a monotone `step_index`, a `solver.order` value
   from `CVodeGetLastOrder`, and a `stages` dictionary with a `post_accepted`
   payload. The JAXDRB runner validates the following field set before
   returning successfully:

```text
Nh
Ph
NVh
ddt(Nh)
ddt(Ph)
ddt(NVh)
SNVh
SNVh_pressure_gradient
SNVh_parallel_viscosity
SNVh_perpendicular_viscosity
```

When the reference `outputVars()` exposes them, the monitor also writes optional
`Dnnh`, `Vh`, and `eta_h` diagnostic-input fields. The accepted-step monitor
patch now extends the reference neutral-mixed component with the same
diagnostic ladder that the native trace writes for the neutral diffusion
coefficient: `Tnlimh`, `logPnlimh`, `grad_logPnlimh`, `Dnnh_raw`,
`Dnnh_flux_max`, `Dnnh_flux_limited`, and `Dnnh_diffusion_limited`. These
fields are still optional rather than part of the required reference schema,
but a patched reference rerun can now split the accepted-step `Dnnh` drift into
temperature flooring, pressure-gradient magnitude, flux limiting, diffusion
limiting, and final boundary application. The patch reads `Dnnh` and `Vh` from
diagnostics, exposes `eta_h` from the neutral viscosity field, and exposes the
pre-boundary diffusion-preparation ladder before the monitor checks for
optional fields.

Each field payload should follow the same compact shape used by JAXDRB native
accepted-step traces:

```json
{
  "active_metrics": {
    "max_abs": 0.0,
    "rms": 0.0,
    "max_abs_index": [0, 0, 0],
    "max_abs_value": 0.0
  },
  "target_adjacent_metrics": {
    "max_abs": 0.0,
    "rms": 0.0,
    "max_abs_index": [0, 0, 0],
    "max_abs_value": 0.0
  },
  "guard_metrics": {
    "max_abs": 0.0,
    "rms": 0.0,
    "max_abs_index": [0, 0, 0],
    "max_abs_value": 0.0
  },
  "target_adjacent_shape": [1, 4, 1],
  "target_adjacent_values": [0.0, 0.0, 0.0, 0.0],
  "guard_shape": [1, 4, 1],
  "guard_values": [0.0, 0.0, 0.0, 0.0],
  "sample_lineout_y_indices": [0, 1],
  "sample_lineout": [0.0, 0.0]
}
```

Use the same index convention as the JAXDRB comparator: active cells are
`xstart:xend` and `ystart:yend`; target-adjacent y cells are `ystart`,
`ystart + 1`, `yend - 1`, and `yend`; guard y cells are `ystart - 2`,
`ystart - 1`, `yend + 1`, and `yend + 2`; the lineout uses the mid active x
index, the mid local z index, and the sorted union of target-adjacent and guard
y indices. `Nh`, `Ph`, and `NVh` should be read from the live species state at
the accepted internal step; `ddt(*)` and `SNVh_*` should be read from the
diagnostic output state after `output_ddt=true` and `diagnose=true`.
The flattened target-adjacent payload is intentionally compact but pointwise:
the comparator reshapes it using `target_adjacent_shape` and reports the worst
native/reference target-cell delta. This keeps legacy max/rms zone summaries
available while avoiding false offender ranking when the largest target-band
value occurs at a different symmetric cell. The flattened guard payload uses
the same convention for the guard band. It is primarily forensic: final
boundary-applied fields should have meaningful guard comparisons, while
pre-boundary limiter ladder diagnostics should be ranked by active and
target-adjacent cells because the reference snapshots are taken before the
final `Dnn.applyBoundary()` call.

## Current Evidence

On June 5, 2026, the two reference patches were applied to a clean disposable
reference checkout at commit `f7bab630`, built successfully with the local
`hermes-3` target, and produced a valid `neutral_mixed_one_step` JSONL trace
with `148` accepted CVODE records. The same clean auto-build path was rerun
with the optional diagnostic-input fields enabled; the produced JSONL includes
`Dnnh`, `Vh`, and `eta_h` alongside the required state, RHS, and `SNVh_*`
source fields.
Native accepted-step traces emit the same 10 required fields as the reference
trace and can replay the reference accepted time grid. A local matched-grid
comparison matches `148/148` accepted points. With timestamp mismatch removed,
the comparator can now separate state inputs, closure inputs, and source terms.
The `parallel_viscosity_input_register` shows final `Dnnh` as the dominant
closure-input drift (`4.46e-3` target-adjacent), followed by `eta_h`
(`3.23e-3`), while `SNVh_parallel_viscosity` is `1.29e-4` pointwise
(`5.35e-5` by the legacy zone-summary active/target comparison).

Native and reference traces now also emit the `Dnnh` preparation ladder
(`Tnlimh`, `logPnlimh`, `grad_logPnlimh`, `Dnnh_raw`, `Dnnh_flux_max`,
`Dnnh_flux_limited`, and `Dnnh_diffusion_limited`). The contextual reference
patch uses BOUT++ `copy(Dnn)` snapshots for the pre-limiter and post-limiter
diffusion fields; ordinary `Field3D` assignment shares field storage and would
turn the raw-diffusion diagnostic into a view of the later limited field. A live
patched-reference rerun with this deep-copy patch produced `148` accepted-step
records with no missing ladder fields. The matched native/reference comparison
identifies `Dnnh_flux_max` as the dominant target-band ladder field (`5.27e-3`),
followed by the flux-limited, diffusion-limited, and final boundary-applied
`Dnnh` fields (`4.46e-3`). The raw diffusion mismatch is much smaller
(`6.07e-4`), so the remaining accepted-step offender is in the flux-limit cap
and near-target state/boundary sequencing rather than raw neutral diffusion
preparation.
On the controlled max-order-2 rerun, the same interpretation holds: the native
replay matches `309/309` accepted times with zero solver-order mismatches, and
the ladder-transition register ranks `Dnnh_raw -> Dnnh_flux_max` first, with
target-pointwise error rising from about `2.83e-4` to about `5.13e-3`.

The latest pointwise target-cell and guard-cell rerun confirms that this is not
only a zone-maximum ordering artifact. At the upper target-adjacent cell with
local target index `[0, 3, 0]`, native `Dnnh_flux_max` is `2.74471293` and the
reference value is `2.73944`, a `5.27e-3` drift. The same cell has essentially
closed temperature and raw diffusion, but native `grad_logPnlimh` is
`0.0130723` versus the reference value `0.0131171`. The new guard payload also
shows that `grad_logPnlimh` and intermediate limiter fields carry large
pre-boundary guard deltas, while final boundary-applied `Dnnh` has a guard
pointwise drift equal to its target pointwise drift (`4.46e-3`). The next
native parity patch should therefore target the accepted-state history feeding
the near-target `Grad(logPnlim)` stencil before changing collision rates or raw
neutral diffusion formulas. A direct algebraic check at the worst
`Dnnh_flux_max` point closes the flux-cap formula itself: both native and
reference records satisfy
`Dmax = flux_limit sqrt(Tnlim/AA)/(grad_logPnlim + 1/lmax)` when `lmax` is
inferred from the local raw diffusion. The input drift at that cell is already
visible in the state: native `Ph` is larger than the reference by
`6.69e-6`, native `Nh` by `6.50e-5`, and native `logPnlimh` by `9.36e-5`.
That points to accepted-step state-history sequencing, not a missing term in
the diffusion cap.

The parity report now records the same conclusion as structured data. Each
`neutral_diffusion_ladder_register` entry includes the state-input fields
(`N*`, `P*`, `NV*`), limiter-input fields (`Tnlim*`, `logPnlim*`,
`grad_logPnlim*`), the dominant state and limiter input offenders, and the
target-pointwise amplification ratios from state inputs to limiter inputs and
from limiter inputs to `Dnn*_flux_max`. This is the next hard gate before a
model patch: if a rerun shows the limiter-input drift is already present in the
accepted state, patch accepted-state/history sequencing; if the state fields
close but `grad_logPnlim*` remains open, patch the near-target gradient stencil
or boundary preparation.

A final-state input-closure cross-check reconstructs `Dnn`, `Vh`, and `eta_h`
from the reference final-state `Nh`, `Ph`, and `NVh` fields and compares those
arrays with the reference `BOUT.dmp.0.nc` diagnostics. This closes the neutral
diffusion, velocity, and viscosity input formulas to roundoff on the current
reference final state, including target-adjacent and guard cells.

## Validation Sequence

After rebuilding the reference executable, generate and compare traces:

```bash
PYTHONPATH=src jax-drb trace-neutral-mixed-accepted-steps \
  --reference-root /path/to/reference-root \
  --case-name neutral_mixed_one_step \
  --internal-substeps 8 \
  --json-out /tmp/native_trace.json

PYTHONPATH=src jax-drb trace-neutral-mixed-reference-accepted-steps \
  --reference-root /path/to/reference-root \
  --workdir /tmp/ref_trace \
  --trace-out /tmp/ref_trace/accepted_steps.jsonl \
  --timeout-seconds 180 \
  --species h \
  --cvode-max-order 2

PYTHONPATH=src jax-drb compare-neutral-mixed-accepted-traces \
  /tmp/native_trace.json \
  /tmp/ref_trace/accepted_steps.jsonl \
  --reference-cvode-max-order 2 \
  --json-out /tmp/neutral_trace_parity.json \
  --time-tolerance 1e-7
```

This diagnostic is the next required evidence before changing neutral-mixed
boundary sequencing or the `NVh` pressure-gradient/viscosity implementation.
When `--hermes-binary` is omitted, JAXDRB prepares a cached clean reference
worktree under the system temporary directory, applies both reference patches,
builds `hermes-3`, and runs that patched binary. Passing `--hermes-binary`
keeps the old explicit-binary behavior for already patched developer builds.

## Controlled Max-Order-2 Reference Lane

The native accepted-step replay is BDF2 after startup, while CVODE may choose a
higher method order. To isolate reference-backed parity from variable-order
history effects, generate a constrained reference trace with
`--cvode-max-order 2`. The runner writes `solver:cvode_max_order = 2` into the
staged `BOUT.inp` and then validates that no emitted `solver.order` exceeds
the configured ceiling. A violation fails the run instead of producing an
ambiguous trace.

Record the same control in the parity report:

```bash
PYTHONPATH=src jax-drb compare-neutral-mixed-accepted-traces \
  /tmp/native_trace.json \
  /tmp/ref_trace/accepted_steps.jsonl \
  --reference-cvode-max-order 2 \
  --json-out /tmp/neutral_trace_parity.json
```

The resulting JSON includes `native_solver_order_summary`,
`reference_solver_order_summary`, and `reference_solver_control`, including the
configured `cvode_max_order`, observed reference max order, and any ceiling
violations found in the trace.
