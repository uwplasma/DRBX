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
  --species h
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
and adding a gated JSONL monitor in the reference application. The monitor
registers its three input options during initialization so BOUT++ input
validation does not reject trace-enabled decks before the first accepted step.
Both reference patches are line-numbered `git apply` patches. Apply the direct
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

[hermes]
neutral_mixed_accepted_step_trace = true
neutral_mixed_accepted_step_trace_file = /tmp/ref_trace/accepted_steps.jsonl
neutral_mixed_accepted_step_trace_species = h
```

3. On rank zero, append one JSON object per accepted internal step. Each record
   must contain `time`, `dt`, a monotone `step_index`, and a `stages`
   dictionary with a `post_accepted` payload. The JAXDRB runner validates the
   following field set before returning successfully:

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
`Vh` and `eta_h` diagnostic-input fields. These extra fields are not part of
the required reference schema, but they match the native accepted-step trace
payloads added to split parallel-viscosity drift into input and stencil pieces.
The patch reads `Vh` from the existing velocity diagnostic and exposes `eta_h`
from the neutral viscosity field before the monitor checks for optional fields.

Each field payload should follow the same compact shape used by JAXDRB native
accepted-step traces:

```json
{
  "active_metrics": {"max_abs": 0.0, "rms": 0.0},
  "target_adjacent_metrics": {"max_abs": 0.0, "rms": 0.0},
  "guard_metrics": {"max_abs": 0.0, "rms": 0.0},
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

## Current Evidence

On June 5, 2026, the two reference patches were applied to a clean disposable
reference checkout at commit `f7bab630`, built successfully with the local
`hermes-3` target, and produced a valid `neutral_mixed_one_step` JSONL trace
with `148` accepted CVODE records. Native accepted-step traces now emit the
same 10 fields as the reference trace and can replay the reference accepted
time grid. A local matched-grid comparison now matches `148/148` accepted
points. With the timestamp mismatch removed, the leading active/target
offender is `SNVh_parallel_viscosity` at about `5.35e-5`; `ddt(Nh)`,
state-level `Nh`, and state-level `NVh` follow at smaller target-adjacent
scales. RHS/source guard metrics are still large and should be treated as
diagnostic-boundary semantics until a guard-specific reference definition is
chosen. The next implementation step is therefore to fix or further localize
the neutral-mixed parallel-viscosity/boundary sequencing path under the
matched-time accepted-step diagnostic before changing broader BDF sequencing.
Native traces now also emit optional `Vh` and `eta_h` diagnostic-input fields.
The reference monitor patch now writes the same payloads when those diagnostics
are exposed by `outputVars()`, so the remaining parallel-viscosity difference
can be split into velocity/viscosity input drift and the
`Div_par_K_Grad_par_mod(eta_h, Vh, false)` stencil itself. The comparator
summarizes this split in `parallel_viscosity_input_register`: missing `Vh` or
`eta_h` marks the trace as insufficient for source-input diagnosis, while
present input fields quantify whether the leading `SNVh_parallel_viscosity`
offender is driven by `Vh`/`eta_h` drift or by the parallel-diffusion stencil
and boundary treatment.

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
  --hermes-binary /path/to/hermes-3 \
  --workdir /tmp/ref_trace \
  --trace-out /tmp/ref_trace/accepted_steps.jsonl \
  --timeout-seconds 180 \
  --species h

PYTHONPATH=src jax-drb compare-neutral-mixed-accepted-traces \
  /tmp/native_trace.json \
  /tmp/ref_trace/accepted_steps.jsonl \
  --json-out /tmp/neutral_trace_parity.json \
  --time-tolerance 1e-7
```

This diagnostic is the next required evidence before changing neutral-mixed
boundary sequencing or the `NVh` pressure-gradient/viscosity implementation.
