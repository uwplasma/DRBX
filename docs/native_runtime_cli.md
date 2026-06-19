# Native Runtime CLI

`jax_drb` now has a standalone native runtime surface for supported inputs. After an editable install,

```bash
pip install -e .
```

both of these console commands are available:

```bash
jax_drb path/to/input.toml
jax-drb path/to/input.toml
```

The bare input-file form is equivalent to:

```bash
jax_drb run path/to/input.toml
```

## Input Layout

The native CLI accepts organized TOML decks and also keeps compatibility with legacy `.inp` decks used by the curated benchmark harness. The intended TOML layout is:

- `[time]`
- `[runtime]`
- `[runtime.logging]`
- `[mesh]`
- `[solver]`
- `[model]`
- `[output]`
- `[restart]`
- `[species.<name>]`
- `[fields.<name>]`

Expression-valued entries can be written explicitly as wrappers:

```toml
[mesh]
dx = { expr = "0.0075 + 0.005*x" }

[fields.Nh]
function = { expr = "1 + H(x - 0.25) * H(0.75-x)" }
```

List-valued component/type entries use standard TOML arrays:

```toml
[model]
components = ["h"]

[species.h]
type = ["evolve_density", "evolve_pressure", "anomalous_diffusion"]
```

## Precision Selection

Precision can be chosen in the input file:

```toml
[runtime]
precision = "float64"

[runtime.logging]
verbosity = "detailed"
verbose = true
quiet = false

[output]
directory = "output/my_case"
write_summary = true
write_arrays = true
write_restart = true
write_log = true
```

or overridden at the terminal:

```bash
jax_drb input.toml --precision float32
```

The terminal logging mode can also be controlled directly:

```bash
jax_drb input.toml --verbose
```

The logging rules are:

- `[runtime.logging].verbose = true` means detailed staged event output
- `[runtime.logging].verbose = false` means the concise summary path
- `[runtime.logging].verbosity = "summary"` or `"detailed"` pins the level explicitly
- `[runtime.logging].quiet = true` suppresses terminal output entirely
- `--verbose` overrides the deck for a one-off detailed run

For the recycling transient lanes, the same deck can now pin the native one-step transient solver explicitly:

```toml
[runtime]
recycling_transient_solver_mode = "adaptive_bdf"
```

Allowed values are:

- `continuation`
- `bdf`
- `bdf_fixed_full_field_jvp`
- `bdf_active_array_jvp`
- `fixed_bdf2_jax_linearized`
- `fixed_bdf2_jax_linearized_lineax`
- `fixed_bdf2_active_array_jax_linearized`
- `fixed_bdf2_active_array_jax_linearized_lineax`
- `adaptive_be`
- `adaptive_bdf`
- `adaptive_bdf_sparse_jvp`
- `adaptive_bdf_jax_linearized`
- `adaptive_bdf_jax_linearized_lineax`
- `adaptive_bdf_active_array_jax_linearized`
- `adaptive_bdf_active_array_jax_linearized_lineax`

That switch is meant for controlled solver sweeps on the open-field/tokamak recycling one-step lanes. The `fixed_bdf2_jax_linearized` modes are opt-in promotion lanes that avoid the SciPy full-output BDF callback: they use a fixed-layout backward-Euler startup step and fixed-layout BDF2 steps with controller integrals packed into the residual state. The curated runner still keeps its default case-specific solver choice unless the deck asks for a different mode.

The JAX-linearized adaptive-BDF modes also support an explicit rejected-history
policy:

```toml
[runtime]
recycling_adaptive_bdf_reuse_rejected_history = true
```

The same setting is available through
`JAX_DRB_RECYCLING_ADAPTIVE_BDF_REUSE_REJECTED_HISTORY`. It defaults to
`true` only for JAX-linearized adaptive-BDF modes and to `false` for the legacy
sparse compatibility mode. When enabled, a rejected BDF2 trial halves the next
trial timestep but keeps the last valid accepted-step history instead of
forcing a startup backward-Euler trial. This reduces restart overhead without
changing the embedded-error acceptance criterion.

The same modes can also reuse the just-computed backward-Euler predictor as the
initial state for the BDF2 corrector:

```toml
[runtime]
recycling_bdf2_use_be_initial_guess = true
```

The matching environment variable is
`JAX_DRB_RECYCLING_BDF2_USE_BE_INITIAL_GUESS`. The default is `true` for
JAX-linearized adaptive-BDF modes and `false` for the sparse compatibility
mode. This is a convergence-path heuristic only: it does not alter the
backward-Euler or BDF2 residual equations, and current promotion gates still
show that the JAX-linearized output-window route is dominated by inner Krylov
linear solves.

The inner JAX Krylov tolerance can be swept independently from the outer
nonlinear residual gate:

```toml
[runtime]
recycling_jax_linear_tolerance_factor = 10
```

The matching environment variable is
`JAX_DRB_RECYCLING_JAX_LINEAR_TOLERANCE_FACTOR`. The effective Krylov
tolerance is `solver.rtol * recycling_jax_linear_tolerance_factor`; the default
factor is `1`, preserving the historical behavior. This knob is for controlled
promotion studies only. The adaptive-BDF residual, embedded-error, fallback,
and linear-solver-health gates still determine whether a run is acceptable.

The JAX-linearized Newton line search can also start from a damped step when a
validated gate repeatedly rejects the full Newton step:

```toml
[runtime]
recycling_jax_linear_line_search_initial_step_scale = 0.25
```

The matching environment variable is
`JAX_DRB_RECYCLING_JAX_LINEAR_LINE_SEARCH_INITIAL_STEP_SCALE`. Values are
bounded to `(0, 1]`, and the default is `1`, preserving the standard backtracking
search. This control is not a physics-model change; it only avoids predictable
rejected residual evaluations in quality-gated JAX-linearized solver studies.
The line-search policy itself can be selected with
`runtime:recycling_jax_linear_line_search_mode=backtracking` or `full_step`, or
with `JAX_DRB_RECYCLING_JAX_LINEAR_LINE_SEARCH_MODE`. The default
`backtracking` evaluates a trial residual before accepting each Newton update.
The opt-in `full_step` path accepts finite full updates and lets the next
`jax.linearize` call perform the residual check. Fixed-BDF2 histories report
`fixed_bdf2_line_search_mode`, and the comparison harness can require it with
`--require-fixed-bdf2-line-search-mode`. Current bounded gates keep
`backtracking` as the default because `full_step` traded fewer standalone
residual evaluations for more linearizations and did not improve wall time.
The same diagnostics report `linear_operator_call_count` and
`linear_operator_dispatch_seconds`, which measure Python-visible calls to the
matrix-free linearized operator during the Krylov solve. These are profiling
diagnostics for solver studies; the full `linear_solve_seconds` value remains
the end-to-end wall-time measurement because JAX device work can be asynchronous.
When a JAX-GMRES preconditioner is enabled, the same reports include
`linear_preconditioner_build_count`, `linear_preconditioner_build_seconds`,
`linear_preconditioner_apply_count`, and
`linear_preconditioner_apply_seconds`; fixed-BDF2 and adaptive-BDF histories
aggregate those as `fixed_bdf2_total_*` and `adaptive_bdf_*` fields.
Current opt-in dynamic choices include diagonal probes, sampled field blocks
(`runtime:recycling_jax_linear_preconditioner=field_block_sample` or
`field_split`), same-cell blocks, and selected parallel-line blocks. The sampled
field-block probe is bounded by
`runtime:recycling_jax_linear_preconditioner_max_field_block_fields`.
The profiling and comparison scripts can enforce these diagnostics with
`--require-max-preconditioner-applies` and
`--require-fixed-bdf2-max-preconditioner-applies` when screening candidate
transport preconditioners. The promotion wrapper also accepts
`--fixed-bdf2-only` for these screens. That option runs only the bounded
fixed-BDF2 JAX-linearized phase, so preconditioner or matrix-free Krylov
experiments are not blocked by the separate SciPy-BDF JVP bridge parity gate.
For heavy matrix-free profiling, the linearized Krylov action can also be
wrapped with `runtime:recycling_jax_linear_jit_linear_operator=true` or
`JAX_DRB_RECYCLING_JAX_LINEAR_JIT_LINEAR_OPERATOR=1`. Fixed-BDF2 histories
report `fixed_bdf2_linear_operator_jitted_steps`, and the comparison harness can
require this route with `--require-fixed-bdf2-linear-operator-jitted`. This is
an opt-in compiler experiment, not a default, because bounded local gates have
not shown a robust speedup.

Fixed-output BDF2 JAX-linearized histories use
`runtime:recycling_jax_linear_initial_residual_mode=linearize` by default. This
keeps the initial convergence check, but gets the first residual norm from the
first `jax.linearize` call instead of evaluating a separate standalone residual
that is immediately followed by linearization. Direct one-step and adaptive
JAX-linearized modes keep the older `residual` default; users can force either
behavior with `runtime:recycling_jax_linear_initial_residual_mode=residual` or
`linearize`, or with
`JAX_DRB_RECYCLING_JAX_LINEAR_INITIAL_RESIDUAL_MODE`.

Fixed-output BDF2 histories also report the initial-guess policy used for BDF2
corrector steps. The default policy is `rhs_predictor`, set with
`runtime:recycling_fixed_bdf2_initial_guess_policy=rhs_predictor`, which builds
the Newton seed from the explicit RHS predictor inside the BDF2 residual context.
The opt-in `history_extrapolation` policy uses the two previous accepted states
to seed \(u^{n+1}\approx u^n+\Delta t_n(u^n-u^{n-1})/\Delta t_{n-1}\), then
applies the same density/pressure and feedback-integral sanitizers as the
production path. The matching environment variable is
`JAX_DRB_RECYCLING_FIXED_BDF2_INITIAL_GUESS_POLICY`. This is a solver-seeding
control only; it does not change the BDF2 residual equation. Local bounded gates
currently keep `rhs_predictor` as the default because history extrapolation did
not reduce Krylov work on the active-array recycling fixture.

For JAX-linearized promotion experiments, the inner matrix-free Krylov backend
can be selected with `runtime:recycling_jax_linear_solver=<backend>` or
`JAX_DRB_RECYCLING_JAX_LINEAR_SOLVER`. Supported values are `jax_gmres`
(default), `lineax_gmres` when the optional Lineax package is installed, and
`jax_bicgstab`. The fixed-BDF2 comparison gate can require the selected backend
with `--require-fixed-bdf2-linear-solver-backend`. The BiCGSTAB path is
diagnostic: local JAX does not report an inner success flag for that solver, so
adaptive-BDF diagnostics count those steps under
`adaptive_bdf_unknown_linear_solver_steps`.

Current status:

- `float64` is the default and the most complete runtime mode.
- `float32` now runs cleanly on the simple diffusion/restart tutorial path and no longer emits the old internal dtype-truncation warnings there.
- broader native paths still contain explicit `float64` requests internally, so `float32` should currently be treated as an opt-in performance experiment, not a parity-default production mode.

For a concrete measurement workflow, use:

```bash
PYTHONPATH=src .venv/bin/python examples/diffusion_precision_benchmark.py
```

The committed example benchmark artifacts are in:

- [docs/runtime_precision_benchmark/data/diffusion_precision_analysis.json](docs/runtime_precision_benchmark/data/diffusion_precision_analysis.json)
- [https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__runtime_precision_benchmark__images__diffusion_precision_elapsed.png](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__runtime_precision_benchmark__images__diffusion_precision_elapsed.png)

On the current machine, the warm second-run `float32` diffusion path is about `1.23x` faster than `float64` (`2.096s` vs `2.584s`) on the same input.

## Runtime Output

A native run can write four main artifact types:

- summary JSON
- arrays NPZ
- restart NPZ
- verbose run-log JSON

Each emitted payload is also expected to carry the current capability-tier label for that run:

- `native_exact`
- `native_operational`
- `scaffolded_reference_backed`

For direct deck-driven native runs, the current default is `native_exact` unless a curated benchmark case explicitly supplies a different tier.

Example:

```bash
jax_drb examples/inputs/restartable_diffusion.toml \
  --precision float32
```

If the deck includes an `[output]` section, the bare `jax_drb input.toml` form is enough. CLI flags still override the deck when you need an ad hoc run location. A typical deck-controlled output block is:

```toml
[output]
directory = "output/restartable_diffusion"
write_summary = true
write_arrays = true
write_restart = true
write_log = true
```

This writes:

- `<output-dir>/<case>_summary.json`
- `<output-dir>/<case>_arrays.npz`
- `<output-dir>/<case>_restart.npz`
- `<output-dir>/<case>_run_log.json`

The terminal output is rich-formatted when `rich` is available and falls back to plain text otherwise. It now has two layers:

- event-style run messages while the simulation is being configured, restarted, launched, and written out
- the final run summary table

For Python driver scripts, the same native entry point now exposes a matching verbose switch:

```python
from jax_drb.native import run_input_case

result = run_input_case(
    "examples/inputs/restartable_diffusion.toml",
    case_name="diffusion_driver",
    parity_mode="run",
    verbose=True,
)
```

`verbose=True` emits the same staged event stream through the native runner, and `event_logger=` can be supplied if a script wants to capture those events instead of printing them.

The same deck-level `recycling_transient_solver_mode` override is honored by the native runner when a Python driver script loads the deck through `run_input_case(...)`.

Both versions report the same core metadata:

- input file
- case name
- runtime precision
- runtime backend/device/cache
- runtime library and machine metadata (`jax_version`, `python_version`, platform, process id)
- time/mesh/solver settings
- scheduled components
- compare variables
- capability tier
- restart provenance
- output artifact paths
- variable min/max/mean/delta summaries

The verbose run-log JSON now also stores the ordered event stream, so a downstream plotting or workflow script can reconstruct what happened during the run.
The same JSON also stores sanitized working-directory and machine/runtime metadata so a saved run can be audited later without leaking workstation-specific absolute paths.
It now also carries `event_count` and `event_stages`, and the native recycling lanes emit interval-level `progress` events so long implicit steps do not appear idle in the CLI. On the live native backward-Euler and adaptive BDF/BE paths those events now also include interval counts, simulated time, accepted timestep, elapsed wall time, and an estimated remaining wall time.

In practice, the detailed runtime stream now covers:

- configuration loading
- restart loading
- native run launch/completion
- recycling transient interval progress
- artifact destination resolution
- per-artifact write completion
- final run summary

## Restart / Resume

To resume from a saved restart bundle from the CLI:

```bash
jax_drb input.toml \
  --output-dir /tmp/jax_drb_run_resume \
  --restart-in /tmp/jax_drb_run/<case>_restart.npz \
  --resume-steps 2
```

The same workflow can also be encoded in the deck:

```toml
[restart]
input = "output/restartable_diffusion/restartable_diffusion_restart.npz"
resume_steps = 2
```

The runnable tutorial for the full flow is:

- [examples/restartable_diffusion_tutorial.py](examples/restartable_diffusion_tutorial.py)

And the simplest shipped example deck is:

- [examples/inputs/restartable_diffusion.toml](examples/inputs/restartable_diffusion.toml)
