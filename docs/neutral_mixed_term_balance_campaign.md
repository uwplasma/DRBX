# Neutral Mixed Term-Balance Campaign

This campaign localizes the remaining `neutral_mixed_one_step` mismatch on
`NVh` by evaluating the native neutral momentum operator on both the native
final state and the Hermès-3 final state.

The diagnostic uses the backward-Euler residual-rate form

```text
R_NVh = (NVh_final - NVh_initial) / dt - RHS_NVh(NVh_final)
```

where `RHS_NVh` is decomposed into the named native terms returned by
`compute_neutral_mixed_rhs`: parallel inertia, pressure gradient,
perpendicular diffusion, parallel viscosity, and perpendicular viscosity. The
same initial state, mesh, metric normalization, and timestep are used for both
final states. A small residual when the native final state is inserted verifies
the native one-step balance. A larger residual when the Hermès-3 final state is
inserted identifies the operator and boundary terms that cannot reproduce the
Hermès update under the current native closure.

![Neutral mixed term-balance audit](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_term_balance_campaign_artifacts__images__neutral_mixed_term_balance_campaign.png)

Current artifact outputs:

- JSON summary: [neutral_mixed_term_balance_campaign.json](data/neutral_mixed_term_balance_campaign_artifacts/data/neutral_mixed_term_balance_campaign.json)
- Hermès-free substep/hybrid JSON: [neutral_mixed_substep_hybrid.json](data/neutral_mixed_substep_hybrid_artifacts/data/neutral_mixed_substep_hybrid.json)
- compact arrays: [neutral_mixed_term_balance_campaign.npz](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_term_balance_campaign_artifacts__data__neutral_mixed_term_balance_campaign.npz)
- figure: [neutral_mixed_term_balance_campaign.png](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_term_balance_campaign_artifacts__images__neutral_mixed_term_balance_campaign.png)

The generated report uses the physical active `x-y` domain for all term
metrics and direct-reference scaling, so guard-cell-only diagnostics do not
contaminate the offender ranking. The current code path adds connected-y guard
reconstruction for non-target neutral-mixed meshes and promotes the one-step
native default from four to eight internal BDF substeps. The tracked JSON report
and release-backed NPZ/PNG bundle have been regenerated from this code path
with the direct Hermès diagnostic NetCDF present. The active-domain final
`NVh` metric is now about `4.47e-6`, compared with the older `5.81e-4` and
`3.37e-3` reports. The same one-step audit keeps the active-domain final `Nh`
and `Ph` metrics near `2.19e-4` and `2.11e-5`, respectively. The short-window
path remains at four internal substeps because the prefix sweep shows that
eight substeps improve center momentum but do not solve the larger
total-density/pressure history drift. The remaining final-state drift is
therefore no longer a missing direct pressure-gradient or viscosity source
formula.

The report now carries a target-adjacent offender register rather than only
aggregate field errors. On the native-minus-Hermès final-state term deltas,
pressure gradient (`8.10e-6`) and parallel viscosity (`1.00e-5`) remain the
largest target-adjacent named differences because the native and Hermès final
states are not identical. Direct source-level diagnostics close the
implementation question:
after active-domain scaling, `SNVh_pressure_gradient`,
`SNVh_parallel_viscosity`, and `SNVh_perpendicular_viscosity` agree with the
matched JAXDRB reconstructions with max absolute differences of about
`2.17e-19`, `1.30e-18`, and `9.93e-23`, respectively.

The JSON report also carries `state_driver_register`, which turns that
interpretation into a regression target. Dividing the final-state differences
by the one-step interval ranks the target-adjacent state-rate errors as `Nh`
(`7.61e-6`), then `Ph` (`7.52e-7`), then `NVh` (`2.02e-7`). The induced
momentum-driver deltas are led by `NVh -> parallel_viscosity` (`1.00e-5`) and
`Ph -> pressure_gradient` (`8.10e-6`). The next implementation target is
therefore the
target-adjacent state history and boundary reconstruction that feeds those
closed operators, not a replacement of the pressure-gradient or viscosity
formula.

The companion `series_errors` payload now stores an `active_edge_history_trace`
for `Nh`, `Ph`, and `NVh`. This trace records the target-band maximum and RMS
history at the first two and last two active-y cells, which is the diagnostic
band used by the connected-y neutral-mixed parity case. It separates a true
initial-condition offset from drift introduced by the final internal substep
and gives future state-sequencing changes a small, Hermès-free regression
surface before rerunning the full live reference campaign.

The companion substep/hybrid diagnostic makes that conclusion reproducible
without a live Hermès rerun. It sweeps the native
`runtime:neutral_mixed_internal_substeps` setting against the committed
`neutral_mixed_one_step` arrays, records failed high-substep points instead of
hiding them, and then swaps one reference final field at a time (`Nh`, `Ph`,
or `NVh`) into the native final state before reevaluating the native momentum
balance. The hybrid ranking asks a precise question: which state variable most
reduces the target-adjacent pressure-gradient or viscosity term delta when it
is made reference-exact? That keeps the remaining parity work focused on the
target-band state/history sequencing rather than on already-closed formulas.
The committed diagnostic now records successful substep counts `1`, `2`, `4`,
and `8`, with `NVh` final max error decreasing from `8.84e-4` at one substep
to `4.47e-6` at eight substeps; the `3` and `6` substep probes are retained as
failed points so the report is explicit about the current controller limits.
Those failed points now include the nonlinear residual-vector size, finite
fraction, RMS, and maximum absolute value; in the committed report both
failures have 1800 finite residual entries with maximum absolute residuals of
about `0.85` and `0.88`.
This is the concrete reason the one-step path remains on the eight-substep
neutral mixed setting and why the next neutral parity work should target the
state/history sequencer rather than the direct pressure-gradient or viscosity
operators.

The next fidelity diagnostic should be an accepted-step trace from the
reference implementation before another native boundary patch is attempted.
JAXDRB now exposes the matching native-side hook through
`advance_neutral_mixed_implicit_history(..., store_internal_substeps=True)`,
which records accepted internal-step time, `dt`, BDF order, final accepted
`Nh`, `Ph`, and `NVh`, nonlinear iteration count, and residual norm without
changing the default history output. That hook is intentionally opt-in because
it is diagnostic evidence, not a different physical model.

Write the matching native trace artifact with:

```bash
PYTHONPATH=src jax-drb trace-neutral-mixed-accepted-steps \
  --case-name neutral_mixed_one_step \
  --internal-substeps 8 \
  --json-out neutral_mixed_native_accepted_step_trace.json
```

If a reference accepted-step JSONL is available, replay its adaptive accepted
time grid directly in the native trace:

```bash
PYTHONPATH=src jax-drb trace-neutral-mixed-accepted-steps \
  --input-path /path/to/neutral_mixed_one_step/BOUT.inp \
  --reference-trace-jsonl /tmp/neutral_mixed_reference_trace/accepted_steps.jsonl \
  --time-tolerance 1e-7 \
  --json-out neutral_mixed_native_reference_grid_trace.json
```

This diagnostic mode keeps the production fixed-substep path unchanged. It
uses backward Euler for the first positive accepted step and variable-step
BDF2 after startup, passing the previous accepted `dt` through the BDF2
residual. Adaptive reference monitors can finish slightly beyond the requested
output time; JAXDRB records both the requested target time and the final
accepted reference time in the native trace report.

For every accepted internal solver step from `t = 0` to the one-step output
time, the native trace writes `time`, `dt`, solver order, and post-accepted
`Nh`, `Ph`, and `NVh` values at the active target-adjacent cells and adjacent
guard cells. It also writes native diagnostic inputs `Vh` and `eta_h`, which
are the velocity and neutral viscosity entering the parallel-viscosity source.
The comparator ignores those fields until the reference JSONL contains the
same payloads, so the addition is backward-compatible with older reference
traces. When a reference executable with the accepted-step monitor patch is
available, write the matching reference JSONL with:

```bash
PYTHONPATH=src jax-drb trace-neutral-mixed-reference-accepted-steps \
  --reference-root /path/to/reference-root \
  --workdir /tmp/neutral_mixed_reference_trace \
  --trace-out /tmp/neutral_mixed_reference_trace/accepted_steps.jsonl
```

If `--hermes-binary` is not supplied, this command now builds a cached clean
patched reference worktree automatically before launching the trace run. That
keeps the authoritative `Vh`/`eta_h` diagnostic rerun independent of any dirty
developer checkout.

Then compare both traces with:

```bash
PYTHONPATH=src jax-drb compare-neutral-mixed-accepted-traces \
  neutral_mixed_native_accepted_step_trace.json \
  /tmp/neutral_mixed_reference_trace/accepted_steps.jsonl \
  --json-out neutral_mixed_accepted_step_trace_parity.json
```

The reference JSONL should include post-boundary/pre-RHS samples, `ddt(Nh)`,
`ddt(Ph)`, `ddt(NVh)`, and the existing direct `SNVh_*` source diagnostics.
That information distinguishes time-integrator history drift from
guard/boundary sequencing differences; JAXDRB-side final-state artifacts alone
do not isolate a unique safe native patch. The current reference-side monitor
has been built and run on a clean disposable checkout and writes valid JSONL.
The native accepted-step trace now writes the same state, RHS, and source field
set and can now replay the reference accepted-step time grid. A local
reference-grid comparison of `neutral_mixed_one_step` matches `148/148`
accepted points. With the timestamp mismatch removed and the reference trace
now writing `Vh` and `eta_h`, the highest matched-time input drift is `eta_h`
at about `3.23e-3` in the target/guard comparison. The next active/target
source offender is `SNVh_parallel_viscosity` at about `5.35e-5`. Large
RHS/source guard deltas remain reported but are not used to rank `ddt(*)` or
`SNVh_*`, because those guard values are diagnostic-boundary semantics rather
than active-domain source formulas. The next native parity patch should
therefore target neutral-viscosity closure preparation or boundary sequencing
under this matched-time diagnostic instead of changing the already-closed
pressure-gradient formula or the parallel-viscosity stencil before its inputs
agree.
The accepted-step comparator now also writes
`parallel_viscosity_input_register`. For each `SNV*_parallel_viscosity`
offender this register reports the matched `V*` and `eta_*` input-field
errors, ranks the matched `Dnn*`, `N*`, `P*`, and `NV*` closure/state-input
errors, lists missing input fields, and labels the diagnostic as either
`input_drift_check_available` or `reference_input_trace_missing`. When the
patched reference JSONL includes `Vh` and `eta_h`, a small input delta with a
large `SNVh_parallel_viscosity` delta points at the
`Div_par_K_Grad_par_mod(eta_h, Vh, false)` stencil or boundary semantics. The
current rerun instead reports a larger `eta_h` input delta, so the immediate
owner is accepted-step state/history sequencing, neutral closure preparation,
or target-boundary reconstruction before the viscosity stencil is changed.
On the current `148/148` matched accepted-step trace, `Nh` is the dominant
state-input drift, but the `eta_h` target-adjacent drift is about `99` times
larger than the largest state-input drift. The same rerun now includes `Dnnh`
and shows that the diffusion-coefficient target-adjacent drift is about
`4.46e-3`, larger than the `eta_h` drift of about `3.23e-3`. That separates the
offender from a directly state-sized density, pressure, or momentum mismatch
and points first at accepted-step `Dnn` preparation or target-boundary
sequencing before viscosity is formed.
The comparator ranks state fields with guard metrics, but ranks `ddt(*)` and
`SNVh_*` fields by active and target-adjacent cells while still reporting guard
deltas separately.
A separate final-state input-closure gate now compares native `Dnn`, `Vh`, and
`eta_h` reconstructions against a reference-style `BOUT.dmp.0.nc` dump. The
public API entry point is
`build_neutral_mixed_reference_input_closure_report`, with JSON output handled
by `write_neutral_mixed_reference_input_closure_json`. On the current
reference final state this gate closes the neutral diffusion, velocity, and
viscosity input formulas to roundoff in active, target-adjacent, and guard
cells. That result is deliberately narrower than the accepted-step comparator:
it shows that the `eta_h` offender should be pursued in accepted-step
state/history or target-boundary sequencing, not by changing the already
matched `eta_h = A_h D_{nn,h} N_h` closure formula.
The `trace-neutral-mixed-reference-accepted-steps` runner now validates this
schema before returning successfully: each accepted-step record must contain
`Nh`, `Ph`, and `NVh` in the `post_accepted` stage, and the JSONL must contain
`ddt(Nh)`, `ddt(Ph)`, `ddt(NVh)`, `SNVh`, `SNVh_pressure_gradient`,
`SNVh_parallel_viscosity`, and `SNVh_perpendicular_viscosity` somewhere in its
stage payloads. A reference binary that only writes aggregate final-state
arrays therefore fails fast with a list of missing fields instead of producing
an ambiguous accepted-step parity report.
The exact reference-side monitor requirements are recorded in
[hermes_neutral_mixed_accepted_step_trace_monitor.md](hermes_neutral_mixed_accepted_step_trace_monitor.md).

Run the Hermès-free diagnostic with:

```bash
jax_drb diagnose-neutral-mixed-substeps \
  --input-path /path/to/BOUT.inp \
  --reference-arrays-npz references/baselines/reference_arrays/neutral_mixed_one_step.npz \
  --substeps 1,2,3,4,6,8 \
  --json-out neutral_mixed_substep_hybrid.json
```

When no native history is supplied, the command can also generate native
histories from a local reference checkout by setting `--reference-root` or
`JAX_DRB_REFERENCE_ROOT`. In CI and release-closeout use, the same report can
be built from committed reference arrays and supplied native histories, so the
diagnostic remains deterministic and does not require Hermès to be installed.

The campaign can now also ingest a one-step Hermès diagnostic NetCDF generated
with `output_ddt = true` and `diagnose = true` under the `neutral_mixed`
component. The committed JSON/NPZ bundle includes the direct Hermès diagnostic
lineouts from that rerun: `ddt(NVh)`, `SNVh`,
`SNVh_pressure_gradient`, `mfh_visc_par_ylow`, `mfh_visc_perp_xlow`,
`mfh_visc_perp_ylow`, `mfh_adv_perp_xlow`, and `mfh_adv_perp_ylow`. The
`SNVh_pressure_gradient`, `SNVh_parallel_viscosity`, and
`SNVh_perpendicular_viscosity` variables come from the local Hermès diagnostic
patch recorded in
[hermes_neutral_mixed_pressure_gradient_diagnostic.patch](hermes_neutral_mixed_pressure_gradient_diagnostic.patch)
and the follow-on local viscosity-source patch used for this audit. They write
the same `-Grad_par(Pn)`, parallel-viscosity, and perpendicular-viscosity
sources that enter the neutral momentum equation. The report still stores the
matched postprocessed reconstructions under
`hermes_diagnostic_outputs.matched_reconstructions` because those are the
normalized JAXDRB-side operator lineouts used for native term balance. The
direct Hermès variables are therefore the reference-side written diagnostics,
while the matched reconstructions are the normalized comparison operators
evaluated on the Hermès final fields.

Regenerate the artifact with:

```bash
PYTHONPATH=src python examples/engineering/neutral_mixed_term_balance_campaign_demo.py
```

To include direct Hermès diagnostic fields, first run the Hermès neutral-mixed
case with `output_ddt = true`, `diagnose = true`, and `nout = 1`, then pass the
resulting dump:

```bash
JAX_DRB_NEUTRAL_MIXED_HERMES_DIAGNOSTIC_NC=/path/to/BOUT.dmp.0.nc \
  PYTHONPATH=src python examples/engineering/neutral_mixed_term_balance_campaign_demo.py
```

The demo can also perform that one-step Hermès diagnostic rerun directly:

```bash
PYTHONPATH=src python examples/engineering/neutral_mixed_term_balance_campaign_demo.py \
  --rerun-hermes-diagnostics \
  --diagnostic-workdir tmp/neutral_mixed_hermes_diagnostics
```

This writes a temporary deck with `nout = 1`, `output_ddt = true`, and
`diagnose = true`, runs the local Hermès executable, and then packages the
resulting `BOUT.dmp.0.nc` into the same JSON/NPZ report.
