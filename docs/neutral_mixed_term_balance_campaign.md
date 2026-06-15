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
The native trace now also writes the full neutral diffusion-coefficient
preparation ladder: `Tnlimh`, `logPnlimh`, `grad_logPnlimh`, `Dnnh_raw`,
`Dnnh_flux_max`, `Dnnh_flux_limited`, and `Dnnh_diffusion_limited`, followed by
the existing boundary-applied final `Dnnh`. Those fields are diagnostic-only
and do not change the production neutral-mixed solve. The reference
accepted-step monitor patch now emits the same optional ladder. Native and
reference payloads also include flattened target-adjacent and guard-band
values, so the comparison reports both the legacy zone max/rms metric drift and
the actual pointwise target-cell or guard-cell drift. This separates a real
target-cell mismatch from a case where the largest value merely occurs at a
different symmetric cell. The comparator also labels ladder fields such as
`Dnnh_flux_max` as `active_target_preboundary_diagnostic`: guard values are
still reported, but the field is ranked by active and target-adjacent cells
because the reference snapshots are taken before the final diffusion boundary
application.
The native `grad_logPnlim*` implementation now evaluates the covariant
`|Grad(logPnlim)|` norm with the carried metric terms `g11`, `g22`, `g33`, and
the supported `g23` cross term, matching the reference vector norm on the
structured metrics represented in JAXDRB.
The comparator ignores those fields when an older reference JSONL does not
contain the same payloads, so the addition is backward-compatible with older
reference traces. The trace parity report also counts native/reference solver
order mismatches using the reference `solver.order` value written from
`CVodeGetLastOrder`, excluding the synthetic native initial state at `t = 0`
from the aggregate order count. A BDF2 native replay should not be interpreted
as a physics-term mismatch when the reference accepted step used a different
multistep order. When a reference executable with the accepted-step monitor patch is
available, write the matching reference JSONL with:

```bash
PYTHONPATH=src jax-drb trace-neutral-mixed-reference-accepted-steps \
  --reference-root /path/to/reference-root \
  --workdir /tmp/neutral_mixed_reference_trace \
  --trace-out /tmp/neutral_mixed_reference_trace/accepted_steps.jsonl \
  --cvode-max-order 2
```

`--cvode-max-order 2` stages `solver:cvode_max_order = 2` in the generated
reference `BOUT.inp` so the CVODE accepted-step trace is constrained to the
same maximum method order as the native BDF2 replay. The runner validates the
emitted `solver.order` values and fails if any accepted reference step exceeds
the configured ceiling. Omit the option only when intentionally auditing the
stock variable-order reference lane.

Recent reference monitor patches also emit `active_shape` and `active_values`
for every traced field. When the comparator is given the corresponding
`BOUT.inp` through `--input-path`, it reconstructs the full active reference
`Nh`, `Ph`, and `NVh` states and evaluates the native backward-Euler or
variable-step BDF2 residual directly on those reference accepted states. The
result appears in `reference_active_state_residual_register`. A large residual
there means the remaining parity offender is still a native RHS, boundary, or
closure mismatch. A small residual there, combined with visible final-state
drift, means the RHS is locally compatible with the reference state and the
next patch should target accepted-state history preparation, nonlinear
tolerance, or multistep sequencing. Older reference traces that do not contain
full active payloads remain valid; their report marks this register as
unavailable instead of failing.

The current live ladder rerun uses a contextual reference patch with deep-copy
snapshots for `Dnn` before and after each limiter stage. It produced `148`
matched accepted-step records and no missing ladder fields. The resulting
`neutral_diffusion_ladder_register` ranks `Dnnh_flux_max` as the dominant
target-band ladder mismatch. The pointwise target-cell comparison confirms that
this is not only a zone-maximum ordering artifact: at the upper target-adjacent
cell corresponding to local target index `[0, 3, 0]`, native `Dnnh_flux_max` is
`2.74471293` while the reference value is `2.73944`, a `5.27e-3` difference.
The final `Dnnh` pointwise target drift is `4.46e-3`, while raw diffusion is
`6.07e-4`. The same point shows raw diffusion and temperature are essentially
closed, but `grad_logPnlimh` differs by about `4.48e-5`. A June 8, 2026 guard
payload rerun shows large pre-boundary guard deltas in `grad_logPnlimh` and the
intermediate limiter fields, while final boundary-applied `Dnnh` has matching
target and guard pointwise drift (`4.46e-3`). The next native patch should
therefore target accepted-state history feeding the near-target
`Grad(logPnlim)` stencil before changing collision rates or the raw diffusion
formula. A direct check at the worst `Dnnh_flux_max` target cell closes the
flux-cap algebra on both sides: the recorded cap is reproduced by
`flux_limit sqrt(Tnlim/AA)/(grad_logPnlim + 1/lmax)` when `lmax` is inferred
from the local raw diffusion. The same point already has native `Ph`, `Nh`,
and `logPnlimh` above the reference by `6.69e-6`, `6.50e-5`, and `9.36e-5`,
respectively, so the remaining blocker is accepted-step state-history
sequencing rather than a missing flux-cap term.

If `--hermes-binary` is not supplied, this command now builds a cached clean
patched reference worktree automatically before launching the trace run. That
keeps the authoritative `Vh`/`eta_h` diagnostic rerun independent of any dirty
developer checkout.

Then compare both traces with:

```bash
PYTHONPATH=src jax-drb compare-neutral-mixed-accepted-traces \
  neutral_mixed_native_accepted_step_trace.json \
  /tmp/neutral_mixed_reference_trace/accepted_steps.jsonl \
  --input-path /tmp/neutral_mixed_reference_trace/reference_run/data/BOUT.inp \
  --reference-cvode-max-order 2 \
  --json-out neutral_mixed_accepted_step_trace_parity.json
```

The reference JSONL should include post-boundary/pre-RHS samples, `ddt(Nh)`,
`ddt(Ph)`, `ddt(NVh)`, and the existing direct `SNVh_*` source diagnostics.
That information distinguishes time-integrator history drift from
guard/boundary sequencing differences; JAXDRB-side final-state artifacts alone
do not isolate a unique safe native patch. The current reference-side monitor
has been built and run on a clean disposable checkout and writes valid JSONL.
The native accepted-step trace now writes the same state, RHS, and source field
set and can replay the reference accepted-step time grid. A local
reference-grid comparison of `neutral_mixed_one_step` matches `148/148`
accepted points. With timestamp mismatch removed and the reference trace
writing `Dnnh`, `Vh`, `eta_h`, and the diffusion-preparation ladder, the largest
remaining pointwise target-band closure drift is final `Dnnh` at about
`4.46e-3`, followed by `eta_h` at about `3.23e-3`. The next active/target source
offender is `SNVh_parallel_viscosity` at about `1.29e-4` pointwise
(`5.35e-5` by the legacy zone-metric comparison). Large RHS/source guard deltas
remain reported but are not used to rank `ddt(*)` or `SNVh_*`, because those
guard values are diagnostic-boundary semantics rather than active-domain source
formulas. The next native parity patch should therefore target the
flux-limit-cap and near-target gradient/boundary sequencing under this
matched-time diagnostic instead of changing the already-closed pressure-gradient
formula or the parallel-viscosity stencil before its inputs agree.
The accepted-step comparator now also writes
`native_solver_order_summary`, `reference_solver_order_summary`,
`reference_solver_control`, `parallel_viscosity_input_register`, and
`accepted_step_state_history_register`.
`reference_solver_control` records the configured `cvode_max_order`, the
observed reference max solver order, whether the trace stayed within the
configured ceiling, and a bounded list of violating points if it did not. For
each `SNV*_parallel_viscosity`
offender this register reports the matched `V*` and `eta_*` input-field
errors, ranks the matched `Dnn*`, `V*`, and `eta_*` closure inputs separately
from the `N*`, `P*`, and `NV*` state inputs, lists missing input fields, and
labels the diagnostic as either `input_drift_check_available` or
`reference_input_trace_missing`. When the patched reference JSONL includes
`Dnnh`, `Vh`, and `eta_h`, a small input delta with a large
`SNVh_parallel_viscosity` delta points at the
`Div_par_K_Grad_par_mod(eta_h, Vh, false)` stencil or boundary semantics. The
current rerun instead reports a larger `Dnnh` closure-input delta. The JSON
now makes that explicit through `dominant_closure_input_field`,
`max_closure_input_*_delta`, and `diffusion_to_state_*_ratio`, so the immediate
owner is accepted-step neutral diffusion-coefficient preparation,
state/history sequencing, or target-boundary reconstruction before the
viscosity stencil is changed.
`accepted_step_state_history_register` then selects the dominant neutral
diffusion ladder offender, finds its worst target-adjacent local index, and
writes a compact time window of matched native/reference values for `N*`,
`P*`, `NV*`, limiter inputs, optional covariant `Grad(logPnlim)` components,
diffusion-ladder fields, `V*`, `eta_*`, and `SNV*_parallel_viscosity`. This
makes the state-to-flux-cap amplification path reproducible from the JSON
report without loading the full trace payloads into an ad hoc analysis script.

A controlled max-order-2 reference rerun is now available as the preferred
neutral NVh parity lane. The staged reference deck used
`solver:cvode_max_order = 2`, emitted `309` accepted points, and the native
replay matched all `309/309` accepted times with zero solver-order mismatches.
The reference solver-control payload reported observed max order `2`, no
configured order-ceiling violations, and `within_configured_max_order = true`.
The leading offender did not move to the pressure-gradient or viscosity
formulas: `Dnnh_flux_max` remains first, with a target-adjacent drift of about
`5.13e-3`; final `Dnnh`, diffusion-limited `Dnnh`, flux-limited `Dnnh`, and
`eta_h` follow. The ladder-transition register now identifies the dominant
max-order-2 jump as `Dnnh_raw -> Dnnh_flux_max`: target-pointwise error rises
from about `2.83e-4` to about `5.13e-3`, while the raw-diffusion formula itself
stays much closer. The practical conclusion is that variable CVODE order is no
longer the dominant explanation for the neutral NVh mismatch. The remaining
patch should compare accepted-step state/history sequencing and the near-target
`Grad(logPnlim)`/flux-limit-cap preparation directly against this max-order-2
trace before changing local source-term formulas.
The June 15, 2026 tight native replay keeps that same conclusion after lowering
the native accepted-step residual tolerance. A follow-up component-enabled
rerun replayed the reference startup order sequence (`1, 1, 2, ...`) and
removed the only native/reference solver-order mismatch. The
`accepted_step_state_history_register` still matches `309/309` accepted-step
points and identifies `Dnnh_flux_max` at `t = 2.7536261188` and local
target-adjacent index `[5, 3, 0]` as the dominant point. At that point `Nh` is
high by `6.70e-5`, `Ph` by `6.62e-6`, `NVh` by `2.04e-6`, `logPnlimh` by
`9.18e-5`, scalar `grad_logPnlimh` by `-4.51e-5`, `Dnnh_flux_max` by
`5.13e-3`, final `Dnnh` by `4.35e-3`, `eta_h` by `3.13e-3`, and
`SNVh_parallel_viscosity` by `-1.07e-5`. Across the full trace, the dominant
state-input drift remains `Nh` (`6.83e-5`) and the dominant scalar limiter
input remains `logPnlimh` (`9.94e-5`), giving flux-cap amplification ratios of
about `51.6x` relative to the scalar limiter input and `75.0x` relative to the
state input. Optional `grad_logPnlim*_x/y/z` component fields are now reported
separately; they expose coordinate-component differences, but the cap itself
uses the scalar `abs(Grad(logPnlim))`. That separates the offender from a
directly state-sized density, pressure, momentum, raw-diffusion, component
gradient, or local source-term formula mismatch and points first at a closer
CVODE-style accepted-step state/history replay for the neutral
pressure/log-pressure limiter.
The accepted-step comparator now also writes an
`accepted_step_error_onset_register`. On the component/order-replay trace, the
first 10% relative onset appears in scalar `logPnlimh` at `t = 0.454`, in
`Nh` at `t = 0.673`, and in `Dnnh_flux_max` at `t = 0.851`, all after the
startup steps have reached order 2. The ordering is important: the cap drift
does not first appear as an isolated late-time algebraic jump; it follows
accumulated accepted-state and scalar-limiter drift. The next high-value
reference patch should therefore emit enough full active-field state to test
the native residual on reference accepted states, or emulate CVODE's accepted
state/history update more closely, before changing neutral source terms.
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
it shows that the accepted-step `Dnnh`/`eta_h` offender should be pursued in
accepted-step state/history, diffusion-coefficient preparation, or
target-boundary sequencing, not by changing the already matched
`eta_h = A_h D_{nn,h} N_h` final-state closure formula.
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
