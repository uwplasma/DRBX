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
For every accepted internal solver step from `t = 0` to the one-step output
time, the trace should write `time`, `dt`, solver order, and both
post-boundary/pre-RHS and post-accepted `Nh`, `Ph`, and `NVh` values at the
active target-adjacent cells and adjacent guard cells. The same sample should
include `ddt(Nh)`, `ddt(Ph)`, `ddt(NVh)`, and the existing direct
`SNVh_*` source diagnostics. That is the missing information needed to
distinguish time-integrator history drift from guard/boundary sequencing
differences; the current JAXDRB-side artifacts alone do not isolate a unique
safe native patch.

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
