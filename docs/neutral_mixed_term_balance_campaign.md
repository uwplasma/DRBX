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
- compact arrays: [neutral_mixed_term_balance_campaign.npz](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_term_balance_campaign_artifacts__data__neutral_mixed_term_balance_campaign.npz)
- figure: [neutral_mixed_term_balance_campaign.png](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_term_balance_campaign_artifacts__images__neutral_mixed_term_balance_campaign.png)

The published generated report uses the physical active `x-y` domain for all
term metrics and direct-reference scaling, so guard-cell-only diagnostics do
not contaminate the offender ranking. That report shows a worst active-domain
final `NVh` difference of about `5.81e-4`, down from the earlier
boundary-local `3.37e-3` mismatch after the neutral-mixed mesh topology and
one-step history substepping were tightened. Inserting the Hermès-3 final state
into the native operator gives a residual-rate max of about `1.66e-4`;
inserting the native final state gives a residual-rate difference of about
`9.01e-5` against the Hermès balance. The remaining final-state drift is
therefore no longer a missing direct pressure-gradient or viscosity source
formula.

The current code-path audit adds connected-y guard reconstruction for
non-target neutral-mixed meshes and promotes the one-step native default from
four to eight internal BDF substeps. Before regenerating the release media,
the combined change reduces the active-domain final `NVh` metric from about
`5.81e-4` in the published report to about `4.47e-6`. The same one-step audit
reduces the active-domain final `Nh` and `Ph` metrics to about `2.19e-4` and
`2.11e-5`, respectively. The short-window path remains at four internal
substeps because the prefix sweep shows that eight substeps improve center
momentum but do not solve the larger total-density/pressure history drift.
The tracked JSON report has been regenerated from this code path. The
remaining release-media task is to publish the matching regenerated NPZ/PNG
bundle to the validation-artifact release so the remote figure reflects the
same numbers.

The report now carries a target-adjacent offender register rather than only
aggregate field errors. On the native-minus-Hermès final-state term deltas,
pressure gradient (`6.60e-4`) and parallel viscosity (`6.42e-4`) remain the
largest named differences because the native and Hermès final states are not
identical. Direct source-level diagnostics close the implementation question:
after active-domain scaling, `SNVh_pressure_gradient`,
`SNVh_parallel_viscosity`, and `SNVh_perpendicular_viscosity` agree with the
matched JAXDRB reconstructions with max absolute differences of about
`1.3e-11`, `1.2e-11`, and machine precision, respectively.

The JSON report also carries `state_driver_register`, which turns that
interpretation into a regression target. Dividing the final-state differences
by the one-step interval ranks the target-adjacent state-rate errors as `Nh`
(`5.11e-4`), then `Ph` (`4.53e-5`), then `NVh` (`2.90e-5`). The induced
momentum-driver deltas are led by `Ph -> pressure_gradient` (`6.60e-4`) and
`NVh -> parallel_viscosity` (`6.42e-4`), with target-to-interior ratios of
about `3.22` and `4.65`. The next implementation target is therefore the
target-adjacent state history and boundary reconstruction that feeds those
closed operators, not a replacement of the pressure-gradient or viscosity
formula.

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
