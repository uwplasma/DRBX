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

![Neutral mixed term-balance audit](data/neutral_mixed_term_balance_campaign_artifacts/images/neutral_mixed_term_balance_campaign.png)

Current artifact outputs:

- JSON summary: [neutral_mixed_term_balance_campaign.json](data/neutral_mixed_term_balance_campaign_artifacts/data/neutral_mixed_term_balance_campaign.json)
- compact arrays: [neutral_mixed_term_balance_campaign.npz](data/neutral_mixed_term_balance_campaign_artifacts/data/neutral_mixed_term_balance_campaign.npz)
- figure: [neutral_mixed_term_balance_campaign.png](data/neutral_mixed_term_balance_campaign_artifacts/images/neutral_mixed_term_balance_campaign.png)

The current generated report shows a worst active-domain final `NVh` difference
of about `3.37e-3`. Inserting the native final state back into the native
operator gives a residual-rate max of about `1.85e-4`, while inserting the
Hermès-3 final state gives about `2.52e-3`. That gap makes the next parity task
specific: compare the pressure-gradient and parallel-viscosity lineouts against
Hermès-3 operator diagnostics or add targeted boundary/closure unit tests for
the neutral-mixed momentum equation.

The campaign can now also ingest a one-step Hermès diagnostic NetCDF generated
with `output_ddt = true` and `diagnose = true` under the `neutral_mixed`
component. The committed JSON/NPZ bundle includes the direct Hermès diagnostic
lineouts from that rerun: `ddt(NVh)`, `SNVh`,
`SNVh_pressure_gradient`, `mfh_visc_par_ylow`, `mfh_visc_perp_xlow`,
`mfh_visc_perp_ylow`, `mfh_adv_perp_xlow`, and `mfh_adv_perp_ylow`. The
`SNVh_pressure_gradient` variable comes from the local Hermès diagnostic patch
recorded in [hermes_neutral_mixed_pressure_gradient_diagnostic.patch](hermes_neutral_mixed_pressure_gradient_diagnostic.patch);
it writes the same `-Grad_par(Pn)` source that enters the neutral momentum
equation. The report still stores the matched postprocessed reconstruction
under `hermes_diagnostic_outputs.matched_reconstructions.pressure_gradient`
because it is the normalized JAXDRB-side operator lineout used for native term
balance. The direct Hermès variable is therefore the reference-side written
diagnostic, while the matched reconstruction is the normalized comparison
operator evaluated on the Hermès final pressure field.

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
