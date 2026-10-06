# Q09 prescribed-phi evolved MMS harness

Status: **prescribed-phi finite-duration evolved MMS passed with documented
spatial-accuracy exceptions (5 October 2026)**. The raw compact-C3 N32/N48/N64
campaign covers 24 integrations, diffusion-only and the six-field parallel RHS,
four D/N/mixed patterns, 5,800 accepted steps and t=1e-4. This scoped acceptance
covers the tested evolution/source/BC implementation; it does not assert
uniformly second-order solution errors or close the full Q09 integration gate.
The [roadmap](parallel_second_order_roadmap.md#q09--certify-evolved-coupled-mms-and-complete-final-integration)
records the acceptance and remaining work.

Vi/Ve fine global orders remain approximately 0.99-1.00 / 1.14-1.15. Omega
retains a 11.9-12.5% global rebound in two BC patterns and 16.5-17.3% bulk
rebound in all four. Subsequent saved-state/closure audits attribute the
leading coupled omega error to Ve O-R propagated through current divergence.
Fixed-location span and raw-versus-filtered magnetic comparisons support a
geometry/error-sampling explanation, not a blanket reference-bug claim.
Keep [raw evolved results](../../../../work/q09_evolved_return_20261005/report.md)
and the [filtered control](../../../../work/q_filtered_velocity_20261005/report.md)
separate: the latter is bounded static evidence only, with residual wall
sampling and wave reconstruction limitations, not an evolved qualification.

The N32 three-dt pilot has differences at roundoff; temporal convergence order
is unmeasured, and the longer campaign uses one dt per resolution. Long-time
stability/CFL, reconstructed phi, perpendicular coupling, multi-GPU evolution,
physical sheath/exterior crossings and production remain unqualified. Magnetic
filtering is a verification control, not a production field selection. Check
raw-field anisotropic resolution and intended physical observables before
making production-accuracy claims; this MMS does not quantify their error.

## Filtered global comparison prepared — 5 October 2026

[`scripts/q09_filtered_global`](../../../scripts/q09_filtered_global/README.md)
prepares the matching 24-run evolved campaign for P's immutable m<=3-per-period
Fourier filter of `J B^i` (four field periods, 64 samples/period, table arm
`e61e1d70a7f14697a10a10fe2b1f1262a8f0a9752a79c237d78e3ebe83c23aae`).
This changes the magnetic field only; the map, owner volumes, smooth MMS,
prescribed phi, D/N/mixed cases, h/32 inner/diffusion and h/16 outer spans,
100/225/400 time steps to t=1e-4 and five snapshots remain fixed.

The filtered adapter regenerates RK4-64 traces, geometry-dependent repair
choices, both paired reconstruction banks, tube coefficients and continuum
forcing. It cannot consume raw-field banks/references under a filtered label.
It reuses the frozen numerical runtime and evolution/source kernels. CPU
workers prepare whole-owner chunks using the table/map only; smooth-only
reference preparation is checked against the full-catalogue authority.
GPU tracing and evolution remain mandatory remotely. New endpoint crossings
stop the campaign; neither the filtered control nor this short-span test
qualifies exterior ghosts or a live sheath law.

Local bounded preparation and portable tests precede the remote hardware gate.
Global filtered results are pending, with no promised second-order rate:
compare all fields and regions, especially Vi/Ve, omega current transfer and
wall errors, against the raw evolved results. This smooth-field campaign does
not replace the documented short-wave reconstruction limitations. The raw
production evaluator/default is unchanged.

## Selected state and exact observation

The numerical unknown is `(n, Te, Ti, Vi, Ve, omega)` in `(6, n_owner)` order.
Phi is prescribed, with no perpendicular evolution or elliptic solve. Reuse
Q08's h/32 traced diffusion and material inner total separation, h/16 outer
characteristic separation, five eta planes, accepted compact28 repair/guard and
wall policy, `compact_c3` magnetic data and saved RK4-64 endpoints. The inner
half separation is `eta_step=(2*pi/N)/64=pi/N/32`; no rows or traces are rebuilt.
The constants are `mu=1836`, `tau=1` and `D=(.01,.02,.03,.04,.05,.06)`.
Complete mode invokes `apply_q_plan(..., characteristic_method='polynomial')`;
diffusion mode calls the same existing cap-gradient diffusion kernel directly.

For `0 <= t <= 1`, define `a(t)=1+0.1*sin(t)`. The initial spatial family is the
frozen Q08 smooth case: primitive deviations proportional to `r*cos(theta)*cos(eta)`
for n/Ti/Ve, and `r*sin(theta)*sin(2*eta)` for Te/Vi, with the existing amplitudes
and offsets. `U*(t)=BASE+a(t)*(U_smooth-BASE)`, where the six-field base ends in
omega=.2, and `dU*/dt=.1*cos(t)*(U_smooth-BASE)`. For `0<=r<=1`, positive lower
bounds are n>=.89, Te>=1.045 and Ti>=.845. Velocities are unconstrained in sign.
Omega remains `WOFF + WC dot (n,Te,Ti,Vi,Ve)` with the frozen coefficients;
it is a manufactured independent evolved unknown, with no enforced polarization
relation. `phi*(t)=a(t)*(.07*sx+.04*sy)`.

`Manufactured.from_members` takes actual raw member coordinates, owner IDs,
physical raw volumes and owner volumes. It checks their correspondence to the
same full bank and uses the accepted physical-volume weighted member observation.
An owner midpoint is never substituted. Phi and the analytic time derivative
use the identical observation. At every RK stage, the analytic D value,
D query value and theta/eta tangential derivatives, or physical-normal N data,
are evaluated through their exact amplitude formula on the bank's original wall
query maps. The outward normal sign comes from `boundary_wall_normal`, with no
extra sign or conversion to a coordinate-radial derivative. Mixed field kinds
and independent phi D/N selection remain explicit. Nonwall padding retains the
existing Q08 affine omega convention.

## Independent scientific source

`ContinuumReference.prepare` reuses Q08's independent continuum coordinate-flux
reference and its three fourth-order derivative spacings (1e-4, 5e-5, 2.5e-5).
It caches the raw-center smooth values, magnetic directional derivatives,
phi derivative, B, coordinate divergence kappa and projected constant-coefficient
diffusion. It verifies t0 complete-action replay against Q08's corrected Ti
continuum equations. Its preparation accepts at most 128 raw rows. The saved-input
adapter reuses the checked smooth Q08 continuum diffusion columns and independently
reevaluates raw center geometry and fourth-order coordinate div(b) for the nonlinear
material source. It never uses numerical flux weights to define this source.
Only R columns 12:24 are consumed; the historical standalone Ti diagnostic sign
in column 29 is irrelevant. Reference sensitivity diagnostics remain attached.
For inputs without saved references, `prepare_chunks` retains the original
bounded whole-owner preparation interface; the shipped adapter requires the
accepted saved reference run.

At each stage the linear diffusion is multiplied by a(t), but the nonlinear
material/current equations are reevaluated using the current analytic primitive
values and gradients. In particular physical current divergence is
`(Vi-Ve)*G(n)+n*(G(Vi)-G(Ve))+kappa*n*(Vi-Ve)`; B²/n is applied to the raw
current before owner projection. The electron equation has the corrected
cancellation of its material Ti compensation with the generalized force.
Projection is last, matching Q08. The source is always
`S(t)=dU*/dt-R_continuum(U*(t),phi*(t))`; neither the evolving numerical action
nor an exact-slot discrete stencil defines scientific forcing. There is no
temporal interpolation and no geometry evaluation at RK stages.

## Execution and acceptance behavior

`SixState` is a research-only `FciModelState` adapter to the existing shared
`Rk4Stepper`. Stage times are t, t+dt/2, t+dt/2 and t+dt. The numerical operator
receives the evolving owner state; exact states are used for initialization,
BCs, prescribed phi, source and error measurement. Prepared plan, BC, observation,
reference and volume arrays are dynamic `QPayload` carry leaves through JIT,
not closed XLA constants. Stage diagnostics retain device-reduced volume integrals
rather than transferring all stage RHS fields to the host.

Every stage checks finite state/source/RHS, positive owner n/Te/Ti and applicable
numerical admissibility; complete mode includes reconstructed-state/characteristic
flags. Every accepted endpoint is checked again. There is no clipping or relaxed
after-the-fact gate. Failure raises before accepting or checkpointing that step.
Reports state actual start/end, accepted step count and stage times, regional and
global volume-weighted RMS/relative/max solution error, integral change, integrated
RK RHS, RK balance residual and signed manufactured-solution integral drift.
The N32 pilot uses three sequential dt/dt2/dt4 runs to compare temporal self
differences separately from spatial MMS error. The full N32/N48/N64 extension
uses one dt per resolution, as requested. Completion of a finite run is not a
stability pass.

## Prepared provider and checkpoints

`PreparedProvider` requires a validated complete Q bank, balanced frozen geometry,
member-observed MMS, independent prepared reference, positive physical owner volumes,
regional boolean masks and explicit content input hashes. It rejects the saved
seven-output-owner fixtures even though their donor IDs address the global state.
The existing bank provenance records the original preparation campaign (e.g.
`c0d64dba...`) and C3 geometry (`C3:ab607bf...`); these are distinct from the
accepted Q08 extraction receipt identity `ff1bff86...`. The provider input hash
map must include `q08_receipt_identity` equal to that accepted receipt and
`prepared_bank_identity` equal to the actual merged bank identity. The provider
adapter verifies the receipt/files chain before supplying those entries.
Policy checks include the actual bank support, trace method/count and half span.
The provider identity hashes bank, geometry, reference, volume, regional masks,
inputs and harness/reference/runtime source files.

Each checkpoint binds provider identity, mode, D/N assignments, time interval
and timestep; array content hashes detect finite-state corruption. Resume checks
accepted count, state positivity/finiteness, exact time grid, stage history and
integral diagnostics. The CLI takes a single-writer output lock, binds the output
directory to one run configuration and removes a prior completed report before
attempting a resume. Failed runs emit no new completed report or pass receipt.

## Saved-input adapter and pilot replay

`scripts.q09_evolved_mms.inputs:load` is now the default CLI provider. Create a
separate configuration directory and copy
`scripts/q09_evolved_mms/inputs.example.json` to its `inputs.json`. The example
records the known Perlmutter input paths and N32; adjust paths and the host
memory budget to the actual allocation. Its 40 GiB budget is an admission
budget, not measured peak consumption. Relative cache paths resolve against
the configuration directory; never put the configuration/cache inside an
immutable input run.

The adapter checks the frozen extraction design/input authority, original
verification and complete CPU-grid receipts, every consumed bank/geometry file,
all geometry-model files, canonical inputs for the selected grid, and the complete
scientific reference receipt chain. Each reference payload must link to the
same bank-chunk receipt. A former scientific `completion.json` is neither needed
nor fabricated: checked R payloads, not the old final report, are the input.
The existing `merge_chunks` implementation validates full owner/raw closure and
replays literal chunk identities. Raw coordinates, physical volumes and topology
are mapped through `bank.raw`, which is in owner-major order, not necessarily
raw grid order.

New raw continuum preparation is cached one chunk at a time under an input/source
identity with content receipts and a single-writer lock. Interrupted payloads
without receipts are recomputed; corrupt receipts/payloads stop reuse. Cached
preparation is CPU work; all live RHS arrays are explicitly staged onto the
selected application device. There is no CPU fallback for the numerical RHS.
No tracing or donor/reconstruction generation occurs.

Canonical C3 inputs are available locally. The 4 October bounded actual-HSX
adapter audit covers seven N32 owners / 46 raw members, including core,
transition, bulk and wall. It reuses returned remote R data: consumed R columns
replay fresh local evaluation within 1.784e-11; reused versus freshly prepared
time-dependent continuum RHS agree within 2.105e-13 at t=0,.4,1. The check took
9.18 seconds with 0.79 GiB peak RSS. These are bounded reference/input checks,
not time-evolved solution or full-domain factory evidence. The full saved banks
were retired locally; use the retained remote extraction run for the pilot.

After selecting an allocation with the required saved inputs, first audit and
prepare without time integration:

```bash
JAX_ENABLE_X64=true PYTHONPATH=src:. python -m scripts.q09_evolved_mms.inputs \
  audit --inputs /path/to/q09-config
JAX_ENABLE_X64=true PYTHONPATH=src:. python -m scripts.q09_evolved_mms.inputs \
  prepare --inputs /path/to/q09-config
```

The completed N32 pilot is frozen in
[`scripts/q09_evolved_mms/README.md`](../../../scripts/q09_evolved_mms/README.md).
Use its `campaign.py verify/tests/preflight/run/validate` sequence for remote
execution. The generic lower-level CLI remains available in this form:

```bash
JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true PYTHONPATH=src:. python -m scripts.q09_evolved_mms.cli \
  --inputs /path/to/q09-config \
  --output /path/to/new/diffusion --mode diffusion --dt 1e-6 --end 1e-5
JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true PYTHONPATH=src:. python -m scripts.q09_evolved_mms.cli \
  --inputs /path/to/q09-config \
  --output /path/to/new/complete --mode complete --dt 1e-6 --end 1e-5
```

Both commands run the three timestep levels sequentially. The campaign applies
`--kinds NNNNNN --phi-kind N` for physical-normal N data, and the existing
`DNDNDN`/`NDNDND` field patterns with the opposite phi kind for mixed data.
Reuse the same arguments and add `--resume` for an interrupted low-level CLI
run; `campaign.py run` already resumes checked levels/cases in the same RUN.
The completed pilot and longer N32/N48/N64 extension are assessed in the
roadmap; these commands retain the pilot replay interface. No remote job or
production selector is changed by documenting their scoped acceptance. The frozen pilot is full-domain N32,
eight mode/BC cases and three levels each, with 10/20/40 steps to exactly 1e-5.
The timestep count tolerates a floating-point integral-ratio roundoff, avoiding
a spurious eleventh step; each level finishes at the exact requested endpoint.
One GPU executable per mode/BC case is reused across timestep levels. Preparation,
staging, first-step compilation and warm steps are recorded separately. Full
GPU execution has not been measured locally. Very small temporal differences
may hit roundoff; no favorable temporal order is required to complete the pilot.
Short-time success alone cannot qualify stability or evolved spatial accuracy.

Portable tests cover analytic derivative/source algebra, shared RK4 order on a
known ODE, stage timing/evolving-state semantics, failures, checkpoint integrity,
JAX dynamic inputs, and actual bounded-HSX t0 action/BC replay. These are bounded
wiring/algebra evidence, not global dynamics or stability qualification.
