# Frozen traced-Q diffusion: accepted static-accuracy baseline

User-approved freeze, 29 September 2026. This is a research qualification and
implementation contract, not a claim that the method is selectable in the
production solver. The [Q roadmap](parallel_second_order_roadmap.md) owns status.

**Current coupled-Q choice, 4 October 2026:** the user selects total cap
separation h/32 for diffusion and the other base parallel operators, retaining
five eta planes. h/16 remains a qualified diffusion alternative; other h/n
choices need consistent prepared geometry/rows and applicable qualification.
This selects between previously evaluated diffusion spans without changing
the operator formula, original campaign receipts or production defaults. See
the [Q08 configuration](parallel_second_order_roadmap.md#current-q08-configuration--user-freeze-4-october-2026)
for the C3 coupled baseline and material's paired outer sampling. The original
29 September evidence and choices below retain their historical identity.

## Identity and acceptance scope

- Accepted campaign identity:
  `c0d64dba761087a3273d29cd4728689c4f4b5b67f838873a845d1cdb1a809f8b`.
- Immutable [freeze manifest](../../../../work/q_fci_diffusion_freeze_20260929/freeze.json),
  [source snapshot](../../../../work/q_fci_diffusion_freeze_20260929/frozen_source.tar.gz),
  and [campaign identity](../../../../work/q_fci_diffusion_freeze_20260929/campaign_identity.json).
  The manifest checksums the complete frozen source and evidence. It pins the
  exact canonical geometry/magnetic inputs through the campaign identity.
- Packaged base revision: `dacf15f14be726374a026ea087fd9d665d35c852`.
  This revision alone does **not** identify the accepted method: its campaign
  uses always-balanced support. The accepted selective policy is the pinned
  `policy.py` and `run.py` overlay in the snapshot.
- [Global comparison](../../../../work/q_fci_selective_global_20260929/report.md),
  [N-O-R orders](../../../../work/q_fci_selective_global_20260929/orders.csv), and
  [interface audit](../../../../work/q_fci_repair_interface_audit_20260929/report.md).
- **Static reconstruction accuracy accepted with exceptions:** localized
  last-two-agglomerated-ring cancellation/truncation loss, the previously
  accepted coarse outer-envelope N-O order exception, and the separate O-R
  reference/discrete-action limitation. This does not assert unconditional
  continuum N-R convergence or transfer historical direct-method evolved and
  structural results to the traced method.
- Regional/max norms remain diagnostics under the shared global accuracy
  contract. No new requirement that every cell/ring achieves order2 is imposed.
  These exceptions are recorded explicitly, not reclassified as literal passes
  of the original every-field/interval continuum-MMS criterion.

## Frozen numerical choices

1. Actual canonical HSX angular-RLP N32/N48/N64 with one evolved value per
   owner. Owner observations and output projection use the stored physical
   raw volumes and raw-midpoint interpretation, including every aggregate member.
2. Raw-midpoint seeds, RK4 with64 steps per leg. Retain both tested variants
   alpha=1/16 and1/32. Alpha is **total cap separation** in units of deta;
   traced legs end at eta plus/minus alpha*deta/2. Keep span explicit in cache
   and application identity; no new default span or shortening is introduced.
3. Inner reconstruction: Cartesian transverse total-degree4,28 donor owners
   per eta plane per raw anchor, common five-plane eta quartic on k-2..k+2.
   Nearest28 retains its existing rank/reproduction repair. The selective
   gradient guard compares compact, always-balanced28, and local40 QR-balanced28.
   It changes support only if **both** degree5/degree6 parallel-gradient shell
   indicators improve by more than10% and coefficient amplification does not
   increase (relative comparison slack1e-12). Choose smallest worst shell ratio;
   candidate iteration local40 before broad balanced fixes tie behavior.
   Amplification is the Euclidean norm across four caps of row coefficient-L1
   sums. Preserve weighted/uniform rank checks, tolerances, pivoting, ordering,
   tie-breaking and rank-failure expansion. Field values never select donors.
4. Structured outer support: seven angular donors with four radial layers and
   radial cubic reconstruction, sharing the five-plane eta quartic. The switch
   remains after last aggregate indices10/15/21. No special ring20 veto.
5. Last two wall rings retain the frozen quartic Dirichlet and physical-normal
   Neumann constructions, including metric normal, prescribed-data contribution
   and fixed-leg ghost-endpoint treatment. Normal data are not coordinate-radial
   derivative data. The35-dimensional wall elimination is setup algebra, not
   a timestep elliptic solve. No wall extrapolation or characteristic model is
   silently substituted during extraction.
6. Reconstruct the original scalar's gradient at the traced caps, then apply
   the frozen outer difference and nonzero-divB correction. With
   g=b dot grad(reconstructed f), the raw action is

   ```text
   B0 * b_eta0 * [(g/B)_plus - (g/B)_minus] / (alpha*deta)
       + (divB0/B0) * g0
   ```

   Here b_eta is the contravariant eta component. Preserve the magnetic and
   metric evaluators and the existing divB coefficient calculation. Do not
   assume interpolated B is divergence-free. Project raw actions to owners
   only with the frozen weights.
7. Float64/complex128 verification; constants, source identities, complete-owner
   coverage, finite coefficients and polynomial reproduction are checked.
   Cross-platform action replay tolerance is1e-8 absolute; the observed maximum
   is2.16e-9, with paired local N-O replay1.58e-11. Report errors, do not erase
   them. No new MMS fields, trace locations, donor degree or selector tuning.

## Q05 extraction scope and current completion decision

Q05 was closed by user acceptance on 29 September 2026 after corrections and
second review. Full-domain replay remains unperformed and is waived as a Q06
prerequisite, not reported as passing. The [current Q06 contract](q06_traced_gradient_divergence_contract.md)
and [direct-gradient assignment](q06_direct_gradient_assignment.md) govern next
work. The extraction scope below is retained to define the shared interface.

This is a staged preparation exception following the user's accepted static
accuracy decision. It does not claim full Q04 evolved/structural certification
or authorize a production-default switch. Historical Q05a direct extraction is
useful implementation precedent; its coefficients/results are not this method.

- Read package architecture/testing guidance, then design one reusable
  host-prepared bundle for geometry identity, raw/owner maps, weights, cap
  locations, support/repair decisions, gradient/value rows, magnetic action
  coefficients and boundary sidecars. Preserve the directional data needed by
  later G/D/transport operators; do not reduce the contract to a symmetric graph.
- Provide pure-JAX application of fixed coefficients to owner fields and
  prescribed boundary data. No tracing, donor search, SVD/QR or wall solve inside
  the timestep. Distinguish preparation from runtime cost. Combine duplicate
  owner indices and masked padding without changing weighted sums;28 donors
  per plane is not a bound on the final union across planes/raw aggregate members.
- Reuse appropriate existing P/Q observation, topology and boundary abstractions
  after checking their semantics. Do not force P's transverse scheme or old
  integrated-face references into this frozen Q action merely to share code.
- Replay all26 fields, both spans and both BCs against saved global actions,
  starting with bounded core/rank-repair/interface/interior/wall patches, then
  complete coverage using existing traces. Verify boundary affine contributions,
  constants, polynomial targets, signed duplicate accumulation, dtype, JIT and
  state JVP. Time/memory preparation and application separately; GPU timing is
  remote when needed. No new continuum-reference campaign is required for replay.
- Preserve evolved-state/restart layout and current defaults. The deliverable
  is a reusable prepared/apply API with tests and an extraction report, not a
  new numerical scheme or copied per-operator campaign code.

**Extraction gate:** unchanged frozen diffusion actions, observations, BC
semantics and reconstruction decisions within justified numerical tolerance;
correct pure-JAX state application; all coverage/identity checks pass.

## Subsequent Q work

1. **Q04 traced evolution/runtime checks:** use the extracted implementation for
   a bounded conduction evolution and timestep check, then the planned evolved
   MMS and remaining structural/execution checks. Historical direct-cubic
   evolution is separate. Keep accepted static-reference exceptions visible.
2. **Q06 gradient/divergence:** reuse directional geometry, supports and boundary
   sidecars to qualify b dot grad(f), div(b f), and relevant compositions on
   real HSX. Preserve nonzero divB/div(b) terms. The frozen diffusion evaluates
   a reconstructed scalar gradient at caps; it is not automatically identical
   to D_h applied to an owner-sampled G_h field. Test that transfer explicitly.
3. **Q07 transport/material terms:** density flux, thermal advection/compression,
   velocity/pressure, current and electrostatic couplings. Qualify individual
   actions and coupled blocks before sharing characteristic/upwind runtime
   machinery. Static diffusion accuracy does not certify advection.
4. **Q08/Q09 coupled RHS and final integration:** consolidate the common API;
   term-resolved full parallel RHS, fixed-time solution MMS and timestep/error
   budgets; prescribed phi first, reconstructed phi with the P dependency.
   Complete the planned structural and execution qualification before changing
   production defaults. Do not equate this freeze with full Q00–Q09 completion.

No further donor-policy search or new global tracing campaign is the next step.
