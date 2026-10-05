# Roadmap: second-order perpendicular operators on angular RLP grids

## P07N physical-normal Neumann static qualification accepted — 27 September 2026

**User decision: P07N is qualified** as a global static qualification of the
physical-normal Neumann diffusion/polarization closure on the canonical
N32/N48/N64 geometries. The field-derived campaign at `274e93e9`, with its
[analysis](../../../../work/p07n_field_derived_274e93e9_20260927T054625Z_72cfa1/local_analysis/report.md)
and [acceptance record](../../../../work/p07n_field_derived_274e93e9_20260927T054625Z_72cfa1/local_analysis/acceptance_decision.md),
is accepted on the closure error. N−O converges at
`2.24–3.85/2.25–3.73` on every field, including the held-out field, and the
wall-normal residual converges at about 4th order. The returned headline flag
`global_order_pass=false` is preserved. N−R ≈ O−R orders are
`1.55–1.74/1.72–1.83`, exactly as the predeclared screen predicted; the held-out
field reaches `1.713/1.719`. That midpoint shortfall is recorded as a geometry-resolution
limitation of the test problem, not as a closure or reference defect. See the
[near-wall toroidal geometry resolution](#near-wall-toroidal-geometry-resolution-coil-ripple--27-september-2026)
section below. The earlier `5930b72c` campaign remains a preserved failure.
Pure-Neumann polarization inversion, energy, evolution and production
integration remain separate.

## P05 direct midpoint static qualification accepted — 26 September 2026

**User decision: P05 is qualified for global static midpoint MMS accuracy**
on the canonical N32/N48/N64 geometries, for the direct reconstructed bracket
and the direct bracket plus the saved owner-normalized `U - A` jump. This
supersedes the P05-unqualified status of the earlier face-flux/cell-correction
assembly below; it does not relabel that older failed campaign as passed.

The [completed campaign and independent recovery audit](../../../../work/p05_direct_midpoint_global_565e1d1a_HsoyFbJ3/local_analysis/report.md)
at producer revision `565e1d1aa2ff4d57414886051afb23067ae8ffb2` verifies all
1,584 numerical and 144 preflight chunks. Both candidates pass the frozen
global RMS order threshold of 1.8 on both intervals for all seven nonconstant
cases, including zero/varying Dirichlet and varying-generator cases. Centered
actual-vorticity orders are `4.030/1.847`; with the saved jump they are
`4.047/1.930`. The six other centered cases have fine orders `2.810–2.989`.
All nonconstant regional RMS errors decrease. The separate
[acceptance record](../../../../work/p05_direct_midpoint_global_565e1d1a_HsoyFbJ3/local_analysis/acceptance_decision.md)
preserves the original inputs, reference, thresholds, flags and recovery provenance.

Freeze the direct strong bracket from reconstructed gradients at each raw
midpoint, then project complete actions with stored physical raw volumes to
owners. Retain the structured/ringwise/coupled-axis and prescribed-Dirichlet
rows. The optional tested jump is the saved `U - A` correction; this acceptance
does not qualify a new runtime jump implementation or promote the previous
face-flux/cell-correction base. Production integration/replay of this direct
candidate, conservation/energy properties, evolved accuracy and stability
remain separate follow-through. Historical package extraction results below
apply to their own operator identities.

**Qualification limit:** actual-vorticity maximum errors rebound outside the
axis: centered `0.273419 → 0.066805 → 0.152426`. Uniform pointwise convergence
and third order for every field are not claimed. The
[reconstruction-row audit](../../../../work/p05_reconstruction_row_audit_20260926/report.md)
finds no row/assembly defect. The
[signed geometry-term audit](../../../../work/p05_omega_geometry_terms_audit_20260926/report.md)
isolates the dominant error to magnetic-projector derivatives in the
manufactured continuum vorticity. The
[fixed-map interpolation comparison](../../../../work/p05_field_interpolation_audit_20260926/report.md)
and [raw MAKEGRID checks](../../../../work/p05_makegrid_native_audit_20260926/report.md)
show strong derivative sensitivity, predominantly in the source toroidal
direction. This is evidence of a difficult interpolation-dependent test field,
not proof that the frozen reference is algebraically wrong or that the whole
hotspot error is nonphysical. It is a nonblocking limitation of the accepted
P05 static milestone; do not redesign P05 or rerun its global campaign solely
to obtain a more favorable actual-vorticity slope.

### Magnetic-field derivative/reference resolution follow-up

**Planned geometry follow-up:** assess the magnetic-field representation and
the derivatives actually consumed by the selected operators/references. The
manufactured-vorticity hotspot does not demonstrate an inherent simulation
error. Independent finer-source validation is valuable, but is not a blanket
prerequisite for simulation work or a retroactive P05 static-pass gate.
Simulation readiness follows integration, stability and resolution-sensitivity
evidence in P08/P10. This note does not launch new campaigns.

Start with the existing two hotspots, nearby control and a small representative
point set, keeping the coordinate map, equilibrium and current configuration
fixed. Prefer exact source-coil field evaluation or a verified finer-toroidal
MAKEGRID. Compare B, first/required second derivatives, magnetic-projector
coefficients and the resulting operator/reference actions; keep source-grid
resolution, interpolation choice and simulation-grid resolution separate.
Use divergence and stored-vector-potential consistency as supporting checks,
not independent truth. Do not select an interpolant because it minimizes MMS
error: quintic interpolation worsened the present coarse-spacing hotspots.
Assess geometry/reference sensitivity relative to the intended simulation's
spatial errors and required accuracy; do not infer its impact solely from this
nested manufactured-field test.

Only the original MAKEGRID was found in the bounded local inventory; independent
finer-source validation therefore awaits a matching upstream input. Same-file
interpolant comparisons remain diagnostics and cannot replace that validation.
The fitted coordinate map has not been independently certified by these tests.
**Status (2 October 2026):**
- The P08 N64 recheck is closed: autodiff K and the autodiff reference divergence removed the finite-difference derivative errors, and `compact_c3` replaced the nonlocal toroidal spline.
- Independent finer-source certification remains open and non-blocking.

If no improved input becomes available, retain the uncertainty explicitly;
simulation refinement alone is not independent geometry certification, but the
missing input does not block P08/P10 integration and evolution work.

### Near-wall toroidal geometry resolution (coil ripple) — 27 September 2026

**Finding:** the HSX geometry the solver consumes contains toroidal structure
that the canonical N32/N48/N64 grids do not resolve near the wall. For
flux-form operators (P07, P07N, and P06's face corrections), this limits
midpoint-target accuracy of fields with near-wall content. It is a property of
geometry resolution, separate from the solver's own error. Evidence is in the
[compatible-field screen report](../../../../work/p07n_compatible_fields_20260927/report.md)
(addenda 1–3) and its JSON artifacts.

- **Source: modular-coil ripple.** HSX has 48 modular coils (12 per field period), so the field carries toroidal mode n = 48, mixed with the 4-period shaping into sidebands 48 ± 4j. On the unit field's logical radial component b^u, which is 0.2–0.44 of b^η because the grid is not field-aligned, the n ≈ 48 band is among the top modes on every surface. Its energy grows about 6× from u = 0.5 to the wall.
- **Amplitude** ([ripple_amplitude.json](../../../../work/p07n_compatible_fields_20260927/ripple_amplitude.json)):

  | u | \|B\| ripple pk-pk / mean | Pitch b^u/b^η ripple vs shaping (rms) | η-derivative ratio ripple/shaping: \|B\| / pitch |
  |---|---|---|---|
  | 0.5 | 1.4% | 0.03 vs 0.20 | 0.41 / 1.2 |
  | 0.75 | 3.1% | 0.08 vs 0.24 | 0.65 / 1.9 |
  | 0.9 | 5.4% | 0.15 vs 0.29 | 0.95 / 2.5 |
  | 1.0 | 8.2% | 0.22 vs 0.35 | 1.33 / 2.9 |

  The ripple is modest in |B| values, but from about u ≈ 0.9 outward it dominates η-derivatives, which feed divergence terms, ∇|B| drifts, curvature and div b. It is therefore not a negligible perturbation for edge/SOL physics.
- **Resolution:** the eta Nyquist is n = 16/24/32 at N32/N48/N64, and n·h_η for n = 48 is 9.4/6.3/4.7. An unresolved mode's cell-average-minus-midpoint gap scales as `1 − sinc(n h/2)`, i.e. `1.21/1.00/0.70`, and barely shrinks. It mixes with the resolved O(h²) part and stalls the observed order as refinement proceeds. N, O and R share identical coefficients, so N−O is unaffected and O−R carries it.
- **Why the Dirichlet fields converge:** the shaping-dominated tensor components (T_uu, T_uθ, T_θθ) have eta-spectrum decay α = 3.1–5.5 even at the wall, which predicts asymptotic second order. Only the b^u-driven part of T_uη is near-flat (α ≈ 0.8–1.4). The accepted Dirichlet fields' u²(1−u²)⁴ envelope keeps them away from the ripple-strong outer band. An envelope sweep with the same angular field shows O−R orders falling monotonically from `1.79/2.07` to `1.30/1.80` as field content moves toward the wall, independent of the boundary-condition type.
- **Excluded causes:** B interpolation between the 1° MAKEGRID planes contributes ≤1e-3 of the ripple at the value level (≤ about 1% for first derivatives). The P05 higher-derivative interpolation sensitivity above remains a separate note. The references are self-consistent: O, N and R use the same fitted map and B. The reference numerical-step uncertainty is about 1e-8.
- **Not yet proven:** aliasing cannot separate ripple from high shaping harmonics, because 48 lies on the 4-period lattice. The only fully decisive test is a consistent ripple-filtered B used in both operator and reference.

**Visibility in midpoint MMS depends on the operator form — this is a test property, not an accuracy property.**
- **Volume (flux-form) operators, P07/P07N and P06's face terms:** the action is a cell average (the divergence theorem), but the midpoint reference samples one point. The cell spans about 1.5/1.0/0.75 ripple wavelengths (7.5° toroidally) at N32/N48/N64, so the average largely smooths the ripple while the midpoint picks up one phase of it, amplified by the reference's coefficient derivatives (n≈48). The midpoint MMS therefore **exposes** the unresolved geometry as O−R error.
- **Pointwise operators, P05's centered bracket and P06's q1 volume terms:** the operator and the reference evaluate identical coefficient functions at the same midpoints, so both alias the ripple at the same phase and the error cancels exactly. The [P06N exact-input screen](../../../../work/p06n_exact_input_screen_20260927/report.md) confirms this by code identity. Their midpoint MMS **hides** the limitation: N−R measures only the field reconstruction and closure.
- **Consequence:** P05/P05N/P06/P06N midpoint passes certify their reconstructions and Neumann closures. They do **not** show near-wall physical accuracy at N32–N64. Pointwise sampling aliases unresolved coefficient structure, and that would only show against an independent cell-averaged or better-resolved reference, or in evolved/physics comparisons. Physically both operator forms see the same unresolved geometry: pointwise operators alias it, volume operators partly filter it.

**Consequences and open decisions for P08/P10 (not scheduled by this note):**
1. Midpoint-accuracy statements for wall-active fields at N32–N64 must cite this limitation. The Dirichlet fine-interval orders (2.06–2.11) are optimistic for near-wall edge structure.
2. Resolving n = 48 to asymptotic second order needs n·h_η ≲ 1, i.e. N_η of order 300 in the outer region. Eta-only refinement would be sufficient for this effect.
3. The alternatives are to deliberately filter the ripple, which is a physics modelling assumption to justify separately and must be applied consistently in operator and reference, or to revisit how the perpendicular operators couple in η on a grid that is not field-aligned. Both are design decisions for P08/P10, not repairs to accepted operators.

## Next P milestone: shared extraction and replay before P08

P05 direct midpoint, P06 midpoint curvature and P07 midpoint-target diffusion
have completed their accepted static qualifications. The
[shared infrastructure design and extraction sequence](../../../../work/p_shared_infrastructure_design_20260926/design.md)
has completed [Phase A bounded point-row extraction and replay](../../../../work/p_shared_point_rows_extraction_20260926/report.md)
and [Phase B integrated-face/live-correction replay](../../../../work/p_shared_face_extraction_20260926/report.md)
at N32/N48/N64. These are not selectable production paths.

Share owner observations, basis/fit primitives, boundary loading, prepared-row
application, topology and eta distribution where their contracts agree. Preserve
P07 integrated-functional support separately from P05/P06 point rows, and keep
physical-volume and P06 evolution-measure reductions explicit. Recompute and
replay P05's saved `U-A` correction from live states before claiming runtime
coverage of the accepted direct-plus-jump candidate.

The bounded real-HSX extraction and JAX replay covers point and integrated
rows, ordinary/aggregate/core/transition/wall/seam faces, P05's live scalar
jump, and P06's separate characteristic side corrections and q1 midpoint
material/remainder closure. P07 integrated rows retain their own support
search and tensor convention; the P05/P06 correction kernels retain their
different signs and owner measures. The bounded receipts separate preparation,
lowering/cache, compilation and repeated application without a whole-grid
speed claim. Next qualify physical-normal Neumann data through the shared
boundary construction and the separate P05N, P06N and P07N operator gates below,
under the [27 September Neumann MMS field contract](#physical-normal-neumann-mms-field-contract--27-september-2026),
before combined P08 certification. Full
model/default integration remains at P10. Equivalent replay does not require a
new global static campaign. The older extraction applies only to its own
operator identity; Q need not finish before this work proceeds.

## Locked midpoint formulation and MMS contract — 25 September 2026

User-approved for the next P05/P06/P07 global campaigns. This supersedes
integrated-primary-reference and high-volume-quadrature requirements in older
campaign descriptions below; historical results and their original reference
identities remain unchanged. This is a frozen research campaign definition,
not a production or new convergence pass.

- Keep physical-raw-volume-weighted member-center owner observations and one
  evolved value per owner. Keep the structured/ringwise/coupled-axis and
  prescribed-Dirichlet reconstruction policies. No fine-cell evolved states.
- **Volume terms:** evaluate reconstructed fields, required gradients and the
  full intended coefficient products once at each raw-cell logical midpoint.
  Sum member contributions for agglomerated owners; do not replace a whole
  aggregate by one representative point. Freeze coefficient placement rather
  than silently replacing averages of products by products of averages.
- **Faces:** retain the existing shared q3 face flux/characteristic rules,
  canonical incidence and BC channels. Midpoint volume evaluation does not
  authorize midpoint replacement of face fluxes or pointwise replacement of
  conservative diffusion. P07 has no numerical cell-volume source to change.
- **Historical P05 face/cell campaign:** q1 cell correction with existing q3 A/B/C/U face construction;
  analytic strong bracket at each raw midpoint projected with stored physical
  raw volumes, including actual continuum vorticity differentiated independently
  of numerical reconstruction. Its derivative-step sensitivity is checked.
  This base assembly failed its midpoint qualification and is superseded by
  the accepted direct midpoint P05 definition above.
- **P06:** q1 complete material/remainder volume expressions plus existing q3
  characteristic face correction. Member mass is `(J/B)(x_c)*Delta_xi_c` for
  both numerical assembly and analytic midpoint reference; divide only after
  owner summation. Recompute normalization of face numerators with that mass.
- **P07:** existing q3 integrated reconstructed face-gradient diffusion action
  divided by stored physical owner volume. Replace only the primary reference
  with the analytic pointwise diffusion expression projected from all raw
  midpoints using stored physical raw volumes.
- Global norms retain frozen physical owner-volume weights for all three
  campaigns. These reporting weights, stored observations and P06 evolution
  masses are separately labeled. Numerical references use analytic fields and
  derivatives, not numerical reconstruction or numerical jump corrections.
- Primary gate remains full-domain operator L2 order >=1.8 on both N32→N48
  and N48→N64 intervals per nontrivial field/component, with existing primary
  form masks. Preserve boundary, axis, transition and aggregate diagnostics.
  Midpoint-reference derivative/geometry uncertainty is screened independently
  on bounded complete owners. Midpoint-to-integrated comparisons and additional
  high-quadrature reference stages are deferred at the user's request to avoid
  their computational cost; they are not scheduled by these campaigns.
- No mandatory high-order reference integration over the full domain. The
  campaigns retain only bounded midpoint-reference half-step checks. Existing
  integrated audit artifacts remain historical evidence; do not recompute them.
  A midpoint pass does not certify exact continuous-average or evolved accuracy.
- Configuration/source identities change. Use a new campaign output folder;
  old integrated-reference chunks must not be relabeled or admitted through
  optimization-only continuation. Compatible numerical artifacts may later be
  imported only through an explicit validated migration. No automatic reuse
  importer or remote launch is authorized by these implementation edits.

The [bounded P06 midpoint test](../../../../work/p06_midpoint_volume_test_20260925/report.md)
found all 44 sampled nonzero component errors decreasing against the midpoint
reference, but only 36/44 descriptive sample slopes meeting 1.8 on both
intervals. Sampled integrated-reference differences can increase. These are
historical bounded findings, not a global pass. Their integrated comparisons
remain archived; the next campaigns do not recompute them.


The [midpoint campaign implementation and bounded checks](../../../../work/p_midpoint_campaign_update_20260925/report.md)
record the P05/P06/P07 edits, 14 passing focused tests, and actual-HSX preflight/
reference checks. P06 passes the midpoint global qualification below.
The returned P05/P07 midpoint campaigns have now been analyzed. P07 is accepted
by user decision as an observed second-order static midpoint pass. The initial
P05 face/cell candidate failed; the subsequent direct midpoint candidate is
qualified as recorded above. Original gate outcomes and diagnostics are retained.

### P05 volume-normalization audit — 26 September 2026

The [bounded volume-measure audit](../../../../work/p05_volume_measure_audit_20260926/report.md)
confirms that canonical raw volumes use q2 Jacobian integration (eight points
per raw cell), rather than midpoint volume. With the campaign's fixed N64
continuous metric, sampled N64 volumes replay within `2.64e-13` relative.
N32/N48 use different producer metric checkpoints; bounded cross-metric q2
differences are `1.64e-3`/`3.46e-4`, not an own-producer replay failure.
Stored-integrated versus stored-midpoint volume relative RMS has global orders
`1.945/1.975`; six fixed-center geometry probes approach order two.

A diagnostic rescaling of saved centered exact-input numerators by midpoint
owner volume does **not** restore convergence: six smooth-case fine orders
become `0.657–1.077`, compared with `1.125–1.373` originally. The normalization
change itself has fine orders `1.936–1.973`. References, anchors, reconstruction
moments and reporting weights were held fixed; this is not a fully rebuilt
alternative operator. No new global computation, integrated MMS reference,
traces or reconstruction were run. Keep canonical normalization and the locked
point reference. At this audit stage the face/cell P05 candidate remained
unqualified; the direct candidate's later acceptance is recorded above.
The then-next investigation isolated the remaining exact-input
face/cell error at fixed geometry centers and decreasing cell widths, with
properly centered coordinate decomposition, before considering a new campaign.
P06 and P07 acceptance remain unchanged.

### P07 midpoint observed second-order pass accepted — 25 September 2026

**User decision:** record job `58882298` as a **passed observed approximately
second-order global static midpoint MMS milestone**. The user accepts the
small N32→N48 shortfall (orders `1.750–1.816`) in light of N48→N64 orders
`2.059–2.107`, passing operational and bounded reference checks, decreasing
regional RMS/max errors, and the independently verified error decomposition.
The [acceptance record](../../../../work/p07_validated_58882298/local_analysis/acceptance_decision.md)
is separate from the original audit and raw results. Preserve
`global_order_pass=false` and `reference_qualified_by_bounded_checks=true`
in the returned artifacts. This is an explicit research acceptance, not a
retroactive statement that every frozen 1.8 interval gate passed.

Retain the q3 conservative diffusion assembly and locked midpoint target.
Reconstruction-to-exact-face error remains above third order; the dominant
exact-face-minus-midpoint contribution gives roughly second-order fine-grid
behavior. The acceptance does not require another P07 consistency campaign
before proceeding. The separate contributions from averaging, geometry,
quadrature and normalization remain an optional research question, not proven
individually by this result. P05, Q, evolved stability/accuracy, energy and
production integration remain separate milestones; none is promoted here.

### P05/P07 midpoint returns: consistency work remains — 25 September 2026

The [P05 job 58880191 audit](../../../../work/p05_failed_58880191/local_analysis/report.md)
verifies all 55,962 scientific chunks from commit `2458dbf6`. All computational
stages completed, but reduction stopped on a bounded/global actual-vorticity
reference mismatch. The largest sampled discrepancy is `1.53e-8`, negligible
against spatial error; a local three-owner probe reproduces batch-sensitive
nested-derivative differences. Preserve the original failed status and repair
the replay comparison with an explicit noise bound or aligned evaluations;
the saved numerical work need not be rerun. Independent diagnostic reduction
also finds a genuine accuracy non-pass: only actual-vorticity A/B/C pass both
intervals (3/27 primary entries). Other cases have fine orders `1.118–1.397`.
Reference step sensitivity is at most `0.08182%` of matching sample error,
and constants/support checks pass. Thus repairing the reducer alone will not
qualify P05. For six smooth nonconstant cases, saved exact-input assembly
already reproduces the poor midpoint orders, while the centered reconstruction
contribution has fine orders `3.04–3.85`. The actual-vorticity swapped oracle
remains unavailable; do not use its B/C exact-input diagnostics.

The [P07 job 58882298 audit](../../../../work/p07_validated_58882298/local_analysis/report.md)
verifies 15,665 scientific chunks and exactly reassembles numerical and
exact-gradient face actions and midpoint references. Operational validation
and bounded reference checks pass, but the frozen global order gate does not:
phi `1.778/2.096`, Ti `1.766/2.107`, regular `1.750/2.080`, and mixed eta
`1.816/2.059`. All regional RMS/max errors decrease; wall fine slopes are only
`0.566–0.927`. The reconstruction-versus-exact-q3-face contribution retains
orders above three on both intervals, while exact-face-minus-midpoint error
dominates the new target comparison. This is approximately second-order
fine-grid behavior. The user-accepted pass above supersedes its status as a
roadmap blocker; the original strict-gate non-pass and lack of a midpoint
third-order claim remain unchanged.

For the then-unqualified face/cell P05 path, the next investigation addressed the exact-input assembled
operator's consistency with the locked midpoint target, including
geometry/coefficient variation, face/cell rules
and normalization, using bounded saved-data diagnostics before donor redesign
or another global campaign. The related Q finding motivates but does not
prove the same mechanism for P. Preserve historical integrated-reference
passes and all returned flags; no automatic target change, gate relaxation,
production promotion, checkpoint migration or remote launch is authorized.

### P06 midpoint static qualification passed — 25 September 2026

The [returned job 58880303 and independent local audit](../../../../work/p06_completed_58880303/local_analysis/report.md)
verify the source content of commit `2458dbf6`, all 28,869 scientific NPZ
payload hashes, complete global coverage, and saved-action/reference reassembly
within `1.78e-15`. All 44 required nonzero primary M/R/total entries pass the
frozen 1.8 global order gate on both intervals: N32→N48 orders `3.010–3.747`,
N48→N64 orders `2.659–2.929`. This includes corrected MMS, held-out smooth,
homogeneous phi Dirichlet, and variable phi Dirichlet states, with nonzero
normal derivatives in both Dirichlet controls. Primary U density/Te/Ti and
centered vorticity remain unchanged; exact-zero vorticity remainder is exempt.

Regional RMS and maxima decrease on both intervals for every primary component,
including walls, axis, first ring, agglomeration and interfaces. Fine-interval
wall orders are `2.297–2.822`; first-ring orders are only `1.049–1.890`, retained
as a local diagnostic without rebound or a global failure. The largest bounded
midpoint-reference half-step sensitivity is `0.003651%` of the corresponding
sampled numerical error, below the 10% screen. This is a bounded reference
check, not a global uncertainty bound. All implementation invariants pass.

Retain q1 complete volume expressions, q3 characteristic faces, midpoint J/B
owner reduction, and independent analytic midpoint references. This qualifies
the static midpoint P06 research construction; it does not certify continuous
integrated-reference accuracy, asymptotic third order, evolved stability,
production integration, or the Q path. Historical integrated campaigns and
their flags remain unchanged. The P05 non-pass and user-accepted P07 midpoint
pass above remain separate from this P06 pass.

Approved research roadmap, 18 September 2026. This document specifies planned
work and acceptance gates, not a claim that the methods below are already
implemented or verified. It is the authoritative progress ledger for future
tasks. Keep numerical evidence and experiment logs in linked research artifacts.

## 1. Objective and acceptance contract

### P07 observed third-order static convergence accepted — 25 September 2026

**User decision:** accept **observed approximately third-order global static
MMS convergence** for the combined structured, BC-conditioned reconstruction
and proceed with the roadmap. Reference refinement is a nonblocking caveat for
this research acceptance, not a prerequisite for further development. This
decision supersedes older active boundary-redesign assignments below; it does
not close all P07 integration/evolution milestones or promote production use.

The [returned global campaign and independent audit](../../../../work/p07_combined_global_analysis_20260925/report.md)
at commit `5e2531fb5ae0ce263c0de78e483ed618881c771b` gives N32→N48 / N48→N64
global volume-weighted RMS orders:

| Field | Observed orders |
|---|---:|
| phi | 3.257 / 3.546 |
| Ti | 3.251 / 3.591 |
| regular | 3.099 / 3.338 |
| mixed eta | 3.098 / 3.327 |

Wall-layer RMS, global maxima, aggregate-interface and coupled/ringwise-join
norms decrease at both refinements. The previously accepted small axis-core
regular/mixed rebound remains diagnostic; first-ring norms decrease globally.
All 15,672 returned chunks were verified and saved flux reassembly reproduced
the global arrays exactly.

The [bounded reference audit](../../../../work/p07_bounded_reference_audit_20260925/report.md)
finds that q3 volume-reference underintegration inflated the highlighted N64
interior hotspots; refined surface estimates reduce those apparent errors by
about 11–33x without changing the numerical operator. Retain the limits: the
reported global orders use the original q3 reference, no corrected global
orders were computed, and the finest bounded strong-volume/surface comparison
still differs by up to 14.4% of local numerical error. The original campaign's
`global_order_pass=true` and `reference_qualified_by_bounded_checks=false`
remain unchanged. This is observed-order acceptance, not proof of asymptotic
third order or completed reference qualification. Numerical q3 face integration
and stored-volume normalization remain separate accuracy contributions.

The [acceptance record](../../../../work/p07_combined_global_analysis_20260925/acceptance_decision.md)
preserves this distinction. No reference rerun, new elliptic solve, energy
optimization or production change is launched by this decision. Historical
D_trace energy findings are evidence about that older operator, not a new
energy assessment of the combined structured candidate.

### Structured reconstruction and resolved-scale response — 24 September 2026

**Literature record requested by the user.** The current layered reconstruction
is an adaptation of established interpolation, functional recovery and boundary
reconstruction ideas, not a reproduction of one published turbulence scheme.
Its four-layer radial cubic, seven-point theta trigonometric interpolation,
four-plane eta cubic and prescribed-trace lifting need their own actual-HSX
qualification. Moment-functional recovery specifies how stored observations
approximate a target derivative or face integral; it does not specify a unique
filter or imply a turbulence closure. Single-member observations are raw-center
values; aggregated observations retain their weighted raw-midpoint contract,
not newly assumed exact continuous cell averages.

| Reference | Relevant result and limit of applicability |
|---|---|
| Mirzaei, Schaback & Dehghan (2012), *On generalized moving least squares and diffuse derivatives*, IMA J. Numer. Anal. 32, 983–1000. [Author manuscript](https://num.math.uni-goettingen.de/~schaback/research/papers/OGMLSaDD.pdf), [DOI](https://doi.org/10.1093/imanum/drr030). | Recovers target linear functionals from observations with reproduction constraints. Supports the functional framework; our structured interpolation is not this paper's particular least-squares algorithm. |
| McCorquodale, Dorr, Hittinger & Colella (2015), *High-order finite-volume methods for hyperbolic conservation laws on mapped multiblock grids*. [Author repository](https://escholarship.org/uc/item/7gs9t74k), [DOI](https://doi.org/10.1016/j.jcp.2015.01.006). | Fourth-order mapped-grid finite-volume construction and interface interpolation. Architectural precedent, not certification of our owner observations, anisotropic diffusion or wall closure. |
| Du & Li (2018), *A two-stage fourth order time-accurate discretization for Lax–Wendroff type flow solvers II. High order numerical boundary conditions*. [Preprint](https://arxiv.org/abs/1801.00990), [DOI](https://doi.org/10.1016/j.jcp.2018.05.002). | Boundary/interior interpolation must be consistent with the discretized equations. Their hyperbolic boundary algorithm differs from our Dirichlet trace lift; it is not our implementation. |
| Lele (1992), *Compact finite difference schemes with spectral-like resolution*. [NASA record](https://ntrs.nasa.gov/citations/19930029704), [DOI](https://doi.org/10.1016/0021-9991(92)90324-R). | Wavelength-dependent derivative resolution complements formal order. These finite-difference schemes are benchmarks for analysis, not a proposed implicit compact solve. |
| Motheau & Wakefield (2021), *On the numerical accuracy in finite-volume methods to accurately capture turbulence in compressible flows*. [Preprint](https://arxiv.org/abs/2106.06585), [DOI](https://doi.org/10.1002/fld.5021). | Face reconstruction strongly affects turbulent spectra despite similar measured overall convergence; higher averaging quadrature alone did not improve spectral accuracy in their cases. Does not establish optimal quadrature or degree for HSX. |
| Denaro (2011), *What does Finite Volume-based implicit filtering really resolve in Large-Eddy Simulations?* [DOI](https://doi.org/10.1016/j.jcp.2011.02.011). | Interpolation, differentiation and integration determine effective filtering of resolved scales. Its continuous-cell-average filtering interpretation cannot be transferred unchanged to our raw-midpoint observations. |
| Ghosal (1996), *An Analysis of Numerical Errors in Large-Eddy Simulations of Turbulence*. [DOI](https://doi.org/10.1006/jcph.1996.0088). | Broad spectra and nonlinear interactions require error analysis beyond single smooth scales; relates numerical errors to nonlinear/subgrid terms. Fluid LES evidence, not a plasma closure prescription. |
| Kravchenko & Moin (1997), *On the Effect of Numerical Errors in Large Eddy Simulations of Turbulent Flows*. [DOI](https://doi.org/10.1006/jcph.1996.5597). | Aliasing, truncation and nonlinear operator form affect turbulent calculations even for finite-difference/spectral methods. Reconstruction consistency alone cannot settle evolved behavior. |
| Almgren, Aspden, Bell & Minion (2013), *On the Use of Higher-Order Projection Methods for Incompressible Turbulent Flow*. [DOI](https://doi.org/10.1137/110829386). | Fourth-order finite-volume evolution reduces the resolution needed for the tested turbulent cascade relative to second-order comparators. Positive suitability evidence, not authorization for a projection/elliptic method here. |

**Plasma-code comparison.** Giacomin et al., *The GBS code for the
self-consistent simulation of plasma turbulence and kinetic neutral dynamics in
the tokamak boundary* (2022), [preprint](https://arxiv.org/abs/2112.03573),
[DOI](https://doi.org/10.1016/j.jcp.2022.111294), describes fourth-order finite
differences/Arakawa brackets, adds perpendicular numerical diffusion for
stability (Sec. 2.1), and checks grid convergence of averaged profiles (Sec. 6).
Body et al., *Treatment of Advanced Divertor Configurations in the
Flux-Coordinate Independent turbulence code GRILLIX* (2020),
[preprint](https://arxiv.org/abs/1908.05398),
[DOI](https://doi.org/10.1002/ctpp.201900139), Sec. 2, explicitly identifies
interpolation-related numerical dissipation and the danger that fast parallel
dynamics can contaminate perpendicular dynamics; careful support-operator
construction is their mitigation. These published versions are not evidence
that finite differences or FCI avoid resolution/dissipation errors. Neither
paper certifies our HSX method or imposes its stabilization on this roadmap.

**Recommended sequencing, not a new run authorization:** finish and review the
[current all-orientation P test](../../../../work/p07_all_orientation_reconstruction_20260924/assignment.md)
before a full bounded HSX response comparison. Use its frozen complete maps
rather than certify the superseded radial-only mixture. Analytic uniform-stencil
symbols can inform test design now but cannot establish HSX effective resolution.
Preserve all passed P05/P06 and frozen Q baselines; an optional replacement
requires its own operator-specific evidence, not a retroactive change of status.

The next response study should freeze wavelength/direction/phase choices before
errors, compare old/radial-only/all-orientation actions on common actual-HSX
patches, and report amplitude, phase and absolute complete-operator errors for
waves or localized packets spanning nominally 12, 8, 6 and 4 local cells. Record
physical wavelength via the geometry, not only coordinate-grid counts; include
oblique directions, wall/transition locations and aggregate support when a legal
construction exists. Measure actual observation response separately from
reconstruction and assembled-operator response. Near exact-null targets report
absolute defects rather than misleading relative ratios. Account for all
recipients and qualify reference quadrature at each tested wavelength. Do not
call static attenuation a per-step damping rate: that requires the complete
evolution operator and timestep. Later nonlinear transport/spectral checks must
separate truncation, aliasing, intentional diffusion and time integration.
No universal cells-per-wavelength threshold, turbulence pass, new filter,
energy optimization, elliptic solve or extra global campaign follows from these
references. This documentation update launches no computation.

### Current P07 direction: boundary-conditioned accuracy reconstruction

**User decision, 24 September 2026:** boundary-cell error growth under refinement
must be investigated even when the global volume-weighted static order passes.
Preserve the D_trace global phi pass (`3.25999/1.91368`), while treating local
absolute accuracy and refinement behavior as explicit design objectives.
Moving selected cells are not fixed-location convergence tests; report their
motion, individual errors and patch/layer norms rather than hiding them in a
global average or claiming that all boundary cells diverge.
Do not accept a proposed wall closure on global order alone: systematic growth
of complete boundary error under controlled refinement remains an unresolved
accuracy defect, even when its global volume-weighted contribution is small.

The [literature reset](../../../../work/rlp_literature_reset_20260924/research_memo.md)
and [wall-only experiment](../../../../work/p07_wall_face_only_20260923/report.md)
motivate a change of research direction. The latter reduces quartic norms by
58–72% without changed neighbors or measured sensitivity growth, but actual
errors remain mixed. Separate Taylor-block minimization is not an adequate
accuracy criterion. Pause fifth-stage recovery and further conic repair scans.

The authorized [next P assignment](../../../../work/p07_boundary_conditioned_reconstruction_20260924/assignment.md)
compares one **compact reconstruction that uses prescribed Dirichlet trace
information directly** against original D_trace and the saved wall-only map.
Use hard boundary constraints or an explicit trace lifting, actual owner means,
and the complete metric/tensor face functional to reconstruct the unresolved
normal derivative. A possible representation is `u=g_ext+d*v`, with a declared
boundary-defining function and trace-data interface; this is a research option,
not a qualified implementation. A basis rotation or Gaussian reweighting of the
same broad footprint alone does not constitute the proposed change.

Retain arbitrary-cubic compatibility with its corresponding nonzero boundary
data, physical incidence and unchanged non-wall faces. Do not impose the MMS
fields' accidental zero normal flux as a generic Dirichlet condition. Additional
known trace samples must be declared as BC data, never analytic interior or
normal-derivative information. Freeze support/fit rules before candidate errors.
Evaluate normal/tangential probes, nonconstant trace and nonzero-flux controls,
held-out smooth fields, complete owner errors and a cheap controlled refinement
problem. Assess true footprint reduction and target observability, not only rank.

Preparation is local and reusable for fixed geometry/coefficients. No extra
global elliptic solve per timestep, global campaign, energy repair or production
promotion is authorized by this direction. The earlier energy defect remains a
separate eventual qualification issue. This current decision supersedes older
P07 next-step recommendations below; their historical evidence is preserved.

### Current result: centered and material-upwind static brackets globally qualified

The clean selection-v3 remote campaign at revision `c54b0552` is complete and
locally verified. The [returned-campaign analysis](../../../../work/p_centered_cubic_c54b0552_kFhdmt_analysis/report.md)
records all three fields passing the agreed real-HSX global operator gate:
vorticity orders `2.4414/2.4690`, regular scalar `3.5135/2.6455`, and eta-varying
scalar `3.6330/3.3759`. A and B pass separately as well as their centered C.
The global vorticity reference is the complete analytic IBP integral; bounded
reference budgets for all fields are 0.31–2.81% of the corresponding global
candidate error. The prior focused-interpolant failure is not used or relabeled.

Freeze the general cubic reconstruction, adaptive selection-v3 support,
shared face values/gradients, matched q3 face/volume integration, continuous
geometry queries, and frozen owner normalization as the centered-bracket
foundation. Midpoint assembly with the same reconstruction still fails the
smooth fields on the fine interval. Face-only q3 also passes these fields;
the volume correction improves eta accuracy but is not separately proven
necessary to reach the threshold. Preserve the complete identity without
field-specific tuning. Regional/max-norm behavior remains diagnostic.

**Milestone decision:** P03 is **passed as an HSX mechanism audit**; its old
"acceptance failed" label described the baseline operator, not failure to
complete the audit. P04 is **passed for numerical design and research
qualification**, demonstrated by the globally qualified centered-bracket
realization. These numerical dependencies are satisfied. Reusable payload
extraction and fixed-shape JAX application remain implementation follow-through;
they do not reopen the passed design gate or require another design campaign.
Other operators must still establish their own accuracy and applicable
structural properties when adopting these functionals.

This is the first complete three-field, two-interval global HSX pass for this
new centered-bracket foundation, with qualified reference budgets. Earlier
single-field, bounded, algebraic, and baseline successes remain valid within
their narrower scopes. The local recheck verified all 106 source files,
8,756 planned face/cell chunks, saved-array hashes, and recomputed weighted
errors and orders. The smallest centered order is 2.4414; even shifting errors
adversely by the empirical reference budgets leaves a minimum of 2.4086.

**P05 material static accuracy also passes.** The frozen nodewise cubic-jump
campaign at revision `257bd55f` is complete and independently verified in the
[returned material-campaign analysis](../../../../work/p05_material_257bd55f_qQnlSX_analysis/report.md).
The required regular-scalar orders are `3.4639/2.7951`; eta-varying scalar orders
are `3.6320/3.2729`. Their reference budgets are only `1.05–1.90%` of spatial
error. Diagnostic upwind vorticity also gives `2.4499/2.3266`; retain the
qualified centered operator for vorticity. All 4,839 new face chunks, frozen
donor identities and central fluxes were verified; independent correction
assembly agrees to `1.11e-16`.

Freeze the recentered nodewise jump, bias `0.75`, selection-v3 supports, q3
nodes, unchanged common value/generator and matched volume contribution, and
existing physical-wall trace treatment. At N64 the material errors are only
`2.38%/0.71%` above centered material A for the two scalars; the correction is
nonzero and its own global RMS decays with orders above two on both intervals.
This demonstrates preserved accuracy, not a requirement that upwinding improve
smooth-field errors. Fine-interval maximum errors flatten and some RLP regions
carry growing error fractions; these are diagnostics, not new acceptance gates.

**Status boundary:** P03/P04 numerical dependencies and both P05 static bracket
accuracy milestones are passed. Shared implementation, structural properties,
curvature, diffusion/polarization, coupled/evolved MMS and production promotion
remain separate work. Do not relabel material static accuracy as pending while
those follow-through items are open. No repeated bracket global campaign,
higher degree, bias scan or new donor search is justified by these results.

### Integration sequencing: consolidate first, integrate at the end

**User decision, 22 September 2026:** full shared-model integration belongs at
the end of this roadmap, in P10. A qualified individual operator is not an
instruction to wire it into every driver or replace production defaults.
First establish the bracket, curvature, and diffusion/polarization designs in
P05–P07; then consolidate the components they demonstrably share before P08's
coupled verification. Reuse the existing extracted bracket machinery while
learning the other operators' requirements instead of implementing separate
copies of donor selection, moments, geometry evaluation, incidence, boundary
classification, preparation, or runtime application for each operator.

Early package extraction, saved-action replay, and small opt-in adapters in
existing diagnostic harnesses remain allowed when needed for the authorized
experiment. These are provisional implementation and verification steps, not
full integration or a reason to freeze the final shared API prematurely.
P08 and P10 may assemble the candidate operators in the verification harness
to perform their required tests; deployment into the shared model/default
workflow follows the final gates. Blob-driver synchronization stays deferred.

Before P08, inventory common versus operator-specific requirements and extract
one reusable preparation/application layer wherever the numerical contracts
agree. Retain distinct physical fluxes, measures, volume/anchor corrections,
boundary data, and stabilization where they differ. Check for reuse with the
[parallel roadmap](parallel_second_order_roadmap.md), but do not force P and Q
to use the same reconstruction or delay either for speculative unification.
Replay each affected action against its saved qualified evidence after
consolidation. An implementation-equivalent refactor does not reopen numerical
design or require another global campaign; a changed numerical action needs
its own scoped qualification.

### Recommended next work: reuse the frozen bracket infrastructure

The completed extraction-and-replay implementation contract and corrected
matching-error denominator audit are recorded in
[P05 bracket extraction and replay](../../../../work/p_wip_archive_20261004/g2_face_flux_bracket/src/drbx/dev_docs/p05_bracket_extraction_replay.md). A
self-contained N32 complete-owner fixture now covers resident eager/JIT/JVP and
changed-field reuse without the campaign artifacts. This closes P05's saved-
action replay follow-through without reopening either static accuracy milestone;
structural/evolved qualification and production integration remain open.

These are bounded follow-up scopes for future assignment, not an instruction
to launch another campaign or promote production behavior now.

1. **P05 implementation follow-through: extract and replay.** Extract the
   minimum reusable host-side geometry/reconstruction payload from the frozen
   successful campaign: owner observation moments and measures; selection-v3
   supports and expansion policy; shared face incidence, orientation and q3
   nodes; central value/derivative and biased side-fit functionals; and the
   matching volume/anchor and physical-boundary data. Evaluate needed geometry
   continuously at its actual quadrature locations. Preserve the current
   midpoint/raw-volume observation convention and evolved/restart layouts.
   Keep geometry/setup work separate from fixed-shape JAX operator application;
   no per-step donor search or SVD. Factor repeated geometry work across fields
   rather than importing campaign/MMS fields into the reusable operator.
2. **Replay the actual extracted actions against saved evidence.** Start with
   a complete N32 HSX action and bounded N48/N64 owner sets covering axis,
   agglomerated bulk, size changes, ordinary interior, walls and periodic seams.
   Replay centered A/B/C and the recentered material correction, including the
   complete volume and boundary terms. Reuse archived reference and baseline
   data. Use justified floating-point tolerances below recorded spatial error;
   bitwise equality is not required. Cached full-array replay is welcome when
   cheap; do not regenerate expensive global geometry/references for this
   engineering check. Record smooth-path JIT/JVP and applicable sharding checks
   during integration. A mismatch requires localization, not retuning the fit.
3. **Bounded material scientific follow-through.** Keep the fit/support/bias
   fixed. Predeclare one held-out smooth positive thermodynamic field on real
   HSX, exercise the actual action and inspect dissipative work and positivity
   of a justified one-step update separately from forced MMS. Account for
   boundary work and the actual material/volume identity; do not infer energy
   stability from the word "upwind" or conservation alone. Record limiter/floor
   activation without introducing a new limiter as part of extraction. These
   checks inform subsequent structural/evolved certification; they neither
   reopen the static pass nor block starting P06/P07 audits. Any numerical
   repair receives its own bounded design and authorization.
4. **P07 combined structured static accuracy accepted, 25 September 2026.**
   The user accepts observed approximately third-order global MMS convergence
   with the bounded reference caveat retained as nonblocking. Preserve the
   combined candidate and the older D_trace baseline separately. Advance the
   remaining integration/evolution roadmap when tasked; do not restart the
   superseded boundary-redesign assignment or infer production readiness.
   Elliptic and energy work remain deferred.

Reuse the extracted bracket implementation and the P06 boundary-functional
machinery for P07 where their numerical contracts agree. Do not insert full bracket-driver/model
integration as the next milestone. Consolidate common machinery before the
combined P08 RHS; complete full integration only at the end of P10 after
evolved certification. Blob-driver synchronization stays deferred.

### Preserved execution contract for the completed remote campaigns

The earlier local P global campaign is paused and preserved as historical
evidence. The matched-q3 centered-bracket N32/N48/N64 qualification was run
as a **new remote campaign**, using
`scripts/hsx_remote_qualification` (removed; see commit `c0269a95`).
The remote handoff contains repository commands only, in
`scripts/hsx_remote_qualification/REMOTE_COMMANDS.md` (removed; see commit `c0269a95`).
The material extension used the separate
P05 runner (`scripts/p05_material_campaign`, removed; see commit `257bd55f`) and reused the
hash-pinned completed centered campaign. Both computations are complete.
The remote setup skill chooses concurrency to fit the active allocation and
worker memory; the campaign requires an explicit worker count. **No remote
scaling study is required**. The
remote task owns execution and monitoring; the local P task remains paused.
Workers do not send unsolicited messages to the parent or other tasks; the
parent inspects progress when requested.

The new `selection-v3` policy resolves near-equal candidate distances by a
documented anchored tolerance group and canonical owner ID, and applies one
half-open snapping convention to **all eight** angular-sector boundaries.
Do not special-case pi/4 or tune ties to reproduce historical donor IDs.
Rebuild the actual downstream dependency chain under this policy, including
physical-boundary cubic derivative rows and all face/cell reconstructions.
The global study does not consume historical bounded-fit, factor, cross, or
global-derivative caches. Geometry, original manufactured owner data, and
independent qualified references may be reused after content verification.
Historical N32/N48/N64 numerical outputs must not be mixed into the new
campaign, including already completed local global chunks.

Fresh HSX implementation checks remain required; replaying every archived
donor ID is **not** a prerequisite for a new campaign. Record donor identities,
polynomial reproduction, constants, and action diagnostics. Floating-point
agreement uses justified tolerances; a policy change receives a new numerical
identity rather than bypassing a stale-cache check. Candidate-specific
reference budgets and the existing two-interval >=1.8 global operator gate
remain unchanged. Remote execution alone is not numerical qualification or
promotion; the successful scientific results establish the milestone decision
above.

The remote performance repair removes persistent query-basis growth, shares
metric evaluations, microbatches metric/curl queries, groups owner memberships
once, isolates preparation stages, and validates resumed chunks before worker
startup. See the campaign implementation validation (`scripts/hsx_remote_qualification/VALIDATION.md`, removed; see commit `c0269a95`).
It changes execution/source identity, not the frozen candidate or acceptance
contract. Start a fresh output folder for this release. Per-worker RSS and cache
telemetry are recorded; full-node runtime and memory remain remotely measured
quantities. The returned successful attempt took 75m49s, including 47m52s
preflight and 26m42s global computation with 48 workers. Chunk worker high-water
RSS stayed below 0.881 GiB; these process samples do not measure whole-node
peak memory or production-timestep cost. Do not infer production performance
from the research campaign.

### Scientific contract

Retain the RLP owner unknowns and regular chart, and establish second-order
convergence for the **current MMS configuration**:

- Material scalar upwind brackets and the centered vorticity bracket.
- Complete coupled curvature, including the potential remainder.
- Conservative perpendicular polarization and the shared evolved perpendicular
  diffusion action.

Other selectable formulations remain diagnostic controls. Parallel operators
have a separate [Q00–Q09 roadmap](parallel_second_order_roadmap.md).
Blob-driver synchronization remains outside both roadmaps.

**Pinned boundary baseline for both roadmaps: the existing frozen MMS
configuration**, `physical_wall_model="legacy-velocity-trace"` with
`parallel_velocity_wall_bc="neumann"`. This supersedes the interim no-flow
and rung-2 choices. The legacy compatibility model supplies owner-extrapolated
ion/electron velocity traces and thermodynamic Neumann conditions, with the
existing potential/vorticity Dirichlet conditions. It is neither the rung-1
no-flow wall nor the rung-2 conducting sheath. Preserve
`neumann_ghost_scheme="physical"`,
`parallel_boundary_pairing="characteristic-sat"`, and
`parallel_characteristic_wall_law="energy-absorbing"` from the frozen MMS.
The numerical characteristic wall law does not turn the legacy trace model
into a conducting sheath.

The source of truth is `simulate_hsx_mms.py`'s `_production_configuration`,
`_production_selector_args`, and runtime assertions, checked against the
32/48/64 frozen-run manifests in the
[HSX RHS summary](../../../work/perpendicular_second_order_hsx_p01_p03/perpendicular_rhs_summary.json).
`native/fci_physical_wall.py` resolves the selected physical traces. Keep
periodic eta, actual HSX wall/termination geometry, and existing axis topology.
Physical sheath-law certification or migration is outside this P/Q scope.

Choose manufactured fields and independent sources/boundary pairing compatible
with this fixed configuration. Explicitly record each operator's effective
scalar/diffusive flux, normal-derivative, polarization, and terminated-leg
closure: the physical-wall selector alone does not specify every numerical
boundary functional or justify dropping wall contributions. Preserve existing
operator-specific support rules. Bounded tests with other manufactured boundary
loads remain labelled diagnostics. Include selectors, effective closures, and
boundary-contract identity in provenance/cache keys. Preserve historical
results; no boundary rerun is required solely to repair the earlier rung label.
Check actual closure differences before certification reuse. This pins the
research configuration, not production or blob-driver defaults.

**Real HSX geometry is the acceptance basis from the first numerical audit.**
Use the actual metric, field-line traces where applicable, angular RLP topology,
axis treatment, and domain boundaries. Idealized maps, slabs, and simple polar
geometries are not prerequisite convergence gates. Small algebra/bookkeeping
tests may remain, but their success does not establish HSX accuracy or complete
an operator milestone. Analytical manufactured fields are still appropriate:
evaluate them and their independent continuum sources on the real HSX geometry.

**HSX-only numerical test policy:** every new or revised numerical accuracy,
convergence, reconstruction, interface, return-map, or boundary test for this
roadmap must target actual HSX geometry. Do not add or run idealized slab,
straight-field, circular-polar, or synthetic-map campaigns as substitute or
preparatory numerical evidence. Cheap regressions may use a bounded extraction
of a real HSX artifact, retaining its metric, topology, measures, traces where
applicable, and boundary semantics; record the parent artifact and selection.
Such a local check diagnoses a mechanism but cannot certify global order.
Geometry-independent unit tests may check algebra, indexing, hashes, and norm
bookkeeping only. Preserve historical idealized results as background without
rerunning them to satisfy milestones. Missing HSX inputs must be reported or
produced through the explicit geometry producer, never replaced by an idealized
fixture. This policy applies to P01–P10, including tests added during repairs.

Begin numerical audits with the existing **32³/48³/64³** HSX artifacts. Keep one
continuous geometry reference across the sequence. Nonmonotone results remain
inconclusive; extending resolution is an explicitly scoped follow-up. Diagnose
failures with regional residuals and controlled changes on the same HSX geometry.

**The primary fast verification gate is second-order global operator convergence.** Retain the fast
midpoint-based MMS as the default convergence check, using independently derived
continuum sources and explicitly documented sampling/averaging conventions.
Use bounded actual-HSX average controls to qualify representation consistency; do not
require high-order quadrature over the full HSX domain. Require observed
global volume-weighted operator-error L2 order **at least 1.8 on both finest
refinement intervals**, assessed per certified operator and nontrivial field/case
rather than hidden in an average across fields. Retain independent elliptic and
fixed-physical-time solution MMS checks with the same global order criterion.

Second-order operator consistency is a stronger verification target than
second-order solution convergence alone; the two are not generally equivalent.
Good solution convergence does not automatically waive a failed global operator
gate. Document such a result and its mechanism for review rather than silently
relaxing the agreed requirement. Local truncation, face, and regional error
orders, including maximum-norm orders, remain diagnostics, not separate second-order gates. Required stability,
conservation, boundary, and algebraic properties remain part of certification.
This operator-plus-solution contract supersedes the interim solution-only rule.

Always report axis, transition, ordinary-interior, and physical-wall errors
separately: normalized RMS, maximum error, physical volume, and contribution to
global squared error. These regional norms need not individually achieve second
order under the selected acceptance contract; their behavior must remain visible.

Geometry-reference and, where applicable, temporal and linear-solve errors must
each be demonstrated to contribute less than **10% of the finest spatial error
in the corresponding operator or solution check**. When an optional
quadrature-qualified reference audit is used, apply the same bound to its
quadrature error. Routine midpoint sampling error may be part of the overall
second-order error budget; it need not satisfy this 10% bound. Qualify its order
with bounded controls on selected actual HSX cells, particularly near the axis and
agglomeration seams. Do not infer order from errors at an independently
established numerical floor.

Use stable task IDs below. Each task records: status, dependency, hypothesis,
implementation revision, configuration, evidence links, measured orders, and
remaining failures.

### Donor coverage and conditioning (shared P/Q lesson)

Both paths have exposed reconstruction failures caused by insufficient or
poorly distributed donors. Donor count alone is not the criterion: nearest
owners can cluster along too few angular columns or eta planes, leaving the
target derivative weakly determined. A full-rank solve and accurate polynomial
reproduction can still produce large coefficients and inaccurate operator
actions. This is a support-design issue to check before rejecting the
reconstruction family or increasing polynomial degree.

- Preflight the actual HSX owner-observation matrix with centered/scaled
  coordinates and rank-revealing solves. Preserve the actual stored owner
  functional; replacing averages by representative point values is a separate
  consistency error that better conditioning cannot repair.
- Inspect directional coverage, donor count, support radius, scaled condition
  numbers, and coefficient amplification. For derivative rows report
  mesh-scaled amplification with explicit coordinate/component conventions;
  regular Cartesian y is not logical theta. Condition numbers depend on
  scaling and basis, so compare them under a stated common convention.
- Do not stop support expansion solely because rank and reproduction pass.
  Compare a modest increase or directionally balanced support on cached HSX
  faces, holding degree, observation functional, and assembly fixed. Select
  support from geometry and algebra, not exact MMS values or errors. Prefer
  compact, adequately distributed donors: wider support can reduce
  amplification while increasing approximation bias and runtime.
- Rerun the complete bounded operator action after repairing support, including
  the relevant gradient/return stages and cancellation-sensitive contributions.
  Better conditioning is evidence of a repaired fit, not proof of operator
  accuracy. No universal condition-number cutoff, formal conditioning-to-error
  bound, or improvement at every sampled owner is an additional acceptance
  requirement. Global HSX operator convergence and the existing structural
  gates remain decisive.
- Record the support policy, scaling, expansion/fallback activity, cost, and
  comparison identity. Invalidate affected reconstruction caches when these
  change; retain independent geometry/reference caches and historical results.

Evidence from both paths supports this workflow. The parallel
[reconstruction audit](../../../../work/parallel_q03_state_refinement_patch_20260919/Q03_RECONSTRUCTION_IMPLEMENTATION_AUDIT.md)
reduced the maximum scaled condition from `6.86e5` to `38.9` with a
geometry/algebra-selected reconstruction using better donor support and
substantially reduced bounded propagated errors. The perpendicular
[supervisor support check](../../../../work/perpendicular_owner_face_supervisor_20260920/report.md)
found that 12 donors per eta plane occupied only two angular owner columns at
representative failing faces; 18 introduced a third, reducing condition numbers
from thousands to about five and sharply improving sampled vorticity
derivatives. These are bounded results, not universal donor counts or global
certification; the full perpendicular support rerun is pending.

### Efficient diagnostic execution (shared P/Q workflow)

Use the [supervise-long-runs skill](/Users/yxie/.codex/skills/supervise-long-runs/SKILL.md)
for reusable stage execution, recovery, and quiet adaptive supervision. The local
supervisor chains authorized stages immediately; the heartbeat provides oversight.
Default heartbeat wakeup is **10 minutes**, adapting to **20 or 30 minutes** for
measured longer stages, with a **30-minute maximum** unless the user changes it.
Scientific scope and acceptance requirements remain in this roadmap.

**Task handoff design requirement:** new bounded assignments must specify the
mathematical actions/functionals, coordinate and component conventions, fixed
inputs/support, independently checked quantities, stage dependencies, acceptance
and stopping decisions, and exact scope of any resulting claim. Require a small
preflight through the actual HSX path before scaling. A prerequisite failure must
not block an independent diagnostic without a demonstrated dependency. Do not
replace shared multi-owner assembly with a one-row balance identity and label it
equivalent. Record deviations explicitly before interpreting their results.

**Implementation flexibility:** fix the scientific question, representation,
boundary contract and acceptance criteria; treat suggested donor counts, patch
sizes and solver choices as starting points unless they define the comparison
itself. Workers may expand local support, improve scaling/rank detection, change
equivalent solvers, repair indexing, and make modest evidence-driven scope
adjustments without another authorization. Formal rank/compatibility is not a
usable-stencil guarantee: failed reproduction or excessive numerical amplification
justifies adapting the support. Preflight failures before expensive rebuilds,
record changes and preserve comparison identities. Report a blocker only when
the scientific requirement or a real resource limit prevents progress, not merely
because an initial implementation recipe failed. This does not authorize global
campaigns, production promotion, or weakening accuracy/structural requirements.

Apply these practices to both roadmaps without weakening the HSX accuracy,
reference-independence, complete-support, or structural acceptance requirements.

- **Scope and time the work before scaling.** State the hypothesis, decision to
  be made, actual owner/face/donor support, reference-point count, and proposed
  resolutions. Run a tiny actual-HSX preflight through the intended coordinate,
  boundary, sampling, and reporting path. Time one representative batch and
  separate setup/compilation, reference evaluation, operator action, and output
  costs. Use these measurements for a provisional runtime/memory estimate;
  refine it from progress. Do not add a separate benchmark campaign or require
  user approval for routine runs already within the assigned scope.
- **Compute only the reference quantities needed.** Prefer narrow Jacobian,
  face-velocity, vorticity, or gradient evaluations over preparation of unrelated
  RHS terms. Batch and deduplicate shared points; reuse metric evaluations at
  common differentiation locations. Avoid computing unused Hessians and
  repeating whole-grid preparation for each bounded candidate. Preserve the
  independent continuum definition and verify optimized paths on bounded HSX
  samples before relying on them. Profile before a broad performance rewrite.
- **Reuse qualified data, not stale conclusions.** Cache geometry/reference
  samples and complete bounded support across candidate comparisons. Key reuse
  by geometry and relevant implementation hashes, field/source and time,
  coordinates/selection/weights, boundary contract, precision, reference
  controls, and schema; key certification additionally by requested checks and
  acceptance rules. Record dependencies and invalidate affected results when
  they change. Preserve historical evidence and independent reference checks.
- **Qualify expensive controls economically.** Midpoint MMS remains the default.
  Use bounded, stratified actual-HSX quadrature, differentiation, or tracing
  refinement to estimate uncertainty in the corresponding completed operator
  residual. Fixed choices such as 128 tracing substeps or full-domain high-order
  quadrature are not requirements in themselves. Use cheaper qualified controls
  when evidence supports them; do not infer sufficiency from endpoint shifts
  alone, truncate necessary support, or weaken the agreed error budget.
- **Checkpoint and chain authorized stages automatically.** Use persistent
  stdout/stderr, stage timings/progress counts, atomic versioned checkpoints,
  process/launch identity, and captured exit status. A launcher should run the
  next authorized chunk or stage immediately after successful completion,
  including final assembly/analysis where already specified. A successful
  chunk exit is not completion of the whole experiment. Resume only missing
  valid stages and guard against duplicate launches. Stop the chain on a real
  failure or a numerical decision requiring review. Do not use an agent
  heartbeat as a scheduler for each small chunk.
- **Match monitoring to expected duration to save tokens.** For short runs,
  inspect completion directly with bounded tool waits instead of creating a
  recurring monitor or repeatedly polling. For runs expected to outlast an
  active turn, use a quiet long-interval heartbeat (normally the user's
  10-minute default, adapting up to 30 minutes), informed by measured runtime. Reuse/update one existing
  monitor per task; do not create duplicates or shorten the interval merely
  to dispatch the next chunk. Stay quiet on unchanged healthy progress;
  resume on completion, actionable failure, or a required decision. Pause or
  remove obsolete monitors when their work finishes. Monitoring cadence must
  not insert idle gaps between authorized computations.
- **Recover failures before repeating work.** Check exit code and logs, validate
  stage inputs, and fix the identified problem with a small HSX preflight.
  SIGKILL alone does not prove memory exhaustion. Preserve useful checkpoints,
  record unknown causes explicitly, and confirm a restarted process is alive
  before reporting it running. Do not interrupt a healthy expensive stage just
  to retrofit logging or optimization; apply changes at a safe stage boundary.
- **Report scientific progress and cost separately.** Summarize completed and
  remaining stages, measured errors/orders, unresolved mechanisms, compute
  versus idle time, and a provisional ETA when useful. A local regression is
  diagnostic, not a requirement for every region to improve. Keep the next
  experiment bounded by a specific decision; do not expand a sampled audit
  into a global campaign without the scope already being assigned.

## 2. First audits: establish where order is lost

### P00 — Freeze the baseline and create the progress ledger

**Dependencies:** none.

- Maintain this durable roadmap and task ledger. Store numerical evidence and
  experiment logs in purpose-named research artifacts linked from the ledger.
- Record the exact MMS selections, enabled reconstruction payloads, source
  revision and dirty-file hashes, geometry artifacts, precision, boundary
  conditions, and sharding.
- Reuse the existing operator-only bracket audit and MMS term ledger; extend
  their diagnostics rather than creating a competing full simulation driver.
- Preserve baseline results before changing numerical formulas. Run one
  resolution per process to avoid accumulating geometry and JAX allocations.

**Deliverable:** reproducible baseline manifest and a results table with separate
bracket, curvature, diffusion, and polarization entries. Existing literature/source
audits count as background evidence, not completed numerical verification.

### P01 — Qualify manufactured fields, averages, sources, and norms

**Dependencies:** P00.

- Keep the existing midpoint-only full-HSX MMS as the fast default. Second-order
  convergence alone does not require higher-order quadrature everywhere.
- Qualify **physical owner-average** controls, boundary data, and continuum
  source averages on selected actual HSX cells/owners. Reuse qualified physical
  moments where possible; distinguish exact identities from approximate
  geometry moments. Use bounded optional quadrature on those cells/regions
  when necessary and cache reusable results. Idealized averaging fixtures are
  unit controls only, not reference qualification for HSX.
- Label midpoint and exact-average reference conventions explicitly. Use the
  latter to isolate representation/differentiation defects, not to replace every
  fast convergence test or make full-domain quadrature a setup dependency.
- Use smooth regular-chart fields containing linear, quadratic, cubic, mixed,
  and non-polynomial components. Include x and y, not only quadratic angular
  harmonics.
- Supply boundary data compatible with each manufactured field; for example,
  testing x with unrelated homogeneous boundary data would invalidate the
  diagnosis.
- Derive sources independently of the discrete operator being tested.
- Validate the regional masks, including overlapping diagnostic masks and a
  disjoint partition for error accounting.
- Fix the same continuum geometry reference across a refinement sequence and
  demonstrate its accuracy independently at every resolution with stratified
  samples covering axis, interfaces, interior, and boundaries.

**Deliverable:** reusable reference/measurement helpers and a qualified
manufactured-field catalogue.

### P02 — Audit reconstruction and differentiation stage by stage

**Dependencies:** P01.

Use the actual HSX geometry and automatic angular RLP topology as the primary
case. Where supported, use q=1 and axis-only owner topologies as controlled
diagnostics on the same continuous HSX geometry, with explicitly identified
topology-dependent artifacts. Do not substitute an idealized polar geometry
for the HSX stage audit or regenerate unrelated magnetic artifacts.

Measure separately:

1. Owner averages → reconstructed raw averages.
2. Owner averages → face values.
3. Owner averages → face derivatives.
4. Face data → integrated face contributions.
5. Integrated contributions → completed owner residuals.

Retain the existing analytical f=x first-face 8/9 result as supporting evidence.
Measure the corresponding average-to-derivative discrepancy and its propagation
through each complete operator on actual HSX cells; do not assume that the
idealized factor or its global effect carries over unchanged.

Test 1, x, y, x², xy, y², selected cubics, and smooth non-polynomial fields. For
brackets, include {x,y} and mixed-degree pairs using the implementation's sign
convention.

**Deliverable:** the earliest failing stage for each operator on HSX, including
whether a supported same-geometry control already fails without agglomeration.

### P03 — Audit geometry, interface coupling, return maps, and closures

**Dependencies:** P01; can proceed alongside P02.

**Status: passed — HSX mechanism audit complete.** The localized failures
justified the subsequent P04 repair; the clean remote result now confirms that
the repaired centered composition meets the global target. P03 is not held
open because the historical baseline failed convergence. Operator-specific
curvature, polarization, and runtime/sharding checks remain with their later
work packages rather than becoming new prerequisites to this audit closure.

**Completed HSX follow-up assignment (retained scope):** use the qualified continuous producer
metric/B reference to re-establish the unchanged production bracket baseline
on the existing 32³/48³/64³ HSX artifacts, and perform the bounded N32
matched-functional comparison below. This includes the necessary supplemental
P01 reference qualification and P02 stage localization. Update each milestone
only against its own evidence and deliver an explicit P04 readiness decision;
do not begin production P04 implementation in this follow-up.

1. Recompute midpoint-projected manufactured inputs and independent continuum
   bracket sources from the same continuous reference at every resolution.
   Cover manufactured vorticity and both existing smooth scalar controls with
   the same generator and frozen boundary contract. Report global physical-
   volume-weighted operator L2 errors and both interval orders, plus regional
   squared-error contributions. An exact common-face discrete oracle remains
   a stage diagnostic, not a substitute for the continuum operator reference.
   Qualify reference/differentiation uncertainty with bounded matched checks;
   retain midpoint MMS as the default and do not introduce full-domain
   high-order quadrature or new resolutions.
2. On the existing eight N32 owners and their complete face/donor support,
   compare the current geometric-moment candidate with a candidate whose
   polynomial observations use exactly the member sample locations and
   physical weights defining the stored midpoint-projected owner data.
   Verify that sampling convention from the implementation first. Keep owner
   values, reconstruction degree, donor-selection policy, production face
   velocity, center/compression term, assembly, and production eta
   reconstruction fixed so the changed observation functional is identifiable.
   Report any unavoidable support changes. Use low-degree reproduction through
   the actual observation functional as an algebraic check, then compare actual
   vorticity and both smooth fields on HSX, especially the non-axis transitions.
   Distinguish this MMS sampling diagnostic from a claim that evolved finite-
   volume averages are point samples. Existing bounded integrated-average
   results are a separate control, not justification for changing stored data.
3. Report face-state errors, signed radial/angular/eta residual changes,
   conditioning, weight amplification, fallback activity, complete support,
   and physical-volume-weighted sample errors. Preserve collapsed-face zero
   flux and retain axis-owner residuals. Update the detailed report, evidence
   identities, and ledger, then recommend whether a candidate warrants global
   qualification. Candidate global qualification is a subsequent scoped step;
   the production 32/48/64 baseline is authorized now.

**Diagnostic versus acceptance gate:** improvement at every sampled owner or
in every region is not required. Regional regressions require explanation and
error-budget reporting, not an automatic veto. The numerical acceptance gate
remains global physical-volume-weighted operator order >=1.8 on both finest
intervals for each nontrivial field, with the applicable structural and
reference requirements. Sample error reduction alone does not certify order.

- Perform the audit on actual HSX geometry from the outset. Use matched
  geometry/source artifacts and actual owner/interface regions for all
  counterfactual comparisons; preserve idealized results as supporting
  diagnostics only.
- Cross exact/production generator derivatives with exact/production transported
  traces to separate bracket errors.
- Compare nonlinear flux evaluated at common quadrature points with products of
  separately averaged quantities.
- Audit shared subface measures, coarse/fine flux summation, metric identities,
  and operator-specific mass factors.
- Separate curvature material updates, potential remainder, and radial/theta/eta
  contributions.
- Test R and H-star = M_owner^-1 H.T M_raw on exact smooth raw averages
  independently of the differentiated operator. Verify R H = I separately.
- Audit axis closure, periodic endpoints, physical Dirichlet and nonzero Neumann
  conditions, and eta-shard boundaries.
- Record reconstruction degree, stencil extent, conditioning, weight
  amplification, and positivity-fallback activation.

**Deliverable:** an HSX error-budget table identifying reconstruction, derivative,
geometry, quadrature, return-map, and boundary contributions.

**Audit gate:** do not begin a broad operator replacement until P02–P03 identify
a failing mechanism on actual HSX geometry and a bounded HSX reproducer. A failure already present on a same-HSX q=1 control
must not be attributed solely to agglomeration. Use stage-localized measurements
to identify what prevents the global operator target before committing to a
broad redesign. P02/P03 themselves remain audits; their completion does not
require operator repair or a passing global convergence result.

## 3. Literature-grounded implementation tasks

The primary approach is **moment-consistent face values and derivatives
evaluated with shared geometry and quadrature**. Keep the existing owner
representation; repair the operations applied to it.

| Literature precedent | Method to carry into this roadmap |
|---|---|
| [Purser, 1988](https://journals.ametsoc.org/view/journals/mwre/116/5/1520-0493_1988_116_1067_andnap_2_0_co_2.xml); [Lima–Peixoto, 2023](https://eartharxiv.org/repository/view/3192/) | Respect regular polar modes when reconstructing and differentiating across angular thinning. Retain the regular-chart basis and test its differentiated modes. |
| [Mignone, 2014](https://arxiv.org/html/1404.0537) | Derive interface quantities from geometric volume moments; distinguish cell averages, centroids, and point values. |
| [Fornax, Skinner et al., 2019, §VII](https://arxiv.org/html/1806.07390) | Coordinate restriction and transverse reconstruction across angular refinement interfaces. |
| [McCorquodale–Colella, 2011](https://msp.org/camcos/2011/6-1/camcos-v6-n1-p01-s.pdf) | Combine polynomial reconstruction, shared-face flux quadrature, and consistent time integration across refinement boundaries. |
| [Almquist–Wang–Werpers, 2019](https://arxiv.org/html/1806.01931) | Design value and derivative transfers jointly with quadrature and stability; adjoint compatibility alone does not preserve order. |
| [Barad–Colella, 2005](https://crd.lbl.gov/assets/pubs_presos/AMCS/ANAG/A158.pdf) | Assess elliptic solution convergence separately from interface truncation error. |

These papers establish applicable methods and precedents; none directly proves
convergence of the current DRBX composition.

### P04 — Build consistent owner-to-face functionals

**Dependencies:** P02–P03.

**Status: passed — numerical design and research qualification.** The general
cubic selection-v3 implementation meets the intended polynomial reproduction
checks on HSX and its matched-q3 centered action passes globally for all three
fields. Preserve this implementation as the qualified design baseline.
Extraction into reusable payloads and host-compiled/JAX application is remaining
engineering follow-through during adoption in P05–P07. Compare that extraction
against the saved qualified action; it is not a reason to reopen donor/degree
design or declare the present convergence result incomplete. This scoped pass
does not certify untested operators or production stability.

- Use the existing regular-chart moment machinery to evaluate **values and
  derivatives** at shared face quadrature points from the actual stored owner
  observation functional. The current candidate uses midpoint/raw-volume
  owner moments; do not silently reinterpret these as exact volume averages.
- Start with cubic reproduction. Differentiate the reconstructed polynomial
  analytically, then apply the chart/metric transformation consistently.
- Use common subfaces and geometry on both sides of an agglomerated interface;
  sum their contributions consistently for each owner.
- Cover every face family required by the certified operators. Audit
  ordinary/direct-face transitions explicitly rather than assuming a narrow
  transition patch is sufficient.
- Retain bounded value-row amplification; monitor mesh-scaled derivative-row
  amplification and stencil locality under refinement.
- Keep host-side compilation and fixed-shape JAX runtime application. Reuse
  existing moment-fit prototypes where their contracts match.
- Make unsupported degree, conditioning failure, and fallback behavior explicit
  in diagnostics.

**Interface changes:** extend internal reconstruction payloads with derivative
rows and quadrature provenance. Preserve evolved-state and restart layouts.
Candidate selection stays explicit in MMS/audit configuration; existing geometry
moments should be reused without regenerating unrelated magnetic artifacts.

**Gate:** reproduce the intended polynomial values/derivatives on actual HSX
owners to fitting tolerance for any new functionals and retain the applicable
structural properties before production integration. Bounded experiments
qualify implementation and motivate candidates; beating a baseline error or
improving every sampled owner is not a prerequisite for global qualification.
Repair only
what is needed for the global operator and solution targets; lower local order
alone does not require a repair if the global gates already pass.

### P05 — Repair and certify the brackets

**Dependencies:** P04.

**Current milestone passed:** the direct midpoint implementation is qualified
for static global MMS accuracy by the 26 September acceptance at the top of
this roadmap. Its geometry-derivative caveat and production/evolution scope
are explicit there. The following shared-face milestones are historical
qualifications of different operator identities, retained for comparison.

**First milestone passed:** the frozen general-cubic, shared-face, matched-q3
centered bracket C qualifies on the original omega, regular, and eta-varying
fields at 32/48/64. The [verified remote results](../../../../work/p_centered_cubic_c54b0552_kFhdmt_analysis/report.md)
include complete global actions and a global analytic IBP omega reference with
bounded independent qualification. All C orders exceed 1.8 on both intervals;
A/B constituents pass separately. C is this milestone's certified operator.
No additional field, resolution, or actual-action gate is imposed retroactively.
A predeclared held-out-field check remains a generality diagnostic during
implementation follow-through, without tuning the frozen design.

**Second milestone passed: material static accuracy.** The
[verified P05 global campaign](../../../../work/p05_material_257bd55f_qQnlSX_analysis/report.md)
qualifies the recentered nodewise cubic-jump action on both required scalars:
regular `3.4639/2.7951`, eta `3.6320/3.2729`, with reference budgets below 1.90%.
The same cubic supports/common generator are retained; only the shared jump
correction is added to material A. Diagnostic upwind vorticity also passes,
but does not replace the centered vorticity path. Shared implementation replay
and structural/evolved qualification remain follow-through; P05 is not yet a
production or complete time-evolution certification.

**Scientific checks, not blockers for the current study:** constant annihilation,
argument antisymmetry, shared-face bookkeeping, and checks of actual boundary
branches are lightweight diagnostics reported alongside the convergence result.
Reuse valid evidence and existing assembly data; do not require separate global
runs, repetition at every resolution, or a separate boundary-convergence gate.
Missing or failed diagnostics do not automatically stop the study or veto the
reported global operator-order result. Record their status and scientific
implications separately. This does not make an incomplete domain, incorrect
reference, or corrupted computation valid evidence; correct concrete execution
or bookkeeping errors when they prevent a meaningful convergence measurement.
Structural certification and production suitability remain separate later
decisions, without adding them as prerequisites to this static study.

Centered and material-upwind static accuracy are separately passed P05
milestones. Curvature, polarization, full-system balance checks, evolved MMS,
and production promotion remain subsequent work; they are not additional
prerequisites for these static bracket results. Completing both milestones does not by itself
complete P05 integration/structural work or certify the full perpendicular system.

**Bounded material-upwind decomposition complete (21 September 2026):** the
[N48/N64 evidence](../../../../work/p05_material_upwind_decomposition_20260921/report.md)
holds the qualified matched material action fixed and decomposes the existing
production scalar traces into mean and jump flux corrections on complete-owner
samples with axis, seam, wall, transition, sign-changing, and near-zero
generator coverage. Full upwinding increases every bounded field error; the
mean-trace replacement dominates, while the smaller jump-only term slightly
improves the eta-varying scalar at both resolutions. Trace, quadrature,
generator, boundary, anchor, volume-correction, and signed-incidence replays
close to roundoff, including a direct replay of the real scalar action using
the generator sign before `-1/rho_star` scaling. The diagnostic therefore does
not support promotion or a global material-upwind campaign in its present
form. It adds no per-owner gate and does not change the centered-static pass.

**Bounded nodewise cubic-jump comparison complete (21 September 2026):** the
[saved N48/N64 evidence](../../../../work/p05_nodewise_cubic_jump_20260921/report.md)
uses the exact prior complete-owner samples and the qualified central fit's
selection-v3 donors, chart, scale, and owner-moment observations. Opposite
fixed `0.75*tanh(s)` weight biases produce two rank-revealing cubic WLS fits;
only their nodewise jump is recentered around the unchanged qualified common
value. The correction is materially nonzero, smaller than the legacy jump,
improves the eta-varying scalar at both resolutions, mildly regresses the
regular scalar, and improves diagnostic vorticity. The unrecentered same-fit
pair no longer exhibits the legacy mean-trace failure but remains diagnostic
because it replaces an already qualified common value. All cubic reproduction,
selection-v3 support, action/sign, incidence, seam, collapsed-axis, and frozen
physical-wall checks pass; positive shifts do not activate the existing trace
floor. These bounded nonidentical samples are not global-order, dissipation,
positivity, or stability evidence. The subsequent authorized global campaign
now qualifies the recentered jump on both required material fields; its complete
domain errors mildly increase relative to centered A while preserving the
required order. The earlier bounded error reductions are not a global accuracy
improvement claim. Keep the support and bias fixed for held-out-field and
structural follow-through, without making those checks retroactive prerequisites
to the completed static campaign. Do not start a donor/bias scan.

- Integrate consistent generator derivatives and transported traces into the MMS
  material and centered-vorticity paths.
- Evaluate the complete face contribution with matched geometry and quadrature.
- Preserve the compatible correction and centered bracket identities. A gradient
  change must be checked against the complete compatible algebra.
- Test both operands independently and together; include variable geometry and
  magnetic coefficients.
- Require the centered vorticity bracket to pass independently of material
  upwinding.

**Current operator-accuracy gate:** second-order global owner-residual convergence
for each bracket path being qualified, with qualified references. Report
constant/antisymmetry and shared-face identities as scientific diagnostics under
the current-study rule above, not additional blockers. Interface orders are
diagnostic, not independent second-order gates. Independent
fixed-time solution checks are retained in P10.

### P06 — Repair and certify complete curvature

**Dependencies:** P04 (numerical dependency satisfied).

**Current milestone:** P1's bounded formulation/reconstruction audit (`p06_curvature_bounded_plan.md`, removed; see commit `c0269a95`)
and remote global campaign are complete. The
[returned-campaign local analysis](../../../../work/p06-curvature-cpu_4bb8336e_8ijYwA5H/local_analysis/report.md)
independently confirms all 22 nonzero primary M/R/total components above order
1.8 on both intervals. The subsequent
[bounded reference-only audit](../../../../work/p06_reference_qualification_20260922/report.md)
reproduces the exact seven-owner q3/q5/q7 preflight, saves all owner/cell
contributions, and localizes the unsettled N64 screen to one ordinary raw cell.
**P06 is closed for roadmap progression by explicit user decision**, accepting
the observed global-order pass and retaining the reference uncertainty as a
nonblocking scientific note. This does not claim that global reference
qualification was completed, alter the archived campaign's false combined
flag, or certify evolved/production behavior. Further reference computation is
not a prerequisite for P07 or subsequent roadmap work.

**New shared-structured P06 campaign (25 September 2026):** The separate
[returned global analysis](../../../../work/p06_structured_global_analysis_20260925/report.md)
found 40/44 nonzero primary component order checks passing against its frozen
q5 J target; four variable-Dirichlet fine-interval checks failed. The
[owner-measure audit](../../../../work/p06_measure_reference_audit_20260925/report.md)
confirmed that this research candidate uses J/B owner mass while that global
target used J. Its correct exact complete-source target is the J/B mean;
physical owner-volume error weights and midpoint observations remain fixed.
The same audit found material q3 cell underintegration at selected hotspots;
q7 improves one settled N48 hotspot but does not resolve all N64 reference
controls. The new v2 campaign implementation separates candidate cell and
reference rules and retains J as a diagnostic. This new shared-structured
global qualification remains **unpassed** pending an independently resolved
reference and a new global evaluation. The earlier user-accepted P06 milestone
above remains historical and is not retroactively changed.

**Boundary policy selected and global campaign prepared (22 September 2026):**
the [bounded A/B/C evidence](../../../../work/p06_boundary_functional_20260922/report.md)
and [point/mean/three comparison](../../../../work/p06_boundary_policy_20260922/report.md)
correct the earlier wall mechanism and close the bounded policy choice.  The
physical-wall characteristic correction is exactly zero under the frozen
contract; the wall-owner defect is in the smooth reconstructed volume gradient,
not a wall-face trace fluctuation.  The new versioned package interface
separates metric-derived wall value/normal/tangential functionals from dynamic
boundary data and exports fixed rank-revealing owner/boundary maps.
Actual-HSX eager/JIT/JVP and dynamic-data tests pass.

Freeze the single wall-center point relation for wall cells and every
geometry-selected wall-reaching central/biased face fit.  It is the smallest
field-independent constraint and avoids the broad coarse-grid regressions of
the three-moment cell policy; the single-mean and three-moment alternatives
remain diagnostics, not pooled-error gates.  Selection-v3 supports, degree,
weights, bias, q3 rule, midpoint/raw-volume observations, wall model, and
characteristic assembly are unchanged.  The unconstrained reconstruction
through the new face assembly replays the saved face correction to
`1.06e-16`; the portable frozen point candidate replays all bounded
N32/N48/N64 centered and U actions to `2.50e-11`.

The computation-only portable global runner (`scripts/p06_curvature_global`, removed; see commit `6f95ecea`)
now covers complete-owner N32/N48/N64 M/R/total actions for both states,
centered and U candidates with shared preparation, directional/region
diagnostics, independent continuous references and bounded q3/q5/q7 reference
qualification.  Actual-HSX complete-owner preflights pass at all three
resolutions; serial/parallel chunks are bitwise equal, checkpoint resume and
identity rejection are verified. The published campaign at revision
`6f95ecea6e493107aaa0ecf0735448eab83ec7b1`, identity `4bb8336e…`, has now
completed on remote CPUs and was independently reduced locally. All required
components pass observed orders; the minimum is `1.94027` (held-out Te material,
32→48). Primary total orders are approximately `2.97–3.43` for corrected frozen
MMS and `1.94–2.52` for the held-out state. Independent use of q3-integrated
volume weights also retains every pass. M+R and directional assembly close to
`2.67e-15`; reported boundary constraints close to `2.65e-13`.

The combined campaign flag remains false solely because 20 of 22 nonzero
maximum-based reference screens exceed 10%. The reference audit shows that a
matching q7-volume-weighted sample RMS is much less conservative: at N64 q5/q7
is below 10% for 21 of 22 components, with a maximum of 10.3954%. The residual
is overwhelmingly raw cell 169732 of ordinary owner 109892; it contributes
more than 99% of the sampled q5/q7 squared difference for the highlighted
frozen-Ti and held-out density/vorticity totals. Composite q5 refinement
through 8 subdivisions per axis remains oscillatory there, while five N64
axis/interface/boundary/seam controls settle from s4 to s8 below 6.05e-7.
Quadrature-dependent denominators are negligible. This implicates the local
phase/intersection with the piecewise-cubic MAKEGRID and shifted derivative
evaluations, not the axis, wall law, or curvature reconstruction.

**Retained nonblocking reference note.** q5/q7 and composite integration have
not settled on raw cell 169732. The favorable 21/22 sampled comparison concerns
q5/q7; the campaign used q3 references, whose matching N64 sampled screen is
below 10% for 14/22 components (maximum 16.21%). Neither sampled comparison is
a global norm estimate. Thus the reported orders remain measured against the
archived q3 references. If this diagnostic is revisited, stabilize that cell
and assess the actual q3 reference bias across all three resolutions, reusing
saved candidate actions. Do not launch another reference campaign or retune
the operator merely to close this note. Evolved physics, shared-component
extraction/consolidation and production integration remain separate. The wall
law and unconstrained omega policy are unchanged.

- Reuse the direct radial quadrature infrastructure.
- Repair remaining face directions, axis/endpoints, and boundaries according to
  the audit.
- Treat the material characteristic update and potential remainder as separate
  diagnostic components, then certify their sum.
- Reconcile quadrature, coefficient products, and the intended V/B versus
  physical-volume normalization through the continuum derivation.
- Preserve the coupled path-conservative balance and compatible geometric
  identities.
- Use positive smooth MMS fields that keep positivity fallbacks inactive; test
  fallback behavior separately.

**Gate:** second-order global residuals for the complete curvature contribution
in each evolved equation, retaining coupled balance and compatible identities.
Report material and remainder residuals separately to expose errors in the
composition; directional and regional orders are diagnostic. Independent
fixed-time solution checks are retained in P10.

### P07 — Repair and certify perpendicular diffusion/polarization

**Current static accuracy status (25 September 2026):** the combined structured,
BC-conditioned candidate has user-accepted observed approximately third-order
global MMS convergence, with reference sensitivity retained as nonblocking.
See the current acceptance record at the start of this roadmap. The older
boundary-conditioned reconstruction assignment and subsequent bounded tests
have culminated in the returned global campaign. Preserve both its evidence
and the historical D_trace baseline. All next-step statements in the historical
entries below are superseded where they conflict with this decision. P07
integration/evolution and production certification are separate outstanding
milestones; no new run is implied.

**Historical decision after local energy-repair feasibility (23 September 2026):**
The [P worker](thread://01a0d0df-8d51-7e93-8549-f6d8b36dd427?hostId=local) completed the
[bounded assignment](../../../../work/p_wip_archive_20261004/g4_p07_assignment_docs/src/drbx/dev_docs/p07_local_energy_repair_assignment.md); the
[report and standalone validation](../../../../work/p07_local_energy_repair_20260923/report.md)
use all 264 frozen faces and preserve the accepted static accuracy pass.
The full 16-owner D_trace symmetric forms have minima
`-1258.68/-3946.61/-7430.33`, but the wall-zero subspaces have positive minima
`1864.03/3175.09/4953.22`: this test does **not** rule out every wall-only repair.
Physical normal-gradient wall flux and cancellation with adjacent radial flux
dominate the saved negative witnesses. Removing the direct tangential wall
flux as a diagnostic leaves negative minima essentially unchanged; six q3
trace directions remain unconstrained, so indirect effects on normal fits
are not excluded. Existing hard-constraint residual penalties have zero action,
and individually positive owner-only wall damping fails quadratic consistency.
One compatible q3 trace-residual/transpose family was derived but rejected
before coefficient selection or application: representing its donor outputs
by physical shared fluxes needs at least **2,123 additional faces**, even
crediting the saved N32 477-face audit, above the 1,000-face cap. No total
energy bound or repaired scheme is claimed. The next design requirement is
a compatible boundary-work and local normal-gradient/flux/divergence identity
with finite output support, actual owner means, cubic consistency and shared
physical fluxes; no unique repair or necessity to alter adjacent faces has
been proved. No new faces, elliptic solve, global campaign, evolution, tracing,
production change or automatic follow-on run was launched. P07 remains open
at its energy gate; elliptic qualification stays deferred. These are
noncontractivity and scoped feasibility results, not global unstable
eigenvalues, eventual blow-up or an explanation of smooth-MMS order loss.

**Historical decision after structural/application audit (23 September 2026):**
The coordinator completed the user-authorized
[bounded fixed-operator audit](../../../../work/p07_structural_application_audit_20260923/report.md)
without elliptic solves or a new global campaign. Both original center-Dirichlet
and frozen trace-moment treatments admit two-owner near-wall perturbations with
negative `u^T M A u` for `A=-div(P grad)` at N32/N48/N64. Independent constrained
gradient integration confirms the result; at N32 every one of the 477 globally
affected faces is included and the complete response conserves signed external
flux to roundoff. The trace witness on the same geometric track gives
`-1258.68/-3946.61/-5886.67` with unit physical-volume norm. This establishes
failure of physical-volume energy dissipation, not an unstable global eigenvalue
or eventual blow-up, and does not identify the cause of smooth-MMS order loss.
Compatible constant boundary loading passes. Cached CPU application of this
bounded face batch takes about 0.10 ms with fixed maps and no per-application
solve; global/JAX runtime cost is not measured. Preserve the static accuracy
pass, but defer the proposed explicit evolution qualification. Next investigate
the local radial flux/energy structure and a fixed local repair that retains
consistency and conservation; no repair or assignment is launched by this
review. Elliptic qualification remains deferred at the user's preference.

**Historical decision after scaling review (23 September 2026):** The bounded
[quartic-response and stencil-scaling test](../../../../work/p_wip_archive_20261004/g4_p07_assignment_docs/src/drbx/dev_docs/p07_dirichlet_error_scaling_assignment.md)
is complete and [independently reviewed](../../../../work/p07_dirichlet_error_scaling_20260923/parent_review.md).
Normalized footprints remain broad but nearly fixed; complete-row norms grow
moderately, and large fourth-order/higher-order contributions change their
cancellation. No concrete reconstruction defect or uniquely justified repair
was identified. The user prefers to avoid elliptic solves: defer that
qualification and recommend a bounded structural/application-cost audit using
fixed operator applications, then a small fixed-step explicit diffusion test
with prescribed potential if supported. Neither adds an elliptic correction
or changes existing runtime polarization handling. These are recommendations,
not newly launched assignments. Preserve the static pass; structural, evolved
and polarization/inversion qualification remain open. Historical recommendations
below to proceed immediately to elliptic testing are superseded by this deferral.

**Current decision after the bounded mechanism test (23 September 2026):**
The matched [Dirichlet-to-flux assignment](../../../../work/p_wip_archive_20261004/g4_p07_assignment_docs/src/drbx/dev_docs/p07_dirichlet_flux_mechanism_assignment.md)
is complete. The [parent review](../../../../work/p07_dirichlet_flux_mechanism_20260923/parent_review.md)
independently verifies archived donors/fluxes, complete signed actions and all
reported group budgets. Preserve the accepted global static-order pass and
frozen center-value-plus-two-trace-moments candidate. P07 remains open pending
structural qualification and an independent manufactured elliptic solve; these
are the recommended next work, not work launched by this review. The mechanism
test identifies interior reconstruction-remainder response and changing radial
cancellation, but does not prove uniform stability or a universal leading
quartic error law. All tested Dirichlet value traces are zero, so general
nonzero boundary loading remains unqualified. No new static policy scan,
production change or change to either Q campaign follows from this result.


**Cached global Dirichlet comparison completed (23 September 2026):** The
[frozen-graph result](../../../../work/p07_cached_global_dirichlet_20260923/report.md)
updates all 24,032/64,752/118,149 wall-reaching phi faces at N32/N48/N64,
independently reassembles the complete face ledger, and reuses unchanged faces
and references. The fixed centre-value-plus-two-trace-moments candidate passes
the static global order gate; details and remaining limits are recorded below.



**Remote global result and current assignment (23 September 2026):** The
[returned campaign analysis](../../../../work/p07-diffusion-cpu_72acda02_yYbcA48I/local_analysis/report.md)
verifies all 22,379 chunks and independently reassembles the full actions.
Ti, regular-Neumann and mixed-eta fields pass both global intervals. Phi has
orders 3.28168/1.74047, narrowly below the 1.8 fine-interval target. Its N64
outermost layer contributes 58.45% of squared error, with loss of wall/adjacent
radial-face cancellation relative to N48. Exact-gradient oracle discrepancies
are small. This baseline static qualification remained open pending the bounded
and cached-global follow-ups below. Archived condition numbers describe the
Neumann map and cannot diagnose phi conditioning. Local orders remain
diagnostic.


**Bounded wall/interior follow-up (23 September 2026):**
[parent analysis](../../../../work/p07_diffusion_polarization_wall_followup_20260923/parent_review.md)
replays the saved signed face actions. Physical-normal Neumann constraints
repair the physical-wall face but leave the neighboring interior radial-face
error dominant; the phi value-constrained variant does not improve its wall
error. The experiment applies three surface moments only to the physical face,
not the P06-selected single wall-center relation throughout wall-reaching
stencils. Non-wall actions remain unchanged. The parent corrected the stale
constant diagnostic separately: homogeneous-Neumann unit-constant action is
`1.89e-10`, rather than the incompatible Dirichlet check stored in the worker's
JSON. Next: apply the established point relation to all appropriate wall-reaching
fits and audit the identified interior radial-face functionals. No degree or
support-count change is justified yet; global order and elliptic checks remain
open.

**P06 point-policy replay (23 September 2026):** The fresh [bounded
comparison](../../../../work/p07_diffusion_polarization_point_policy_20260923/report.md)
applies the selected single wall-centre relation at the actual physical-wall
location to all six geometry-selected wall-reaching cubic fits (including the
previously untouched adjacent radial fit), while retaining inherited and
three-moment physical-face controls. The rank-one geometry preparation and
complete signed-incidence face replay validate. The all-wall-reaching policy
substantially reduces the N32 sample RMS for the two Neumann stress fields
(`regular: 3.16807 -> 1.11511`, `mixed: 3.13721 -> 1.15836`), but is not a
uniform improvement (`phi: 0.31856 -> 0.43296`; Ti's physical-only point
diagnostic is better than its all-wall-reaching value). This localizes a real
policy sensitivity and retains the existing bulk radial-face errors; it is not
a global-order or production qualification. Next audit the value/target-gradient
functional on the six selected fits and close the agglomerated/ordinary radial
face reproduction record without changing degree or support-count policy.

**Global-preflight input receipt (23 September 2026):** The frozen bounded
32/48/64 P07 preflight contract, tracks and streaming implementation plan are
recorded in [the preparation receipt](../../../../work/p07_global_preflight_20260923/README.md).
The initial missing-input report was corrected: all three geometry directories
are valid symlinks to the existing `prototype_runs/geometry/hsx_fci_*` artifacts.
The parent verified all 24 manifest-listed component hashes and located the
continuous-reference sidecar; see the preparation folder's
`parent_input_verification.json`. Inputs are available; the bounded harness
still requires implementation and execution. First reproduce the saved N32
overlap, then complete the bounded N48/N64 comparison before deciding on the
separate global campaign.

The remaining adapter and bounded replay work is assigned to the replacement
P worker through the [implementation handoff](../../../../work/p_wip_archive_20261004/g4_p07_assignment_docs/src/drbx/dev_docs/p07_refinement_worker_handoff.md).
Existing track selections and topology preparation are reusable; they are not
completed numerical refinement results.

**Bounded refinement completed (23 September 2026):** The replacement P worker
implemented the resolution-aware [research adapter and report](../../../../work/p07_global_refinement_20260923/report.md).
It freshly reconstructs only the frozen complete wall/agglomerated/ordinary
closures, applies the single wall-centre relation to every donor-selected
wall-reaching fit, and assembles canonical signed incidence from topology. The
N32 all-wall-reaching overlap reproduces archived completed actions to
`1.07e-14` and face fluxes to `4.34e-19`; N48/N64 complete closures are finite,
have six wall-reaching fits and retain q3 face/oracle/q7 continuum evidence.
The fresh unit-Dirichlet/homogeneous-normal point-policy constant receipts are
`4.77e-11`, `2.06e-10`, and `9.70e-11` at N32/N48/N64.  Regular and mixed
Neumann three-owner RMS values decrease over both intervals, while phi and Ti
rebound on the selected N64 wall track.  These small mapped-owner slopes are
diagnostic, not a global acceptance gate or a reason to choose policies by
field.  The report retains physical-wall/adjacent/other signed components,
stored-versus-continuous volumes, measured cost/RSS, and a conservative global
cost extrapolation.  The next global campaign remains a separate decision: it
must batch and qualify reference evaluation and evaluate all complete owners
before claiming the roadmap's physical-volume L2 order criterion. The
[parent review](../../../../work/p07_global_refinement_20260923/parent_review.md)
independently verifies raw-membership incidence and both Dirichlet and Neumann
constant receipts; it supports advancing to global-runner preparation, with
representative reference-cost measurement before launch.  No cubic degree,
support-count, field-policy, production, structural, or elliptic milestone has
changed.

**Phi Dirichlet-wall audit completed (23 September 2026):** The corrected
[complete-owner audit](../../../../work/p07_phi_dirichlet_wall_audit_20260923/corrected_v2/report.md)
retains the historical partial result and implements the assigned wall-centre
value plus two centered tangential trace moments on every wall-reaching fit in
eight wall/hotspot closures and their inward controls. Archived point actions
replay within `3.94e-11`; all four fields are retained and Neumann controls are
unchanged. Independent raw-midpoint cubic reproduction is within `3.60e-14`
in gradient and `1.11e-15` in integrated flux. Actual target sensitivities and
cubic reproduction support further study of truncation/cancellation rather
than rank loss on these samples. The candidate worsens selected N32/N48 errors
but improves N64 (`1.17475 -> 1.09784`). The
[parent review](../../../../work/p07_phi_dirichlet_wall_audit_20260923/corrected_v2/parent_review.md)
independently confirms complete closure and action assembly, and notes that
coarse-sample regression is not a rejection gate. It recommends one cached
global flux-difference comparison on all wall-reaching fits, reusing unchanged
faces and continuum references. The purported quartic test is not a pure
regular-chart quartic; inward-owner component labels also require correction.
These do not invalidate completed MMS actions. Neither promotion nor global
qualification is claimed; structural, elliptic, evolved and production
milestones remain open.

**Cached global trace candidate passes static accuracy (23 September 2026):**
The completed [comparison and analysis](../../../../work/p07_cached_global_dirichlet_20260923/report.md)
uses the archived donor graph and changes only phi fluxes on every archived
wall-reaching fit. Complete candidate phi L2 errors are
`0.590880/0.157559/0.0908551`, with orders `3.25999/1.91368`; the fine order
therefore clears the 1.8 gate that the point baseline narrowly missed. Ti and
both Neumann controls are unchanged and retain passing orders. Independent
full-ledger assembly agrees with baseline-plus-delta assembly within
`2.38e-13`, and baseline replay agrees within the same bound. The candidate's
absolute phi error is nevertheless 7.25%, 8.20%, and 2.94% larger at the three
resolutions. At N64 it slightly improves outermost-layer RMS
`0.371343→0.370347` while worsening the inward-adjacent and ordinary-owner
budgets; this is an order pass, not uniform local improvement. Saved full
radial profiles, exact-gradient controls, genuine regular-chart quartic
responses, coverage and execution receipts accompany the report. Freeze this
candidate identity for the smallest next step: P07 structural and elliptic
checks. Do not launch another boundary-policy scan or static global campaign,
and do not infer production/evolved qualification.

**Bounded Dirichlet-to-flux mechanism test completed (23 September 2026):**
The [matched six-field treatment matrix and Taylor budget](../../../../work/p07_dirichlet_flux_mechanism_20260923/report.md)
reuses the exact sixteen owners and 88-face closures at each resolution with
archived donors and once-prepared maps. D-point, D-trace and a separate exact-
normal N-point diagnostic replay their saved anchors within `4.01e-12` in
completed action; independent saved-array assembly is exact. Phi is not unique:
Ti reproduces its N48-to-N64 bounded rebound under both Dirichlet treatments,
while the regular and mixed-eta functions retain substantial Dirichlet wall
errors without rebounding. All four original functions share a fourth-order-
flat wall envelope, so their exact value and first gradient vanish there. An
independent symbolic regular-chart Taylor decomposition shows that the hotspot
wall-face phi error is entirely the interior owner-observation remainder
response; adjacent-face error is also owner-remainder dominated, with cubic
reproduction at `2.09e-17` or below. Signed wall/adjacent/further-face
cancellation changes strongly with resolution and treatment. The exact-normal
diagnostic reduces phi error but redistributes it to the adjacent face and is a
different boundary problem, not a replacement policy. Genuine nonzero-normal
quartics support boundary-map sensitivity without exposing a rank or arithmetic
defect. Preserve the static pass and proceed to the frozen candidate's
structural/elliptic checks; no repair, policy scan or new static campaign is
supported.

**Bounded Dirichlet error-scaling test completed (23 September 2026):** The
[common-owner quartic-response analysis](../../../../work/p07_dirichlet_error_scaling_20260923/report.md)
uses all 15 degree-four monomials in the fixed chart
`z=(x,y,eta/eta_period)`, compatible degree≤3 controls, and direct independent
applications of phi's p3, q4 and exact fifth-and-higher remainder. Normalized
supports are broad but nearly fixed across N32/N48/N64: raw-member RMS spans
about 5–7 cells in x/y, maxima reach about 13 cells, and eta maxima remain 2.5
cells. Complete `h²`-scaled owner-row medians grow 14–16% and maxima 24–28%; the
fixed-angle geometry quartic-response norm grows about 25%, while hotspot
responses are larger and include track relocation. This is moderate,
heterogeneous finite-resolution change, not a demonstrated blow-up or uniform
bound. The full phi decomposition rejects a single leading-quartic account:
q4 and r5 RMS are each several times the completed error and their signed cross
shares switch between cancellation and reinforcement. On the geometry-only
D-trace subset, actual/q4/r5 RMS divided by `h²` changes
`970/2471/3363 -> 239/2938/2976 -> 681/3195/2526`. Exact symmetric splits show
both analytic profile and retained response changes contribute; the response
term still contains metric/location effects. Replays, all 15 direct-versus-
contracted responses and independent row/term assembly pass. No concrete
reconstruction defect or repair is identified. Preserve the static pass and
proceed to structural then independent elliptic qualification with the broad-
support/pre-asymptotic cancellation limitation documented; do not add a global
Ti run, policy scan or degree/support change.

**Earlier parent verification and decision:** the [independent follow-up](../../../../work/p07_diffusion_polarization_point_policy_20260923/parent_review.md)
closes the targeted reproduction audit on twelve implicated faces. Donors and
stored bases match; direct raw-midpoint polynomial observations agree to
`2.33e-13`; unconstrained face-flux reproduction is within `5.60e-13` relative,
and point-constrained reproduction within `7.21e-15`. Independent fixture-based
incidence replay agrees within `3.02e-14`. The N32 phi regression is not a
local improvement gate or proof of failed global order. Next: freeze the
all-wall-reaching point policy for a bounded 32/48/64 complete-owner refinement
comparison as a cost/mechanism preflight for global qualification. Retain
face-resolved signed errors and independent oracle controls; no speculative
degree or donor-count change is justified by this audit.

**Dependencies:** P04 and P03's return-map audit.

**Next bounded assignment: physical-action audit and shared-face comparison.**
The [P07 worker assignment](../../../../work/p_wip_archive_20261004/g4_p07_assignment_docs/src/drbx/dev_docs/p07_diffusion_polarization_bounded_assignment.md)
provides the execution design, direct code entry points, and artifact locations.
Start on the existing N32 HSX artifact with complete owners selected by geometry
across axis, agglomerated bulk, size transitions, ordinary interior, physical
wall and periodic seams. This diagnoses mechanisms; it does not certify order.

1. Freeze the actual MMS selectors and trace the conservative perpendicular
   action through field reconstruction, fine face fluxes, owner restriction,
   `LocalPerpLaplacianInverseSolver` application/inversion and evolved diffusion. Record
   the physical tensor, signs, coefficients, normalization and effective
   boundary data. Separate physical action, affine boundary load, solver
   regularization and preconditioner. The existing `apply_positive_operator`
   is the shared phi/Ti application point; verify its use rather than inventing
   another polarization equation.
2. On those complete owners compare the baseline with fluxes assembled from
   the qualified cubic owner-to-face derivatives and continuous geometry at
   the matched face quadrature nodes. Use the audited physical perpendicular
   tensor, including its metric cross terms. Assemble each oriented flux once
   and distribute it with opposite incidence signs, then divide by physical
   owner volume. Do not substitute the bracket or curvature flux formula.
3. Use analytic gradients at the same faces as an oracle to separate geometry,
   integration/assembly and boundary errors from numerical reconstruction.
   Compare against an independent continuum owner functional, not the discrete
   action applied to the manufactured field. Reuse qualified reference methods
   and bounded integration checks; do not impose universal high-order rules.
4. Reuse the boundary-functional machinery with the correct operator-specific
   constraints. Physical normal derivatives and perpendicular conormal fluxes
   are not interchangeable. Keep the frozen wall model, collapsed-axis and
   periodic conventions; separate homogeneous action from nonzero loads.
   Record conservation/boundary balance, the applicable constant/nullspace
   behavior, weighted pairing and dissipative work. Do not assume that cubic
   accuracy or conservative incidence guarantees symmetry/energy stability.
5. If the bounded comparison supports the candidate, proceed to existing
   N32/N48/N64 global operator qualification with one frozen policy, then an
   independent manufactured elliptic solve using that same physical action.
   Include smooth regular-chart angular and mixed-eta fields and the actual
   variable coefficient/metric factors required by the selected model. Keep
   solver error below 10% of spatial error; handle a Neumann gauge only where
   the selected boundary problem requires it. A discrete self-generated RHS
   may check the solver but cannot establish continuum solution accuracy.

The initial assignment ends at the bounded audit/comparison and a justified
next-step decision; a full global launch is separately scoped. Reuse saved
geometry and fixed coefficient maps, measure setup versus application costs,
and checkpoint expensive preparation. No production selector, wall-law change,
new transport physics, or evolved/global campaign is implied by this plan.

- First retain the current conservative composition and replace the identified
  inconsistent derivative/flux stages.
- Verify the actual owner action against the owner-averaged continuum operator,
  including variable metric cross terms and physical boundaries.
- If the return map prevents the global operator or solution gate from passing,
  prototype an explicitly consistent mass/source formulation using the same owner unknowns. Keep this
  as a separately identified candidate.
- Derive boundary loads and distinguish physical normal derivatives from
  conormal fluxes.
- Use exactly the same selected physical action for phi inversion, Ti
  polarization, and evolved perpendicular diffusion; exclude solver
  regularization from physical diffusion.

**Gate:** second-order global operator residuals and independent elliptic
solutions, correct nullspace and conservation/boundary balances, and the
applicable energy property. Fixed-time diffusion solution MMS is retained in
P10. Regional operator orders remain diagnostic.

**Decision rule:** promote the smallest candidate that passes all gates. If both
the retained composition and the explicit mass/source candidate fail, record
the structural blocker and reopen the numerical design; do not redefine the
reference or weaken the success criterion.

## 4. MMS certification, integration, and progress tracking

### Physical-normal Neumann MMS field contract — 27 September 2026

**User decision, applying to P05N, P06N and P07N:** qualify the Neumann
closures with nonzero physical-normal data taken from the manufactured field
itself, `g_N = n · grad_x f` at the wall. The fields are smooth, axis-regular,
theta/eta-periodic analytic functions with nonconstant wall traces. They carry
no boundary-correction term, and their values, gradients and Hessians are exact.
The HSX wall geometry enters only through the prescribed data `g_N`, which is
what the closure must consume. Each campaign freezes a held-out field together
with its catalogue, before any numerical action is evaluated.

**Fields must lie outside the reconstruction's exactness space — 28 September 2026.** The interior reconstruction reproduces exactly any field that is at most cubic in u with θ harmonics ≤ 3: 4-point radial Lagrange, 7-point θ trigonometric. For such fields the two one-sided face reconstructions coincide, to about 1e-15, so the upwind (P05 U − A) and characteristic (P06 q3) corrections vanish on interior radial and θ faces, and interior errors come only from η and the boundary rows. The P07N/P05N `frozen_v1` fields are in that space. Catalogues that must exercise upwinding therefore include fields with non-polynomial radial profiles, radial degree above 3 and θ harmonics of 4 or more. Accepted Dirichlet P05's fields were outside it; its saved U − A is active through the interior, at RMS 1e-7 to 1e-3 per layer.

Zero data is not a separate global field requirement. The shared Neumann rows are
linear in owner observations and prescribed data. The `g_N = 0` path is instead
covered by a bounded row-level check that the restored traces/gradients satisfy
`row(f, g) = row(f, 0) + row(0, g)`, and that zero data replays the homogeneous
application exactly. This supersedes the earlier "zero and spatially varying
nonzero data" field requirement below.

**Reason:** a zero-normal field with a nonconstant trace must satisfy
`f_u = -(r_theta f_theta + r_eta f_eta)` at the wall, with
`r_* = g^{u*}/g^{uu}`. On HSX, `r_theta` has standard deviation 0.43 and eta
harmonics n = 4–20 ([wall spectrum](../../../../work/p07n_compatible_fields_20260927/wall_spectrum.json)).
That requirement forces wall non-orthogonality into the field itself, through a
correction differentiated by finite differences. The failed P07N campaign below
localized its error to exactly this construction. It is a property of the test
field, not of the operator.

**Field admission:** before a catalogue is frozen, screen each candidate
globally on N32/N48/N64 with the exact-gradient face action O against the locked
midpoint target R. No reconstruction is evaluated, so candidates are not chosen
by numerical-operator error. The [admissibility design](../../../../work/p07n_compatible_fields_20260927/screen_design.json)
was frozen before any result: O−R global L2 order ≥ 1.8 on both intervals, and
≥ 1.9 preferred. A conservative flux-form action is at best second order against
the midpoint target, so a field failing this screen cannot fairly test P07N at
these grids. P05N/P06N apply the analogous exact-input/midpoint screen for their
own actions. Every screened candidate, including rejected ones, is reported.

**Wall-face exterior-state contract for P05N and P06N — user decision, 27 September 2026.**
At the wall face, the exterior state entering P05's `U − A` jump and P06's characteristic wall
solve is the **recovered Neumann trace** from the shared rows; for Dirichlet fields it is the
prescribed trace, as accepted. The wall-face jump and characteristic correction are therefore zero
by construction, exactly as in the accepted P05 and P06 qualifications. No manufactured wall state
and no independent constraint is used. This qualifies the operators and their Neumann
reconstruction, not a physical characteristic wall law. The rung ladder or production wall-trace closure,
including the curvature drift's wall-normal component and the wall-normal E×B velocity for
Neumann φ, is a separate later qualification, and both acceptance records must say so. P07N's
elliptic flux closure has no exterior state. The [design](../../../../work/p_neumann_p05n_p06n_design_20260927/design.md)
records the shared row families and the exact-input screens.

### P07N — Explicit Neumann boundary implementation and qualification

**Status: passed — user-accepted closure qualification, 27 September 2026.** See the
[acceptance record](../../../../work/p07n_field_derived_274e93e9_20260927T054625Z_72cfa1/local_analysis/acceptance_decision.md)
and the summary at the top of this roadmap. The field-derived campaign `274e93e9`
is accepted on N−O. Its midpoint accuracy for wall-active fields is limited by
[near-wall toroidal geometry resolution](#near-wall-toroidal-geometry-resolution-coil-ripple--27-september-2026).

**Preserved earlier failure:** the frozen static global campaign ran at `5930b72c` and
failed its gate; that result is preserved unchanged. The
[local analysis](../../../../work/p07n_static_global_5930b72c_20260926T2355Z_a91d3c/local_analysis/report.md)
verifies all 24,495 chunks and reassembles the arrays exactly. It records N−R
orders of `1.548/1.830` for the smooth nonzero control, `0.677/0.747` for
zero-normal m1, `0.669/0.747` for prescribed-nonzero m1 and `1.077/1.257` for the
held-out field. The constant and wall-residual trend checks pass. N−D stays
about `1e-5` of N−R, so the physical-normal closure does not explain the
failure. The dominant error is O−R in the interior correction collar
(u = 0.25–0.75, 91–92% of squared error). The
[reference/face audit](../../../../work/p07n_reference_face_audit_20260926/report.md)
excludes derivative, reference, extraction and face-quadrature defects. It shows
the exact-gradient gap closing at about second order under box shrinking, mostly
through eta. The
[eta-span comparison](../../../../work/p07n_eta_span_comparison_20260926/report.md)
confirms that, locally, midpoint error falls monotonically with the eta extent;
true eta refinement is location-dependent. The next P07N campaign uses the field
contract above, with a new catalogue and output folder; it is not a rerun.
Shared extraction and replay are complete for the bounded scope. This does not
reopen the accepted Dirichlet static passes.

**Next campaign definition — user decisions, 27 September 2026:**
- **Frozen catalogue:** `field_b1` (the frozen smooth-control base), `field_e3` and `field_e12`, which give weak and strong theta wall-trace gradients and so exercise `r_theta` non-orthogonality. `heldout_field_b2` (m2/m3 content) is held out, and the constant is included. All wall data are field-derived.
- **Reporting and acceptance:** N−R global L2 order ≥ 1.8 on both intervals for every nonconstant field stays the headline gate, as traditional MMS reporting. The constant and wall-trend checks stay as well. The acceptance decision is the user's, made after the results by examining N−O (reconstruction/closure error against exact face fluxes) and O−R (exact-flux consistency with the midpoint target) separately. This follows the Dirichlet P07 observed-order acceptance. The frozen gate flag is preserved as returned, whatever the decision.
- **Predeclared expectation:** the [admission screen](../../../../work/p07n_compatible_fields_20260927/report.md) gives exact-gradient O−R orders of `1.548/1.830`, `1.739/1.782`, `1.684/1.809` and `1.712/1.719` respectively. The accepted Dirichlet P07 fields give `1.75–1.81/2.06–2.11` under the same screen. For smooth fields, N−O is about `5e-5` of O−R, so N−R is expected to track O−R. A gate failure consistent with this prediction is therefore expected, and is a field/geometry/target property rather than a closure defect.
- **Why Neumann converges more slowly than Dirichlet (measured 27 September):** an [envelope sweep](../../../../work/p07n_compatible_fields_20260927/report.md#addendum-why-the-dirichlet-fields-converge-faster-27-september-2026) shows O−R order is controlled by how much field content sits in u>7/8, where the perpendicular-tensor coefficients' sub-Nyquist eta content grows about tenfold. This holds even for fields with zero wall value and gradient. The accepted Dirichlet fields peak at u≈0.45. A Neumann test must have wall content. The Dirichlet fine-interval orders are therefore optimistic for near-wall edge/SOL structure, and P08/P10 resolution planning should use the outer-band behavior.
- **Outcome and attribution:** the returned campaign matched this expectation exactly (N−O `2.24–3.85/2.25–3.73`; N−R ≈ O−R as predicted), and the user accepted it on 27 September. The residual midpoint stall of wall-active fields, including the held-out field, is most plausibly unresolved modular-coil ripple. See [near-wall toroidal geometry resolution](#near-wall-toroidal-geometry-resolution-coil-ripple--27-september-2026).

The [bounded pilot](../../../../work/p_neumann_structured_trace_20260926/report.md)
passes its algebra and 23 focused tests. Its operator errors are N-O comparisons
against exact-gradient q3 face assembly, not midpoint N-R certification. The
[parent review](../../../../work/p_neumann_structured_trace_20260926/parent_review/review.md)
identifies the largest original homogeneous-data discrepancy on unchanged
interior rows. The [error audit](../../../../work/p_neumann_error_audit_20260926/report.md)
separates matched Dirichlet, exact q3 face and midpoint terms. The
[axis-regular preflight](../../../../work/p_neumann_axis_regular_preflight_20260926/report.md)
qualifies globally valid periodic fields and their numerical reference controls
on complete wall, interior, seam and correction-transition owners at N32/N48/N64.
The bounded Neumann increment relative to matched Dirichlet is small. Shared
face-versus-midpoint consistency error and the held-out B2 off-node
physical-normal residual remain explicit global diagnostics, not
reference-uncertainty findings. The [local campaign preparation](../../../../work/p07n_global_prep_20260926/report.md)
freezes the static global contract and computation-only handoff draft without
launching it or promoting a package default.

The accepted shared construction uses prescribed Dirichlet traces. Fields named
`regular_neumann` and `mixed_eta_neumann` were evaluated with their analytic wall
values; they establish accuracy on Neumann-compatible fields, not qualification
of a boundary closure supplied only with physical-normal derivative data.

The supported input for these new gates is prescribed `g_N = n · grad_x f`.
Qualification fields supply nonzero, spatially varying data from the field
itself, and zero data is verified by the bounded row-linearity check, both per
the field contract above. Logical radial derivatives and prescribed tensor fluxes
are not alternative APIs in this work package.
On the nonorthogonal grid use `a = E^{-1} n`, so the constraint is
`a · grad_q f = g_N`; zero normal derivative does not generally imply zero
anisotropic diffusive flux.

1. **Freeze the boundary functional.** Distinguish logical radial derivative,
   physical outward-normal derivative, and outward diffusive flux through the
   anisotropic tensor. These are not interchangeable on HSX geometry. State the
   precise supported condition, metric factors, sign and units for each operator;
   qualify each advertised variant explicitly. Keep physical characteristic wall
   states separate from this reconstruction boundary constraint.
2. **Implement through the shared preparation/application layer.** Construct
   geometry-dependent boundary rows using prescribed physical-normal data and
   interior owner observations. Recover unknown wall values and required
   tangential behavior from those inputs; do not supply exact manufactured wall
   values or tangential derivatives of that unknown trace. The MMS oracle may use
   analytic fields independently to score the result. Preserve owner unknowns,
   interior policy, axis treatment and operator-specific measures. No additional
   global elliptic reconstruction solve is part of this step.
3. **Run bounded real-HSX tests first.** Use multiple admitted manufactured
   fields with field-derived nonzero data and unknown, nontrivial wall traces,
   plus the row-linearity/zero-data check. Include wall-adjacent rows, periodic seams and
   the boundary-to-interior transition at N32/N48/N64. Check boundary-functional
   satisfaction, recovered wall values/gradients, complete selected-owner actions,
   constants where applicable, JIT/JVP and unchanged Dirichlet replay. Retain
   different reconstruction targets for P07 integrated rows and P05/P06 point rows.
4. **Qualify complete operators with the new boundary path.** After bounded
   checks pass, prepare a frozen N32/N48/N64 global static midpoint campaign for
   P07 diffusion/polarization actions. P05N and P06N below separately qualify
   their full actions, including applicable jump/characteristic terms.
   Use the existing operator-order criterion and term/regional diagnostics;
   no mandatory high-order volume-reference integration. Report exactly which
   fields, boundary functionals and operators are covered. Campaign execution is
   a later scoped task, not launched by this roadmap edit.

**Gate:** true physical-normal-derivative-only boundary input, demonstrated boundary
consistency and global static operator convergence for the supported Neumann
variants, with unchanged Dirichlet behavior. Include both boundary families in
P08/P10 for physically consistent field/closure combinations. Pure-Neumann
polarization inversion additionally needs compatibility and gauge/nullspace
handling in the later solution tests; a static operator pass does not certify
that solve, and this step does not require a new inversion method.

### P05N — Physical-normal Neumann bracket qualification

**Status: passed — user-accepted static qualification, 28 September 2026**, from `frozen_v1` and `upwind_v1` together. See the [acceptance record](../../../../work/p05n_p06n_43250ccf_20260928T053254Z_c415a4cd/local_analysis/acceptance_decision.md). Preserve the accepted P05 Dirichlet qualification.

- Qualify the complete accepted direct midpoint bracket, first centered and then
  with its live scalar face-jump correction. The three-wall-owner jump pilot in
  the P07N bundle is useful evidence but does not qualify the full bracket.
- Prescribe physical-normal derivatives for the declared generator and
  transported-field combinations, taken from each field (`g_N = n · grad_x f`)
  per the field contract above. Recover unknown wall values and all required
  gradients from owner observations and those data; keep common gradients,
  fixed-anchor side reconstructions and physical exterior-state policy explicit.
  Exact traces are scoring or matched-Dirichlet controls only.
- Run bounded N32/N48/N64 wall, seam and boundary-to-interior checks with
  field-derived nonzero data, nonconstant wall traces, held-out fields and the
  row-linearity/zero-data check. Cover
  complete owner actions and separate centered/jump errors, constants where
  applicable, and JIT/JVP in owner and boundary data. Verify periodicity and axis
  regularity of fields before expanding to the entire grid.
- After bounded gates pass, run a frozen global midpoint MMS campaign including
  ordinary and aggregate/core owners and regional norms. Retain the existing
  order criteria and P05 magnetic-reference caveats; use independent reference
  sensitivity checks where needed, without mandatory volume integration.

**Frozen catalogue — user decision, 27 September 2026**, recorded in [p05n_catalogue.json](../../../scripts/p05n_field_derived_global/p05n_catalogue.json):
- **(a) Neumann generator × Neumann transported field:** `field_b1→field_e3` and `field_e12→field_b1`; held-out `heldout_field_b2→field_e12`.
- **(b) Dirichlet generator × Neumann transported field:** `field_b1(D)→field_e3` and `zero_trace_generator(D)→field_e12`; held-out `heldout_field_b2(D)→field_b1`. The new `zero_trace_generator`, (1−u²)(0.12x cos η + 0.08(x²−y²) sin 2η), has zero wall value and nonzero normal derivative, i.e. the physical φ=0 wall.
- **Controls:** a constant in each slot.
- **Candidates:** centered, and centered plus the live `U − A` jump, under the recovered-trace wall-exterior contract.
- **Predeclared expectation:** N−R ≈ N−O (pointwise operator, no O−R ceiling). A pass does not certify near-wall physical accuracy.
- **Held-out disclosure:** the bounded construction (below) chose its own pairings before the freeze. One of them, `field_e12→heldout_field_b2`, is by exact antisymmetry the negative of the frozen held-out centered action, so that pair's centered N−R was seen at 52 near-wall owners per grid. Nothing was tuned from it, and `heldout_field_b2` was already evaluated globally in P07N. The user kept the held-out pair (27 September 2026).

**Bounded construction — passed, 27 September 2026.** At the 54 selected owners per grid on N32/N48/N64, every structural check passes:
- the all-Dirichlet replay of accepted P05 (at most 3% of an atol + rtol = 1e-12 tolerance);
- Neumann conditioning (≤ 7) and row linearity with zero data;
- constant action (≤ 3e-13) and swapped-argument antisymmetry (≤ 3e-16);
- a zero live jump on the physical-wall, radial n−1 and last-two-layer transverse faces, which holds by construction under the wall contract;
- JAX lowering.

The Neumann and accepted Dirichlet rows share identical donor patches at every near-wall face quadrature point. The batched global package `scripts/p05n_field_derived_global/` reproduces the per-owner construction on those owners for every frozen pairing, to at most 3% of the same tolerance. The package batches rows across fields and pairings, caches the physical normal and wall data once per grid on the wall lattice, and memoizes the frozen cardinal exactly. The local N32 smoke run cost about 35 CPU-minutes, so all three grids are projected at about 7 CPU-hours. Row construction is one-time geometry setup, not per-step runtime work. Its hot spot was the frozen cardinal/Lagrange primitives' Python loops; they were vectorized on 28 September with the same operation order (outputs byte-identical to the loop versions, including layout). A 60-chunk one-off replay at N32 reproduces 58 chunks bitwise; one differs at 5e-13 relative, far below the replay tolerances. The P06 characteristic action `_absolute_action` is now batched over face points, bitwise equal to the per-point version and 3× faster per call. Measured N32 chunk speedups: P06N faces 1.5×, P06N raw 1.6×, P05N faces 1.6×, P05N raw 1.9×; the same chunk as above is the only one not bitwise identical. Remaining setup cost: the face metric evaluation and the `_coupled` near-axis fits.

**`frozen_v1` global campaign returned, 28 September 2026 (accepted together with `upwind_v1`).** Campaign `05be9063`, job 58976096; see the [local analysis](../../../../work/p05n_field_derived_05be9063_20260927T230447Z_6140a1/local_analysis/report.md).
- **Integrity:** all 18,402 chunk receipts are valid, reassembly is exact, and a local replay of 24 chunks agrees to 7e-13.
- **Headline gate:** `global_order_pass=true` for all 12 gated pair × candidate combinations. Centered orders are 2.31–3.41 and centered-plus-jump orders 3.06–3.54 over both intervals (fine intervals 2.92–3.54).
- **Controls:** constants ≤ 2.2e-12, and the wall-adjacent live jump is exactly 0.
- **Limitation:** the frozen fields lie in the reconstruction's exactness space (above). So the live jump is roundoff on interior radial and θ faces, and upwinding is exercised only on η faces and at the n−2 interior/boundary interface.
- **Supplementary local test:** a u ≥ 0.5 outer-region run with two rich fields ([report](../../../../work/p05n_rich_fields_20260928/report.md)) activates the interior jump. Every region and candidate converges at order ≥ 4.4 (pre-asymptotic).

**`upwind_v1` catalogue — user decision, 28 September 2026**, recorded in [p05n_upwind_catalogue.json](../../../scripts/p05n_field_derived_global/p05n_upwind_catalogue.json) and now the package's active catalogue:
- rich_a × rich_f and rich_f × rich_a (Neumann × Neumann), and rich_a(D) × rich_f (main pairs);
- a held-out pair with the new, unevaluated `heldout_rich_g`, once with a Neumann and once with a Dirichlet generator;
- b1 × e3 as a gated regression tie to `05be9063`;
- constant controls.

rich_a and rich_f were seen in the outer-region test and are disclosed as main pairs only. P05N acceptance is decided from `frozen_v1` and `upwind_v1` together.

**`upwind_v1` global campaign — returned and accepted, 28 September 2026.** Commit `43250ccf`, job 58995223; see the [local analysis](../../../../work/p05n_p06n_43250ccf_20260928T053254Z_c415a4cd/local_analysis/report.md).
- **Headline gate:** `global_order_pass=true`. Main and held-out pairs run at 5.17–5.87 for both candidates. The b1×e3 regression pair reproduces `05be9063` digit for digit (centered 2.91/3.14, with jump 3.31/3.46).
- **Upwinding exercised:** the live jump is active everywhere except the zero-jump families of the outer two layers. It converges at about 3.5, and its share of the error grows from about 8% (N32) to 30–70% (N64).
- **Pre-asymptotic orders:** the 5–6 orders come from the θ error on harmonics 4–6 (outside the 7-point stencil's exact space). They are not the asymptotic rate, which the low-degree fields put at about 3. The lowest-order region is the RLP aggregation transition (about 3.0 on N48→N64).
- **Integrity:** O ≡ R exactly, antisymmetry ≤ 7e-15, and a local chunk replay agrees to 6e-13.

**Gate:** full centered and centered-plus-live-jump actions qualified with the
new input contract and unchanged Dirichlet replay. No production/default or
full evolution promotion follows automatically from this static pass.

### P06N — Physical-normal Neumann curvature qualification

**Status: passed — user-accepted static qualification, 28 September 2026**, under the recovered-trace wall-exterior contract. See the [acceptance record](../../../../work/p05n_p06n_43250ccf_20260928T053254Z_c415a4cd/local_analysis/acceptance_decision.md).

**Bounded construction — passed, 28 September 2026** ([results](../../../../work/p06n_bounded_20260927/)). At 55 selected owners per grid, all checks pass on N32/N48/N64:
- replay of accepted P06 against a mirror of its assembly: at most 3.5e-4 of an atol + rtol = 1e-12 tolerance. The mirror matches the saved Perlmutter artifact to at most 2.7e-10 absolute on values up to 3.15, consistent with cross-platform roundoff amplified by the finite-difference curvature;
- Neumann condition ≤ 6.8 and row linearity 0;
- constant controls ≤ 8e-13;
- the physical-wall characteristic correction and the zero-jump families exactly 0 under the recovered-trace contract;
- ω-independence 0, exact input O ≡ R, and JAX lowering.

The provisional bounded fields were in the exactness space, so these checks are structural.

**Accepted P06 seam defect — found and fixed 28 September 2026.** The accepted global run planned all 3(n+1)n² face slots, and θ slot n and η slot n alias slot 0 with identical lower and upper cells. Run 58880303 therefore applied every θ = 0 and η = 0 seam face's characteristic correction twice.
- **Why it was invisible:** those corrections were about 1e-12 or smaller for its fields.
- **Scope:** the centered candidate is unaffected. The upwinded (U) qualification carried the defect.
- **Fix:** `scripts/p06_structured_global/numerics.py` now skips the aliases (`_periodic_duplicate_face`, with a test). The accepted result stays pinned to `2458dbf6`.
- **Requirement:** P08/P10 production integration must use the deduplicated census.

**Frozen catalogue — user decision, 28 September 2026**, in [p06n_catalogue.json](../../../scripts/p06n_field_derived_global/p06n_catalogue.json):
- n, Te, Ti = rich_f, rich_a, rich_c (Neumann), outside the exactness space so the characteristic correction is active;
- φ either Neumann (rich_a − 1) or Dirichlet (the zero-trace φ = 0 wall);
- a held-out case pair with the new, unevaluated `heldout_rich_h` as n;
- an all-Dirichlet rich case (accepted lifted rows), which re-qualifies accepted P06's upwinded operator with the corrected census;
- constant controls; ω inert.

**Global package — prepared 28 September 2026** (`scripts/p06n_field_derived_global/`):
- **Structure:** batched q1 raw-cell and q3 face stages over the accepted face index space. Periodic aliases are skipped; internal aggregation seams are kept, as accepted P06 does.
- **Campaign preflight** (all through the batched kernels, about 2 minutes for three grids):
  - accepted-P06 replay in accepted-census mode, headroom 0.12 against atol 1e-9;
  - the structural gates;
  - a seam check showing the deduplicated census removes exactly one duplicate contribution.
- **One-off `verify-equivalence` at N32:** the batched kernels match the per-owner oracle at 9% of an atol + rtol = 1e-12 tolerance, over 15 owners × 14 variants.
- **End-to-end:** an N32 local run completed operationally in 23 minutes on 6 workers, about 2.1 CPU-hours. That projects to about 26 CPU-hours for three grids.

**Global campaign — returned and accepted, 28 September 2026.** Commit `43250ccf`, job 58995223 (18 CPU-hours); see the [local analysis](../../../../work/p05n_p06n_43250ccf_20260928T053254Z_c415a4cd/local_analysis/report.md).
- **Headline gate:** `global_order_pass=true` on all 30 gated case × equation × candidate entries, including the held-out cases. Centered orders are 4.38–6.15, U orders 4.47–6.12, and ω (reported) 4.0–5.8.
- **Characteristic correction exercised:** its norm is comparable to the discretization error (about 1e-4 at N64), and it converges at 4.2–5.4.
- **Structure:** the φ-Neumann and φ-Dirichlet variants have bitwise-identical material and correction arrays; φ enters only through the q1 remainder.
- **Seam re-qualification:** the all-Dirichlet rich case re-qualifies accepted P06's U operator with the deduplicated census (U orders 5.1–6.0, N−D exactly 0).
- **Limitations:**
  - The orders are pre-asymptotic, as in P05N.
  - The RLP aggregation transition region is the weakest region: N48→N64 orders 2.2–2.6 (U), up to 30–40% of the squared error in some cases.
  - Exact-input defect 0, wall-exterior defect 0, and the zero-jump families are exactly 0. Local chunk replay agrees to 2.3e-10 (raw) and 5e-13 (faces).

Preserve the accepted P06 Dirichlet qualification.

- Qualify the full accepted midpoint curvature construction: q1 midpoint
  material/remainder terms with their J/B measure, q3 characteristic corrections,
  and the total coupled action. A scalar trace or gradient test alone is not
  sufficient.
- State which fields receive physical-normal data, taken from each field
  (`g_N = n · grad_x f`) per the field contract above, and recover their wall
  traces consistently. Separately freeze a compatible physical characteristic exterior
  state and wall law. A scalar Neumann datum does not determine that state;
  do not fill the gap with the exact unknown manufactured wall state or impose
  incompatible independent constraints. Respect the existing wall-rung contract.
- Run bounded N32/N48/N64 checks with field-derived nonzero data, nonconstant
  traces, eta dependence, wall/seam/transition owners, held-out fields and the
  row-linearity/zero-data check. Report each
  material/remainder/characteristic contribution and the total, together with
  boundary residuals and JIT/JVP on smooth admitted states.
- Once both reconstruction and physical wall-state contracts pass, run the
  global midpoint MMS campaign with existing order and regional criteria,
  complete owner reductions and unchanged Dirichlet replay. Report the exact
  field/closure combinations covered; do not generalize to all wall rungs.

**Gate:** complete curvature actions qualified for the declared Neumann and
physical-wall combination. Missing characteristic wall-state compatibility
blocks P06N, not a separate P07-only static campaign. No new global elliptic
reconstruction solve or mandatory integrated MMS reference is introduced.
The 27 September recovered-trace wall-exterior contract (see P07N above) supersedes the
separately frozen wall-state requirement for this static gate. The physical characteristic
wall state and wall law move to the rung wall-law qualification.

### P08 — Certify the combined frozen HSX perpendicular RHS

**Dependencies:** P05–P07, shared extraction/replay, and P05N/P06N/P07N.

- Consolidate and replay the qualified operators' shared components as
  described in the integration-sequencing contract above. Assemble them through
  a common opt-in verification path; this is not production/default promotion.
- Use the current MMS configuration with parallel terms disabled or reported
  separately.
- Include the qualified Dirichlet and explicit Neumann boundary variants for
  their supported field/physical-closure combinations; keep results separate.
- First prescribe exact manufactured phi; then repeat with reconstructed phi to
  expose polarization error.
- Report each perpendicular term and their sum.
- Use the same **32/48/64** HSX sequence established in the first audits. Any
  extension to resolve asymptotic behavior is an explicitly scoped follow-up;
  an inconclusive sequence remains incomplete.
- Recheck geometry accuracy and the qualified reference convention on the finest
  accepted mesh; use bounded reference spot checks rather than mandatory
  full-domain high-order quadrature.
- Test matched single-device and eta-sharded execution.

**Execution plan — 28 September 2026.** Status: step 1 accepted; steps 2 and 3 done (30 September); operator-change bundle (3b): autodiff K, q2 (P05/P06) and inner donor support C3 adopted 30 September and locked 1 October after the remote C1/C2/C3 campaign; the bundle is complete and final. Standalone campaign A dropped (user decision, 30 September): see step 4. **P08 passed (user decision, 2 October 2026); see step 7.** Step 4 complete (1 October). Step 5 complete (2 October): φ solver chosen and the 5.3 combined campaign passed; φ is Dirichlet only, with Neumann-type φ in P11. Open: step 6 (transverse-wave check, matched single-device/η-sharded execution, N64 geometry/reference recheck), a one-time artifact rebuild and reference re-freeze if the B-field evaluator changes, and step 7. Update each step's status here as it completes.

Starting point (code inventory, 28 September):
- Every qualified action is computed only by host NumPy in `scripts/`, across six packages that each reimplement the runner, observation functional, wall lattice and face census.
- The package already has JAX apply kernels for:
  - point rows;
  - Neumann point rows;
  - P07 integrated face rows;
  - the P05 jump;
  - P06 q1 and q3 corrections.
- Still missing:
  - a package version of the P05 midpoint bracket;
  - a combined RHS;
  - any η-sharding.
- `fci_perpendicular_bracket` is the superseded face-flux bracket and is not a basis for P08.

Steps:
1. **Consolidate the host layer** (required by the integration-sequencing contract).
   - One shared P-path layer for the wall lattice, observation functional, deduplicated face census, quadrature/context/sidecar, and a single chunked runner.
   - A field-independent **per-grid row artifact**: all point, side, Neumann and integrated rows, plus face geometry, built once per grid with an identity. Each operator then becomes an application of that artifact.
   - Gate: a one-off replay of the six accepted campaigns' saved owner arrays (complete N32; N48/N64 owners preselected across every region), within 0.1% of each archived spatial error. The accepted P06 run is replayed with its duplicated seam census.
   - Decisions (user, 28 September; design in `work/p08_step1_consolidation_design_20260928/design.md`):
     - The one-time setup goes in a new package layer, `drbx.stencils`, built after `drbx.geometry`: the face census, geometry coefficients at stencil nodes, and the row artifact with its I/O and identity.
     - The research harness is `scripts/p_shared/`.
     - The six accepted campaign packages stay frozen as oracles.
     - The operator geometry is the frozen MMS-reference metric.
     - Precontracting the P07N Neumann rows (a summation reorder of about 1e-15) is accepted.
     - Polarization is Boussinesq, a physics model choice: the geometry-only tensor J(g^ij − b^i b^j) that P07 qualified and production uses. The artifact stores rows already contracted with it. A density-weighted, non-Boussinesq coefficient would be a separate, separately qualified extension.
     - ~~Build and replay N32 locally.~~ Superseded by a later user decision the same day: every full-grid build and replay, N32 included, runs remotely as one unit-parallel campaign (`scripts/p08_step1_global/`). Local runs are bounded unit subsets only. A monolithic local N32 replay ran for 50 minutes at 7 GB without output before it was stopped.
     - The step-1 artifact is plain CSR. Step 2a replaced its storage with schema v3 and exact tensor factoring.
   - Status: **accepted, 29 September (user decision)**, on the remote campaign below plus the bounded owner closure at every grid.
     - **Artifact schema v2.** Neumann rows are tagged by request (R1–R4) and radial degree: at wall faces, R2 and R3 rows share a key but use degrees 4 and 3. R3 side rows are value-only. N32 size: 9.2 GB. Row counts: 326,656 point, 195,584 Neumann, 91,904 integrated.
     - **Parallel geometry stage** in 4096-row units. The 4096 chunking is a numerical parameter: moving batch boundaries changes the finite-difference-derived K and P07 divergence by up to about 4.5e-9 relative. Bounded units reproduce the serial N32 geometry bitwise.
     - **Unit-parallel replay** (`scripts/p_shared/replay_units.py`):
       - It reads stored rows, including the tagged Neumann rows, face geometry from `geometry.npz`, and wall data cached once per grid on the n×n lattice.
       - It pins each oracle's catalogue explicitly.
       - The owner-closure check (`scripts/p_shared/owner_closure.py`) compares the replay with the six frozen campaigns at 12 owners spanning the interior, wall, transition, aggregate, both seams and the axis, with every incident face. All 143 campaign × term entries pass at N32, N48 and N64: roundoff for P05/P05N/P07 (≤ 4e-13), and ≤ 3.4e-10 for P06/P06N/P07N (the cross-platform finite-difference floor). It is the campaign's bounded preflight at every grid.
     - **Replay defects found and fixed** before the gate passed:
       1. Neumann rows were rebuilt at the wall-projected trace point instead of the true query point.
       2. Plain P07's wall families use the Dirichlet lift, not Neumann restoration.
       3. The P05N/P06N exterior-side Neumann fallback was wrong.
       4. P06N face corrections are stored undivided.
       5. P07N family-0 rows sit at u = 0.
       The stored artifact rows were correct throughout.
     - **Remote campaign** (Perlmutter, one CPU node, commit `e936dac4`; [results](../../../../work/p08_step1_e936dac4_20260929T140249Z_52372a/)):
       - Scope: complete build and replay at N32 and N48. N64 was cancelled by decision during its face stage: the gate needs only preselected owners there, which the owner closure covers, and the N64 CSR artifact (about 75 GB) is not a usable runtime operator (see step 2).
       - Tier B passes every term of all six campaigns. The worst ratio to the archived spatial error is 1.8e-10 at N32 and 1.7e-9 at N48 (P06 finite-difference floor), against the 1e-3 tolerance. P05/P05N/P07 are at ≤ 2e-11.
       - The pointwise caps flagged entries in p05, p05n_frozen, p07 and p07n. They are all roundoff. The entrywise cap, 1e-11 × |saved|, cannot be met where an entry is far below the terms that produced it:
         - the p05 jump check compares with the archived U − A, a subtraction of terms up to 36 leaving results 3–7 orders smaller; the largest difference is 3.5e-15;
         - the other 1–3 entries per term differ by 8e-19 to 8e-14, about 1e-16 of their array's scale.
       - The cap rule was corrected after this run, as an explicit record: it now scales with the term's column magnitude (for the p05 jump, the centered term's), so step 2's replay against these outputs uses the corrected rule. Re-reducing the returned N32/N48 replay outputs locally under it: every term of all six campaigns passes, with zero pointwise violations and unchanged Tier-B ratios ([re-reduction](../../../../work/p08_step1_e936dac4_20260929T140249Z_52372a/local_rereduction_corrected_cap/)).
       - Operational fixes from the run: per-worker peak RSS (forked workers reported the controller's high-water mark; the controller itself peaks near 29 GB at N48), resume after an assembled artifact (the remote worker regenerated N32/N48 geometry, and it hashed identical to the originals), and the replay report's host label.
       - Cost: N32 build plus replay took about 23 minutes of wall time on 96 workers. N48 ran across restarts, so it has no single wall time.
     - **Open performance item:** face replay units take about 40 s per N32 unit warm, against a 30 s target.
2. **JAX application layer.**
   - **2a, artifact layout and runtime loader: done 29 September** ([design and measurements](../../../../work/p08_step2_layout_loader_design_20260929/design.md)). Numerics unchanged: every stored weight decodes bitwise, and every new apply path matches the per-row kernels to roundoff (≤ 5e-16 relative).
     - **Schema v3** (`drbx/stencils/artifact.py`) stores donor lists, tags and boundary queries once per source instead of once per quadrature node: about 1.5× smaller than v2 on face rows. v2 remains readable; only v3 is written.
     - **Exact tensor-factored rows** (`drbx/stencils/tensor_rows.py`). The unconditioned singleton, ringwise and centered_radial rows are products of 1-D Lagrange factors (ringwise: per-ring angular vectors). The builder captures the factors it multiplies (`rows_with_factors`) and stores bit-deduplicated tables plus a few indices per node. Decoding reproduces the rows bitwise, checked on every source at encode time with CSR fallback (0 fallbacks on real N32 units). On real N32 units: 400 → 129 MB on disk; pure singleton/ringwise units shrink 25–135×. Coupled-quartic and the conditioned wall families stay CSR. Estimated N64 point rows: v2 about 68 GB, v3 45 GB, factored about 5.5 GB.
     - **Vectorized loader** (`drbx/stencils/loader.py`) lowers chunks to JAX payloads without per-row Python loops, streaming per chunk, with grid-global deduplicated boundary queries and factor tables, and per-source η-offset halos for sharding.
     - **Kernels** (`native/fci_perpendicular_source_rows.py`, `native/fci_perpendicular_tensor_rows.py`). CSR sources gather donors once per source and keep gradients only where stored. Tensor sources never materialize weights: θ is precontracted once per apply per used (θ entry, layer) and per ring entry, then each node gathers 16 rows and contracts η and radial. Eager and JIT are bitwise equal, and JVP is linear. On real N32 units the runtime plan is 120 MB against 398 MB all-CSR (the remainder is coupled-quartic), and apply takes about half the CSR time.
     - The one-time artifact migration tools were removed.
   - **2b, JAX operator assembly: implemented and locally gated 29 September** ([design, API and gate results](../../../../work/p08_step2b_operator_assembly_design_20260929/design.md)). The spec is the step-1 host replay, including its five fixes.
     - **Package layers.**
       - `drbx/stencils/operator_plan.py`: one field-independent plan per grid or bounded owner set. It holds the loader payloads, `geometry.npz` geometry, census maps, precomputed q1 evolution volumes, and two grid-global boundary point tables that every payload indexes.
       - `native/fci_perpendicular_reconstruction_state.py`: per-field Dirichlet/Neumann reconstruction with the host's missing-side rules.
       - Operators: `native/fci_perpendicular_midpoint_bracket.py` (the package P05 bracket), `fci_perpendicular_p05_operator.py` (centered plus live jump; P05N per-role D/N), `fci_perpendicular_p06_operator.py` (q1 plus q3, including the wall solve and the legacy seam multiplier), `fci_perpendicular_p07_operator.py`. They reuse the qualified JAX kernels.
     - **Harness.** `scripts/p_shared/campaign_fields.py` moves the campaigns' boundary callbacks out of the replay closures (host replay bitwise unchanged). `scripts/p_shared/jax_replay.py` builds the JAX owner closure and outputs the host's owner-term format.
     - **G1**, bounded owner closure at N32, N48 and N64, all seven campaign keys:
       - JAX vs host: non-cancellation terms ≤ 1e-11 of scale (worst P07N-D, 7.7e-12 at N64). Cancellation terms (face jumps, q3 corrections), ≤ 10× their measured one-ulp conditioning floor: worst 0.87× the floor.
       - `compare_to_oracle` with the JAX terms: 143/143 rows pass at every grid.
       - Policy decision: cancellation terms carry no fraction-of-scale bound. Their scale shrinks with refinement below input noise; at N64, P05N-frozen `face_D` has a floor of 7e-8 of its scale.
     - **G2:** eager and JIT are bitwise equal. JVP equals the linear action (P07; P06 q1 in the gradients) and agrees with finite differences for P05/P06 (≤ 5e-10).
     - **G3, accepted 30 September (user decision):** full-grid JAX replay at N32 and N48 on Perlmutter (`scripts/p08_step2_global/`, commit `1ad24091`; [results](../../../../work/p08_step2b_g3_1ad24091_20260930T034152Z_d2297c/)). MMS reference terms stay host-only and were gated in step 1.
       - P06-legacy, P07 and P07N pass every term. Every real-variant Tier-B ratio is ≤ 2.4e-8, against the 1e-3 tolerance.
       - The literal step-1 criteria flag two structural effects, neither a defect:
         - P06N raw terms show a ratio of 0.13 on the four constant-field control variants only. Their exact result is zero, so the archived error is roundoff (about 1e-12). The ten real variants are ≤ 5.6e-10.
         - The entrywise pointwise cap flags the P05 per-face jumps and P05N `face_N`/`face_D` (cancellation terms): absolute differences 1e-19 to 4e-13, ≤ 1.9e-8 of the column scale. That is the same relative size as at the owner closure, 2–6× the one-ulp floor measured there, and it is not concentrated at small owners. Step 1 passed these entries only because its host replay repeated the frozen NumPy arithmetic.
       - **Gate carry-forward:** do not reuse Tier B on control variants whose archived error is roundoff (give them an absolute floor), or the entrywise cap on cancellation terms (gate them on their conditioning floor).
       - Infrastructure at full scale:
         - 265,216 (N32) and 942,336 (N48) tensor-factored point-row sources, with 0 fallbacks.
         - Face rows 1.25 / 3.19 GB (step 1: 8.4 GB at N32); whole artifact about 2.3 / 5.8 GB, of which Neumann rows are now the largest part (0.86 / 1.93 GB).
         - Build 10 / 19 min; JAX replay 5.5 / 16 min at 5 / 15 GB peak; 56 min in total on one CPU node.
   - **Step 2 closed, 30 September.**
3. **Combined perpendicular RHS: done 30 September** ([design, API and gate results](../../../../work/p08_step3_combined_rhs_design_20260930/design.md)). An opt-in verification path, not production.
   - **Scope (user decision):** the fields n, Te, Ti and ω, with φ prescribed; polarization is step 5. Vi/Ve are deferred; production gives them bracket and diffusion only. The internals are field-generic.
   - **`native/fci_perpendicular_rhs.py`:** `perpendicular_rhs` returns per-field terms under production's names. One shared cell/face reconstruction feeds P05 and P06; P07 uses its integrated rows.
     - `poisson_bracket` = `P05(φ, g)/ρ*` (production's `−[φ,g]/ρ*`).
     - `curvature` = P06 q1 + q3 correction / evolution volume.
     - `perpendicular_diffusion` = `−D_f·P07(f)`. P07 is the positive operator −∇·(P⊥∇f). This sign was corrected during design after the host reference found it.
   - **Sign audit** against the published normalized Boussinesq GBS equations (arXiv:2508.04881 §2; Giacomin et al. 2022) and production's code:
     - The bracket, C(f), all four curvature rows (the code's M plus the ψ = φ + τTi remainder, reduced symbolically) and the diffusion match exactly.
     - The upwind corrections have no continuum counterpart; they are covered by the convergence checks.
   - **Gates,** bounded owner closure, N32/N48/N64:
     - G3.1: the combined call is bitwise equal to the separate operator calls.
     - G3.2: each frozen campaign's term through the combined path reproduces E6's operators (bitwise or roundoff), and `compare_to_oracle` passes 143/143 rows at every grid.
     - G3.3: against the host published-form continuum reference (`scripts/p_shared/perpendicular_reference_rhs.py`) on the P06N `main_*` states, every term converges. The worst relative L2 is 3.6e-2 / 6.0e-3 / 7.3e-4, and the least-squares scale fit is 0.9997–1.0000 at N64, where a sign error would give −1.
     - G3.4: eager and JIT are bitwise equal; JVP matches finite differences to 3e-10.
     - Sharing the reconstruction gives no measurable speedup at the closure size.
   - **Also 30 September:** the comparison core gained a roundoff-floor Tier B for control variants and conditioning-floor caps for cancellation terms (`replay_support.py`). Re-comparing G3 under them passes every term at N32/N48.
3b. **Operator-change bundle, before the full-grid campaigns (user decision, 29 September).** The step-6 candidates that change the qualified operator — autodiff curvature K, q2 face quadrature for P05/P06, and the transverse-reconstruction support at the coupled/ringwise switch — are settled here, after G3 and step 3, so the expensive campaigns run once on the final operator.
   - For each candidate: the bounded checks listed in step 6, then a decision, then re-frozen references and re-qualification through the JAX owner closure.
   - G3 runs first, because afterwards the step-1 frozen oracles no longer describe the operator.
4. **Full-grid artifact build and smoke check on the final bundle operator.** This replaces a standalone campaign A (user decision, 30 September 2026).
   - **Status: complete, 1 October 2026 (commit 53e068af, remote job 59158624).** `scripts/p08_step4_global/` built N32/N48/N64 artifacts with the options pinned explicitly (autodiff, q2, fixed_radius). Results: [local analysis](../../../../work/p08-step4-53e068af-20261001T134159Z-2a344d/local_analysis.md).
     - Gates pass on every grid. Preflight: 29/29 host-vs-JAX policy rows. Smoke: 0 non-finite out of 10.8M / 36.4M / 85.7M term elements. Validate also passes.
     - Artifacts are 1.77 / 4.94 / 11.0 GB, of which the P07 rows are 0.17 / 0.57 / 1.34 GB. There are 70,656 / 228,096 / 528,384 coupled-quartic point rows and no tensor-encoding fallbacks. The N64 full replay peaks at 41 GiB.
     - Change against the frozen fd/q3/profile7 oracles (informational):
       - P07: ≤ 8% of the archived error. P07N: unchanged (≤ 1e-5).
       - P05/P05N raw terms change only in the C3 region, by 0.5–0.9×.
       - The P05N face term changes by a constant factor of 2–3×, converging at the same order: the known q2 effect.
       - P06 legacy changes by up to 1.55× in the C3 interface band (the documented exception).
       - At rings 5–10, P06N error drops from about 5e-4 to about 1e-5 (C3).
     - The one apparent regression, P06N at the N64 physical wall (ratio 2.34), is the frozen reference's finite-difference K. At 8 wall points (one per field period, stellarator-symmetric) FD K is off by 3e-4 at step 2e-4 and converges onto autodiff as the step shrinks. The old operator carried the same error and cancelled it ([check](../../../../work/p08_p06n_wall_n64_20261001/README.md)). Resolved by the step-5.3 re-freeze: the references now use autodiff K as well.
     - **The FD-curvature option is kept.** New campaigns use autodiff K, and so do the 5.3 references. Removing FD would break:
       - replay of the accepted campaigns: the step-1/2 runners, `replay_units`, `curvature_gates` and the frozen-reproduction tests all pin `curvature="fd"`;
       - artifact builds, which still compute the perpendicular divergence by FD.
   - **Why:** a prescribed-φ campaign on its own would be repeated anyway. Its unique value, separating forward-operator error from φ-solve error, comes from a prescribed-φ arm inside the step-5 campaign. The full perpendicular RHS (Vi/Ve, sheath and wall BCs) will need its own full-grid campaign later.
   - **Remote:** build N32/N48/N64 artifacts on the final operator (autodiff K, q2 for P05/P06, the chosen inner support). Clean-export check first. Step 5 needs these artifacts anyway.
   - **Smoke only, no MMS reductions:**
     - build receipts, row and family statistics, and artifact bytes;
     - JAX replay finite on every owner;
     - host-vs-JAX agreement at sampled owners under the step 3.0 rules.
   - **Accuracy before step 5:** provisional full-grid errors from bounded stratified global samples with the difference estimator, as for q2 (`work/p08_q2_face_quadrature_20260930/`).
5. **Reconstructed φ — user decision, 28 September.** The solver must invert the same operator the forward RHS uses for the perpendicular Laplacian. Keep the production `LocalPerpLaplacianInverseSolver` machinery (matrix-free FGMRES, preconditioner, augmented-Neumann gauge), and add an `operator_form` that applies the qualified P07 action from the row artifact.
   - Sequence:
     1. Check that the new form's apply matches the P07 host action.
     2. Qualify the solve against a host direct sparse solve of the same operator at N32/N48. That solve is an oracle only; it separates the linear-solve error from the discretization error.
     3. **Combined full-grid MMS campaign, remote: the first full-grid qualification of the bundle.**
        - N32/N48/N64 on a frozen catalogue, with held-out fields, and Dirichlet and Neumann variants kept separate.
        - Rich fields (upwinding active) and low-degree fields (asymptotic rate), plus Q's transverse-wave fields.
        - Two arms in the same run: **prescribed exact φ** (the forward-operator control) and **solved φ**.
        - Report each term and the sum, with regional budgets that name the RLP transition region.
        - This re-freezes the references, replacing the frozen fd/q3/C0 step-1 oracles.
   - **Preconditioner development is expected.** `fci_polarization_coarse` was built for the production RLP operator, so its convergence on the P07 operator must be measured, and new preconditioners are likely needed.
   - All-Neumann φ must first establish that the operator's null space is exactly the constants before `solve_augmented_neumann` applies.
   - **Decisions and preparation, 1 October 2026** ([design notes](../../../../work/p08_step5_phi_audits_20261001/design_notes.md); [wall-condition derivation and literature](https://claude.ai/artifact/N56XrXUwQoJ1vN9uR3UuHp)):
     - **Bounded audit, N32.** P07 has no negative direction on any patch tested. A·1 is at roundoff. The operator is non-symmetric by 2–9% in the interior and up to 15% at a Dirichlet wall.
     - **Neumann compatibility.** The logical wall is not a flux surface: |b·n̂| is 0.043 RMS at u = 1, with a maximum of 0.156. So under normal Neumann the left null vector is not the volume weights, and the augmented solve's λ is an O(1) inconsistency.
     - **Prepare both inversions (user decision).**
       - Dirichlet φ, as in GBS and GRILLIX, comes first.
       - Normal-Neumann φ runs in report mode: check the null space, compare the left null vector ℓ with M·1, and report λ.
       - The conormal and mixed conditions wait for the wall-model work.
     - **Build remotely, solve locally (user decision).** P07 is exported as sparse `A u + B g` and solved locally. The production `LocalPerpLaplacianInverseSolver` integration follows when φ is wired into the simulation.
       - `drbx.native.fci_perpendicular_p07_sparse` exports for both kinds. On the real N32 closure it equals `p07_action` to 3e-16 / 7e-16 relative.
       - `drbx.native.fci_perpendicular_p07_solve` runs solvax FGMRES in the M-inner product, with a Jacobi or user-supplied preconditioner, plus a sparse-LU oracle.
       - The remote export of the step-4 artifacts is `scripts/p08_step5_export/`.
     - **Reference finite differences** ([check](../../../../work/p08_step5_reference_fd_check_20261001/README.md)). At the same 8 wall points, the reference's FD divergence ∂_i(J P⊥^{ij}) is off by 7e-5 at the default step. That is negligible for P07's errors, but the 5.3 re-freeze should compute it by autodiff, as for K.
   - **Local solver studies on the exported full-grid P07, 1 October 2026** ([notes](../../../../work/p08_step5_solver_studies_20261001/README.md)). Export: `work/p08-step5-export-2ff50718-20261001T165909Z-ebcbd2`. It reproduces `p07_action` to ≤ 1e-15 relative, and with the harness wall data it reproduces the frozen P07N D/N actions to ≤ 2e-11. Harness: `scripts/p08_step5_local/`.
     - **Dirichlet φ.**
       - Positive definite: generalized λ_min(sym MA) = 177 against λ_max = 2.1e6 at N32. 10.5% non-symmetric.
       - Solving the frozen discrete action returns φ̄ to 7e-14 (direct) or 2e-10 (FGMRES).
       - The solve's own discretization error (rhs = exact cell-average flux O_q3) converges at **order 3.9–4.0**: 4e-8 to 3e-6 relative at N32, 2e-9 to 2e-7 at N64.
       - The headline error against the midpoint reference R is 1.2e-4 to 2.3e-4 at N64, at orders 1.5–2.1. It is entirely A⁻¹(R − O), the reference mismatch.
     - **Normal-Neumann φ, report mode.** The kernel is exactly the constants, but there is a near-null cluster of toroidal-only modes.
       - Eigenvalues are 0.002–0.09, several negative, against 177 for Dirichlet.
       - The continuum normal-Neumann quadratic form is itself indefinite on toroidal modes. The bulk term is m²⟨J P⊥^{ηη}⟩, which is 7e-4 to 5e-2 because P⊥^{ηη} is 0.15% of g^{ηη}. The oblique wall term, −½∮f²∇_Γ·(b_n b_t), is ±3e-2 to 6e-2.
       - Consequences: the left null vector is 81× away from the volume weights, and Neumann φ errors run from 1e-3 up to 100× the field.
       - A conormal condition would remove only the wall term. Dirichlet removes both.
       - **φ wall condition: Dirichlet only through P08–P10 (user decision, 2 October 2026).** Every Neumann-type φ condition is deferred to [P11](#p11--φ-wall-model-bookkeeping-for-future-physics-work-not-a-gate), which is bookkeeping, not a gate:
         - normal Neumann (Loizu's magnetic-presheath-entrance condition);
         - conormal Neumann (the polarization-current condition);
         - a mixed Dirichlet/Neumann condition chosen by grazing angle.
       - **Open issues recorded in P11** ([grazing-angle map](../../../../work/p08_wall_grazing_angle_20261002/)):
         - Normal Neumann's compatibility condition is weighted by ℓ, not the volume weights, so λ is spurious.
         - Every Neumann form leaves the constant potential level free.
         - The toroidal near-null modes remain.
         - The Bohm sign flips across b_n = 0, so the data are discontinuous at tangency curves.
         - The form that differentiates along b degenerates at tangency.
         - The theory does not hold below the critical angle: 10% of the wall is below 0.3° and 33% below 1°.
         - A mixed condition with a sharp switch is singular at the junctions (the Zaremba problem).
     - **Linear solver (user decisions).** FGMRES right-preconditioned by **block-Jacobi over η planes**, with each plane block solved exactly by a JAX ring-block LDU in **float32**, **warm starts**, and **default rtol = 1e-8**. The matrix is applied as BCSR, which is 6.6× faster per matvec than BCOO.
       - Iterations are 10 / 9 / 9 at N32 / N48 / N64, against Jacobi's 341 / 566 / 760.
       - At N64 a solve takes 0.73 s cold at 1e-10. Warm starts take it to 0.31–0.59 s at 1e-8.
       - Over 100 solves this is 44× faster end to end than Jacobi, with 369 MB of factors.
       - Rejected on wall time: smoothed-aggregation AMG (its coarse operators densify), ILU (setup grows about 10× per refinement), Chebyshev, and block Gauss–Seidel across planes.
   - **Step 5.3, static Dirichlet-φ combined campaign: complete, 2 October 2026** (`scripts/p08_step5_combined/`, commit f4de626e, remote job 59177170; [local analysis](../../../../work/p08-step5-combined-f4de626e-20261001T231445Z-a13f6c/local_analysis.md)). It ran on the step-4 artifacts, so nothing was rebuilt.
     - **Results.** All solver gates pass at N32/N48/N64: the consistency error is ≤ 4e-11, and the ψ solve takes 10–11 iterations at rtol 1e-11.
       - The informational headline order criterion passes for every variant, field and arm. The total-term N−R at N64 is 2e-5 to 1.2e-4, at orders 4.3–5.9.
       - The lowest regional order is 4.1.
       - The solved arm differs from the prescribed arm by 2–3.5e-5 relative at N64, and the error of φ_h is 1–1.4e-5 relative at order about 4.5. Solving φ inside the RHS costs no accuracy or order.
       - The re-frozen references for the final operator are in the campaign folder. **Passed (user decision, 2 October 2026).**
     - **Production φ solver:** `drbx.native.fci_perpendicular_phi_solver`, built from the plan, with `drbx.native.fci_perpendicular_plane_preconditioner`.
     - **Reference divergence:** `drbx.geometry.curvature_autodiff.AutodiffPerpendicularGeometry` and `p_shared.curvature_reference.perpendicular_geometry(..., method="autodiff")`. It agrees with converged FD to 2e-10 per point. Artifact builds keep their FD divergence.
     - **Catalogue:** the P06N variants with Dirichlet φ: `main_phi_dirichlet`, `heldout_phi_dirichlet`, `dirichlet_rich` and `control_constant_dirichlet`. Low-degree and transverse-wave controls move to the step-6 transverse-wave check.
     - **Parameters:** ρ* = 0.05, τ = 1, D_f = 1e-2.
     - **Arms:** prescribed φ̄, and solved φ. The solved arm solves P07_D ψ_h = Ō(ψ) for ψ = φ + τTi, using the exact wall trace of ψ and the exact cell-average flux Ō, then sets φ_h = ψ_h − τT̄i. The measurement uses rtol 1e-11; the production default is 1e-8.
     - **Re-frozen references** (bracket, curvature, diffusion O_q3) are computed with autodiff K, and the midpoint reference of ψ with the autodiff divergence.
     - **Gates:** a discrete-consistency solve must recover ψ̄ to ≤ 1e-8 relative, every solve must converge, and every term must be finite. The headline order is informational; the user decides acceptance.
     - **Timing:** 5.3 ran on the current canonical evaluator. An evaluator change triggers one rebuild and re-freeze (see below).
   - **Magnetic-field evaluator (Q path finding, 1 October 2026; [Q roadmap](parallel_second_order_roadmap.md), "Q07 sharp geometry feature" and "Compact magnetic evaluator"; [qualification](../../../../work/compact_bfield_qualification_20261001/report.md)).**
     - **The finding.** The canonical MAKEGRID B evaluator's periodic cubic toroidal prefilter couples all toroidal planes. Sharp field structure from planes whose R,Z samples lie outside the wall (near the coils) therefore leaks into interior queries.
       - The spline's holdout B error reaches 0.044 T at most. Its div B is 4.2e-3 RMS and 0.12 T/m at most.
       - It drove Q07's N48 density hotspot. The compact toroidal evaluator `compact_c3` (seven raw planes, septic Hermite, C3 in φ, unchanged R/Z cubic) reduces that hotspot by about 650×. It also improves div B to 8.8e-4 RMS and 0.011 T/m at most.
       - `compact_c3` is experimental and uncommitted. Canonical geometry, the default and every frozen campaign are unchanged.
     - **Effect on P so far.**
       - Every P qualification is an MMS consistency check in which the operator and the reference share the same evaluator. So the operator decisions stand, because they were made on N−O, which is reconstruction error with B as a common coefficient: C3 inner support, q2, autodiff K, and the step-4 pass.
       - What can move is O−R, which is coefficient variation inside a cell. P07N's O−R dominates its N−R, with orders 1.4–2.0. The bounded `compact_c3` comparison below found it unchanged, so it is not this artifact.
       - The 8-point FD-K wall feature is ~0.3° from the nearest MAKEGRID toroidal plane, so it is presumably an R/Z-spline effect, which `compact_c3` keeps. That is not verified.
     - **Plan.**
       - Step-5 solver studies on the exported P07 continue. Conditioning, preconditioners and the null-space structure do not depend on ~0.2% B changes.
       - **Step 5.3 ran on the current canonical evaluator (2 October).** If `compact_c3`, or another evaluator, becomes canonical, the geometry identity changes. Then rebuild the step-4 artifacts and re-run the 5.3 campaign once on the final evaluator to re-freeze the references. Fold this into the final P08 acceptance campaign. The decision is pending on the Q path.
       - Candidate P contribution: a bounded matched spline/`compact_c3` replay of P07N and P06N (N−O / O−R / N−R) at core, transition, bulk and wall owners, as the Q roadmap's next item asks.
     - **`compact_c3` is the P baseline (user decision, 2 October 2026).** Q adopted it in commit 8dadf696. P treats Q's evidence as motivation, not qualification; P's own evidence is the bounded comparison below.
       - Naming: "C3" alone means the inner donor support (`inner_support="fixed_radius"`). The magnetic evaluator is always written `compact_c3`.
       - Every new P campaign pins `bfield_toroidal="compact_c3"` explicitly. The code default stays `"spline"` for now, because these packages pass only three options and rely on it:
         - the frozen step-1/2 runners and the frozen-reproduction tests;
         - the completed step-4, step-5 export and 5.3 packages, which would otherwise mix a `compact_c3` environment with spline artifacts.
         - Flipping the default needs those call sites pinned to `"spline"` first. That is deferred to P10 integration.
       - **Option:** `bfield_toroidal` = `"spline"` (default, frozen) | `"compact_c3"` (`scripts/p_shared/bfield.py`), threaded like `inner_support`.
         - The frozen reference builder (`hsx_mms_continuum_reference.py`) and `p07_diffusion_global/numerics.py` are hash-pinned by frozen manifests, so they are not edited.
         - Instead, `p_shared` replaces `reference.bfield_evaluator` after the reference is built and before the provider wraps it. The JAX evaluator for autodiff K and the P07 divergence is derived from it.
       - **What depends on B in P:**
         - b, |B|, h, K and the P07 tensor and its divergence;
         - therefore the artifacts' geometry and rows, and the MMS references.
         - P reads no field-line traces.
       - **Regeneration:**
         - The step-4 artifacts and 5.3 references are rebuilt on `compact_c3`. The spline ones stay labelled spline and are never relabelled.
         - The coordinate map (the `MetricEvaluator` checkpoint) is kept as an independent coordinate input. It was fitted with the spline field, which is recorded in provenance. MMS consistency does not depend on it.
         - The grazing-angle map and the Jarvis wall diagnostic are regenerated through `build_environment(..., bfield_toroidal=...)`.
       - **Provenance:** a non-spline build records `policy.bfield_toroidal` and hashes the B-evaluator sources (`Bfield_evaluator.py`, `compact_toroidal.py`, `jax_bfield_evaluator.py`, `p_shared/bfield.py`). The MAKEGRID file and its currents (via the sidecar) are already hashed. `check_artifact_options` treats a missing key as spline.
       - **Bounded comparison:** `scripts/p08_bfield_eval` (removed; see commit `6c4de657`). For both evaluators on identical owners at N32/N48/N64, it reports:
         - N−R per term, plus N−O, O−R and N−R for diffusion;
         - Dirichlet and physical-normal Neumann variants;
         - six regions × {on-plane, mid-plane} MAKEGRID knot classes;
         - the coefficient change between evaluators.
         - **Result, 2 October 2026** ([report](../../../../work/p08_bfield_eval_20261002/report.md)): 66–72 owners per grid, none dropped, 2 min and < 5 GiB per run.
           - `compact_c3`/spline error ratios are 0.975–1.012 over every variant, field, term, comparison, region and knot class at N32/N48/N64 (2,976 entries). Observed orders are identical. There are no flags.
           - So `compact_c3` costs no MMS accuracy, and it does not shrink the P07 O−R gap either. That gap is about 0.1–0.2 relative in the outer two layers and is unchanged, so it is not the toroidal-prefilter artifact.
           - Coefficient change (RMS/max relative):
             - B: 1e-5 / 2e-4;
             - P07 tensor: 3e-6 / 4e-5;
             - P07 divergence: 2e-4 / 4e-3;
             - K at raw midpoints: 1e-3 / 2e-2;
             - K at wall faces: 1–4% RMS, up to 0.25–0.53 max.
           - MMS cannot say which B is more physical, since operator and reference share it. That evidence is Q's (holdout error and div B).
           - **Limits:**
             - The owner sample is redrawn per grid, so the orders are noisy; the `compact_c3`/spline ratio is the robust statistic.
             - A cell spans several MAKEGRID planes (90 per field period) even at N64, so the knot classes barely separate.
       - **Re-freeze on `compact_c3`: complete, 2 October 2026** (`scripts/p08_step5_compact_c3/`, commit 6c4de657, job 59207210; [local analysis](../../../../work/p08-step5-compact-c3-6c4de657-20261002T150857Z-27485/local_analysis.md)).
         - The N32/N48/N64 artifacts were rebuilt on `compact_c3` and the 5.3 stages re-run. Every gate passes and the headline criterion passes.
         - Every row agrees with the spline 5.3 run within a factor 0.997–1.004 (orders within 0.02). Total-term N−R at N64 is 1.4e-5 to 1.2e-4, at orders 4.3–5.9. φ_h is at orders 4.8 / 4.2. The lowest regional order is 4.08.
         - The references in that folder are the re-frozen references of the final P08 operator. The artifacts stay on Perlmutter scratch (`summary/artifacts.json`). **Passed (user decision, 2 October 2026).**
6. **Remaining gates.**
   - **Bounded geometry/reference recheck on N64: closed (user decision, 2 October 2026).** Evidence:
     - The one N64 anomaly (P06N wall ratio 2.34) was finite-difference K error in the old reference ([N64 P06N wall check](../../../../work/p08_p06n_wall_n64_20261001/README.md)). Autodiff K removed it.
     - The reference's finite-difference divergence was off by 7e-5 ([reference FD check](../../../../work/p08_step5_reference_fd_check_20261001/README.md)). The references now use autodiff K and the autodiff divergence.
     - The `compact_c3` comparison and re-freeze show no N64 anomaly.
     - **Still open and non-blocking:** certification of the MAKEGRID field and the fitted coordinate map against a finer independent source (see "Magnetic-field derivative/reference resolution follow-up").
   - **Scoped change: autodiff curvature K (user decision, 28 September).** Compute K = (B/2J)∇×(b_cov/B) with `jax.jacfwd` through the JAX metric and B-field evaluators. These are the same interpolants the NumPy reference uses, and they agree to 4e-15. The frozen fourth-order finite difference it replaces uses step 2e-4, shrunk near u = 0 and u = 1, with a one-sided rule at the wall.
     - [Comparison](../../../../work/p08_autodiff_curvature_20260928/) on N32 raw midpoints and face nodes:
       - median relative difference 3e-11, 99th percentile ~1e-7, max 1.5e-5;
       - the finite-difference error falls with the step at fourth order, bottoms out near 2e-4, then grows as roundoff/h; at the worst points it converges onto the autodiff value, so it is finite-difference truncation, most likely at MAKEGRID cubic-spline knots;
       - autodiff is ~14× faster per point.
     - Adoption:
       1. Switch the operator and the MMS reference together, so K cancels in the pointwise q1 N−R.
       2. Replace the one-sided wall rule.
       3. Show that the P06/P06N actions change far below their archived spatial errors (bounded check), or rerun a bounded P06 check.
     - Step 1 keeps the finite-difference K, because it must replay the accepted campaigns. After adoption, the face-geometry build cost drops accordingly.
     - **Adopted 30 September 2026 (user decision); the default for new builds is `curvature="autodiff"` (`p_shared.curvature_reference.DEFAULT_CURVATURE`).** [Qualification](../../../../work/p08_bundle_autodiff_curvature_20260930/design.md) at N32/N48/N64 (QK1–QK4):
       - K vs finite difference: median 3e-11. The finite difference converges onto autodiff at 4th order at all worst points.
       - The divergence identity holds at roundoff (≤ 4e-15), against 7e-12 to 4e-11 for the finite difference.
       - Operator change against the archived error: 7e-9, 4e-8 and 5e-6, below the 1e-2 gate. The region error ratio is 1.000000005, and G3.3 is unchanged.
       - Geometry is bitwise chunk-independent (block mode) and about 2× faster.
       - N64 frozen-oracle rows: 20 of 126 move just past the oracles' 1e-5 equivalence clause, at 1.5e-5 to 4e-5. They are FD-baked.
       - The step-1/step-2 campaign runners, `replay_units` and the frozen-reproduction tests pin `curvature="fd"` explicitly.
       - References are re-frozen once, after the whole bundle, through the combined full-grid campaign (step 5.3).
   - **q2 for P05/P06 implemented as an option, 30 September 2026** (`face_quadrature="q2"`, default `"q3"`; P07 stays q3). [Design and gate results](../../../../work/p08_q2_face_quadrature_20260930/design.md):
     - Accuracy on a stratified global sample: the q2 N−R global L2 order is ≥ 3.5 for every P05N pair and P06N case × equation, against the 1.8 acceptance gate. The absolute error is 1.00–1.22× q3 at N64.
     - Implementation: host q2 = frozen `face_chunk(order=2)` to 1e-14; JAX q2 = host q2 (29/29).
     - Combined RHS: G3.3 at q2 converges with fit 1.
     - Cost: the P06 face-apply time halves.
     - **Adopted 30 September 2026 (user decision): the default is `face_quadrature="q2"` for P05/P06, and P07 stays q3** (`p_shared.face_quadrature.DEFAULT_FACE_QUADRATURE`). Earlier bounded wall-weighted evidence: [28 September report](../../../../work/p08_face_quadrature_q2_20260928/report.md).
       - The frozen step-1/step-2 runners, `replay_units`, `curvature_gates` and the frozen-reproduction tests pin `face_quadrature="q3"`.
       - Explicit q3 policies and identities are exactly the historic ones.
       - References are re-frozen once, after the bundle, through the combined full-grid campaign (step 5.3).
   - **Inner donor support: C3 locked, 1 October 2026** ([design](../../../../work/p08_donor_support_c1_20260930/design.md)). User decision: evaluate C1 only against the current C0.
     - C1 (`inner_support="last_aggregate"`) is Q's layout with P's own coupled quartic fit. The coupled fit is used through the last agglomerated ring, so the switch sits at a fixed radius (after ring 10/15/21).
     - Tested with Q's 26-field catalogue and a short-wave (λ = 0.5, 0.25) response report. The default stays C0 (`"profile7"`) until a decision.
     - **Adopted 30 September 2026 (user decision): C3, `inner_support="fixed_radius"`.** The coupled quartic is used for stencils whose anchor-ring centre lies below u = 0.21, which is the measured C1/C0 per-band crossing. C0's rule applies beyond.
       - C1 was worse than C0 in u 0.21–0.40 (1.3–2.4×). C1b was no better than C1.
       - C3 keeps C1's 8–25× gain in u 0.06–0.21, where C0 fails (P07 N−O orders −3.6 and 1.5), and is identical to C0 from u 0.27 out. Its interface band costs 1.6× C0 at order 2.9.
       - Documented exceptions:
         - u 0.12–0.21: P07 N−O order 0.37 on 48→64, error 8× below C0.
         - Near axis: order 0.5–0.9, pre-existing in all candidates.
       - Frozen reproduction pins `"profile7"`.
       - **C2 evaluated and rejected; C3 locked, 1 October 2026 (user decision).** C2 (`inner_support="last_aggregate_nearest28"`) is C1's layout with Q's isotropic nearest-28 donors.
         - Remote campaign `scripts/p08_inner_support_eval` (removed; see commit `078633ce`): C1/C2/C3 at N32/N48/N64, 12 owners per ring (521 at N64). [Results](../../../../work/p08_donor_support_c1_20260930/design.md#remote-decision-campaign-c1-c2-c3-at-n32n48n64-1-october-2026).
         - P07 N−R is identical across C0–C3 to about 1%, because O−R dominates it.
         - C2 has the most accurate reconstruction (pooled 0.11× C0, min order 3.1). But its P07/P06 convergence is irregular: per-ring N−O at N64 is up to 7× C0 at u 0.27. On 48→64 it stalls at u 0.06–0.12 (P07 N−O order 0.32) and u 0.27–0.33 (order 1.1; P06 0.11). It adds 33 new rebound flags, against 10 for C3.
         - C3 confirmed on the larger sample: interface band 0.21–0.27 is 1.8× C0 at orders 5.4/4.4; the band 0.12–0.21 is 8× below C0 at order 1.8/1.4.
         - C1, C1b and C2 were removed from the code on 4 October 2026 (user decision); default numerics unchanged.
   - **Background: transverse reconstruction at the coupled/ringwise switch.** This was found by the Q path and confirmed for P, 29 September. The [P audit](../../../../work/p_transverse_wave_audit_20260929/report.md) and its [follow-up](../../../../work/p_transverse_wave_audit_20260929/followup/report.md) are bounded: Q's 36 owners plus fresh phases and fixed-coordinate tracks, N32/N48/N64, run through P's own assemblies.
     - **Controls.** Fixed-wavelength transverse waves exp(i[2πx/λ + η]) and the y analogue, with λ = 2 and 4, in computational disk coordinates.
     - **Finding.**
       - The ringwise construction (ringwise angular with radial cubic) reproduces Cartesian cubics but not quartics.
       - At the first ringwise layer, its transverse gradient error rebounds 2.2× from N48 to N64 (2.6× at face q3 nodes). λ = 4 shows no rebound.
       - The switch radius moves inward with refinement. A fixed point it crosses flips from coupled quartic to ringwise, and its transverse error jumps 13–60×.
       - The η-free controls reproduce every ratio, so the effect is purely transverse. The η-derivative error is a separate, spatially uniform, roughly third-order floor.
     - **Effect on the operators:**
       - P06 q1 and the P05 bracket with a smooth partner amplify the rebound in place (about 1.9×).
       - The P05 wave × wave bracket and the P05 jump do not rebound.
       - P07 carries it through the radial face shared with the switch layer into the next layer out (3.2×). There, N−O reaches 50–60% of O−R, and the N−R order drops to 0.85 (y, λ = 2).
       - N−R still decreases everywhere sampled, because O−R dominates it. O−R is the transverse face-flux versus midpoint gap, not an η effect.
     - **Why the accepted campaigns missed it.** The low-degree catalogue fields are Cartesian polynomials of degree 3 or less, by the axis-regularity design, so they lie inside ringwise's exact space. The rich fields' quartic-and-higher terms scale as uᵐ and are negligible at the switch radius (u ≈ 0.06–0.14). The global volume-weighted gates weight the switch layers at about 1–2%. The accepted passes are unaffected, but they do not certify under-resolved transverse structure near the switch. P07 is the most exposed operator, and it is the one inverted for φ.
     - **Candidate remedy, from the [Q donor-support study](../../../../work/q_fci_donor_support_20260928/report.md).** Keep the coupled Cartesian-quartic basis and change only the donor support:
       - nearest-40 complete-owner centroids per η plane removes the wave rebounds in all six sampled roles on both intervals;
       - nearest-28 gives the lowest pooled error but keeps one coarse-interval rebound;
       - neither dominates ringwise uniformly. At the first singleton (N64, y, λ = 2), ringwise gives 1.2e-5, nearest-40 gives 1.1e-4 and nearest-28 gives 3.7e-5.
       - This is not adopted for P. It would change the qualified reconstruction and require re-qualifying P05–P07 against their oracles.
     - **Status, 2 October 2026:**
       - Planned check 2 was done by the C1/C2/C3 remote campaign (C3 locked).
       - Planned check 1 was done at full grid by `scripts/p08_step6_global/` ([local analysis](../../../../work/p08-step6-global-b1746484-20261002T191943Z-11690/local_analysis.md)).
     - **Full-grid transverse-wave result:**
       - **Fields:** the transverse field set has Q waves for n, Te and Ti (1 + 0.5 × wave), and degree-4 harmonics × B(u²) at the switch for ω and φ.
       - **Gates and headline criterion:** both pass.
       - **Global orders:** switch-localized content converges at about third order globally (total N−R at N64 is 4e-3 to 7e-3, orders 2.9–3.4), against about fifth order for the catalogue. φ_h is at orders 4.9 / 3.4.
       - **Bands:** the weakness is confined to the documented C3 exceptions.
         - u 0.12–0.21: ω total orders 1.3 / 1.6; ψ N−O orders 2.5 / 1.0.
         - u < 0.06: ω total orders 1.6 / 1.3.
       - Bands u ≥ 0.21 converge at 2.7 or better. No new defect appeared.
       - **Accepted (user decision, 2 October 2026)** with the core-band limitation. Mechanism: see the step-7 record and the [report](../../../../work/p08_core_convergence_20261002/report.md).
     - **Planned checks** (historical):
       1. Add a transverse-wave MMS control with order-one degree-≥4 Cartesian content at the switch radius. Score it per layer and at fixed coordinates across the switch, reporting the transverse and η parts separately, alongside the global gates.
       2. Evaluate the Q support candidates in P's own assemblies at the same owners. Priority is P07 N−O/N−R at switch_plus1, and the first-singleton regression.
       3. Only then decide on a P support or switch-policy change, then re-qualify and re-freeze the row artifact.
   - **Matched single-device and η-sharded execution: implemented, local checks pass (2 October 2026; [design](../../../../work/p08_step6_sharding_20261002/design.md)).** Full-grid matched check pending (remote).
     - **Layout:**
       - Every owner lies in a single η plane, and every plane has the same owner count: 793 / 1,792 / 3,161 at N32/N48/N64.
       - Owners are relabelled plane-major. Each device owns a contiguous block of planes, and halos come from two ring `ppermute` calls.
     - **RHS** (`drbx.native.fci_perpendicular_sharding`):
       - Each shard gets a local `PerpendicularPlan` over its planes plus a halo of **3** planes. The halo is 3, not 2: a face on a block boundary is computed on both shards, and its side rows reach 3 planes past the block. The lowering checks the donor reach and raises if it is exceeded.
       - The unchanged `perpendicular_rhs` runs inside `shard_map`.
       - **Local result:** sharded equals single-device **bitwise** for Sz = 1, 2, 4. This holds on a synthetic plan and on a real N32 closure with targets on every η plane (`compact_c3`, Dirichlet and Neumann variants).
       - **Limits:**
         - `raw_pairs`, `jump_mask` and `face_multiplier` overrides are not yet supported.
         - Tensor-source θ tables are still built for all η planes.
         - P06 diagnostic counters are per shard.
     - **φ solve** (`drbx.native.fci_perpendicular_phi_sharding`):
       - Halo of 2 planes. The P07 matrix couples −2..+2 planes, measured.
       - The inner product is a `psum`; solvax routes every reduction through it.
       - The per-plane preconditioner factors are slices of the global ones.
       - **Local result** on the N32/N48 exports: the same 10 (smooth) or 12 (white-noise) iterations at every Sz, with ‖Δφ‖_M/‖φ‖_M ≤ 3e-15.
     - **Full-grid matched check: passed (user decision, 2 October 2026)** (`scripts/p08_step6_global/`, commit b1746484, job 59221294; [local analysis](../../../../work/p08-step6-global-b1746484-20261002T191943Z-11690/local_analysis.md)).
       - Run on the `compact_c3` artifacts at N32/N48/N64, single device against Sz = 2, 4, 8.
       - The RHS is **bitwise equal** for `main_phi_dirichlet` and `transverse_dirichlet`, every field and term.
       - The ψ solves take identical iteration counts, with ‖Δψ‖_M/‖ψ‖_M ≤ 3e-16.
       - Timing in that run was not a scaling measurement (forced host devices). One single-device RHS evaluation took 41 s at N64, a P10 performance item.
7. **Acceptance record and roadmap update.**
   - **P08 acceptance record: P08 passed (user decision, 2 October 2026).** The core-band limitation stays under investigation and is re-evaluated with its report.
     - **Final operator:**
       - P05/P05N bracket, P06/P06N curvature and P07/P07N diffusion/polarization on the row artifact;
       - autodiff K, q2 faces for P05/P06 (P07 at q3), inner support C3 (`fixed_radius`);
       - B evaluator `compact_c3` (pinned explicitly; the code default stays `spline` until P10).
     - **φ:** Dirichlet only (Neumann-type φ is in P11). Solved by FGMRES with block-Jacobi over η planes, float32 plane LDU factors, warm starts, production rtol 1e-8.
     - **Gate, part by part:**
       - **Certified operator contributions and the combined residual:** the `compact_c3` re-freeze (`scripts/p08_step5_compact_c3/`, commit 6c4de657, job 59207210, identity 2eeecb16…).
         - Every solver and finiteness gate passes. The global total order is ≥ 4.3 in every variant, field and arm (criterion 1.8), and the lowest regional order is 4.08.
         - The results reproduce the spline 5.3 run within 0.4%. The 5.3 run was passed on 2 October.
       - **Qualified references:** re-frozen on the final operator in that campaign (autodiff K and divergence). The step-4 N64 wall anomaly is resolved; the N64 geometry/reference recheck is closed.
       - **Matched sharding:** bitwise RHS and identical φ solves at N32/N48/N64 for Sz = 2, 4, 8 (`scripts/p08_step6_global/`, commit b1746484, job 59221294).
       - **Source:** satisfied by the qualified references.
       - **φ diagnostics:**
         - prescribed and solved arms in 5.3 and in the re-freeze;
         - φ_h error 1.4e-5 at order 4.8 (main);
         - ψ solve: 10–11 iterations, consistency error ≤ 4e-11.
       - **Error budgets:** term-resolved and regional (P06N regions and u-bands) in the campaign summaries.
     - **Step 6:**
       - autodiff K, q2 and C3 adopted;
       - `compact_c3` adopted after a bounded comparison (operator errors change < 2.5%) and the re-freeze;
       - transverse-wave check accepted, with the core-band limitation;
       - sharding passed.
     - **Documented limitations:**
       1. Core bands u < 0.06 and 0.12–0.21 (inside the coupled-quartic region): fine transverse structure converges at about first order on N48→N64. The error is about 1e-2 relative at N64 for stressed fields, and it does not spread outward through the φ solve. **Investigated, 2 October 2026** ([report](../../../../work/p08_core_convergence_20261002/report.md)); re-evaluation with the user pending.
          - **u < 0.06 is mostly a measurement effect.** At a fixed ring index, N32→64 converges at order about 4. N48 has a different near-axis agglomeration (1, 3, 6, 6 owners against 1, 4, 4, 8), so 32→48 and 48→64 orders straddle two stencil families. The 2–4-ring band and its reference also change with N.
          - **u 0.12–0.21 is a real stencil limit.**
            - The 4-ring × 7-owner coupled-quartic donor arc becomes long and thin as N grows: its error constant drifts upward with N, its conditioning grows with N, P07 face errors don't cancel, and the ring after each owner-count doubling doesn't converge.
            - Basis, metric and η handling are ruled out.
            - Wider radial support (6–8 rings) restores about third order (32→64), but costs about 8× in the 0.21–0.27 interface band unless the transition is also changed.
          - The campaign's ω total in these bands is dominated by the P05 bracket, which shows the same mechanisms.
       2. Near-wall P07 O−R (face flux against midpoint reference) is about 3e-2 at order about 1.7–2.0. It comes from under-resolved near-wall coefficient structure (coil ripple), not from the reconstruction.
       3. Generic fields with degree-≥4 content converge at about third order (the cubic in-plane and η reconstruction). Catalogue fields show about fifth order because they lie near the exactness space.
       4. The MAKEGRID field and the fitted coordinate map are not certified against a finer independent source (non-blocking).
     - **Carry-forwards to P10:**
       - time-dependent source/boundary pairing;
       - reconciling the P-path Neumann point rows with production's physical halos;
       - the production wall model (P11 route A first);
       - flipping the code default to `compact_c3` after pinning the frozen call sites to spline;
       - sharded `raw_pairs` / `jump_mask` / `face_multiplier` overrides, and per-shard tensor θ tables;
       - RHS performance: one single-device evaluation takes about 41 s at N64 on CPU;
       - Vi/Ve terms (deferred);
       - the core-convergence follow-up.
     - **Deferred candidate: the "Γ-switch" quartic/ringwise rule** ([study](../../../../work/p08_quartic_rule_20261002/report.md), 2 October 2026; user decision: deferred).
       - **What it changes:** the coupled 4×7 quartic is used for stencils with anchor ring a ≤ a*, where a* is the largest non-full-stencil anchor with m_min < 7 or Γ(a) = A4(m_min)·(a+½)³·N² > 7.6e4.
         - A4(m) is ringwise's measured cos 4θ aliasing error on an m-owner ring, roughly ∝ m⁻⁶.
         - The rule uses only the logical owner layout, so it is geometry-independent. The switch lands just after an owner-count doubling (u* ≈ 0.17–0.18 at N32–64, about 0.09 at N ≥ 192).
       - **Effect at N32–64:**
         - one ring per grid changes; everything else is bitwise unchanged;
         - the 0.21–0.27 interface band improves about 2×;
         - the slow 0.12–0.21 P07 order is not fixed: those rows are still 4×7.
       - **Effect at large N** (synthetic, to N256): coupled conditioning and the error constant stay bounded, whereas C3 grows ∝ N and ends up 50× worse at the interface.
       - **Adopt** when the next re-freeze happens anyway, or before going beyond N64. That means a new `inner_support` option threaded like C3, a C3-vs-new remote campaign, and a re-freeze of the changed rows.
       - **Not recommended:** deeper blocks (6×5). They fix the P07 order in 0.12–0.21, but make the R1/R2 gradients and a P05 proxy 1.4–4.5× worse; a real P05 check would be needed first.
     - **Core-reconstruction design study: scheduled after the P10.0 RHS performance batch C (user decision, 3 October 2026).**
       - **Root cause** ([doubling study](../../../../work/p08_doubling_rootcause_20261002/report.md), synthetic):
         - **Primarily, owner aspect against a fixed 4-ring stencil.** The 4×7 quartic patch is about 7·aspect/4 times longer than it is deep. Its arc straightens as coverage shrinks, so it starves of radial information; the face-flux error constant is fitted as ∝ aspect^1.9·N^1.7, and the effect is present even without jumps.
         - **Secondarily, the ring after each owner-count doubling.** Its two radial faces use donor patches from rings with different owner counts, so the face-flux errors do not cancel. This defect dominates beyond N128.
       - **Study:** a geometry-adaptive rule, set per stencil:
         - (i) quartic or ringwise, by Γ;
         - (ii) the quartic's patch depth and width from owner aspect, arc angle or conditioning (deeper only where the patch is thin and straight; uniformly deeper hurts gradients);
         - (iii) face-consistent patches for owners at a doubling.
       - **Scoring:** fixed u, 32→64 pairs, synthetic to N256, and a real P05/P06 check (about 2.5 GiB).

Pending decisions:
- **Neumann closure in the harness: closed for P08 (user decision, 2 October 2026).** P08 uses the qualified P-path point rows. Reconciling them with production's physical halos is a P10 integration item.
- **The combined catalogue**, frozen before evaluation. Resolved for step 5.3: `scripts/p08_step5_combined/configuration.json`.

Carry-forwards:
- the production wall model (deferred to full RHS wiring, user decision 30 September).
  - Under the P-path operators a wall model becomes a provider of per-field kinds and `BoundaryData` on the plan's tables.
  - `no-flow` and `simple-conducting-sheath` map directly. `simplified-gbs-mpe` needs a Robin density condition, the augmented-Neumann φ solve and a derived ω.
  - **φ (user decision, 2 October 2026):** Dirichlet with a sheath-informed wall value (P11 route A).
    - The first form is insulating: φ_w = ΛTe, with j∥ = 0 (as in GRILLIX).
    - The optional form carries current: φ_w = (Te/e)[Λ − w(α) ln(1 − j∥/(e n c_s))], where w(α) switches the sheath current on smoothly across the critical-angle band.
    - The φ solver already accepts any per-point Dirichlet data. The wall model must supply the ψ = φ + τTi trace value and its tangential (θ, η) gradient at the plan's Dirichlet points.
    - j∥ is lagged by one step, or Picard-iterated with warm-started solves, and clipped below ion saturation.
    - Before relying on the current-carrying form, check:
      - the stability of the lagged sheath coupling;
      - the mapping between the Dirichlet points and the parallel wall-intersection points.
  - The CLI-default `legacy-velocity-trace` is to be removed then. Its legacy "neumann" velocity condition stores the owner velocity as the Neumann value.
- the deduplicated periodic census;
- the recovered-trace wall contract (not a wall law);
- pre-asymptotic rich-field orders read alongside low-degree fields;
- the coupled/ringwise switch weakness for transverse degree-≥4 structure (see the step-6 later check). The transition layers are the first place to look for order loss.

**Gate:** the certified operator contributions and the combined perpendicular
residual meet the global operator-order criterion, with qualified references
and matched sharding/source/phi diagnostics ("source" is satisfied by the qualified re-frozen references, user decision
2 October 2026; time-dependent source/boundary pairing is P10). Keep term-resolved and regional
error budgets so a summed residual cannot hide a failing operator. No separate
regional second-order gate is imposed. P10 independently checks solutions.

### P09 — Energy-stable perpendicular operators (SBP)

**Dependencies:** P08. This is the gate before P10 (user decision, 4 October 2026). It was added on 3 October as P11, an investigation track.

**Status:** in progress. The SBP split-form bracket is the production bracket (user decision, 4 October 2026); only details of its construction remain. Option B rounds 1–3 are done (round 3: the seam error is closure-row truncation on a steep manufactured field, not the trace). An integration design and the τp_i model change run in parallel.

**Gate:** the perpendicular RHS has no numerical energy source beyond the physical compressibility bound.
- **Bracket (required):** the discrete energy identity holds to round-off, and the rightmost eigenvalue of the frozen-φ operator is at most max(½c).
- **P06, P07 and the φ operator:** included only if the step-3 audit finds a positive H-symmetric part beyond the continuum operator's; then converted until it is gone.
- **Accuracy:** every replaced operator re-passes its static gates (P05/P05N, P06/P06N or P07/P07N), and the P08 references of the changed terms are re-frozen, before P10 uses it.
- **Method:** SBP split form with SAT is the primary route. If it fails, record the mechanism and choose another route that meets the same gate.
- **Out of scope:** the Q path (step 6 is coordination only).

**Why: the evolved-RHS linear-stability diagnosis (N32, 3 October 2026).** Evidence: the [instability report](../../../../work/p09_instability_20261003/report.md) and the [symmetric-rows report](../../../../work/p09_symrows_20261003/report.md).
- **Already in the package:**
  - the corrected, dissipative P05 jump sign (`43197e1b`);
  - opt-in centred cell rows, `cell_stencil="symmetric"` (`eb13a9f3`; the default stays `"biased"`);
  - an opt-in outflow wall closure for the bracket, `wall_transport="characteristic"` (`40e0dce6`; the default stays `"dirichlet"`). It removes the wall vorticity modes.
- **Remaining:** the ring-3 family, about +1.33e4 at N32 (step-6 φ, ρ* = 0.05).
  - It is a grid-scale θ dipole on a convergence line of the E×B flow.
  - It has no continuum counterpart. The physical compressibility S = ½∇·V is at most 470 at ring 3 and 5.58e3 globally.
- **Cause:** the bracket is in advective form, and its rows are not antisymmetric near the axis (the coupled-quartic fits, anchor ring below u = 0.21) or at the wall (one-sided cubics through the wall value).
  - Centred stencils reverse the group velocity of grid-scale content, so that content grows on a convergence line.
  - Symmetric rows or a Fourier θ basis alone therefore do not remove the mode.
- **Invisible to a short evolved run:** with the roughly 170× smaller |φ|/ρ* that P10.1's dt needs anyway (user decision pending), the mode grows only about 13× over the 50-step run. P10.1 would likely pass with it present, which is why this gate comes before P10.

**Terms.** The new bracket is an *SBP split-form* bracket, and it needs both ingredients.
- **Split form** (a rewriting of the continuum operator): V·∇g = ½[V^i∂_i g + |J|⁻¹∂_i(|J|V^i g)] − ½cg, with c = ∇·V = 2S.
  - The bracketed transport part is skew-adjoint, so it changes ∫g² dV only through the wall flux.
- **SBP** (a property of the discrete derivative): D = W⁻¹Q with diagonal W > 0, where Q + Qᵀ is nonzero only in boundary terms. It makes integration by parts exact in the discrete norm.
- **SAT:** boundary and interface conditions imposed weakly, as penalties whose strengths come from the energy estimate.
- **Why both:**
  - With an SBP D and H = |J|W, the discrete split form gives gᵀH·(transport) = the wall term exactly.
  - The advective form alone leaves a commutator of V with the antisymmetric part of D. That term is not sign-definite at the grid scale.
  - A derivative that is not SBP leaves interior terms in either form.
- **Scope:** this makes each advected field's g² neutral for a given φ, which is what the frozen-φ eigenvalue test measures.
  - The E×B energy exchange ∫φ[φ, ω] is a separate property.
  - With the split form it is expected to close only to truncation error, because the discrete product rule fails.
  - Arakawa's 2-D Jacobian closes it exactly by also being skew in φ.

**Step 1 — Option B: the SBP split-form P05 bracket (prototype, in progress).** See the [design](../../../../work/p09_optionB_20261003/design.md). The prototype is NumPy/SciPy, outside the package.
- **Discretization:**
  - one node per owner per η plane, at a raw-cell centre;
  - H = |J|·w^u·(2π/N_i)·Δη;
  - D_θ: Fourier on each ring;
  - D_η: centred (1, −8, 0, 8, −1)/12;
  - D_u: applied per physical Fourier amplitude with ring masks, so truncation and zero-padding between rings are exact transposes. It uses the same centred interior stencil, with generalized-SBP closures for cell-centred nodes (faces at the axis and the wall);
  - axis: zero radial flux, imposed by SAT;
  - wall: a characteristic SAT, with inflow penalty τ = |v_w| and nothing at outflow;
  - c: from the same operators, so constants are preserved exactly.
- **Stages:**
  - **1a:** derive the radial closure and check the operators.
  - **1b:** an analytic disk × periodic-η testbed: n = 16–128, straight and curved h, a saddle near the axis, and an advective-form control.
  - **1c:** HSX N32, from one extraction.
  - **1d:** upwind dissipation −H⁻¹Σ_d G_dᵀ diag(½w_f|U_n|) G_d. This is today's jump, scattered through Gᵀ instead of ±1 to the two owners.
- **Decisions (user, 3 October 2026):**
  - Target at least 2nd order overall, starting from the current 4th-order interior; no 6th-order stencils.
  - Zero-flux axis closure. Parity ghosts are dropped; a Zernike core remains a later option.
  - The velocity uses the discrete ∇φ, with the exact (autodiff) gradient as a control.
  - Keep upwinding, in the 1d form, to damp grid-scale modes.
- **Success criteria:**
  1. Energy identity to round-off: for random g the transport part contributes only the wall term, and the axis term is exactly zero.
  2. No ring-3 mode: the rightmost eigenvalue is at most max(½c).
  3. Accuracy: at N32, regional static accuracy comparable to today's C3 rows with `cell_stencil="symmetric"`. On the testbed, at least 2nd order overall with a 4th-order interior.
- **1d checks:** negative semidefinite to round-off. Also report the change in static error, the eigenvalues with dissipation on, and the damping rate of grid-scale modes.
- **Evidence:** the prototype report and `results.json` in the design folder. If a criterion fails, record the mechanism and stop before step 2.

**Step 1 results — round 1 (4 October 2026).** See the [round-1 report](../../../../work/p09_optionB_20261003/report.md).
- **Energy identity:** met to round-off (≤1.8e-17 on the testbed, 5.5e-18 at HSX N32); the axis term is exactly zero.
- **Eigenvalue bound:** met only with the **D7+ wall correction** ½H⁻¹ω t_R[t_Rᵀ(F^u g) − v_w t_Rᵀ g], which makes the wall term exactly −½ω v_w (t_Rᵀ g)². D7+ is adopted.
  - At HSX N32 the rightmost eigenvalue is +3.25e3 at rings 7–9, against +1.33e4 for today's operator. It equals its mode-weighted ½c and is within 0.2% of the physical S: physical compressibility.
  - With D7+ and the 1d dissipation nothing grows (RK4 rate about −53).
- **Accuracy:**
  - Testbed: overall order 3.2–3.8.
  - HSX N32 on today's pyramid: rings 0–6 are 10–100× worse than C3 (axis 0.34–1.09, near-axis 0.085–0.23); interior ω 0.161 against 0.046; wall 40–60× better.
- **Causes near the axis:**
  - Discrete ½c reaches 1.93e4 at ring 1, where the physical S is 260. Amplitudes zeroed on ring 0 but present on rings 1–2 sit inside the axis extrapolation's support.
  - On 8-node rings, φ's sin 4θ is the Nyquist mode.
  - At the 16→32 transition, the 16-node rings truncate the m = 8 content of F·ω.
- **Radial closure:** b = 4, s = 3, unique. The interior is exact to degree 4, the closures to degree 2.
- **Time step:** RK4 Δt 3.25e-6, against 3.7–4.1e-6 today.

**Step 1 results — round 2: Zernike core and SBP ring levels (4 October 2026).** See the [round-2 report](../../../../work/p09_optionB_core_20261004/report.md).
- **Construction:**
  - **Core:** a p = 6 polar-Gauss polynomial core inside u = K/n. Its operator is projection-type SBP: exact on degree p, and D_x and D_y commute.
  - **Ring levels:** rings are grouped into constant-N levels, each its own SBP radial block.
  - **Coupling:** levels and core are coupled by SAT. Each side's flux trace is set against a symmetric velocity-weighted interpolant of the other side's, plus an upwind penalty Γ = ½|v|.
  - **Velocity:** from a trace-matched φ (deviation D5c).
  - **Dissipation:** the 1d jumps in the rings, plus shell damping −κHP_h of the core modes above degree p.
- **Layout rules:**
  - L1: θ arc ≤ αΔu, with α = 4.27 fitted to today's rule.
  - L2: every level is at least 8 rings wide, so the closures at the two ends of a level do not overlap.
  - L3: K is the smallest ring index for which L2's pass leaves only levels at least 8 wide. This gives K = 5 at n = 32, 64, 128 and 256, and K = 11 at n = 48.
  - At HSX N32 this gives the core plus one 32-node level on rings 5–31, with no level-to-level face.
- **What it fixed:**
  - energy identity to round-off including the core and every interface (2.6e-17 at HSX);
  - numerical abscissa = max ½c in every case;
  - ½c = 0 on a straight field (2.5e-4);
  - HSX core error 0.04–0.11, against round 1's 0.34–1.09;
  - interior ω 0.047, against 0.161;
  - wall 5–30× better than C3+S1 (recomputed like-for-like).
- **What remains, against C3+S1:**
  - **Near the axis:** the core and rings 5–8 are 2–2.5× worse.
  - **The seam band u 0.12–0.21:** 5.5–10× worse. Inferred cause: the ring block's degree-2 inner closure takes a quadratic-extrapolated trace on the core circle (flux mismatch 3.9% at n = 32, where the core's own trace is accurate to about 1e-5), and the SAT lifts the mismatch into the first rows. A p = 8 core and K = 3 gave no gain.
  - **Level faces (n ≥ 64):** they converge at 2nd order, limited by the closures rather than the transfer.
- **Transfer pair at level faces:** Almquist-type and plain Fourier transpose give level-band errors within 8% of each other. Only the Fourier pair gives ½c = 0 exactly on a straight field.
- **Eigenvalues at HSX N32:**
  - **Default:** +408.8, 99.9% in the core: physical compressibility (mode-weighted ½c 411 against S 402.5).
  - **Centred interfaces:** a grid-scale +3281 appears at rings 6–9; the upwind penalty removes it.
  - **With dissipation:** no residual-gated eigenvalue was found. RK4 to T = 0.04 ends at −50.3, still drifting up by about 3 per window. No growth seen; not proven.
- **Time step:** see Cross-cutting.

**Round 3 — the core–ring seam (4 October 2026; [report](../../../../work/p09_optionB_trace_20261004/report.md)).** The trace is not the cause, so V1 and V2 were not built.
- **Trace substitution has little effect.** Replacing the ring block's trace on the core circle with the exact trace or the core's trace barely moves the band error: n 0.122 → 0.107–0.109; ω gets worse.
  - The 3.7% mismatch is in the flux F^u, not the field; the field trace error is 1e-4.
  - The core's own F^u trace is just as wrong (3.5%), because D5c matches the core gradient to the ring trace.
- **The decisive control.** Same resolution with no block boundary in the band: round 1's single full-ring radial block, whose closure sits only at rings 0–3. It gives n 0.013 and ω 0.030 at n = 32, 9× and 3× better than round 2, and about level with C3+S1.
- **Mechanism (inferred from the per-term breakdown and that control):** truncation in the degree-2 closure rows of the conservative half, at the seam (rings 5–8) and in the core's outer rows. The SAT cannot cancel it.
- **It is a property of the manufactured field.**
  - The step-6 ω carries a bump exp(−((u² − 0.21²)/0.06)²) peaking at u = 0.21, right on the K = 5 seam.
  - The testbed shows the same band error without HSX geometry.
  - With the smoother phi_wave field the band error is 7× lower at n = 32 (n 0.016), and 3.7e-4 at n = 64.
  - Any steep structure that crosses a block boundary meets the same 2nd-order closure behaviour.
- **Remaining levers (layout or design decisions, not trace fixes):**
  - put the seam outside steep regions: K ≥ 9, where a p = 6 core is too coarse;
  - or avoid a radial closure at the seam, for example a single radial block running down to a small inner core.
- **Decided (user, 4 October 2026): the seam behaviour is accepted.**
  - The closure rows at block boundaries stay degree 2, so steep structure crossing a block boundary converges locally at 2nd order.
  - Asymptotically, global order is unaffected. At gate resolutions it depends on the layout family across N32/48/64 (see the [integration plan](../../../../work/p09_integration_design_20261004/plan.md), finding 4):
    - the rule-generated K is 5 / 11 / 5;
    - N64 adds a level face at u ≈ 0.34;
    - on the testbed, the all-regions order for 32→64 is 0.8 for n and 0 for ω.
  - Regional orders remain diagnostic, as in the P10 gate.
  - Revisit the layout levers above only if re-qualifying the bracket against the P05 static gates fails on the seam.

**Step 2 — Decide on and integrate the bracket (after step 1; user decision).**
- **Decided (user, 4 October 2026):**
  - The SBP split-form bracket is the production bracket. Only construction details remain.
  - Level faces use the plain Fourier transpose pair, because the Almquist-type pair gives no accuracy gain here.
  - **D1 = nodal point values** (user, 4 October 2026).
    - Unknowns are point values at nodes: ring nodes at raw-cell centres, core nodes at Gauss points.
    - H holds the quadrature weights.
    - Inside the operator there is no raw-cell restriction or prolongation. The only transfers are the Fourier ones at level faces.
    - This amends "Retain the RLP owner unknowns" for the P path.
  - **Gates use the H-weighted L2 norm against exact nodal values** (user, 4 October 2026). This amends the locked MMS observation contract for the P path. C3 numbers are historical and compared only like-for-like.
  - **Wall inflow data (D7 a)** (user, 4 October 2026). Every advected field gets the inflow SAT at the wall, including Neumann-kind fields. The inflow trace is the manufactured trace in MMS and an extrapolated trace in production; diffusion carries the Neumann SAT.
  - **Layout: family A** (user, 4 October 2026; [M1 report](../../../../work/p09_m1_layout_20261004/report.md)).
    - **Rule:** a core of fixed radius R_c = 1/8 (K = n/8 rings) with degree p = min(K + 2, 12), plus one ring level at the full count N = n. There are no level faces up to n = 128.
    - **Gate:** at HSX it passes the projected P05 gate on the step-6 transverse set. Global order is ≥ 2.55 on 32→48 and ≥ 2.75 on 48→64 for every field, and the abscissa equals max ½c at N32, N48 and N64.
    - **Time step:** RK4 Δt at HSX N32 is 2.97e-6 (1.79e-6 at N48, 1.26e-6 at N64), 20–28% below today's. Accepted.
    - **Known limit, not gating:** the φ_wave Ti order at HSX is 1.30 / 1.70 for every family. It comes from the η product-rule defect, because the HSX metric changes by 21–29% between adjacent η planes.
    - **The rule needs n divisible by 8.**
    - **Core basis: Zernike polynomials** (user, 4 October 2026).
      - The production core (M3) builds its node-space matrices from the Zernike basis, not monomials. The polynomial space, nodes, norm and operator are unchanged.
      - This removes the monomial conditioning cap (p ≤ 12), so p follows K for every n.
      - M3 checks agreement with the monomial matrices to round-off at p ≤ 10, and runs a time-step check with a field that has finite flow across the axis, because core eigenvalues grow roughly like p².
    - **Core compressibility artifact: accepted** (user, 4 October 2026).
      - Discrete ½c in the core exceeds the physical S: 2127 against 642 at N32, 1249 against 646 at N48, 975 against 678 at N64.
      - The excess falls monotonically, at order about 2.2 / 2.5.
      - The global max ½c is set in the interior, so the eigenvalue bound and the time step are unaffected.
  - **Dissipation defaults** (user, 4 October 2026):
    - **On by default.** Ring face jumps at full upwind strength ½|U_n|, interface penalty Γ = ½|v|, wall inflow τ = |v_w|. None of these has a tunable factor.
    - **Core shell damping −κP_h** with κ = c_κ·max|V|·p/R_c and **c_κ = 1 fixed by rule**.
      - It is a modal filter for the core modes the projection-type derivative cannot see, so no characteristic speed sets it.
      - M3 checks that results are insensitive over c_κ = 0.1–1.
  - **Ring derivative application** (user, 4 October 2026; [benchmark](../../../../work/p09_theta_bench_20261004/report.md)):
    - **D_θ stays pseudospectral (Fourier) and is applied by FFT:** rfft, multiply by ik with the Nyquist mode zeroed, then irfft. It is the same operator as the dense matrix to round-off (≤ 4.5e-14), at any node offset.
    - **The radial SBP derivative is applied banded:** closure rows plus the 5-point interior.
    - The dense `Du`/`Dth` stay in the plan as test references and for probing.
    - **Measured share of bracket time saved by both changes:** −5% (slower) at n = 32, 12% at n = 64, 32% at n = 128. FFT alone is about 2× slower than the dense matrix at N = 32 and wins from N = 128 up. A periodic FD stencil in θ was rejected: its phase error is about 20% at m = 8 on N = 32.
  - **HSX coil ripple and the two geometry arms** (user, 4 October 2026; [wall-band diagnosis](../../../../work/p09_wall_band_20261004/report.md)):
    - **What happened.** M3c's P05N wall-band failures, and the φ_wave Ti "η limit", come from the HSX metric's η content: up to k ≈ 64 per turn, about 30× stronger at the wall than near the axis. Its k = 48 line is HSX's 48-coil ripple (confirmed by the user).
    - **Mechanism.** The conservative half of the split form differentiates F^η·g in η. F^η is built from h_u and h_θ, which carry the ripple. At n_eta ≤ 64 that content is unresolved and aliases (48 → 16 at N32/N64, → 0 at N48), so the orders were erratic.
    - **The old operator could not see it.** It used a pointwise metric in advective form, with the same pointwise metric in the reference (O ≡ R), so it never differentiated the metric in η.
    - **The operator is not at fault:**
      - its wall closure, inflow SAT, wall correction, advective half and dissipation all converge at about 2–3;
      - the testbed converges at 2.3–2.5;
      - η-low-passing the metric (k ≤ 12, in both N and R) makes every failing pair pass at 2.2–3.0.
    - **Rule.** The final MMS tests (static and evolved) run two geometry arms:
      - **Ripple arm (full HSX field):** always run; errors, orders and stability results are reported, but nothing on this arm is gated.
      - **η-filtered arm:** the order gate.
    - **How to filter.** Apply it to the field, not the operator. Filter B's cylindrical components in φ at fixed (R, Z) above a fixed physical cutoff; this preserves ∇·B = 0. Derive h, J, K and the FCI maps from the filtered field, so each arm has one geometry identity.
    - **Scope.** M5 and M6 plan both arms from the start.
  - **Open, to be decided with the SBP Laplacian design (step 4):**
    - narrow vs wide second derivatives for P07/φ, and whether φ uses the bracket's gradient;
    - how P06/P07 act on the nodal state.
- **Integration design (in progress):** moving from C3 owner averages to the nodal layout. It covers what that means for the P05–P07 operators, Q's FCI maps, the wall and sheath, MMS fields and diagnostics, and which parts are independent of the seam.
- **D1 for production:** nodal point values (the prototype) or owner averages (finite volume, as today).
  - Owner averages cost an O((mΔθ)²) mismatch at owner-count doublings, or a non-diagonal H.
  - With steps 4–5 the choice becomes system-wide.
- **Representation:** how the Fourier ring and amplitude operators enter the plan, the row artifact and sharding. They are not today's sparse rows.
- **Wall:** the SAT replaces both the one-sided wall rows and `wall_transport="characteristic"`.
- **Possible earlier change:** if 1d holds up, scatter today's jump through Gᵀ. This is a small production change, independent of the rest of Option B, but it changes the qualified jump candidate.
- **Then:**
  1. re-qualify the P05/P05N static gates on the new bracket;
  2. re-freeze the P08 references of the changed terms;
  3. P10.1 then runs on the new bracket.

**Integration milestones (adopted 4 October 2026; [integration plan](../../../../work/p09_integration_design_20261004/plan.md)).** These move the opt-in P stack from C3 owner averages to the nodal SBP layout. The production RHS (`fci_drb_EB_rhs.py`) keeps its own operators until P10 integrates the P stack.

| Milestone | Content | Gate |
|---|---|---|
| M0 | Decisions and contract amendments (Step 2 above) | User sign-off; done except items deferred to the Laplacian study |
| M1 | Layout-family prototype at HSX N32/48/64 (fixed core radius, few or no level faces) | Projected P05 order ≥ 1.8 on both intervals; abscissa = max ½c; time step reported |
| M2 | Package infrastructure independent of the seam: nodal layout and plan, nodal metric gather, node-plane sharding, ring-level operators, SAT boundary data, H norms, audit tools | SBP identities, Parseval, transfer transposes; single-device vs η-sharded agreement |
| M3 | JAX P05 SBP bracket with dissipation and D5c | Equivalence with the prototype to round-off; energy identity; ω = max ½c; Cayley evidence with dissipation; P05/P05N static gates at 32/48/64 in H; insensitivity over c_κ = 0.1–1; P08 bracket re-freeze |
| M4 | Step-3 audit of today's P06/P07/φ and the interchange pair | Report, after τp_i stage 1 |
| Laplacian study | Prototype of the SBP perpendicular Laplacian on nodes (Step 4 design); settles narrow vs wide and how P06/P07 act on nodes | User decision |
| M5 | P07 and φ on nodes (Step 4), CG | P07/P07N, φ elliptic controls, H-symmetry and definiteness, P08 re-freeze |
| M6 | P06 on nodes (Step 5), coordinated with the τp_i re-split | P06/P06N, H-antisymmetry up to ∇·K, W2 interchange audit, P08 re-freeze |
| M7 | Q interface: nodal index space and shared H (Q-owned; P-only with an explicit transfer until then) | Q's gates |
| M8 | P10 on the nodal stack | P10 gate |

**Step 3 — Audit the other operators (cheap; N32, local).**
- **Check:** the largest eigenvalue of the H-symmetric part of P06 curvature, P07 diffusion/polarization and the φ operator. A positive value marks a source of numerical energy.
- **Prior result:** the first diagnosis bounded P06's symmetric part (q1 + q3) at λ_max = +5.9e3, at the wall rings (i = 29–31). That is an energy bound, not a proven mode.
- **Also:** whether the φ matrix is symmetric in H, which would allow CG in place of FGMRES.
- **Cost:** one sparse eigenvalue solve per operator, with matrix-free products. The assembled attempt that included diffusion needed 9.3 GB and was dropped.
- **Parallel (Q-path) operators:** only if Q's owners agree.
- **Result (M4, 4 October 2026; [report](../../../../work/p09_m4_audit_20261004/report.md)):** HSX N32, production field kinds.
  - **P06 curvature is a numerical energy source, with growth confirmed.**
    - Rightmost eigenvalue: +1983 at τ = 1 and +1004 ± 5312i at τ = 0 (residuals ≤ 1e-9), both reproduced by RK4. That is 3.5× the continuum symmetric bound of 564.
    - Two families of growing modes: wall modes on the outer two rings (Neumann closure on inflow characteristics) and interior grid-scale checkerboards, one at the 16→32 agglomeration step.
    - The centred material term carries the growth; q3 does not damp it.
    - **The Step 5 gate condition is met.**
  - **P07 Dirichlet is dissipative** (ω = −177). **The φ matrix is 20% non-H-symmetric,** 87% of that from the wall rows on the outer two rings, so CG is not admissible until the Step 4 SAT wall closure.
  - **P07N:** ω = +1.2, from the wall faces; growth not resolved, negligible against operator rates of about 1e3.

**Step 4 — P07 diffusion/polarization and the φ solve, in the same H.**
- **Construction:** second derivatives that are symmetric negative semidefinite in H by construction.
  - **Cross terms:** the H-adjoint form −G†(D⊥)G.
  - **Diagonal terms:** narrow-stencil variable-coefficient operators (Mattsson 2012). The wide form leaves the Nyquist mode undamped.
- **Polarization:** the same H-adjoint form, so the discrete E×B energy ½∫n|∇⊥φ|² is a positive quadratic form in the same H. Whether G must be the bracket's gradient (the design's "consistency of the φ solve with D") is decided here.
- **Boundary data:** Dirichlet (a symmetric, Nitsche-like penalty) and Neumann (a flux penalty) as SAT. The existing `BoundaryData` targets feed the penalties instead of being written into rows.
- **Solver:** a symmetric definite φ matrix permits CG.
- **Then:** re-qualify the P07/P07N static gates and the φ elliptic controls.

**Step 5 — P06 curvature.**
- C = K·∇ with a fixed vector K. The interchange terms ∫φ C(p) and ∫p C(φ) cancel between equations only if C is antisymmetric in H, up to the ∇·K term.
- **Construction:** the same D and split form as the bracket. The q3 face upwinding is rewritten as −Gᵀ|A|G, so it can only damp.
- **Wall closure: option (a), as for the bracket** (user, 4 October 2026).
  - Inflow SAT on P06's incoming characteristics, with τ from the characteristic speed, replacing today's Neumann closure.
  - Reason: M4 found the curvature drift crossing the wall (k^u ≈ +21), with two of three characteristics entering the domain. Neumann closures there leave even the continuum problem without an energy bound.
- **Then:** re-qualify the P06/P06N static gates.

**Step 6 — Q path (parallel operators), only with Q's owners.**
- **Support operators**, as in GRILLIX:
  - the parallel divergence is minus the H-adjoint of the parallel gradient;
  - parallel diffusion becomes −∇∥†χ∇∥;
  - sheath and MPE conditions are imposed as SAT.
- **Shared norm:** the perpendicular and parallel operators must share one H, or the whole-system energy estimate does not close.
- This is recorded for coordination and is not scheduled on the P path.

**Watch items: energy exchange between equations (recorded 4 October 2026; not gates).** The gate above covers each operator acting on its own field. The items below concern how energy moves between equations in the coupled system. Check them when the relevant step runs, and record what is found. None blocks P09 or P10.
- **The target is a hierarchy, not exact exchange everywhere:**

  | Priority | Property | Status here |
  |---|---|---|
  | 1 | Each operator creates no numerical energy on its own | The P09 gate |
  | 2 | Linear coupling pairs are adjoint in H, so the coupled linear system has no spurious growth | Watch item W2 |
  | 3 | Nonlinear exchanges are exact, such as the E×B energy exchange | Watch item W1; left to dissipation |
  | 4 | Exact exchange with neutrals, sheath and sources | Not attainable; only consistency is required |
- **Background: what an exchange is.**
  - The total energy splits into reservoirs: E×B kinetic energy E_K = ½∫n|∇⊥φ|² dV, thermal energy (n, Te, Ti), parallel kinetic energy, and magnetic or sheath energy.
  - With ω = ∇·(n∇⊥φ), integrating by parts gives dE_K/dt = −∫φ ∂ω/∂t dV plus wall terms. So multiplying a vorticity-equation term by −φ and integrating shows what that term does to E_K.
  - A term either moves energy between reservoirs (an exchange) or removes it (dissipation). An exchange is exact on the grid only if the paired discrete terms are H-transposes of each other, so that what one equation loses the other gains.
- **W1 — the E×B energy exchange through the bracket (nonlinear; accounting, not stability).**
  - **Continuum:** ∫φ V·∇ω dV = −∫c φ ω dV plus boundary terms (V ⊥ ∇φ, c = ∇·V). This is zero in 2-D with straight field lines, and a small compressible exchange in 3-D.
  - **Split-form bracket (paper derivation, untested):** φᵀH·B_h(φ, ω) = −ωᵀH(c∘φ) − ½ωᵀH(Kφ − c∘φ) plus boundary terms.
    - K is the discrete conservative operator, Kφ = |J|⁻¹D_i(|J|V^i φ).
    - The advective half annihilates φ exactly, because the component cross product makes V ⊥ ∇φ at every node.
    - The leftover ½ωᵀH(Kφ − c∘φ) is a discrete product-rule error, since D(Vφ) ≠ φ·DV. It has no partner term in another equation, so it is a small net source or sink in the total energy, not a transfer between fields, and its sign is not fixed.
  - **Size:** for resolved fields it is truncation error and should shrink with refinement. Grid-scale content does not shrink, so the defect then depends on how much energy sits at the grid scale. Dissipation (stage 1d) keeps that small.
  - **Exact alternative:** a 3-D Arakawa-type bracket, which reproduces both each field's own ∫g² balance and the compressible E×B exchange. Its existence with curved h is open (Arakawa 2-D: the average of the advective form and both flux forms).
  - **Measured on round 1 (4 October 2026; [W1 report](../../../../work/p09_optionB_w1_20261004/report.md)):**
    - **Identity:** confirmed to ≤3.6e-16. The defect is about 4th order on resolved fields.
    - **Size at HSX N32:** 17% of the physical exchange for ω_η2, with 73% of it coming from the 16→32 transition rings 9–12.
    - **Checkerboard response:** a θ checkerboard gives about 2000× more on 16-node rings than on full rings.
    - **Open:** a 1st-order response to a radial checkerboard in φ.
  - **Test-catalogue lesson (η-parity):** the step-6 manufactured ω and n give W = P = 0 exactly. By η-parity, d and c∘φ carry only η-harmonics 0 and 2, while these g carry harmonic 1. So step-6 fields cannot exercise this exchange; use η-wavenumber-2 variants. The P10.1 field catalogue should be checked for the same blind spot.
- **W2 — linear coupling pairs (linear; a stability issue, not only accounting).**
  - When two fields are coupled linearly, a non-adjoint discrete pair can make the coupled linear system grow, even if each operator passes the gate on its own field. This is the ring-3 problem spread across two fields.
  - |U_n|-weighted upwinding does not necessarily cover it, and it vanishes on stagnation lines in any case.
  - **Interchange pair (P path):** C(p) in the vorticity equation against C(φ) in the pressure equations, linearized about the background profiles. In an energy-consistent model, ∫φ C(p) = −∫p C(φ). Step 3 should extend its audit to the H-symmetric part of this two-field operator, not only each operator on its own field.
  - **Parallel pairs (Q path; corrected 4 October 2026).** There is no direct ∇∥p ↔ ∇∥·V∥ pair in this model: the pressure equations contain D∥Ve and D∥j, not D∥Vi, and the Vi force pairs only up to the single-fluid term L_M = O(me/mi). The pairs that exist are:
    - the current pair: D∥(j) in the vorticity equation against G∥(ψ) in Ve. The code already builds it as a weighted negative adjoint (`fci_support_pair.py`). The same D∥(j) should also feed the (2T/3n) terms in Te and Ti, which currently use different boundary traces.
    - the electron pressure pairs: D∥(nVe) ↔ G(n), D∥(Ve) in Te ↔ G(Te), and D∥(Ve) in Ti ↔ the μτ G(Ti) column.

    Flag these to Q's owners.
  - **Answered (4 October 2026):** the continuum model is energy-consistent only at τ = 0 (with ρ* = 1, uniform B, and m_e/m_i neglected). At τ > 0 the polarization variable φ + τTi breaks the linear pairing and causes growth in the actual code. See "Model change" below. Until that change lands, the step-3 two-field audit gates at τ = 0, and at τ > 0 it compares against the continuum defect (2τ, 2τ²) rather than zero.
- **Why exact exchange everywhere is not the goal.**
  - **Neutrals:** ionization, recombination and charge exchange move energy through local source terms. The coupled plasma–neutral system does not conserve a simple quadratic energy for a discretization to mirror.
  - **Sheath:** sheath losses are physical sinks.
  - **What is relied on instead:** the gate (no operator creates energy), adjoint linear pairs (W2), and dissipation for grid-scale content (upwinding where the flow is nonzero).
- **Residual risk:** on stagnation and convergence lines |U_n| = 0, so upwinding does not damp grid-scale content there. If evolved runs show grid-scale noise accumulating there, the options are strain-aware jump speeds or explicit hyperdiffusion.
- **Level transitions and coarse-ring aliasing (recorded 4 October 2026; outcome after round 2 at the end).** Two separate mechanisms add error where the ring node count changes. The round-2 prototype targets only the first.
  - **Masked-amplitude jump (round 2 targets this).** A Fourier amplitude that is zeroed on the coarse ring but present on the fine ring is a jump of size a_m, so the radial stencil makes an error of about a_m/Δu. In round 1 this caused the fake compressibility at ring 1 and the 16→32 error for ω. The W1 check found most of the HSX N32 ω defect on the 16→32 transition rings. Round 2 makes each constant-N level its own SBP block, coupled by SAT with Fourier transfer, and compares Almquist's order-preserving interpolation with the plain transpose pair. The jump is then never differentiated; the expected interface error is about a_m.
  - **Product aliasing on coarse rings (round 2 does not target this).** Products such as V·ω on a 16-node ring fold harmonics m ≥ 8 back onto lower ones. Interfaces do not change how products are formed within a ring. The W1 check found that this amplifies grid-scale content on the 16-node rings by about 2000× compared with full rings.
  - **Not fixable by either:** harmonics with m ≥ N_coarse/2 are lost at the interface. This is a resolution limit; layout rule L1 (θ spacing ≤ α·Δu) keeps it small.
  - **Candidate fix, not adopted:** product dealiasing, forming products on a 3/2-padded θ grid and truncating back. It is cheap to implement but adds runtime. Decide after round 2 reports, based on whether the transition-band error and the W1 defect remain large once the interfaces are in place. If it is adopted, re-run the W1 check on the round-2 operator with and without it.
  - **Outcome after round 2 (4 October 2026): dealiasing is not adopted.**
    - The level-band error is closure-limited: the Almquist-type and Fourier pairs give the same error.
    - Content that the coarse side cannot represent enters through the SAT lift, scaled by a_m/Δu. For steep modes it is the same size as the closure error. Neither mechanism is product aliasing.
    - At HSX N32 the round-2 layout has no 16-node rings, so the aliasing W1 measured there is gone by construction.
    - Revisit if evolved runs show grid-scale noise on coarse levels.

**Model change: hot-ion polarization variable ψ = φ + τp_i (decided 4 October 2026).** This changes the model equations, not the discretization. It fixes a continuum defect that no SBP operator can remove.
- **The change.**
  - Boussinesq polarization: ω = ∇⊥²(φ + τTi) becomes ω = ∇⊥²(φ + τp_i), with p_i = nTi in normalized units (n₀ = 1).
  - It is still Boussinesq: the φ solve stays a constant-coefficient Laplacian, and only its right-hand side changes, from τ∇⊥²Ti to τ∇⊥²p_i.
  - **Only the polarization relation is a model change (corrected 4 October 2026, from the [scoping survey](../../../../work/p09_continuum_energy_20261004/model_change_scoping.md)).**
    - The code also uses ψ in three places as an internal *split* of φ = ψ − τTi:
      - the curvature remainder C(ψ), together with the elimination column in `curvature_principal_matrix`;
      - the composite G(ψ) in Ve;
      - the μτ column of the parallel flux matrix.
    - In the continuum each split cancels: the net terms are C(φ) and μ∇∥φ.
    - Changing one half of a split alone changes the physics. A re-split must update each pair together:
      - **Curvature:** column 0 gains c·τTi and column 2 becomes c·τn, with c = (2n, 4Te/3, 4Ti/3).
      - **Ve:** A[4,0] = μTe/n + μτTi, A[4,2] = μτn, and G(φ + τnTi).
    - Keeping the old split is continuum-correct. The upwind and characteristic dissipation, built from those local matrices, then misses the new local coupling.
- **Why: the continuum model.**
  - The Ti form drops τñ from the polarization while the curvature and parallel terms keep it. The linearized pairs then mismatch: n–Te by 2τ, n–Ti by 2τ², n–Ve by τ.
  - With these terms there is no quadratic energy. About a uniform, gradient-free background, the model has growing modes with growth ∝ |k| at τ > 0: 0.148|k| at τ = 1 and large k⊥², 0.009|k| at τ = 0.1. Without dissipation the system is ill-posed.
  - With ψ = φ + τ(ñ + T̃i), every curvature pair is symmetric and the growth is zero. The only defect left is the single-fluid ion term, O(me/mi).
- **Why: the code.**
  - Dense Jacobian of the actual RHS: 4×32×32 slab, uniform background, ρ* = 1, φ solve included, three parallel paths.
  - Largest growth rate:
    - τ = 0: about 0 on every path (≤ 2e-4).
    - τ = 0.1: about 0.005–0.018.
    - τ = 1: about 0.5–0.8.
  - It is linear in k at low k, matching the hand model (0.148, 0.290, 0.413 against 0.149, 0.299, 0.448).
  - The default perpendicular diffusion (1e-5) barely changes it.
  - Patching only the polarization (φ += −τñ) removes the growth on the coordinate and fci-legacy paths with centred curvature (about 1e-12).
  - It leaves 4.7e-3 with upwind production curvature, and 0.09 (upwind) or 0.19 (centred) on the production characteristic parallel path, at τ = 1.
  - Likely cause (inferred, untested): those paths' local matrices still carry the old elimination column, so their dissipation is built for the old split.
- **Why: the literature.**
  - The Ti form comes from the GBS stellarator papers (Coelho et al. 2022, [doi:10.1088/1741-4326/ac6ad2](https://doi.org/10.1088/1741-4326/ac6ad2); 2024), which cite the cold-ion Ricci et al. 2012 for the Boussinesq step. The sign audit above used the same normalized form (arXiv:2508.04881).
  - The energy-consistent forms all keep p_i:
    - Scott 2007 ([doi:10.1063/1.2783993](https://doi.org/10.1063/1.2783993)): W = φ + τ(ñ + T̃i);
    - Hermes-3 (Dudson et al. 2026, [doi:10.1088/1741-4326/ae3627](https://doi.org/10.1088/1741-4326/ae3627)): Boussinesq with p_i/n₀, and a proven energy theorem;
    - the non-Boussinesq GBS (Halpern et al. 2016; Giacomin et al. 2022) and GRILLIX.
- **Where it lands (staged).**
  1. **The model change itself (stage 1).**
     - The polarization relation: the φ solve's right-hand side and its p_i boundary data, the vorticity-from-polarization helpers, the MPE gauge multiplier, and the polarization balance terms. In `native/fci_drb_EB_rhs.py`, about lines 1711–1850 and 5537–5670.
     - Its callers: `fci_boundary_imex_*` and `simulate_hsx_blob.py`.
     - The hand-built copy in `linear/dispersion.py`.
     - The MMS references: `hsx_mms_continuum_reference.py` and `scripts/p09_evolved_mms/source.py`.
     - `docs/physics_models.md`.
     - The P07 operator is unchanged.
  2. **The lockstep re-split (stage 2, only if the stage-1 Jacobian still shows growth on the upwind or characteristic paths).**
     - **Curvature:** `curvature_principal_matrix`, the remainder ψ, and the P06 closed-form |M|, which must be re-derived or switched to `lapack4`.
     - **Ve:** the composite G(ψ), the μτ column in `fci_parallel_production_flux.py`, the short-leg Jacobian, and the `q_*` modules, including re-deriving the characteristic quartic.
     - The P path makes the Q-side changes as well (user decision, 4 October 2026), so both paths agree; Q's owners are informed.
     - New forms go behind a keyword whose default is the old form. The frozen P06 oracle imports the live `curvature_principal_matrix`.
  - **Result (4 October 2026; [acceptance report](../../../../work/p09_tau_pi_acceptance_20261004/report.md)): stage 2 is needed.**
    - **Stage 1 (394a9144):**
      - Criterion 2 passes: the continuum pairing is symmetric and growth is ≤ 1.4e-15 at every tested k⊥, τ and μ.
      - Criterion 1 passes only for centred curvature on the coordinate and fci-legacy paths.
      - At τ = 1 there is still growth: upwind curvature 4.7e-3 on every path (the τ = 0 baseline is 1.6e-4), and on the production characteristic path 0.086 (upwind) and 0.187 (centred). The old form gave 0.515 and 0.708.
    - **Stage 2, with the full lockstep as in-process overrides:** 7.5e-5 (upwind, every path) and 4.9e-13 (production centred) at τ = 1, and 1.3e-4 at τ = 0.1. All are below the τ = 0 baseline. Either half of a pair alone grows (0.03–0.59).
    - The stage-1 production growth comes from the characteristic |A| dissipation, which is built on the old split (A[4,0] lacks μτTi).
    - **Stage 2 is in implementation:** both pairs, the P06 closed-form |M| and the parallel characteristic quartic, behind the polarization selector. The old-form default stays on the shared matrix functions. The acceptance runs are repeated on the committed code.
- **Sequencing.**
  - Make the change before the P10.1 evolved-MMS harness is built, so the manufactured sources are written once, for the final model.
  - The operators P05–P07 are unchanged, so their static gates stand. The MMS manufactured fields and sources change.
- **Acceptance.**
  1. The uniform-background slab Jacobian shows no τ-driven growth beyond the τ = 0 baseline, on every parallel path including the production characteristic one, after stage 1 or, if needed, stage 2.
  2. The symbolic pairing check with the F₂ weights passes (`work/p09_continuum_energy_20261004/scripts/lin10.py`, `lin9.py`).
  3. `docs/physics_models.md` states the polarization form and cites its sources.
- **Companion defects (same audit; separate decisions, not part of this change).**
  - Semi-Boussinesq B²/n factor: the vorticity sources carry a local B²/n, but the polarization operator has no n/B². Hermes-3 moved a B² factor to restore energy conservation.
  - ρ*: it multiplies only the bracket, not the explicit compression terms, so particle conservation holds only at ρ* = 1. The MMS lane uses 1; the blob driver takes `--rho_star`.
  - Ohmic heating: νj² is absent from Te. García Herreros et al. 2026 (arXiv:2609.17425) found that restoring it raised transport and brought the target heat flux closer to experiment.
- **Evidence:** [continuum energy report](../../../../work/p09_continuum_energy_20261004/report.md), [literature table](../../../../work/p09_continuum_energy_20261004/literature.md), Jacobian logs in `work/p09_continuum_energy_20261004/jacobian/`.

**Cross-cutting.**
- **SBP fixes the spatial operators, not the time step.** Neutral operators have imaginary eigenvalues, and classical RK4 needs |λ|dt ≲ 2.8 there.
- **Time step on the round-2 layout (deferred, user decision 4 October 2026).**
  - **Size:** RK4 Δt at HSX N32 is 2.69e-6, against 3.25e-6 in round 1 and 3.7–4.1e-6 today.
  - **Not the core:** its |λ| is 0.18–0.36 of the rings'.
  - **The limiting mode** sits on rings 6–9. Rule L2 refines today's 6-ring, 16-node level (rings 5–10) to 32 nodes, which halves the θ arc there; that accounts for the 17% loss from round 1. The rest of the gap to today comes from the SBP operator itself.
  - **Later fix:** the layout rule can keep the agglomeration coarser, for example by extending a narrow level outward instead of refining it.
- **State-dependent wall data (P11):** each condition first needs a continuous energy estimate, and the SAT strengths mirror it (Nordström 2017). Without one, linearize about the state and check eigenvalues.
- **Reading:** an annotated [SBP reading list](../../../../../plasma%20papers/Theory/summation-by-parts/reading-list.md) is kept outside the repository. The core references:

| Reference | Used for |
|---|---|
| [Strand, 1994](https://doi.org/10.1006/jcph.1994.1005); [Del Rey Fernández–Boom–Zingg, 2014](https://doi.org/10.1016/j.jcp.2014.01.038) | Deriving diagonal-norm closures; generalized SBP for cell-centred nodes (1a) |
| [Svärd–Nordström, 2014](https://doi.org/10.1016/j.jcp.2014.02.031); [Svärd–Nordström, 2006](https://doi.org/10.1016/j.jcp.2006.02.014) | The energy method; why lower-order closures still give a higher global order |
| [Carpenter–Gottlieb–Abarbanel, 1994](https://doi.org/10.1006/jcph.1994.1057); [Nordström, 2017](https://doi.org/10.1007/s10915-016-0303-9) | SAT penalties; well-posed boundary conditions first, then their discrete mirror |
| [Fisher–Carpenter, 2013](https://doi.org/10.1016/j.jcp.2013.06.014); [Gassner, 2013](https://doi.org/10.1137/120890144) | Split forms on SBP operators |
| [Mattsson–Svärd–Nordström, 2004](https://doi.org/10.1023/B:JOMP.0000027955.75872.3f); [Mattsson, 2017](https://doi.org/10.1016/j.jcp.2017.01.042) | Dissipation that keeps the energy estimate (1d) |
| [Mattsson–Nordström, 2004](https://doi.org/10.1016/j.jcp.2004.03.001); [Mattsson, 2012](https://doi.org/10.1007/s10915-011-9525-z) | Second derivatives, including variable coefficients (step 4) |
| [Goodman–Hou–Tadmor, 1994](https://doi.org/10.1007/s002110050019) | Fourier collocation of the advective form is not guaranteed stable when the velocity changes sign |
| [Stegmeir et al., 2016](https://doi.org/10.1016/j.cpc.2015.09.016) | Support operators for the parallel direction (step 6) |

### P10 — Certify evolved MMS and complete final integration

**Dependencies:** P08, P09.

- Run fixed-final-time perpendicular MMS with independently controlled temporal
  and solver errors.
- Include diffusion-only evolution, bracket/curvature evolution, and the coupled
  perpendicular system.
- Recheck phi reconstruction and source/boundary pairing.
- Run on the operators that pass the [P09](#p09--energy-stable-perpendicular-operators-sbp) energy-stability gate. A short evolved run cannot detect slow grid-scale growth by itself.
- φ uses the Dirichlet wall condition with manufactured, time-dependent trace data. The solver accepts any per-point data, so the sheath-informed values of P11 need no solver change. Neumann-type φ conditions are out of scope.
- Add compact regression tests for the discovered failure mechanisms; keep
  expensive convergence campaigns as reproducible research artifacts.
- Verify JIT, JVP/gradient behavior on the smooth path, sharding agreement, and
  record compilation/runtime/memory costs.
- Update architecture documentation to describe the accepted implementation
  and its verified limits.
- After the gates pass, complete integration of the consolidated operators
  into the shared model/MMS workflow, reusing the verified preparation and
  runtime path rather than implementing each operator's infrastructure again.
  Record final selector/default decisions explicitly. Blob-driver
  synchronization remains deferred.

**Gate:** independent fixed-final-time MMS solutions meet the global
volume-weighted L2 order criterion in Section 1 for the isolated controls and
coupled perpendicular system; elliptic solution controls also pass. Temporal,
linear-solve, and geometry-reference budgets are qualified. P05–P08 global
operator gates and the P09 energy-stability gate also pass; local and regional error orders remain diagnostic.

**Final deliverable:** a certification bundle containing commands, immutable
configuration manifests, machine-readable errors/orders, plots, regional
diagnostics, and passing regression evidence, together with the consolidated
shared implementation and final model/MMS integration record.

### P11 — φ wall model (bookkeeping for future physics work, not a gate)

**Status:** this records the physics options for the φ wall condition so they are not lost.
- It is **not** part of the P operator/MMS roadmap.
- It has no acceptance gate.
- P05–P10, and roadmap completion, do not depend on it.
- The P path qualifies Dirichlet φ with manufactured wall data only.

**Inputs:**
- the presheath-theory literature review, `work/presheath_literature_review_20261002/report.md` (in progress, 2 October 2026);
- the [HSX grazing-angle map](../../../../work/p08_wall_grazing_angle_20261002/): wall area is 10% below 0.3°, 23% at 0.3–1°, 51% at 1–3° and 16% above 3°. b·n̂ changes sign on the wall, so tangency curves exist;
- P10, for anything that uses time-dependent wall data.

**Common background:**
- The sheath current–voltage law is j∥ = e n c_s[1 − exp(Λ − eφ/Te)], with Λ = ½ ln(mi/(2π me)).
- The inversion is for ψ = φ + τTi, so wall data are ψ traces and, for Dirichlet, their tangential gradients.
- An elliptic problem takes one scalar condition per wall point. Prescribing the normal derivative and the tangential variation separately over-determines it.

**Route A — sheath-informed Dirichlet everywhere (current production choice).**
- A1, insulating: φ_w = ΛTe with j∥ = 0 (as in GRILLIX).
- A2, current-carrying: φ_w = (Te/e)[Λ − w(α) ln(1 − j∥/(e n c_s))], with w(α) switching the sheath current on smoothly across the critical-angle band.
- A2 captures sheath current closure. Neither form has the Bohm (ion-acceleration) constraint of the magnetic presheath entrance.
- The solver is ready. Still needed:
  - a wall-model provider of ψ traces and their tangential gradients;
  - Te and Ti wall traces;
  - j∥ at the wall from the parallel solver;
  - the mapping between the Dirichlet points and the parallel wall-intersection points;
  - a stability check of the lagged j∥ coupling (or Picard iteration);
  - clipping below ion saturation;
  - the choice of w(α).

**Route B — mixed Dirichlet + MPE Neumann by grazing angle (as in GBS).**
- Use route A's value where α < α_c. Where α ≥ α_c, use Loizu's normal condition, ∂_nψ = ∓(mi c_s/e)√(1+Ti/Te) ∂_n v∥i + τ∂_nTi, with the Bohm sign set by sign(b_n). Blend smoothly (Robin) across the transition.
- The Dirichlet part fixes the potential level and removes the compatibility condition. The tangency curves and the Bohm sign flip lie inside the Dirichlet band.
- A presheath condition that keeps tangential terms, ∂_nψ + c·∇_Γψ = g, fits the same slot if one is derived.
- Still needed:
  - a continuum coercivity check of the oblique wall term on the MPE part;
  - per-face condition kinds and an oblique condition kind in the export, i.e. per-face constraint vectors in the reconstruction;
  - an N32 nonsingularity and near-null-mode check;
  - an MMS order campaign including an α_c sweep;
  - Robin blend weights, which are a modelling choice with no derivation yet;
  - MPE data ∂_n v∥i, plus the companion n, Te and ω conditions;
  - an estimate of the error from the 1-D presheath assumption.

**Route C — conormal Neumann everywhere plus a potential-level equation (research).**
- The condition is ν·∇ψ = g_ν (ν = P⊥n̂), the polarization charge that has crossed the wall. It is well-posed at tangency.
- Compatibility is charge conservation. It holds automatically only if one wall-current model feeds both the vorticity evolution and the inversion.
- A gauge makes the solve unique, but the physical level of φ (which the sheath law needs) requires a global equation, such as net wall current balance, or a Dirichlet anchor.
- Still needed:
  - a presheath model that gives the conormal flux with tangential gradients (none known yet);
  - deflation of the toroidal near-null modes (bulk eigenvalues about 7e-4 to 5e-2 against about 2e6);
  - a study of how data errors are amplified.

**Excluded:**
- normal Neumann on the whole wall: λ is spurious, the level is free, the data are discontinuous at tangency curves, and the theory does not hold below the critical angle;
- the derivative-along-b form (degenerate at tangency);
- a sharp Dirichlet/Neumann switch (singular at the junctions; the Zaremba problem).

**State-dependent wall data (recorded 3 October 2026; needed by routes A and B, and by any sheath condition on n, Te or Ti).**
- **What exists:** `BoundaryData` (Dirichlet values and tangential gradients at `dirichlet_points`, g_N at `neumann_points`) is an input to every RHS call and to the φ solve. The plan stores only where the data sit and how rows weight them. A caller may compute the data from the current state inside the traced RHS, and JAX then differentiates through it. Every P campaign so far used prescribed MMS data, so this is untested.
- **Missing 1, interior traces at the boundary point tables:** a row family giving a field's value and wall-tangential gradient at `dirichlet_points` / `neumann_points`, built from the same reconstruction, including that field's own wall condition. Face rows give values only at face quadrature nodes, and the Neumann rows take g_N as an input. This is lowering work, not new numerics.
- **Missing 2, self-referential data:**
  - **Cross-field data are explicit.** For example, φ_w = ΛTe_w, with Te under its own Neumann condition: reconstruct the Te trace first, then form φ's data. Route A needs only missing 1.
  - **Data depending on the same field are implicit**, for example sheath heat transmission (∂ₙTe depends on Te_w), which is Robin-type. Options:
    - **Linear Robin:** a new row kind folded in exactly at lowering.
    - **Nonlinear:** a per-point Newton solve on the trace, which is cheap because the trace is affine in the data.
    - **Lagging** from the previous stage: costs time accuracy and needs a stability check.
- **Not a characteristic solve.** All wall data enter through the reconstruction. Weak characteristic imposition of a value-type thermodynamic wall condition is a separate design option: the q3 wall solve with genuinely different interior and target states, done in closed form from the P06 eigenstructure. It would apply only if a time-integration stability check shows wall-driven growth.

**Decision record:** none yet. Revisit after the literature review, and record any evidence here.

### Tracking and future-task rules

- P02/P03 may run independently after P01. P05/P06/P07 may become separate
  future tasks after P04.
- Each future task receives its ID, dependencies, bounded objective, acceptance
  gate, and required evidence.
- Use statuses: `pending`, `ready`, `in progress`, `blocked`, `passed`.
- Mark a task `passed` only when its numerical gate is satisfied and evidence
  is linked. Code completion alone is insufficient.
- Record implementation/configuration identity, geometry/source hashes,
  reference qualification, per-term and regional errors, fallback activity,
  unresolved defects, and the next bounded task. Changing geometry,
  implementation, field catalogue, or acceptance rules invalidates certification
  reuse; retain the historical evidence unchanged.
- Preserve failed candidates and their measured failure mechanisms in research
  artifacts, keeping current-behavior documentation free of experiment history.
- The roadmap is complete only when P10 passes, which requires P09 (P11 is bookkeeping and not required); no conservation identity,
  interpolation test, frozen residual slope, or elliptic solution result alone
  substitutes for the agreed global operator-plus-solution contract.
- **Renumbering, 4 October 2026 (user decision):** the energy-stability milestone, added on 3 October as P11, is now P09 and gates evolved certification. Evolved MMS and final integration moved from P09 to P10 (its stages P09.0–P09.2 are now P10.0–P10.2), and the φ wall model moved from P10 to P11. Names created before then keep the old numbers: the `work/p09_*` folders, `scripts/p09_evolved_mms/`, `tests/test_p09_evolved_source.py` and commit messages.
- **P06 absolute-matrix action, 4 October 2026:** the intermediate `absolute_method="block_lapack"` (3x3-block `eig`, introduced at `cbbcc1ff` between the 4x4 `lapack4` and the `closed_form` default) was removed. Removed 4 October 2026; default numerics unchanged. `lapack4` stays as the bitwise reproduction pin.

### Progress ledger

P00–P01 evidence was produced by [Set up perpendicular convergence roadmap P00–P01](thread://01a0b506-1907-73d3-997d-a66637cd7e8c?hostId=local), using GPT-5.6 Sol.

**Current campaign status:** both clean remote centered and material-upwind
static campaigns are complete and independently verified. P03/P04 are passed
in the scopes stated above; both P05 static accuracy milestones are passed.
Shared implementation, structural and evolved certification remain open.
No new worker run is launched by this roadmap update. Earlier worker assignments
and the bounded/global diagnostics below are historical provenance, not current
instructions to repeat those experiments.

**Historical HSX qualification assignment:** P01–P03 had completed their
original audit scope and were reopened for actual-HSX qualification. P00
remained baseline preservation evidence, and P04 was then pending on the
supplemental P02/P03 evidence; prior idealized success did not satisfy that
dependency. That P03 task included the required P01/P02 supplemental work and
stopped before P04 implementation. Preserve those results and task records;
the current P04/P05 status is recorded in the table below.

**Current corrected-reference evidence:** the [continuous-reference sidecar](../../../work/perpendicular_second_order_hsx_p01_p03/continuous_reference_sidecar.json)
qualifies the producer evaluator against 4096 artifact samples and records
bounded differentiation checks. The [corrected N32 orientation experiment](../../../work/perpendicular_second_order_hsx_p01_p03/face_candidate_orientation_continuous_N32.json)
uses that reference for manufactured inputs and face data. Prior quantitative
near-axis omega conclusions derived from serialized polar metric interpolation
are superseded, including the claim that the planar candidate barely improves
omega. The [orientation report](../../../work/perpendicular_second_order_hsx_p01_p03/candidate_orientation_report.md)
now leads with the corrected result and preserves its former rejection only as
historical evidence; that rejection is not the current design decision.
Historical common-face oracle and global error
results using the old reference cannot establish corrected continuum accuracy.
Algebraic identities remain separate supporting evidence.

The corrected planar-only candidate, retaining production eta reconstruction,
reduces physical-volume-weighted L2 error over the fixed eight-owner sample by
60.1% for omega, 16.7% for the regular scalar, and 52.9% for the eta-varying
scalar. Omega relative errors at rings 0/1/2/3 change from
0.05459/0.04652/0.05384/0.27486 to 0.02649/0.04702/0.02066/0.02810;
non-axis transitions regress from 0.00394 to 0.03986. These compare completed
residuals with a common-face midpoint reference, not global continuum operator
errors. Integrated donor averages give mixed results. The axis-placeholder
cleanup leaves owner residuals unchanged: representative u=0 state/metric
values are unavailable and excluded from face-state comparisons, collapsed
integrated flux is exactly zero, and axis-owner errors remain included.
The candidate was promising enough for the bounded matched-functional audit.
That audit is now complete: the midpoint/raw-volume observation functional
retains identical support on 258/268 rows, reproduces quadratics to at most
`1.67e-15`, and reduces aggregate selected-sample L2 error relative to
production by 41.1%/52.5%/49.7% for omega/regular/eta-varying fields. The
[corrected global baseline](../../../work/perpendicular_second_order_hsx_p01_p03/continuous_global_baseline/summary.json)
passes both intervals for actual omega (`1.9305`, `2.2761`) but fails the fine
interval for the smooth regular (`1.5764`) and eta-varying (`1.3302`) controls.
The [structured-omega uncertainty audit](../../../work/perpendicular_second_order_hsx_p01_p03/structured_omega_differentiation.json)
shows source-difference orders `4.80` and `3.96`, so reference differentiation
does not explain that historical baseline failure. P03's audit gate is passed;
the baseline's failed operator gate motivated the repair and is not the current
candidate status. The [bounded N48/N64 fine-interval bridge](../../../../work/perpendicular_p03_fine_interval_design_20260919/report.md)
then refines the baseline into disjoint axis, boundary, true size-change,
agglomerated, and ordinary regions and tests frozen eight-owner samples before
candidate evaluation. True interfaces carry 94.6%/91.9% of scalar-upwind omega
squared error at N48/N64, whereas the centered omega budget is led by the axis
at 44.9%/51.1%; the scalar-upwind omega pass therefore does not certify the
centered operator. G improves every sampled scalar-upwind and centered field at
both resolutions. O improves both smooth upwind fields and all centered
combinations, but regresses scalar-upwind omega at both resolutions. Centered A
and B improve separately, with constant, orientation, decomposition, and
swapped-antisymmetry closures at roundoff. This closes the bounded P02/P03
mechanism audit and made P04 ready only for a separately authorized,
non-production P/G/O 32/48/64 global diagnostic qualification. The subsequently
authorized [P04 global diagnostic](../../../../work/perpendicular_p04_global_bracket_20260919/report.md)
is now complete. Production replay matches the qualified global L2 values
exactly. Of 36 per-field/operator/rule lanes, only production scalar-upwind
omega and G/O for the isolated centered-B eta-varying action pass both order
intervals. G/O fail every scalar-upwind lane and every complete centered-C lane;
their common-value scalar-upwind experiment explicitly does not preserve
production left/right stabilization. Structural, periodic, orientation,
constant, and bounded-equivalence checks close at roundoff, but the result does
not support a universal face rule or production promotion.

The subsequent [bounded P03 failure-localization report](../../../../work/perpendicular_p03_failure_localization_20260920/report.md)
reuses the frozen eight-owner N48/N64 samples and saved P04 rows with complete
incident support. On the centered-vorticity O/C sample, exact transported face
values do not improve the `0.761/0.819` numerical L2 error, while the exact
combined generator factor plus matching center correction reduces it to
`0.0540/0.0417`; both exact operands give `0.0312/0.0224`. Smooth controls also
favor the generator substitution, although cancellation prevents treating that
cross as a candidate operator. Center-state replacement is negligible, and
removing the production upwind jump produces only a modest change. Replays,
mean/jump algebra, reference-step sensitivity, and saved-row conditioning close
at their stated tolerances. That cross changed face geometry and the generator
gradient together and therefore did not by itself distinguish them. The
follow-on [four-way generator-factor report](../../../../work/perpendicular_generator_factorization_20260920/report.md)
separates HH, geometry-only EH, derivative-only HE, and combined EE while
preserving the face and matching center terms. EH does not remove the important
error, whereas HE reduces O/C vorticity from `0.761/0.819` to
`0.0430/0.0417` and reduces completed smooth-control errors for both O and G.
The interaction is small; N64 production and continuous one-forms agree to
roundoff. The supported next experiment is therefore matched owner-to-face
generator differentiation with the production one-form retained. This remains
bounded diagnostic evidence, not global convergence certification or a
production repair.

The initial [matched derivative experiment](../../../../work/perpendicular_owner_face_derivative_20260920/report.md)
used the actual midpoint/raw-volume owner functional and a full 3-D quadratic
in regular `(x,y,eta)`, but stopped support expansion as soon as rank and
polynomial reproduction passed. Its `597/1168` maximum regular-Cartesian `y`
amplification and O/vorticity regression from `0.761/0.819` to `3.754/5.152`
therefore reject that nearest-12 support rule, not degree-two differentiation.
The subsequent [donor-support correction](../../../../work/perpendicular_owner_face_derivative_support_20260920/report.md)
tests fixed 18 and 24 donors per eta plane and a geometry-only rule that starts
at 18 and expands to 24 when a plane has fewer than three distinct angular
owner columns. The coverage rule expands no N48 faces and two N64 faces,
reduces maximum condition to `12.3/13.6` and regular `x/y/eta` amplification to
`1.56/1.38/1.08` and `1.61/1.49/1.08`, and improves O/vorticity centered C to
`0.351/0.326`. However, G/vorticity and every smooth-control P/G/O centered-C
lane remain worse than production, with A/B failures exposed separately.
HH/HE replay exactly and no production selector changed. At that stage global
qualification of this earlier candidate was unwarranted; the remaining
stable-row truncation/cancellation defect motivated the follow-on work below.
This historical rejection does not apply to the now-qualified matched-q3
cubic candidate.

The follow-on [adaptive-support and truncation report](../../../../work/perpendicular_owner_face_derivative_adaptive_20260920/report.md)
keeps coverage18_24 as the baseline and localizes its remaining O/C error to
radial/angular generator differences with strong face/center cancellation.
Saved-owner rankings reproduce the independent concentration at true
interfaces and agglomerated interiors for the smooth controls and at the
axis/inner ring/interface for vorticity. A geometry-only sector-balanced
quadratic support leaves completed errors essentially unchanged, so support
placement is no longer the dominant defect. Actual owner observations of all
degree-three regular-chart monomials give persistent mesh-scaled quadratic-row
responses, while one adequately supported degree-three candidate (24 donors
per plane, five/six eta planes) reproduces degree three to `1.44e-12`, keeps
maximum condition at `50.7/87.9`, and improves all nine bounded centered-C
field/P-G-O lanes over production at N48/N64. O/vorticity becomes
`0.112/0.078`, regular O becomes `7.21e-5/4.80e-5`, and eta-varying O becomes
`4.20e-4/3.12e-4`. A/B improve separately and direct numerical-minus-HE
assembly closes below `7.2e-15`. This identifies degree-two truncation as the
dominant bounded mechanism, with residual compatible-composition sensitivity.
It supports a scoped global static qualification of the fixed cubic policy but
does not itself establish order or authorize production integration.

The authorized [global balanced-cubic qualification](../../../../work/perpendicular_cubic_global_qualification_20260920/report.md)
then applies that fixed degree-three policy at N32/N48/N64, with a single
geometry-only deficiency schedule used at every resolution. The optimized
construction reproduces saved bounded rows to roundoff, retains cubic
reproduction to `4.45e-10`, and closes replay and structural checks below
`7.82e-14`. O/centered-C vorticity passes both global order intervals
(`2.3048`, `2.3077`), but the smooth regular (`2.2140`, `1.3777`) and
eta-varying (`2.2527`, `1.4582`) controls fail the fine interval. O/A also
fails the fine interval for all three fields; O/B passes for vorticity and the
eta-varying control but fails for the regular control. At N64, 89.19% of the
regular-control and 72.06% of the eta-varying-control O/C squared error lies in
agglomerated interiors, with another 12.92% of the eta-varying error at true
size-change interfaces; neither the axis nor the physical boundary explains
the failure. This rejects the fixed global cubic candidate for certification
without invalidating its stable construction evidence. No production source,
boundary contract, evolved MMS, curvature, diffusion/polarization, or blob
driver changed.

The subsequent [bounded value/derivative cross](../../../../work/perpendicular_value_derivative_cross_20260920/report.md)
uses deterministic, geometry-stratified 28-owner N48/N64 samples with complete
incident neighborhoods and both base and expanded cubic supports. Holding the
globally qualified cubic derivative fixed, cubic planar face values regress
completed centered C relative to current O values at both resolutions:
vorticity ratios are `1.4487/1.3064`, smooth-regular ratios are
`1.0357/1.0356`, and eta-varying ratios are `1.0382/1.0601`; adding cubic eta
values is immaterial. Analytic values improve vorticity and eta-varying C, but
leave the regular C essentially unchanged at N48 and 7.10% worse at N64 even
though its A/B constituents improve, exposing cancellation/integration
sensitivity. Analytic derivatives reduce vorticity C from
`0.1534/0.04041` to `0.03722/0.01998`, so the cubic derivative family remains
useful diagnostic evidence rather than being rejected wholesale. For the
smooth controls, value and derivative substitutions both matter and their
completed-action behavior cannot be inferred from isolated A/B improvement.
Cross, decomposition, antisymmetry, derivative replay, value reproduction, and
full-operator replay close at or below `4.70e-13`. Under the midpoint target
used in that audit, the fixed cubic value candidates appeared rejected and the
workflow stopped before another global run. The integrated-reference result
below supersedes that target-dependent ranking. No production change was made.

The [integrated-reference requalification](../../../../work/perpendicular_bracket_reference_requalification_20260920/report.md)
supersedes that reference-sensitive interpretation while preserving its raw
midpoint evidence. The supervisor and P-campaign geometry paths resolve to
byte-identical payloads. Direct q3 physical-volume integration passes the 10%
reference budget for both smooth fields at every resolution; the global
O/centered-C orders become `2.1457/1.3143` for the regular control and
`2.1305/1.2130` for the eta-varying control, so the completed production
composition still fails the fine interval. On the frozen bounded cross,
cubic-planar values with the qualified cubic derivative now improve completed
C by 34.18%/48.79% for the regular control and 13.02%/2.42% for the
eta-varying control at N48/N64. The earlier blanket rejection of cubic values
is therefore withdrawn; it was not robust to integrated cell references.
Vorticity direct/IBP q3 passes the bounded budget at N32/N48/N64, but an N64
throughput sample extrapolates to 9.92 hours globally and the affordable
focused interpolant fails the N32 budget at 14.95%. No global integrated
vorticity result was built, so its historical midpoint pass is preserved but
not recertified. This motivated the matched face-and-volume experiment below.
No production source or frozen numerical input was changed.

That [bounded matched face-and-volume experiment](../../../../work/perpendicular_matched_face_volume_20260920/report.md)
is now complete on the frozen 28-owner N48/N64 samples. The corrected
supervisor prerequisite is preserved explicitly: O2/Dh vorticity C is
`0.075041/0.033573`, cubic-planar/Dh is `0.121014/0.026683`, O2 with the
analytical derivative is `0.111712/0.049318`, and analytical value/derivative
is `0.121491/0.050820`; analytical substitution alone therefore does not
repair midpoint assembly. Under matched q3 integration, the analytical volume
correction reduces face-only centered-C error by 93.6%/96.3% for vorticity and
99.7--99.8% for both smooth controls, supporting the continuum identity, sign,
and normalization. With values and derivatives from the same balanced cubic
fit, face-product integration reduces the two smooth-control errors by about
45--55%, but regresses vorticity. The numerical volume correction adds only a
0.9--6.5% smooth improvement and worsens vorticity by 2.9--5.0%. The remaining
vorticity error is concentrated in the boundary-footprint and axis owner strata;
the footprint label is not a radial-wall-face diagnosis. The smooth
error remains concentrated in agglomerated and true size-change strata. This
closes the requested bounded P04 integration audit with a mixed result: the
integration mechanism is real, but the complete numerical candidate is not
globally qualified.

The subsequent [supervisor face-factor audit](../../../../work/perpendicular_face_factor_audit_20260920/report.md)
supersedes the proposed volume-only, one-sided vorticity repair. With the
analytical volume correction held fixed, replacing only face values changes
vorticity C error from `0.144000/0.0305823` to `0.0907829/0.0191711` at
N48/N64; replacing only generator gradients gives `0.0545676/0.0139092`.
Both factors matter, while their bilinear interaction is only 0.84%/1.07%
of the total face-defect RMS. The same diagnosis survives the numerical
volume correction. There are zero physical radial-boundary faces in these
samples: the `physical_boundary_footprint` owner label includes short-leg
and double-hit topology masks and does not establish a wall-reconstruction
defect. No evidence justifies a field-specific repair, degree increase, or
automatic donor expansion. The user has authorized global 32/48/64
qualification of the frozen general cubic matched-integration candidate,
with the global vorticity reference explicitly costed and qualified; the
study has now completed; its clean remote qualification is recorded above.
Bounded error magnitudes do not establish its order, and improvement at every
sampled owner is not a prerequisite. The audit itself changed no production
path; the subsequently authorized study remains research qualification.

Update this table in place. Link detailed evidence rather than appending a
chronological run diary here. Each evidence bundle must identify its hypothesis,
revision, configuration, measured results, and unresolved failures.

| ID | Work package | Dependencies | Status | Task / evidence / remaining failures |
|---|---|---|---|---|
| P00 | Baseline and reproducibility ledger | None | passed | Revision `6c2b005d`; immutable dirty-source and 32/48/64 geometry hashes, exact selectors/boundaries/precision/sharding, and a single-process N=32 bracket/curvature/diffusion/polarization baseline are in the [P00–P01 evidence bundle](../../../work/perpendicular_second_order_p00_p01/report.md) and [machine-readable manifest](../../../work/perpendicular_second_order_p00_p01/p00_p01_manifest_and_results.json). This is preservation evidence, not a convergence claim. |
| P01 | Independent references, averages, sources, norms | P00 | qualified for centered and material static catalogues | The [remote campaign analysis](../../../../work/p_centered_cubic_c54b0552_kFhdmt_analysis/report.md) verifies a global complete-IBP vorticity reference and the existing direct-q3 smooth references, with candidate-relative bounded HSX qualification budgets below 2.82%. Direct/IBP independence and quadrature/step qualification remain bounded checks, not a claimed global exact-error bound. The failed focused interpolant and historical midpoint results remain separate. Frozen owner volume remains primary. The [material audit](../../../../work/p05_material_257bd55f_qQnlSX_analysis/report.md) requalifies its required-field budgets at 1.05–1.90% of the new errors. Other physical operators need their own references. |
| P02 | Reconstruction and derivative audit | P01 | complete; mechanism localized | [Task](thread://01a0b527-4aaf-7950-a06d-97a46ab517c4?hostId=local), GPT-5.6 Sol. N32 matched-functional evidence and the [N48/N64 bridge](../../../../work/perpendicular_p03_fine_interval_design_20260919/report.md) qualify the existing G/O degree-two mechanisms without changing degree or donor policy. G improves every bounded operator/field sample at both fine resolutions; O improves the failing smooth upwind controls but regresses scalar-upwind omega. This is bounded mechanism closure, not global convergence certification. |
| P03 | Geometry, interfaces, return maps, closures | P01 | passed — HSX mechanism audit | The [bounded localization](../../../../work/perpendicular_p03_failure_localization_20260920/report.md), [factorization](../../../../work/perpendicular_generator_factorization_20260920/report.md), and [face-factor audit](../../../../work/perpendicular_face_factor_audit_20260920/report.md) identify the derivative/value/integration mechanisms on actual HSX geometry. The baseline failures remain historical evidence, not a failed task gate. The [clean remote pass](../../../../work/p_centered_cubic_c54b0552_kFhdmt_analysis/report.md) validates the resulting centered-bracket repair. P04 readiness is satisfied; other operator and runtime/sharding checks follow their own work packages. |
| P04 | Consistent owner-to-face functionals | P02, P03 | passed — numerical design/research qualification | The [clean remote evidence](../../../../work/p_centered_cubic_c54b0552_kFhdmt_analysis/report.md) verifies polynomial reproduction and qualifies general cubic selection-v3 reconstruction with shared face values/gradients and matched q3 face/volume integration globally on all three fields. Freeze adaptive support, continuous geometry queries, and the boundary/owner conventions. Reusable payload/JAX extraction with saved-output replay is engineering follow-through during P05–P07 adoption, not a remaining numerical-design blocker. Other operators retain separate qualifications. |
| P05 | Brackets | P04 | qualified — direct midpoint global static MMS; production integration/evolution pending | User acceptance 26 September: [completed direct campaign](../../../../work/p05_direct_midpoint_global_565e1d1a_HsoyFbJ3/local_analysis/report.md), producer `565e1d1a`, passes both centered and centered-plus-saved-jump candidates for all seven nonconstant cases on both intervals. Actual-vorticity orders: centered `4.030/1.847`, with jump `4.047/1.930`; six other centered fine orders `2.810–2.989`. Preserve the [acceptance scope](../../../../work/p05_direct_midpoint_global_565e1d1a_HsoyFbJ3/local_analysis/acceptance_decision.md): localized omega maximum rebound and interpolation-sensitive geometry derivatives remain documented, without blocking this static pass. The [bounded Phase B replay](../../../../work/p_shared_face_extraction_20260926/report.md) recomputes the accepted jump from live state and matches saved `U-A`; coupled production integration, structural and evolved checks remain. Magnetic-field derivative/reference resolution remains a nonblocking accuracy follow-up. |
| P06 | Complete curvature | P04 | passed — midpoint static qualification 25 September; earlier integrated-reference closure retained | Current qualification: [job 58880303 at `2458dbf6`](../../../../work/p06_completed_58880303/local_analysis/report.md) (`scripts/p06_structured_global`) passes all 44 primary nonzero M/R/total entries, with orders `3.010–3.747/2.659–2.929`. It uses a prescribed Dirichlet trace lift for every field, and its physical-wall characteristic correction is identically zero because the wall solve receives the interior reconstruction as its trace. Its MMS states have zero thermodynamic wall gradients; only the two phi-Dirichlet controls carry nonzero normal derivatives. Historical: remote campaign `4bb8336e…` at `6f95ecea` completed; [local analysis](../../../../work/p06-curvature-cpu_4bb8336e_8ijYwA5H/local_analysis/report.md) replays all 22 required nonzero M/R/total components above 1.8 on both intervals (minimum 1.94027). User accepts milestone closure with the [bounded reference audit](../../../../work/p06_reference_qualification_20260922/report.md) retained as a nonblocking note: q5/q7 remains unsettled on raw cell 169732; q3 global-reference uncertainty is not fully quantified. Archived flags and results remain unchanged. Proceed to P07; no automatic reference rerun or production promotion. Seam defect (28 September): the accepted run counted the periodic θ/η slot-n alias faces twice in the U correction (invisible at its ~1e-12 corrections). Fixed in the runner and re-qualified 28 September by the P06N all-Dirichlet rich case (U orders 5.1–6.0, deduplicated census). |
| P07 | Perpendicular diffusion/polarization | P04, P03 | Combined structured static accuracy accepted — observed approximately third order; reference caveat nonblocking; integration/evolution pending | User decision 25 September: accept the [combined global campaign](../../../../work/p07_combined_global_analysis_20260925/report.md), orders phi `3.257/3.546`, Ti `3.251/3.591`, regular `3.099/3.338`, mixed `3.098/3.327`. The [bounded reference audit](../../../../work/p07_bounded_reference_audit_20260925/report.md) supports the accuracy assessment but does not recompute global orders. Preserve the archived failed reference flag and the [acceptance distinction](../../../../work/p07_combined_global_analysis_20260925/acceptance_decision.md). Preserve older D_trace results separately; its energy defect is not a measured defect of the new candidate. Elliptic/energy work remains deferred, and no production promotion or automatic run is authorized. |
| P05N | Physical-normal Neumann brackets | Shared extraction/replay and bounded Neumann reconstruction admission | passed — user-accepted static qualification 28 September; recovered-trace wall contract | [Acceptance record](../../../../work/p05n_p06n_43250ccf_20260928T053254Z_c415a4cd/local_analysis/acceptance_decision.md) from `frozen_v1` (`05be9063`: centered 2.31–3.41, with jump 3.06–3.54; jump active only at n−2 and on η faces) and `upwind_v1` (`43250ccf`: rich pairs 5.17–5.87 pre-asymptotic, jump active and converging at about 3.5, b1×e3 regression reproduced exactly). O ≡ R. The wall-face and outer-two-layer jumps are zero (contract and row structure); this is not a physical wall-law qualification. Transition region lowest (about 3.0). Evolution, production integration and the rung wall law remain open. |
| P06N | Physical-normal Neumann curvature | Shared extraction/replay and bounded Neumann reconstruction admission | passed — user-accepted static qualification 28 September; recovered-trace wall contract | [Acceptance record](../../../../work/p05n_p06n_43250ccf_20260928T053254Z_c415a4cd/local_analysis/acceptance_decision.md) for `43250ccf`, job 58995223: all 30 gated entries pass, including held-out (centered 4.38–6.15, U 4.47–6.12, pre-asymptotic). The q3 correction is active and converges at 4.2–5.4. φ enters only through the remainder (bitwise check). The all-Dirichlet rich case closes the P06 seam defect. The wall characteristic correction is zero by contract; not a physical wall-law qualification. Transition region lowest (2.2–2.6 on N48→N64). Evolution, production integration and the rung wall law remain open. |
| P07N | Physical-normal Neumann diffusion/polarization | P05–P07 shared extraction/replay | passed — user-accepted closure qualification 27 September; midpoint accuracy geometry-limited | [Acceptance record](../../../../work/p07n_field_derived_274e93e9_20260927T054625Z_72cfa1/local_analysis/acceptance_decision.md) for campaign `274e93e9`: N−O `2.24–3.85/2.25–3.73` on every field including held-out; wall-normal residual about 4th order; returned `global_order_pass=false` preserved; N−R ≈ O−R `1.55–1.74/1.72–1.83` limited by unresolved near-wall toroidal geometry (most plausibly coil ripple). The `5930b72c` failure is preserved. Inversion/gauge, energy, evolution and production integration remain open. |
| P08 | Combined frozen HSX perpendicular RHS | P05, P06, P07, shared extraction/replay, P05N/P06N/P07N | in progress — steps 1–2 (host consolidation, row artifact, JAX operators) accepted 29–30 September; step 3 next; see the P08 execution plan | Include separately qualified Dirichlet and Neumann variants. Use the deduplicated periodic face census. Decide or reconcile the Neumann closure (P-path point rows vs production physical halos). Watch the RLP transition region. Reconstructed φ: production FGMRES inverting the qualified P07 operator (new `operator_form`); new preconditioners likely. |
| P09 | Energy-stable perpendicular operators (SBP) | P08 | in progress — SBP split form chosen for production (4 October 2026); Option B rounds 1–3 done | [Design](../../../../work/p09_optionB_20261003/design.md); [round 1](../../../../work/p09_optionB_20261003/report.md), [round 2](../../../../work/p09_optionB_core_20261004/report.md); motivation in the [instability](../../../../work/p09_instability_20261003/report.md) and [symmetric-rows](../../../../work/p09_symrows_20261003/report.md) reports. Gate before P10 (user decision, 4 October 2026); replaced operators re-pass their static gates. Added on 3 October as P11. |
| P10 | Evolved MMS and promotion | P08, P09 | pending | Was P09 before 4 October 2026. |


### P07 portable global campaign preparation — 2026-09-23

The P07 global campaign contract (`p07_global_qualification_campaign.md`, removed; see commit `72acda02`) now
provides a frozen, node-local parallel CPU 32³/48³/64³ qualification runner.
Clean-checkout replay matches the prior bounded actions to `1.53e-13`,
serial/parallel N32 preflight arrays agree exactly, and complete-owner
q3/q5/q7 preflight passes its bounded reference check at all resolutions.
This established readiness for the now-completed cached comparison. The
[measured result](../../../../work/p07_cached_global_dirichlet_20260923/report.md)
passes the static global accuracy gate. Structural, elliptic and evolved
qualifications remain separate.

### P05 direct raw-midpoint candidate — 2026-09-26

**Completed and user-accepted:** the N32/N48/N64 global campaign at `565e1d1a`
passes both candidates' frozen static RMS gates. Local postprocessing repaired
only a nested summary-key lookup; all regenerated owner arrays exactly match
the remote partial products. See the acceptance and linked evidence at the
top of this roadmap. The [runner and frozen contract](../../../scripts/p05_direct_midpoint_global/README.md)
remain the reproducible campaign definition; its handoff is historical and
does not request a repeat run. Geometry-derivative uncertainty, current-candidate
runtime integration, conservation and evolved stability remain separate.
