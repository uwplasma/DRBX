# Roadmap: nodal FCI parallel operators on HSX (QF00–QF07)

Approved research roadmap, 12 October 2026. It is the authoritative progress ledger for the Q (parallel) path. It
specifies planned work and acceptance gates, not implemented or verified capabilities. The numerical method is defined
in the [nodal FCI design contract](q_fci_nodal_design_contract.md). The [perpendicular roadmap](perpendicular_second_order_roadmap.md)
(P00–P11) is unchanged, and both paths now share one node layout.

## 0. Why a new roadmap

The [archived Q00–Q09 roadmap](parallel_second_order_roadmap.md) qualified a different construction:
- owner/agglomerated GMLS reconstruction;
- traced spans of h/32 (caps at ±h/64, outer samples at ±h/32), which never reach the wall;
- a wall closure applied by radial ring (the last two rings).

The [Q09 wall audit](../../../../work/q09_wall_audit_20261012/codex/task_01_result.txt) showed the following:
- Parallel boundary treatment was assigned to every node in rings ≥ N−2, and no trace crossed the wall.
- In the complete RHS, the wall-layer orders lag (outermost-ring ω RMS 1.06, DDDDDD). In two D/N patterns the
  maximum-error order of ω was negative.

On 12 October 2026 the user decided:
- the intended FCI framework labels a node as a parallel boundary node only if its trace hits the wall;
- Q is redesigned as standard plane-to-plane FCI on P's nodal layout, with finite differences plus the support
  operator;
- the past Q gates are reopened.

**Retained:**
- the scientific contract (§1);
- the MMS field catalogue and evolved MMS harness conventions (§2, §3);
- the baseline evidence (§4);
- the execution workflow (§6).

**Retired:**
- owners and agglomeration;
- raw/owner midpoint observations and returns;
- support selection, rank repair and guards;
- short-span caps;
- ring-based wall closure;
- the characteristic exact-slot correction.

Historical results keep their identities and status. They are not relabelled, and they do not qualify the new method.

## 1. Scientific contract

These rules are carried over from the archived roadmap and adapted to the nodal method:

- **Real HSX geometry only.** Accuracy, convergence and boundary tests use the actual HSX metric, field, wall and
  topology from the first numerical audit. Idealized fixtures are algebra or bookkeeping controls only.
  - A bounded extract of real HSX data may serve as a cheap regression, but it cannot certify global order.
- **Gate.** For each certified operator and each nonconstant primary field:
  - the global volume-weighted RMS of N−R must have order **≥ 1.8 on both N32→N48 and N48→N64**, using actual
    spacing ratios;
  - the norm is `E = sqrt(Σ H e²/Σ H)` in P's norm H;
  - **N−O and O−R are always reported alongside N−R** (see the contract, §8). The user decides on the evidence.
- **Diagnostics, not gates:** regional and maximum-norm orders, per-trace-class bins (`none`/`forward`/`backward`/`both`)
  and wall-distance bins.
  - Use a disjoint partition for accounting: core block, interior rings and the outer four rings.
- **Resolutions:** begin with N = 32/48/64, using one continuous geometry reference across the sequence.
  - Nonmonotone results are inconclusive.
  - Do not infer slopes at a numerical floor.
  - Extending the resolution range is a separately scoped follow-up.
- **Continuous geometry.** Metric, |J|, B and normals are evaluated from the continuous evaluators at the points used:
  nodes, trace stages and wall crossings.
- **Error budget.** Reference, temporal and solver errors must each be below 10% of the corresponding finest spatial
  error. A solution pass does not waive an operator failure, and algebraic identities do not prove accuracy.
- **Structural requirements for diffusion:** H-symmetry, dissipation, constants and conservation (contract §10).
  - The old mandatory discrete minimum principle conflicts with cubic interpolation, whose weights can be negative.
    By decision D4 it is now a reported diagnostic; evolved n and T must stay positive.
- **Field arms.** The η-filtered field (m ≤ 3 per period) and the raw compact C3 field are kept as separate arms. The
  policy is the same as P (decision D5): gate the filtered arm, report the raw arm, then decide on what is observed.
- **Boundary scope.** The MMS uses prescribed Dirichlet and Neumann data at the trace-wall crossings. Physical sheath
  laws, inflow/outflow selection and the production wall model stay out of scope until QF05/QF06.

## 2. Retained static MMS catalogue

The catalogue keeps the user decision of 27 September 2026 on the Q fields. With `x = u cos θ`, `y = u sin θ` and
the full torus period 2π in η:

```text
s = 1 + 0.1*x*cos(eta) + 0.05*y*sin(eta) + 0.02*(x*x-y*y)*cos(2*eta)
common          = s
homogeneous_D   = (1-u*u)^2*s
simple_zero_N   = 1 + (1-u*u)^2*s
waves           = exp(i*(2*pi*x_or_y/lambda + eta))   lambda = 2, 4 (controls); 0.5, 0.25 (stress probes)
constant        = 1
```

**Changes for the nodal method:**
- **Reference.** The reference is the analytic operator evaluated pointwise at the nodes. This supersedes the raw-cell
  midpoint owner projection of 25 September 2026, which was owner-specific.
- **Dirichlet data** are exact values at the crossing points.
- **Neumann data** follow contract decision D1: the parallel derivative `b·∇f` at the crossing.
  The physical-normal datum `g_N = (g^{uj}/√g^{uu})∂_j f` remains P's convention.
- **Controls kept:** constant, affine-shift and data-linearity checks. The wavelengths are fixed in logical x/y units
  under refinement.

**Field-aligned family (user decision, 13 October 2026).**
- **Why it is needed.** The catalogue above is smooth in the logical coordinates but not aligned with B, so it varies
  along a field line as fast as the line wobbles through those coordinates (about 0.4 rad in θ at k_η = 4). That
  makes the three-point l²·f''' error large: 8.3% at N64 with n_eta = n
  ([refinement](../../../../work/qf01_maps_20261012/refine.json)). Turbulence is nearly aligned, so accuracy is
  assessed on a trace-aligned family.
- **Construction.** `f = g(X₀(x)) · w_j(η)`:
  - `X₀` is the footpoint on a reference plane η₀, traced backward with the operators' own field;
  - `g` is a footpoint profile with compact support inside the confined region;
  - `w_j` is a C⁷ window over half the torus, times `cos(k(η − η₀))` for k ∈ {0, 4, 12}.
- **Exact gradient reference:** `R_G = g(X₀) (B^η/|B|) w_j'`.
- **Roles.** The original catalogue remains the order and stress test. The aligned family measures the accuracy that
  the plane count (D8) is decided on.

Evidence for the earlier catalogue:
- [field-derived Neumann catalogue](../../../../work/q_fci_field_derived_neumann_20260927/catalogue.json);
- [manufactured-continuation audit](../../../../work/q07_continuation_audit_20261002/report.md). Its profiles are kept as
  held-out wall stress controls.

## 3. Retained evolved MMS harness conventions (from Q09)

The [Q09 harness contract](q09_evolved_mms_contract.md) supplies the evolved MMS for QF06, adapted from owners to nodes.

- **State:** `(n, Te, Ti, Vi, Ve, ω)` at nodes, with φ prescribed: `φ* = a(t)(0.07 sx + 0.04 sy)`.
- **Amplitude:** `a(t) = 1 + 0.1 sin t`, and `U* = BASE + a(t)(U_smooth − BASE)`.
  - The smooth deviations go as `r cos θ cos η` for n/Ti/Ve and as `r sin θ sin 2η` for Te/Vi.
  - ω is `WOFF + WC·(n,Te,Ti,Vi,Ve)`.
  - The positive lower bounds are n ≥ 0.89, Te ≥ 1.045 and Ti ≥ 0.845.
- **Constants:** `μ = 1836`, `τ = 1`, and diffusion coefficients `(0.01, …, 0.06)`.
- **Source:** `S(t) = dU*/dt − R_continuum(U*, φ*)`, built from an independent continuum reference. It is never built
  from discrete stencils.
- **Boundary patterns:** DDDDDD/φD, NNNNNN/φN, DNDNDN/φN and NDNDND/φD. The letters apply in the order n, Te, Ti, Vi,
  Ve, ω.
- **Modes:** diffusion-only and the complete six-field parallel RHS.
- **Timing:** T = 1e-4 with 100/225/400 RK4 steps at N32/48/64 and five snapshots, plus the three-dt N32 temporal pilot.
- **Checks:** finite and positive states at every stage, with no clipping. Regional, global and maximum errors are
  reported, along with integral drift.

## 4. Baseline evidence to beat (archived operator, final time, complete RHS, N48→N64)

| Quantity | η-filtered (m ≤ 3) | raw C3 |
|---|---|---|
| global RMS order, n/Te/Ti | 1.85–1.92 | 3.7–5.7 |
| global RMS order, Vi / Ve | 2.32–2.38 / 2.20–2.24 | 0.99–1.00 / 1.14–1.15 |
| global RMS order, ω | 1.83–1.95 | −0.41 to 0.81 |
| wall / outermost-ring ω RMS order (worst pattern) | 1.18 / 1.06 | — |
| ω max-error order, DDDDDD / DNDNDN | −0.57 / −0.28 | — |

Sources: [filtered three-grid review](../../../../work/q09_filtered_final_20261012/report.md),
[raw review](../../../../work/q09_evolved_return_20261005/report.md),
[wall audit](../../../../work/q09_wall_audit_20261012/codex/task_01_result.txt).
The Jarvis diagnostic **MMS tests → Q09** plots these results. The new method is compared on the same fields, patterns
and norms.

## 5. Stages

### QF00 — Archive, baseline and ledger

- **Tasks:**
  - Archive the Q00–Q09 roadmap.
  - Record the baseline evidence (§4) and pin P's layout modules and commit.
  - Fix the region partition and trace-class bins.
  - Close out pending work on the old operator: the Q full-domain mode campaign (`scripts/q_full_domain_modes`,
    commit 2575a42) was **dropped by the user on 12 October 2026** and will not be run remotely. Its scripts stay as
    archived records.
- **Gate:** this document and the design contract are accepted, and the open decisions are logged.

### QF01 — Plane-to-plane maps and wall-crossing census

- **Tasks:**
  - Build `M_full` (planes → planes ±Δη) and `M_half` (half-planes → planes ±Δη/2) on the family-A layout, at
    N = 32/48/64 for both field arms. Use RK4 in Cartesian (x, y) with step doubling and first-hit termination.
  - Record per node and direction: endpoint or crossing point, arc length, `l_w`, incidence and angle.
  - **Census:** count boundary nodes by direction, class and ring. Report the distributions of `l_w`, the shallowest
    crossings, nodes hitting in both directions, re-entry after a first hit, and the sign and range of B^η.
- **Gate:**
  - every node is classified;
  - no endpoint lies outside the domain;
  - endpoints and lengths converge under step doubling within the stated tolerance;
  - the round trip `M₊∘M₋` returns to the start within tolerance;
  - map artifacts are saved with identities.
- **Cost:** full-grid tracing (N64: 64 planes × 3,716 nodes per plane, two directions, two map sets, two arms). Any
  full-grid run goes remote. Local work is limited to bounded subsets.

### QF02 — Endpoint interpolation `I`

- **Tasks:**
  - Ring region, two arms (decision D7):
    - **primary:** Fourier in θ × cubic Lagrange in u, matching P's θ representation. It is applied by `oversampled`
      (σ = 8, m = 8) by default, with `exact` as an option (contract §4, user decision 13 October 2026);
    - **comparison:** tensor cubic Lagrange (4 × 4), with the quintic escalation rule.
  - Core: P's Zernike evaluation through the projector Π.
  - Wall and core-interface stencil shifts as specified in the contract, §4.
- **Gate:**
  - exact reproduction: the primary arm on ring trigonometric polynomials (up to the ring band limit) × cubics in u;
    the comparison arm on tensor degree ≤ 3; both on `P_p` in the core;
  - the interpolation error of the exact catalogue fields at the actual QF01 endpoints converges at order ≥ 3.8,
    globally and per region, for each nonconstant field;
  - the half-cell wall extrapolation class is reported separately.

### QF03 — Collocated parallel gradient `G` with trace-hit LVF boundaries

- **Tasks:** build `G` (contract §5) with Dirichlet and Neumann LVF rows. Run the static MMS on the §2 catalogue and
  the field-aligned family for both arms. The along-line formula is the non-uniform three-point form (user decision,
  13 October 2026); the five-point form is a measured comparison only.
- **Gate:**
  - the global N−R order is ≥ 1.8 on both intervals for each nonconstant field, under both D and N;
  - N−O and O−R are reported by region, trace class and `l_w` bin;
  - fallback rows (small legs, both-direction hits) are counted and their error shares reported;
  - constants and quadratic-along-line exactness are checked.

### QF04 — Support-operator divergence and parallel diffusion

- **Tasks:**
  - Build `G½`, the adjoint divergence `D`, and `L_∥ = D χ½ G½` (contract §6), including the boundary half-plane rows.
  - Compare `D` against the control divergence `B ∇∥(q/B)`.
- **Gate:**
  - the structural checks pass (H-symmetry, negative semidefiniteness, conservation, constants, data linearity);
  - the N−R order is ≥ 1.8 on both intervals for `D` and `L_∥`, under both D and N;
  - the minimum principle is reported (D4), not gated.
- **Note:** this stage reopens the archived Q03/Q04 diffusion gates.

### QF05 — Transport, current divergence and the coupled static six-field RHS

- **Tasks:**
  - Assemble the six-field parallel RHS from `G`, `D` and `L_∥`, using the Q07/Q08 term definitions (material
    transport, `∇∥p`, current divergence into ω, generalized force).
  - Using bounded evidence, compare centred against upwinded interior terms with characteristic (inflow-only) wall data (D3), and check that
    collocated velocities control checkerboard modes. Switch to staggered only if they cannot (D2).
- **Gate:** the global static N−R order is ≥ 1.8 on both intervals for each field and BC pattern, with N−O and O−R
  reported.
- **Note:** this stage reopens the archived Q06–Q08.

### QF06 — Evolved six-field MMS

- **Tasks:** run the §3 harness on the nodal operators: 24 integrations, both modes, all patterns, both arms.
- **Gate:**
  - the global solution-error order is ≥ 1.8 on both intervals;
  - the three-dt temporal check is within 10% of the spatial error;
  - positive n/T throughout;
  - regional, trace-class and maximum errors are reported against the §4 baseline.
- **Note:** this stage reopens the archived Q09.

### QF07 — Coupling with P and package integration

- **Tasks:**
  - Couple with P's perpendicular SBP operators on the shared layout, including the combined energy identity.
  - Then integrate into the package: consolidate first and integrate at the end.
- **Gate:** combined P+Q evolved MMS, and long-time stability evidence, to be defined when QF06 passes.

## 6. Execution workflow (retained)

- **Local:** bounded subsets only, and one heavy process at a time.
  - Pin JAX threads (`NPROC`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`).
  - Check memory headroom before each heavy step.
  - Check free disk space before writing gigabytes.
- **Remote:**
  - full-grid runs use a pinned commit, tested from a clean export;
  - a remote handoff carries computation only, with no scientific interpretation;
  - each campaign writes to one folder.
- **Identities:** prepared artifacts carry identities. Cross-platform checks use tolerances, never float-byte hashes.
- **Equivalence checks** between optimized and reference code run once per code change, not in every preflight.
- **Delegation:** research harnesses and analyses are delegated to workers. The parent reviews each result before it
  is reported, and the user decides each gate.

## 7. Progress ledger

| Stage | Status | Date | Evidence |
|---|---|---|---|
| QF00 | in progress: roadmap and contract drafted; D1–D8 decided; old full-domain modes campaign dropped | 2026-10-12 | this document; [design contract](q_fci_nodal_design_contract.md) |
| QF01–QF07 | not started | — | — |

## 8. Archived Q records

These documents describe the superseded owner/short-span method. They are kept unchanged for provenance:

- [Q00–Q09 roadmap](parallel_second_order_roadmap.md): the full history, including the traced-diffusion freeze of
  29 Sep, the Q06 traced gradient of 30 Sep, the long-leg boundary prototype and working-span decision of 1 Oct, the
  Q08 freeze of 4 Oct, and Q09.
- [FCI/direct architecture comparison](parallel_fci_direct_design_comparison.md),
  [traced diffusion freeze](q_traced_diffusion_frozen_contract.md), [traced extraction](q_traced_extraction.md),
  [Q06 traced contract](q06_traced_gradient_divergence_contract.md), [Q07 transport](q07_transport_contract.md),
  [Q08 extraction plan](q08_shared_extraction_plan.md), [Q09 harness](q09_evolved_mms_contract.md), and the
  `q0*_assignment.md` / `q_fci_*` assignment and design records.
- Evidence of particular relevance to the redesign:
  - [long-leg boundary prototype](../../../../work/q07_c3_longleg_boundary_20261001/report.md): first-hit cut legs
    at span h, with crossings from the fifth ring;
  - [projected-FCI global result](../../../../work/q_fci_projected_global_analysis_20260925/report.md): the face-return
    failure at span h;
  - [Q09 wall audit](../../../../work/q09_wall_audit_20261012/codex/task_01_result.txt).
