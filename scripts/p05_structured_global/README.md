# Structured P05 global qualification

This fresh static N32/N48/N64 campaign applies the common P07-derived structured
point reconstruction to the Poisson bracket. Previous P05 centered/material
passes remain historical evidence for their original implementations. This run
qualifies a changed implementation, including prescribed Dirichlet wall traces.
No elliptic solve, evolved simulation, or production promotion is included.

## Numerical contract

All owner observations are physical-volume weighted raw member-center values.
Canonical distinct fine faces are shared with opposite incidence signs; internal
aggregate faces are removed and collapsed-axis fluxes are exactly zero. The
common reconstruction is `scripts/perpendicular_structured/reconstruction.py`:
structured cubic radial/four-plane eta/seven-point trigonometric theta,
ringwise owner moments, bounded coupled Cartesian quartic near the axis, and
quartic radial wall/adjacent closures with the trace lift `f-g`.

Both values and gradients now use that service at face and volume quadrature
points. This is stronger point-functional support qualification than the P07
integrated diffusion row. Unsupported rows fail explicitly.

With `h=b_cov/B`, `U_a=-h cross grad(a)` and rho*=1, primitive A is
`[sum_faces integral U_a*f - c_f sum_faces integral U_a
 - integral (f-c_f) div(U_a)]/V_owner`, where
`div(U_a)=-curl(h).grad(a)`. B is the negative argument-swapped primitive;
C=(A+B)/2. The owner anchor c is the same constant throughout each complete
owner. Candidate face and volume rules are q3. Curvature of geometry enters
curl(h) through fourth-order centered differences, step 2e-4.

The new U form is A plus a shared dissipative flux
`-0.5 integral |U_a.normal| (f_right-f_left)`. Left/right states come from
structured stencils anchored at the adjacent cells; the common velocity and
common value are unchanged. Equivalently, states can be re-centered about the
common value without changing their jump. The physical-wall exterior state is
the prescribed trace, and compatible wall reconstructions have that same trace.
This replaces the *old correction's* biased unstructured WLS construction; it
must earn its own accuracy qualification. No energy qualification is claimed.

Cases include the original MMS phi/actual generalized-potential vorticity,
smooth regular and eta-dependent scalars, zero Dirichlet scalar with nonzero
normal derivative, spatially varying Dirichlet scalar, constant control, and a
nonconstant-wall generator paired with both new scalars. Compatible zero-slope
smooth fields are controls evaluated with their analytic traces, not a new
Neumann-boundary implementation. All A/B/C forms and U on the smooth cases are
primary; actual-vorticity U and the constant case are diagnostics. The order
criterion is >=1.8 on both intervals, separately from reference qualification.

## Independent reference and diagnostics

Primary smooth references integrate the analytic strong bracket with q5 volume
quadrature. Actual vorticity is obtained from the continuum generalized-potential
operator and referenced by complete integration by parts with q5 surface and
volume rules, avoiding differentiation of omega in the reference bracket.
The existing wall convention evaluates omega at r=1-3*FD_step; this approximation
is retained explicitly and included in the half-step reference controls.
Only its tangential derivatives are required for the prescribed trace lift.

Ten geometry-selected complete owners per resolution cover axis, first ring,
agglomerates, transition, ordinary, seams, wall and two inward layers. Return
q5/q7/q9 reference comparisons and q7 half-step results there. Smooth strong and
integration-by-parts values are both saved, including continuous owner volumes.
Do not interpret a q3/q5 difference alone as a qualified higher-order reference.
The bounded RMS budget sums absolute q7-q5, q9-q7 and q7 half-step changes;
its fraction of each global error must be <=10% for the separate machine
reference flag. Raw evidence is returned for local scientific review; a machine
order pass alone is not a reference-qualified pass. This is bounded evidence,
not a rigorous bound on every reference cell.

Candidate q3 exact-field flux/correction arrays are retained as oracle diagnostics.
The actual-omega swapped-B oracle and strong-reference slot are unavailable
(zero placeholders): analytic omega gradients are deliberately not evaluated.
Only its A integration-by-parts reference is scientifically meaningful. All
smooth-case oracle slots are available.
All global owner arrays, regional RMS/max, shared face fluxes, and support/constant
checks are saved. Complete face closure is independently reassembled for each
control owner and compared against global scatter. No crop substitutes for the
full-domain norm. Constant-field actions are separately reported; the global implementation flag
requires their maximum absolute action <=1e-8 in the frozen rho*=1 units.

## Commands and resources

Run from the repository root with immutable workspace-layout INPUT_ROOT (the
nine file identities and paths are in `input_manifest.json`) and a new OUTPUT:

```bash
python scripts/p05_structured_global/campaign.py verify-inputs --input-root "$INPUT_ROOT" --output "$OUTPUT"
python scripts/p05_structured_global/campaign.py preflight --input-root "$INPUT_ROOT" --output "$OUTPUT" --workers "$WORKERS" --memory-budget-gib "$MEMORY_GIB" --worker-memory-gib "$WORKER_GIB" --memory-reserve-gib "$RESERVE_GIB"
python scripts/p05_structured_global/campaign.py run --input-root "$INPUT_ROOT" --output "$OUTPUT" --workers "$WORKERS" --memory-budget-gib "$MEMORY_GIB" --worker-memory-gib "$WORKER_GIB" --memory-reserve-gib "$RESERVE_GIB"
python scripts/p05_structured_global/campaign.py validate --input-root "$INPUT_ROOT" --output "$OUTPUT"
```

The nine required files are the six canonical geometry/topology files, the
continuous-reference sidecar, metric cache, and MAKEGRID file. Historical
`prototype_runs/geometry/hsx_fci_64x64x64/*` artifacts named in the sidecar
are provenance only for this runner; the continuous-reference loader does not
read them. They are intentionally absent from the runtime input manifest.

Allocation setup chooses the resource variables. Execution is **node-local CPU
multiprocessing**, not MPI: one writer and one node per campaign. Preflight uses
the same process pool and memory cap. Inputs are verified once in full; source,
configuration, commit, localized sidecar, topology and selection identities guard
subsequent commands. Complete chunks have SHA256 receipts. Interrupted stages
resume the same output; corrupt chunks are recomputed, incompatible campaigns
are rejected. Preserve the pinned revision/configuration during resume.

Each worker loads one resolution context; BLAS threads=1; dense point rows are
streamed per entity rather than stored for every global face. Immutable owner
observations are loaded once per worker. Matrix and boundary-trace caches are
bounded; workers recycle after 64 chunks by default. Choose concurrency from
measured peak worker RSS with controller headroom. Record allocation, effective
workers, memory, timing, exit status and every attempt in the campaign folder.

`scripts/.../campaign.py smoke --resolution N --sample-index I` executes one
bounded preflight owner after verification; it is for local development or an
operational footprint check, not a substitute for the full preflight command.
Do not launch a full local run to prepare the remote handoff.

## Optimized execution and existing campaigns

Identical metric queries are reused within each batch, singleton interpolation
weights have bounded exact-coordinate caches, side states skip unused gradient
contractions, and topology arrays are loaded once per worker. No quadrature,
donor policy, numerical coefficient or acceptance threshold is reduced.
For q7/q9 control points whose fixed radial curl stencil would leave [0,1],
only that radial step is reduced to 0.2 times the distance to the boundary.
Previously those controls raised a geometry-domain error. Canonical N32/48/64
q3 candidate and q5 global-reference points retain the fixed step.

An existing campaign from `2ad730c82de8a669205d7d635fe450e1255b17cb`
can explicitly adopt the tested release using `adopt-optimization` with its
original `--input-root` and `--output`, after stopping the old controller and
workers. Use the same checkout location. The immutable original manifest and
checkpoint identities remain the numerical lineage; a separate
`optimization_execution.json` records the new execution sources/commit.
The tracked shared `optimization_release.json` admits only the tested source
pair and unchanged inputs/configuration. Every newly computed chunk records
the execution certificate. Ordinary resume checks still validate hashes and
coverage; corrupt chunks are never accepted by the upgrade. Do not edit
receipts or manifests manually. Run `preflight`, `run`, then `validate` after
adoption; valid work is reused. Other source chains require a separate audit.
