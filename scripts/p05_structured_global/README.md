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
owner. Candidate face rule is q3; the cell-volume correction uses q1 midpoint evaluation. Curvature of geometry enters
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

The primary reference evaluates the analytic strong bracket once at every raw
cell midpoint, including the actual continuum generalized-potential vorticity.
It is projected using stored physical raw volumes and divided by stored owner
volume. Actual-omega derivatives use fourth-order finite differences of the
continuum expression, with radial steps clipped inside the domain; no numerical
reconstruction enters the reference. Wall trace conventions remain unchanged.

Bounded complete-owner controls compare midpoint references at the nominal and
half differentiation steps. That empirical derivative uncertainty must be below
10% of the corresponding same-sample primary numerical error. Integrated reference controls are disabled at the user's request. All A/B/C/U forms use the
same primary midpoint reference. The old global q5 face-reference stage and
q5/q7/q9 reference controls are removed from this campaign.

Candidate face/cell exact-input arrays remain diagnostics. The actual-omega
swapped face oracle is unavailable because face reference evaluation does not
request its full gradient; it is never the primary strong midpoint reference.
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

## Identity and historical campaigns

This is a new midpoint numerical configuration (v2). Existing integrated-reference
campaigns cannot resume or use `adopt-optimization` under it. Use a new output
folder. No checkpoint importer is implemented. Historical results retain their
original method/reference identities. Geometry caching, worker parallelism,
q3 shared-face construction and reconstruction policies remain in place.
