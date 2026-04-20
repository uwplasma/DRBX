# JCP Manuscript Outline

This page is the tighter manuscript-facing companion to [PLAN.md](../PLAN.md).
It is meant to keep the introduction, results narrative, and claim boundary
coherent while the paper is being drafted.

## Working paper position

The current paper should be written as a computational-physics methods and
verification/validation paper with a clear software-and-reproducibility contract.
It should not be written as a broad “complete replacement for the full mature
reference-code ecosystem” paper.

That means the introduction and results should keep returning to the same
message:

- `jax_drb` is a research-grade, restartable, JAX-native drift-reduced Braginskii code;
- it has explicit capability tiers and a versioned parity/validation ladder;
- the selected promoted native lanes plus the general 3D adapter infrastructure
  are the supported claim boundary;
- broader production temperature/detachment workflows and broader full 3D
  production workflows remain future work, not hidden assumptions.

## Introduction tightening

The introduction should have four short jobs.

### 1. State the computational problem

- Edge/SOL simulation needs coupled transport, sources, closures, geometry
  handling, restartability, and strong verification/validation practices.
- Existing frameworks such as Hermes-3, GBS, GRILLIX, TOKAM3X, and
  related reduced-model tools set a high bar for physics scope and benchmark
  discipline.

### 2. State the software/methods gap

- JAX-native scientific solvers often emphasize differentiability or kernel
  performance before they demonstrate parity, restart, benchmarking, and
  research-code usability.
- Plasma edge codes often have strong physics scope but are not built around a
  JAX-native execution and differentiability stack.

### 3. State the contribution precisely

The paper contribution is not “everything is done.” It is:

- a restartable, research-grade JAX-native solver stack;
- explicit parity and capability-tier accounting;
- promoted native lanes with exact or tightly bounded reference-backed evidence;
- reusable 3D geometry/observable/parity infrastructure spanning tokamak,
  traced-field-line, and VMEC-style families;
- a compact but real differentiable lane with sensitivity, uncertainty
  propagation, inverse design, and scaling artifacts.

### 4. State the claim boundary

The introduction should say this explicitly:

- selected promoted native lanes plus general 3D infrastructure: yes;
- broad parity-complete standalone DRB replacement across the full target matrix: no.

That statement belongs early, not buried in the discussion.

## Results-section tightening

The results should be ordered as an argument, not as a changelog.

### Result 1. The code is numerically correct on promoted operators

- MMS and observed-order evidence first.
- This is the verification floor for everything else.

### Result 2. The promoted native 1D/2D lanes match the reference on explicit compare surfaces

- Use Hermes-backed parity summaries and bounded transient windows.
- Keep capability tiers visible in every figure/table caption.

### Result 3. The code reproduces physics-facing 2D dynamics and benchmark geometry visuals

- Blob and diverted-tokamak visuals belong here.
- Use compact diagnostics and carefully chosen stills, not only movies.

### Result 4. The 3D architecture is genuinely general, not tokamak-only

- Show the shared tokamak / traced-field-line / VMEC adapter model.
- Present reduced native rungs as infrastructure evidence rather than as full
  production turbulence claims.

### Result 5. The JAX-native claim is real, not branding

- Runtime/scaling on reduced native kernels.
- Profile audit and compilation/warm-run behavior.
- Differentiability results: sensitivity, uncertainty propagation, inverse
  design, and strong scaling on the same native differentiable lane.

### Result 6. The software is reproducible and reviewable

- committed artifact bundles;
- versioned scripts;
- release/package/reproducibility surface;
- explicit statement of what is supplemental vs main text.

## Recommended main-text figure order

1. Architecture and validation ladder
2. Governing-equation and geometry summary
3. Verification: MMS and observed order
4. Hermes parity summary across promoted native lanes
5. Controller/recycling/detachment reduced-lane evidence
6. Neutral plus direct-tokamak transient validation panel
7. 2D dynamics and benchmark geometry visuals
8. General 3D geometry infrastructure panel
9. Reduced native 3D runtime/profile audit
10. Differentiable lane: sensitivity, uncertainty, inverse design, scaling

## Discussion and limitations

The discussion section should not sound apologetic, but it should be explicit.

What the paper can say strongly:

- the selected-lane matrix is defensible;
- the 3D architecture is already broader than one benchmark geometry;
- the code is reproducible, packaged, and reviewable;
- the differentiable lane is real and not only conceptual.

What the paper should still mark as future work:

- broader production temperature/detachment workflows;
- broader end-to-end production 3D workflows;
- the broad standalone parity-complete replacement claim.

## Drafting rule

If a result does not already have:

- a committed script entry point,
- a committed artifact bundle,
- a documented capability tier,
- and a bounded interpretation,

then it should not be promoted to the paper’s main claim surface.
