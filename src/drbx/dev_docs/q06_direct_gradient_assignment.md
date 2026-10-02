# Q06 direct parallel gradient: implementation and bounded verification

Implement the direct gradient under the
[current frozen Q06 contract](q06_traced_gradient_divergence_contract.md).
Do the implementation and bounded tests, then report for parent code review.
Do not implement divergence/composition yet, launch a global run, retrace,
change frozen numerics, promote defaults, commit or push.

## Starting point and scope

Read workspace AGENTS.md, docs/code_structure.md, docs/testing_strategy.md,
src/drbx/dev_docs/README.md, q_traced_extraction.md, the frozen diffusion contract,
and work/q05_traced_extraction_20260929/review_v2/report.md (workspace-relative).
The Q05 user closure is an acceptance decision on bounded evidence; full-domain
action replay remains unperformed and is not this assignment's prerequisite.
Concurrent P/production edits share this checkout. Touch only Q files needed for
this task and preserve all unrelated edits. Work in the saved workspace directly.

Existing package entry points:

- src/drbx/stencils/q_parallel.py: PreparedQ and checked schema-v2 artifacts.
- src/drbx/native/q_parallel.py: current diffusion apply and QBoundaryData.
- tests/test_q_parallel_hsx_portable.py and tests/data/q05_actual_hsx/: portable
  real-HSX fixture, topology, geometry responses and BC observations.
- scripts/q05_bounded_matrix.py: fixed 21-site actual-HSX selection.
- workspace work/q05_traced_extraction_20260929/review_v2/chunks/: corrected
  saved preparations; old parent-directory v1 artifacts must not be reused.

Prefer new q_parallel_gradient modules in stencils/native and a new Q06 adapter,
rather than altering Q05 numerical preparation or its evidence scripts.
Reuse small common helpers where sensible; avoid a broad refactor or a second
independent reconstruction implementation. Existing prepared derivative rows
are sufficient: no geometry evaluation or new fitting is needed for contraction.

## Host construction

Add a clearly named host builder, e.g. prepare_parallel_gradient(prepared_q).
Validate its source PreparedQ and preserve source/geometry/topology/trace/span
identities, donor selection and complete-owner projection.
The three stored slots are minus cap, plus cap, midpoint (index 2); assert this
against source metadata/points, not an inferred fine-cell face convention.
For raw member r, slot s and donor d, form

    C_kind[r,s,d] = sum_a magnetic_b[r,s,a] * row_gradient_kind[r,s,a,d]

These derivative rows and b components are both in logical coordinates; do not
apply another coordinate Jacobian or physical-normal conversion.
Prepare affine lifts using the SAME contraction:

    LD_node[r,s,j] = sum_a b[r,s,a]*boundary_gradient_D_trace[r,s,a,j]
    LN_normal[r,s,j] = sum_a b[r,s,a]*boundary_gradient_N_normal[r,s,a,j]
    LD_tangent[r,s,:] = b[r,s,1:] on wall rows, exactly zero elsewhere.

The tangential term multiplies prescribed query theta/eta derivatives; it has
no radial component. Dirichlet query value is unused by G. Keep that explicit
without dropping it from the shared QBoundaryData contract.
Audit signs and slot axes against existing wall algebra. Never derive individual
gradient lifts by dividing the diffusion boundary rows by magnetic_L: outer
contraction has already combined slots and may contain zeros/cancellation.

Provide a lean center-only runtime view and an optional three-slot diagnostic
view. Do not stage full Cartesian derivative arrays, support diagnostics or
geometry with the runtime gradient. Reuse donor/owner arrays where feasible.
Use a distinct operator identity; do not label G artifacts as diffusion actions.
If persisting a gradient bundle, require full array checksums and dependency
identities and reject incompatible schema; never overwrite Q05 preparations.
Persisting separate coefficients is optional if deterministically derived from
the checked PreparedQ at setup. Receipt identity for the new implementation is
mandatory either way.

## Runtime API

Implement an explicit pure-JAX apply_parallel_gradient(runtime, owner_values,
boundary, kind='D'|'N') returning (..., n_chunk_owner): gather donors, contract
center rows, add center BC lifts, then project complete raw members with stored
weights. Batched input shape is (..., n_owner). Validate BC field axes consistent
with Q05. Support real64 and complex128 and changing BCs under a reused JIT.

Expose an optional separately named raw-slot diagnostic application returning
(..., n_raw, 3), with documented slot order. Keep arbitrary field batch axes
unambiguous; do not average cap values as though they lived at owner centers.
Stage the actual lean runtime object used by eager/JIT application.
No runtime fitting, NumPy/SciPy evaluation, tracing, wall solve, state-dependent
support changes or Python callbacks are allowed.

## Verification

1. Portable tests built from actual-HSX fixture: explicit contraction of stored
   Cartesian derivative rows and b agrees with precontracted application, including
   D/N lifts. Independent manufacture/algebra controls should accompany this
   internal consistency test; avoid merely asserting one implementation twice.
2. Compatible constant data yield zero gradient. Poison finite unused nonwall
   boundary entries and prove exact invariance; separately prove wall data have
   an effect. Test nonzero D tangential and nonzero physical-normal N data.
3. Compare eager/JIT and batched vs separate application, complex inputs, JVP
   with respect to state and BC data, and fixed coefficients with changed BCs.
   Preserve the affine interpretation of homogeneous versus prescribed lifts.
4. Recontract the three-slot gradients with PreparedQ.magnetic_L and compare to
   existing raw diffusion and projected apply_q for both spans/BCs. This is a
   wiring equivalence control, NOT proof that D_h(G_h f) equals frozen diffusion.
   Use unchanged 1e-8 absolute diffusion replay tolerance and report actual maxima.
5. Use the 21 predefined complete-owner patches at N32/N48/N64, both spans and
   both BCs with all 26 frozen fields. Read corrected saved chunks, validate
   provenance; derive matching observations/BCs with the frozen research adapter.
   Compare center G and raw cap G against analytic manufactured derivatives
   contracted with the frozen b. The reference must not use reconstructed
   derivatives or the implementation under test. For midpoint G, O=R by contract.
   Report absolute RMS/max, field/region breakdown, repair choices and worst
   owners. Compare center rows/actions across spans: support is frozen common to
   both, so differences should be roundoff; diagnose rather than retune if not.
   Site orders are descriptive only, not a global convergence claim.
6. Rerun focused Q05 regression tests after changes and relevant curated checks.
   Preserve all existing diffusion outputs, selection choices and tolerances.

Existing Q05 fixture includes accepted diffusion actions, not an independent
gradient accuracy oracle. Extend a small portable fixture or use analytic scalar
controls at its actual-HSX points for gradient verification; document provenance.
Keep broad scientific MMS scoring in the bounded adapter, outside package code.

## Performance and execution limits

Local CPU only, BLAS threads one, at most two CPU workers, total target RSS <=6GiB,
incremental disk <=2GiB. Prefer sequential streaming, verified saved artifacts
and one-resolution observation caches. No full-grid preparation or long global
replay. Measure host contraction, staging, first JIT, warmed apply, changing-BC
apply, runtime bytes and peak process RSS separately. No GPU speed claim.
If bounded work unexpectedly expands beyond about 30 minutes, preserve progress
and report the measured bottleneck instead of silently widening the run.

## Deliverables

- Shared host and pure-JAX direct-gradient code, meaningful portable tests,
  updated selectable-interface documentation and a Q-specific bounded adapter.
- workspace work/q06_direct_gradient_20260929/report.md, validation.json,
  performance.json, machine-readable per-field/site errors and source/input
  hashes. Record O=R for direct midpoint G and keep cap errors separately named.
- Parent-review summary: changes, exact validation commands/results, limitations,
  whether bounded G is ready for review, and next D_direct/D_tube comparison.
  Do not claim Q06/global/production certification from this bounded exercise.
- Update only the Q roadmap's current worker-result pointer/status after reporting;
  retain user Q05 closure and all reference/structural caveats. Do not edit P files.
