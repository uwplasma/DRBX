# P07 global perpendicular diffusion/polarization qualification

The bounded shared-face cubic candidate is ready for a full actual-HSX
32³/48³/64³ operator accuracy campaign. This advances the perpendicular
roadmap's P07 static accuracy investigation; it does not mark P07 complete.

The runnable contract is
[`scripts/p07_diffusion_global/README.md`](../../../scripts/p07_diffusion_global/README.md).
Its configuration and source/input manifests freeze the numerical choices.
The four independent fields are phi MMS, Ti MMS, regular Neumann and mixed-eta
Neumann. The wall-reaching cubic fit retains the existing point value/physical
normal derivative constraint, selection-v3 and midpoint owner observations.
The positive operator is assembled from shared q3 face fluxes and stored owner
volumes. The reference remains the independent continuous physical-volume
average, evaluated globally with q3 and qualified on bounded complete owners
with q5/q7. Geometry is queried continuously at the required locations.

The preceding three-owner audit is archived locally at
`work/p07_global_refinement_20260923` in the HSX workspace. The portable runner
replays its actions, face fluxes and q7 references; the evidence and validation
limits are recorded in the campaign's `local_validation.md`. Complete-owner
preflight adds axis, transition and periodic-seam coverage without introducing
regional-order gates. The previous wall-track nonmonotonicity remains a
scientific observation for the global campaign to resolve, not a reason to
retune the frozen candidate before running it.

The global gate is physical-volume-weighted operator L2 order >=1.8 on both
refinement intervals for each field separately. Regional squared-error
contributions, maxima, constants, face-oracle comparisons, boundary/support
information and reference-quadrature differences are returned for local
analysis. Scientific pass/fail flags do not authorize remote design changes.

The remote task is computation-only with node-local parallel CPU execution.
The allocation/environment skill chooses resources. Every output, log, cache,
checkpoint and operational receipt belongs in one new downloadable folder.
Only validated, identity-matching chunks may be reused. Final validation
reassembles the actions from chunk data and checks the recorded statistics.

After return, analyze all three resolutions locally and update P07 status.
Structural properties, elliptic-solver checks, evolved checks and eventual
shared model integration remain separate milestones.
