# Actual HSX Q05 replay patch

`patch.npz` retains N32 canonical topology and two complete owners: the first
owner in saved trace blocks22 (ordinary non-wall) and63 (wall). It contains
their existing GPU64 endpoints, frozen evaluator responses at exactly the
preparation queries, sparse donor observations for all26 manufactured fields,
explicit D/physical-normal N data, and the independently archived accepted
global actions for both spans. The metadata records owner ids, source trace
hashes and the accepted result hash from the selective campaign
`c0d64dba761087a3273d29cd4728689c4f4b5b67f838873a845d1cdb1a809f8b`.

The topology arrays are from
`geometry_artifacts/rlp_convergence_32_48_64_20260917/32x32x32`.
The source trace blocks are from
`work/q-fci-balanced28-dacf15f1-20260929T171025Z/global/N32`.
Expected actions come directly from `work/q_fci_selective_global_20260929/results_N32.npz`.
Field observations and geometry/BC responses were evaluated using the verified
`work/q_fci_diffusion_freeze_20260929/source` snapshot. No trajectories or
continuum reference were recomputed.

The portable test rebuilds support, magnetic coefficients and both wall lifts
with package code. Its callbacks only replay stored actual HSX geometry
queries; they reject any other query. Boundary-padding invariance is tested
alongside the independent global actions, so both a leaked interior wall lift
and accidentally removed legitimate wall coefficients are detectable in CI.
This is bounded extraction evidence, not new convergence qualification.
