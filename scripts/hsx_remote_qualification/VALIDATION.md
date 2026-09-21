# Local validation of the clean remote campaign

This validates the handoff implementation, not global operator convergence or
execution on Perlmutter. No new global study or remote scaling run was made.

* Twelve focused tests pass: all-octant perturbations, near-cutoff candidate
  inclusion, canonical storage-order independence, anchored non-transitive tie
  groups, stale-schema rejection, global plan coverage, gate bookkeeping and
  existing persistent-worker controls.
* [Selection audit](../../../work/hsx_clean_remote_v3_selection_check/summary.json):
  78 actual HSX rows at each of N32/N48/N64, across all face orientations and
  several eta planes, including the reported `[24,23]` location. Indexed and
  exhaustive donor sets and weights agree exactly. Fresh boundary derivatives
  are finite for all 1,024 / 2,304 / 4,096 radial boundary rows.
* [Fresh N32 preflight](../../../work/hsx_clean_remote_v3_validation/N32.preflight.json):
  42 faces and seven cells, including collapsed-axis and physical-wall faces;
  finite outputs, constant reconstruction errors at floating-point scale,
  and indexed/exhaustive agreement. Preparation was rebuilt under v3. The
  verification reused a local JAX compilation cache; it is not a cold-runtime
  performance claim.
* [Execution and recovery check](../../../work/hsx_clean_remote_v3_execution_check/summary.json):
  the eight-chunk actual-HSX N32 plan gives identical arrays, donor hashes and
  complete sampled-cell actions with one and two workers. Deleting one owned
  test checkpoint causes exactly one recomputation; the other seven are reused.
  The recovered arrays/actions are again identical. This is a correctness
  check, not a remote scaling study.
* Before packaging, full content verification passed for 45 immutable input
  files and the original broad source manifest. The committed manifest now
  covers 106 transitive local Python dependencies, including the HSX blob
  driver imported by the audit helpers. It excludes unrelated research work.
  Every source entry was checked against its staged Git content. Eight campaign
  tests pass from an export of that exact staged tree; all 99 imported local
  modules are covered, with no imports leaking back into the dirty workspace.
* The clean staged-tree [N32 preflight](../../../work/hsx_clean_remote_v3_commit_check/N32.preflight.json)
  also passes after full input/source verification. All 17 numerical preparation
  arrays equal the earlier local validation exactly. This checks that excluding
  unrelated workspace edits does not change the prepared numerical inputs.
  The staged-tree [execution/recovery repeat](../../../work/hsx_clean_remote_v3_commit_execution/summary.json)
  passes too: eight chunks, identical serial/parallel arrays and sampled-cell
  actions, and exactly one chunk recomputed after deleting its checkpoint.

The remote `preflight` command deliberately runs fresh checks at **all three**
resolutions on that platform. Cross-platform donor equality with historical
v1/v2 outputs is not required. The `run` command defaults to 64 workers and
computes a new N32/N48/N64 chain. Its convergence/reference verdict remains
unknown until those remote calculations complete.

The local P pause is independently recorded in the
[pause receipt](../../../work/perpendicular_matched_global_qualification_20260920/supervision_v1_4_parallel_2w/pause_receipt.json):
no owned workers remain, its heartbeat is paused, and 566 historical N64
checkpoints were preserved. Those checkpoints are not inputs to v3.
