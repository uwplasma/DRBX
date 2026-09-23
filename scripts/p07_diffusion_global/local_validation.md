# Local portability and replay evidence — 2026-09-23

Validation ran from a separate checkout of committed base `12dcf7c7` with only
this campaign and its focused test file added. It did not import the loose P07
prototype driver or uncommitted package extensions. All 50 source-manifest
entries were checked against committed files or this campaign's new files.

Local evidence directory, relative to the HSX workspace:
`work/p07_portable_validation_20260923/`.

- **Archived replay:** `replay/replay.json` compares every saved face and all
  three complete owner actions at N32/N48/N64 against
  `work/p07_global_refinement_20260923/N*.evidence.npz`. Donor counts and IDs
  exactly match the frozen package face builder. Across all four fields and
  all resolutions, maximum action difference is `1.53e-13`, maximum face-flux
  difference `6.51e-19`, and maximum q7 reference difference `6.22e-15`.
  Hashes of the archived inputs are included. Later runner changes added
  operational receipt history and stronger validation; numerical kernels are
  unchanged from this replay.
- **Actual-HSX preflight:** all seven complete-owner selections completed at
  all three resolutions, including axis and periodic seams. The largest
  selected-sample weighted q7-minus-q3 reference discrepancy was 0.820% of that
  sample's operator error. This is bounded reference evidence, not a global
  reference-error proof.
- **Parallel execution:** `release_serial/N32/preflight.npz` and
  `release_parallel/N32/preflight.npz` agree exactly for every saved numerical
  and topology array. This compares one versus two CPU processes.
- **Recycling:** the two-process run used a two-chunk process lifetime to
  exercise replacement workers. N32/N48/N64 completed with 4/5/6 distinct
  worker PIDs. The production default is 128 chunks per process to amortize
  initialization. Preflight worker execution elapsed times were approximately
  27/27/36 seconds with deliberately frequent recycling. These are bounded
  preflight timings, not full-campaign runtime estimates.
- **Memory:** peak observed worker RSS across those checks was 0.836 GiB.
  This is per worker, not total job memory. The controller also holds a
  geometry/evaluator context. Remote resource choices belong to the remote
  setup skill and should include headroom.
- **Resume:** repeating N32 preflight in the same folder with two workers
  reused all eight completed units and executed zero numerical units. Saved
  arrays remained identical. Invocation and execution histories are retained.
- **Focused tests:** `pytest -q tests/test_p07_diffusion_global_campaign.py`
  passed all seven tests: face numbering/incidence, chunk coverage, memory
  concurrency cap, exclusive writer, checkpoint integrity/identity rejection,
  signed assembly/normalizations, and independent completed-result validation
  including rejection of changed summary statistics. These are bookkeeping
  tests; the real-HSX preflight/replay supplies numerical evidence.

The complete global runner has not been executed locally. The remote campaign
must produce and validate full N32/N48/N64 arrays before global order can be
assessed. No global accuracy or production pass is claimed by these checks.
