# Single-center-seed global comparison campaign

Run with `--seeds 1` at N32, N48 and N64. This is a separate numerical candidate
for comparison with the RK4 four-seed campaign pinned to
`fd1daa275ce51c9d6bdd1abac658baa48192da2a`. The assignment does not cancel,
restart, migrate or reinterpret that campaign. Keep both output folders.

## Matched numerical contract

The only intended mathematical change is transverse sampling: one seed at the
logical cell center, with the whole cell's transverse area, replaces the four
quarter-cell midpoint seeds weighted by one quarter of that area. Both trace
one eta interval in each direction with the same compiled float64 RK4, 64
substeps, and fourth-order connection-length integration. One directional
observation still comes from each cell. There are still four scalar fields.
For one seed the magnetic weight cancels, giving direction*(T_end-T_source)/ell.
The corresponding potential-moment observation matrix uses that same endpoint
functional; it must not reuse the four-seed face weights.

Frozen inputs, metric/MAKEGRID software, currents, midpoint owner states,
cubic endpoint fitting, cubic-potential return basis, nearest-96 and interval
coverage rules, halo expansion, q5 face target, q9/q11 reference faces, q5/q7
continuous volumes, strong-volume checks, boundary loads and global recipients
are unchanged. The exact same rank/target, constant, conservation and global
accuracy gates apply. Each nonconstant field must attain global weighted L2
order >=1.8 on both refinement intervals, with empirical reference fraction
<0.1. This is static accuracy qualification, not evolved/production certification.

The same geometry-based support algorithm is used, not a forced identical list
of selected rows: seed-dependent flux centroids and wall validity can change
which observations are admissible or selected. Such differences are part of
the candidate and are retained in face diagnostics. Axis and wall owners remain
in both global domains. A failed seed excludes an observation, not its recipient.

## Cost and reproducibility

Observation counts stay 65,536 / 221,184 / 524,288. Single-seed trajectory-leg
counts equal those counts: 811,008 overall, versus 3,244,032 for four seeds.
Endpoint-reconstruction counts also fall by four. Face fits and reference
integration remain, so total walltime does not scale by exactly four.

This standalone campaign recomputes its references using the identical rules.
Although those references are mathematically seed-independent, importing foreign
checkpoints would bypass the existing whole-campaign identity contract. Do not
copy or relabel any four-seed checkpoint. Reuse only the immutable input dataset
and software environment. Reference and action arrays, owner/face IDs, numerical
and exact-secant channels, support diagnostics and timings are returned for a
later local comparison; remote scientific comparison is not requested.

`configuration.json` is the frozen four-seed base. The CLI resolves the selected
seed count, seed pattern and observation description into `campaign.json`.
Its identity and every downstream checkpoint identity include that resolved
configuration. A mixed-seed resume is rejected. Source changes also require a
new folder, even when selecting four seeds. The seed dimension in trajectory
arrays is 1 or 4; the scalar-field dimension remains 4.

## Remote execution

Use the README's exact verify/preflight/run/validate/status sequence with
`SEEDS=1`. The remote run must perform fresh seven-owner preflight at all three
resolutions before global execution. A preflight's bounded error slopes are
outputs, not a new global acceptance gate. Unresolved face functionals are
execution failures to return for local follow-up, not permission to tune.

Preflight and global tracing, endpoint reconstruction, face maps/references,
volume integrals and preflight strong-volume controls use the selected node-local
CPU process pool. All substantial kernels are parallel; stage dependencies,
resolutions and final deterministic reductions remain ordered. One parent writes
the campaign. Choose allocation, affinity, workers and memory through the remote
setup skill; record actual worker memory, and keep all caches and logs inside
the single new campaign folder.

## Bounded local validation

The reproducible actual-HSX check is `validation/single_seed_local.py`. It runs
one complete ordinary owner per resolution through the real two-worker trace,
face, continuous-volume and assembly stages; checks center coordinates and the
one-seed secant identity; exercises checkpoint resume; probes axis/wall/seam
trajectories at 64 and 128 steps; replays a stored four-seed batch; and rejects
mixed-seed reuse. Results are recorded in `validation/single_seed.json`.
This is not a full local preflight or a global convergence claim.

Run from the repository root using the existing HSX workspace inputs and the
historical four-seed smoke file under `work/q_fci_rk4_validation_20260923`:

```bash
python scripts/q_fci_return_campaign/validation/single_seed_local.py \
  --workspace "$WORKSPACE" --output "$CHECK"
pytest -q tests/test_q_fci_return_campaign.py
```

The geometry-independent tests check checkpoint corruption, identity/coverage
mismatch, writer exclusion, supported seed patterns and equality of all frozen
configuration fields except the intended sampling metadata.

### Recorded results (2026-09-23)

All bounded checks passed. Seven checkpoint/configuration tests pass.

| N | Observations | Complete faces | Minimum rank | Maximum target defect | Maximum constant action |
|---|---:|---:|---:|---:|---:|
| 32 | 260 | 6 | 19 | 4.707e-14 | 1.478e-14 |
| 48 | 260 | 6 | 19 | 1.053e-13 | 7.976e-14 |
| 64 | 260 | 6 | 19 | 2.916e-14 | 8.277e-14 |

The N32 resume test reused 68 trace checkpoints without changing payload hashes or modification times.
Axis/wall/seam probes retained identical 64/128-step validity decisions; the largest regular-coordinate endpoint change was 1.484e-09.
Recorded peak RSS across bounded face/volume workers was 0.79 GiB; this is not a full-campaign memory bound.
The stored four-seed batch replay passed, and attempting four-seed verification in the single-seed output folder was rejected.
No global or full seven-owner preflight was run locally.
