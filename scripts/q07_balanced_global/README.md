# Paired geometry-consistent Q07 material campaign

Compare the frozen original tube with
`D_bal(F) = D_old(F) + [div(b) - D_old(1)] F_center`.
This enforces the prepared geometry's constant-response identity while retaining
nonzero div(B), original h/32 caps, h/16 outer characteristic samples,
selective repair choices, and D/physical-normal N/mixed wall reconstruction.

All complete owners at N32/N48/N64, 18 matched MMS states, four boundary
combinations, and nine regions. Eighteen outputs are the two formulations' three
actions (centered/correction/combined), each for n/Te/Ti. The characteristic
correction is evaluated once and shared exactly. Vi/Ve, vorticity and diffusion
are unchanged and excluded. This is a comparison, not production promotion.
No field-line tracing, field-dependent donor selection or numerical tuning.

## Frozen runtime and inputs

The checked-in `source_bundle.tar.gz` contains the exact research runtime,
including dependencies not yet extracted into committed shared infrastructure.
Always use this snapshot, not an installed DRBX checkout. `design.json` pins
every runtime source and `inputs_manifest.json` pins all data by relative path.
The bundle's repository baseline was 078633ce8a44e7490dc4514b6adbbc5903eeef81;
the campaign commit and source hashes identify the actual executable snapshot.

Extract `source_bundle.tar.gz` into this campaign directory before importing
Python. Extract the supplied `q07-balanced-inputs.tar.gz` into the same directory
(produces `inputs/`). Set `Q07_INPUT_ROOT` to an immutable data root containing
the eight canonical paths in the manifest. The separate supplied canonical
archive contains those exact files if unavailable remotely; never regenerate.

Set `Q07_OUTPUT` to a new campaign result directory. Set both
`JAX_COMPILATION_CACHE_DIR` and `DRBX_CACHE_DIR` under that campaign folder.
Use Python 3.12, CPU JAX x64, single-threaded BLAS. Tested with NumPy 2.4.6,
SciPy 1.17.1, JAX/JAXLIB 0.9.2, netCDF4 1.7.3, h5py 3.16.0, pydantic 2.13.4.

```
python campaign.py verify
python campaign.py preflight
python campaign.py pilot
python campaign.py run --n 32 --workers "$Q07_WORKERS"
python campaign.py run --n 48 --workers "$Q07_WORKERS"
python campaign.py run --n 64 --workers "$Q07_WORKERS"
python campaign.py analyze
python campaign.py validate-completion
```

Do not run `freeze` remotely: the design is already frozen. Each invocation
checks it. `verify` hashes every supplied trace, frozen choice array, baseline,
and canonical geometry/magnetic file. Preflight reproduces 27 saved matched
owners' independent original/balanced N/O/R actions, and six core/bulk owners.
The local test suite separately checks row algebra, constants, owner batching,
volume reductions, checkpoint corruption and independent completion reduction.

## Parallelism, resources and recovery

The runner uses node-local spawn processes with dynamic chunk scheduling;
793 independently resumable complete-owner chunks. It does not use multiple
nodes. Choose CPU workers from actual allocated CPU affinity and host memory.
`Q07_WORKER_GIB` defaults to 3 GiB; set `Q07_TOTAL_MEMORY_GIB` to the usable host
memory budget (the runner reserves an additional 2 GiB for the parent). Maintain
at least 3 GiB free disk. Local execution is limited to two workers for the
pilot projection; the remote runner permits an allocation-appropriate count.
Workers recycle after 24 chunks. Avoid concurrent writers to the same output;
`run.lock` prevents them. Repeat the same `run` command on interruption: only
complete payloads with matching identity, checksum, coverage and gates are reused.

No row cache survived the prior campaign: reconstruction rows are rebuilt from
saved GPU64 endpoints and frozen support choices. Geometry calls shared between
spans are cached within each batch. No new trajectories are integrated. Unchanged
velocity/diffusion calculation and expensive diffusion references are omitted.
The output contains aggregate norms and maximum owner IDs, not all owner actions.

Replay tolerances remain 1e-8; constant N-O stiff-row tolerance 1e-7; the
previously authorized center-b reference guard is 1e-10. No numerical tolerance
has been relaxed. Initial canonical/source and finite/positive-state gates,
complete-owner coverage, frozen choices and trace crossing checks are mandatory.
Scientific order regressions do not halt or authorize changes.

The deterministic analysis produces `totals.npz`, `analysis.json`, `orders.csv`,
`report.md`, `original_global_replay.json`, and `completion.json`. N-O/O-R/N-R
physical-volume RMS, relative RMS, maxima/owners, signed integrals and orders
remain separate, as do smooth/short-wave cases and boundary types. Original
results are independently compared against the prior complete global campaign;
completion independently reduces all chunks again and verifies artifact hashes.
Return the whole campaign output folder and operational receipt for local
scientific interpretation. Conservation/current-phi/SAT, evolution and
production qualification remain open.

## Local readiness evidence

Final preflight: 33 owners, maximum independent bounded N/O/R replay difference
5.87e-12. Seven tests passed, including independent full reduction and corruption
rejection. Two spawned workers matched serial actions exactly. Relocated archive
extraction verified all input/source hashes and passed the same seven tests.
The N64 regional pilot projects 102–158 minutes with two local workers, above
the requested 30-minute limit. No local global run was launched. Peak pilot RSS
was 1.27 GiB; compressed checkpoint projection about 0.11 GiB before caches and
logs. See verification/ for machine-readable evidence. These are local pilot
estimates, not remote timing guarantees.

## Timestamp-independent checksum recovery

The first Perlmutter attempt stopped before numerical preflight: a same-size
rewrite retained the inode, size, mtime_ns and ctime_ns, causing the metadata-only
SHA cache to return old content. This was a real integrity-cache defect, not a
scientific failure or grounds for weakening the regression test.

The runner now hashes current bytes on every checksum request. Stat information
is only a best-effort concurrent-write check. Large canonical inputs are hashed
at verification and worker initialization, not in the numerical chunk loop.
The original seven tests remain, with two deterministic tests for identical-stat
rewrites and corrupted same-size checkpoints. Numerical code, source bundle,
input archive, canonical hashes and tolerances are unchanged. Only the runner's
source digest changes the campaign identity to
`117daaec352ad2817b3dc9a4c4db5b9f6ee939fb9c4fb1b98a3404fed646694b`.
The preceding evidence retains its original identity; do not relabel it.
Recovery evidence is in `verification/hash_recovery/`.

For the reported preflight-only failure, retain the same remote RUN and existing
immutable inputs. Export the new pinned campaign revision into a separate
revision directory under RUN, keep the old source/logs/verification receipts,
and use a new result/log subdirectory. Rerun verify, all nine tests, preflight,
pilot and then the prescribed global stages. No scientific chunks existed, so
there is no numerical checkpoint migration. If any chunks are discovered,
stop rather than silently reusing them under the new identity.
