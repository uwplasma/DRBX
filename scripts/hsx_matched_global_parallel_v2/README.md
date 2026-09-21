# Portable matched-cubic HSX runner (v2)

This directory is a separate, immutable execution version of the frozen
`joint_cubic_matched_q3` global qualification.  It does not modify the live v1
implementation or any production DRBX operator.  The numerical routines in
`frozen_numerics_v2.py` differ from v1 only in explicit deployment/source/cache
root resolution; the face fits, cell fits, quadrature, reference evaluations,
assembly, fields, and acceptance gate are unchanged.

## Entry points

`parallel_runner.py` takes three explicit roots on every command:

- `--deployment-root`: root of the exact DRBX checkout containing `src/` and
  `scripts/`.
- `--input-root`: directory containing paths listed in
  `portable_configuration.json`.
- `--output-root`: new checkpoint directory for this execution.

The command sequence is:

1. `frozen-stage --stage prepare --resolution N` (or provide a validated
   `N.prepare.npz`).
2. `plan --coverage global --resolution N --plan PLAN`.
3. `execute --plan PLAN --workers W`.
4. `validate --plan PLAN`.
5. `assemble --plan PLAN`.
6. `frozen-stage --stage validate:case_N`.
7. After N32/N48/N64, `frozen-stage --stage merge`, then
   `frozen-stage --stage validate:merge`.

The unchanged implementation preflight and held-out diagnostic are exposed as
`frozen-stage --stage preflight` and `frozen-stage --stage heldout` with their
matching `validate:preflight` and `validate:heldout` stages.

All commands use normal argv arrays and are suitable as stages in the existing
checkpointing supervisor.  No scheduler, transfer, SSH, GPU-launch, or remote
environment setup is included here.

## Work and identity contract

- Global face and raw-cell ranges are deterministic and non-overlapping.
- Workers use the spawn start method and load the context, continuous
  reference, and prepare cache once, then process multiple chunks.
- BLAS/OpenMP thread counts are capped at one per worker.
- Every chunk is atomically replaced only after complete serialization and
  carries array, numerical, execution, prepare, and unit identities.
- A held output lock rejects a second writer. Missing chunks prevent assembly;
  stale identities, corrupt arrays, duplicate indices, and incomplete global
  coverage are rejected.
- Resume reuses only validated chunks and computes missing units.
- Scientific identity is content-addressed by `source_manifest.json`,
  `input_manifest.json`, the portable numerical configuration, and the prepare
  cache hash. Absolute paths are kept only in the execution identity.

`input_manifest.json` enumerates 3,222 files (6,657,076,009 logical bytes) with
sizes and SHA-256 identities. The 5.4 GiB MAKEGRID checksum is inherited from
the qualified continuous-reference sidecar and its size is rechecked; all
other listed hashes are recomputed in `build_manifests.py`.

## Local qualification evidence

The current N32 test plan executes the production face/cell kernels for 48
unique faces and eight raw cells. It includes two collapsed-axis faces, two
physical radial-boundary faces, four cells owned by multi-cell agglomerates,
ordinary interior cells, and outer radial cells. Every selected cell has all
six incident faces, enabling partial A/B/C action and independent omega
reference assembly.

Serial and two-worker outputs are bitwise identical (`maximum_array_abs_difference
= 0`). Their partial action norms are A=6.516466211316343,
B=6.675475446998873, C=6.595361588560376, and the omega-reference norm is
6.362932554882879. A controlled failure preserved two completed chunks; resume
reused exactly those two, computed six, and reproduced the serial arrays and
actions bitwise. Missing, stale/root-mismatched, overlapping-plan, and
concurrent-writer cases were all rejected.

The bounded cold timings are evidence, not a speedup promise. Serial elapsed
was 10.72 s. Two-worker elapsed was 17.85 s because each spawned worker spent
roughly 13.46 s initializing under concurrent I/O; after initialization,
scheduled kernel work completed in roughly 4.14 s versus 5.16 s serial.
Per-worker peak RSS was about 0.65–0.67 GiB, versus 0.85 GiB serial.
Four workers were not attempted while the independent live N64 preparation was
active and the host showed memory compression.

A fresh-process relocation smoke test ran from
`/private/tmp/hsx_matched_relocation_v2_20260920`, resolved DRBX and audit
imports there, resolved every runtime input beneath that root, and reproduced
the original serial arrays/actions bitwise. The local implementation is
CPU-bound NumPy/SciPy/host-loop work. GPU speedup and multi-node execution are
untested and must not be claimed.
