# Q short-span midpoint research campaign

This isolated Q campaign code implements the frozen 2026-09-27 raw-midpoint divergence and complete-owner stored-volume projection. It does not change the production `drbx` operator. The exact P/R screen and numerical N/E runner have separate immutable designs. The numerical runner requires explicit owner IDs and has no global default.

Run from the `DRBX/` directory in the `drb` environment. Substitute an input root containing the canonical geometry and MAKEGRID files, and a writable output directory. Use the preflight cache only when its SHA256 map identities pass. The commands below document a new output, not a request to rerun the already completed local screen.

```bash
conda run -n drb python -m scripts.q_fci_shortspan_midpoint_global.exact_screen freeze \
  --input-root '/Users/yxie/Desktop/HSX drbx' --output /path/to/exact
conda run -n drb python -m scripts.q_fci_shortspan_midpoint_global.checked_run \
  --input-root '/Users/yxie/Desktop/HSX drbx' --output /path/to/exact --N 32 --workers 2
conda run -n drb python -m scripts.q_fci_shortspan_midpoint_global.checked_run \
  --input-root '/Users/yxie/Desktop/HSX drbx' --output /path/to/exact --N 48 --workers 2
conda run -n drb python -m scripts.q_fci_shortspan_midpoint_global.checked_run \
  --input-root '/Users/yxie/Desktop/HSX drbx' --output /path/to/exact --N 64 --workers 2
```

`checked_run` validates all frozen input hashes and the method before each resume. The exact-screen design binds the initial absolute input root. Each atomic owner chunk carries a SHA256 receipt; valid chunks are skipped.

```bash
conda run -n drb python -m scripts.q_fci_shortspan_midpoint_global.numerical_runner freeze \
  --input-root '/Users/yxie/Desktop/HSX drbx' --output /path/to/numerical \
  --exact-screen /path/to/exact \
  --preflight-cache '/Users/yxie/Desktop/HSX drbx/work/q_fci_cross_region_preflight_20260927'
conda run -n drb python -m scripts.q_fci_shortspan_midpoint_global.numerical_runner run \
  --input-root '/Users/yxie/Desktop/HSX drbx' --output /path/to/numerical \
  --exact-screen /path/to/exact --N 32 --owners 9 3464 --workers 2
```

The numerical design saves canonical input SHA256 hashes, repository-relative source hashes, an exact-screen design hash, copied preflight traces, three selected-map archives with member hashes, and the 39-owner full-map representative subset. The exact reuse manifest verifies all 1,227 original chunks and their receipts after relocation. `run` checks these identities before work. It writes compact N/E owner and BC rows, signed actions and diagnostics in stable complete-owner chunks. The optional `--force-fresh-raw` list on `freeze` is for bounded pilots; it never alters the frozen field or candidate method.

For the full campaign, use `campaign.py` through its module entry point. A previously frozen starter directory can be transferred intact and used at a different input/software/exact-screen path. It contains a stable dispatch manifest with 64-owner global groups, a 39-owner cached preflight, a fresh two-owner ordinary/wall pilot at each N, and separate output subdirectories. `freeze` is only needed when creating a new starter where the historical preflight maps are locally accessible; remote execution can start with a transferred frozen starter.

```bash
python -m scripts.q_fci_shortspan_midpoint_global.campaign verify \
  --campaign "$CAMPAIGN" --input-root "$INPUT_ROOT" --exact-screen "$EXACT_SCREEN"
python -m scripts.q_fci_shortspan_midpoint_global.campaign run \
  --campaign "$CAMPAIGN" --stage preflight --N 32 \
  --input-root "$INPUT_ROOT" --exact-screen "$EXACT_SCREEN" \
  --workers "$WORKERS" --host-memory-gib "$HOST_GIB" --worker-memory-gib "$WORKER_GIB"
python -m scripts.q_fci_shortspan_midpoint_global.campaign validate \
  --campaign "$CAMPAIGN" --stage preflight --N 32
```

Repeat the bounded preflight and pilot for N48/N64. The worker count is node-local and must fit both the allocated CPU affinity and the explicit host-memory budget. The CPU backend and one-thread BLAS/JAX settings are established before numerical imports. `rk4_512_raw_ids` are frozen from the 39-owner geometry sample: all raw trajectories receive RK4-256 validity/crossing/reentry/reach checks, while only these 39 IDs receive live RK4-512 checks. An already verified 512 sample can be imported with `--rk4-check-cache` at freeze time when input and trace-source hashes agree; the original receipt hashes and signed endpoint sensitivity are preserved. The remote starter uses live 512 checks.

After a reviewed remote pilot, explicitly run `--stage global` for N32, N48 and N64 using the same command form. Then validate each N and run:

```bash
python -m scripts.q_fci_shortspan_midpoint_global.campaign reduce \
  --campaign "$CAMPAIGN" --stage global \
  --input-root "$INPUT_ROOT" --exact-screen "$EXACT_SCREEN"
```

The reducer streams signed complete-owner N/E/P/R and nominal/half R, sums directional contributions before physical-volume norms, and reports global, topology and fixed-radial-band metrics, both spans and both interval orders. P/R is unavailable for lambda0.5/lambda0.25 stress fields, so they have N/E only and are excluded from the primary N−R gate. Constant controls have zero reference and undefined convergence order. Execution completeness and scientific order results are separate fields. A failed order never triggers a method change.

Chunk files and receipts are atomic. A valid chunk is skipped on resume even if the worker count changes. A missing partner, corrupt hash, conflicting method or stale `.lock` rejects the run for inspection; remove a stale lock only after confirming no writer remains. All new results, caches and logs should stay under one campaign folder. Exact P/R arrays and the eight canonical input files may be transferred as external immutable inputs with their hashes and relative locations intact.

The completed screen and verification receipts are in `work/q_fci_optimized_global_preparation_20260927/`. See its `report.md` for the gate interpretation and resource bounds. No global numerical N/E campaign was executed here.
