# Frozen Q03 direct-cubic static global accuracy campaign

Use `python -m q03_direct_campaign.remote` from the repository root with
`PYTHONPATH=src:scripts`. The remote worker uses its **run drbx on perlmutter**
skill for allocation, environment setup, concurrency and supervision.
Computation only: no scientific interpretation, optimization or numerical edits.

## Frozen inputs and numerical scope

Transfer and extract the separately supplied `q03_direct_seeds_20260922.tar.gz`.
`seed_manifest.json` pins its root manifest, which checks every numeric file.
This 333-MB uncompressed bundle includes all N32/N48/N64 scalar-owner moments,
states, canonical face charts/incidence, q9 polynomial targets, volume and
region measures, grid-face coordinates and bounded preflight evidence. It does
not contain the previous exchange candidate's outputs or expensive leg pools.
The local-only exporter is `export_seed.py`; do not run it remotely.

Two existing immutable datasets remain external:
`hsx_metric_d58d392545fd3917efeb83b6.npz` and
`mgrid_res2p5cm_180pln.nc`. Both are pinned in the seed manifest. The direct
prototype's historical metric alias resolves to this corrected d58 cache.
Old Q01 source manifests retain their historical cache hash; the new reference
identity explicitly records the actual d58 cache. No substitution is permitted.

The candidate uses all 20 cubic modes, 120 scalar owners on five eta planes,
raw-midpoint volume-weighted owner means, fixed distance weights, shared flux
and opposite incidence, unit diffusivity, and prescribed continuous boundary
flux. Fields are radial_eta, angular_x, mixed_y_eta and constant. No limiter,
donor exchange, support tuning or higher degree is included. Positivity fails
for the unrestricted action on some nonnegative states; this is a static
accuracy study, not Q04 certification. Dissipation and evolved MMS are pending.

Fresh q9/q11 analytical face reference fluxes are generated remotely using
the continuous evaluator. The analytical field class and quadrature routine
are frozen extractions of the audited research definitions. No geometry
regeneration, cell-center metric interpolation or historical workspace import
is performed. No complete reference quadrature point cloud is retained.

## Execution

`SEEDS` is the extracted seed directory. `CAMPAIGN` is one new output folder.
`METRIC` and `MAKEGRID` identify the two external files. The setup skill must
choose `Q_WORKERS` and `Q_REFERENCE_WORKERS` independently: reference workers
load continuous geometry and use more memory than the NumPy reconstruction.
Do not prescribe either count or an allocation in the handoff.

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
python -m q03_direct_campaign.remote verify \
  --seeds "$SEEDS" --metric-cache "$METRIC" --makegrid "$MAKEGRID" --output "$CAMPAIGN"
python -m q03_direct_campaign.remote preflight \
  --seeds "$SEEDS" --metric-cache "$METRIC" --makegrid "$MAKEGRID" --output "$CAMPAIGN" \
  --workers "$Q_WORKERS" --reference-workers "$Q_REFERENCE_WORKERS"
python -m q03_direct_campaign.remote run \
  --seeds "$SEEDS" --metric-cache "$METRIC" --makegrid "$MAKEGRID" --output "$CAMPAIGN" \
  --workers "$Q_WORKERS" --reference-workers "$Q_REFERENCE_WORKERS"
python -m q03_direct_campaign.remote validate \
  --seeds "$SEEDS" --metric-cache "$METRIC" --makegrid "$MAKEGRID" --output "$CAMPAIGN"
```

Verify checks hashes and records the Python/NumPy/SciPy environment.
Preflight evaluates fresh references on the existing 379/523/620-face HSX
samples, compares saved references/prototype actions, and compares serial
against the requested parallel execution. It is an implementation check,
not an additional convergence gate or a requirement of historical donor IDs.
Run prepares and executes N32, then N48, then N64. Validate verifies every
chunk receipt, reassembles all actions, and writes `operator_orders.json`.
Global volume-weighted L2 order >=1.8 on both intervals, with the reference
difference below 10% of spatial error, is evaluated separately for all three
nonconstant fields. Constants remain an algebra check. Scientific failure
returns completed computation and unmodified flags; it does not authorize tuning.

## Output and recovery

Keep every generated file beneath the single campaign folder, including logs,
scheduler output, temporary files, caches and receipts. Set TMPDIR,
XDG_CACHE_HOME, DRBX_CACHE_DIR, JAX_COMPILATION_CACHE_DIR and HSX_JAX_CACHE there;
set PYTHONDONTWRITEBYTECODE=1. Collect any unavoidable external generated
artifacts into the folder as real files before returning it.

One controller lock prevents concurrent campaign writers. Resume the same
folder, source, environment and numerical inputs. Reference chunks checkpoint
every 240 faces; action chunks every 512 faces. Incomplete chunks are rebuilt,
valid chunks reused, incompatible/corrupt receipts rejected. Worker counts
may change. Re-run the missing/failed prescribed stage, then the later stages.
Do not delete manifests, alter numerical parameters or force reuse. Return
numerical/identity failures for local follow-up.

Return the entire folder: campaign/input provenance, `preflight/`,
`preflight.json`, `inputs/N*/` (including reference checkpoints),
`results/N*/chunks/`, action arrays and summaries, `operator_orders.json`,
`validation.json`, `operations/`, logs, job records and an operational receipt.
The receipt lists the pinned revision, input identities, job IDs, concurrency,
elapsed times, exit statuses, completed/missing stages and the exact failed
command if needed. Return the absolute download path. Do not interpret results.
