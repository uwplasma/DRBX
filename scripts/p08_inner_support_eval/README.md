# P08 inner-support evaluation: C0 vs candidate inner supports (remote campaign)

## Purpose

Decides between candidate inner donor supports for P's perpendicular rows by comparing each against the C0 baseline
(`inner_support="profile7"`) **on identical owners**, with the real P row builders and operators and Q's field
catalogue (26 scored fields, autodiff curvature K, q2 face quadrature). For each grid and candidate, `run.py` builds the
owner-closure rows for C0 and the candidate, then evaluates

* reconstruction errors (R1 cell value/gradient at raw midpoints, R2 face-node value/gradient) by component;
* P07 per owner: N (integrated R4 rows), O (exact gradient on the same q3 integrand), R (continuum); N-O, O-R, N-R;
* P06 q1 (K.grad) and the P05 centered bracket and face jump;
* a short-wave response report (never part of an order or pooled statistic);
* the candidate-aware **dispatch check**: a row that is coupled in neither C0 nor the candidate is bitwise identical;
  a row coupled in both is identical unless the candidate's rows carry `donor_policy == "nearest28"`; every changed
  row is `coupled_quartic` in the candidate (R4: family 7 stands for coupled; nearest-28 candidates may change
  family-7 rows). Zero violations is required; the list is capped at 20 but the count is complete.

Candidates (`configuration.json`): `C1=last_aggregate`, `C2=last_aggregate_nearest28`, `C3=fixed_radius`. In every
results file the candidate is mode key `C1` (C0 is `C0`); the campaign name is recorded as `candidate`. Grids 32, 48, 64,
12 owners per ring and phase (`--per-ring 12`), fixed u-bands in `configuration.json`.

Nothing here edits `src/drbx`, the frozen `p0[0-7]*_global` and `perpendicular_structured` packages, or
`p08_step1_global` (imported read-only: `localize_sidecar`, the input manifest). The evaluation is single-node.

## Inputs

* `--input-root`: the immutable HSX input root, the one that holds the files listed in
  `scripts/p08_step1_global/input_manifest.json` (17 files incl. the ~5.8 GB `mgrid_res2p5cm_180pln.nc`; the same root the
  P05N/P06N/P07N/P08 step-1 campaigns use). Locally it is `/Users/yxie/Desktop/HSX drbx`. No oracles are needed.
* The continuum sidecar is localized into `<output>/localized_sidecar.json` (paths rewritten under `--input-root`) and
  passed to every `run.py` as `--sidecar`.
* `sample.json` (13.6 KB, Q's 36-owner audit sample of 28 Sep 2026) is vendored in this package and read package-relative.

## Memory and cores

Each `run.py` is **single-threaded** and peaks at about **5-6 GiB RSS at N64** (N32 and N48 are smaller; a per-ring-1
N32 preflight peaks at ~2.7 GiB). The campaign starts up to `--jobs J` of them at once, finest grid first, so choose
`J <= floor(available GiB / 6)` and `J <= cores`; there are 9 independent (candidate, grid) tasks, so J > 9 gains
nothing. Every subprocess gets `JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Rough time per task on a loaded laptop at per-ring 6: N32 5 min, N48 8 min,
N64 12 min; per-ring 12 roughly doubles the owner count.

## Exact command sequence

From `DRBX/scripts` of the pinned checkout (test the commit from `git archive`, not a working tree), with the `drbx`
environment active; `IN` is the input root and `OUT` a new campaign folder:

```bash
cd DRBX/scripts
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
IN=/path/to/hsx_inputs        # holds input_manifest.json's files
OUT=/path/to/p08_inner_support_eval_out

python -m p08_inner_support_eval.campaign verify-inputs --input-root "$IN" --output "$OUT"
python -m p08_inner_support_eval.campaign preflight     --input-root "$IN" --output "$OUT" --candidates C2
python -m p08_inner_support_eval.campaign run           --input-root "$IN" --output "$OUT" \
    --grids 32,48,64 --candidates C1=last_aggregate,C2=last_aggregate_nearest28,C3=fixed_radius --per-ring 12 --jobs J
python -m p08_inner_support_eval.campaign validate      --output "$OUT"
python -m p08_inner_support_eval.campaign analyze       --output "$OUT"
```

`--candidates ... --per-ring 12` are the defaults of `run` (and of `verify-inputs`); `--jobs` defaults to 1. Give
`verify-inputs` and `run` the same `--candidates` and `--per-ring`, otherwise the identity check refuses the folder.

* `verify-inputs`: hashes every file of the input manifest (sha256, size; ~10 s to minutes depending on the file system),
  localizes the sidecar, writes `campaign_manifest.json` and `inputs_verified.json`. `run` and `preflight` re-hash only
  if that stamp is missing or belongs to another manifest or input root; otherwise they just check existence and size.
* `preflight`: one candidate (`--candidates`, default the first configured), N32, `--per-ring 1`, in its own campaign
  folder `<output>/preflight` (own manifest and identity; it never mixes with the real outputs). Bounded by a 10 min
  timeout; prints wall time, peak RSS, dispatch-check counts and failures, and exits non-zero on any problem.
* `run`: one `run.py` subprocess per grid x candidate, `J` at a time. A failed task does not stop the others; the exit
  code is non-zero if any failed. Ctrl-C terminates the children.
* `validate`: every expected file exists (`results`, `receipt`, `log` per candidate and grid), identity and
  (n, inner_support, per_ring) match the manifest, `failures` is empty, and the dispatch check has 0 violations
  (and equal row keys). Writes `validation.json`; exit 1 on any problem.
* `analyze`: `analyze.py` per candidate (orders, C1/C0 ratios, rebound flags, row statistics) and `compare_candidates`
  over all candidates (u-band tables, pooled-inner ratios). Needs all three grids present and identity-consistent.

## Outputs (all under `OUT`)

```
campaign_manifest.json     git commit, dirty flag (and dirty source files), configuration, candidates, per-ring,
                           input-manifest sha256, localized-sidecar sha256, source hashes, identity
inputs_verified.json       input hash stamp
localized_sidecar.json     continuum sidecar under the input root
<cand>/results_N{n}.json   per-group metrics for C0 and the candidate, dispatch_check, failures   (~2 MB each)
<cand>/receipt_N{n}.json   timings, peak RSS, owner count, closure sizes, identity               (~1 KB each)
<cand>/run_N{n}.log        stdout/stderr of run.py
validation.json            from validate
summary/<cand>_summary.json, summary/<cand>_analyze.txt   from analyze (per candidate; summary JSON ~3 MB)
summary/comparison.json, summary/comparison.txt           candidate comparison over all candidates
preflight/                 the preflight campaign folder
```

## Identity, refusal and resume

The campaign identity is the digest of the configuration, the candidates, per-ring, the `input_manifest.json` hash, the
localized-sidecar hash (which depends on the input root) and the hashes of the sources this evaluation depends on
(`campaign.SOURCE_FILES`). A folder whose `campaign_manifest.json` has a different identity (another per-ring or
candidate set, another input root, changed source) is refused: use a new `OUT`.

`run` is resumable: a (candidate, grid) is skipped when its `results_N{n}.json` and `receipt_N{n}.json` exist with the
campaign identity, the right n, inner_support and per-ring, and no failures. Anything else is deleted and rerun. Both
files are written atomically, the receipt last. Re-issue the same `run` command after an interruption. A run holds an
exclusive lock on `OUT/.campaign.lock`, so two `run`s cannot share a folder.

## Return archive

Return the results and the summary only: `campaign_manifest.json`, `validation.json`, `<cand>/results_N*.json`,
`<cand>/receipt_N*.json`, `<cand>/run_N*.log` and `summary/` (about 30 MB in total). Do not return
`localized_sidecar.json`, the `preflight/` folder or any input file.
