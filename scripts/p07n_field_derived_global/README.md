# P07N field-derived static global Neumann campaign

This tracked computation-only runner prepares a fresh N32/N48/N64 global
static diffusion qualification for a **new** catalogue of physical-normal
Neumann fields. It supersedes nothing: the frozen, failed
`scripts/p07_neumann_global/` (zero-normal manufactured fields needing a
finite-difference wall correction) is preserved unchanged. This package does
not evolve a field, solve a new elliptic problem, or promote a default.

## Field-derived wall data and catalogue

Every field has **nonzero physical-normal** wall data taken from the field
itself: `g_N = a.grad_x f` at the wall, `a = fields.normal(ref,q)` (same
formula as frozen). No finite-difference correction, cutoff or shifted
stencil: all derivatives are analytic. `fields.py` is self-contained
(no import from `work/` or the frozen package). The catalogue is `field_b1`
and held-out `heldout_field_b2` (frozen `base(q,1|2,period)`, verbatim),
`field_e3`/`field_e12` (theta/eta wall-trace fields at theta-gradient
strengths 0.03/0.12, from the design screen at
`work/p07n_compatible_fields_20260927/fields_v2.py`, verbatim), and
`constant` (1, zero derivatives). `core.py` reuses the tracked P07 canonical
topology, structured boundary-family trace elimination and integrated q3
rows unchanged.

## Linearity check

The Neumann row application is linear in (owner values, boundary data), so
`core.face_chunk(...,linearity=True)` also returns, per boundary-family
(1/2/4) face, `N_zero_data` (boundary data zeroed) and `N_zero_owner` (owner
values zeroed); `N` must equal their sum. The preflight face stage runs with
`linearity=True` and records, per grid, `row_linearity_max_abs` and
`row_linearity_pass` (`<= 1e-12*max(1,max|N|)`) — an operational check, not
a scientific gate.

## Inputs, stages and gate (unchanged from frozen)

`input_manifest.json` is an identical copy: the same 17 immutable files.
Source hashes cover this package's own six files plus the same tracked
`p07_combined_global`/`p07_diffusion_global`/`hsx_mms_continuum_reference.py`/
`drbx.geometry` dependencies. `topology`, `preflight`, `run` and `validate`
stages, chunking, resume and the matched-Dirichlet D diagnostic are
unchanged. The scientific gate is unchanged: for each nonconstant field,
global L2 order of `N-R` >= 1.8 on both N32->N48 and N48->N64; held-out
`heldout_field_b2` has no exemption; constant max `|N-R|` <= 1e-8; wall RMS
nonincreasing; bounded reference-uncertainty check. The `N-R` order gate is
the headline traditional MMS report; the acceptance decision is made by the
user after results, by examining `N_minus_O` (closure vs exact face flux)
and `O_minus_R` (exact flux vs midpoint target) separately, and the returned
gate flag is preserved unchanged.

## Ungated accuracy record

`summary.json` additionally reports, **not gated**, `O_minus_R_orders` and
`N_minus_O_orders` per field next to the `N-R` `orders`, each with the
per-grid regional L2 for `N-O`/`O-R` that `stats()` already computes, so the
decomposition reads straight from the file, plus `row_linearity_pass` folded
in from preflight. `configuration.json` predeclares the exact-gradient
screen's `screen_predicted_O_minus_R_orders` (`screen_evidence`:
`work/p07n_compatible_fields_20260927/report.md`, run before freezing) for
comparison; it is evidence, not a gate.

## Commands and recovery

From the DRBX repository root, with a **new unique** `OUTPUT` folder and an
immutable `INPUT_ROOT` containing the 17 mapped files:

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$OUTPUT/cache" DRBX_CACHE_DIR="$OUTPUT/cache/jax" TMPDIR="$OUTPUT/scratch"
python scripts/p07n_field_derived_global/campaign.py verify-inputs --input-root "$INPUT_ROOT" --output "$OUTPUT"
python scripts/p07n_field_derived_global/campaign.py preflight --input-root "$INPUT_ROOT" --output "$OUTPUT" --workers "$CPU_WORKERS" --memory-budget-gib "$MEMORY_BUDGET_GIB" --worker-memory-gib "$WORKER_MEMORY_GIB" --memory-reserve-gib "$MEMORY_RESERVE_GIB"
python scripts/p07n_field_derived_global/campaign.py run --input-root "$INPUT_ROOT" --output "$OUTPUT" --workers "$CPU_WORKERS" --memory-budget-gib "$MEMORY_BUDGET_GIB" --worker-memory-gib "$WORKER_MEMORY_GIB" --memory-reserve-gib "$MEMORY_RESERVE_GIB"
python scripts/p07n_field_derived_global/campaign.py validate --input-root "$INPUT_ROOT" --output "$OUTPUT"
```

Repeat `preflight` or `run` with the same arguments and output folder after
an interruption. `run-stage --stage STAGE` permits operational replay of one
stage; `--max-units` is for a bounded local recovery test only. Keep logs,
caches, scratch, chunks, manifests and receipts inside the single `OUTPUT`
folder.
