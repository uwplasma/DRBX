# P08 step 5: sparse export of the P07 owner action

Computation only (no commits, no pushes, no analysis). CPU, one process.

## Purpose

The step-4 campaign left full-grid row artifacts (N32 / N48 / N64) on the remote. This campaign lowers **only the
P07 part** of each artifact, exports the P07 owner action as scipy sparse matrices with
`drbx.native.fci_perpendicular_p07_sparse.export_p07_sparse` (kinds `dirichlet` and `neumann`), gates each export
against `p07_action` on the full grid, and writes small files to download for local linear-solver studies.

## Inputs

- `--step4-campaign DIR`: the step-4 output's `campaign/` folder with `artifact/`, `localized_sidecar.json`,
  `validation.json` and `campaign_manifest.json`. Every requested grid must have `smoke_pass` and `preflight_pass`
  true in step 4's `validation.json`, and its artifact must carry the pinned options
  (`curvature=autodiff`, `face_quadrature=q2`, `inner_support=fixed_radius`; pinned in `configuration.json`).
- `--input-root DIR`: the same immutable input root the step-4 run used.
- `--output DIR`: a new folder (the campaign identity is recorded; a changed identity refuses to reuse it).

## Commands (from `DRBX/scripts`)

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m p08_step5_export.campaign verify-inputs --step4-campaign S4/campaign --input-root IN --output OUT
python -m p08_step5_export.campaign run           --step4-campaign S4/campaign --input-root IN --output OUT [--grids 32 48 64]
python -m p08_step5_export.campaign validate      --step4-campaign S4/campaign --input-root IN --output OUT [--grids ...]
```

`run` repeats `verify-inputs`, skips a grid whose `N{n}/export_receipt.json` has this identity and passed, and ends
with `validate`. If a gate fails the receipt is still written (`pass: false`), the remaining grids are not run and the
exit status is nonzero. Re-running `run` recomputes a failed grid.

## Outputs (`OUT/`)

- `provenance/inputs.json`: campaign identity, step-4 identity, per-grid artifact identity hash, paths, JAX info.
- `N{n}/p07_dirichlet.npz`, `N{n}/p07_neumann.npz`: `load_p07_sparse` files (blocks `matrix`, `dirichlet_value`,
  `dirichlet_tangential`, `neumann_normal`, plus owner volume and boundary points), identity-stamped.
- `N{n}/owner_map.npz`: `owner_volume`, `raw_to_owner`, `centers_u`, `centers_theta`, `centers_eta`.
- `N{n}/export_receipt.json`: gates (full action and linear part: max abs diff, scale, pass), nnz and shapes per
  block and kind, file bytes and sha256, seconds per phase, process peak RSS.
- `summary/step5_export_summary.json`, `validation.json`, `last_exit.json`, `invocations/`.

Gate: `max|apply_p07_sparse - p07_action| <= 1e-11 * max|p07_action|` with random fields (2 columns, seed 0) and
random boundary data; the linear part is gated against zero boundary data.

## Resources

Single process (no worker pool). The P07 rows are 1.25 GiB in the N64 artifact; the step-4 N64 full replay peaked at
41 GiB, but only P07 is lowered here (no cells/faces, no device placement), so the peak is far lower. The
environment build (geometry, census) is the other main cost. Download `OUT/N*/` (the `.npz` files and receipts).
