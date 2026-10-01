# P08 step 5: local solver studies on the exported P07 operators

`solver_study.py` consumes the remote export (`EXPORT/N{n}/p07_dirichlet.npz`, `p07_neumann.npz`, `owner_map.npz`;
see `drbx.native.fci_perpendicular_p07_sparse`) together with the frozen P07N campaign
(`work/p07n_field_derived_274e93e9_.../N{n}.owner_values.npz` and `N{n}.global.npz`) and studies the discrete
operator `A` (per-volume action `A u + B g`) on the five campaign fields. Everything is local, one process, CPU.

## What is computed (per grid)

* **Matrix diagnostics** (both kinds): size, nnz, nnz/row, asymmetry `||MA-(MA)^T||_F/||MA+(MA)^T||_F`
  (`M = diag(owner_volume)`), `max|A 1| / max abs row sum`, and the extreme eigenvalues of the generalised problem
  `sym(MA) x = lam M x` (smallest `k` by shift-invert, largest by Lanczos; dense for tiny matrices). The Dirichlet
  kind should be positive definite; the Neumann kind positive semi-definite with `lam_min ~ 0`. Only eigenvalues
  nearest zero are inspected, so a far-negative eigenvalue would not be seen. Failures are recorded, not raised.
* **Consistency**: `max|A phi_bar + B g - frozen|` per field (Dirichlet vs frozen `D`, Neumann vs frozen `N`),
  absolute and relative to `max|frozen|`. Expected ~1e-6 relative at worst (the operators agree with the frozen ones
  to ~1e-6), D and N to ~3e-9 absolute.
* **Dirichlet solves** `A phi = rhs - B g` (direct SuperLU, factored once; FGMRES with no / Jacobi preconditioner via
  `solve_p07_dirichlet_jit`, M-weighted inner product), error `e = phi - phi_bar` as volume-weighted L2, max,
  L2 relative to `L2(phi_bar)`, and L2 per region; iterations, convergence, wall time, residual, and the FGMRES-vs-direct
  difference. Right-hand-side sets:
  * `D` -- the frozen Dirichlet action of `phi_bar`: pure linear-solve check, must return `phi_bar`;
  * `O_q3` -- exact face-flux owner average of the continuum operator (the natural cell-average rhs): the
    discretisation error of the solve against exact cell averages, `e_O = A^-1 (O_q3 - D)`;
  * `R` -- continuum operator at the raw-volume midpoint: the headline error, including the O-R reference mismatch,
    `e_R = e_O + A^-1 (R - O_q3)`. The decomposition table reports `A^-1 (R - O_q3)` (direct solve, zero boundary
    term) and the defect of the identity (rounding level).
* **Neumann report** (`A 1 = 0`): right null residual; the bordered matrix `K = [[A, 1], [m^T, 0]]`, `m = V/sum V`
  (sparse LU; a failure or a huge condition estimate means `ker A != span{1}`) with a 1-norm condition estimate
  (`onenormest` of `K` and `K^-1`); the left null vector `l` (`l^T A = 0`, `sum l = 1`, from `K^T [l; mu] = [0; 1]`,
  `mu = 0` identically) compared with `w = V/sum V` globally and per region (`l = w` iff the `M A` rows sum to zero, i.e.
  the owner action is exactly conservative; wall faces whose flux enters only one owner break it); and for the rhs sets
  `N` (consistency: `lambda ~ 0`, `phi = phi_bar`), `O_q3`, `R` the augmented solve
  `K [phi; lambda] = [rhs - B g; m^T phi_bar]` (gauge: volume mean of `phi_bar`), with `lambda`, `lambda` relative to the
  volume-weighted RMS of the rhs, the solvability defect `l^T (rhs - B g) / (l^T 1)` (equals `lambda`) and the `phi` error.

Regions are the frozen `region_*` masks (prefix dropped) plus `interior = ~(boundary | physical_wall)`.

## Wall data

Dirichlet value / tangential-gradient and Neumann normal data at the export's point tables come from the campaign
field adapter (`p_shared.campaign_fields.P07NAdapter`, environment `curvature=autodiff, face_quadrature=q2,
inner_support=fixed_radius`; ~20 s per grid to build). They are cached in `OUT/N{n}/boundary_data.npz` and reused when
the sha256 of both point tables match.

## Running

From `DRBX/scripts` (conda env `drb`; the module sets `JAX_PLATFORMS=cpu`, x64 and single-thread BLAS itself):

```bash
python -m p08_step5_local.solver_study --export EXPORT --out OUT --grids 32 48 \
    [--frozen DIR] [--skip-neumann] [--solvers direct none jacobi] [--rtol 1e-10] \
    [--restart 50] [--max-restarts 20] [--k-eigs 6] [--batch 8192] [--input-root DIR] [--sidecar FILE]
```

Without `"direct"` in `--solvers` the direct reference, the FGMRES-vs-direct differences and the decomposition are
skipped. The harness refuses to run if the export's `owner_volume` (or owner count) differs from the frozen `volume`
(rtol 1e-12).

## Outputs

* `OUT/N{n}/results.json` -- all numbers (JSON-safe; non-finite values become `null`);
* `OUT/N{n}/boundary_data.npz` -- cached wall data; `OUT/N{n}/neumann_left_null.npz` -- `ell` and `w`;
* `OUT/report_N<grids>.md` -- markdown tables of the grids of that invocation (diagnostics, consistency, Dirichlet errors
  and FGMRES iterations / times, reference mismatch, Neumann null / left-null / lambda tables).

Tests: `tests/test_p08_step5_solver_study.py` (synthetic, seconds).
