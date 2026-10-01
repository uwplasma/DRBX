## Preconditioner candidates for the P07 Dirichlet phi solve

Code: `preconditioners.py` (builders), `bench_preconditioners.py` (end-to-end benchmark),
`../../tests/test_p08_step5_preconditioners.py` (fast synthetic tests). Every builder takes
`(op, *, owner_plane=None, **params)` and returns a `Preconditioner` (`name`, `apply`, `setup_seconds`, `nbytes`,
`jax_native`, `info`). `apply` maps `r -> M^{-1} r` on float64 vectors and is passed as the static `preconditioner`
argument of `solve_p07_dirichlet_jit`. All candidates are fixed linear operators, so plain right preconditioning is
valid (the solvax flexible GMRES tolerates them too). JAX-native candidates embed their data in the compiled program as
constants (compile time and memory grow roughly linearly with their size; this lands in the warm-up column; `plane_jax`/`plane_jax32` instead pass their data as a jit argument, see below).

| registry name | builder and parameters | notes |
|---|---|---|
| `jacobi` | `jacobi` | `r / diag(A)`; the speedup reference |
| `cheb4`, `cheb8` | `chebyshev(degree=4 / 8, lower_ratio=30, power_iters=30)` | Chebyshev iteration for `D^-1 A z = D^-1 r`, `z0 = 0`, on `[lam_max/30, 1.1 lam_max]`, `lam_max` by power iteration; `degree - 1` BCSR matvecs per application; `degree = 1` is a scaled Jacobi |
| `plane` | `plane_block` | exact SuperLU of every eta-plane diagonal block `A[k, k]` (host, `pure_callback`); needs the owner-to-plane index |
| `sa_jac` | `smoothed_aggregation(smoother="jacobi")` | smoothed-aggregation AMG V-cycle, defaults `strength=0.08, max_coarse=1500, presmooth=postsmooth=1, prolongator_smoothing=True, max_levels=10`; damped Jacobi `omega = 4/(3 rho)` per level |
| `sa_cheb` | `smoothed_aggregation(smoother="chebyshev")` | same hierarchy, Chebyshev smoother `cheb_degree=2` on `[rho/10, 1.1 rho]` per level |
| `sa_jac_vs`, `sa_cheb_vs` | as above with `volume_scaling=True` | hierarchy built for `M^{1/2} A M^{-1/2}` (M-symmetrised), cycle wrapped as `M^{-1/2} cycle(M^{1/2} r)`; recommended to try because `A = M^-1 K` with `K` nearly symmetric, so plain Galerkin `P^T A P` is not the natural coarse operator when the volumes vary |
| `sa_jac_f`, `sa_cheb_f` | as above with `filter_smoothing=True` | prolongator smoothed with `A` filtered to its strong pattern (weak entries lumped to the diagonal): sparser `P` and coarse operators (lower operator complexity, cheaper setup) |
| `ilu` | `ilu(drop_tol=1e-4, fill_factor=10)` | SuperLU incomplete LU, host sequential triangular solves; reference only |
| `plane_jax` | `plane_ring_block` | same exact solve as `plane`, JAX-native (no callback): banded block LU over radial super-rings, see below |
| `plane_jax32` | `plane_ring_block(factor_dtype="float32")` | as `plane_jax` with the `L`, `U`, `D^-1` blocks stored (and both scans run) in float32, `r` cast in and `z` cast back to float64; half the storage and memory traffic |
| `bgs_fwd` | `plane_bgs_forward` | one sequential forward block Gauss-Seidel sweep over the planes `k = 0..n-1` (plane solves of `plane_jax`); iteration-count reference |
| `bgs_mc4` | `plane_bgs_multicolor(colors=4)` | multicolor block Gauss-Seidel, color `k mod 4`, one forward sweep over the colors |
| `bgs_mc4_sym` | `plane_bgs_multicolor(colors=4, symmetric=True)` | forward and backward color sweep |
| `bgs_mc4_host` | `plane_bgs_multicolor(colors=4, plane_solver="host")` | as `bgs_mc4` with the SuperLU/`pure_callback` plane solves, for comparison |

The `_vs` and `_f` variants are optional extensions (not in the default `--candidates`); add more variants by adding
entries to `REGISTRY` in `bench_preconditioners.py`. `info["operator_complexity"]`, `["grid_complexity"]`, per-level
`(n, nnz, nnz/row)` and the setup breakdown of the AMG candidates are written to the json and summarised in the md.

### Ring/plane block solvers (`plane_jax`, `bgs_*`)

These take, besides `owner_plane`, the radial index `owner_ring` and an angular key `owner_theta` of every owner; the
benchmark derives them from `owner_map.npz` (`owner_geometry_index(raw_to_owner, n, n_owners)`: `ring = raw // n^2`,
`plane = raw % n`, `theta = ` smallest raw `j = (raw // n) % n` of the owner; the order of the key within a ring and plane
is the angular order, everything else is data driven).

**`plane_ring_block` (`plane_jax`).** Super-rings: within a plane the rings are merged greedily in radial order while the merged
owner count stays `<= B` (`B` = largest ring in one plane = `n`, or `max_block`); a ring never splits. All planes share the
partition (computed from the per-ring maximum over planes; a plane with fewer owners in a ring is padded with identity
diagonal and no coupling). The block bandwidth `w` is the largest super-ring distance of any in-plane coupling, computed from
the data (`info["w"]`, `info["ring_reach"]` per ring, `info["super_rings"]`). Layout: unknowns of a plane ordered by
(super-ring, ring, theta position), padded to `S x B`; the gather index `idx[S, planes, B]` (padding reads 0) and the
scatter map `pos[n]` are precomputed. Factor form: block LDU without pivoting (banded block Gaussian elimination, all
planes at once with batched matmuls, in place, fill stays in the band): multipliers `L_{r,s} = A~_{r,s} D_s^-1`
(`S x w` blocks), upper blocks `U_{s,c}` (`S x w`), and the explicit inverses `D_s^-1` of the Schur-complement diagonal
blocks (`S` blocks); `ValueError` if a diagonal block has a 1-norm condition number above `cond_max` (default `1e12`;
the largest one is reported in `info["max_block_cond1"]`). Apply: gather, forward `lax.scan` over super-rings
(`y_s = b_s - sum_j L_{s,j} y_{s-w+j}`, carry = last `w` solutions), backward scan
(`x_s = D_s^-1 (y_s - sum_e U_{s,e} x_{s+1+e})`), scatter; matmuls only, batched over the planes.

Storage: `bytes = 8 S P (2 w + 1) B^2 + 4 (S P B + n_owners)` (`P` planes, float64 blocks + two int32 index arrays;
`info["storage_bytes"]`). From the real `owner_map.npz` files: N32 (25,376 owners, `B = 32`) `S = 25`; N48 (86,016, `B = 48`)
`S = 38`; N64 (202,304, `B = 64`) `S = 50` (padded fraction 0.9 / 1.8 / 1.2 %). That gives, for `w = 2` / `w = 3`:
N32 0.033 / 0.046 GB, N48 0.169 / 0.236 GB, N64 0.526 / 0.736 GB (the measured `w` is in the benchmark json; a wall ring
reaching 3 rings would force `w = 3`). Setup flops are about `2 w^2 B^3 S P` (N64, `w = 3`: ~4e10), apply reads the whole
store once per forward and backward scan.

**Block Gauss-Seidel (`bgs_*`).** With `P` = exact plane-block solve (block-diagonal `D_P^-1`), `plane_bgs_multicolor` colors the
planes `c = k mod colors` (`colors = 4` needs an inter-plane reach `<= 2` and `n % 4 == 0`; both are verified from the
matrix and a `ValueError` is raised otherwise) and for `c = 0..colors-1` does `z|_c += A_cc^-1 (r - A z)|_c` from `z = 0`
(the plane solves of one color are batched over its planes; `(A z)|_c` uses a precomputed row-subset BCSR `A_c`, color 0 of
the forward sweep needs none, so one sweep costs about one matvec plus the plane solves; each color holds its own factor
arrays, so the storage is that of `plane_jax` plus the BCSR copy of `A`). `symmetric=True` adds the backward colors
`colors-2..0` (the repeated last color would be an exactly zero update, so it is skipped). `plane_bgs_forward` is the same
with one plane per color (n sequential plane solves, unrolled in the jitted program: compile-heavy; iteration-count
reference only). `plane_solver="host"` uses the SuperLU plane solves of `plane_block` (through `pure_callback`) instead. All
are fixed linear operators (`z` starts at 0), valid for right-preconditioned FGMRES.

**`factor_dtype`.** The banded block LU is always computed in float64 on the host; `factor_dtype="float32"` casts the stored
blocks afterwards. The apply casts the gathered `r` to `factor_dtype`, runs both scans in it and returns float64 (relative
error of `M^-1 r` ~1e-6 on the synthetic problem; as an FGMRES preconditioner it needs at most about as many iterations).
Storage becomes `4 S P (2 w + 1) B^2 + 4 (S P B + n_owners)`, i.e. about half. `w_per_superring` (per-super-ring bandwidth to
skip structurally zero band blocks) was **not** implemented: the lower profile is per block row but the upper profile is per
block column, so it would need several scans over segments with different `w` and a carry sliced per segment; that is not a
small change. The wall-ring reach would still inflate `w` for the bulk by one block (~28 % of the storage).

### Preconditioner data as a jit argument

`Preconditioner` has optional `data` (a pytree of arrays) and `apply_with(data, r)` (`apply(r) == apply_with(data, r)`); the
closure `apply` is kept. Only `plane_jax` / `plane_jax32` provide them so far (the other candidates fall back to the
closure `apply`). `preconditioners.make_solver(system, prec, config)` returns `run(rhs, boundary_term=None, x0=None) ->
(x, info)`, a jitted FGMRES solve (same solve as `solve_p07_dirichlet`) in which the system arrays *and* `prec.data` are
jit arguments, so the factors (736 MB at N64) are not embedded as program constants and the warm-up column of the benchmark
measures the compile of the program only (the benchmark and `bench_warm_start` use it for every candidate). A test checks
that the closure apply's jaxpr carries the factors as constants and `apply_with`'s does not; the first-call time on the tiny
synthetic problem is dominated by compile either way (no timing assertions).

### Warm-start proxy (`bench_warm_start.py`)

Time-stepping proxy on the real data: the trajectory `phi_s = phi_bar[:,0] cos(w s) + phi_bar[:,1] sin(w s)`,
`B g_s` = the same combination of the cached boundary-source columns, `rhs_s = A phi_s + B g_s` (exact solution `phi_s`),
`w` chosen so that the mean M-weighted relative change per step is `delta` (`--deltas`, default 1e-2 1e-3 1e-4). Each
(preconditioner, rtol, delta) is solved cold (`x0 = 0`) and warm (`x0` = previous solution; step 0 cold in both), averages
over steps 1..S-1. **The solver's convergence test is `||rhs_eff - A x||_M <= rtol ||rhs_eff||_M`, relative to the norm of
the right-hand side, not of the initial residual**, so a warm start saves iterations only through its smaller initial
residual (about `delta`), and more so for looser `rtol` and smaller `delta`. Outputs `OUT/warm_N{n}.json` (per-step iterations,
wall times, true M-weighted relative errors, convergence flags) and `.md` (prec, rtol, delta, cold/warm its per step, cold/warm
s per step, warm speedup, max error).

```bash
python -m p08_step5_local.bench_warm_start --export EXPORT --study-out STUDY_OUT --grids 32 \
  --preconds jacobi plane_jax plane_jax32 --steps 20 --deltas 1e-2 1e-3 1e-4 --rtols 1e-10 1e-8 --out OUT
```

### Timing protocol (per grid, per candidate)

1. **setup**: host wall time of the builder (includes conversion to JAX arrays).
2. **warm-up**: first solve (rhs `D`, field 0) of the jitted solve (`make_solver`: preconditioner data as an argument where available) with this preconditioner: JIT compilation plus one solve.
3. **solve**: mean wall time of `--repeats` timed solves of rhs `D` field 0 (known solution `phi_bar`; the true M-weighted
   relative error `||phi - phi_bar||_M / ||phi_bar||_M` is recorded) and of rhs `O_q3` field 1, each synchronised
   with `block_until_ready`; the relative residual is recomputed independently on the host with scipy.
4. `time_1 = setup + warm-up + solve`, `time_100 = setup + warm-up + 100 * solve` (literal formulas; the warm-up
   already contains one real solve, so they are very slightly conservative). `speedup_k = time_k(jacobi) / time_k(candidate)`.
   Peak-RSS high-water mark and its growth over each candidate are in the json.

Jacobi is always run first, jit caches are cleared after each candidate, and a failing candidate is recorded
(`error`) without stopping the run. Boundary data is read from `STUDY_OUT/N{n}/boundary_data.npz` (written by
`solver_study`); it must match the export's point tables.

### Commands (from `DRBX/scripts`, single process, CPU, x64, one thread)

```bash
# unit tests (seconds)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m pytest -q ../tests/test_p08_step5_preconditioners.py

# benchmark (heavy on the real exports; run when the machine is free)
python -m p08_step5_local.bench_preconditioners --export EXPORT --study-out STUDY_OUT --grids 32 48 \
  --candidates jacobi cheb4 cheb8 plane sa_jac sa_cheb ilu --rtol 1e-10 --restart 50 --max-restarts 40 \
  --repeats 2 --out OUT
# results: OUT/bench_N32.json, OUT/bench_N32.md, ...
```
