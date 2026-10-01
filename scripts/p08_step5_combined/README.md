# P08 step 5.3: static Dirichlet-phi combined full-grid MMS

Computation only (CPU). Roadmap: P08 step 5.3. Per grid (N32, N48, N64) this campaign measures the combined perpendicular
RHS (`perpendicular_rhs`: bracket + curvature + diffusion of n, Te, Ti, omega) on the FULL grid with the final operator
and two arms per frozen P06N variant, against re-frozen owner references. Nothing is accepted or rejected by the
campaign: the solver gates and finiteness are recorded as pass/fail, the headline order criterion is recorded as
informational, and **the user decides acceptance**.

## Scientific contract (`configuration.json`; also checked against literals in `campaign.py`, hashed into the identity)

| item | value |
|---|---|
| schema | `drbx.p08-step5-combined-dirichlet-v1` |
| resolutions | 32, 48, 64 |
| operator options | `curvature=autodiff`, `face_quadrature=q2`, `inner_support=fixed_radius` (a drifted configuration is refused) |
| parameters | `rho_star = 0.05`, `tau = 1`, `D_f = 1e-2` for n, Te, Ti and omega (the step-3 G3.3 values) |
| phi solve | Dirichlet, block-Jacobi eta-plane preconditioner, float32 factors, restart 50, max_restarts 40, **rtol 1e-11** |
| catalogue | frozen P06N variants with Dirichlet phi (below) |

**phi-solve tolerance.** `1e-8` (`PHI_RTOL_DEFAULT`) is the *production* default; `1e-11` is the *measurement* setting of
this campaign (`phi_solve_rtol`; the production value is recorded as `phi_solve_rtol_production_default`). Measured on the
real exports, `rtol = 1e-8` leaves a linear-solve error of about 2e-8 relative, which exceeds the phi-solve discretization
error at N48 / N64 (7e-9 / 2.4e-9 for the smoothest field) and would corrupt the measured orders. The linear-solve error
of every variant is reported per solve through the discrete-consistency gate (a) below (threshold 1e-8, M-weighted).

**Catalogue** (P06N variants, `p06n_catalogue.json`; kinds are the catalogue's, pinned and checked at run time):

| variant | field set | kinds n, Te, Ti, omega, phi | role |
|---|---|---|---|
| `main_phi_dirichlet` | `main` | N, N, N, D, D | main |
| `heldout_phi_dirichlet` | `heldout` | N, N, N, D, D | held-out fields |
| `dirichlet_rich` | `main` | D, D, D, D, D | all Dirichlet |
| `control_constant_dirichlet` | `constant` | D, D, D, D, D | constant control (references vanish identically) |

**Deferred (recorded in the configuration):** the Neumann-phi variants (`main_phi_neumann`, `heldout_phi_neumann`,
`control_constant_neumann`) are deferred until after this step (user decision, 1 October 2026). The low-degree /
transverse-wave controls belong to the step-6 transverse-wave check, not to this step.

## The two arms

* **prescribed**: `perpendicular_rhs` with the exact owner-averaged `phi_bar` (the frozen P06N `owner_values` column; the
  exact average of every catalogue column, path key `p05n_p06n_upwind`).
* **solved**: production polarization is `omega = div(P_perp grad(phi + tau Ti))`, so with `psi = phi + tau Ti` the
  campaign solves the P07 Dirichlet problem `A psi_h = O(psi) - B g` where
  * `O(psi)` is the owner average of the POSITIVE operator `-div(P_perp grad psi)` built exactly like the P07N `O_q3`
    reference (exact gradients of psi on the q3 face nodes of every P07 face, scattered lower -, upper +, divided by the owner
    volume): `references.psi_positive_operator` = `diffusion_reference(..., positive_operator=True)` of psi;
  * `g` is psi's exact wall trace at the plan's Dirichlet points (value and tangential gradient), taken from the same
    five-column P06N state data the arms use: `trace(phi) + tau trace(Ti)`;
  * the solve is cold-started (iterations and seconds recorded per variant);

  then `phi_h = psi_h - tau * Ti_bar` and `perpendicular_rhs` runs with `phi_h`.

omega is the catalogue's own independent field: the arms test the phi-solve-in-the-RHS pipeline; omega-phi consistency
is a P09 time-evolution concern.

## Stages per grid (`run` does all of them for each grid, 32 -> 48 -> 64; every stage is resumable)

1. **references** (`references.py`; `--workers` process pool): per owner chunk (`owner_chunk_size`, via
   `runner.chunk_units` and `runner.run_stage`; each worker builds its own `build_environment(..., **options)` once in the
   initializer): for every distinct field set (`main`, `heldout`, `constant`) `reference_rhs` over ALL owners returning, per
   field and term, owner arrays of `poisson_bracket` (P05N `raw_R`), `curvature` / `curvature_material` /
   `curvature_remainder` (P06N evolution-weighted `raw_R`), `perpendicular_diffusion` (P07N `O_q3` exact face flux,
   `D_f div(P grad f)`) and `total`; plus `psi__<fs>__O` (positive `O(psi)`) and `psi__<fs>__R_mid`, the diffusion midpoint
   reference `-div(P grad psi)` at the raw midpoints owner-projected by raw volume over owner volume, with the AUTODIFF
   divergence (`curvature_reference.perpendicular_geometry(..., method="autodiff")`) and the exact gradient / Hessian of
   psi (numerator `div . grad psi + tensor : Hess psi`, over `|J|`). Merged into `N{n}/references/references.npz` +
   `manifest.json` (identity, sha256); `N{n}/context.npz` holds the owner volumes and the P06N region masks. These are the
   new frozen references for the final operator.
2. **jax** (`jaxstage.py`; one process): `build_environment`, the step-4 artifact (`load_artifact`, options and identity
   checks), the FULL plan (`lower_plan`: cells, faces, P07), the phi solver once
   (`build_phi_solver(plan, raw_to_owner=env.t.ro, n=n, rtol=1e-11, ...)`), then per variant (a `runner` unit, so a crash
   resumes at the next variant): five-column state, kinds and boundary data (see below), prescribed arm, solved arm and the
   solver gates. Arrays of both arms (every field, every term) and the potentials are checkpointed.
3. **reduce** (`reduction.py`; pure NumPy): `N{n}/results.npz` + `N{n}/summary.json`.

Boundary data: `boundary_data_from_callables(plan, dirichlet, normal)` with the exact state of the variant restricted to its
five columns (`n, Te, Ti, omega, phi`), the same data `JaxOwnerClosure` builds from the whole catalogue (verified equal on
the N32 closure by the slow test). `JaxOwnerClosure` itself is not used for the full plan: it evaluates all ~20 catalogue
columns at every wall point.

## Solver gates (recorded pass/fail per variant and grid)

* **(a) discrete consistency**: `rhs_c = A psi_bar + B g` (`A`, `B` from `solver.op`, `psi_bar = phi_bar + tau Ti_bar`) is
  solved through the same solver and must return `psi_bar` to a relative M-weighted L2 error <= 1e-8
  (`consistency_tolerance`). This is the solver's linear-solve error, reported per variant.
* **(b) `||O(psi) - (A psi_bar + B g)|| / ||O(psi)||`**: the discrete operator's N - O for psi (report only; also the sign
  check of `O`: a sign error would give ~2). Expected small (~4e-3 at the N32 closure owners).
* **(c) every solve converged** (consistency solve and the psi solve) and **every array finite**.

`grid_pass` = (a) and (c) for every variant and everything finite. The headline order criterion (global relative-L2 order
of the `total` term >= 1.8 on both intervals, every non-constant variant and field, prescribed and solved arm against the
reference) is recorded in `validation.json` and the analysis, marked `informational` / `user_decides`.

## Reduction (what is stored)

Per variant, field (density, Te, Ti, vorticity), term (`poisson_bracket`, `curvature`, `perpendicular_diffusion`, `total`,
and the `curvature_material` / `curvature_remainder` details), region (`global` and every P06N region mask:
`physical_wall`, `transverse_last_two_layers`, `transition`, `aggregate`, `ordinary`, `interior`) and comparison
(`presc_vs_ref` = `N_presc - R`, `solved_vs_ref` = `N_solved - R`, `solved_vs_presc` = `N_solved - N_presc`): owner-volume-weighted
L2 and max of the difference, absolute and relative to the reference's own L2 / max. Degenerate references follow the
step-3 design rule (section 8.2): a term whose reference L2 is below `1e-8` of the largest production-term reference L2
of the same variant, field and region uses that largest term as scale (`degenerate` flag); when even that is zero (the
constant control) the relative errors are undefined (NaN in `results.npz`, `null` in JSON) and only absolute errors are
meaningful. Also: the potential error `phi_h - phi_bar` (relative to `phi_bar`, per region), the psi solve (iterations,
seconds, residual), and the psi diffusion triple `O - R_mid`, `N - O`, `N - R_mid` (per region).

`analyze` computes, for every row, the values on the available grids and the observed orders between consecutive grids
(`log(e_coarse/e_fine)/log(n_fine/n_coarse)`) of the relative L2, absolute L2 and max, and writes
`summary/step5_combined_report.md` and `summary/step5_combined_summary.json`.

## Inputs

* `--step4-campaign DIR`: the step-4 output's `campaign/` folder with `artifact/N{n}`, `localized_sidecar.json`,
  `validation.json`, `oracle_manifest.json`, `campaign_manifest.json`. Every requested grid must have `smoke_pass` and
  `preflight_pass` true, and the artifact must carry the pinned options. Nothing is rebuilt.
* `--input-root DIR`: the same immutable input root as step 4 (existence and sizes of the 17 files are checked; their
  hashes are covered by the step-4 identity).
* `--oracle-root DIR` (default: `--input-root`): where the files of the step-4 oracle manifest live. Only the P06N
  `owner_values` files of the requested grids are read (and hash-checked against the step-4 oracle manifest).
* `--output DIR`: a new folder (identity-checked).

The campaign identity is the sha256 over the configuration, this package, the P-path sources it depends on
(`perpendicular_rhs`, the P05/P06/P07 operators, reconstruction state, operator plan, `p07_sparse` / `p07_solve` /
`phi_solver` / `plane_preconditioner`, the reference module, `curvature_reference`, `campaign_fields`,
`replay_support`, ...), the step-4 campaign identity, the localized-sidecar hash and the P06N oracle entries.

## Commands (from `DRBX/scripts`)

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
A=(--step4-campaign "$STEP4/campaign" --input-root "$INPUT_ROOT" --oracle-root "$ORACLE_ROOT" --output "$OUT")

python -m p08_step5_combined.campaign verify-inputs "${A[@]}"
python -m p08_step5_combined.campaign preflight     "${A[@]}"                       # N32, bounded; must pass before run
python -m p08_step5_combined.campaign run           "${A[@]}" --resolutions 32 48 64 --workers 96
python -m p08_step5_combined.campaign validate      "${A[@]}" --resolutions 32 48 64
python -m p08_step5_combined.campaign analyze       "${A[@]}" --resolutions 32 48 64
# recovery, stage by stage (not preflight-gated)
python -m p08_step5_combined.campaign run-stage     "${A[@]}" --stage references --n 64 --workers 96
python -m p08_step5_combined.campaign run-stage     "${A[@]}" --stage jax --n 64
python -m p08_step5_combined.campaign run-stage     "${A[@]}" --stage reduce --n 64
```

`--workers` is the reference-stage pool only (the JAX stage is one process); `--memory-budget-gib B --worker-memory-gib W
--memory-reserve-gib R` cap the pool as in steps 1-4; `--max-tasks-per-worker` recycles workers.

**Preflight** (N32, bounded): `references.reference_chunk` on the 12 owners of `owner_closure.select_owners` for the `main`
field set (no pool), and the JAX stage on the FULL N32 plan for `main_phi_dirichlet` (both arms, solver gates). Because the
full-grid `O(psi)` does not exist yet, the preflight's solved arm solves with `A psi_bar + B g`; the subset reference
`O(psi)` is compared with `A psi_bar + B g` (N - O: a sign / consistency check of the reference). Pass = finite, solver
gates (a)(c), N - O of psi <= 0.1, prescribed-arm sanity at the subset (worst relative L2 of the `total` terms against the
reference, relative to the field's largest term, <= 0.5) and the arm difference <= 1e-5. `run` refuses without a matching
passing preflight; a failed preflight is recomputed on the next invocation.

`run` stops after the first grid whose `grid_pass` is false (its data are checkpointed) so the finer grids are not burned,
runs `validate` and `analyze` on the grids completed, and exits nonzero (`last_exit.json` status `failed_gates`). Every
command appends `invocations/<ns>_<command>.json`, writes `provenance/inputs.json` and `last_exit.json`; progress lines
with timestamps go to stdout (tee into `logs/`). The package refuses a non-CPU backend or missing x64.

## Outputs (`$OUT/`)

`provenance/inputs.json`, `preflight.json`, `preflight/N32_preflight.json`, `N{n}/context.npz`,
`N{n}/references/{references.npz,manifest.json}`, `N{n}/jax_stage.json`, `N{n}/results.npz`, `N{n}/summary.json`,
`validation.json`, `summary/step5_combined_report.md`, `summary/step5_combined_summary.json`, `invocations/`,
`executions/`, `last_exit.json`, and `work/_chunks/N{n}/{references,jax_<variant>}/` (checkpoints; the JAX chunks are the
full per-owner arrays of both arms, ~100 MB per variant at N64: delete `work/_chunks` after `reduce` to save space, keep
it to redo the reduction). Return `summary/`, `validation.json`, `N*/results.npz`, `N*/summary.json`, `N*/references/` and
`preflight*`; leave `work/` out of the return archive.

## Resources

CPU only. The references stage is embarrassingly parallel over owner chunks (each worker holds one environment, a few GiB
at N32 and more at N64); the JAX stage holds the full plan like the step-4 replay (peak ~36 GiB at N64 from step 4's
estimate, plus the sparse P07 operator, the plane-preconditioner factors and the per-variant arrays) and runs one process:
the phi solves (two per variant, up to 50 x 40 GMRES iterations at rtol 1e-11) dominate. Disk: the chunk checkpoints and
merged references are small next to the step-4 artifacts.

## Local tests

`tests/test_p08_step5_combined_campaign.py` (fast, ~3 s): pinned contract and drift refusal, catalogue check, identity,
`verify-inputs` against a synthetic step-4 folder, CLI, reference stage (shape / merge / resume / sha), the psi references,
both arms of the JAX stage fed the right phi, the solver gates enforced, checkpoint resume, preflight verdict, reduction
math, orders and headline criterion, `validate`, `analyze`, and the run flow. `tests/test_p08_step5_combined_campaign_real.py`
(slow, ~35 s): the reference core and the wiring of both arms on the real N32 owner closure with the final options.
