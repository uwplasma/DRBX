# Full-domain N32 Q modes: one remote CPU trip

Install this package as `DRBX/scripts/q_full_domain_modes/` in the reviewed
checkout. The remote worker uses its `run drbx on perlmutter` setup skill to
pin the approved revision, choose the environment/allocation and size
WORKERS, WORKER_GIB, HOST_GIB, STAGE_SECONDS and KRYLOV_SECONDS. A GPU
allocation is allowed, but every calculation uses CPUs and GPUs stay hidden.
No runtime dependency under `work/` exists. Set DRBX_SOURCE_REVISION to the
pinned commit for the source receipt; the campaign performs no git operations.

Run once from `DRBX/`, without an inspection stop between pilot and solve:

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m scripts.q_full_domain_modes.run_all --run "$RUN" \
  --canonical-root "$CANONICAL" --workers "$WORKERS" \
  --worker-gib "$WORKER_GIB" --host-gib "$HOST_GIB" \
  --stage-seconds "$STAGE_SECONDS" --krylov-seconds "$KRYLOV_SECONDS" \
  --through validate --remote-ok
```

Order: source, preflight, prepare, pilot, solve, compare-derivatives, validate.
`--through STAGE` remains available; `--dry-run` prints the ordered commands.
`--stage-seconds` applies separately to each stage. The setup skill must allow
walltime for the sum of all seven stage budgets, plus startup, hash verification
and output/controller overhead. Pilot results automatically include diagnostic
RK4 separation for growth rates 1 and 10, rho, warm matvec time and estimated
restart cost. They do not change the method or dt=0.3/rho.

CPU admission is min(requested workers, inherited CPU affinity, floor((host
budget - reserve)/worker budget)), with reserve=max(1 GiB, 5% of host budget).
Each admitted child has its own worker RSS cap and time guard. Slots carry
explicit core indices; Linux pins each child to its own admitted core. macOS
cannot pin affinity and records that limitation. BLAS/JAX threading is restricted.
Pilot and validation run cases concurrently. Solve runs independent case/seed
jobs concurrently, then queues one analyses job per case after its seeds finish;
analyses can overlap other cases. Derivative comparisons remain sequential.
All outputs have one writer. An execution error waits for the other launched
jobs and stops the chain without a successful stage receipt. Only this runner's
own children can be signalled by its guards.

Each spectrum worker loops warm Ritz-vector restarts until convergence, the
search budget or `--max-attempts` is reached. Default max attempts is unlimited
within the budget; an explicit limit counts attempts in this invocation. Each
attempt has immutable JSON/NPZ artifacts and an atomic current-checkpoint
pointer. Exact Krylov basis resume is unavailable in the unchanged P driver.
Solve reserves the final 20% of its stage budget for analysis; an unfinished
analysis gets a clearly incomplete budget summary. Missing/unconverged seeds
are listed. Budget exhaustion continues the chain to derivative comparison and
validation; execution errors stop it. Partial validation rechecks every saved
mode and reports missing seeds, rather than claiming convergence. Same RUN and
source/input/science/solver identity resume checked payloads. Resource and time
caps may change on resume. Validation reruns when solver checkpoints change.

CANONICAL uses the existing Q08/Q09 workspace-root layout:
`geometry_artifacts/rlp_convergence_32_48_64_20260917/32x32x32/` with
`base_geometry.npz` and `rlp_topology.npz`, plus the root-level metric NPZ and
MAKEGRID NC named in `input_manifest.json`. Controllers SHA-check these
immutable inputs; workers reuse the checked receipt. Retained RK4 endpoints,
frozen support choices, complete owner chunk plan, and the bounded seven-owner
bank/geometry ship in `inputs/`. Frozen compact-C3 producer sources are in
`geometry_model/`; `eigcore.py` is unchanged, with original SHA recorded.
All generated files, caches, receipts and locks are below RUN.

Science is unchanged: Q09 smooth MMS at t=0, physical-volume observations,
fixed prescribed phi and exact base BC, homogeneous perturbations, no forcing,
and Q09 frozen-projector JVP. H repeats owner volumes without another Jacobian.
Abscissa uses H^-1/2 S H^-1/2 and saves its Lanczos direction for validation.
The true directional derivative comparison uses refined central FD of the
original nonlinear RHS with projectors recomputed and reports its uncertainty.
Negative candidates cannot certify an exhaustive absence of growing modes.
Use portable campaign RSS receipts; unchanged eigcore log RSS uses Darwin units.

Bounded local controls: prepare with `--chunks` or `--max-chunks` selecting fewer
than all chunks. `--interrupt-after-payload INDEX` fails once before chunk commit
and tests recovery. `--bounded-test` uses only the immutable seven-owner fixture
for pilot/solve/compare/validate, and never loads the full prepared bank.
`--test-kill-job action_operator_bc_seed` kills one scheduler-owned child once;
`--test-first-krylov-seconds` tests a tiny first attempt followed by normal warm
attempts. Both require bounded-test. Their spectra are artificial-closure tooling
evidence only. Full stages require `--remote-ok` and full owner/raw/donor closure.
Preflight checks JVP/FD, VJP adjoint, energy identity, dense/Krylov and dense/Lanczos
agreement for four cases, plus checkpoint reloads and synthetic Krylov restarts.
Return RUN receipts/results/checkpoints/logs (omit compilation caches), pinned
source and manifests for parent analysis.
