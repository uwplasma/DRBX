# HSX/RLP Stage 7 manufactured-solution campaign

The Stage 7 test is compatible with radius-dependent angular RLP. The
continuum solution is evaluated on one fixed smooth 64-cubed HSX metric,
sampled at each raw-cell midpoint, and then volume-restricted to canonical RLP
owners. Numerical owner error and fine-grid RLP representation error are
reported separately.

The canonical entry point is `simulate_hsx_mms.py`. It uses real
resolution-local FCI maps and the production-split configuration:

- support-core FCI pairing;
- characteristic-SAT, energy-absorbing physical-wall closure;
- homogeneous current/phi pair plus the affine characteristic-current lift;
- central operator inflow traces;
- local backward Euler on all physical wall material legs;
- IMEX-SSP222 with forcing evaluated at every actual stage time;
- material-scalar third-order upwind Poisson brackets;
- conservative projected-fine RLP curvature;
- automatic radius-dependent angular RLP; and
- four FCI trace substeps.

The potential inversion uses the same production policy as the attached
64-cubed run and current driver defaults: line-u preconditioning, target
`1e-8`, acceptance `5e-5`, 500 maximum iterations, restart 100, and one
reliable true-residual correction. Evolved MMS trajectories start from the
analytical manufactured phi, giving exactly zero initial-condition error as
required for a classical MMS convergence test. The separate frozen audit
evaluates both analytical phi and reconstructed phi so spatial-operator and
elliptic-reconstruction errors remain distinguishable.

The production-active physical coefficient in this MMS is perpendicular
diffusion: `density_D_perp`, `electron_temperature_D_perp`,
`ion_temperature_D_perp`, `Vi_D_perp`, `Ve_D_perp`, and `vorticity_D_perp` are
all `1e-5`, and the independent continuum reference uses the matching
`perp_diffusion=1e-5` operator.  Parallel diffusion/viscosity and `Ve_nu`
collisions are zero by design and are recorded in `physical_parameters_json`.
Production MMS enables a static axis-regular generalized potential
`psi`, with `phi=-tau*(Ti-1)+psi` and exact `omega=L_perp psi`.  The omega
scalar is evaluated from the independent cached continuum metric
tensor/divergence; its gradient and Hessian are assembled on the structured
`n x n x n` cell-midpoint tensor using nonuniform fourth-order five-point
weights (periodic theta/eta wrapping and one-sided radial end stencils).
Midpoint projection is second-order accurate and intentionally matches the
target order of the production scheme; it avoids the eightfold reference
work and memory of the former `2 x 2 x 2` Gauss rule.
Thus the vorticity Poisson-bracket, parallel-advection, and perpendicular-
diffusion lanes are active and independently sourced.  The zero-omega mode
is retained as the analytic-reference self-test regression only.

The continuum source is independent of the production RHS. Frozen residuals
combine the explicit ARK partition with the selected implicit wall residual,
so they measure the complete semidiscrete `F + G` operator. The output also
contains per-term continuum error ledgers and observed orders, both globally
and for each spatial region.

## Production-fidelity audit

The audit used the launch command and persisted metadata from
`prototype_runs/fci_qhs_64_material_scalar_pb3_characteristic_sat_imex_allwalls_trace4_rhsterms_t015_800steps_26d9e506_20260901`.
All 42 metadata keys shared by that run and the MMS production contract agree
exactly. Removed historical selectors were traced to their current fixed
implementations rather than reintroduced: centered cell-centred FCI, central
operator traces, the homogeneous current/phi pair and affine SAT lift, full
conservative characteristic curvature, and projected-fine RLP behavior.

The audit corrected the trajectory-affecting solver discrepancy: the MMS had
used an 80-iteration/restart-40/no-correction GMRES policy with acceptance
`1e-5`. It now uses the production solver policy described above. Analytical
initial phi is deliberately retained as an MMS condition rather than treated
as production drift. The analyzer requires these settings in
`production_configuration_json`; evolved histories also carry the same flat
metadata rather than an abbreviated configuration.

Differences retained deliberately are the manufactured state and independent
source, the short verification time horizon, float64 history storage, and the
absence of optional per-step RHS-statistics replay. None changes the live
spatial or time-advance operators.

## Local gates completed

Run from the repository root with the repository source tree on `PYTHONPATH`.

```bash
PYTHONPATH=src python3 simulate_hsx_mms.py --self-test --wiring-only
PYTHONPATH=src pytest -q \
  tests/test_simulate_hsx_mms.py \
  tests/test_analyze_hsx_mms.py \
  tests/test_simulate_hsx_blob_production_selectors.py
```

The corrected local frozen smoke is:

```bash
PYTHONPATH=src python3 simulate_hsx_mms.py \
  --resolutions 8 \
  --time 1e-6 --final-time 1e-6 --dt 1e-6 \
  --metric-cache-dir ../.hsx_metric_cache \
  --output ../work/hsx_mms_frozen8_current.npz
```

That final 8-cubed artifact has real maps (56 forward and 56 backward wall
hits), converged phi reconstruction, finite regional and reconstructed-phi
ledgers, and a maximum independent-source pairing error of
`1.02e-15`. Its production vorticity Poisson-bracket, parallel-advection,
parallel-current, curvature, and perpendicular-diffusion lanes are all
nonzero in populated regions. It is a wiring/frozen-residual smoke only, not a
convergence point.

A pre-audit one-step evolved smoke is stored as
`../work/hsx_mms_evolved8_history_contract.npz`, with its production history
and short-leg companion beside it.  It reused the fixed 64-cubed metric,
traced the same 56/56 real FCI wall hits at 8 cubed, completed one compiled
IMEX-SSP222 step with all four phi solves accepted, and obtained independent
source pairing `1.32e-15`.  The history is float64 and contains 424 active RLP
owners with finite positive aggregate volumes.  The exact initial frame has
zero absolute high-mode RMS and zero localized jump. It predates the aligned
production GMRES policy and initial-phi reconstruction, so it is retained as
historical wiring evidence only, not current acceptance evidence.

The completed local 32-cubed file `../work/hsx_mms_frozen32.npz` predates the
correction that adds production `div(b)` terms and the implicit wall residual
to the independent reference comparison. Retain it as a setup/wiring artifact,
but do not use it in a convergence fit. The later 21-frame evolved 32-cubed
history under `../work/stage7_mms_20step_local/` is finite through `2e-5`, but
it too predates the aligned GMRES/initial-phi correction and is not admissible
in the final MMS campaign.

The corrected analytical-initial-phi evolved smoke is
`../work/hsx_mms_analytic_phi_evolved8_t1e-6.npz`, with the history and analyzer
report beside it. It loaded the fixed 64-cubed metric, traced 56 forward and 56
backward physical-wall FCI hits at 8 cubed, constructed the radius-dependent
RLP owner topology, explicitly skipped initial GMRES reconstruction, and then
completed one full compiled IMEX-SSP222 step. The history contains two finite
float64 frames; all four stage phi solves were accepted and the independent
source-pairing error was `1.32e-15`. The analyzer passes every applicable
single-resolution check. The older
`../work/hsx_mms_postfix_evolved8_t1e-6.npz` artifact is retained only as
operator-wiring evidence and is superseded because it reconstructed phi at
the initial time. As intended, spatial and temporal convergence remain
unavailable until the actual multi-resolution campaign is run.

## Allocation-neutral campaign launcher and preflight

`scripts/submit_stage7_mms.sbatch` is the canonical campaign wrapper (the
historical filename is retained). It contains no `#SBATCH` resource request,
does not invoke `srun`, and does not require `SLURM_JOB_ID`; the person
initializing a run chooses the machine, scheduler/account, wall time, and
resource allocation. A real campaign still checks for four visible JAX GPU
devices because it uses eta sharding `--shard-counts 1 1 4`. The frozen and
evolved spatial campaigns each keep
32/48/64 in one Python invocation, which is required for in-memory reuse of
the fixed 64-grid metric evaluator. Direct driver invocations remain under the
operator's control; use the launcher for the documented remote campaign.

Set the cluster paths explicitly when submitting:

```bash
export DRBX_ROOT=/pscratch/sd/y/yiqunx/DRBX
export MAKEGRID=/pscratch/sd/y/yiqunx/uw_summer/mgrid_res2p5cm_180pln.nc
export VESSEL=/pscratch/sd/y/yiqunx/uw_summer/vessel_hsx_flare.txt
export OUTPUT_ROOT=/pscratch/sd/y/yiqunx/uw_summer/production_runs/stage7_mms
export PYTHON_BIN=python

CAMPAIGN_KIND=frozen bash scripts/submit_stage7_mms.sbatch
# Wait for the frozen job; the evolved launcher rechecks its spatial gate:
CAMPAIGN_KIND=evolved bash scripts/submit_stage7_mms.sbatch
# Wait for the evolved baseline before either temporal refinement:
CAMPAIGN_KIND=temporal_5e-7 bash scripts/submit_stage7_mms.sbatch
CAMPAIGN_KIND=temporal_2p5e-7 bash scripts/submit_stage7_mms.sbatch
```

The launcher shares reusable metric and JAX compilation caches beneath
`OUTPUT_ROOT/cache`; override `METRIC_CACHE_DIR` or `JAX_CACHE_DIR` when a
different persistent location is needed. Do not run these four jobs
concurrently against the same cache directories. Every campaign gets a
separate directory containing logs, the exact shell-escaped command, source
SHA-256 hashes, Git status, start/end times, and an exit status recorded by an
EXIT trap.

The sequence is enforced from artifact contents, not only documented.  An
`evolved` job runs the analyzer with `--require-spatial` on the canonical
frozen aggregate before importing JAX or inspecting GPUs.  Either temporal job
runs `--require-evolved` on both the frozen aggregate and the evolved baseline;
that gate requires exact 32/48/64 rows, `start=0`, `final=2e-5`, `dt=1e-6`,
20 steps, finite evolved-field errors, and one resolvable finite short-leg
diagnostic per resolution.  A `positive-growth` classification remains valid
evidence for the temporal investigation and is reported as a warning rather
than treated as a malformed artifact.

Every downstream manifest records the prerequisite aggregate paths and
SHA-256 digests.  The launcher also compares each prerequisite manifest's
deterministic source-hash block byte-for-byte with the current job before the
artifact gate runs.  If any production source changed after the frozen or
baseline campaign, use a fresh `OUTPUT_ROOT` and rerun the prerequisite chain;
mixing results from different source states is rejected.

A local or login-node dry run constructs the complete manifest and command but
does not invoke Python, `srun`, JAX, GPUs, or geometry:

```bash
DRY_RUN=1 CAMPAIGN_KIND=frozen \
  OUTPUT_ROOT=/tmp/stage7_mms_preflight \
  bash scripts/submit_stage7_mms.sbatch
```

For evolved and temporal dry runs, the manifest and terminal output include
the exact prerequisite gate command and expected artifact paths, but the gate
is not executed.  This preserves the dry run's no-Python contract.

The launcher deliberately refuses to overwrite an existing aggregate. The
MMS driver does not yet provide a checkpoint restart; `--reuse-history` is a
postprocessing path, not a resumable production advance, and the launcher
does not enable it. After a failed advance, preserve its logs and partial
history, then use a fresh `OUTPUT_ROOT` for the replacement campaign. The
metric and JAX caches may still be reused.

## Remote frozen campaign

Use one invocation for all three resolutions. This guarantees that the same
in-memory continuous metric evaluator and the same derived reference magnetic
field are reused throughout the comparison.

Both the frozen residual audit and the production advance are eta-sharded
across the four-GPU cluster with `--shard-counts 1 1 4`
(`frozen_execution=eta-sharded`, `evolved_execution=eta-sharded`).  The frozen
audit reuses the production driver's sharded model construction, invariant
full-torus curvature packing, real FCI maps, RLP payload, and boundary closure.
Its independent continuum projection and owner-volume norm reductions remain
host-side postprocessing.  A one-device development smoke retains the smaller
host-local frozen path and records `frozen_execution=host-single-device`.

Set `--metric-cache-dir` to a campaign-local persistent directory. Only the
initial fixed 64-grid evaluator uses this cache; each resolution-local build
reuses the in-memory `metric_context`. Add `--rebuild-metric-cache` only when
deliberately regenerating that fixed evaluator.

```bash
mkdir -p work/stage7_mms
PYTHONPATH=src python3 simulate_hsx_mms.py \
  --resolutions 32,48,64 \
  --shard-counts 1 1 4 \
  --metric-cache-dir work/stage7_mms/metric_cache \
  --time 1e-6 --final-time 1e-6 --dt 1e-6 \
  --advance-execution compiled \
  --output work/stage7_mms/hsx_mms_frozen_32_48_64.npz
```

Inspect these arrays before evolving:

- `exact_phi_residual` and `exact_phi_observed_order`;
- `phi_reconstruction_difference`;
- `rhs_term_error_norms` and `rhs_term_error_observed_order`;
- `partitioned_exact_phi_residual`;
- `partitioned_rhs_term_error_norms` and its observed order;
- `representation_error` and its observed order; and
- `region_cell_counts`.

The output records `generalized_potential_enabled`,
`reference_projection_method`, `reference_projection_order`,
`reference_derivative_method`, `reference_derivative_order`, and the periodic
coordinate-domain metadata so the midpoint reference and omega reconstruction
are auditable from the artifact itself.

The six disjoint regions are ordinary bulk, RLP rings, RLP-transition rings,
physical-wall cells, short-leg/topology-transition cells, and double-hit
cells. A non-finite value in a populated region, a source-pairing error above
roundoff, or failed phi reconstruction is a failed gate.

## Remote evolved campaign

After the frozen gate passes, use a 20-step baseline trajectory with every
step saved:

```bash
PYTHONPATH=src python3 simulate_hsx_mms.py \
  --resolutions 32,48,64 \
  --shard-counts 1 1 4 \
  --metric-cache-dir work/stage7_mms/metric_cache \
  --time 0 --final-time 2e-5 --dt 1e-6 \
  --advance-execution compiled --save-every 1 \
  --output work/stage7_mms/hsx_mms_evolved_32_48_64_t2e-5_dt1e-6.npz
```

Each resolution writes its own history and `short_leg_modes.npz` file. The
short-leg diagnostic records absolute localized high-mode RMS, maximum
poloidal jump, late-time log-growth rate, growth factor, fit R-squared, and a
classification. A normalized high-mode fraction is retained as optional
context, but it is not an MMS acceptance quantity and is not required by the
complete gate. `positive-growth` requires at least three late samples,
a fitted amplification of at least 1.25, and fit R-squared of at least 0.5;
otherwise a sufficiently sampled trajectory is classified as a bounded or
decaying closure layer.

The evolved trajectory does not enable the production driver's optional
per-step RHS-statistics history. That flag does not alter the numerical
trajectory and would add another full diagnostic RHS evaluation at every
saved step. Instead, the frozen MMS evaluates and stores the complete
per-field RHS ledger directly, which is the quantity used for term-by-term
spatial convergence.

To separate temporal error from spatial error, repeat a fixed-resolution run
over the same `2e-5` interval with successively halved timesteps. This gives
20, 40, and 80 steps while preserving 21 saved frames in every history. Use
distinct output names:

```bash
PYTHONPATH=src python3 simulate_hsx_mms.py \
  --resolutions 64 --time 0 --final-time 2e-5 --dt 5e-7 \
  --shard-counts 1 1 4 \
  --metric-cache-dir work/stage7_mms/metric_cache \
  --advance-execution compiled --save-every 2 \
  --output work/stage7_mms/hsx_mms_evolved_N64_t2e-5_dt5e-7.npz

PYTHONPATH=src python3 simulate_hsx_mms.py \
  --resolutions 64 --time 0 --final-time 2e-5 --dt 2.5e-7 \
  --shard-counts 1 1 4 \
  --metric-cache-dir work/stage7_mms/metric_cache \
  --advance-execution compiled --save-every 4 \
  --output work/stage7_mms/hsx_mms_evolved_N64_t2e-5_dt2p5e-7.npz
```

The production integrator here is second-order IMEX-SSP222, not RK4. For a
joint refinement against a second-order spatial method, balancing
`O(dt^2)` with `O(h^2)` requires `dt` proportional to `h`, so halving the
grid spacing calls for halving `dt`. The `sqrt(2)` rule would instead apply
to balancing fourth-order RK4 time error, `O(dt^4)`, against `O(h^2)`.

The aggregate's compatibility fields `integration_error_by_field` and
`integration_error` are owner-volume-weighted errors against the continuum
solution.  They include spatial truncation, omega-to-phi reconstruction, and
time-integration error, so their ratios across timesteps are diagnostic only
and are not used as a temporal-order estimate.  The same values are recorded
under the explicit names `continuum_total_error_by_field` and
`continuum_total_error`, with `integration_error_definition` documenting
their semantics.

For temporal acceptance, the analyzer reads the final N64 states from the
three production histories at `dt=1e-6`, `5e-7`, and `2.5e-7`.  Production
histories carry `owner_active` and `owner_aggregate_volume`, allowing the
login-node calculation to form
`||U_dt-U_dt/2||_V / ||U_dt/2-U_dt/4||_V` without rebuilding geometry.  The
common fixed-N64 spatial truncation cancels from these pairwise differences.
`--require-complete` requires temporal self-convergence order at least 1.8 for
every evolved field.  It never constructs a discrete manufactured source or
re-evaluates a production operator.

The compact artifact records the exact command, production configuration,
field and region names, RHS term names, fixed metric resolution, reference
magnetic field, field period count, and FCI trace-substep count. Do not begin a
128-cubed confirmation until the 32/48/64 frozen and evolved trends are
understood.

Every aggregate also carries `physical_parameters_json`, using the exact
`FciDrbEBRhsParameters` field names.  The intended MMS values are
`tau=1`, `mi_over_me=1836`, `rho_star=1`, all six perpendicular diffusion
coefficients equal to `1e-5`, all parallel diffusion/viscosity coefficients
equal to zero, and `Ve_nu=0`.  The analyzer requires this scalar for new
artifacts and requires exact cross-artifact equality.  The known historical
`hsx_mms_frozen8_final.npz` is reported with an explicit legacy warning when
this metadata is absent.

## Login-node artifact analysis

The simulation does not need to be rerun to merge campaigns or perform the
gates.  After copying the aggregate files, their production `*.history.npz`
files, and their `*.short_leg_modes.npz` companions to a login node, run:

```bash
PYTHONPATH=src python3 scripts/analyze_hsx_mms.py \
  work/stage7_mms/frozen_32_48_64/frozen_32_48_64.npz \
  work/stage7_mms/evolved_32_48_64_t2e-5_dt1e-6/evolved_32_48_64_t2e-5_dt1e-6.npz \
  work/stage7_mms/evolved_N64_t2e-5_dt5e-7/evolved_N64_t2e-5_dt5e-7.npz \
  work/stage7_mms/evolved_N64_t2e-5_dt2p5e-7/evolved_N64_t2e-5_dt2p5e-7.npz \
  --require-complete \
  --output work/stage7_mms/stage7_mms_analysis.json
```

The analyzer validates the production selectors, fixed `(64,64,64)` metric
and magnetic-field identity, finite norms in populated RLP regions, and
independent-source pairing.  It reports spatial and per-RHS-term orders,
owner-grid versus fine-grid representation error, production-history temporal
self-convergence order, and each short-leg growth classification.  The
complete gate requires exactly the canonical four artifacts, the intended
N64 timestep trio, float64 final histories with identical owner measures, and
finite aligned short-leg diagnostics for all evolved campaigns.  A
positive-growth classification
is reported as a warning for investigation; malformed configuration, missing
source pairing, or non-finite populated-region data is a hard failure.  Use
`--strict` when a campaign is required to contain every optional temporal and
short-leg diagnostic.  Temporal grouping uses the artifact's
`actual_timestep`, `start_time`, and `final_time` metadata when available, then
falls back to the recorded command line (and the filename `dt...` convention).

The two launcher prerequisite modes can also be reproduced directly on a
login node:

```bash
PYTHONPATH=src python3 scripts/analyze_hsx_mms.py \
  work/stage7_mms/frozen_32_48_64/frozen_32_48_64.npz \
  --require-spatial --output work/stage7_mms/frozen_gate.json

PYTHONPATH=src python3 scripts/analyze_hsx_mms.py \
  work/stage7_mms/frozen_32_48_64/frozen_32_48_64.npz \
  work/stage7_mms/evolved_32_48_64_t2e-5_dt1e-6/evolved_32_48_64_t2e-5_dt1e-6.npz \
  --require-evolved --output work/stage7_mms/evolved_gate.json
```
