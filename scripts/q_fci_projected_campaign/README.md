# Projected fourth-order traced FCI: global static qualification

This is a new standalone research campaign. Run the specified computation and
return its artifacts for local interpretation. It is not a production solver,
energy certificate, evolved test, or a guarantee of second-order convergence.
The previous single/four-seed return campaigns are separate methods; their
checkpoints must not be relabeled or resumed here.

## Frozen computation

On all canonical interior faces at N32/N48/N64, use q5 physical face points,
a ten-coefficient physical-area cubic projection, actual HSX field tracing,
and the fixed OwnerMoments transverse cubic plus four-plane eta interpolation.
All numerical endpoint values come from actual raw-volume weighted midpoint
owner observations, including real aggregates. The base span is H=2*pi/N:

```
D_full = [T(F(+H/2))-T(F(-H/2))]/H
D_half = [T(F(+H/4))-T(F(-H/4))]/(H/2)
D_fourth = (4 D_half-D_full)/3
```

The coefficient cancellation is fourth order in the line interval, not a claim
of fourth-order full-mesh convergence. Both endpoints are reconstructed directly
from owner data; no interpolation between the full endpoints supplies half
values. Exact endpoint and gradient channels are diagnostics only. The geometry
model and endpoint selector are inherited verbatim from the pinned preceding
research campaign. RK4 uses 64 substeps per distinct leg. Full/half maps and
all four fields share geometry, support selection and sparse rows.

### Explicit near-wall interval policy

The fixed-span interior method is not defined globally: a bounded N32 coverage
check found 115 out of 13,800 legs leaving the domain. The user authorized one
bounded boundary-adjacent test before global preparation continued. Its results
and design are in `validation/boundary_summary.json` and
`validation/boundary_design.md`.

At each quadrature point, if any full/half leg leaves 0<r<1, halve that point's
span and retry, reusing the existing half legs as the new full legs. Only two
new shorter legs are needed per retry. At most eight halvings are permitted.
All derivatives use the accepted span in their denominator. This is a frozen,
geometry-only rule; scalar values, error norms and other points' errors never
select a span. Bad/nonpositive b^eta or an unresolved exit is a hard failure.
No seed, face or owner may be dropped. No old-operator fallback, outside-domain
geometry query or analytic interior endpoint is permitted.

This policy preserves unshortened interior points and retains the prior
prescribed integrated wall-flux contract. The four manufactured fields have
zero gradient at r=1, so physical-wall face flux is prescribed as exactly zero.
No scalar Dirichlet datum is introduced along a clipped field line. This does
not qualify general physical-wall or sheath treatment.

The bounded six-pair test resolved all tested exits (spans down to H/64), passed
constants/cubic transfer/shared-flux and host/step controls, but has mixed
accuracy: 12/18 pair RMS comparisons improve against the original return, with
some hotspot regressions up to 4.20x and substantial q5/q7 candidate changes.
This limitation is deliberate evidence to carry into qualification, not an
accuracy pass or a reason to tune the remote run.

## Qualification and references

Score **every owner on the same complete physical domain** at N32/N48/N64.
One positive-coordinate canonical flux contributes plus to its lower owner
and minus to its upper owner. Collapsed-axis and same-owner internal faces
are absent in the frozen topology; no additional owner cropping is allowed.
Numerical actions divide by the original midpoint physical owner volumes.

The primary independent reference is exact analytic q11 face divergence divided
by summed continuous q7 owner volumes. q9 faces/q5 volumes give the empirical
reference difference. The static scientific gate remains L2 order >=1.8 on
both refinement intervals for each nonconstant field, with reference difference
less than 10% of its numerical error. Insufficient reference resolution is a
scientific non-pass, not authorization for new remote quadrature or method tuning.
Preflight also records independent strong-volume q9 controls on ordinary and
wall owners, and q7 candidate construction on every preflight face. Those
bounded construction checks are not a full-domain q7 bound. Full-domain q5
candidate quadrature remains part of the method being qualified.

Output channels include full/half/fourth numerical and exact endpoint actions,
exact-gradient projection, q5/q9/q11 reference actions, per-axis fourth action,
physical volumes, error components, RMS/maxima, canonical face maps and trace
validity/interval levels. Fixed logical regions and axis/aggregate/wall strata
are reported without changing the global denominator. The old global return
is not recomputed: archived comparison and scientific diagnosis occur locally.

## Source and inputs

The runner uses the existing tracked `scripts/q_fci_return_campaign/`
`geometry_source.tar.gz` and `geometry_source_manifest.json` to extract and
verify the frozen 127-file geometry package under the campaign's `software/`.
It imports that source before an installed editable DRBX. Do not rebuild the
bundle remotely. The committed endpoint/geometry kernels and frozen MMS module
are included in the campaign source identity, along with all configuration.

Use the established immutable input root from the prior campaigns. Its eight
files are verified against `scripts/q_fci_return_campaign/input_manifest.json`:

- `hsx_metric_d58d392545fd3917efeb83b6.npz`
- `mgrid_res2p5cm_180pln.nc`
- `geometry_artifacts/rlp_convergence_32_48_64_20260917/{N}x{N}x{N}/base_geometry.npz`
- The corresponding `rlp_topology.npz`, N=32,48,64.

Inputs/environment may remain external. Record the absolute input root and
identities; no cache regeneration or package installation is part of this
campaign. Old return checkpoints and local patch maps are not campaign resume
inputs. Only this runner's compatible same-folder checkpoints may be resumed.

## CPU execution and optimization

This is a **single-node multiprocessing spawn runner**, not MPI/distributed.
One parent coordinates the campaign and reduces units in deterministic order;
each child owns distinct batch checkpoints. Linux children bind to one CPU
from the scheduler affinity mask before JAX initialization. BLAS/OpenMP threads
are one. Worker/memory options cap process count; memory allowances are planning
limits, not an OS resource enforcer. Actual RSS is recorded per completed batch
and on worker failures.

Preflight and global preparation both use the selected worker pool for traces,
endpoint fits, references and volume integration. All fields share the maps.
No separate writer should be launched per resolution. Routine final reductions,
hashing and bookkeeping are serial. Source/input verification precedes work.

Optimizations that preserve the frozen numerical choices:

- Build faces directly, eliminating the old whole-grid secant catalogue,
  neighbor catalogue expansion and 19-mode return SVD.
- Fixed vectorized RK4 batches of 200, with one padded shape and a shared
  campaign-local compilation cache; do not retrace prior half legs on halving.
- Reuse one position/Jacobian/field evaluation for face projection and q5
  exact-gradient integration; avoid unused field Hessians in endpoint/reference
  evaluation and the selector's unused quadratic diagnostic fit.
- Save full/half sparse maps; derive the fourth map algebraically instead of
  storing a third duplicate. Shared maps apply to all scalar fields.
- Persistent per-batch trace checkpoints precede endpoint fitting. Prepared
  maps precede field/reference work, so interruption can reuse expensive setup.
- Global batches contain 32 faces, reducing filesystem metadata and scheduling
  overhead; preflight batches contain two to expose parallel work. Volume units
  are 32 raw cells globally and eight in preflight. No scaling study is required.

JIT/setup, reference and interpolation costs must be included in allocation
planning. Small sparse application timings do not estimate preparation time.
Each preflight summary reports measured CPU/peak-memory costs and a deliberately
rough full-domain CPU-work projection. The remote setup skill chooses allocation,
affinity, workers, memory allowances and walltime from these measurements.

## Commands, one folder, recovery

Run from the repository root. Use one newly created unique absolute `CAMPAIGN`
folder for **all** N32/N48/N64 stages, scheduler output, logs, caches, provenance,
job records and the final operational receipt. Keep real files, not symlinks
to campaign-generated outputs elsewhere. Immutable inputs may remain external.

The remote **run drbx on perlmutter** skill supplies `INPUT_ROOT`, `WORKERS`,
`MEMORY_GIB`, `WORKER_MEMORY_GIB` and `RESERVE_GIB`; do not infer them from GPUs.

```bash
mkdir -p "$CAMPAIGN"/{logs,tmp,cache,provenance}
export TMPDIR="$CAMPAIGN/tmp" XDG_CACHE_HOME="$CAMPAIGN/cache/xdg"
export DRBX_CACHE_DIR="$CAMPAIGN/cache/jax"
export JAX_COMPILATION_CACHE_DIR="$CAMPAIGN/cache/jax"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu
RUNNER=scripts/q_fci_projected_campaign/campaign.py
COMMON=(--input-root "$INPUT_ROOT" --output "$CAMPAIGN")
CPU=(--workers "$WORKERS" --memory-budget-gib "$MEMORY_GIB"
     --worker-memory-gib "$WORKER_MEMORY_GIB" --memory-reserve-gib "$RESERVE_GIB")
python "$RUNNER" verify "${COMMON[@]}"
python "$RUNNER" preflight "${COMMON[@]}" "${CPU[@]}"
python "$RUNNER" run "${COMMON[@]}" "${CPU[@]}" --resolutions 32
python "$RUNNER" run "${COMMON[@]}" "${CPU[@]}" --resolutions 48 64
python "$RUNNER" validate "${COMMON[@]}"
python "$RUNNER" status "${COMMON[@]}"
```

Capture stdout/stderr and exit status for each command, with all scheduler logs
under `CAMPAIGN/logs`. Preflight must complete at **all three resolutions**
before any `run`. It includes complete ordinary, aggregate, transition, axis,
wall, inward and seam owner actions. A local `smoke --resolutions 32` command
exercises one complete ordinary owner and is never a substitute for preflight.

The N32 stage precedes N48/N64. Operational/coverage/identity/finite/constant
failures stop execution; preserve their evidence and return it. A poor scientific
error or order does not authorize changing method or aborting the prescribed
resolution sequence if operational checks pass. The final `validate` without
a resolution filter computes the complete three-resolution scientific flags.

Resume by repeating the interrupted command with the same output folder,
revision and numerical inputs. Completed payloads, maps and traces are SHA256
checked; missing units continue. Worker and memory settings may change. An
exclusive coordinator lock rejects duplicate writers. Do not delete receipts,
edit manifests, reinterpret old checkpoints or bypass a corrupt-payload failure.
A source/numerical-policy change requires a fresh campaign folder.

## Return

Return the single absolute campaign folder with `campaign.json`, input locations,
environment, status, extracted software, all preflight/global selections and
plans, trace/map/face/volume checkpoints and receipts, failure evidence if any,
actions, summaries, `preflight_validation.json`, `global_validation.json`,
logs/job records and `operational_receipt.md`. Summaries contain machine-produced
errors/orders/pass-fail flags; return them unchanged without interpretation.

The operational receipt must state commit/identities, job/allocation IDs,
requested/effective CPU workers, memory allowances/observed peaks, elapsed times,
exit statuses, completed/failed/incomplete stages, validation statuses, exact
failed command if relevant, and the absolute folder path plus inventory.

Run the specified computation and return its artifacts and operational receipt.
Do not analyze the scientific results; we will do that locally.
