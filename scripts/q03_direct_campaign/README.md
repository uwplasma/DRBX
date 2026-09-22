# Q03 direct cubic scalar-owner flux campaign

Research runner; **not a production diffusion operator or Q04 certification**.
The parallel roadmap owns numerical acceptance. The remote N32/N48/N64 static accuracy study is authorized; use
[REMOTE.md](REMOTE.md) for its frozen execution contract. The unrestricted
linear candidate has a demonstrated positivity/minimum-principle defect, so
passing this study cannot establish a production diffusion method.

The candidate computes one shared integrated parallel flux per physical face
from 120 scalar owner observations (24 per eta plane, five planes). It preserves
the bounded prototype's cubic basis including the constant, owner raw-midpoint
volume means, distance weights, continuous-geometry q9 face targets, opposite
incidence, and prescribed continuum boundary flux. No donor exchange, limiter,
field-dependent support tuning, or polynomial-degree increase is introduced.
The operator uses unit parallel diffusivity, matching the bounded comparison;
the historical source-catalogue time/chi parameters are provenance, not an
extra coefficient multiplying this static action.

## Setup and runtime changes

- Build one owner spatial index per eta plane. Query a conservative Euclidean
  ball containing the anisotropic nearest-neighbor ellipse, then sort by the
  exact original distance and owner ID. Trees prune searches; they do not
  replace the donor-selection metric. No ULP snapping or historical donor
  forcing is introduced.
- Precompute ten centered XY owner moments once. Translate these moments to
  each face chart, including periodic eta. This preserves actual agglomerated
  owner averages rather than substituting centroids. Owners spanning eta planes
  are rejected by this particular input contract.
- Solve 20-by-120 weighted systems with batched SVD. Do not change to normal
  equations, remove weak modes, or use a condition-number acceptance cutoff.
- Independent bounded chunks, process workers, read-only memory-mapped input,
  and at most two queued chunks per worker avoid a domain-sized reconstruction
  matrix or large worker-result transfers.
- By default retain fluxes and per-face diagnostics, not all 120 coefficients.
  `--save-coefficients` is an explicit disk-space tradeoff, useful for bounded
  replay and later structural audits. Coefficients can otherwise be regenerated
  from immutable scalar inputs. The execution kernel performs no primal replay
  solve; independent validation does.

This reduces **one-time setup work**, not only repeat-run caching. Geometry and
reference production remains separate and may dominate the next campaign.

## Inputs and reference accuracy

Run from `DRBX/`, with the `drb` environment and `PYTHONPATH=scripts`.
Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` before using multiple workers.
The portable remote path does not use historical workspace files.
Only the local producer adapter below needs the historical workspace research files
and geometry evaluators. The prepared directory and this Python package are
sufficient for the numerical runner; it imports only NumPy/SciPy and stdlib.

Bounded preparation and execution (the example input uses actual HSX):

```bash
python -m q03_direct_campaign.prepare \
  --workspace .. --N 32 \
  --sample ../work/parallel_q03_direct_owner_flux_20260921/N32 \
  --output ../work/direct_campaign/inputs/N32
python -m q03_direct_campaign.campaign run \
  --input ../work/direct_campaign/inputs/N32 \
  --output ../work/direct_campaign/results/N32 \
  --workers 2 --chunk-size 512 --batch-size 32 --save-coefficients
```

For local research preparation, omit `--sample`, prepare N32/N48/N64
into one campaign's inputs directories, then run each prepared directory.
`prepare` defaults to continuous q9/q11 reference fluxes on all selected faces,
evaluated in batches of 24 with checkpoints every 240 faces. Analytic geometry
and analytical field derivatives supply these references. The existing q9
polynomial targets are reused after frozen-source and geometry identity checks.
If a bounded sample has qualified q9/q11 data, that data is reused by default;
`--fresh-references` exercises the same producer path used by full preparation.
`--reference-workers` parallelizes these independent reference chunks with a
bounded queue; each process loads its continuous geometry once. This stage has
a larger per-worker footprint than the NumPy/SciPy flux runner. Its receipts
record peak memory and timing; choose concurrency using that stage's footprint.
On a tiny sample, worker startup can cost more than parallelism saves.

The low/high references use identical high-order prescribed boundary flux, so
their difference isolates the reference error relevant to the fixed-boundary
operator comparison. This is an empirical quadrature qualification, not a
rigorous error bound or physical wall-law certification.

`--reference-mode frozen` is available for cost/replay diagnostics only. Such
inputs explicitly lack a reference qualification and **cannot certify order**.
Old q4/q5 references cannot silently qualify the much smaller new errors.

## Recovery, output and acceptance

Keep all inputs/results in one campaign folder. No scheduler or allocation
setup is provided. For long authorized work use `supervise-long-runs`: stage
supervision, quiet 10-minute wakeups adaptive to 20–30 minutes, completion and
exit-status validation. Short validation needs no heartbeat.

An exclusive lock rejects duplicate runs. Resume the same command/directory:
chunk receipts validate source, input manifest, policy, range and output hash.
Worker count may change on resume; batch/chunk size, source, numerical policy
or coefficient retention requires a new output directory. A partial chunk
without a receipt is recomputed. Corrupt/incompatible completed chunks fail
explicitly. Preparation likewise checks its source/geometry/field identity.

`status.json` distinguishes running/failed/completed. `summary.json` contains
complete-owner weighted L2/max errors, regional squared-error contributions,
reference budgets and cubic defects. `actions.npz` contains complete actions;
chunks retain canonical face IDs, fluxes, conditions and reproduction defects.
Regions may overlap and should not be summed blindly. Bounded sample norms
are never interpreted as global orders.

After complete global runs of one policy/source:

```bash
python -m q03_direct_campaign.campaign orders \
  ../work/direct_campaign/results/N32 \
  ../work/direct_campaign/results/N48 \
  ../work/direct_campaign/results/N64 \
  --output ../work/direct_campaign/operator_orders.json
```

The comparison requires each nonconstant field to have global physical-volume
weighted L2 order >=1.8 on both intervals and reference error below 10% of
spatial error. Q04 remains false even if this static accuracy check passes:
positivity/minimum principle, dissipation, solution MMS, actual JAX action and
applicable integration qualifications remain separate pending work. Shared
incidence alone does not certify those properties.

The runner deliberately has no timestep or end-time parameter: it performs
static operator checks only. A future evolved MMS must introduce a separate
identity including its time integration and requested qualification checks.
