# HSX FCI Braginskii backend (drbx.fci_braginskii)

The `drbx.fci_braginskii` package is the FCI drift-reduced Braginskii plasma
backend from the 2D_fci line at `d340f4d638e0d9495546d1fd2e2c66c1c206704a`,
vendored into its own namespace (`drbx.fci_braginskii.native`,
`drbx.fci_braginskii.geometry`) so it is independent of the `drbx.native` FCI
modules. Its plasma operators and IMEX advance are those of that commit, apart
from tracer-safe host guards (`drbx.fci_braginskii._host_guards`) and
dead-code removal, which leave the compiled graph unchanged. Geometry is
produced separately: `simulate_hsx_blob.py` requires `--geometry` and only
deserializes the schema-v1 base geometry, center maps, RLP topology and cell
positions (`drbx.fci_braginskii.geometry.fci_simulation_geometry`). It never
builds physical geometry and does not activate the artifact's newer overlap,
vertex reconstruction, endpoint-field extensions, or curvature edge one-form.
Initial filament label tracing and device lowering still occur.

The canonical 32-cubed bundle uses 64 trace substeps and a 64-cubed coordinate
fit sample, unlike the historical 48-cubed/four-substep run. This is a test of
the old plasma architecture with canonical geometry, not an exact replay.
The bundle controls resolution, topology and producer geometry metadata.

From the repository root:

```bash
python simulate_hsx_blob.py --geometry /path/to/hsx_fci_32x32x32 \
  --flux-framework production-split --parallel-operator-scheme fci \
  --parallel-flux-pairing support-core \
  --parallel-boundary-pairing characteristic-sat \
  --parallel-characteristic-wall-law physical-boundary-state \
  --parallel-velocity-wall-bc dirichlet-zero \
  --parallel-short-leg-treatment local-backward-euler \
  --parallel-short-leg-selection all-physical-walls \
  --parallel-short-leg-cfl-limit 2.5 \
  --time-integrator imex-ssp222 --advance-execution staged-compiled --no-phase-timing \
  --shard-counts 1 1 1 --halo-width 2 --neumann-ghost-scheme physical \
  --poisson-bracket-scheme material-scalar-third-order-upwind \
  --parallel-diffusion 0.0 --perp-diffusion 1e-5 \
  --gmres-preconditioner line-u --final-time 0.0075 --num-steps 32 \
  --save-every 8 --checkpoint-every 8 --diagnostic-every 8 \
  --filament-cache-dir /path/to/run/initialization_cache \
  --output /path/to/run/history.npz
```

`staged-compiled` compiles four smaller kernels (implicit stage with potential
solve, explicit right-hand side, standalone potential, stage diagnostics)
instead of one fused IMEX step; on the N32 case it was faster both with a cold
and with a warm compilation cache. `compiled` remains available.

The driver imports `drbx` from this repository's `src` by default;
`DRBX_SOURCE_ROOT` overrides that. Set `DRBX_CACHE_DIR` to a writable,
run-local compilation cache.

The velocity selector is the historical no-flow wall implementation, not the
later named physical-wall selector. Density/temperature physical Neumann and
potential/vorticity Dirichlet conditions retain historical axis/periodic handling.
This plasma-only smoke does not qualify wall budgets, neutral reactions, coupling,
MMS accuracy, or long-time stability. No sheath/recycling model is introduced.

Focused checks:

```bash
DRBX_TEST_GEOMETRY_BUNDLE=/path/to/hsx_fci_32x32x32 python -m pytest -q tests/fci_braginskii
```

Report geometry loading/lowering, compilation, advance and total wall time
separately. A short successful evolution does not establish a ten-minute full
physics run; cold and warm-cache runs must be distinguished.
