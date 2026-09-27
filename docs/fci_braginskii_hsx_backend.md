# HSX FCI Braginskii backend (drbx.fci_braginskii)

`drbx.fci_braginskii` is a compact FCI drift-reduced Braginskii plasma backend
for HSX. It is the plasma model of the 2D_fci line at
`d340f4d638e0d9495546d1fd2e2c66c1c206704a`, vendored into its own namespace
(`drbx.fci_braginskii.native`, `drbx.fci_braginskii.geometry`) and independent
of the `drbx.native` FCI modules. The code was reduced to the one production
configuration below; with `--rho-star 1 --no-implicit-current-phi-pair` the
compiled advance and its fields are identical to the base commit. The
defaults add two changes on top of that: a consistent rho* normalization and
an implicit current/potential pair (see Normalization below). Tracing is
about ten times faster because host-side validation no longer builds JAX's
jaxpr-walking error messages (`drbx.fci_braginskii._host_guards`).

The backend consumes a precomputed geometry artifact. It contains no geometry
production (field-line tracing, metric fitting, RLP construction); those live
in the 2D_fci geometry pipeline.

## Fixed configuration

- Geometry: toroidal FCI geometry with the radius-dependent angular (RLP)
  topology, read from a schema-v1 artifact.
- Time advance: IMEX-SSP222. Material fluxes on all physical wall legs are
  treated by local backward Euler inside each stage. By default, each
  implicit stage also does one linearized backward-Euler coupled potential
  solve treating mu*grad_par(phi) and the homogeneous parallel current
  divergence implicitly, with every other coefficient frozen at the stage
  state; the wall current lift stays explicit.
  `--no-implicit-current-phi-pair` restores the explicit pair.
- Parallel transport: FCI parallel operator, production-path characteristic
  material fluxes with support-core pairing and characteristic-SAT
  boundaries, physical-boundary-state characteristic wall law.
- E x B advection: material-scalar third-order upwind bracket (compatible
  centred bracket for vorticity).
- Boundaries: no-flow walls (Vi = Ve = 0 face traces), physical-normal Neumann
  ghosts for density and temperatures, Dirichlet potential and vorticity.
- Potential solve: SOLVAX FGMRES with the line-u preconditioner.

No sheath, presheath, recycling or neutral model is included.

## Normalization

Lengths in the geometry are in metres (the artifact is SI: major radius
0.87-1.53 m). |B| is in tesla (B_ref = 1 T). n, Te, Ti are in reference
units, and time is in L_ref/c_s with L_ref = 1 m. rho* = rho_s / L_ref,
rho_s = sqrt(Te m_i) / (e B_ref); E x B and curvature drift terms scale with
rho*, and the polarization relation scales with rho*^2. The default rho* is
5e-4 (hydrogen, Te ~ 24 eV, B = 1 T; rho* scales as sqrt(Te)/B, e.g. 4.6e-4
at 20 eV). With these values c_s ~ 4.8e4 m/s, so one time unit is about 21
microseconds and t = 0.15 is about 3 microseconds. The base commit used
rho* only as a divisor of the E x B terms; this is the same operator at
rho* = 1.

## Geometry artifact

`--geometry` names a directory with `manifest.json` (schema
`drbx.fci_simulation_geometry`, version 1). The reader
(`drbx.fci_braginskii.geometry.fci_simulation_geometry`) loads the base
geometry, centre maps, RLP topology and cell positions, and ignores the
artifact's newer overlap, vertex-reconstruction, endpoint-field and curvature
edge one-form components. Resolution, topology and producer metadata come from
the artifact.

The canonical 32-cubed bundle uses 64 trace substeps and a 64-cubed
coordinate-fit sample, unlike the historical 48-cubed, four-substep runs, so it
tests the old plasma model on canonical geometry rather than replaying a
historical run.

Restore the canonical bundle with
`python scripts/fetch_example_artifacts.py --skip-media --hsx-geometry`, which
downloads the `fci-braginskii-geometry-v1` release asset into
`artifacts/geometry/hsx_fci_32x32x32` and checks every file against the
manifest checksums. Until that release is published, obtain the bundle from
the maintainers and place it at the same path.

## Running

From the repository root:

```bash
python simulate_hsx_blob.py --geometry artifacts/geometry/hsx_fci_32x32x32 \
  --save-every 10 --checkpoint-every 50 --diagnostic-every 10 \
  --filament-cache-dir /path/to/run/initialization_cache \
  --output /path/to/run/history.npz
```

The defaults run to t = 0.15 in 200 steps (dt = 7.5e-4) at rho* = 5e-4 with
the implicit current/potential pair. With the explicit pair this rho* limits
dt to about 1e-4. With the implicit pair, 32-cubed runs are stable at dt =
1.5e-3 (tested to t = 0.3) and fail at 5e-3, where the explicit electron
parallel terms set the limit. Runs at dt = 1.5e-3, 7.5e-4 and 3.75e-4 to
t = 0.15 converge at about order 1.5 in dt. At the default step the
estimated error is 0.1% or less (relative L2) for density and temperatures,
and the blob centroid is converged to 1e-5 m. The potential is 2 to 5% off
in amplitude and about 13% off in relative L2. That error is smooth and sits
in the bulk around the filament, not at the walls. Ve behaves the same way,
and vorticity converges more slowly. Where the potential matters, use more
steps: 400 steps (dt = 3.75e-4) bring the potential error to about 5%.

The driver imports `drbx` from this repository's `src`; `DRBX_SOURCE_ROOT`
overrides that. Set `DRBX_CACHE_DIR` to a writable, run-local JAX compilation
cache so repeated runs skip compilation.

Remaining options cover time and output cadence, snapshots and restarts, the
initial filament (`--blob-*`, `--density-amplitude`), physical parameters
(`--tau`, `--rho-star`, `--mi-over-me`, `--perp-diffusion`,
`--parallel-diffusion` (default 0), `--electron-collision-frequency`) and GMRES
tolerances; see `python simulate_hsx_blob.py --help`.

`--advance-execution` selects how each step runs:

- `staged-compiled` (default): four reusable kernels (implicit stage with
  potential solve, explicit right-hand side, standalone potential, stage
  diagnostics). Fastest on the N32 case with both cold and warm caches.
- `compiled`: one fused IMEX step.
- `eager`: no outer `jax.jit`; slow, but Python-level debugging and printing
  work inside the right-hand side.

Phase timing, which reports the operator and GMRES shares of each step, is on
by default; `--no-phase-timing` disables it. At N32 with the defaults the
potential solves take a little over half of each step (about 90 GMRES
iterations per solve on average, four solves per step: two coupled
current/potential solves and two standalone potential solves), and a step
takes about 16 s on an 8-core M1.

Multi-device runs decompose only the toroidal (eta) direction:
`--shard-counts 1 1 N`, with N dividing the eta resolution. On CPU, emulate N
devices with `DRBX_HOST_DEVICE_COUNT=N`.

## Neutral-model coupling

The backend has no source-term or neutral-coupling hook. Coupling a
neutral model, including reaction sources that depend on the plasma
density and temperatures, is left to the neutral-model implementation;
the right-hand side is assembled in `LocalFciDrbEBRhs.evaluate_stage`.

## Known limitations

- Multi-device runs differ from single-device runs at the eta shard
  interfaces. After one N32 step on two shards the largest relative
  differences are 5e-4 (vorticity) and 3e-4 (potential), confined to the
  planes adjacent to the two interfaces, and unchanged by a tighter GMRES
  tolerance. The discrepancy is inherited from the base commit. Treat
  single-device runs as the reference.
- The largest relative GMRES residual in the first N32 step is about 8e-4 at
  the default tolerances, above the 5e-5 acceptance tolerance; the driver logs
  it as `gmres-relres`.
- Under the physical-boundary-state wall law, an isolated stiff two-wall row
  (the historical hotspot) is admissible, but its selected parallel material
  residual is not sign-restoring on its own.
- A short successful evolution does not qualify wall budgets, MMS accuracy or
  long-time stability.

## Tests

```bash
DRBX_TEST_GEOMETRY_BUNDLE=artifacts/geometry/hsx_fci_32x32x32 python -m pytest -q tests/fci_braginskii
XLA_FLAGS=--xla_force_host_platform_device_count=4 DRBX_HOST_DEVICE_COUNT=4 \
  python -m pytest -q tests/fci_braginskii
```

The second run also covers the multi-device halo exchange. The tests find the
bundle at `artifacts/geometry/hsx_fci_32x32x32` automatically, so
`DRBX_TEST_GEOMETRY_BUNDLE` can be omitted once it is restored there.
