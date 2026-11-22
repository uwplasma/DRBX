# FCI nodal manufactured solutions

`drbx.fci_nodal` builds and evolves electromagnetic drift-reduced Braginskii
manufactured solutions (MMS) on an HSX vessel. The perpendicular operators
and parallel flux-box operators use one Zernike core-ring layout. The geometry
build traces field lines and constructs a polygon representation of the wall.
One wall-band closure supplies the parallel boundary terms for the conducting
wall (conducting_wall) and the floating wall (floating_wall).

At each stage of explicit third-order SSP Runge–Kutta (SSPRK3), the backend
solves for the electrostatic potential by polarization and for the parallel
vector potential by Ampère's law with electron inertia. The manufactured
configuration runs on CPU in float64, uses algebraic flux correction (AFC),
and has magnetic flutter off.

## Device inputs and symmetry

Supply a MAKEGRID field archive, its coil currents in archive order, and the
vessel file. The build uses a fixed spline fit and exact level-set planes.
It derives one filtered field table from the device inputs and uses it
throughout the case.

| `fci_nodal.symmetry` | Build region |
| --- | --- |
| `nfp` | Build source geometry for one field period and replicate it around the torus. The number of eta planes must be divisible by `field_periods`. |
| `none` | Build source geometry over the full turn without period replication. `field_periods` still describes the device and field filter. |

## Build and run a case

Use a TOML deck with `[model] backend = "fci_nodal"`. The
[Deck Reference](decks.md#nodal-fci-manufactured-cases) lists all keys.

| Deck table | Settings |
| --- | --- |
| `[fci_nodal]` | `case_dir` holds build and run products. `resolution` sets the nodal resolution and number of eta planes. `symmetry` and `field_periods` set the geometry layout. `owners` sets the number of owner slabs and defaults to one. |
| `[geometry]` | Device paths and currents, polygon `rings`, dense metric `resolution`, fixed map and fit settings and geometry build `workers`. |
| `[parallel]` | Build `workers` for indexed parallel construction. |
| `[field_filter]` | `max_harmonic_per_period`, `samples_per_period` and `table_grid` set the shared field table. |
| `[mms]` | `wall_law` selects `conducting_wall` or `floating_wall`. `field_profile` selects `wall_flat_temperature` or `prescribed_wall_heat_flux`. `max_steps` caps the advance. `capture_static_rhs` records the initial RHS. Owner launch options configure CPU execution. |

Set the MAKEGRID and vessel paths in `examples/inputs/fci_nodal_mms.toml`.
The example uses the shipped vessel and an external MAKEGRID archive.
Its dense metric output grid is `[16, 16, 16]`. The fit sample grid and
metric mesh are fixed at `[8, 9, 8]` and `[64, 64, 64]`. Relative paths
resolve from the deck file. Override a setting with `--set table.key=value`.

```bash
drbx inspect examples/inputs/fci_nodal_mms.toml
drbx build examples/inputs/fci_nodal_mms.toml
drbx build examples/inputs/fci_nodal_mms.toml --verify-only
drbx run examples/inputs/fci_nodal_mms.toml
```

`inspect` shows the resolved deck and case options. `build` creates or
resumes the case and publishes it. `build --verify-only` checks the published
receipt chain and deck build settings without rebuilding. `run` uses
that publication and starts from the exact manufactured initial state.
The case can reuse matching MMS receipts. Changing the MMS field profile or step
controls can reuse the built geometry and operators.

The Python case API is available through `drbx.fci_nodal.case`.
`build_case(options, case_dir)` takes a `CaseOptions` and returns
`owner_inputs/receipt.json`. `run_mms(case_dir, mms_options)` uses that
receipt and returns the owner state directory and summary fields.

## Build products

| Stage | Products |
| --- | --- |
| `metric` | `hsx_metric_*.npz` with the dense metric, evaluator data and reference magnetic field |
| `field_table` | `field_table_*.npz` shared by geometry and operators |
| `polygon` | Polygon geometry, merged wall rows, face flux and wall rows bound by `polygon_case.json` |
| `geometry` | Indexed physical volumes, faces, polygon wall rows, fits and provenance |
| `parallel_operators` | Indexed owner construction with per-plane parallel rows and moment-reduced face quadrature recipe |
| `perpendicular_operators` | Owner perpendicular reference and SBP leaves with a build report |
| `owner_inputs` | Receipt binding those build inputs and the owner parallel application products |
| `wall_data` | Wall rows, limiter graph and routing by owner |
| `mms_prepare` | Prepared parallel geometry and conducting_wall/floating_wall manufactured_field_data rows by owner |
| `mms_run` | Final state by owner and shared run summary |

The parallel operator uses the absolute-flux action and its analytic derivative in
`operators/parallel/absolute_flux.py`. Wall preparation evaluates the magnetic
field direction through `evaluators/magnetic_field.py`, also used for
metric differentiation in geometry and parallel construction. MMS preparation
evaluates its divergence.

Set `geometry.workers` and `parallel.workers` to control local build concurrency.
Both default to one. MMS execution uses the owner count independently.

## Owner-sharded builds and MMS

An owner holds a contiguous slab of whole toroidal (eta) planes. The perpendicular and parallel operators
use the same slab boundaries and canonical nodal order. Each owner must
hold at least four planes. For example, at resolution 16 with two owners, the slabs
have eight planes each.

Omit `owners` for one slab, or set it in `[fci_nodal]` to split the full
layout into several slabs. Use a separate case directory when changing the
owner count.

The build publishes `owner_inputs/receipt.json` with the indexed geometry,
shared metric and field table, direct perpendicular slabs, and owner parallel construction and
application products. Per-owner inputs include perpendicular metric and setup leaves,
parallel operator rows and face records, wall and AFC data, and prepared manufactured
data. Some slab files include periodic halo planes. The run writes
`node_ids.npy` and the final state `nodal_state.npy` under each owner's directory in
`mms_run/attempt-*/owners/`. With `mms.capture_static_rhs = true`,
each owner also writes `static_rhs.npz` with the raw and manufactured initial
right-hand sides. The attempt has a shared `result.json` with scalar summaries.

`mms/evolved.py` assembles manufactured stages in `owner_stage` and
constructs the conducting_wall and floating_wall wall data. `mms/cache.py` validates and places
prepared manufactured_field_data rows on the owner layout. The conducting_wall and floating_wall closures are
created from the residual and exterior state for each stage.

With `mms.launch_mode = "local"` (default), a single process uses one JAX CPU
device per owner. With `"per_owner"`, the advance uses one process and one CPU
device per owner, coordinated over loopback on one machine. With
`"scheduler"`, the driver appends a rank entry to `mms.rank_command`
and invokes it once. Supply the command and its arguments as a TOML list.
The command must start exactly one task per owner, pass the appended command
to every task, supply a unique zero-based task index, and wait for the team.
By default the launcher reads the index from `SLURM_PROCID`. Set
`mms.rank_index_env` to use a different variable. Set
`mms.collectives_interface` to select the CPU collectives interface, or omit it
to let the collectives library choose.

Run the driver in the batch script body, outside a scheduler step and on
the host where rank 0 will run. All ranks need to reach that hostname and use
the same case path and Python environment. The parent verifies all rank
outputs before writing the run receipt.
In either ranked mode, optionally set
`mms.rank_rss_limit_bytes` and `mms.run_timeout_seconds` for the
per-rank RSS cap and the launch wall-clock timeout. Omitting them leaves both
deck limits unset. The worker guard and coordinator lifecycle limits still apply.

Single-machine checks at resolution 16 used saved spline/exact inputs for wall_flat_temperature floating_wall.
They exercised adopted one- and two-owner cases and a direct two-owner
build from saved upstream products. A two-process loopback check compared
the static raw and manufactured RHS and the third-step state per owner
with the single-process owner run. The corresponding arrays matched byte
for byte. Cold owner builds with fresh tracing, multi-node runs, other
resolutions, additional owner layouts, and owner runs for other wall
field profiles or wall laws are outside this evidence.
Scheduler argument dispatch, rank identity and deck routing have data-free
coverage. The opt-in resolution 16 scheduler comparison uses a local task command.
Live scheduler-mode runs and cross-host comparisons are pending.

## Not supported

This backend runs manufactured solutions only. The following workflows are
outside its interface.

- Physical initial states and physical time evolution, including restart,
  checkpoints and snapshots. Setting `fci_nodal.initial_state` reports
  the missing physical-state loader and layout validation.
- A physical sheath wall model with current closure.
- Implicit–explicit (IMEX) time stepping.
- Coupled neutral transport and recycling.

## Verification status

Verification coverage for this backend is limited to resolution 16. A spline-map
order-of-accuracy study is still outstanding.

### One resolution 16 example on a Mac laptop

A cold spline/exact resolution 16 `nfp` case with CPU float64 reached three wall_flat_temperature floating_wall
SSPRK3 steps in about 56 min 28 s. It traced 158,733 rays. Peak RSS was
2.747 GiB in the metric stage. Tracing stages used one library thread.
These figures describe the archived gathered build on one laptop. That build
spent 758.577 s and traced 29,970 rays in the retired wall-entry stage.
The archived nodal reference and magnetic companion stages took 0.996 s
and 3.848 s, respectively.

| Stage | Seconds | Peak RSS (GiB) | Rays |
| --- | ---: | ---: | ---: |
| Metric and cold spline fit | 175.459 | 2.747 | 0 |
| Field table | 133.646 | 0.947 | 0 |
| base polygon and wall classification/repaired polygon and merged wall rows | 1,129.557 | 1.330 | 71,579 |
| Indexed geometry | 1,051.894 | 1.020 | 57,184 |
| parallel operators | 32.342 | 1.081 | 0 |
| perpendicular bundle | 33.586 | 1.547 | 0 |
| Publish | 6.445 | 0.246 | 0 |
| Wallfix preparation | 19.899 | 0.924 | 0 |
| Wallfix floating_wall static and three steps | 26.639 | 1.578 | 0 |

The time step was 1.31506e−4 and the positivity bound was 1.46117e−4 in
normalized units. The third step ended at 3.94517e−4.
