# FCI simulation geometry artifacts

HSX physical geometry is generated independently from every simulation.
The canonical bundle is a directory with `manifest.json` plus separate files
for base geometry, cell-center maps, the complete canonical raw-vertex trace
atlas, polar/RLP topology and volumes, the lean owner-overlap graph, and any
optional curvature edge one-form.

Generate a bundle once for a resolution:

```bash
conda run -n drb python generate_hsx_fci_geometry.py \
  --resolution 32 32 32 \
  --metric-mesh-shape 32 32 32 \
  --trace-backend jax \
  --output /path/to/hsx-fci-32
```

The producer traces cell centers and raw transverse vertices with 64 RK4
substeps everywhere, builds the current direct
owner-boundary/straight-edge/centroid graph, and validates the completed
object before publishing it atomically. The 128-substep comparison remains an
explicit development diagnostic for changes to the magnetic configuration,
eta spacing, or tracing algorithm; it is not repeated during artifact
generation. Completed
stage checkpoints, `process_status.json`, and `run.log` are kept in the
output's sibling `.NAME.producer-checkpoints/` directory; that directory is
never a valid simulation input.
Producer-internal cache/checkpoint options affect only generation and are not
simulation fallbacks.

The default `jax` trace backend compiles the continuous Fourier--Zernike
metric evaluation, MAKEGRID cubic-spline evaluation, Cartesian-to-logical
field transform, and all RK4 substeps together. It shards only the leading
trajectory batch and replicates the read-only metric/field coefficients, so
there are no collectives between trajectories. By default all local JAX
devices participate. `--trace-device-count N` selects a local-device prefix
and `--trace-batch-size N` fixes the reusable compiled batch shape; padding is
masked and omitted from output. The `numpy` backend remains available as the
reference implementation and produces the same artifact contract. On GPU,
use a JAX installation with 64-bit support and an FP64-capable device.
The trace metadata records coefficient bytes per device, aggregate replicated
coefficient bytes, peak host RSS, and XLA argument/output/temporary-buffer
estimates. Temporary storage scales approximately with `--trace-batch-size`,
so reduce that option if the compiler estimate approaches available device
memory. The producer releases the device-resident coefficient state before
the polygon-overlap stage. On a shared GPU, setting
`XLA_PYTHON_CLIENT_PREALLOCATE=false` before launch avoids JAX reserving most
of the device up front; this is an environment policy rather than an artifact
or simulation setting.
The subsequent owner-overlap build remains interface-streamed: polygon and
quadrature temporaries are discarded after each eta interface, and completed
link dictionaries are compacted to arrays instead of accumulating Python
objects through the full torus.

Run a consumer by naming the bundle explicitly:

```bash
conda run -n drb python simulate_hsx_blob.py \
  --geometry /path/to/hsx-fci-32 \
  [simulation options]
```

The consumer deserializes the named bundle and lowers it for the requested
device layout. It does not validate physical quality, search for another
bundle, fit a metric, trace a map, rebuild overlap, or regenerate a missing
component. A missing or incompatible component therefore fails directly.

The manifest's SHA-256 values are informational integrity records. They are
never used to select, substitute, or rebuild geometry.
`audit_fci_simulation_geometry_checksums(path)` checks those records on demand;
the ordinary loader deliberately does not call it.

If an old run has usable cache files, the one-time migration utility can be
given those exact files explicitly:

```bash
conda run -n drb python convert_legacy_hsx_geometry_cache.py \
  --legacy-metric-cache /path/to/hsx_metric_old.npz \
  --legacy-map-cache /path/to/hsx_maps_old.npz \
  --makegrid /path/to/mgrid.nc --vessel /path/to/vessel.txt \
  --resolution 32 32 32 --metric-mesh-shape 32 32 32 \
  --output /path/to/hsx-fci-32
```

This command is producer-only. It stages copies of the named cache files,
rebuilds and qualifies any missing or incompatible artifact components, and
publishes a new bundle. Simulation never invokes it and never searches the
legacy cache directories.
