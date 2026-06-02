# Diverted Tokamak Movie Demo

This demo generates a detailed 2D diverted tokamak GIF with:

- toroidally averaged field fluctuations on the full poloidal mesh
- LCFS overlay from `psixy = 0`
- wall and divertor target curves from `tokamak.nc`
- saved analysis JSON and assembled NPZ payloads for reuse

The current committed artifact is generated from the exact `tokamak_turbulence_short_window` benchmark lane. The figure is therefore benchmark-backed, not a claim that the full-domain diverted tokamak transient is already promoted as a native exact lane.

## Run It

Restore the committed movie, figures, and arrays from the private release:

```bash
gh auth login --hostname github.com
python scripts/fetch_example_artifacts.py --skip-baselines
```

This is enough to inspect the README/docs GIF and the saved
`diverted_tokamak_turbulence_arrays.npz` payload. With the restored arrays in
place, `examples/diverted_tokamak_movie_demo.py` regenerates the PNG/GIF package
without launching a fresh external run. A fresh benchmark rerun requires the
external reference suite described below.

Fresh benchmark run:

```bash
export JAX_DRB_REFERENCE_ROOT=/path/to/hermes-3
export JAX_DRB_REFERENCE_BINARY=/path/to/hermes-3/build/hermes-3
PYTHONPATH=src python examples/diverted_tokamak_movie_demo.py
```

The example follows the SIMSOPT-style script pattern used by the 3D geometry
examples: edit the constants near the top of
`examples/diverted_tokamak_movie_demo.py`, then run the file. Set
`REFERENCE_ROOT` to a reference-suite checkout, `WORKDIR_IN` to an existing
work directory with `BOUT.dmp.*.nc` files when you want to reuse a kept run,
`MESH_PATH` to a specific `tokamak.nc` when the work directory does not contain
one, `OUTPUT_ROOT` to the artifact directory, and `FIELD_NAME` to the saved
field to render. The default field is `phi`, which gives the clearest
diverted-geometry fluctuation movie on the current exact turbulence rung.
Set `USE_RELEASE_ARRAYS_IF_AVAILABLE = False` when you explicitly want to
ignore the restored arrays and force a fresh reference-backed run.

The reference root is a local external benchmark checkout that contains both
`tests/integrated` and `examples/tokamak-2D`. For the default movie case, the
mesh is expected at
`$JAX_DRB_REFERENCE_ROOT/examples/tokamak-2D/tokamak.nc`, and the fresh run
creates `BOUT.dmp.*.nc` files by launching the curated
`tokamak_turbulence_short_window` reference case. If the auto-discovery helper
cannot find such a checkout, set `JAX_DRB_REFERENCE_ROOT` explicitly before
running the example.

## Output Files

- assembled arrays: [diverted_tokamak_turbulence_arrays.npz](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__data__diverted_tokamak_turbulence_arrays.npz)
- analysis JSON: [diverted_tokamak_turbulence_analysis.json](data/diverted_tokamak_turbulence_artifacts/data/diverted_tokamak_turbulence_analysis.json)
- snapshot panel: [diverted_tokamak_turbulence_snapshots.png](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__images__diverted_tokamak_turbulence_snapshots.png)
- poster frame: [diverted_tokamak_turbulence_poster.png](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__images__diverted_tokamak_turbulence_poster.png)
- GIF: [diverted_tokamak_turbulence.gif](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__movies__diverted_tokamak_turbulence.gif)

## What The Script Does

1. launches or reuses the curated `tokamak_turbulence_short_window` benchmark case
2. stitches the multi-rank `BOUT.dmp.*.nc` files into one full-domain field history
3. reduces the 3D field to a toroidally averaged 2D fluctuation history
4. loads `Rxy`, `Zxy`, and `psixy` from `tokamak.nc`
5. renders a snapshot panel, poster frame, and animated GIF with LCFS, wall, and divertor overlays

## Why This Is Useful

This closes a practical gap in the current 2D program:

- the exact direct tokamak parity lanes are currently rank-local for native compare surfaces
- summary figures need the stitched full diverted geometry
- this script turns the same validated benchmark output into a summary geometry figure without pretending that the whole direct tokamak recycling transient is already a claim-bearing native exact lane
