# Diverted Tokamak Movie Demo

This demo generates a detailed 2D diverted tokamak GIF with:

- toroidally averaged field fluctuations on the full poloidal mesh
- LCFS overlay from `psixy = 0`
- wall and divertor target curves from `tokamak.nc`
- saved analysis JSON and assembled NPZ payloads for reuse

The current committed artifact is generated from the exact `tokamak_turbulence_short_window` benchmark lane. The figure is therefore benchmark-backed, not a claim that the full-domain diverted tokamak transient is already promoted as a native exact lane.

## Run It

Fresh benchmark run:

```bash
PYTHONPATH=src .venv/bin/python examples/diverted_tokamak_movie_demo.py
```

The example follows the SIMSOPT-style script pattern used by the 3D geometry
examples: edit the constants near the top of
`examples/diverted_tokamak_movie_demo.py`, then run the file. Set
`REFERENCE_ROOT` to a reference-suite checkout, `WORKDIR_IN` to an existing
work directory with `BOUT.dmp.*.nc` files when you want to reuse a kept run,
`OUTPUT_ROOT` to the artifact directory, and `FIELD_NAME` to the saved field to
render. The default field is `phi`, which gives the clearest diverted-geometry
fluctuation movie on the current exact turbulence rung.

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
