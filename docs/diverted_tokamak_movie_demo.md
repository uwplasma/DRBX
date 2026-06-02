# Diverted Tokamak Movie Demo

This self-contained demo generates a detailed 2D diverted tokamak GIF with:

- toroidally averaged field fluctuations on the full poloidal mesh
- LCFS overlay from `psixy = 0`
- wall and divertor target curves from `tokamak.nc`
- saved analysis JSON and assembled NPZ payloads for reuse

The current committed artifact is generated from the exact
`tokamak_turbulence_short_window` benchmark lane and stored in the private
release artifact bundle. Normal users do not need an external reference-code
checkout to run the demo or regenerate the PNG/GIF package.

## Run It

Restore the committed movie, figures, and arrays from the private release:

```bash
gh auth login --hostname github.com
python scripts/fetch_example_artifacts.py --skip-baselines
```

This is enough to inspect the README/docs GIF and the saved
`diverted_tokamak_turbulence_arrays.npz` payload. With the restored arrays in
place, `examples/diverted_tokamak_movie_demo.py` regenerates the PNG/GIF package
without launching a fresh external run:

```bash
PYTHONPATH=src python examples/diverted_tokamak_movie_demo.py
```

The example follows the SIMSOPT-style script pattern used by the 3D geometry
examples: edit the constants near the top of
`examples/diverted_tokamak_movie_demo.py`, then run the file. For the
self-contained path, leave `REFERENCE_ROOT = None`, keep
`USE_RELEASE_ARRAYS_IF_AVAILABLE = True`, and adjust `OUTPUT_ROOT`,
`FIELD_NAME`, `FPS`, or `FRAMES_PER_INTERVAL` if desired.

To analyze the same release-backed arrays with radial profiles, target
lineouts, RMS traces, and a final diverted-domain field map, run:

```bash
PYTHONPATH=src python examples/diverted_tokamak_profile_analysis_demo.py
```

## Optional Developer Reference Regeneration

The public example path above is self-contained. Developers who are refreshing
the validation bundle can still set `REFERENCE_ROOT` to a local reference-suite
checkout, set `USE_RELEASE_ARRAYS_IF_AVAILABLE = False`, and rerun the case from
fresh `tokamak.nc` and `BOUT.dmp.*.nc` files. That path is for maintaining the
benchmark artifact bundle; it is not required for users running JAXDRB examples.

## Output Files

- assembled arrays: [diverted_tokamak_turbulence_arrays.npz](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__data__diverted_tokamak_turbulence_arrays.npz)
- analysis JSON: [diverted_tokamak_turbulence_analysis.json](data/diverted_tokamak_turbulence_artifacts/data/diverted_tokamak_turbulence_analysis.json)
- snapshot panel: [diverted_tokamak_turbulence_snapshots.png](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__images__diverted_tokamak_turbulence_snapshots.png)
- poster frame: [diverted_tokamak_turbulence_poster.png](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__images__diverted_tokamak_turbulence_poster.png)
- GIF: [diverted_tokamak_turbulence.gif](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__movies__diverted_tokamak_turbulence.gif)
- profile analysis: `docs/data/diverted_tokamak_turbulence_artifacts/images/diverted_tokamak_turbulence_profiles.png`

## What The Script Does

1. restores or reuses the release-backed `diverted_tokamak_turbulence_arrays.npz`
2. reconstructs the diverted geometry and toroidally averaged field history
3. renders a snapshot panel, poster frame, and animated GIF with LCFS, wall, and divertor overlays
4. optionally runs the profile-analysis script to inspect radial profiles, target response, RMS traces, and the final geometry map

## Why This Is Useful

This closes a practical gap in the current 2D program:

- the README/docs can show the full diverted geometry without storing large media in git
- users can regenerate the visible figures and GIFs from a fresh clone
- developers can refresh the same artifact bundle separately when the reference validation suite is updated
