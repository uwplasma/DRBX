# Blob2D Meeting Demo

This page documents a fast Blob2D visualization workflow. The full native `blob2d_short_window` run is still expensive enough that it is not the right default for quick meeting preparation. For immediate visualization, the example can read a saved portable `.npz` payload and render Matplotlib figures plus 2D/3D movies without rerunning the simulation.

## Fast Saved-Result Command

Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python examples/blob2d_meeting_demo.py \
  --arrays-in references/baselines/reference_arrays/blob2d_one_step.npz \
  --output-root docs \
  --skip-parity
```

The `--skip-parity` option is intentional for this quick movie path: `blob2d_one_step.npz` is not on the same output timeline as the committed `blob2d_short_window` parity metrics. The script still writes the saved analysis JSON, snapshots, a poster image, and movies.

## Outputs

- [analysis JSON](docs/data/blob2d_meeting_analysis.json)
- [parity-skipped JSON](docs/data/blob2d_meeting_parity_skipped.json)
- [snapshot panel](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__blob2d_meeting_snapshots.png)
- [movie poster](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__blob2d_meeting_movie_poster.png)
- [2D movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__movies__blob2d_meeting_2d.mp4)
- [3D movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__movies__blob2d_meeting_3d.mp4)

## Figures

![Blob2D meeting snapshots](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__blob2d_meeting_snapshots.png)

![Blob2D meeting poster](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__blob2d_meeting_movie_poster.png)

## How To Customize

The example script deliberately exposes separate functions for the main workflow:

- `build_demo_settings(...)` collects the case name, saved-array input path, density variable, background density, movie FPS, and output directories.
- `describe_requested_case(...)` prints the user-facing setup and the curated reference metadata when running a live case.
- `run_or_load_arrays(...)` either runs `jax_drb` or reads an existing portable `.npz` file.
- `create_plots_and_movies(...)` dispatches to the plotting layer.
- `create_blob2d_visualization_package_without_parity(...)` is the fast saved-payload path for quick movie generation.

For a full parity-oriented Blob2D movie, first add or generate a matching short-window `.npz` array payload, then run the same example without `--skip-parity`.
