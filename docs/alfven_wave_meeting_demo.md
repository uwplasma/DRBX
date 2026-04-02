# Alfven-Wave Meeting Demo

This page collects a meeting-ready visual package generated from the native `alfven_wave_short_window` rung. It gives one reproducible command that produces publication-ready benchmark plots plus both 2D and 3D movies from a fast, stable electromagnetic case.

## Example Command

Run the demo from the repository root:

```bash
PYTHONPATH=src .venv/bin/python examples/alfven_wave_meeting_demo.py \
  --reference-root /Users/rogerio/local/hermes-3
```

By default this writes:

- [analysis JSON](/Users/rogerio/local/jax_drb/docs/data/alfven_wave_meeting_analysis.json)
- [parity JSON](/Users/rogerio/local/jax_drb/docs/data/alfven_wave_meeting_parity.json)
- [snapshot panel](/Users/rogerio/local/jax_drb/docs/images/alfven_wave_meeting_snapshots.png)
- [diagnostics plot](/Users/rogerio/local/jax_drb/docs/images/alfven_wave_meeting_diagnostics.png)
- [parity plot](/Users/rogerio/local/jax_drb/docs/images/alfven_wave_meeting_parity.png)
- [movie poster](/Users/rogerio/local/jax_drb/docs/images/alfven_wave_meeting_movie_poster.png)
- [2D movie](/Users/rogerio/local/jax_drb/docs/movies/alfven_wave_meeting_2d.mp4)
- [3D movie](/Users/rogerio/local/jax_drb/docs/movies/alfven_wave_meeting_3d.mp4)

## Figures

![Alfven-wave meeting snapshots](/Users/rogerio/local/jax_drb/docs/images/alfven_wave_meeting_snapshots.png)

![Alfven-wave meeting diagnostics](/Users/rogerio/local/jax_drb/docs/images/alfven_wave_meeting_diagnostics.png)

![Alfven-wave meeting parity](/Users/rogerio/local/jax_drb/docs/images/alfven_wave_meeting_parity.png)

![Alfven-wave meeting poster](/Users/rogerio/local/jax_drb/docs/images/alfven_wave_meeting_movie_poster.png)

## Why This Case

- It runs quickly enough to regenerate during a meeting-prep pass.
- The same transient supports both benchmark-grade diagnostics and readable 2D/3D visualizations.
- The committed short-window ladder already has exact summary and array parity on the current native scaffold.
