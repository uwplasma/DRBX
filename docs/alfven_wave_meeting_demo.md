# Alfven-Wave Meeting Demo

This page collects a meeting-ready visual package generated from the native `alfven_wave_short_window` rung. It gives one reproducible command that produces detailed benchmark plots plus both 2D and 3D movies from a fast, stable electromagnetic case.

## Example Command

Run the demo from the repository root:

```bash
PYTHONPATH=src .venv/bin/python examples/alfven_wave_meeting_demo.py \
  --reference-root /path/to/reference-suite
```

To regenerate plots and movies from an existing `.npz` payload without rerunning the case:

```bash
PYTHONPATH=src .venv/bin/python examples/alfven_wave_meeting_demo.py \
  --arrays-in https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__alfven_wave_short_window_native.npz \
  --output-root docs
```

By default this writes:

- [analysis JSON](docs/data/alfven_wave_meeting_analysis.json)
- [parity JSON](docs/data/alfven_wave_meeting_parity.json)
- [snapshot panel](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__alfven_wave_meeting_snapshots.png)
- [diagnostics plot](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__alfven_wave_meeting_diagnostics.png)
- [parity plot](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__alfven_wave_meeting_parity.png)
- [movie poster](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__alfven_wave_meeting_movie_poster.png)
- [2D movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__movies__alfven_wave_meeting_2d.mp4)
- [3D movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__movies__alfven_wave_meeting_3d.mp4)

## Figures

![Alfven-wave meeting snapshots](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__alfven_wave_meeting_snapshots.png)

![Alfven-wave meeting diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__alfven_wave_meeting_diagnostics.png)

![Alfven-wave meeting parity](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__alfven_wave_meeting_parity.png)

![Alfven-wave meeting poster](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__alfven_wave_meeting_movie_poster.png)

## Why This Case

- It runs quickly enough to regenerate during a meeting-prep pass.
- The same transient supports both benchmark-grade diagnostics and readable 2D/3D visualizations.
- The committed short-window ladder already has exact summary and array parity on the current native scaffold.
- The example is intentionally tutorial-style: setup, run/load, `.npz` saving, payload summaries, plotting, and artifact reporting are separate functions users can copy and customize.
