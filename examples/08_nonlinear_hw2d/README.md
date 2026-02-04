# Nonlinear HW2D milestone

Scripts for the 2D nonlinear HW-like drift-wave testbed used to validate operators and support
nonlinear-prep development.

- `hw2d_driftwave_turbulence.py`: baseline HW2D turbulence run (periodic, spectral operators).
- `hw2d_neutrals_effect.py`: HW2D run with minimal neutral exchange enabled.
- `hw2d_movie.py`: produces a short animated GIF for a fast nonlinear run.
- `hw2d_camargo1995_validation.py`: invariants + energy-budget closure checks (Camargo/Biskamp/Scott 1995-style).
- `mms_hw2d_convergence.py`: method of manufactured solutions (MMS) convergence check.
