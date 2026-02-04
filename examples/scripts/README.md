# Example scripts

This folder contains runnable scripts organized by **topic** (not by “difficulty”).

The intent is:

- make it easy to find an example for a given module (geometry, sheath BCs, literature workflows, …),
- keep scripts short and hackable,
- ensure every run produces informative plots and prints progress.

## Folders

- `01_linear_basics/`: quick linear scans (slab/tokamak/s-α, kx–ky, EM, hot ions).
- `02_geometry/`: tabulated geometry I/O and sanity checks.
- `03_sheath_mpse/`: open-field-line MPSE/sheath closures (Bohm, Loizu 2012, heat/SEE).
- `04_closures_transport/`: parallel closures, Braginskii/Spitzer scalings.
- `05_jax_autodiff/`: JAX differentiation/optimization workflows.
- `06_literature_tokamak_sol/`: literature-aligned workflows (Mosetto/Halpern/Jorge).
- `07_essos_geometries/`: ESSOS-driven stellarator/VMEC/Biot–Savart geometry examples.
- `08_nonlinear_hw2d/`: nonlinear HW2D milestone (turbulence, neutrals, MMS, movie).

## Conventions

- Prefer explicit `--out out_<name>/` style outputs to avoid overwriting results.
- Figures should be ready to drop into slides/notes without manual cleanup.
- If an example depends on optional packages (e.g. ESSOS), it should fail gracefully with a clear message.

