# Examples

All runnable scripts live directly under:

`examples/`

and are organized by **topic** (not by “difficulty”). Each subfolder contains a `README.md`
describing what the scripts do and what figures to expect.

## Folders

- `examples/01_linear_basics/`: quick linear scans (slab/tokamak/s–α, kx–ky, EM, hot ions)
- `examples/02_geometry/`: tabulated geometry I/O and sanity checks
- `examples/03_sheath_mpse/`: open-field-line MPSE/sheath closures (Bohm, Loizu 2012, heat/SEE)
- `examples/04_closures_transport/`: parallel closures, Braginskii/Spitzer scalings
- `examples/05_jax_autodiff/`: JAX differentiation/optimization workflows
- `examples/06_literature_tokamak_sol/`: literature-aligned workflows (Mosetto/Halpern/Jorge)
- `examples/07_essos_geometries/`: ESSOS-driven stellarator/VMEC/Biot–Savart geometry examples
- `examples/08_nonlinear_hw2d/`: nonlinear HW2D milestone (turbulence, neutrals, MMS, movie)
- `examples/09_fci/`: flux-coordinate independent (FCI) preparation milestone (field-line maps + parallel operators)
- `examples/10_verification/`: verification examples (elliptic solves, MMS, conservation checks)

## Running

From the repository root:

```bash
python examples/01_linear_basics/slab_ky_scan.py
```

Most scripts write a small results folder under `out/` (or a user-provided `--out` path) containing:

- `.npz` outputs with raw scan data,
- publication-friendly `.png` plots.
