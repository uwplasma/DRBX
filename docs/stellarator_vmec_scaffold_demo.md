# Stellarator VMEC Scaffold Demo

This demo exercises the third 3D geometry adapter in the tree: a VMEC-style stellarator equilibrium scaffold on the same manifest, observable, figure, and movie infrastructure used by the tokamak and traced-field-line packages.

Run it with:

```bash
PYTHONPATH=src .venv/bin/python examples/geometry-3D/stellarator-vmec/scaffold_demo.py
```

The script is a SIMSOPT-style driver: edit `OUTPUT_ROOT` and
`EQUILIBRIUM_PATH` near the top of
`examples/geometry-3D/stellarator-vmec/scaffold_demo.py` to point at a VMEC
`wout*.nc` file or to change the artifact directory.

The scaffold writes:

- a geometry-adapter manifest;
- an input report with VMEC/source metadata;
- a validation contract for profile and sampled-surface gates;
- a radial profile bundle for `iota`, `pressure`, and `toroidal_flux`;
- a sampled `R`/`Z` flux-surface cross-section bundle with a summary summary figure and GIF;
- a shared observable report on the generic 3D schema.

Committed preview bundle:

- `docs/data/stellarator_vmec_scaffold_artifacts/data/stellarator_vmec_scaffold_manifest.json`
- `docs/data/stellarator_vmec_scaffold_artifacts/data/stellarator_vmec_scaffold_input_report.json`
- `docs/data/stellarator_vmec_scaffold_artifacts/data/stellarator_vmec_scaffold_validation_contract.json`
- `docs/data/stellarator_vmec_scaffold_artifacts/data/stellarator_vmec_scaffold_profile_report.json`
- `docs/data/stellarator_vmec_scaffold_artifacts/data/stellarator_vmec_scaffold_surface_report.json`
- `docs/data/stellarator_vmec_scaffold_artifacts/data/stellarator_vmec_scaffold_observable_report.json`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_vmec_scaffold_artifacts__images__stellarator_vmec_scaffold_profiles.png`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_vmec_scaffold_artifacts__images__stellarator_vmec_scaffold_surface_summary.png`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_vmec_scaffold_artifacts__images__stellarator_vmec_scaffold_surface_movie.gif`

This is still a scaffold package, not a native 3D execution claim. Its job is to prove that the public 3D diagnostics layer can support a third geometry family with different source data and different observable structure before the broader 3D program is widened further.
