# JAXDRB Documentation

JAXDRB is a JAX-native edge and scrape-off-layer plasma code for
drift-reduced Braginskii models, electrostatic turbulence, neutral transport,
curated tokamak workflows, and reusable 3D geometry diagnostics.

The documentation is organized around the same validation and release surface
used by the test suite:

- [Native runtime CLI](native_runtime_cli.md) explains how to run decks,
  restarts, progress reporting, and output artifacts.
- [Physics models](physics_models.md) summarizes the governing equations,
  closures, boundary models, and current implementation status.
- [Validation gallery](validation_gallery.md) collects the public
  publication-ready figures and their backing campaign artifacts.
- [VMEC-extender edge field import](vmec_extender_edge_fields.md) documents the
  prescribed exterior-field NetCDF contract and JAX interpolation/RHS bridge.
- [Stellarator examples](stellarator_examples.md) explains the 3D stellarator
  geometry, linear mode, nonlinear turbulence, plotting, equations, and
  literature links behind the example scripts.
- [Research-grade validation matrix](research_grade_validation_matrix.md)
  separates primary scientific evidence from supporting engineering gates.
- [Profiling runtime](profiling_runtime.md) documents the runtime,
  differentiability, and profiling evidence used to guide performance work.
- [Release packaging](release_packaging.md) records the public packaging,
  CI/CD, PyPI, and artifact-release expectations.

The project README remains the shortest installation and quick-start path:
[README](https://github.com/uwplasma/jax_drb#readme).

Large rendered media are release-backed so that the repository stays small.
For the current private repository, unauthenticated readers may need repository
access to render linked release media on ReadTheDocs; local users can restore
the same docs media and self-contained example payloads with
`python scripts/fetch_example_artifacts.py --skip-baselines`, or restore the
full media plus reference-baseline set with
`python scripts/fetch_example_artifacts.py`.
