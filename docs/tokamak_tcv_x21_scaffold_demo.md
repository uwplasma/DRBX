# TCV-X21 Tokamak Scaffold Demo

This page documents the first honest 3D tokamak kickoff package in `jax_drb`.
It is intentionally labeled `scaffolded_reference_backed`: the package resolves
the manifest hook for `tokamak_tcv_x21_escalation`, records whether a local
TCV-X21 reference tree is present, and can generate a polished 3D benchmark
bundle from either:

- a real public TCV-X21 benchmark-data root containing `TCV_forward_field.nc`,
  `TCV_ortho.nc`, `snaps00000.nc`, and `vgrid.nc`;
- a real local reference workdir with dump files and mesh;
- or a tiny synthetic preview workdir when no external data root is available yet.

The committed artifact bundle in this repository is generated from the real
public benchmark-data mode. The synthetic preview mode still exists so the 3D
launch path remains testable when no external data root is available.

## Run It

Public benchmark-data mode, which is now the preferred reproducible path:

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/tcv-x21/scaffold_demo.py \
  --reference-root /path/to/reference-suite \
  --download-public-benchmark-data \
  --benchmark-data-root /tmp/tcv_x21_public_benchmark \
  --output-root docs/data/tokamak_tcv_x21_scaffold_artifacts
```

Preview mode, if no external benchmark or workdir tree is available:

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/tcv-x21/scaffold_demo.py \
  --output-root docs/data/tokamak_tcv_x21_scaffold_artifacts
```

If you have a local TCV-X21 checkout with a populated workdir and mesh, bind it
in explicitly:

```bash
PYTHONPATH=src .venv/bin/python examples/tokamak-3D/tcv-x21/scaffold_demo.py \
  --reference-root /path/to/reference-suite \
  --workdir-in /path/to/tcv-x21-workdir \
  --mesh-path /path/to/tcv-x21-workdir/tokamak.nc \
  --output-root docs/data/tokamak_tcv_x21_scaffold_artifacts
```

## Output Files

- manifest report: [tokamak_tcv_x21_scaffold_manifest.json](data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_manifest.json)
- input/deck report: [tokamak_tcv_x21_scaffold_input_report.json](data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_input_report.json)
- benchmark-data report: [tokamak_tcv_x21_scaffold_benchmark_data_report.json](data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_benchmark_data_report.json)
- validation contract: [tokamak_tcv_x21_scaffold_validation_contract.json](data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_validation_contract.json)
- observable report: [tokamak_tcv_x21_scaffold_observable_report.json](data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_observable_report.json)
- profile report: [tokamak_tcv_x21_scaffold_profile_report.json](data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_profile_report.json)
- profile arrays: [tokamak_tcv_x21_scaffold_profile_arrays.npz](data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_profile_arrays.npz)
- assembled arrays: [tokamak_tcv_x21_scaffold_arrays.npz](data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_arrays.npz)
- analysis JSON: [tokamak_tcv_x21_scaffold_analysis.json](data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_analysis.json)
- profile summary plot: [tokamak_tcv_x21_scaffold_profiles.png](data/tokamak_tcv_x21_scaffold_artifacts/images/tokamak_tcv_x21_scaffold_profiles.png)
- snapshot panel: [tokamak_tcv_x21_scaffold_snapshots.png](data/tokamak_tcv_x21_scaffold_artifacts/images/tokamak_tcv_x21_scaffold_snapshots.png)
- poster frame: [tokamak_tcv_x21_scaffold_poster.png](data/tokamak_tcv_x21_scaffold_artifacts/images/tokamak_tcv_x21_scaffold_poster.png)
- GIF: [tokamak_tcv_x21_scaffold.gif](data/tokamak_tcv_x21_scaffold_artifacts/movies/tokamak_tcv_x21_scaffold.gif)

## Preview

![TCV-X21 scaffold movie](data/tokamak_tcv_x21_scaffold_artifacts/movies/tokamak_tcv_x21_scaffold.gif)

![TCV-X21 scaffold profiles](data/tokamak_tcv_x21_scaffold_artifacts/images/tokamak_tcv_x21_scaffold_profiles.png)

## What This Package Does

1. resolves the `tokamak_tcv_x21_escalation` manifest entry;
2. records whether a local 3D reference tree is actually present;
3. parses the reference deck into a structured input report with time, mesh,
   solver, component, and compare-surface metadata when the deck is present;
4. writes a benchmark-data report that records which public TCV-X21 benchmark
   files are present and which sample field is being visualized;
5. writes a benchmark validation contract that records the planned TCV-X21
   observables, profile metrics, and promotion gates for the 3D lane;
6. writes a geometry-adapter observable report that lifts the named benchmark
   profile families onto the shared 3D observable schema;
7. extracts the staged `FHRP`, `LFS-LP`, and `HFS-LP` profile families from the
   public benchmark observable record into a structured report and compact NPZ bundle;
8. renders a publication-style profile summary figure from that same bundle;
9. renders a publication-style GIF, snapshot panel, and poster from the public
   sample geometry and snapshot files;
10. keeps the first 3D kickoff honest by labeling it as scaffolded/reference-backed.

## Benchmark Gate Design

The validation contract follows the same observable families used in the local
TCV-X21 helper workflow:

- `FHRP`: outboard-midplane density, temperature, potential, and floating-potential profiles
- `LFS-LP`: low-field-side target density, temperature, potential, current, and floating-potential profiles
- `HFS-LP`: high-field-side target density, temperature, potential, current, and floating-potential profiles

The immediate promotion gates are now:

1. scaffold gate: manifest, deck report, validation contract, and geometry figure bundle
2. external benchmark-data gate: the same artifact bundle driven by the public TCV-X21 benchmark data root
3. selected-field parity gate: a compact native/reference compare surface on a reduced 3D rung
4. benchmark validation gate: publication-ready TCV-X21 profile plots and methods note

## What It Does Not Do Yet

- it does not claim a native 3D tokamak solver path;
- it does not replace the future TCV-X21 execution lane;
- it does not depend on a heavy local 3D solve for the first public benchmark deliverable.
- it does not yet turn the deck report into a live native 3D run configuration.
