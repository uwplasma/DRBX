# TCV-X21 Tokamak Scaffold Demo

This page documents the first honest 3D tokamak kickoff package in `jax_drb`.
It is intentionally labeled `scaffolded_reference_backed`: the package resolves
the manifest hook for `tokamak_tcv_x21_escalation`, records whether a local
TCV-X21 reference tree is present, and can generate a polished diverted-geometry
preview from either:

- a real local reference workdir with `BOUT.dmp.*.nc` files and `tokamak.nc`;
- or a tiny synthetic preview workdir when no 3D output tree is available yet.

The preview mode is the default in this repository. It exists so the 3D launch
path is testable now without pretending the full TCV-X21 solver lane is already
native-exact.

## Run It

Preview mode, which is the default if no external 3D workdir is provided:

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
- assembled arrays: [tokamak_tcv_x21_scaffold_arrays.npz](data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_arrays.npz)
- analysis JSON: [tokamak_tcv_x21_scaffold_analysis.json](data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_analysis.json)
- snapshot panel: [tokamak_tcv_x21_scaffold_snapshots.png](data/tokamak_tcv_x21_scaffold_artifacts/images/tokamak_tcv_x21_scaffold_snapshots.png)
- poster frame: [tokamak_tcv_x21_scaffold_poster.png](data/tokamak_tcv_x21_scaffold_artifacts/images/tokamak_tcv_x21_scaffold_poster.png)
- GIF: [tokamak_tcv_x21_scaffold.gif](data/tokamak_tcv_x21_scaffold_artifacts/movies/tokamak_tcv_x21_scaffold.gif)

## Preview

![TCV-X21 scaffold movie](data/tokamak_tcv_x21_scaffold_artifacts/movies/tokamak_tcv_x21_scaffold.gif)

## What This Package Does

1. resolves the `tokamak_tcv_x21_escalation` manifest entry;
2. records whether a local 3D reference tree is actually present;
3. reuses the existing diverted-tokamak geometry/movie pipeline;
4. renders a publication-style 2D GIF plus a poster frame with LCFS, wall, and
   divertor overlays;
5. keeps the first 3D kickoff honest by labeling it as scaffolded/reference-backed.

## What It Does Not Do Yet

- it does not claim a native 3D tokamak solver path;
- it does not replace the future TCV-X21 execution lane;
- it does not depend on a heavy 3D solve for the first visual deliverable.
