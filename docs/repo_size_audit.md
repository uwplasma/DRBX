# Repository Size Audit

Last local audit: 2026-04-28.

## Current Footprint

The lightweight tree keeps source, JSON reports, tests, and validation logic in
git, while large generated arrays, figures, movies, traces, and profile files
live in the GitHub release
[validation-artifacts-2026-04-28](https://github.com/uwplasma/jax_drb/releases/tag/validation-artifacts-2026-04-28).

Current tracked checkout size after removing release-backed artifacts:

- tracked checkout: about `7M`;
- tracked `docs/data`: about `704K`, mostly JSON reports;
- tracked `references`: about `404K`, mostly JSON summaries and metrics;
- release-backed reference baseline bundle: about `24M`;
- release-backed docs/media bundle: about `31M`.

The local `.git` directory remains large until the history rewrite is applied
because earlier commits still contain generated `.npz`, media, and trace blobs.
After `git filter-repo` and garbage collection, the clone-relevant history
should be dominated by source, tests, JSON metadata, and documentation text.

## Release-Backed Artifacts

The docs and README render images and movies from path-encoded release assets,
for example:

```text
https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__pytree_drb__images__stellarator_drb_pytree_campaign.png
```

The full manifest is tracked as
[release_artifacts_manifest.json](release_artifacts_manifest.json). Heavy
test baselines are stored in
[jax_drb_reference_baselines.zip](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/jax_drb_reference_baselines.zip).

Tests call `ensure_reference_baselines()` through `tests/conftest.py`. On a
lightweight clone, this restores ignored `.npz` baselines under
`references/baselines/reference_arrays/` and
`references/baselines/reference_snapshots/` from the release. Local developers
can set `JAX_DRB_OFFLINE_ARTIFACTS=1` to require preexisting artifacts, or
`JAX_DRB_ARTIFACT_CACHE=/path/to/cache` to choose the download cache.

## Rewrite Target

The history rewrite should remove generated blobs from all earlier commits:

```bash
git filter-repo --force \
  --path references/baselines/reference_arrays \
  --path references/baselines/reference_snapshots \
  --path docs/data \
  --path docs/images \
  --path docs/movies \
  --path docs/runtime_precision_benchmark/images \
  --invert-paths
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force-with-lease origin main
```

This intentionally keeps source files, tests, JSON reference summaries,
validation reports, and documentation text in git, while release assets retain
the publication figures, movies, and heavyweight baselines.
