# Repository Size Audit

Last local audit: 2026-04-28.

## Current Footprint

The current working tree is large because it contains local development
artifacts, but the clone-relevant size is the tracked checkout and `.git`
object database:

- full working tree: about `1.5G`, dominated by local `.venv` and ignored
  profile/cache directories;
- `.git`: about `73M`;
- packed history: about `62M`;
- tracked checkout: about `58M`;
- tracked `docs/data`: about `37M`;
- tracked `references`: about `26M`.

The new non-axisymmetric validation artifacts are kept intentionally smaller:
publication-facing PNG/GIF/JSON outputs remain in
`docs/data/stellarator_fci_validation_artifacts/`, while generated `.npz`
arrays under that directory are ignored because they are reproducible from
`examples/geometry-3D/stellarator-fci/validation_campaign_demo.py`.

## Largest Current History Blobs

The largest clone-size offenders are generated baselines and media files:

- `references/baselines/reference_arrays/annulus_he_emag_short_window.npz`,
  about `4.6M`;
- `docs/data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_arrays.npz`,
  about `3.6M`;
- `references/baselines/reference_snapshots/tokamak_turbulence_short_window_field_history.npz`,
  about `3.2M`;
- `references/baselines/reference_arrays/alfven_wave_medium_window.npz`,
  about `2.2M`;
- `references/baselines/reference_arrays/tokamak_turbulence_short_window.npz`,
  about `2.2M`;
- `docs/data/tokamak_tcv_x21_toroidal_movie_artifacts/movies/tokamak_tcv_x21_toroidal.gif`,
  about `2.1M`;
- `docs/data/diverted_tokamak_turbulence_artifacts/movies/diverted_tokamak_turbulence.gif`,
  about `2.0M`.

## Slimming Strategy

The safe near-term policy is to keep small JSON reports and essential
publication images in the repository, but move heavyweight `.npz`, trace,
profile, and long-movie artifacts to release assets or a separate artifact
store. Tests should use compact analytic baselines, generated fixtures, or
explicit opt-in downloads for heavy reference arrays.

The highest-impact history rewrite is therefore:

1. Move heavyweight baseline arrays and snapshots out of the tracked tree.
2. Replace tests that require those files by compact generated fixtures or
   opt-in artifact downloads.
3. Rewrite history to remove the old blobs.
4. Force-push only after the implementation branch is committed and a backup
   branch/tag exists.

Candidate rewrite command, after the tree is clean and backed up:

```bash
git filter-repo --force \
  --path references/baselines/reference_arrays \
  --path references/baselines/reference_snapshots \
  --path docs/data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_arrays.npz \
  --path docs/data/tokamak_tcv_x21_toroidal_movie_artifacts/movies \
  --path docs/data/diverted_tokamak_turbulence_artifacts/movies \
  --path docs/movies \
  --invert-paths
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force-with-lease origin main
```

This is expected to cut the `.git` object database by well over half if the
large baseline directories are also removed from the current tree. It should
not be run from an active dirty implementation checkout because it rewrites
commit IDs and makes conflict recovery harder.

## Current Decision

Do not rewrite history until the current 3D PyTree/JVP/GPU profiling lane is
committed. The immediate low-risk cleanup already done in this branch is:

- ignore local profile, trace, cache, and XLA dump directories;
- ignore generated non-axisymmetric validation `.npz` arrays under `docs/data`;
- keep only publication-facing non-axisymmetric JSON/PNG/GIF artifacts in the
  working tree.
