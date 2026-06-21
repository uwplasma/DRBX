# Release Notes: 1.0.3

`jax_drb 1.0.3` is a release-readiness, documentation, artifact, and
solver-boundary release. It keeps the stable full-output recycling default on
the validated compatibility BDF path, promotes only the JAX/JVP surfaces that
have matching evidence, and makes the user examples self-contained through
release-backed media and compact fixtures.

## Highlights

- The README, examples, release-packaging docs, and canonical execution plan
  now state the same solver boundary: compact native solvers, selected
  operator gates, fixed-layout residual seams, and differentiable examples are
  promoted where their evidence says so; full output-window recycling still
  defaults to compatibility BDF, while JAX-linearized, sparse-JVP,
  fixed-BDF2, active-array, and matrix-free paths remain opt-in research gates.
- The private docs-media release bundle has been refreshed against
  `docs/release_artifacts_manifest.json`. The bundle now restores all `174`
  manifest media files, including the diverted-tokamak movie, 3D tokamak movie,
  compact stellarator FCI showcase, and imported-field QA-hybrid
  stationarity/Jacobi movie used by the README.
- `scripts/fetch_example_artifacts.py --skip-baselines --force` has been
  tested with an isolated root and cache and restored `174/174` manifest media
  files from the private release bundle.
- The artifact downloader now accepts `JAX_DRB_ARTIFACT_CACHE_DIR` for shared
  CI or cluster caches, preserves the older `JAX_DRB_ARTIFACT_CACHE` alias, and
  honors `JAX_DRB_ARTIFACT_DOWNLOAD_TIMEOUT` plus
  `JAX_DRB_ARTIFACT_DOWNLOAD_ATTEMPTS` in the HTTPS fallback path.
- The promoted solver/public-surface coverage gate now includes the meaningful
  recycling source, target, state, boundary, collision, reaction,
  JVP-promotion, runner, and integrated-recycling evidence layer rather than
  relying on smoke-only coverage.
- Focused recycling tests lock positive upstream feedback clamps, active/full
  source slicing, species source override mapping, sheath-energy and feedback
  source active-layout mapping, promoted zero fallback behavior, and mixed
  backward-Euler/BDF2 residual formulas.
- The package and public docs now use `1.0.3` release metadata because the
  `v1.0.2` tag is already a published historical release.
- A `CITATION.cff` file has been added for manuscript and software-citation
  workflows.

## Validation

The current release candidate passes the bounded closeout gate at `96.0%`
coverage with `88` focused release-surface tests and passes the promoted
native-solver/public-surface gate at `95.16%` coverage with `804` passed,
`14` skipped, `10` deselected, and `1` expected xfail on the local developer
machine. The fast bounded research-check wrapper also passes all default slices
locally, and `mkdocs build --strict --clean` passes with only existing
informational notices for excluded generated artifacts and external example
references.

The self-contained docs/example subprocess slice passes with `11` tests.
Representative user commands for the diverted-tokamak movie/profile,
model-selection guide, stellarator geometry, VMEC-extender import, and compact
nonlinear stellarator movie also ran locally without any external
reference-code install.

The latest footprint and package audit remains lightweight: no tracked or
reachable-history blobs exceed the configured `1 MiB` audit threshold, the
reachable git pack is about `6.43 MiB`, the wheel is about `709 KiB`, and the
sdist is about `614 KiB`.

Live-reference and large `all-gpu` campaigns remain manual self-hosted runs:
they require a valid reference checkout and CUDA-visible devices. Their
commands are exposed in the research-campaign workflow dispatch and tested
against the bundle script, but the retained release evidence should still be
read as committed-profile evidence rather than a blanket full-output-window
GPU speedup claim.

## Current Boundary

The full output-window recycling BDF default remains the stable
finite-difference compatibility path. JVP and JAX-linearized GMRES modes are
audited opt-in lanes for transformable residual surfaces; they should not yet
be described as a blanket end-to-end differentiable heavy recycling backend.

The compact stellarator and imported-field movie examples are validated as
reproducible release examples with connection-length, FCI, refinement, and
movie-QA support where documented. They are not promoted as device-scale
long-window predictive turbulence calculations for HSX, NCSX,
Landreman-Paul QA, or Dommaschk configurations.
