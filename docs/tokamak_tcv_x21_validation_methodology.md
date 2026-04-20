# TCV-X21 Validation Methodology

This page records the validation logic for one selected 3D tokamak benchmark
adapter in `jax_drb`. The current lane is still `scaffolded_reference_backed`,
so the goal here is not to overclaim a native 3D solver. The goal is to make
the gate design explicit before the native 3D solver surface is promoted and to
keep this benchmark package clearly separate from the general 3D architecture.

## External Context

The current gate design follows the benchmark and verification literature that
is most relevant to diverted edge/SOL turbulence:

- [TCV-X21 validation benchmark](https://arxiv.org/abs/2109.01618)
- [TCV-X21 turbulence validation follow-up](https://arxiv.org/abs/2506.12180)
- [Verification and validation procedures with applications to plasma-edge turbulence simulations](https://graphsearch.epfl.ch/fr/publication/676d35db-935c-423f-93e4-ac4b85683a4c)
- [Detachment scalings derived from 1D scrape-off-layer simulations](https://arxiv.org/abs/2406.16375)

These references motivate a layered gate structure:

1. code verification on reduced/operator slices
2. compact native/reference parity on selected fields
3. benchmark-facing observable comparisons
4. publication-ready uncertainty and methods reporting

This benchmark-specific methodology sits on top of the broader geometry plan in
[geometry_roadmap.md](geometry_roadmap.md). TCV-X21 is the first serious 3D
benchmark adapter, not the definition of the whole 3D code path.

The current scaffold implementation now consumes a shared 3D diagnostic-profile
layer for report building, NPZ export, and publication plotting. That is the
intended direction for future benchmark adapters too.

## Observable Families

The first selected 3D benchmark lane should be organized around the same
observable families used in the local TCV-X21 helper workflow:

- `FHRP`
  - outboard-midplane reciprocating-probe profiles
  - density
  - electron temperature
  - ion temperature
  - plasma potential
  - floating potential
- `LFS-LP`
  - low-field-side target Langmuir-probe profiles
  - density
  - electron temperature
  - ion temperature
  - plasma potential
  - parallel current
  - floating potential
- `HFS-LP`
  - high-field-side target Langmuir-probe profiles
  - density
  - electron temperature
  - ion temperature
  - plasma potential
  - parallel current
  - floating potential

## Gate Sequence

### 1. Scaffold Gate

Required now:

- manifest resolution
- deck/input report
- benchmark validation contract
- geometry/movie preview with LCFS, wall, and divertor overlays

This is the current in-tree state.

### 2. External Benchmark-Data Gate

Now landed in a reproducible public-data form:

- the same artifact bundle can be produced from a real public TCV-X21 benchmark-data root
- the profile-summary JSON is built from the public benchmark observable record
- the movie/poster/snapshot figures are built from the public sample geometry and snapshot files

This gate is still `scaffolded_reference_backed`, but it turns the 3D lane from
synthetic preview into an honest public benchmark-preparation path.

Still missing after that checkpoint:

- a reduced native 3D compare surface against the same benchmark families
- runtime/provenance summaries from an actual native 3D execution rung

### 3. Selected-Field Parity Gate

Required before any native 3D claim:

- a compact selected-field compare surface
- bounded native/reference parity tests
- restart/provenance summary on the selected reduced 3D rung

This is where the 3D lane first becomes comparable to the promoted 2D gates.

The reduced selected-field parity package now exists in-tree for compact
`Ne`/`Pe`/`phi` surfaces. On the tokamak lane, that reduced gate now also runs
from the real public TCV-X21 benchmark-data root, using the public sample
bundle as the reference side and a deterministic derived candidate as the
reproducible compact compare target. What is still missing is the first reduced
native 3D rung beyond that benchmark-backed gate or a real external
reference/candidate workdir pair on that same selected-field surface. The first
native reduced rung is now also in-tree as `tokamak_native_selected_field`,
which runs a promoted native tokamak one-step case on the same compact
surface and writes runtime/provenance metadata alongside the parity bundle. The
non-tokamak traced-field-line lane has already moved beyond synthetic-only
operation by running the same reduced gate from a real external FCI grid with a
deterministic derived candidate.

### 4. Benchmark Validation Gate

Required before publication-grade 3D benchmark claims:

- TCV-X21 observable package covering `FHRP`, `LFS-LP`, `HFS-LP`
- publication-ready profile comparisons
- profile-shape and absolute-level diagnostics
- a short methods note recording mesh, run time window, averaging window, and
  observable extraction methodology

## Verification Requirements

The 3D lane should not skip verification just because the benchmark is complex.
Before a native 3D rung is promoted:

- reduced operators should already have MMS/order-of-accuracy evidence
- selected-field parity should already be bounded on a reduced rung
- restart equivalence should already be demonstrated
- artifact provenance should already be written to the public bundle

## Local Source Mapping

The current scaffold implementation and validation package lives in:

- `src/jax_drb/validation/tokamak_tcv_x21_scaffold.py`
- `examples/tokamak-3D/tcv-x21/scaffold_demo.py`
- `tests/test_validation_tcv_x21_scaffold.py`

The current public artifact bundle lives in:

- `docs/data/tokamak_tcv_x21_scaffold_artifacts/`

The current geometry rendering path reused by the scaffold lives in:

- `src/jax_drb/validation/diverted_tokamak_movie.py`

The broader geometry direction for 3D support is tracked in:

- `docs/geometry_roadmap.md`
