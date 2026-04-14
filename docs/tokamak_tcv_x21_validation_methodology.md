# TCV-X21 Validation Methodology

This page records the validation logic for the selected 3D tokamak kickoff lane
in `jax_drb`. The current lane is still `scaffolded_reference_backed`, so the
goal here is not to overclaim a native 3D solver. The goal is to make the gate
design explicit before the native 3D solver surface is promoted.

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

### 2. External Workdir Gate

Required next:

- same artifact bundle produced from a real external workdir and mesh
- profile-summary JSON built from actual 3D output
- reproducible publication-style figures from that workdir

This gate is still `scaffolded_reference_backed`, but it turns the 3D lane from
synthetic preview into an honest benchmark-preparation path.

### 3. Selected-Field Parity Gate

Required before any native 3D claim:

- a compact selected-field compare surface
- bounded native/reference parity tests
- restart/provenance summary on the selected reduced 3D rung

This is where the 3D lane first becomes comparable to the promoted 2D gates.

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
