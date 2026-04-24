# Hermes Live Rerun Campaign

This campaign reruns a representative set of curated native cases and the
matching Hermès-3 reference cases on the same machine, then compares the live
native outputs against the live reference outputs on the guarded compare
surface.

The current case set is intentionally broad rather than narrow:

- 1D neutral mixed
- 1D recycling
- 1D multispecies D/T/He recycling
- integrated 2D recycling
- direct tokamak recycling
- tokamak isothermal transport
- tokamak turbulence
- tokamak diffusion/transport short window
- annulus electromagnetic
- Alfvén wave

The output bundle is:

- JSON report with per-case fidelity, runtime, and sampled peak-RSS metrics
- NPZ arrays for plotting and secondary analysis
- publication-grade PNG figure

The live campaign is meant to do two things at once:

1. strengthen the reviewer-facing code-to-code validation surface against
   Hermès-3
2. expose where the native solver is already competitive in wall time and where
   it still needs work

That is closer to the role played by cross-code figures in the TCV-X21 and
Hermès-3 validation literature than to a pure engineering dashboard. The point
is not only to show that a committed JSON exists; it is to show what the native
solver reproduces live, on the same machine, and where the remaining fidelity
or runtime gaps still are.

The current 3D evidence is still carried by the selected-field reference-backed
packages. Full live 3D Hermès reruns are not yet part of this matrix.

Generated artifact:

![Hermes live rerun campaign](data/hermes_live_rerun_campaign_artifacts/images/hermes_live_rerun_campaign.png)

Key current observations from the refreshed live rerun matrix:

- the hardest remaining fidelity lane in the current selected set is still the
  neutral mixed case:
  `neutral_mixed_one_step` has worst normalized RMS error about `9.17e-1`,
  runtime ratio about `2.93x`, and the dominant mismatch field is `NVh`
- the heavy 1D recycling ladders remain the main runtime gap, but their
  fidelity is still tight:
  - `recycling_1d_one_step`
    - worst normalized RMS error about `4.62e-3`
    - runtime ratio about `3.65x`
    - dominant normalized field `Pd+`
  - `recycling_dthe_one_step`
    - worst normalized RMS error about `4.92e-3`
    - runtime ratio about `7.82x`
    - dominant field `NVd`
- the integrated and direct tokamak recycling one-step lanes are no longer the
  main runtime concern:
  - `integrated_2d_recycling_one_step` runs at about `0.85x` the Hermès wall
    time on the same machine
  - `tokamak_recycling_one_step` runs at about `0.39x` the Hermès wall time
  - both still show bounded relative mismatch on the guarded compare surface,
    but the updated report now flags them as normalization-sensitive because
    the dominant field is near-zero `NVd`
  - the corresponding worst absolute max-errors are still small:
    `7.48e-12` and `3.09e-7`
- the compact tokamak and annulus lanes remain exact on the current guarded
  compare surface:
  `tokamak_isothermal_one_step`, `tokamak_turbulence_one_step`,
  `tokamak_diffusion_transport_short_window`, and `annulus_he_emag_one_step`
- the best current runtime ratios are now on the compact tokamak transport and
  turbulence lanes, with native/reference ratios down to about `1.2e-3`
- the refreshed report now samples process-tree peak RSS during each native
  and Hermès run:
  - the largest native peak is currently the integrated 2D recycling lane at
    about `722 MiB`
  - the largest native/Hermès peak-RSS ratio is currently
    `recycling_dthe_one_step` at about `0.95`
  - no lane in this matrix currently has a native peak RSS larger than the
    matching Hermès peak RSS, so the next memory work should be phase-resolved
    profiling rather than broad memory triage

The figure now includes both normalized error and absolute max-error. That is
important because the literature-facing interpretation is different for the two
main non-exact classes:

- the neutral mixed mismatch is a real fidelity problem and stays large even in
  absolute units
- the integrated/direct tokamak recycling one-step mismatch is currently better
  interpreted as a near-zero normalization artifact than as a large physical
  profile error

That distinction is critical for future paper use. The next paper-facing
comparison step should move from raw one-step field norms alone to more
physical tokamak observables such as target profiles, source/ionization
lineouts, and target flux summaries, in the same spirit as the TCV-X21,
SOLPS-ITER, and Hermès-3 comparison literature.

The figure and report are intended to feed both the docs and future paper
comparison panels.
