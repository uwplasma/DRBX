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

- JSON report with per-case fidelity and runtime metrics
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

Key current observations from the live rerun matrix:

- the most difficult one-step mismatch in the current selected set is still the
  neutral mixed lane, with worst RMS error normalized by reference amplitude
  about `9.17e-1` and native/reference wall-time ratio about `21.18`
- the heavy 1D recycling lanes are much tighter in fidelity but still slower
  than Hermès-3 on this machine:
  `recycling_1d_one_step` and `recycling_dthe_one_step` have worst normalized
  RMS errors about `4.62e-3` and `4.92e-3`, with runtime ratios about `5.35`
  and `9.53`
- the integrated 2D recycling and direct tokamak recycling one-step lanes are
  already around wall-time parity or faster on this machine, but still show
  visible bounded mismatch on the guarded compare surface, with worst
  normalized RMS errors about `1.79e-1` and `1.62e-1`
- the tokamak isothermal, tokamak transport, tokamak turbulence, and annulus
  electromagnetic one-step lanes are effectively exact on the current compare
  surface, and the compact tokamak transport/turbulence lanes are much faster
- runtime competitiveness is mixed rather than uniformly good:
  - heavy 1D recycling paths are still slower than Hermès-3 on this machine
  - integrated and direct tokamak one-step lanes can already be competitive or
    faster
  - compact tokamak transport/turbulence one-step lanes are substantially
    faster
- four cases in the current matrix are exact on the current compare surface:
  `tokamak_isothermal_one_step`, `tokamak_turbulence_one_step`,
  `tokamak_diffusion_transport_short_window`, and `annulus_he_emag_one_step`

The figure and report are intended to feed both the docs and future paper
comparison panels.
