# P05 direct raw-midpoint bracket qualification

This is a new, static accuracy campaign for the direct centered raw-midpoint
bracket on the pinned P05 HSX data. It compares two frozen candidates:

1. the pointwise centered expression
   `(-cross(b_cov/B, grad(a))) dot grad(f) / abs(J)` evaluated at every raw
   midpoint using the existing P05 structured reconstruction; and
2. that complete raw action plus the saved owner-normalized `U - A` jump.

Both candidates are projected to owners using every stored physical raw-cell
volume. The same saved analytic strong-bracket reference and owner volumes
are used for comparison. This campaign does not claim conservation, nonlinear
stability, production readiness, or an evolved-simulation result.

## Frozen coverage and decision rules

The input identity is pinned to the P05 producer commit recorded in
`reuse_bundle/reuse_manifest.json`. The campaign includes N32, N48, and N64,
all eight case/pair entries in `configuration.json`, and both candidates. The
seven nonconstant cases must each have observed RMS order at least 1.8 on
both refinement intervals for a candidate's global accuracy gate to pass.
The constant control, finite values, reconstruction support residual,
pointwise antisymmetry, and constant-gradient checks are reported as
implementation checks. Regional orders and error rebounds are descriptive;
they do not add a gate.

The historical actual-vorticity analytic reference has measured batch replay
noise. Its `1.6e-8` envelope applies only to that analytic-reference replay;
the smooth reference envelope is `1e-10`. Neither envelope applies to
reconstructed actions. The producer's exact-input O oracle is omitted wherever
its validity mask is false, including actual-vorticity B/C; missing oracle
values are not synthesized.

The bounded preflight samples eight deterministic angular locations at each
of the last six radial layers. For every selected owner it processes the
complete raw membership, including analytic reference replay, constant and
antisymmetry checks, and boundary-conditioned reconstruction checks. The
preflight covers every resolution before the global stage can start.

## Inputs and reuse

`input_manifest.json` under `scripts/p05_structured_global/` is the runtime
input authority. It requires nine files: the reference sidecar, six geometry
and topology arrays, the metric cache, and the MAKEGRID file. The file sizes
and SHA256 identities are recorded in `reuse_bundle/reuse_manifest.json`.
The geometry files may be supplied at their manifest paths, or at the exact
`prototype_runs/geometry/hsx_fci_NxNxN/` canonical paths; the runner verifies
the same bytes in either layout.

`reuse_bundle/reuse_inputs_v1.zip` contains the exact saved P05 owner
observations, reference, volumes, old centered result, saved `U - A` jump,
validity-masked exact-input O diagnostics, and bounded control data for each
resolution. The archive and every extracted array are SHA256 checked. It is
materialized inside the campaign output folder so the downloaded result is
self-contained for local analysis.

## Runner and resume contract

Run `campaign.py` from any directory. `verify-inputs` validates the committed
source identity, reuse bundle, and all nine runtime input hashes, then creates
the fresh output identity and materializes reuse arrays. `preflight` is
required before `run`. `run` writes restartable raw-index chunks; only the
parent process writes result/checkpoint files. `validate` requires complete
preflight and run markers, checks every raw index exactly once, writes owner
results and the frozen machine summary, and emits `completion.json`.

Use one output folder for all commands and resume. A checkpoint is reusable
only when its campaign identity, unit, and SHA256 receipt match. The runner
refuses a changed source/revision/input identity. Each command takes an
exclusive folder lock. The process pool is node-local and uses spawn; all
workers verify JAX's CPU backend. BLAS thread counts are fixed at one. The
remote allocation skill must select worker count and memory limits from the
actual allocated GPU node; the runner caps effective workers by the supplied
host budget, worker estimate, and reserve. One node and one writer are
supported; do not launch separate writers into the same folder.

Exact command sequence and remote input-transfer identities are in
[`handoff.md`](handoff.md). The handoff is computation-only and deliberately
stops after the prescribed operational validation; scientific interpretation
and any follow-up run remain local decisions.
