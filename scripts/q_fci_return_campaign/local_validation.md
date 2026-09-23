# Local readiness evidence — 23 September 2026

This records preparation evidence, not global scientific qualification. No
full-domain numerical run was launched locally. The cancelled Q runner remains
under `work/q_fci_global_preflight_20260923/`; its supervisor and child were
stopped and partial outputs retained. This standalone replacement is under
`scripts/q_fci_return_campaign/`. It does not depend on historical workspace
imports at runtime.

## Independent historical replay and topology audit

The corrected endpoint-matrix numerical and exact-secant face fluxes were
replayed on the archived, actual HSX catalogues at all three resolutions.
Historical row selection was held fixed for this test; it is not confused with
the new canonical face-owned selector. The audit independently enumerated raw
cell faces and checked canonical topology incidence throughout each domain.

| N | Owners | Canonical faces | Max flux replay difference | Max action replay difference |
|---|---:|---:|---:|---:|
| 32 | 25,376 | 90,880 | 2.55e-20 | 1.05e-15 |
| 48 | 86,016 | 307,152 | 5.09e-21 | 3.06e-16 |
| 64 | 202,304 | 726,528 | 2.44e-21 | 9.72e-16 |

Fresh two-row traces and endpoint reconstructions at each resolution differed
from the archived observations by at most 4.63e-10. Different adaptive batch
sizes account for roundoff/tolerance-scale trajectory differences. All maps
replayed with the same potential chart and scales. Machine-readable evidence
is in `validation/historical_replay.json`; the local audit scripts preserve
provenance and are not remote entry points.

Five geometry-independent tests pass: incomplete/corrupt checkpoint handling,
source/coverage mismatch rejection, and duplicate-writer exclusion. They are
in `tests/test_q_fci_return_campaign.py`. Numerical checks use actual HSX data.

## Completed N32 canonical preflight

Seven complete owners span the original ordinary/agglomerated/transition tracks,
the collapsed-axis ring, a wall owner, its inward neighbor, and periodic seams.
They cover 139 canonical faces and 2,276 traced observations. Thirty-seven
footprints were excluded by the conservative interior-domain admissibility
rule. Every recipient owner and its complete incident-face action remains.

All 138 non-wall face maps have rank 19. Maximum scaled condition is 1.34e6 and
maximum target-relative defect is 1.21e-12; no support expansion was needed in
this bounded sample. Maximum constant response is 2.79e-13, and maximum exterior
balance residual across all channels is 8.05e-21. Saved weights reproduce saved
fluxes exactly after reload. Independent raw-cell incidence reproduces complete
actions within 1.78e-17. A changed linear combination of fields reuses the
saved geometry map successfully.

The numerical RMS for radial/angular/mixed fields is
`[1.9652e-3, 6.4684e-3, 3.4372e-3]` on the seven-owner sample. This sample and the
canonical support differ from the historical three-owner test; these errors
are not a global norm or a local-order acceptance gate. The q9/q11 face plus
q5/q7 volume reference differences are below 0.015% of these errors. Independent
strong-volume q9 checks at the ordinary and wall owners retain a larger
quadrature discrepancy (ordinary angular about 3.87e-5), so the close face-rule
agreement must not be presented as a rigorous reference bound.

A deliberate interruption followed by a worker-count change reused 44 completed
trace checkpoints without rewriting their payloads. Source, input and numerical
identities were unchanged. P's unrelated computation was left running.

## Measured cost and resource implications

N32 tracing consumed about 1,599 accumulated worker seconds (27 CPU-minutes).
Mean measured costs were 0.702 seconds/trace row, 0.204 seconds/face including
reference integrations, and 0.094 seconds/raw-cell continuous-volume integration.
Peak measured N32 worker RSS was 0.75 GiB. Boundary/seam batches are much slower
than typical interior batches: rejection and recursive batch splitting can
repeat substantial ODE work. Fresh tracing dominates the bounded preflight.

The deduplicated global workloads are:

| N | Trace rows | Trace units | Face units | Volume units |
|---|---:|---:|---:|---:|
| 32 | 65,536 | 2,752 | 1,420 | 1,024 |
| 48 | 221,184 | 9,216 | 4,800 | 3,456 |
| 64 | 524,288 | 21,888 | 11,352 | 8,192 |

Using the N32 sample uniformly gives approximately 19/64/151 accumulated worker
hours for N32/N48/N64, before allocation scaling and startup/I/O overhead.
These are planning estimates with at least factor 2–3 uncertainty; the sample
overrepresents axis/wall cases, while unsampled global exceptions can add cost.
They are not walltime predictions. Measured checkpoint bytes suggest order-GB
returned data, plus dense row catalogues, action arrays, logs and safety margin.
The remote setup skill must choose allocation, concurrency, memory headroom and
walltime from its environment; do not hard-code the local worker count.

## Final local scope and remote prerequisite

The user explicitly stopped the remaining local N48/N64 preflight because of
its cost. N32 is complete. N48 has 40 preserved trace chunks and no completed
face or volume stage; N64 was not started. These partial outputs are not a
qualification result. Cancellation stopped the exact parent and all four
workers, released the writer lock, and left P's computation untouched.

**Ready for remote preflight followed by the guarded global computation.**
This does not claim that the three-resolution canonical preflight has passed.
The remote `preflight` command must freshly complete N32/N48/N64 before `run`
will launch any global work. If a refinement reveals an unresolved numerical
case, return that evidence for local review; no remote numerical tuning is
authorized. Local partial checkpoints are retained locally and are not part
of remote checkpoint reuse.

The bundled `validation/` evidence includes the N32 summary, selection, action
arrays and completion receipt; all-resolution historical replay; independent
N32 reload/incidence and cost audit; and the local cancellation receipt. Full
local checkpoints remain in `work/q_fci_handoff_preflight_v2_20260923/` relative
to the research workspace. All outputs from the remote execution must remain
in its own uniquely named campaign folder.
