# Balanced-28 campaign preparation — 29 September 2026

The current runner is the always-balanced28 comparison candidate, with26 fields
and the same GPU RK4-64 tracing contract. It is prepared locally; a published
revision and successful remote gates are required before global qualification.

## Candidate extraction replay

`local_balanced_validate.py` compares signed complete-owner N-O against the
previous bounded support experiment, without importing its implementation.
All216 owners,812 raw members,26 fields and both spans were replayed for both
compact and balanced selections. Maximum absolute discrepancy was1.63e-11 for
compact and1.73e-12 for balanced. Every balanced plane uses exactly28 donors;
maximum moment reproduction residual was6.35e-14. See
`validation/balanced_replay.json`.

The source selection rule is copied from the tested geometry-only candidate;
no optimization against these fields has been introduced. The original18 field
indices remain stable and8 previously bounded fresh orientations are appended.

## Focused automated checks

Eight tests passed in `tests/test_q_fci_layered_campaign.py`. They cover atomic
checkpoint corruption/identity checks, complete-owner chunking, resource caps,
analytic gradients including all fresh controls, catalogue stability, N-O-R
reduction and maximum orders/worst-owner IDs. Canonical-HSX tests independently
check quartic moment reproduction, unchanged outer rows and rejection when no
exactly28-donor balanced set passes. The two input-dependent HSX tests skip in
CI environments lacking the research inputs; both ran locally.

## CPU numerical preflight

The expanded test-mode preflight covers five phase locations including the seam,
known hotspot/rank-repair rings, the axis, both sides of the aggregate join, and
wall layers. It evaluates all26 fields, both D/N and h/16+h/32 through the actual
campaign worker. All eight canonical inputs are SHA256 verified. Local tracing
uses CPU64 with CPU256 checks, not a GPU claim. The final receipt is stored in
`validation/balanced_preflight.json`; exhaustive all-location geometry and GPU
checks are required remotely.

All three numerical preflights passed.

| N | Owners | Raw cells | Constant action max | CPU64/256 cell-scaled difference | Peak worker GiB |
|---|---:|---:|---:|---:|---:|
| 32 | 75 | 320 | 1.051e-12 | 3.908e-14 | 0.793 |
| 48 | 75 | 440 | 9.294e-12 | 7.461e-14 | 0.914 |
| 64 | 80 | 570 | 1.708e-11 | 8.527e-14 | 1.588 |

Every inspected inner plane selected exactly28 donors. The extra candidate-pool
expansion levels were not needed on this preflight panel. A repeated N32 score
retained its checkpoint checksum (`validation/balanced_resume.json`). The
geometry-only worker was separately exercised on axis, last-aggregate and wall
owners at all three N (`validation/balanced_geometry_smoke.json`); this is a
bounded branch check, not the required exhaustive remote geometry gate.

## Scope of evidence

The bounded short-wave response experiments show both attenuation and phase
errors for balanced support and possible amplification for compact support.
This campaign tests global convergence and regional errors, not turbulence-scale
resolution or time stability. Keep the known O-R limitations separate from N-O.
No source changes to P or production runtime are part of this campaign.

The following section records the completed extraction checks for the original
compact implementation. Those numbers describe that baseline, not the new
balanced campaign's runtime or coverage.

---

# Historical compact campaign validation — 29 September 2026

This is a portable research campaign ready for remote preflight. It is not a
completed global scientific qualification or a production promotion.

## Frozen extraction replay

`local_validate.py` compared the packaged implementation against the bounded
research artifacts: 4,564 comparisons passed. Interior actions were bitwise
identical; maximum absolute differences were 4.35e-11 for wall actions,
2.91e-13 for O, and 4.14e-11 for R. See `validation/replay.json`.

Evidence behind the selected method remains in the workspace reports
`work/q_fci_rank_outer_comparison_20260929/report.md` and
`work/q_fci_layered_dn_qualification_20260929/report.md`. These are evidence,
not runtime dependencies; the portable campaign imports none of them.

## Complete-owner local CPU preflight

Python 3.12, JAX/jaxlib 0.9.2. The release used `--test-mode`, raw chunks of512,
and `--backend cpu-test`; all18 fields, D/N, and both spans were scored.
Initialization verified all eight canonical input hashes. All numerical,
coverage, constant-action and finite checks passed at all three resolutions.

| N | Owners | Raw cells | RK comparison samples | Maximum constant action | RK64/256 cell-scaled difference |
|---|---:|---:|---:|---:|---:|
| 32 | 45 | 192 | 30 | 1.05e-12 | 6.32e-13 |
| 48 | 45 | 264 | 29 | 9.32e-12 | 7.46e-14 |
| 64 | 45 | 336 | 29 | 1.72e-11 | 8.53e-14 |

The receipt key `cpu_gpu_scaled_error` comes from the common validator. In these
local `backend: cpu-test` receipts it compares two CPU evaluations; it is **not
GPU validation**. Actual GPU64/CPU64 agreement remains mandatory remotely.

A repeated N32 score reused its checkpoint without changing its SHA256
(`validation/resume.json`). Five focused pytest tests passed, covering complete
owner batching, corruption/source identity rejection, worker resource caps,
analytic field gradients and N/O/R reduction bookkeeping:

```bash
python -m pytest -q tests/test_q_fci_layered_campaign.py
```

## Required remote gates

No GPU was available locally. Actual multi-GPU execution and exhaustive
geometry coverage have not been run locally. The pinned launcher requires
parallel exhaustive geometry/rank/wall-system checks, GPU64 versus CPU64 and
CPU256 checks, numerical preflight, and a throughput pilot before global
scoring. These gates must pass unchanged; failures return for local repair.
Local preflight timings are not remote throughput estimates. Global accuracy,
regional rebounds and the known O-R reference limitation are to be assessed
locally from returned N/O/R arrays.
