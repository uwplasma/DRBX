# Local release validation — 29 September 2026

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
