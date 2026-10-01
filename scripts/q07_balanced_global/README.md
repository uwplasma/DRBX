# Paired h/32–h/128 geometry-consistent Q07 material campaign

Compare the **balanced** tube at total inner cap separation h/32 and h/128,
with outer characteristic separations h/16 and h/64 respectively, h=2*pi/N.
Both use `D_bal(F) = D_old(F) + [div(b) - D_old(1)] F_center`.
Neither assumes div(B)=0. Delta in the five-point stencil is half the inner
separation: offsets (-2,-1,0,1,2)*delta. Shortening scales both derivative legs.

All complete owners at N32/N48/N64; 22 states (the previous 18 plus four held-out
waves at wavelengths 0.7/0.35 and orientations 10/110 degrees); four D/physical-
normal N/mixed boundary combinations; nine regions. Thirty outputs cover each
span's centered/correction/combined n, Te, Ti, Vi and Ve actions. Diffusion,
vorticity, current–phi/SAT, evolution and production promotion are outside scope.

## Reuse and reference contract

The source bundle, canonical geometry, original GPU RK4-64 h/32+h/16 endpoints,
frozen gradient-repair choices, polynomial degrees, eta support and all donor
policies are unchanged. `span_rows.py` calls the same frozen low-level builders
at all ten query points together. It does not relabel prepared-operator metadata.
Only the additional h/128+h/64 endpoints need tracing: CPU-batched RK4 with 64
steps, in the same parallel worker pool, saved with checksums and identities.
No exterior endpoint, crossing or re-entry is accepted; exterior ghosts remain
unqualified. New traces are not interpolations of the old endpoints.

N uses reconstructed slot values; O uses exact manufactured values at the same
slots and the same tube/correction algebra; R is the unchanged continuum material
reference. Every raw member contributes to the complete-owner physical-volume
projection. N-O/O-R/N-R stay separate. h/128 is a candidate, not a default.
The prior bounded improvement mostly followed geometry-dependent transverse
motion along the actual traced field. This is not proof of a coil-ripple-only
cause and does not remove the reconstruction limitation of short-wave controls.

## Frozen runtime and inputs

Extract checked-in `source_bundle.tar.gz` into this directory. Use this frozen
runtime, not the installed DRBX library. `design.json` pins runtime sources;
`inputs_manifest.json` pins all input bytes. Hashes are always computed from
bytes, never cached using timestamps (the prior remote integrity bug stays fixed).

Extract the unchanged `q07-balanced-inputs.tar.gz` here, then the new
`q07-h128-supplement.tar.gz`. The supplement adds independent bounded evidence and
completed balanced baseline totals; it does not replace the old trace data.
Set `Q07_INPUT_ROOT` to the immutable root containing all eight canonical files.
The unchanged `q07-balanced-canonical.tar.gz` supplies them if missing; do not
regenerate or substitute geometry. Set `Q07_OUTPUT` to a **new** result directory.
Prior scientific chunks have a different identity and cannot be reused.

Use Python 3.12, CPU JAX/JAXLIB 0.9.2, x64, single-threaded BLAS. Dependencies in
the prior campaign environment remain applicable. Put JAX_COMPILATION_CACHE_DIR,
DRBX_CACHE_DIR and logs beneath the new run folder. Choose Q07_WORKERS from actual
CPU affinity and memory on the allocation, with Q07_WORKER_GIB (default 4 GiB)
and Q07_TOTAL_MEMORY_GIB (usable host budget; includes 2 GiB parent reserve).

```sh
python campaign.py verify
python verification/test_campaign.py
python campaign.py preflight --workers "$Q07_WORKERS"
python campaign.py pilot --workers "$Q07_WORKERS"
python campaign.py run --n 32 --workers "$Q07_WORKERS"
python campaign.py run --n 48 --workers "$Q07_WORKERS"
python campaign.py run --n 64 --workers "$Q07_WORKERS"
python campaign.py analyze
python campaign.py validate-completion
```

Do not run `freeze` remotely. Parallel preflight checks all 42 bounded owners,
all 22 states, four BCs, five fields and both spans against independent saved
N/O/R actions. Ten focused tests cover constants, complete-owner batching,
volume reductions, exact trace offsets, trace/checkpoint corruption, identical-
stat rewrites and independent completion reduction. The four-region N64 pilot
includes fresh short tracing and row construction; its local projection is not
a remote timing guarantee. See verification/h128/ for readiness evidence.

## Execution, safeguards and recovery

Node-local spawn processes dynamically schedule 793 resumable chunks. Preflight
and pilot also use spawned workers. Workers recycle after 24 chunks; compiled
fixed-size material and trace kernels are reused between chunks. There is no
multi-node distribution and no GPU kernel requirement. Use a GPU allocation
with CPU computation as selected by the remote setup skill. Keep at least 3 GiB
free disk. A lock prevents concurrent writers. Resume identical commands into
the same output directory; verified complete chunks and short traces are reused.
Do not delete valid chunks on interruption or silently migrate old identities.

Required gates: byte identities, complete owners, finite/positive states,
admissible characteristic split, exact boundary/choice policy, trace validity,
bounded centered/correction replay <=1e-8, bounded combined replay <=2e-8,
global baseline replay <=1e-8 and center-b replay <=1e-10. Bounded actual and
expected arrays must each satisfy combined = centered + correction within
128*float64_eps*(1+abs(centered)+abs(correction)+abs(combined)), allowing
floating-point owner reductions. Both spans, all fields and N/O/R use this
same constituent-sum policy; no field- or site-specific exceptions apply.
Constant N-O uses 1e-7 at h/32 and 4e-7 at h/128. A pilot bulk row had
constant slot residual 1.38e-14, amplified to electron errors 4.71e-8 and
1.85e-7 respectively. Exact constant owner inputs give the same residual.
The new h/128-only roundoff allowance scales with inverse span; it changes no
operator arithmetic or scientific replay gate. The initial failed pilot and
audit are preserved in verification/h128/. Scientific regional/order regressions are
results, not reasons to tune, stop early or change the policy.

Analysis verifies h/32 scalar statistics against the completed balanced campaign
and h/32 velocity statistics against the previous all-five campaign (original
18 states only). It produces totals.npz, analysis.json, orders.csv, report.md,
original_global_replay.json, analysis_receipt.json and completion.json. Completion
independently reduces all saved chunks and verifies artifact checksums. Preserve
all output, short traces/receipts, logs, source, manifests and operational
provenance in one downloadable run folder. Remote work is computation-only;
scientific interpretation and operator selection happen locally.

## Original h/128 readiness — before replay-policy revision

Frozen identity `0a16837968506693afdbc530216dc24911821cb8c7b4558589c870f833bbc659`. All 42 complete-owner bounded replays
passed with two workers; worst absolute N/O/R difference
`1.273e-11`. Relocated extraction of the unchanged source
and input archives plus the supplement passed byte verification and ten tests.
The four-region N64 pilot projects **131–201 minutes on two local workers**,
so execution is handed off remotely. Peak worker RSS was 1.21 GiB; statistical
checkpoint projection 0.46 GiB excludes traces, caches, source and logs.
No local full global campaign was launched. Remote timings must come from its
own allocation and pilot. Prior verification files outside `verification/h128/`
are historical evidence for the preceding campaign, with their original identities.

Two full N64 bulk/wall chunks also passed baseline statistic replay and
checkpoint reuse with two spawned workers; maximum difference
6.268e-09 < 1e-8. This is a bounded execution check,
not a completed global campaign.

## Replay-policy recovery — 1 October 2026

Current frozen identity:
`0f6bc3a6a2222441903e20777f2066401e7cd6037f952786d9060cd19b12ef88`.
The original remote N48 preflight stopped at a combined Ve replay difference
of 1.0621988622e-8. Its centered and correction differences were individually
3.6311575968e-9 and 6.9909696294e-9, both below 1e-8. The revised combined
budget is their two absolute budgets added by the triangle inequality;
constituent checks and the independent sum identity remain mandatory.
Scientific tolerances, global-statistics replay, operator arithmetic,
reconstruction, B evaluator, traces, fields and input archives are unchanged.
Fifteen focused tests include all 55,440 returned N48 replay comparisons and
explicit constituent-cancellation, sum-corruption and nonfinite rejection.
See `verification/replay_recovery/` for original identity, immutable returned
evidence, local audit and readiness receipts.

Use a new unique recovery RUN and output, retaining the failed run unchanged.
Reuse the same input/canonical archives and h/128 supplement after checksum
verification. Do not relabel old short-trace caches or preflight receipts with
the new identity. There are no global scientific chunks from the failed run.
Rerun verify, tests, all-resolution preflight and pilot before global stages.
