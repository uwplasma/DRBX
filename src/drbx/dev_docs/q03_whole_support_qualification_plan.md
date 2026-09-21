# Q03: whole-support cubic exchange toward global qualification

Assigned 21 September 2026 to the existing Q worker. This is the next bounded
work package under the [parallel roadmap](parallel_second_order_roadmap.md),
not a new Q milestone or a change to its acceptance contract.

**Latest user scope update:** stop Q's local campaign and prepare the frozen
N32/N48/N64 global computation for remote execution. This supersedes the local
launch instructions below, which remain historical context. Q is idle and no
local Q compute process remains. The parent prepared the self-contained
[`scripts/q03_exchange_campaign`](../../../scripts/q03_exchange_campaign/README.md)
runner, immutable input bundle and bounded validation. The remote task performs
only the prescribed computation/operational checks; scientific analysis returns
to the local task. The proposed broader local sample is not a prerequisite to
the now-authorized full global comparison. Do not restart local global work.

## Evidence and candidate

Read the research reports and their design/summary files, relative to the
workspace root (the directory containing DRBX):

- `work/parallel_q03_selector_hotspot_replay_20260921/report.md`: full-SVD and
  cached rank-one enrichment select identical sampled rows/actions. The fast
  enrichment optimization is not the sampled defect.
- `work/parallel_q03_support_extent_investigation_20260921/report.md`: original
  observations set important support outliers; additions-only compaction is
  insufficient. Strict historical label counts/envelopes are not requirements.
- `work/parallel_q03_whole_support_exchange_20260921/report.md`: the frozen
  whole-support exchange, protected-original control and exact/G3 comparisons.
- `work/parallel_q03_full_update_exchange_20260921/report.md`: complete update
  of the same 168 owners, 864 incident faces per resolution. N48 G3 error ratios
  are 0.998/1.132/1.029 and N64 0.963/0.744/0.814, radial/angular/mixed.
  Added residual-blind 84-owner patches improve in all three G3 fields at both
  resolutions. Exact-row and regional tradeoffs remain. These are not global
  orders. Four prescribed boundary faces per resolution retain the exact
  continuum flux used by the frozen campaign.

Freeze the **whole-support cubic exchange with 64 accepted swaps maximum** as
the candidate. The numerical oracle is `selector` in
`work/parallel_q03_whole_support_exchange_20260921/exchange.py`.
Reuse unchanged geometry, traces, observation rows, G3, qualified integrated
references, q9 face targets, candidate pools, distance weights and cubic fitter.
Minimize the existing cubic information objective, keeping total/per-plane row
budgets and initially represented sector/direction coverage. Original rows
are replaceable. Keep canonical shared-face incidence. Do not add extent caps,
moment penalties, more donors, higher degree, new limiters or a different
boundary law to improve this sample. The protected-original exchange is a
bounded control, not an additional mandatory global candidate.

**Available implementation optimization:** the parent profiled and implemented
`work/parallel_q03_exchange_optimization_20260921/optimized_exchange.py`.
Read its report and summary. Plane-batched scoring and cached label bookkeeping
preserve the same selector policy and direct SVD verification, with scalar
ranking fallback near ties. Paired speedup is 3.36–3.37x; 1,900 HSX faces across
N32/N48/N64 give identical donor trajectories, coefficients and exact/G3 fluxes.
Use this implementation in the runner after the integrated bounded checks;
do not redo the optimization study. Global calls can use `record_swaps=False`.
Keep the oracle and explicit implementation/dependency identities. Revised
selector-only projection is about 9 CPU-hours across all three resolutions,
or 4.5 hours at ideal two-worker scaling; measure integrated throughput for
the actual ETA. N32's saved chunks/actions are at the campaign root rather
than an additional N32 subdirectory, as the optimization benchmark documents.

## 1. Freeze a broader residual-blind sample and execute it

Use the actual N32/N48/N64 artifacts. Independently select at each resolution:

| Disjoint region | Target owners |
|---|---:|
| axis | 16 |
| RLP transition-adjacent | 16 |
| agglomerated bulk | 16 |
| ordinary interior | 32 |
| boundary | 16 |

Use the saved `region:*` masks. First verify their disjoint coverage of the
owner population; if they overlap, use the existing reporting partition or
document a deterministic precedence before selection. Select uniformly without
replacement from sorted owner IDs using a recorded NumPy generator seed
`20260921 + N`. Use all owners if a stratum has fewer than requested. Generate
and hash all three manifests before reading candidate errors. No refinement
mapping, hotspot hunting, rejection based on errors, or redraw after results.
Report overlap with previous samples but do not exclude it after seeing errors.

For **every selected owner**, include every incident canonical face. Axis and
RLP owners can have many incident tiles: count actual faces rather than assuming
six or cropping their stencil. Reuse matching baseline/enriched and newly
exchanged face artifacts by identity; solve only missing faces. Boundary faces
retain the frozen campaign's prescribed continuum flux for all variants.
Record this effective diagnostic boundary treatment separately from the shared
model's pinned physical-wall settings. A prescribed flux pass does not certify
boundary reconstruction or wall physics.

Compare baseline enriched, protected-original exchange and whole-support
exchange on exact/G3 observations for radial_eta, angular_x, mixed_y_eta, plus
constant controls. Separate any new partially updated fringe from fully updated
selected owners. Do not grow another halo recursively. Keep prior 168-owner
results as independent supporting evidence; no need to rerun them.

Report per-field volume-weighted errors, regional squared-error contributions,
individual owner errors, fit rank/reproduction, amplification, condition,
support/endpoint extent, cap hits, fallback activity (normally none), and
complete-cell leading moment/face-cancellation diagnostics. No expensive
reference regeneration: use the qualified integrated face reference; only add
bounded qualification if an observed numerical floor makes it necessary.

Sample RMS and sample slopes are diagnostics, not global orders. Optionally
report a design-weighted scout estimate of global SSE using
`sum_s (M_s/k_s) * sum_sample_s(volume * error**2)` and the known total volume,
where M_s and k_s are population and sample counts. Clearly label sampling
uncertainty; this estimator is not certification and a small sample must not
be presented as a precise global prediction. Within-stratum contributions to
sample error are not automatically contributions to full-domain error.

**Completion:** reproducible complete-cell comparison across the actual HSX
regions, with valid provenance and usable numerical actions. Do not require
local order 1.8, monotone sample errors, universal improvement, a condition-number
cutoff, or a quartic-norm decrease. Continue through all resolutions despite
mixed scientific results. If rank/compatibility fails, diagnose a bounded
example and report it; do not silently retain baseline on failed faces and
call that the frozen candidate.

## 2. Prepare and test the global computation path

Build on the existing streamed Q campaign machinery, not the rich per-face
Python dictionaries used for the bounded diagnostics. Preserve the oracle's
numerical policy. Performance changes should be one-off compute improvements;
avoid speculative cache infrastructure or another numerical optimization sweep.

- Expose explicit input/output roots, resolution and face-range/chunk selection;
  eliminate dependence on the original absolute workspace path in the delivered
  runnable bundle. Inventory all helper/source and data dependencies, including
  N32's different saved-enrichment campaign location and actual chunk metadata.
  Do not assume the N48/N64-only audit loader supports N32 unchanged.
- Use compact ragged supports/coefficients, scalar diagnostics and streamed
  flux/action assembly. Retain full swap histories only for the bounded sample
  or an explicitly small debug subset, not every global face.
- Load immutable resolution inputs once per persistent worker; bound submitted
  work and result queues. Avoid copying entire geometry for every face/chunk.
  Measure load, selection, fit, assembly and output costs separately.
- Identity-check geometry, implementation/policy, rows, targets, reference,
  field catalogue, boundaries and requested checks. Each support change
  invalidates dependent outputs. Reuse independent inputs. Use atomic completed
  chunk receipts and resume only matching, validated chunks.
- Test serial versus two-worker execution on a small shared face list spanning
  the five HSX strata. Compare against the frozen selector, canonical face
  coverage, supports, coefficients, fluxes and complete actions. On this same
  host, investigate donor differences rather than accepting a changed policy.
  Portable runs must be internally consistent; do not force historical
  platform-specific ties or mix dependent caches from different donor graphs.
- Test bounded restart/assembly by reusing one valid chunk and rebuilding one
  missing chunk in a disposable test output. A zero exit code alone is not a
  completeness check. Do not write/delete actual campaign checkpoints for this.

Use bounded local parallelism chosen from measured available memory; do not
launch a 64-process local pool. The planned remote capacity is a target the
runner may support, not a local resource request. No remote allocation,
environment setup, scheduler integration, remote launch or P campaign edits
belong to this assignment. If putting code inside DRBX, follow its package
guidance and run the focused checks appropriate to those changes.

## 3. Deliver the global-readiness decision

Produce one new output folder with the frozen design, sample manifest,
measurements, validated runner/source identities, input manifest, exact CLI
examples, serial/parallel/restart evidence, measured setup/runtime/memory/disk
costs and a report. Estimate the full N32/N48/N64 workload from actual face
counts and representative region costs; separate CPU work from unmeasured
parallel wall-time scaling. Do not extrapolate one ordinary-interior face to
all regions or imply that 64 workers guarantee 64-fold speedup.

State whether to proceed with this frozen candidate, whether a reproducible
implementation defect needs repair, or whether broad accuracy evidence favors
revisiting the formulation. A regional regression alone is not a blocker.
Avoid extra diagnostic rounds without naming the uncertainty they resolve.
Update the main roadmap with evidence and the next bounded decision. After
implementation/provenance checks pass, continue directly to the authorized
local global campaign in step 4; do not wait for a parent acknowledgment or
require the regional sample to meet a new accuracy threshold. Report a real
implementation/input/resource blocker if present instead of silently changing
the candidate. Avoid infrastructure work unnecessary for the local campaign;
explicit paths and dependency identity remain useful.

## 4. Authorized local global qualification

The local computation will evaluate the same frozen whole-support candidate
on all faces/owners at N32/N48/N64, with the saved baseline comparison. Run
all three resolutions under one candidate identity and assemble each validated
global action. Individual negative results should not silently stop the chain.
Report the exact/G3 distinction, per-field physical-volume-weighted L2, both
refinement intervals, diagnostic regional/max norms and global SSE partition.
The agreed global operator target remains order >=1.8 on both intervals for
every nontrivial field of the action being certified; an exact-row pass cannot
substitute for the reconstructed G3 action. Requalify reference error against
the new spatial error if needed, rather than assuming an old relative bound
survives a much smaller candidate error.

Use the measured serial/two-worker preflight to choose bounded local concurrency
with one BLAS thread per worker. Start with two persistent workers; increase
only if measurements and current machine headroom support it. As a conservative
operational starting budget, target at most 8 GiB aggregate campaign RSS,
including worker input copies and assembly, and reduce concurrency if needed.
This is a scheduling choice, not a numerical acceptance criterion. Process
resolutions sequentially and let the script supervisor advance immediately
after validated completion, without waiting for heartbeat wakes. Reuse all
identity-matched sample chunks. Run only the whole-support candidate globally;
the protected control remains bounded and the existing global baseline is
reused. Retain restartable chunks and avoid full global swap-history retention.
Use measured progress to revise completion estimates. Do not launch remotely
or raise concurrency merely to meet an optimistic timing estimate.

A pass establishes this operator-accuracy rung, not automatic Q04 completion:
structural diffusion properties and the roadmap's independent fixed-time and
integration checks still have their own scope. There is no N128 extension,
physical-wall-law migration or production promotion in this assignment.

## Execution and coordination

Use `/Users/yxie/.codex/skills/supervise-long-runs/SKILL.md` for work that outlasts
the active turn. Inspect existing processes/checkpoints/automations before
launching. Chain authorized stages in a lightweight script supervisor; quiet
heartbeat default 10 minutes, adaptive 20/30 minutes, not above 30. Short tests
need bounded waits, not a heartbeat or repeated short polling. Honor the user's
request to avoid continuously watching. Stop/pause monitoring on completion.

Do not message the parent, P, O or other tasks. Keep results and status in this
Q task and its artifacts; the parent will read them. Preserve unrelated work.
Do not commit/push or create additional worker tasks as part of this assignment.
