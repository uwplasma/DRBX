# P06 complete-curvature global qualification

This directory is the portable, computation-only N32/N48/N64 campaign for the
P06 complete curvature operator on actual HSX geometry. It does not modify the
production model, run an evolved state, or change the frozen physical-wall
baseline.

The selected reconstruction is one field-independent wall-center point
relation for wall cells and every geometry-selected wall-reaching central and
biased face fit. Thermodynamic fields impose homogeneous physical-normal
derivative data, phi imposes its zero value, and omega remains unconstrained.
Selection-v3 supports, cubic degree, weights, bias 0.75, q3 nodes, raw-midpoint
owner observations, physical-wall characteristic solve, and recentered side
jump are frozen.

## Numerical contract

Both `corrected_frozen_mms` and `regular_chart_heldout` are evaluated. Every
density, Te, Ti, and vorticity material/remainder/total component is saved for
centered and U actions, including logical-direction splits. U is primary for
density/Te/Ti; centered is primary for vorticity. The other action is a visible
diagnostic. The primary nonzero components must have global physical-volume L2
order at least 1.8 on both intervals. Vorticity's analytically zero remainder
is recorded but is not assigned a fictitious order. No result is hidden in a
pooled score.

The global target is the independent continuous evaluator integrated with q3
at the actual cell locations. Preflight evaluates complete representative
owners from every region plus periodic seams with q3/q5/q7. Its bounded
q3-to-q7 and q5-to-q7 differences are propagated as a reference budget; it is
not a second candidate-accuracy gate.

## Inputs and outputs

`configuration.json` contains only paths relative to an explicit input root.
`input_manifest.json` and `source_manifest.json` content-hash every required
external input and source dependency. `verify-inputs` rejects missing or
changed content. Runtime sidecar localization changes only the metric and
MAKEGRID paths, preserving their hashes.

All generated files—localized configuration, manifest, preparation, q3/q5/q7
preflight, work plans, chunks, receipts, logs, JAX cache, assembled cases and
summary—must be placed below one new campaign folder. Repeating `run` with the
same command validates and reuses completed chunks. A changed identity is
rejected; use a new folder rather than relabeling checkpoints.

## Command interface

Run from the DRBX repository. The remote setup skill chooses the environment,
allocation and a positive worker count; this campaign supplies no resource
count.

```bash
export P06_INPUT_ROOT=/path/to/downloaded/input/workspace
export P06_CAMPAIGN=/path/to/one/new/downloadable/p06-curvature-campaign
export P06_WORKERS="${P06_WORKERS:?set by the remote setup skill}"

python scripts/p06_curvature_global/campaign.py verify-inputs \
  --input-root "$P06_INPUT_ROOT" --output "$P06_CAMPAIGN"

python scripts/p06_curvature_global/campaign.py preflight \
  --input-root "$P06_INPUT_ROOT" --output "$P06_CAMPAIGN"

python scripts/p06_curvature_global/campaign.py run \
  --input-root "$P06_INPUT_ROOT" --output "$P06_CAMPAIGN" \
  --workers "$P06_WORKERS"

# Exact resume command: identical to run. Valid chunks are checked before work.
python scripts/p06_curvature_global/campaign.py run \
  --input-root "$P06_INPUT_ROOT" --output "$P06_CAMPAIGN" \
  --workers "$P06_WORKERS"

python scripts/p06_curvature_global/campaign.py validate \
  --input-root "$P06_INPUT_ROOT" --output "$P06_CAMPAIGN"
```

`summary.json` distinguishes computation completion, invariant validation,
reference qualification and scientific convergence. A completed computation
may legitimately report that the numerical acceptance gate failed.

## Local validation only

`replay_bounded.py` compares the portable kernels with local bounded evidence.
Those archived answers are never remote inputs. The repository tests also cover
seams, wall-patch ties, identity rejection, resume and serial/parallel equality.
