# P05 frozen nodewise cubic material-upwind accuracy campaign

Research-only global static operator comparison on actual HSX N32/N48/N64.
The user authorized this remote accuracy campaign after the bounded nodewise
jump study. Dissipation, positivity, evolved MMS and production promotion remain
unqualified; those are not prerequisites to measuring this static global order.

This is the **P material-upwind campaign**, not Q parallel diffusion and not a
repeat of the completed centered study.

## Frozen numerical contract

Use the previous centered campaign at `c54b0552` as an immutable baseline.
Its matched material A action, common q3 face value, phi generator, anchor,
volume correction, physical owner volume and independent reference remain
unchanged. The candidate is A plus the conservative action of

```
delta_F = -1/2 integral_face |U| (r_right - r_left).
```

Both cubic side fits use the same selection-v3 donors and actual owner moments
as the central fit, with fixed squared weights
`(1+d^2)^-2 * (1 +/- 0.75*tanh(s))`, the bounded study's directional coordinate,
and rank-revealing least squares. Evaluate the side jump at all nine q3 nodes,
then recenter around the unchanged common value. No bias/donor search,
limiter, different quadrature, or field-specific reconstruction is introduced.

The global donor graph and central face flux are replayed against the immutable
centered chunks on **every face**. A mismatch is reported, not fixed by mixing
graphs or changing tolerances. Collapsed-axis flux is zero; physical radial
walls retain the previous center-to-boundary trace jump. The original periodic
endpoint incidence is preserved (both stored endpoint faces must not each be
wrapped to both endpoint cells).

Required material fields are `smooth_regular_scalar` and
`smooth_eta_varying_scalar`. Actual manufactured vorticity is retained as a
diagnostic; its production operator remains the qualified centered bracket.
Report global physical-volume L2 order >=1.8 on both refinement intervals,
per field. Re-divide the existing absolute reference-uncertainty budgets by the
new errors and require each below 10%. Failed numerical orders or insufficient
reference budgets are completed scientific outputs, not permission to tune.

## Inputs and provenance

The runner requires two independent input roots:

- `--input-root`: the existing immutable HSX input workspace. Layout and hashes
  are inherited from `hsx_remote_qualification/input_manifest.json`.
- `--centered-root`: the **completed** `p_centered_cubic_c54b0552_kFhdmt`
  campaign. `centered_manifest.json` pins 2,431 required files: case/preparation
  arrays, case summaries, campaign identity, summary, and all face chunks.
  Cell chunks are unnecessary because the complete centered action already
  contains the fixed volume contribution.

The existing remote receipt locates these at
`/pscratch/sd/y/yiqunx/uw_summer` and
`/pscratch/sd/y/yiqunx/uw_summer/work/p_centered_cubic_c54b0552_kFhdmt`.
Relocation is supported through the explicit arguments; contents must match.
Do not regenerate geometry, manufactured inputs or the completed centered run.
Missing/different content requires returning failure evidence for local review.

Source manifests cover the frozen centered dependencies and new campaign code.
The dirty local research workspace is not an input. No loose `work/*.py`
research scripts are imported by this runner.

## Commands and one-folder outputs

Run from the pinned DRBX repository with the environment established by the
remote worker's **run drbx on perlmutter** skill. That skill chooses the
allocation, worker count and supervision. `CAMPAIGN_WORKERS` must be explicit.
The entry point sets one numeric-library thread per worker before imports.

Create one new uniquely named `CAMPAIGN_DIR` for all stages and resolutions.
Put logs, scheduler stdout/stderr, launch records, temporary files, caches and
the operational receipt beneath it. Existing immutable inputs may remain
outside; their locations/hashes are recorded. Downloading this one folder must
retrieve real outputs, not symlinks to external output files.

```bash
python scripts/p05_material_campaign/campaign.py verify-inputs --input-root "$HSX_INPUT_ROOT" --centered-root "$P_CENTERED_ROOT" --output "$CAMPAIGN_DIR"
python scripts/p05_material_campaign/campaign.py preflight --input-root "$HSX_INPUT_ROOT" --centered-root "$P_CENTERED_ROOT" --output "$CAMPAIGN_DIR" --resolutions 32 48 64
python scripts/p05_material_campaign/campaign.py run --input-root "$HSX_INPUT_ROOT" --centered-root "$P_CENTERED_ROOT" --output "$CAMPAIGN_DIR" --resolutions 32 48 64 --workers "$CAMPAIGN_WORKERS"
python scripts/p05_material_campaign/campaign.py validate --input-root "$HSX_INPUT_ROOT" --centered-root "$P_CENTERED_ROOT" --output "$CAMPAIGN_DIR" --resolutions 32 48 64
```

Preflight includes the N32 HSX axis/interior/wall/seam replay and the complete
N48/N64 bounded P study samples, including frozen jump and completed-owner
action replay. It is an implementation check, not another scientific gate or
scaling campaign. Local parallel/restart validation uses `check_execution.py`
with the same roots/output and explicit `--workers`; it is not an additional
required remote scaling study.

## Execution, recovery and completion

Only one controller holds the campaign lock. Independent chunks contain 256
faces; workers retain one geometry context and recycle after 16 chunks.
Completed chunks are verified before dispatch. Repeating the same command
resumes valid chunks with identical source/input/policy. Worker count may change
with allocation. Corrupt or incompatible receipts fail explicitly. A chunk
without a completion receipt may be recomputed. Do not edit a manifest, change
the candidate, or relabel old outputs to force continuation.

The same `run` command assembles each resolution and generates `summary.json`
after all three. `validate` independently verifies chunk hashes/coverage,
reassembles actions and regenerates the summary. `N*.npz` contains owner keys,
volumes, references, centered A/C, material A, total/wall corrections and region
masks. `N*.json` records global and regional errors, signed error budgets,
reference budgets and provenance. Chunks retain corrections, central-flux
replay, donor hashes, conditions, fallback and polynomial diagnostics; receipts
record time and peak memory. No dense global reconstruction matrix is built.

Numerical fields and actions are frozen. There is no timestep evolution in
this run. `material_operator_convergence_passed` is separate from
`positivity_dissipation_certified`, `solution_certified` and production status.
Do not reinterpret those pending qualifications as a reason to withhold a
completed accuracy result.

The remote task performs computation and operational verification only. Return
all outputs and a receipt with the commit, identities, allocation/job IDs,
workers, timings, exits, stage/validation status, absolute campaign folder and
inventory. Return failures with the exact command/error. Scientific analysis,
numerical repairs, optimization, additional experiments and promotion happen
locally after download.
