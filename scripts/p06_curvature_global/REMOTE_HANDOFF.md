# P06 curvature Perlmutter handoff — publication-pending draft

Status: **do not execute yet**. The complete campaign implementation exists
locally but has not been committed or pushed. Replace
`<PUBLISHED_P06_COMMIT>` only after publication and verify that exact revision
on the remote host. The current remote branch tip must not be treated as if it
contains this campaign.

Publication must include every exact path and hash in `source_manifest.json`.
The P06 additions themselves are this whole directory,
`scripts/audit_p06_boundary_policy.py`,
`src/drbx/geometry/fci_boundary_functional_reconstruction.py`,
`src/drbx/geometry/fci_perpendicular_bracket.py`, their package exports and
tests, and the P06 roadmap update. Several transitive HSX research drivers are
also currently unpublished; `source_manifest.json` is the exhaustive
publication checklist and `verify-inputs` will reject an incomplete revision.

Use the **run drbx on perlmutter** skill for repository checkout, environment,
allocation, resource choice, worker count, monitoring and artifact return. Do
not perform a scaling study. Do not edit the numerical policy, repair results,
or analyze scientific outcomes.

Repository identity to verify after publication:

- URL: `git@github.com:uwplasma/DRBX.git`
- branch: `2D_fci`
- required revision: `<PUBLISHED_P06_COMMIT>`
- campaign entry point: `scripts/p06_curvature_global/campaign.py`

Immutable external inputs are the relative paths and SHA-256 records in
`scripts/p06_curvature_global/input_manifest.json`. The input root must expose
that exact layout. The repository source is independently checked by
`source_manifest.json`.

Create one new downloadable campaign folder and keep every generated output,
log, checkpoint, cache, manifest, provenance record and operational receipt
inside it. Set the input root to the staged immutable-input workspace and let
the remote skill choose a positive worker count.

```bash
cd /path/to/DRBX
git fetch origin 2D_fci
git checkout --detach <PUBLISHED_P06_COMMIT>
test "$(git rev-parse HEAD)" = "<PUBLISHED_P06_COMMIT>"
git remote get-url origin

export P06_INPUT_ROOT=/path/to/staged/p06-input-workspace
export P06_CAMPAIGN=/path/to/one/new/downloadable/p06-curvature-campaign
export P06_WORKERS="${P06_WORKERS:?choose through the run drbx on perlmutter skill}"

python scripts/p06_curvature_global/campaign.py verify-inputs \
  --input-root "$P06_INPUT_ROOT" --output "$P06_CAMPAIGN"
python scripts/p06_curvature_global/campaign.py preflight \
  --input-root "$P06_INPUT_ROOT" --output "$P06_CAMPAIGN"
python scripts/p06_curvature_global/campaign.py run \
  --input-root "$P06_INPUT_ROOT" --output "$P06_CAMPAIGN" \
  --workers "$P06_WORKERS"

# Resume, if interrupted, with the identical command.
python scripts/p06_curvature_global/campaign.py run \
  --input-root "$P06_INPUT_ROOT" --output "$P06_CAMPAIGN" \
  --workers "$P06_WORKERS"

python scripts/p06_curvature_global/campaign.py validate \
  --input-root "$P06_INPUT_ROOT" --output "$P06_CAMPAIGN"
```

Return the single campaign folder with `campaign_manifest.json`, all
`N*.prepare.npz`, `N*.preflight.*`, `N*.plan.json`, chunk directories,
parallel receipts, assembled `N*.npz`/`N*.json`, `summary.json`, logs and the
remote skill's operational receipt. Report operational completion or failure;
do not interpret convergence or alter inputs to obtain a pass.

Run the specified computation and return its artifacts and operational receipt. Do not analyze the scientific results; we will do that locally.
