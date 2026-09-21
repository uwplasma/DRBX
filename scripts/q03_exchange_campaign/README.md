# Frozen Q03 whole-support cubic exchange campaign

Computation-only NumPy consumer of qualified actual HSX inputs at N32/N48/N64.
This is the optimized 64-swap cubic whole-support candidate. It is not the P
perpendicular campaign and is not production diffusion certification.

## Inputs

Use the separately supplied `q03_remote_inputs_20260921.tar.gz`. Its root
`manifest.json` identifies all resolution manifests; those identify every
numeric array by SHA256. Exported arrays contain HSX leg endpoints/lengths,
source planes/directions, canonical face incidence/charts, owner volumes,
original and enriched seed supports, frozen five-plane nearest-128 candidate
IDs, q9 face moments, exact/G3 observations, qualified integrated references,
saved baseline actions and the disjoint region masks. No remote geometry or
reference generation is required. These immutable inputs are independent of
the candidate outputs and may be reused after identity verification.

Candidate-pool membership is frozen to avoid redoing platform-sensitive tree
cutoffs. The new exchange graph is generated once within the remote campaign;
no downstream locally generated exchange cache is spliced into it. Preflight
compares optimized versus scalar selection on the current host, rather than
requiring historical local donor IDs on a different platform. Source, input,
NumPy/Python and machine identity are part of the checkpoint contract.

The input manifest stores the geometry/reference lineage and seed-chunk
identities. The artifact is about 4.46 GB uncompressed. Absolute paths in
provenance are historical labels only; all active inputs resolve from `--inputs`.

## Commands

Run from the repository root in the DRBX Python environment. `Q03_INPUTS` is
the extracted immutable input directory; `Q03_CAMPAIGN` is one new campaign
output folder. `Q03_WORKERS` is explicitly chosen by the allocation/setup skill.
The remote handoff must not prescribe allocation or a worker count.

```bash
python scripts/q03_exchange_campaign/campaign.py verify \
  --inputs "$Q03_INPUTS" --output "$Q03_CAMPAIGN"
python scripts/q03_exchange_campaign/campaign.py preflight \
  --inputs "$Q03_INPUTS" --output "$Q03_CAMPAIGN" --workers "$Q03_WORKERS"
python scripts/q03_exchange_campaign/campaign.py run \
  --inputs "$Q03_INPUTS" --output "$Q03_CAMPAIGN" --workers "$Q03_WORKERS"
python scripts/q03_exchange_campaign/campaign.py validate \
  --inputs "$Q03_INPUTS" --output "$Q03_CAMPAIGN"
```

Capture each command's logs and exit status under the campaign folder. Also
place scheduler stdout/stderr, launch/job records, temporary/cache outputs and
the final receipt there. Set `TMPDIR`, `XDG_CACHE_HOME` and
`PYTHONPYCACHEPREFIX` to subdirectories there when launching. There is no JAX
or geometry runtime in this consumer. Each process uses one BLAS thread;
parallelism is at the persistent process level with read-only memory-mapped
inputs and a bounded job queue.

`verify` checks source hashes and every input file. `preflight` compares scalar
and optimized trajectories, then serial/parallel serialization and bounded
checkpoint reuse/rebuild on the prescribed HSX faces. `run` executes N32, N48,
N64 sequentially, with parallel face chunks within each resolution, and
assembles results after every resolution. `validate` checks all expected chunk
receipts and reassembles the complete actions and summaries. The default chunk
size is 512 and must remain unchanged when resuming the same folder.

## Resume and failure handling

Rerun `run` with exactly the same inputs/source/environment and output folder;
valid chunks are skipped. The worker count may change without changing the
numerical chunk identity. Re-run the matching preflight if missing. A file lock
prevents two parent writers. Atomic chunk receipts include face IDs, payload
hash and computation identity. Mismatches/corruption raise errors rather than
relabeling outputs. Partial files without a valid receipt are recomputed.

Do not delete or bypass identity checks to force reuse. Preserve failing logs
and the `operations/` receipts. Numerical code, data or policy changes belong
to a local follow-up, not the remote worker. Scientific order failure is stored
in summaries and does not make an otherwise complete computation an execution
failure or authorize a different experiment.

The final fitter preserves the original research `usable` checks: rank 19,
finite coefficients, relative compatibility/residual <=1e-10 and existing
roundoff indicator <=1e-8. These are inherited implementation checks, not a new
condition-number or regional-accuracy requirement. Boundary faces retain the
frozen campaign's prescribed continuum flux, and are excluded from fit-rank
checks; boundary wall-law accuracy is not being certified.

## Outputs and operational receipt

- `campaign.json`, `input_verification.json`, `preflight.json`: identities/checks.
- `progress.json`, `operations/*.json`: progress, commands, PIDs, job IDs,
  worker settings, elapsed times, exit statuses and errors.
- `N*/chunks/*.npz` and receipts: compact supports, coefficients, fluxes and
  fit/selection diagnostics; no full exchange histories.
- `N*/face_flux.npy`, `actions.npz`, `summary.json`: complete canonical fluxes,
  owner actions, baseline/reference arrays, volumes, region masks and errors.
- `global_summary.json`: built-in per-field exact/G3 errors, both refinement
  orders, order-gate flags, and an explicit absence of full certification.

Diagnostic columns in `actions.npz` are named in `summary.json`; boundary
entries have zero fit diagnostics. Existing integrated-reference qualification
must be assessed against new spatial errors locally. The built-in order flag
does not establish reference, positivity, dissipation or fixed-time/Q04 gates.

Return the entire real output folder, not symlinks to files elsewhere, and a
short `operational_receipt.md` within it: pinned commit; source/input identities;
job IDs and worker settings; stage elapsed times/statuses; completed/missing
outputs; absolute download path; and exact command/error if execution stopped.
Return machine-generated scientific results unchanged. Do not interpret them.
