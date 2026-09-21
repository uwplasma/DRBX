# Remote execution commands

Run from the pinned DRBX repository. The remote `run drbx on perlmutter` skill
sets `HSX_INPUT_ROOT` to the existing immutable-input workspace, chooses
`CAMPAIGN_WORKERS`, and creates one unique `CAMPAIGN_DIR` for the entire run.
Put logs, scheduler output, caches, provenance and the final receipt below that
folder too. Resume the same folder and numerical identity after interruption.

```bash
python scripts/hsx_remote_qualification/campaign.py verify-inputs --input-root "$HSX_INPUT_ROOT" --output "$CAMPAIGN_DIR"
python scripts/hsx_remote_qualification/campaign.py preflight --input-root "$HSX_INPUT_ROOT" --output "$CAMPAIGN_DIR" --resolutions 32 48 64
python scripts/hsx_remote_qualification/campaign.py run --input-root "$HSX_INPUT_ROOT" --output "$CAMPAIGN_DIR" --resolutions 32 48 64 --workers "$CAMPAIGN_WORKERS"
python scripts/hsx_remote_qualification/campaign.py validate --input-root "$HSX_INPUT_ROOT" --output "$CAMPAIGN_DIR" --resolutions 32 48 64
```

The setup skill may add the optional memory-budget flags documented in README.
There is no automatic reuse of chunks from the prior source revision. Do not
alter numerical parameters, input hashes or manifests to bypass a rejection.
A completed but scientifically failing summary is an output to return for local
analysis, not authorization to tune the experiment.
