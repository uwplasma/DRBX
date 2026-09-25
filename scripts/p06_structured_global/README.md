# P06 midpoint curvature global qualification

This v3 research campaign implements the user-approved midpoint formulation on
canonical N32/N48/N64. It is not production promotion or an evolved test.

## Frozen numerical and reference contract

Keep midpoint/raw-volume owner observations, the shared structured value/gradient
reconstruction, all four states, M/R/total terms, primary centered/U choices,
Dirichlet trace lift and characteristic wall model. Evaluate the complete
nonlinear curvature expressions once at each raw-cell midpoint (q1). Sum
`Delta_xi*J/B*S` and `Delta_xi*J/B` across every owner before division. Keep the
integrated q3 characteristic face numerators, dividing by this same midpoint
owner mass. No one-point-per-aggregate substitute or new evolved value is used.

The primary exact reference uses analytic fields/derivatives at those raw
midpoints, the same J/B midpoint mass and complete-owner reduction. It does not
use reconstructed numerical fields. Physical J reference values remain labeled
diagnostics. Error-norm weights stay the frozen physical owner volumes; they
are explicitly separate from the candidate quadrature volumes.

Global operator L2 order must be >=1.8 on both intervals for each nonzero primary
component. The analytic-zero vorticity remainder is exempt. Preflight compares
the midpoint reference to a half-step geometry/derivative evaluation on the
same complete owners, using their corresponding numerical errors for the 10%
reference screen. No midpoint-to-integrated reference differences are computed; those controls
are deferred at the user's request. Global q5/q7
references and mandatory bounded q5/q7 controls are not part of this campaign.

## Execution and recovery

The existing node-local CPU worker pool parallelizes preflight, numerical cell
and face work, and independent reference chunks. One BLAS thread per worker,
bounded memory/concurrency, deterministic reduction and payload hash checks
remain. Immutable inputs are unchanged. The remote allocation skill selects
resources; no full local run is needed for preparation.

```bash
python scripts/p06_structured_global/campaign.py verify-inputs --input-root "$INPUT_ROOT" --output "$OUTPUT"
python scripts/p06_structured_global/campaign.py preflight --input-root "$INPUT_ROOT" --output "$OUTPUT" --workers "$WORKERS" --memory-budget-gib "$MEMORY_GIB" --worker-memory-gib "$WORKER_GIB" --memory-reserve-gib "$RESERVE_GIB"
python scripts/p06_structured_global/campaign.py run --input-root "$INPUT_ROOT" --output "$OUTPUT" --workers "$WORKERS" --memory-budget-gib "$MEMORY_GIB" --worker-memory-gib "$WORKER_GIB" --memory-reserve-gib "$RESERVE_GIB"
python scripts/p06_structured_global/campaign.py validate --input-root "$INPUT_ROOT" --output "$OUTPUT"
```

Use a fresh output folder and pinned source identity. Historical v1 physical-J
and v2 integrated-J/B chunks cannot resume as midpoint v3. Optimization-only
adoption cannot migrate a changed method. No automated importer is implemented;
future selective reuse requires verified numerical dependencies and new receipts.
Keep all previous scientific flags unchanged. Return raw errors, regional data,
step sensitivity, identities and resource receipts.
