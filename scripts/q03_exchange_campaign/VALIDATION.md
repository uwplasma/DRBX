# Local preparation evidence

The parent stopped Q's local campaign before creating this remote computation
path. Only frozen-input preparation and bounded validation were run locally.

- Original optimized selector: 1,900 real HSX face comparisons across N32/N48/N64;
  identical ordered exchanges, supports, coefficients and exact/G3 fluxes;
  approximately 3.36x paired speedup. Evidence remains under workspace
  `work/parallel_q03_exchange_optimization_20260921/`.
- Standalone consumer numerics: 15 real HSX faces per resolution, spanning all
  five saved region classes. Moment arrays, donor trajectories, coefficients
  and exact/G3 fluxes agree exactly with the original research helper chain.
- Input preparation verifies the frozen batched tree-query pools against the
  original scalar pool builder on these faces. Full exported arrays and their
  source identities are hashed. No candidate global action was computed.
- Actual HSX excerpt tests cover optimized/oracle selection, valid resume,
  identity mismatch, payload corruption, streamed action assembly and missing
  chunks: `python -m pytest -q tests/test_q03_exchange_campaign.py`.
- Final command-level preflight uses two local workers on the 45 prescribed
  HSX faces. It compares serial/parallel coefficients, supports, fluxes and
  diagnostics, then validates checkpoint reuse and rebuild. Runtime receipts
  are in workspace `work/q03_remote_validation_final_20260921/`.

The small checked-in test fixture is an actual HSX N32 excerpt with monotone
owner/row renumbering and explicit parent provenance. Its tests verify algebra
and bookkeeping, not local or global convergence order. The remote worker
must run the same command-level preflight in its own environment.
