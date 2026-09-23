# Bounded RK4 replacement validation

2026-09-23. This changes the portable research campaign, not the production
operator. The former tracer was explicit adaptive DOP853, not implicit.
The replacement uses float64 compiled CPU RK4 with 64 substeps per eta interval
for both regular x/y position and connection length. The four transverse seeds,
field catalogue, endpoint cubic fits, face policy, references and acceptance
criteria remain frozen. No full local preflight or global run was launched.

## Actual-HSX accuracy checks

`validation/rk4_actions.json` records the runtime source hashes and results.
The controls are immutable historical DOP853 rows and the corrected historical
endpoint-moment actions, using exactly the same selected face observations:

| Check | N32 | N64 |
|---|---:|---:|
| Complete owners | 3 (ordinary/agglomerated/transition) | 1 ordinary |
| Shared faces | 28 | 6 |
| Observation rows | 510 | 150 |
| Largest 64-step action change / existing spatial RMS error | 1.54e-7 | 5.51e-8 |
| Largest 64-to-128-step action change / existing spatial RMS error | 4.42e-8 | 1.61e-8 |
| Largest regular endpoint difference, RK4-64 vs DOP853 | 4.78e-8 | 3.28e-9 |

Fractions are maxima across the three nonconstant fields. The constant channel
remains at rounding scale. These bounded tests support the tracer substitution;
they do not certify global convergence or replace canonical remote preflight.
The comparison threshold was 1% of the existing spatial RMS error, not a
comparison of trajectory positions alone. Doubling the RK4 steps also passes.
Historical DOP853 accuracy is a control, not an exact solution.

The optimized face-map implementation exactly reproduces the old implementation's
fluxes on these frozen observations. Lean geometry values agree with the full
metric evaluator to absolute 1e-13 / relative 1e-12 in the checked points.

Additional N32 checks cover 62 rows in axis, ordinary and wall/seam batches:
RK4-64 and RK4-128 retain the DOP853 validity decisions for every checked row.
Additional N64 axis/wall/seam checks cover eight rows per direction: 64 and 128
steps agree on validity (6/8 backward, 8/8 forward); their largest regular
endpoint difference is 1.54e-10. This samples domain admissibility but does not
establish identical classifications for every trajectory in the full domain.

`validation/rk4_smoke.json` records measured local elapsed times. The first
64-step batch, including compilation, took 21.43 seconds. Subsequent ordinary
24-row and wall/seam 14-row batches took 1.53 and 1.18 seconds, including endpoint
reconstruction. These are local observations, not a Perlmutter runtime promise.
JAX compilation has an initial cost; padded shapes and a campaign-local cache
avoid recompiling each short batch. Historical DOP853 whole-run timing estimates
must not be applied to the new runner.

## Reproduction

The bounded scripts consume existing local research artifacts, deliberately
avoiding regeneration of the large campaigns. From the repository root, set
`WORKSPACE` to the HSX workspace and `CHECK` to a local validation directory:

```bash
mkdir -p "$CHECK"
git show 31138a312351f2685f761a32883dc595debc6db0:scripts/q_fci_return_campaign/numerics.py > "$CHECK/dop853_reference.py"
python scripts/q_fci_return_campaign/validation/rk4_local.py --workspace "$WORKSPACE" --output "$CHECK"
python scripts/q_fci_return_campaign/validation/rk4_smoke.py --workspace "$WORKSPACE" --output "$CHECK"
python scripts/q_fci_return_campaign/validation/rk4_pool.py --workspace "$WORKSPACE" --output "$CHECK"
pytest -q tests/test_q_fci_return_campaign.py
```

The historical controls live under `work/parallel_fci_local_return_20260923`
and `work/parallel_fci_local_return_refinement_20260923`. The separate pool check
uses the saved serial smoke files in the local validation directory. It verifies
two actual spawned workers, checkpoint receipts, and agreement with serial
results; see `validation/rk4_pool.json`. The two small N32 workers peaked at 1.13 and 1.15 GiB RSS; larger resolutions
and compilation require additional memory headroom. Linux scheduler affinity cannot be
validated on this macOS host and must be checked during remote launch.
Five checkpoint/identity/writer-lock tests pass.

## Seed count evidence

The four seeds are the 2x2 composite-midpoint pattern at transverse fractions
(1/4,1/4), (1/4,3/4), (3/4,1/4), (3/4,3/4), all on the source eta plane.
They are not corners. Four legs are combined into each directional observation.

The earlier N32 nine-owner/42-face comparison in
`work/parallel_fci_return_basis_comparison_20260923/report.md` found numerical
RMS errors [4.4649e-4, 5.6431e-4, 5.5615e-4] with one seed, versus
[2.0684e-4, 9.1388e-4, 1.4121e-3] with four seeds, for the geometry-aware
96-observation return. One seed was competitive and better on two fields.
However, their observation catalogues and selected supports differ, and this is
one resolution on an ordinary patch. It provides no convergence slope, no global
qualification, and no axis/wall evidence for the one-seed method. Four seeds are
retained while changing and checking the tracer. A separate matched refinement
experiment would be needed before reducing the frozen seed count.
