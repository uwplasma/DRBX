# Parallel architecture comparison: traced FCI and direct reconstruction

Research decision reopened by the user on 22 September 2026. This is an open
design document, not a production capability claim. The
[Q roadmap](parallel_second_order_roadmap.md) remains the acceptance authority.

## Preference and decision boundary

Retaining traced FCI is heavily preferred for the parallel system. It is not
mandatory at any cost; a direct replacement should be selected only after
credible traced alternatives appear impractical or unable to meet the intended
requirements. Failure of one return formulation is not an FCI impossibility
result. Do not invent new idealized, pointwise-order, exact-cancellation or
conditioning gates to decide this question.

| Aspect | Traced FCI branch | Direct branch |
|---|---|---|
| Parallel connection | Follow field lines between planes | Use evaluated magnetic direction in a spatial differential functional |
| Scalar information | Owner observations reconstructed at traced points or mapped footprints | Owner observations reconstructed in local spatial patches |
| Differentiation/flux | Along traced connections; compatible return/projection | Direct polynomial derivatives and coordinate-face flux integration |
| Conservation | Must be built through shared mapped fluxes or compatible weighted assembly | Shared canonical physical-face incidence already implemented |
| Current evidence | Owner-overlap baseline and many mapped-gradient-to-face prototypes; no general rejection | Qualified diffusion accuracy and extracted kernel; positivity defect remains |
| Important unresolved test | Moment-consistent scalar transfer plus independent traced assembly, including unequal eta resolution | Anisotropic resolution efficiency and geometry integration; full G/D qualification |

The old mapped-return prototype is a third, specific construction inside this
history: traced directional secants were inverted into coordinate-face fluxes.
Its weak observability and return errors cannot be assigned to every traced
FCI discretization. Similarly, a symmetric diffusion matrix is not a transport
operator and an energy-stable adjoint construction is not automatically locally
consistent or positivity preserving.

## Bounded parent audit and experiment

1. Inventory actual prior scalar point/average transfers, differentiation and
   return choices, identifying what was tested rather than names alone.
2. Use existing actual HSX geometry and original nontrivial manufactured fields.
   Compare exact traced data with moment-consistent reconstructed scalar data,
   separating transfer, along-line discretization and return errors.
3. Retain complete owner membership and necessary neighboring closure for any
   assembled-action claim; explicitly distinguish point/row diagnostics from
   a completed conservative operator. Use trace-based differentiation, not
   direct b-dot-polynomial-gradient substituted under an FCI label.
4. Include a bounded unequal-eta/perpendicular-resolution diagnostic, with
   fixed-direction floors reported honestly. Existing trace composition or a
   bounded retrace may be used when qualified; no silent geometry replacement.
5. Preserve direct actions as benchmarks. No new full campaign, production
   promotion or forced architecture decision follows automatically from this
   bounded experiment. Report the next supported assembly test when closure
   or geometry uncertainties prevent a complete comparison.

The [completed parent audit and bounded experiment](../../../../work/parallel_fci_reopening_20260922/report.md)
finds that scalar transfer is a small part of the sampled three-point traced
diffusion error. Exact-data eta refinement approaches second order; a plain
weighted-adjoint assembly and an eta-coordinate strong-form rewrite do not
by themselves repair coarse-spacing accuracy. The next bounded target is
traced flux-tube integration and matched owner projection, with exact-data
and reconstruction controls. No traced global accuracy or wall pass is
claimed. Existing direct qualifications remain intact.

## Literature basis

- Stegmeir et al., *The field line map approach for simulations of magnetically
  confined plasmas*, https://arxiv.org/abs/1505.02040: traced differentiation and
  support-operator parallel diffusion; weighted adjoints provide structure but
  do not independently guarantee the HSX operator-order contract.
- Wiesenberger and Held, *A finite volume flux coordinate independent approach*,
  https://doi.org/10.1016/j.cpc.2023.108838: locally field-aligned finite volumes,
  integrated transfers and conservative fluid discretization. Its smoothing
  and conservation qualifications must not be omitted when adapting the idea.

For this study, reconstruction moments, continuous geometry evaluators,
reference qualification, provenance and streaming infrastructure can be shared
between branches. Their numerical identities and certification remain separate.
