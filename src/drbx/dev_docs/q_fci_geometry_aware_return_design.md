# Q: geometry-aware FCI return basis and support contract

Status: research candidate supported by a bounded N32 comparison; not a
selectable production operator or global convergence qualification. The
[parallel roadmap](parallel_second_order_roadmap.md) remains the progress and
acceptance authority.

## Evidence and design decision

Use the magnetic geometry inside the return basis, and select enough
independent traced-leg observations to determine the required face functional.
These are separate requirements: neither a suitable basis with inadequate
support nor a large support with an unsuitable basis worked in the comparison.

The [cached comparison report](../../../../work/parallel_fci_return_basis_comparison_20260923/report.md)
records the implementation, inputs and results. Its drivers are
`work/parallel_fci_return_basis_comparison_20260923/run.py` and
`support_controls.py` relative to the workspace. They reuse the Q worker's
`work/parallel_fci_shared_face_return_quadratic_20260923/` bundle.

On nine ordinary N32 owners, using identical 96-row supports, the numerical
geometry-aware return reduced physical-volume-weighted operator errors by
approximately 8–51 times relative to a free quadratic flux fit. Its m1 errors
were `[4.46e-4, 5.64e-4, 5.56e-4]` and m2 errors
`[2.07e-4, 9.14e-4, 1.41e-3]` for radial, angular and mixed fields.
This motivates the next bounded robustness experiment. It does not establish
order, a globally sufficient stencil, or structural properties beyond the
shared-face conservation already checked.

## Basis and matching functionals

Let the physical parallel flux scalar be

\[
a=\boldsymbol b\cdot\nabla T.
\]

Use locally centered and scaled regular coordinates
`(x=u cos(theta), y=u sin(theta), unwrapped eta)` and total-degree cubic
potential polynomials `p_j`. The candidate flux basis is

\[
\psi_j=\boldsymbol b\cdot\nabla p_j.
\]

There are 20 cubic monomials in three dimensions. Remove the constant-potential
column, whose derivative is exactly zero, leaving 19 columns. This is not a
free cubic fit of the flux. A cubic potential does not produce a quadratic
physical parallel gradient when the magnetic geometry varies; the basis
retains that known variation explicitly.

In logical coordinates, the directional derivative must include the chart
transformation. For example,

\[
b_x^{\rm chart}=b^u\cos\theta-u b^\theta\sin\theta,
\quad
b_y^{\rm chart}=b^u\sin\theta+u b^\theta\cos\theta,
\quad
b_\eta^{\rm chart}=b^\eta.
\]

Apply the local coordinate scales when differentiating the monomials. Evaluate
continuous magnetic geometry at the actual trajectory and face quadrature
nodes. Do not replace it by a nominal cell-center value.

For each existing arc-gradient observation, retain its functional:

\[
\lambda_r(a)=\frac{1}{Z_r}\sum_q F_{rq}\int_{\mathrm{leg}_{rq}}a\,ds,
\qquad Z_r=\sum_q F_{rq}\ell_{rq},
\]

with the frozen source magnetic-flux weights
`F = source_quadrature_weight * J0 * B0 * abs(b0^eta)`.
Build the observation matrix and oriented face targets as

\[
A_{rj}=\lambda_r(\psi_j),\qquad
 t_{fj}=\int_f J b^\alpha\psi_j\,d\xi^\beta d\xi^\gamma.
\]

The bounded comparison uses positive `b^eta` and the recorded forward/backward
leg orientation. On those legs the numerical observation is
`d * sum_q F_q (T_endpoint - T_source) / Z`, using the established cubic
endpoint reconstruction. Preserve and verify orientation when extending the
selection; do not assume a new trace has the same sign convention by accident.

A scaled, weighted rank-revealing solve provides a linear observation-to-face
map `w_f = t_f A_W^+`. The physical face flux is `w_f g`, where `g` contains
numerical FCI gradient observations. Assemble one canonical flux per face,
with opposite incidences on neighboring cells and the established physical
owner-volume normalization. Include all incident raw faces needed by each
complete owner. Eta, radial and theta faces all contribute to divergence on
the fixed HSX grid.

The candidate remains FCI: traced endpoint differences provide its numerical
data. Exact gradients at trajectory points are an oracle control only. No
manufactured state or reference answer enters the geometry-only face map.

## Adequate support: requirements and diagnostics

### What is being counted

The 24/48/96 counts refer to **traced-leg observation rows**, not owner-cell
values, endpoint donors, polynomial coefficients or face quadrature nodes.
Each row has its own source footprint, traced interval and endpoint
reconstruction. Adequate endpoint donor support and adequate return-row
support are distinct issues.

### Required selection behavior

1. Choose support by geometry and observation functionals before inspecting
   manufactured-field errors. Use the same policy for every field.
2. Include independent transverse coverage and distinct along-field intervals.
   Reversing a leg, repeating quadrature, or changing m1 to m2 does not by
   itself supply a new longitudinal interval. The original two-interval
   catalogue failed to constrain a quadratic eta mode; the outward intervals
   repaired that particular defect.
3. Use deterministic scaled distances, ties, interval accounting and a bounded
   neighborhood expansion schedule. Let the selected count vary with geometry
   and resolution. Record row identities and physical/scaled support extent.
4. Use a rank-revealing solve and inspect the face functional's sensitivity to
   weak observation modes. The necessary question is whether the observations
   determine `t_f`, rather than whether every coefficient can be identified.
   If `A v=0` but `t_f v` is nonzero, that support cannot determine the face
   flux. If both vanish, coefficient nonuniqueness alone is not a rejection.
5. Do not stop solely because nominal full rank or polynomial reproduction is
   achieved. Record target reproduction, singular values and observation-to-
   face amplification; these expose near-null modes that matter to the output.
   Compare bounded support expansions during qualification to check that actual
   completed actions are robust. The eventual runtime support policy must use
   geometry/algebraic information, not MMS errors.

No universal condition-number cutoff, donor count or new per-cell accuracy
acceptance rule is established here. Conditioning is a diagnostic connected
to output sensitivity. The roadmap's HSX global accuracy contract remains the
scientific acceptance criterion.

### Observed support sensitivity

With the geometry-aware basis, the initial 24–27-row supports had maximum
condition numbers of roughly 3–4 million and poor actions despite small
polynomial reproduction residuals. At 48 rows the maximum was about 28,000;
at 96 it was about 4,800–7,500, with much smaller operator errors.

Ninety-six rows are a reproducible **comparison setting**, not a production
minimum or proof of support sufficiency everywhere. Larger support also spans
more variation, costs more and can increase approximation bias; expansion is
not guaranteed to improve accuracy indefinitely. The free quadratic flux
control actually worsened under some support expansions.

m1 and m2 catalogues have different available rows and selected supports.
Keep their results separate. Their present difference is not a clean
transverse-refinement order measurement.

## Next bounded implementation and verification

### 1. Freeze and replay the successful comparison

Record the basis, arc observation convention, orientation, coordinate scales,
weights, support expansion, geometry evaluator and input identities. Replay
both m1/m2 96-row actions on the existing patch. Retain the free-quadratic fit
on identical supports as a comparison. Preserve the existing sources and
results unchanged; put new evidence in a fresh folder.

Check the directional polynomial derivatives and the **target face map** on
known polynomial-potential inputs. Compare exact-arc, exact endpoint secant
and numerical endpoint actions separately. Numerical endpoint differences are
not assumed identical to continuous arc integrals at finite quadrature.
Reuse saved exact face and volume references where their uncertainty is small
enough for the comparison; refine only a bounded reference check if necessary.

### 2. Test limited geometric diversity

Select one additional ordinary-interior owner, one agglomerated-interior
owner and one RLP-transition owner at varied eta locations, using geometry
alone. Resolve complete raw membership and incident faces before evaluating
errors. Reuse shared observations and cached traces where possible; preflight
the cost of the missing trace closure before expanding the computation.

Use the existing 48/96-row settings as controls and inspect target sensitivity
and support expansion. Derive a reproducible geometry-only support policy
from these diagnostics. Keep m2 as the main comparison and m1 as a diagnostic;
the current patch does not establish that either catalogue is universally
better. New axis and physical-boundary treatment remain explicit later
coverage requirements, with the roadmap's existing boundary convention.

Report component errors and completed owner actions for every field. No rule
requires each owner to improve or each diagnostic to meet the global order
target. The decision is whether improvement is sufficiently robust to justify
a broader qualification, or whether a reproducible defect needs another
specific repair.

### 3. Move toward global qualification

If the bounded extension is promising, freeze the candidate and support policy,
measure preparation/application cost, and plan the 32/48/64 HSX campaign with
explicit whole-domain axis/boundary coverage. Certification requires the
existing global operator order gate for every nontrivial field; the present
patch errors and conditioning reductions cannot substitute for it. Conservation,
energy behavior, positivity, evolved checks and final package/model integration
retain their separately recorded roadmap status.

This document authorizes no worker dispatch or numerical campaign by itself;
it records the current design and proposed next bounded task.
