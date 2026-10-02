# Q06 assignment: bounded parallel gradient/divergence certification

This assignment authorizes one bounded real-HSX study at N32, N48, and N64.
It reuses the frozen Q05a owner selections, raw owner memberships, direct-cubic
support policy, and actual geometry evaluator. It does not authorize a global
campaign, evolved solve, remote run, Q07 work, or production selector change.

Durable numerical evidence belongs in the bounded
[Q06 report](../../../../work/parallel_q06_gradient_divergence_20260922/report.md).

## Continuum and discrete contracts

Let `b = B/|B|` in logical coordinates. The continuum operators are

```text
G f = b dot grad(f)
D g = div(b g) = J^-1 partial_alpha(J b^alpha g)
    = G g + g div(b).
```

The diffusion continuum is `D(G f)`, but Q06 must not force the independently
fitted discrete `G` and conservative discrete `D` to factor the qualified Q05a
diffusion action.

| Functional | Location | Target | Observation and support | Boundary |
|---|---|---|---|---|
| `D g` | canonical oriented face to owner incidence | `lambda[p] = integral_face (b dot n) p dS`, q9 initially | frozen raw-midpoint physical-volume owner averages; 120 owners on five eta planes; incident owners retained | replace the fitted boundary row exactly once with the dynamic analytic trace `g` integrated against `b dot n` |
| `G f` | compact owner | `gamma[p] = V^-1 integral_owner b dot grad(p) dV`, q9 initially | same observation and support policy, owner-centred geometry-only support; evaluated owner retained; all raw members integrated | no separate face trace |
| `g div(b)` | compact owner | `delta[p] = V^-1 integral_owner p div(b) dV`, q9 initially | same owner-centred observation and support policy | no separate face trace |

The product-identity check compares conservative `D g` with independently
fitted `G g + <g div(b)>`; it must not substitute a product of owner averages.
The primary geometry definition is the direct derivative
`div(b) = J^-1 partial_alpha(J b^alpha)`. The identity
`div(b) = -b dot grad(log |B|)` is only a cross-check.

The axis face is zero measure and contributes zero flux. All nonzero raw-cell
members of a selected aggregate owner participate in volume integration.
Physical wall traces stay analytic and dynamic. Production incidence signs,
metric factors, and runtime boundary replacement are audited without changing
any production selector.

## Required stages and gates

1. Correct the Q05a replay gate from saved norms: divide replay L2 by the
   corresponding accepted static spatial error, with no numerical rerun.
2. Freeze versioned functional identities naming target kind, integration,
   evaluation location, geometry/topology/evaluator hashes, precision, and
   boundary convention.
3. Run one small preflight before the three bounded resolutions. Use the three
   frozen manufactured fields plus the constant field and independent analytic
   field derivatives.
4. Qualify q9 targets against a higher quadrature check and record fit rank,
   defect, condition, coefficient L1, support extent, and constant behavior.
5. Check face fluxes, incidence assembly, owner `G`, owner `D`, `div(b)`, the
   product identity, and `D(G f)` separately. Report regional values for axis,
   agglomerated bulk, transition-adjacent, ordinary interior, boundary, theta
   seam, and eta seam.
6. Report only bounded sample trends across N32/N48/N64; do not claim global
   convergence order. Pairing is diagnostic and must not be imposed by
   symmetrization or an adjoint replacement.

Peak RSS must remain below 10 GiB. Tests must use actual HSX data and retain the
small synthetic shared-face regression. Completion requires a reproducible
comparison and a recommendation limited to either targeted correction or a
separately scoped global `G`/`D` qualification.
