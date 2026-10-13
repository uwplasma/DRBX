# Nodal FCI parallel operators — design contract

Status: **approved design direction, 12 October 2026; not implemented.** This contract defines the numerical method that
the [nodal FCI roadmap](parallel_fci_roadmap.md) qualifies. It replaces the owner-based GMLS / short-span traced Q
operators described in the archived [Q00–Q09 roadmap](parallel_second_order_roadmap.md). The decisions recorded
on 12 October 2026 are in §11.

## 1. Decisions this contract implements

Approved by the user on 12 October 2026:

1. **One node layout for P and Q.** The parallel operators act on P's nodal layout (family A). Q's owners,
   agglomeration, raw/owner returns, support selection and moment reconstruction are retired.
2. **The finite-difference interpretation, as in P.** The state is nodal point values. Interpolation uses fixed
   Lagrange stencils. There are no least-squares fits, cell averages or moments.
3. **Standard plane-to-plane FCI.** Field lines are traced from each node to the neighbouring η planes, the value is
   interpolated in that plane only, and a finite difference is taken along the line.
4. **A node is a parallel boundary node only if its trace hits the wall,** separately for each direction. The
   boundary condition is applied at the crossing point using the parallel distance to it (Leg Value Fill, as in Hill,
   Shanahan & Dudson 2017). Nodes whose traces stay inside never see wall data.
5. **Finite differences plus the support-operator method.** Divergence and parallel diffusion are the adjoints of the
   gradient in P's diagonal norm H (Stegmeir et al. 2016). Parallel diffusion is then H-symmetric and dissipative,
   consistent with P's SBP perpendicular operators.
6. **Core disk:** points inside it are evaluated with P's own core representation (Zernike basis, projector Π).

## 2. Node layout

The parallel operators use the unchanged P layout from `drbx.geometry.nodal_families.build_family_a_layout(n, n_eta)`
(see `drbx.geometry.nodal_layout`):

- **Rings.** Rings sit at `u_i = (i + 1/2)/n` for `i = K … n−1`, with `n` Fourier nodes per ring at
  `θ_j = π/n + 2πj/n`.
- **Core.** The core block covers `u ≤ R_c = K/n`, with `K = n/8`. It has degree `p = K + 2`, polar Gauss nodes and
  a Zernike basis (`drbx.geometry.sbp_core`).
- **Planes.** Planes sit at `η_k = (k + 1/2)Δη`, with `Δη = 2π/n_eta` and `n_eta = n` initially. η covers the full
  torus (four HSX field periods) and is periodic.
- **Half-planes**, used by the support operator: `η_{k+1/2} = (k + 1)Δη`. They carry the same in-plane layout.
- **Norm.** `H = Hp·Δη` on planes and `H½ = Hp½·Δη` on half-planes, with `Hp = wxy·|J_block|` evaluated at each node,
  exactly as P defines it (the Cartesian core block carries `|J|/u`). H is diagonal.
- **The wall** is `u = 1`, half a cell outside the last ring.

## 3. Magnetic field and field-line maps

**Field arms.** As in P, two field arms are kept:
- the raw compact C3 evaluator;
- P's η-filtered field, with J·B^i Fourier-filtered to m ≤ 3 per field period (table `e61e1d70…`).

Each map artifact records its arm.

**Trace equation.** In logical coordinates, with η as the independent variable:

```text
dX/dη = B^X / B^η   (X = u, θ),      dl/dη = |B| / |B^η|
```

- The integration is done in Cartesian `(x, y) = (u cos θ, u sin θ)` so that it is regular at the axis, as in the
  existing Q trace kernel.
- Preparation requires `B^η ≠ 0` and records its sign. If B^η changes sign, preparation fails.
- RK4 with `S` steps per hop. Each map is checked by step doubling: endpoints and lengths must agree within a
  recorded tolerance. Tolerances are absolute and relative, never float-byte hashes (portable identity rule).

**Two map sets.**

| Map | From | To | Used by |
|---|---|---|---|
| `M_full` | plane nodes | planes `k ± 1` (`±Δη`) | collocated gradient `G` (§5) |
| `M_half` | half-plane nodes | planes `k`, `k+1` (`±Δη/2`) | support-operator gradient `G½` (§6) |

**One map entry**, for each node and direction σ = ±, records:
- the node identity and direction;
- either the in-plane endpoint `(u*, θ*)` and arc length `l_σ`;
- or, if the trace reaches `u = 1` first, the first crossing `(θ_w, η_w)`, the arc length `l_w` to it, and the
  incidence `b·n̂` together with the angle between the line and the wall.

**First-hit termination.** A trajectory stops at its first wall crossing. Later re-entry is never used, and
extended-chart points outside the domain never supply values.

**Classes.** Each node is classified per map as `none`, `forward`, `backward` or `both`. Both directions hit on lines
with a short connection length.

## 4. Interpolation at an endpoint (`I`)

The value `f(u*, θ*)` on the endpoint plane comes only from that plane's nodal values. `I` is a fixed linear operator,
built once, with an exact transpose `Iᵀ` (needed by the support-operator adjoints). The tensor arm is stored as a
sparse matrix; the Fourier arm is applied matrix-free (§4 primary arm).

**Ring region, two interpolation arms (decision D7).**

- **Primary arm: Fourier in θ × cubic Lagrange in u.** This matches P's θ representation.
  - On each of the 4 rings in the radial stencil, the ring's n nodal values are interpolated trigonometrically at θ*.
    This is exact on P's band-limited ring representation.
  - The 4 ring values are then combined with 1-D cubic Lagrange weights in u.
  - Each row uses `4n` values, and the weights include negative values (Dirichlet kernel).
  - **Application (user decision, 13 October 2026).** The definition above is applied in one of two ways:
    - **`oversampled` (default, σ = 8, m = 8).** Each ring is upsampled exactly by FFT zero-padding to σn uniform
      points (Nyquist split symmetrically), then evaluated at θ* with an m-point local Lagrange stencil on the fine
      ring, combined with the same radial cubic.
      - Cost: O(σn log σn) per ring plus O(4m) per target, so O(n³ log n) per full grid. Storage is O(1) per target.
      - Constants and node values are reproduced exactly, and the transpose is exact (stencil scatter, then the
        transpose of the upsampling).
      - Difference from `exact`: 1e-14 relative on resolved modes, about 3e-7 at k = n/2 − 1, and ≤ 4e-7 on the
        stress waves at the real QF01 endpoints (QF02 task 03).
    - **`exact` (option).** The Dirichlet-kernel row `D(θ* − θ_j)` is stored per target and applied in real
      arithmetic after the radial combination. Cost: O(n) per target and O(n⁴) per full grid; storage is O(n) per
      target, about 5 GB per map at N128.
  - **Differentiation caveat.** `oversampled` is a piecewise-polynomial stencil in θ*. It is continuous where the
    stencil shifts, but ∂/∂θ* jumps there by about the size of its difference from `exact`. Derivatives with respect to
    the endpoint positions are therefore only piecewise smooth: geometry or shape sensitivities, or anything that
    differentiates through the maps. Use `exact` for those. Derivatives with respect to the nodal values are exact for
    both methods, because both are fixed linear operators with exact transposes.
- **Comparison arm: tensor cubic Lagrange (4 × 4)**, with the escalation rule below.

**Tensor cubic Lagrange (comparison arm).**
- Use 4 consecutive rings and 4 consecutive θ nodes around `(u*, θ*)`, periodic in θ.
- The weights are products of 1-D Lagrange weights, so the interpolant is exact for tensor polynomials of degree ≤ 3.
- **Near the wall:** the radial stencil is shifted inward to rings `n−4 … n−1`. Endpoints with `u_{n−1} < u* < 1`
  extrapolate by at most half a cell, from interior data only. This is a measured exception, not a wall condition.
- **Near the core interface:** the radial stencil is shifted outward to rings `K … K+3` while `u* ≥ R_c`.

**Core disk (`u* < R_c`): P's core representation.**
- Evaluate `f(x*, y*) = Z(x*, y*)·c`, where `c` are the Zernike coefficients of P's H-orthogonal projection Π of the
  core nodal values (`sbp_core.zernike_vandermonde`, `shell_projector`).
- This is exact on polynomials of total degree ≤ p and matches how P represents a field in the core.

**Degree.** Cubic in u is the minimum that keeps both G and the diffusion second order (§9). Both arms share the
radial stencil and its shifts near the wall and core.

For the comparison arm only: if N−O is the larger part of N−R on the finest interval in QF03/QF04, escalate to tensor
quintic (6 × 6). That is a new prepared plan with its own identity, not a runtime switch.

## 5. Collocated parallel gradient `G` (planes → planes)

**Interior.** For a node `i` with `M_full` endpoints at signed arc positions `−l₋` and `+l₊`, take
`f₋ = (I₋ f)_i` and `f₊ = (I₊ f)_i`. Then

```text
(G f)_i = − l₊/(l₋(l₋+l₊)) f₋ + (l₊−l₋)/(l₋ l₊) f_i + l₋/(l₊(l₋+l₊)) f₊
```

This is the non-uniform three-point formula, second order along the line.

**Boundary legs: one generic constraint-quadratic row (decisions B1–B6, 13 October 2026).** Every row of `G`,
interior or boundary, is the slope at the node (`l = 0`) of the quadratic `q(l) = q₀ + q₁ l + q₂ l²` fixed by three
linear conditions along the line. `G_i = q₁ = e₁ᵀ M⁻¹ c`, so the row weights are one row of a 3×3 inverse. A condition
is one of:

- **State values:** an interpolated endpoint value `q(−l₋) = f₋` or `q(+l₊) = f₊`, or the node value `q(0) = f_i`.
- **Wall functionals** at a hit, at signed distance `±l_w`:
  - Dirichlet `q(±l_w) = f_b`;
  - Neumann `q′(±l_w) = g∥` (decision D1);
  - Robin `α q(±l_w) + β q′(±l_w) = r`, included now for the sheath conditions and tested with D and N only.

**Condition sets by case (before fallback):**

| Case | Conditions | Order |
|---|---|---|
| no hit | `f₋`, `f_i`, `f₊` | 2 (the non-uniform three-point formula) |
| one direction hits | the in-domain endpoint, `f_i`, the wall functional | 2 |
| both directions hit | `f_i` and the two wall functionals | 2. Neumann on both sides reduces to linear interpolation of `g₋`, `g₊` and does not involve `f_i`. |

**Conditioning fallback (B2).**
- **Why.** A value-type wall condition at a very short leg puts a weight of order `1/l_w` on the state, mainly on
  `f_i`. That makes the operator stiff and amplifies state error. Large weights on exact data are harmless.
- **Trigger.** A row falls back when any of its **state** weights exceeds `C_w/l_nom`. `C_w` is set so that a pure
  Dirichlet one-sided leg trips at exactly `l_w = ε_l·l_nom` (D6, `ε_l = 0.01`). Neumann conditions never trip, because
  the row tends smoothly to `g∥` as `l_w → 0`. Robin conditions trip according to their actual weights.
- **What replaces a flagged row.** The flagged wall point is within `ε_l·l_nom` of the node, so its exact datum stands
  in for the node and `f_i` is dropped:
  - **one direction hits (B3):** the linear fit through the in-domain endpoint and the wall condition, for example
    `(f_b − f₋)/(l₋ + l_w)`. First order, with weights of order `1/l`;
  - **both directions hit (B4):** the linear fit through the two wall conditions,
    `(f_b₊ − f_b₋)/(l_w₋ + l_w₊)`. If one side is Neumann, `G = g∥` of that side. First order, and data only.
  - **Switching (B5):** at the threshold the switch is hard, with no blending.
- **Second-order option (not enabled).** Replace `f_i` by a second in-domain endpoint `f₋₂` at `−l₋₂`, one extra
  backward hop, and keep a quadratic. This is to be enabled only if the first-order fallback rows cause a serious order
  stall. Its main use would be to allow a larger `ε_l` (decision C5, on hold).
- **Reporting.** Per grid, report the counts of rows by case and by fallback, their error shares, and the largest state
  and data weights.

**Geometry of short legs.** For a node at normal distance `d` from the wall, `l_w ≈ d / sin α`, where `α` is the
field-line incidence angle. Wall-ring nodes sit half a cell from the wall, which gives
`l_w/l_nom ≈ (a/4πR)/sin α ≈ 0.008/sin α`, independent of n at `n_eta = n`. So `l_w < 0.01·l_nom` needs steep incidence,
roughly α ≳ 50°. HSX lines mostly graze the vessel: the 1% quantile of `l_w/l_nom` was 0.06–0.08 on the sampled planes.

**Non-smoothness.** The hard switch (B5), like the hit/no-hit classification itself, makes `G` only piecewise smooth
in the geometry. Treat geometry or shape derivatives with the same care as the `oversampled` interpolation caveat (§4).

Wall data enter G only through these rows, and every boundary row is affine in `(f, f_b or g∥)`.

**Interpretation and use.** `G` approximates `b·∇f` at plane nodes. It supplies the gradient terms in the six-field
RHS, such as `∇∥p`. It also gives the control divergence `∇·(q b) = B ∇∥(q/B)` (§6), used for comparison only.

## 6. Support-operator gradient, divergence and diffusion

**Half-plane gradient.** For a half-plane node `j`, `M_half` gives endpoints on planes `k` (at `−l₋`) and `k+1` (at
`+l₊`). Then

```text
(G½ f)_j = (f₊ − f₋)/(l₋ + l₊)
```

This is second order at the half-plane node, because `l₊ − l₋ = O(Δη²)`. Boundary half-plane nodes use LVF in the
same way, through a second-order fill or one-sided form. The exact boundary rows are part of QF04 and must keep each
row affine and second order.

**Divergence and diffusion.**

```text
D q        = −H⁻¹ G½ᵀ H½ q  (+ wall-flux terms from hit legs)          half-planes → planes
L_∥ f      =  D (χ½ G½ f)                                              parallel diffusion
```

- `χ½` is the conductivity at half-plane nodes.
- With homogeneous wall data, `H L_∥ = −G½ᵀ H½ χ½ G½`, which is symmetric and negative semidefinite. This is the
  parallel counterpart of P's SBP identities, and it holds to roundoff by construction.
- **Conservation:** `1ᵀ H D q = −(G½ 1)ᵀ H½ q = 0` up to the wall-flux terms, because `G½ 1 = 0` exactly.
- **Accuracy of the adjoint divergence** relies on `H` carrying the physical volume `|J|`, so that the transpose
  approximates `∇·(q b) = B ∇∥(q/B)`. This is measured, not assumed: QF04 compares `D` against the control
  divergence built from `G`.

**Values on half-planes.** Plane fields are carried to half-planes by interpolation along the line:
`q½ = (l₊ I₋ q + l₋ I₊ q)/(l₋ + l₊)`, using the `M_half` endpoints. This is second order.

## 7. Boundary data and the MMS conventions

- **Dirichlet data:** the exact manufactured value at the crossing point `(1, θ_w, η_w)`.
- **Neumann data (D1, decided):** the parallel derivative `g∥ = b·∇f` at the crossing, because only the field-line
  leg enters the parallel operator. The previous Q catalogue used the physical-normal derivative
  `g_N = (g^{uj}/√g^{uu})∂_j f`, which remains the perpendicular (P) Neumann datum.
- **Manufactured fields:** the retained Q catalogue (§10 of the roadmap), evaluated pointwise at nodes and at the
  crossing points.
- **Field-wise patterns:** the D/N patterns from Q09 (DDDDDD/phiD, NNNNNN/phiN, DNDNDN/phiN, NDNDND/phiD), with the
  letters in the order n, Te, Ti, Vi, Ve, ω.
- **Out of scope until QF05/QF06:** inflow/outflow (characteristic) selection for the hyperbolic terms (D3), and
  physical sheath laws at the crossing.

## 8. Reference, error decomposition and norms

- **R:** the analytic continuum operator evaluated pointwise at the nodes, as in P. An independent continuum reference
  provides it, never numerical stencils.
- **N:** the discrete operator applied to exact nodal values, with exact wall data.
- **O:** the same along-line stencils and LVF rows, but every interpolated endpoint `I_σ f` is replaced by the exact
  field value at the traced endpoint.
  - **N−O** is the in-plane interpolation error.
  - **O−R** is the along-line and LVF error, plus the tracing error.
- **Norms.** `E = sqrt(Σ H e² / Σ H)`. Orders are `p = ln(E_c/E_f)/ln(N_f/N_c)` on N32→48, N48→64 and N32→64.
- **Report** N−R as the headline together with N−O and O−R, globally and by region.
- **Regions:** a disjoint partition (core block; interior rings; outer four rings), plus per-map trace-class bins
  (`none` / `forward` / `backward` / `both`) and bins of wall distance `l_w`. The maximum norm is reported as a
  diagnostic.

## 9. Expected orders (design basis)

On the isotropic sequence, with in-plane spacing Δ ~ Δη, a degree-q interpolant has error `O(Δ^{q+1})`:

| Operator | Interpolation contribution | Along-line contribution | Expected |
|---|---|---|---|
| `G`, `G½` | `O(Δ^q)`; cubic gives 3 | `O(Δη²)` | 2 |
| `L_∥ = D χ G½` | `O(Δ^{q−1})`; cubic gives 2 | `O(Δη²)` | 2 |
| `D` (adjoint) | measured, QF04 | measured | 2 required |

Bilinear interpolation (q = 1) loses order, matching Hill et al. 2017. Cubic is therefore the minimum for second-order
diffusion.

## 10. Structural checks (prepared once, not in every preflight)

- **Reproduction:**
  - `I` reproduces polynomials of degree ≤ 3 in the rings and `P_p` in the core.
  - `G 1 = 0` and `G½ 1 = 0` exactly.
  - LVF rows are exact for quadratics along the line.
- **Maps:**
  - `M₊ ∘ M₋ ≈ identity`, to the trace tolerance.
  - Step-doubling convergence.
  - Every node is classified, and no endpoint lies outside the domain.
- **Support operator:**
  - `H L_∥` is symmetric to roundoff.
  - `λ_max(H L_∥) ≤ roundoff`.
  - The conservation identity holds.
  - Wall-data linearity holds.
- **Portability:** identities are checked to tolerance. A float-byte SHA is provenance only.

## 11. Decisions (12–13 October 2026)

| ID | Question | Decision |
|---|---|---|
| D1 | Neumann datum at the wall crossing | **Parallel derivative** `g∥ = b·∇f`. P keeps the physical-normal `g_N`. |
| D2 | Where velocities live | **Collocated** (user, 12 Oct): all state stays on P's plane nodes. Half-planes serve only as internal flux points for `G½` / `D` / `L_∥` and hold no state. Switch to staggered Vi/Ve/j∥ only if collocated checkerboard modes cannot be controlled in QF05. |
| D3 | Stabilising hyperbolic terms; wall data | **Static stages (QF03/QF04):** data on every hit leg. **QF05/QF06:** compare interior (a) centred against (c) upwinding, and use characteristic (inflow-only) data at the wall, imposed weakly in SAT form consistent with P's wall characteristic SAT. The user leans toward upwinding both in the interior and at the wall, if any stabilisation is used; decided on evidence. Explicit 4th-order dissipation is not considered. |
| D4 | Positivity / minimum principle for homogeneous diffusion | **Reported diagnostic,** not a gate. Evolved n and T must stay positive (QF06). Limiters only if the evidence demands them. |
| D5 | Field arm gating | **Gate the η-filtered arm, report the raw arm,** then decide on what is observed. |
| D6 | Small-leg cutoff | **`ε_l = 0.01`** (BOUT++), implemented through the state-weight trigger of B2. Raising it, together with the second-order fallback, is on hold (13 Oct). |
| D7 | Interpolation in the ring region | **Primary: Fourier in θ × cubic in u** (matches P). **Also run:** tensor cubic Lagrange with the quintic escalation rule. |
| D8 | `n_eta` | **Provisional (user, 13 Oct):** `n_eta = n` for development, while the field-aligned accuracy is acceptable. The publication MMS uses `n_eta = c·n` (c ≈ 2–4, chosen from the field-aligned results), refining in-plane and toroidal resolution together. Evidence: [Δη refinement](../../../../work/qf01_maps_20261012/refine.json). Comparable stellarator FCI/GBS runs use about 25–40 planes per period, against 16 at N64. |
| B1 | Boundary row framework | **One generic constraint-quadratic row** (§5) with value, derivative and Robin wall functionals (13 Oct). |
| B2 | Fallback trigger | **State-weight bound** `C_w/l_nom`, calibrated to `ε_l` for pure Dirichlet. Neumann never trips. |
| B3 | One-sided short-leg fallback | **First-order linear fit** through the in-domain endpoint and the wall datum. The second-order extra-hop form is kept in reserve for an order stall. |
| B4 | Both-hit short-leg fallback | **First-order, data only:** a linear fit through both wall data, or `g∥` of the Neumann side. |
| B5 | Switching | **Hard switch** at the trigger. |
| B6 | Robin | **Included now** in the framework and tested later, with the sheath conditions. |

## 12. Implementation boundaries

- **Preparation** (maps, `I`, LVF rows, `H`, region bins) is host NumPy, built once, saved with identities, and
  replayed.
- **Runtime** applies fixed linear operators to the state (a JAX sparse matvec, or per-ring FFTs plus fixed
  evaluation weights for the Fourier arm), with no geometry evaluation at RK stages.
- **Location:** research harnesses come first under `scripts/qf*/`; the package extraction comes last (consolidate
  first, integrate at the end).
- **Reuse** without editing: P's layout, core and norm modules; the HSX metric and magnetic evaluators; the continuum
  reference conventions.
- **Frozen paths stay frozen:** Q07–Q09 runtimes, `hsx_mms_continuum_reference.py` and the P campaign sources.

## References

- P. Hill, B. Shanahan, B. Dudson, *Dirichlet boundary conditions for arbitrary-shaped boundaries in stellarator-like
  magnetic fields for the FCI method*, Comput. Phys. Commun. (2017), arXiv:1608.02416. Trace-hit boundary set and LVF.
- A. Stegmeir et al., *Field line map approach for simulations of magnetically confined plasmas*, Comput. Phys.
  Commun. 198 (2016), arXiv:1505.02040. Support-operator method.
- T. Body, A. Stegmeir et al., *Treatment of advanced divertor configurations in the FCI turbulence code GRILLIX*,
  Contrib. Plasma Phys. (2020), arXiv:1908.05398. Penalisation built from traced angles.
- M. Wiesenberger, M. Held, *A finite volume flux coordinate independent approach*, Comput. Phys. Commun. (2023),
  doi:10.1016/j.cpc.2023.108838. A conservative alternative, not selected.
- J. Bold, B. Shanahan, *Numerical methods for stellarator simulations in BOUT++* (2026), arXiv:2603.28221. Fallbacks
  for short connection lengths and small legs.
