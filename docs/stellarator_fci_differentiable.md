# Differentiable FCI on a non-axisymmetric flux tube (stellarator)

The stellarator-side Phase 6 flagship shows the paper's two core claims -- the
flux-coordinate-independent (FCI) discretization *and* end-to-end
differentiability -- on a non-axisymmetric flux-tube geometry, using the reduced
drift-reduced two-field FCI model (`dkx.native.fci_2_field_rhs`):

```
d/dt n = -{phi, n}/(rho* B) + (2/B) K(n) - (2 n/B) K(phi) - n grad_par(v_par)
d/dt v_par = -{phi, v_par}/(rho* B)
```

with `phi = log(n / n0)` derived internally, the `E x B` Poisson bracket `{.,.}`,
the curvature operator `K`, and the direct parallel gradient `grad_par = b^i d_i`
built from the helical contravariant field. The whole right-hand side is JAX, so
a rollout is `jit`-compiled and differentiable.

## Geometry

The geometry is the shifted-torus metric promoted from the verified two-field MMS
scaffold into the package as
[`dkx.geometry.build_shifted_torus_geometry`](../src/dkx/geometry/shifted_torus.py).
Logical coordinates are `(x, theta, zeta)`; the poloidal angle is sheared,
`Theta = theta + sigma (x - x_mid)`, so the metric carries genuine off-diagonal
cross terms (`g12` / `g_12`) and the coordinate frame is non-orthogonal. The
field lines are helical (rotational transform `iota`).

Honest scope: this is a stellarator-relevant non-orthogonal helical flux tube,
not a full 3D equilibrium, and (as in the MMS scaffold) the field-following FCI
maps are placeholders (`construct_fci_maps=False`). The Poisson-bracket and
curvature operators use the full non-orthogonal metric; the parallel gradient is
the direct `b^i d_i` operator on the helical field. What is demonstrated is a
differentiable DRB operator stack on non-orthogonal helical geometry -- **not** a
claim of saturated stellarator turbulence.

## End-to-end differentiability

`examples/stellarator/fci_differentiable.py` seeds a smooth perturbation
(amplitude `amp`), holds simple fixed zero-Dirichlet fluctuation walls in the
radial direction (periodic in `theta`, `zeta`), and advances a short bounded RK4
rollout that stays finite. It then takes `jax.grad` of a scalar diagnostic of the
*evolved* state -- the total density variance -- with respect to `amp`, and
compares it to a central finite difference. The gradient flows through every RK4
stage and every FCI operator.

On the shipped `16x16x8`, `sigma = 0.6`, 24-step configuration the autodiff
gradient and the central finite difference agree to a relative error of order
`1e-10` (well inside the `1e-3` gate), and the figure overlays the autodiff
tangent on directly sampled `J(amp)` values so the match is visible. A cheaper
single-RHS gradient is reported as a secondary witness. The differentiation path
used is the **multi-step rollout**; the single-RHS check is the documented
fallback for the case where a free rollout is ill-posed (it is not here).

The gate is `tests/test_fci_differentiable.py`
(`test_rollout_grad_matches_finite_difference`), kept fast with a small grid and
two RK4 steps. The reusable case API lives in
`dkx.native.fci_differentiable_case`, which the tests import directly; the
example script is a flat driver over that API (no `main()` function).

## Reproduce

```bash
PYTHONPATH=src python examples/stellarator/fci_differentiable.py
```

writes `output/fci_differentiable/fci_differentiable.png` (evolved density
poloidal slice + the grad-vs-finite-difference tangent) and
`output/fci_differentiable/fci_differentiable_summary.json` (the geometry,
rollout, and gradient/FD/relative-error numbers). The fast gate is:

```bash
python -m pytest -q tests/test_fci_differentiable.py
```
