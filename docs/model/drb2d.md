# DRB2D nonlinear testbed

This page documents the 2D nonlinear DRB testbed used to validate conservative operators
before full field-line nonlinear DRB is introduced.

## Equations (slab, periodic)

We evolve five fields on a 2D periodic grid:

$$
Y = (n, \Omega, v_{\parallel e}, v_{\parallel i}, T_e),
$$

with electrostatic potential from a Poisson solve:

$$
\Omega = \nabla_\perp^2 \phi.
$$

The nonlinear equations are

$$
\partial_t n + [\phi, n] = -\nabla_\parallel v_{\parallel e} + C(p) - C(\phi) + S_n,
$$

$$
\partial_t \Omega + [\phi, \Omega] = \nabla_\parallel (v_{\parallel i} - v_{\parallel e}) + C(p) + S_\Omega,
$$

$$
\partial_t v_{\parallel e} + [\phi, v_{\parallel e}] = \nabla_\parallel (\phi - n - 1.71 T_e) - \eta (v_{\parallel e}-v_{\parallel i}) + S_{v_e},
$$

$$
\partial_t v_{\parallel i} + [\phi, v_{\parallel i}] = -\nabla_\parallel \phi + S_{v_i},
$$

$$
\partial_t T_e + [\phi, T_e] = -\tfrac{2}{3} \nabla_\parallel v_{\parallel e} + \tfrac{2}{3} C\left(\tfrac{7}{2} T_e + n - \phi\right) + S_{T_e}.
$$

Here the curvature operator is a simple slab interchange model:

$$
C(f) = -\omega_c\,\partial_y f.
$$

The source terms $S_\cdot$ include optional background-gradient drives and diffusion terms,
controlled by `DRB2DParams`.

## Notes

- The conservative subset sets drives, curvature, and dissipation to zero.
- The Poisson bracket is discretized using Arakawa's conservative Jacobian for periodic grids.
- This model is intended as a nonlinear verification milestone, not as a full SOL code.
- In the ideal limit with `vpar_e=vpar_i=Te=0` and `kpar=0`, the `(n, omega)` subsystem reduces
  to the ideal HW2D advection equations, providing a direct HW2D limit check.

## Examples

- Conservative energy gate: `examples/08_nonlinear_drb2d/drb2d_conservative_gate.py`
- Linear-phase benchmark: `examples/08_nonlinear_drb2d/drb2d_linear_phase_benchmark.py`
- Nonlinear movie: `examples/08_nonlinear_drb2d/drb2d_movie.py`
