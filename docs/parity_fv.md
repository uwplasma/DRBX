# Parity-FV Rewrite Track

This page documents the new `parity_fv` engine, which is the strict
Hermes-parity rewrite path.

## Scope (current)

Implemented modules:
- `/Users/rogerio/local/jax_drb/src/jaxdrb/parity_fv/params.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/parity_fv/state.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/parity_fv/geometry.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/parity_fv/flux_reconstruct.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/parity_fv/flux_parallel.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/parity_fv/poisson_vorticity.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/parity_fv/rhs.py`

## Numerical policy

The rewrite follows Hermes/BOUT finite-volume semantics first, then extends
physics only after parity gates pass.

### Parallel transport kernel

The current parallel kernel uses limited reconstruction plus Rusanov/Lax form:

\[
\Gamma_{f,i+1/2} = \frac12\left(f_L v_L + f_R v_R\right)
+ \frac12 a_{\max}(f_L - f_R),
\]

with

\[
(\nabla_\parallel f)_i \approx
\frac{\Gamma_{i+1/2} - \Gamma_{i-1/2}}{\Delta z}.
\]

This matches the Hermes documentation in
`solver_numerics.rst` (slope-limited FV + Lax term).

### Poisson/vorticity guard-cell semantics (new)

From Hermes `vorticity.cxx`, parity path now mirrors these boundary semantics:

1. **INVERT_SET midpoint guard rule** for `phi + Pi_hat` at radial guards:
\[
(\phi+\Pi)_{x_g} \leftarrow \tfrac12\left[(\phi+\Pi)_{x_g} + (\phi+\Pi)_{x_{in}}\right].
\]
2. **Outer radial guard fill** after solve by copy from nearest guard.
3. **Parallel free-guard update** used for derivative stencils:
\[
\phi_{y_g} = 2\phi_{y_{in}} - \phi_{y_{in\pm1}}.
\]

Implemented in:
- `prepare_phi_plus_pi_for_poisson(...)`
- `finalize_phi_after_poisson(...)`

## Tests

Added parity-fv tests:
- `/Users/rogerio/local/jax_drb/tests/test_parity_fv_scaffold.py`
- `/Users/rogerio/local/jax_drb/tests/test_parity_fv_parallel_flux.py`
- `/Users/rogerio/local/jax_drb/tests/test_parity_fv_poisson_vorticity_guards.py`

These tests verify reconstruction, FV boundary-flux balance, and guard-cell
rules mapped directly from Hermes vorticity solver behavior.

## Next steps

- Integrate `parity_fv` guard semantics into the new vorticity/Poisson solve path.
- Add one-step Hermes vs JAX parity gate that uses `parity_fv` operators only.
- Promote `engine = "parity_fv"` into CLI once term-level short-window gate passes.
