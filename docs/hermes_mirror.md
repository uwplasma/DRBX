# Hermes Mirror Path

## Purpose

`jax_drb` now has a dedicated `hermes_mirror` translation path under
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror`.

This path exists for one reason: close Stage 1 Hermes parity by translating the
actual Hermes/BOUT operator stack into JAX function by function, instead of
continuing to patch the old approximation path in place.

The mirror path is temporary in architecture and strict in semantics:

- Hermes source files are the source of truth.
- Each mirrored function is landed with a direct source citation.
- Each mirrored function gets a tiny synthetic test before it is wired into an
  operator or engine path.
- Each operator is expected to have both:
  - a readable reference form, and
  - a fused production form suitable for JIT execution.

## Array Layout Contract

Hermes/BOUT fields are written conceptually as `(x, y, z)`. The active JAX
codebase uses `(nz, nx, ny)`. The mirror path keeps the JAX layout so that:

- it can be wired into the existing solver without repeated transposes,
- JIT compilation stays compatible with the current geometry and state layout,
- gradients stay end to end through the same array representation used
  elsewhere in `jax_drb`.

The translation rule is therefore:

\[
(x, y, z)_\text{Hermes} \longrightarrow (z, x, y)_\text{JAX mirror}
\]

Every mirror helper must document which Hermes indices it corresponds to.

## Implemented Phase 1 Primitives

### 1. `limit_free`

Source:
- `/Users/rogerio/local/hermes-3/src/sheath_boundary_simple.cxx`
- function `limitFree`

This helper constructs a free boundary extrapolation that does not create an
unphysical increase into the sheath and avoids invalid values at very low
density.

Let:

- \(f_m\): the cell one step inside the boundary,
- \(f_c\): the cell adjacent to the boundary,
- \(f_p\): the ghost/guard value beyond the boundary.

Hermes implements:

\[
f_p =
\begin{cases}
f_c, & \text{if } f_m < f_c \text{ and mode} = 0 \\
f_c, & \text{if } f_m < 10^{-10} \\
\dfrac{f_c^2}{f_m}, & \text{if mode} \in \{0,1\} \\
2 f_c - f_m, & \text{if mode} = 2
\end{cases}
\]

The exponential branch follows from linear extrapolation in \(\log f\):

\[
\log f_p = 2 \log f_c - \log f_m
\quad \Rightarrow \quad
f_p = \frac{f_c^2}{f_m}
\]

In JAX this is implemented with `jnp.where`, preserving differentiability in
the active branch while avoiding singular division in the low-density case.

### 2. `apply_neumann_boundary_average_z`

Sources:
- `/Users/rogerio/local/hermes-3/src/evolve_density.cxx`
- `/Users/rogerio/local/hermes-3/src/evolve_pressure.cxx`
- field option `neumann_boundary_average_z`

Hermes applies a radial Neumann boundary by first averaging the edge value in
the toroidal direction and then reflecting around that averaged midpoint.

For the lower radial boundary:

\[
\bar{f}_{x_s}(y) = \frac{1}{N_z} \sum_k f(x_s, y, k)
\]

\[
f(x_s - 1, y, k) = 2 \bar{f}_{x_s}(y) - f(x_s, y, k)
\]

\[
f(x_s - 2, y, k) = f(x_s - 1, y, k)
\]

The upper boundary uses the same rule at \(x_e\).

In the mirror path, the z-average is `axis=0` because arrays are stored as
`(nz, nx, ny)`.

### 3. `set_boundary_to_midpoint`

Source:
- `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/field/field3d.cxx`
- method `Field3D::setBoundaryTo(const Field3D&)`

This helper does not copy guard cells directly. It matches the boundary
midpoint of a target field `u` to that of a reference field `v`.

Hermes computes:

\[
\frac{u_g + u_i}{2} = \frac{v_g + v_i}{2}
\]

so:

\[
u_g = v_g + v_i - u_i
\]

This relation is recursive for the outer guard layers because the “interior”
cell for the outer guard is the previously written guard cell.

That recursion matters for parity: replacing it with a one-shot copy or a
vectorized edge-only update changes the actual guard values seen by later
operators.

## Differentiability Rules

The mirror path is required to stay end to end differentiable on the production
solver path. The implementation rules are therefore:

- use pure array functions that return new arrays,
- prefer `jnp.where`, `lax.cond`, and `.at[...]` updates over Python-side
  mutation,
- avoid host callbacks and side effects,
- keep control flow driven by static configuration or array primitives,
- make reference implementations differentiable too, so their equality to fused
  versions can be tested under autodiff.

The Phase 1 primitive tests explicitly verify `jax.grad(...)` on:

- `limit_free`,
- `apply_neumann_boundary_average_z`,
- `set_boundary_to_midpoint`.

## Performance Rules

Literal translation is not a license to ship slow code. The intended workflow is:

1. land a readable reference implementation,
2. validate it against Hermes-backed fixtures,
3. land a fused implementation with the same semantics,
4. prove fused equals reference on the same fixtures,
5. only then wire it into the active strict parity engine.

This prevents a repeat of the previous situation where “close” operator code was
difficult to reason about and difficult to validate at the function level.

## Current Status

Phase 1 started on 2026-03-06 with the first mirrored boundary primitives:

- `limit_free`
- `mc_limiter`
- `apply_neumann_boundary_average_z`
- `set_boundary_to_midpoint`

The first dump-backed primitive fixture is:

- `/Users/rogerio/local/jax_drb/tests/fixtures/hermes_mirror_ne_local_rank0_t1.npz`

It is built from:

- `/Users/rogerio/local/jax_drb/runs/hermes_open_field_terms_t01_vortterms/data/BOUT.dmp.0.nc`
- variable `Ne`
- time index `1` (`t = 0.01`)

and records the local guard-domain field in JAX `(nz, nx, ny)` layout together
with the Hermes-source expected `neumann_boundary_average_z` guard values.

The runtime `engine = "hermes_mirror"` is intentionally not wired yet. That
will start only after the primitive and transform phases have executable tests
and dump-backed fixtures.

## References

- Dudson et al., Hermes-3 code and documentation:
  [Hermes repository](https://github.com/boutproject/hermes-3)
- BOUT++ field and boundary implementation:
  [BOUT++ repository](https://github.com/boutproject/BOUT-dev)
- Hermes solver and model overview:
  [Hermes solver numerics](https://hermes3.readthedocs.io/en/stable/solver_numerics.html)
