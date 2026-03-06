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

### Local Field-Aligned Fixture Layout

For shifted-transform and Y-flux preparation work there is now a second,
explicitly named local fixture layout:

\[
(x, y, z)_\text{Hermes local} \longrightarrow (n_\parallel, x, n_\mathrm{binorm})_\text{mirror prep}
\]

Numerically this is:

\[
(x, y, z)_\text{Hermes local} \longrightarrow (y, x, z)_\text{mirror prep}
\]

This is used only for dump-backed local-rank preparation helpers that need to
follow the Hermes `toFieldAligned(...)` path literally. The active runtime
solver is not switched to this representation; it exists to avoid mixing
boundary-primitives and shifted-transform validation in one ambiguous fixture
format.

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

### 4. `apply_neumann_field3d`

Source:
- `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/boundary_standard.cxx`
- function `BoundaryNeumann::apply(Field3D&, BoutReal)` in the non-staggered
  `CELL_CENTRE` branch

For a centred field and a Neumann boundary condition with prescribed gradient
\(g\), Hermes applies:

\[
f_g = f_i + \Delta g
\]

where \(\Delta\) is the signed metric spacing from the interior cell centre to
the guard cell centre. For the first lower guard cell, \(\Delta = -dx\); for
the first upper guard cell, \(\Delta = +dx\).

When two guard cells are present, Hermes then uses:

\[
f_{g,2} = f_{i,2} + 3 \Delta g
\]

where \(f_{i,2}\) is the second interior point along the same boundary-normal
line.

The mirror implementation lands this centred-field branch first and takes the
boundary axis explicitly. That keeps the source formula literal while avoiding a
hard-coded guess about coordinate names after reordering into the active JAX
layout.

## Phase 2 Transform Slice

Source files:

- `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/parallel/shiftedmetricinterp.cxx`
- `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/coordinates.cxx`

The mirror path now includes a first transform implementation for the
shifted-metric interpolation used by Hermes/BOUT field-aligned operators.

### What is implemented

- precomputed linear interpolation weights in
  `ShiftedFieldAlignedWeights`
- `to_field_aligned_nox_ref`
- `to_field_aligned_nox`
- `from_field_aligned_nobndry_ref`
- `from_field_aligned_nobndry`

These target the two region variants used most heavily in the Hermes parity
path:

- `RGN_NOX`: shift interior x columns and preserve radial boundary columns
- `RGN_NOBNDRY`: shift only non-boundary cells and preserve both radial and
  open-field parallel boundary planes

### Interpolation formula

For a field \(f(s, x, \zeta)\), where:

- \(s\) is the field-aligned/parallel index,
- \(x\) is the radial index,
- \(\zeta\) is the shifted/binormal interpolation index,

and a shift measured in index space \(\Delta(s, x)\), the mirror transform uses
the same linear interpolation structure as the current JAX shifted-transform
path:

\[
\zeta_{\mathrm{src}} = (\zeta + \Delta) \bmod N_\zeta
\]

\[
\zeta_0 = \lfloor \zeta_{\mathrm{src}} \rfloor,
\qquad
\zeta_1 = (\zeta_0 + 1) \bmod N_\zeta
\]

\[
\alpha = \zeta_{\mathrm{src}} - \zeta_0
\]

\[
f_{\mathrm{shift}} = (1-\alpha) f(\zeta_0) + \alpha f(\zeta_1)
\]

The forward transform uses \(+\Delta\). The inverse transform uses \(-\Delta\).

### Why precompute weights

Hermes/BOUT caches interpolation weights inside `ShiftedMetricInterp`. The JAX
mirror path does the same for three reasons:

- the transform becomes a pure gather-and-blend operation under JIT,
- reference and fused implementations can share the same semantics,
- later ExB and parallel operators can reuse the same cached weights without
  re-deriving interpolation indices on every RHS evaluation.

### Validation status

The current transform slice is validated against the existing JAX geometry
adapter in the overlap region where both use the same linear shifted transform:

- `to_field_aligned_nox` matches `FieldAlignedGeometryAdapter.to_field_aligned_nox`
- `from_field_aligned_nobndry` matches the current inverse transform on the
  interior region and additionally enforces `RGN_NOBNDRY` boundary preservation

This is not yet a Hermes dump-backed transform fixture. That requires a
dedicated extraction of Hermes-aligned/interpolated fields and remains part of
the next Phase 2 work.

The source-true Hermes path for the current open-field benchmark is
`ShiftedMetric`, not `ShiftedMetricInterp`, because the Hermes input uses:

```ini
[mesh:paralleltransform]
type = shifted
```

The mirror path therefore now also carries an FFT-based phase-cache
implementation matching `ShiftedMetric::shiftZ(...)`. The linear path remains in
tree only as an overlap check against the pre-existing JAX shifted-transform
implementation.

The first stitched global transform fixture is:

- `/Users/rogerio/local/jax_drb/tests/fixtures/hermes_mirror_shiftedmetric_global_t1.npz`

It is produced by:

- `/Users/rogerio/local/jax_drb/tools/build_hermes_mirror_transform_fixture.py`

from:

- `/Users/rogerio/local/jax_drb/runs/hermes_open_field_terms_t01_vortterms/data`
- field `Ne`
- time index `1` (`t = 0.01`)

and contains:

- the global interior field in local mirror transform layout `(y, x, z)`
- the stitched global `zShift(y, x)` field
- the toroidal domain length used for the FFT phase cache

The remaining open question in Phase 1 is not the centred Neumann formula
itself, which is now landed, but the exact axis/region naming to use when this
helper is wired into a full mirror geometry/runtime path.

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

Phase 3 started on 2026-03-06 with the first mirrored ExB transport slice:

- `div_n_bxgrad_f_b_xppm_xz`
- `div_n_bxgrad_f_b_xppm_xz_ref`

This slice mirrors only the X-Z branch of Hermes
`Div_n_bxGrad_f_B_XPPM`, which is the part of the operator that already maps
cleanly onto the existing JAX `(nz, nx, ny)` storage and current
`exb_flux_divergence(..., hermes_xppm)` implementation when poloidal ExB flows
are disabled.

The current validation for this slice is intentionally layered:

- synthetic fused-versus-reference equality for MC and Fromm reconstruction,
- equality with the existing unified X-Z `hermes_xppm` path when
  `exb_poloidal_flows = false`,
- `jax.grad(...)` coverage to keep the mirror operator differentiable before the
  full ExB term is assembled.

The full mirror ExB operator is still incomplete because the Y-flux branch,
field-aligned transform wiring, and dump-backed parity fixtures are not landed
yet.

The transform layer now also carries the missing `RGN_ALL` variants:

- `to_field_aligned_all`
- `from_field_aligned_all`
- `to_field_aligned_all_fft`
- `from_field_aligned_all_fft`

These are needed because the Hermes Y-flux branch in
`Div_n_bxGrad_f_B_XPPM` uses plain `toFieldAligned(...)` and
`fromFieldAligned(...)`, not the `RGN_NOX`/`RGN_NOBNDRY` restricted forms used
in other interpolation helpers.

The next mirrored preparation slice is now also landed:

- `ddx_centered_guarded`

This helper is intentionally scoped to local guard-inclusive arrays read
straight from `BOUT.dmp.*.nc`. It mirrors the centred `DDX` stage that Hermes
applies before the Y-flux transform, including the fact that boundary interior
cells still use centred stencils because x-guard values are present in the
local dump.

That distinction matters. On the Hermes dump used for strict parity work, the
guard-aware left-boundary `DDX(phi)` differs from the current guardless JAX
boundary derivative by an RMS amount larger than the Hermes boundary derivative
itself. In other words, the active runtime mismatch is not in the transform
region bookkeeping alone; it is upstream in the state-preparation chain.

## Phase 4 Local Prep Slice

The first field-aligned local preparation helper is now landed in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/species.py`:

- `prepare_poloidal_y_dfdx_local_ref`
- `prepare_poloidal_y_dfdx_local`

This mirrors the specific Hermes chain used by the Y-flux branch of
`Div_n_bxGrad_f_B_XPPM`:

1. `DDX(phi)`
2. `mesh->communicate(dfdx)`
3. `dfdx.applyBoundary("neumann")`
4. `toFieldAligned(dfdx)`

The current local fixture for that path is:

- `/Users/rogerio/local/jax_drb/tests/fixtures/hermes_mirror_phi_field_aligned_local_rank0_t1.npz`

It is extracted from:

- `/Users/rogerio/local/jax_drb/runs/hermes_open_field_terms_t01_vortterms/data/BOUT.dmp.0.nc`

with local field-aligned layout `(npar, nx, nbinorm) = (y, x, z)`.

The corresponding tests in
`/Users/rogerio/local/jax_drb/tests/hermes_mirror/test_species.py`
cover:

- zero-shift synthetic behavior,
- autodiff,
- dump-backed deterministic RMS values,
- and a regression that shows the literal local prep path differs materially
  from the current guardless approximation.

## Phase 3 Local Y-Flux Slice

The next mirrored ExB slice now exists in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/exb.py`:

- `div_n_bxgrad_f_b_xppm_xy_y_local_ref`
- `div_n_bxgrad_f_b_xppm_xy_y_local`
- `div_n_bxgrad_f_b_xppm_xy_y_local_from_fields_ref`
- `div_n_bxgrad_f_b_xppm_xy_y_local_from_fields`

This is still a local field-aligned mirror, not the final runtime-facing
assembled operator. It mirrors the Hermes Y-flux branch of
`Div_n_bxGrad_f_B_XPPM` after the guard-aware local preparation chain has
already been applied:

1. prepare `DDX(phi)` locally with guard cells and Neumann x-boundaries,
2. shift both `dfdx` and the advected field into field-aligned coordinates,
3. compute the local Y-face velocity from
   `J * (g11 * g23 / B^2) * dfdx`,
4. apply the open-field sheath-sign restrictions at the lower and upper
   parallel boundaries,
5. apply the Fromm upwind state on the local field-aligned arrays,
6. accumulate the flux divergence into the two adjacent field-aligned cells.

The dump-backed fixture for this slice is:

- `/Users/rogerio/local/jax_drb/tests/fixtures/hermes_mirror_exb_local_rank0_t1.npz`

It is extracted from:

- `/Users/rogerio/local/jax_drb/runs/hermes_open_field_terms_t01_vortterms/data/BOUT.dmp.0.nc`

with the combined local fields:

- `phi`
- `Ne`
- `Pe`
- `dx`
- `dy`
- `J`
- `g11`
- `g23`
- `Bxy`
- `zShift`
- `dz`

The corresponding tests in
`/Users/rogerio/local/jax_drb/tests/hermes_mirror/test_exb_y_local.py`
cover:

- zero-metric sanity (`g23 = 0` gives zero Y flux),
- fused-versus-reference equality on synthetic fields,
- autodiff through the full local-from-fields path,
- dump-backed deterministic RMS values for both `Ne` and `Pe`.

On that local dump-backed fixture, the current reference values are:

- `Ne` total RMS: `5.266245270548453e-03`
- `Ne` interior RMS: `2.3208656645780424e-03`
- `Pe` total RMS: `5.06778021563735e-03`
- `Pe` interior RMS: `2.1455021379486773e-03`

This still does not change the strict Hermes audit by itself, because the
runtime path is still using the old assembled ExB implementation. The next
Phase 3 step is to combine the already-landed X-Z slice and this local Y-flux
slice into a full mirror `Div_n_bxGrad_f_B_XPPM` entrypoint before touching
the active strict engine.

## Phase 3 Local X-Flux Slice

The missing poloidal X-flux slice is now also landed in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/exb.py`:

- `div_n_bxgrad_f_b_xppm_xy_x_local_ref`
- `div_n_bxgrad_f_b_xppm_xy_x_local`
- `div_n_bxgrad_f_b_xppm_xy_x_local_from_fields_ref`
- `div_n_bxgrad_f_b_xppm_xy_x_local_from_fields`

This mirrors the Hermes X-flux branch of `Div_n_bxGrad_f_B_XPPM`, which stays
in the unshifted local field-aligned layout and depends on the separate
preparation chain:

1. `DDY(phi)`
2. `mesh->communicate(dfdy)`
3. `dfdy.applyBoundary("neumann")`
4. average `(g11 * g23 / B^2) * dfdy` to the right x-face
5. apply the Fromm upwind state in the x direction
6. accumulate the x-face flux divergence into the neighboring cells

The preparation helper for this slice now also exists in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/species.py`:

- `prepare_poloidal_x_dfdy_local_ref`
- `prepare_poloidal_x_dfdy_local`

The corresponding tests in
`/Users/rogerio/local/jax_drb/tests/hermes_mirror/test_exb_x_local.py`
cover fused-versus-reference equality, autodiff, and dump-backed deterministic
RMS values on the shared local ExB fixture:

- `Ne` total RMS: `5.391187274308899e-03`
- `Pe` total RMS: `5.289137581776043e-03`

## Phase 3 Local Full Operator

The first assembled local full mirror now also exists in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/exb.py`:

- `div_n_bxgrad_f_b_xppm_local_ref`
- `div_n_bxgrad_f_b_xppm_local`

These functions assemble the current Phase 3 slices in the same order as the
Hermes source:

1. X-Z branch via `div_n_bxgrad_f_b_xppm_xz`
2. local poloidal X-flux via `div_n_bxgrad_f_b_xppm_xy_x_local_from_fields`
3. local field-aligned Y-flux via `div_n_bxgrad_f_b_xppm_xy_y_local_from_fields`
4. `fromFieldAligned(...)` applied to the Y-flux contribution
5. summed full local ExB divergence

The corresponding tests in
`/Users/rogerio/local/jax_drb/tests/hermes_mirror/test_exb_local_full.py`
cover:

- equality with the X-Z slice when `poloidal = false`,
- fused-versus-reference equality for the assembled local operator,
- autodiff through the full assembled local path,
- dump-backed deterministic RMS values for both `Ne` and `Pe`.

On the shared local ExB fixture, the current assembled reference values are:

- `Ne` total RMS: `7.204357588601792e-03`
- `Ne` interior RMS: `1.4914956178042702e-03`
- `Pe` total RMS: `7.071453164542667e-03`
- `Pe` interior RMS: `1.3741825962166174e-03`

This is still a mirror-only local operator. It is the first point where the
full Hermes ExB structure exists in JAX as one testable function, but it is
not yet wired into the strict runtime engine.

## Phase 4 Transform Helpers

The first runtime-facing species state-preparation helpers are now landed in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/species.py`:

- `density_transform_impl`
- `pressure_transform_impl`

These mirror the Stage 1 parts of Hermes
`/Users/rogerio/local/hermes-3/src/evolve_density.cxx` and
`/Users/rogerio/local/hermes-3/src/evolve_pressure.cxx` that prepare the
species fields before the ExB and parallel terms are assembled:

1. optional `exp(...)` for log-evolved variables,
2. `neumann_boundary_average_z` x-guard reconstruction using the binormal
   average at the first/last interior x cell,
3. density floor at zero for the stored species density,
4. pressure floor at zero,
5. temperature reconstruction from `Pfloor / softFloor(N, density_floor)`,
6. pressure consistency reset `P = N * T`.

The corresponding tests in
`/Users/rogerio/local/jax_drb/tests/hermes_mirror/test_transform_impl.py`
cover:

- restoring dump-backed x-guard states after clobbering the guard cells,
- nonnegative density flooring,
- pressure/temperature reconstruction against a dump-backed fixture,
- autodiff through the pressure transform path.

On the shared local fixture, the current deterministic values are:

- density RMS: `1.7785461475277795`
- density interior RMS: `1.8245458655422153`
- temperature RMS: `5.928697471001826e-01`
- temperature interior RMS: `6.081834468193776e-01`

This is the first Phase 4 bridge between the mirror operator stack and the
prepared species states Hermes actually uses. The next remaining work is to
mirror the `finally` ordering and connect these prepared states to the strict
runtime path.

## References

- Dudson et al., Hermes-3 code and documentation:
  [Hermes repository](https://github.com/boutproject/hermes-3)
- BOUT++ field and boundary implementation:
  [BOUT++ repository](https://github.com/boutproject/BOUT-dev)
- Hermes solver and model overview:
  [Hermes solver numerics](https://hermes3.readthedocs.io/en/stable/solver_numerics.html)
