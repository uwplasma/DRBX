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

## Phase 3 Status

The local dump-backed ExB mirror is now at full-cell Hermes parity for the
`Ne` and `Pe` ExB diagnostic terms on the reference open-field fixture. The
last structural gap was not in the X-flux algebra itself, but in the
preparation chain for `DDY(f)` used by that branch.

Hermes does:

\[
DDY(f) \rightarrow \text{communicate} \rightarrow \text{applyBoundary("neumann")}
\]

before the X-flux face loop. The earlier mirror helper only applied the radial
Neumann boundary. The lower-open parallel guard planes were therefore left with
raw centred-difference values instead of the copied Neumann state Hermes uses
at the sheath boundary.

The landed fix extends `prepare_poloidal_x_dfdy_local_ref` so that, when the
parallel direction is not periodic, it also applies the centred-field Neumann
boundary on the local parallel axis with side-aware control:

- lower open boundary: enabled
- upper open boundary: enabled only when the local fixture has that boundary

This restores the correct lower guard recursion:

\[
(\partial_y f)_{g_1} = (\partial_y f)_{i_1},
\qquad
(\partial_y f)_{g_2} = (\partial_y f)_{i_2}
\]

for zero prescribed gradient, in exactly the same way as the centred BOUT
`BoundaryNeumann::apply(Field3D&)` branch.

On the dump-backed full local ExB term fixture:

- `Ne` all-cell diff RMS: `3.072901445531812e-05`
- `Pe` all-cell diff RMS: `1.3376334360587529e-05`
- `Ne` all-cell correlation: `0.9999820919602114`
- `Pe` all-cell correlation: `0.9999963535995172`

That means the remaining ExB work is now runtime wiring and species-state
ordering, not further local operator reconstruction.

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
- dump-backed deterministic RMS values for both `Ne` and `Pe`,
- direct Hermes-term comparison on the physical interior cells.

On the shared local ExB fixture, the current assembled reference values are:

- `Ne` total RMS: `7.204357588601792e-03`
- `Ne` interior RMS: `1.4914956178042702e-03`
- `Pe` total RMS: `7.071453164542667e-03`
- `Pe` interior RMS: `1.3741825962166174e-03`

A second dump-backed fixture,
`/Users/rogerio/local/jax_drb/tests/fixtures/hermes_mirror_exb_term_local_rank0_t1.npz`,
now also stores the raw Hermes `term_Ne_exb` and `term_Pe_exb` arrays from the
same local rank and time slice. On that fixture, the assembled mirror operator
matches Hermes on the physical interior cells to tight tolerance:

- `Ne` interior diff RMS: `2.8867991448834276e-05`
- `Pe` interior diff RMS: `1.2432835191026055e-05`
- `Ne` interior correlation: `0.9998132247422601`
- `Pe` interior correlation: `0.9999591421467119`

The remaining mismatch is not in the interior operator algebra. It is
concentrated in the lower open-boundary guard cells, especially the lower-x
guard, lower-y guard, and lower-left corner. For the `Ne` local term
comparison, the current diff RMS is:

- interior: `2.8867991448834276e-05`
- `xlow_guard`: `3.03297619604618e-02`
- `ylow_guard`: `1.7801910843170923e-02`
- lower-left corner: `1.0506537581007377e-01`

That means Phase 3 is now structurally closed on the physical interior, and the
remaining work is the guard/boundary diagnostic semantics around the mirrored
X-flux open-boundary path.

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

## Runtime Promotion Status

An opt-in runtime wrapper is now landed in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/exb.py` as
`div_n_bxgrad_f_b_xppm`, and the active field-aligned geometry adapter can call
it with `exb_flux_scheme = "hermes_mirror"`.

This wrapper reconstructs the guard-inclusive local Hermes/BOUT storage
contract from the global JAX `(nz, nx, ny)` arrays, runs the validated local
mirror ExB operator, and slices the physical interior cells back out for the
live geometry path.

On the dump-backed local-rank fixture, feeding only the interior cells into the
runtime wrapper gives:

- `Ne` diff RMS: `2.488462499110523e-04`
- `Pe` diff RMS: `2.6183313968993464e-04`
- `Ne` correlation: `0.9872236467215821`
- `Pe` correlation: `0.9839313048079569`

That keeps the wrapper viable as the promotion vehicle, but it is not yet
strict-parity quality.

The next runtime slice adds a stitched global Hermes fixture and a hybrid
open-boundary wrapper. The new fixture builder is
`/Users/rogerio/local/jax_drb/tools/build_hermes_mirror_runtime_fixture.py`,
the checked-in artifact is
`/Users/rogerio/local/jax_drb/tests/fixtures/hermes_mirror_exb_global_t1.npz`,
and the regression is
`/Users/rogerio/local/jax_drb/tests/hermes_mirror/test_exb_runtime_global.py`.

That regression localizes the runtime residual to the first and last open
parallel blocks. The new numerics knob
`hermes_mirror_parallel_edge_block` therefore re-evaluates only those end
blocks with the local guard-inclusive mirror operator while leaving the middle
of the domain on the whole-domain runtime path.

On the stitched global Hermes fixture, setting
`hermes_mirror_parallel_edge_block = 8` improves the direct runtime mirror
arrays to:

- `Ne` diff RMS: `2.7785371223075885e-04`
- `Pe` diff RMS: `2.9023628701603716e-04`
- `Ne` correlation: `0.99676569423027`
- `Pe` correlation: `0.9964150807456237`

For reference, the same fixture with the whole-domain runtime wrapper gives:

- `Ne` diff RMS: `9.281612304656274e-04`
- `Pe` diff RMS: `9.436398753984853e-04`

The first live 3-step Hermes-state audit for that runtime path is recorded in
`/Users/rogerio/local/jax_drb/runs/audit_hermes_mirror_runtime_3step_v2`. Even
after correcting the shifted-transform FFT length to use `metric_dz * nbinorm`
instead of `grid.perp.dy * nbinorm`, the fail-fast leaders are still worse than
the current strict baseline:

- `omega advection/exb`: `0.06804918916596805`
- `n advection/exb`: `0.04636472581495929`
- `Pe advection/exb`: `0.038900114007649214`

The already-closed parallel channels remain unchanged:

- `n parallel/par`: `0.0029585637833904267`
- `Pe parallel/par_total`: `0.0025796150980648175`
- `omega parallel/jpar`: `0.001995419920917737`

With the edge-block wrapper, the smallest strict gate is now
`/Users/rogerio/local/jax_drb/runs/audit_hermes_mirror_edge_block_1step`.
Its scalar fail-fast metric improves only slightly:

- `omega advection/exb`: `0.06804918916596805 -> 0.06712108791244092`
- `Pe advection/exb`: `0.038900114007649214 -> 0.03873682407548267`

while the `n` scalar row still looks worse in `term_mismatch.csv`. That does
not reflect the direct operator arrays: on the same built-system Hermes state,
`term_map["advection"].n` and the reconstructed pressure advection term now
match the direct mirror operator exactly, with array RMS against Hermes of:

- `n`: `2.7785371223075885e-04`
- `Pe`: `2.9023628701559654e-04`

The audit tooling now also writes array-difference metrics for each matched
term:

- `array_diff_rms`
- `array_rel_diff`
- `array_corr`
- `weighted_array_rel`

and `first_failing_terms.csv` now ranks by the array metric by default
(`--term-ranking-metric=array`). That change matters because the promoted
runtime mirror path really is much better on the dominant density/pressure ExB
arrays than the old scalar RMS-only ranking suggested.

With the updated audit in place, the strict early parity config
`/Users/rogerio/local/jax_drb/examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml`
is now switched to:

- `exb_flux_scheme = "hermes_mirror"`
- `hermes_mirror_parallel_edge_block = 8`

The mirror runtime path also now supports the same non-unit poloidal scaling
contract as the older geometry path:

- `exb_poloidal_scale`
- `exb_poloidal_x_scale`
- `exb_poloidal_y_scale`

which is required by the strict Hermes baseline (`exb_poloidal_y_scale = 1.24`).

The promoted 1-step Hermes-state audit lives in
`/Users/rogerio/local/jax_drb/runs/audit_strict_early_mirror_promoted_1step`.
Against the previous strict baseline, its dominant array-weighted ExB channels
improve as follows:

- `n advection/exb`: `0.6415487257460786 -> 0.30603226941513645`
- `Pe advection/exb`: `0.43066567430657776 -> 0.20417452847516265`
- `n` correlation: `0.7888450540689848 -> 0.9947894182550701`
- `Pe` correlation: `0.7699527249045816 -> 0.9952771323120512`

The parallel channels remain unchanged, and the remaining follow-up is now
narrower:

- `omega parallel/jpar`: still `0.2107103945115671` in weighted-array metric
- `omega advection/exb`: worsens from `0.007979974955211428` to
  `0.09741634145346564`

So the remaining blocker is no longer the local mirrored ExB algebra or the
dominant density/pressure ExB transport. It is now the vorticity ExB
composition on top of the promoted runtime mirror path.

The newest literal-refactor slice adds the missing vorticity-side primitives:

- `src/jaxdrb/hermes_mirror/boundary.py::apply_free_o2_field3d`
- `src/jaxdrb/hermes_mirror/fv.py::div_a_grad_perp_local`
- `src/jaxdrb/hermes_mirror/fv.py::div_a_grad_perp`
- `src/jaxdrb/hermes_mirror/vorticity.py::full_omega_exb_advection`

It also adds the stitched dump-backed fixture
`/Users/rogerio/local/jax_drb/tests/fixtures/hermes_mirror_vorticity_global_t1.npz`
via
`/Users/rogerio/local/jax_drb/tools/build_hermes_mirror_vorticity_fixture.py`,
plus operator regressions in:

- `/Users/rogerio/local/jax_drb/tests/hermes_mirror/test_primitives.py`
- `/Users/rogerio/local/jax_drb/tests/hermes_mirror/test_fv.py`
- `/Users/rogerio/local/jax_drb/tests/hermes_mirror/test_vorticity.py`

That literal vorticity slice is now promoted through a dedicated
`Delp2(phi)` implementation in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/delp2.py`. The geometry
adapter in
`/Users/rogerio/local/jax_drb/src/jaxdrb/core/geometry_field_aligned.py` now
ingests optional Hermes `G1`, `G3`, and `d1_dx` coefficient planes, the
converter
`/Users/rogerio/local/jax_drb/tools/convert_hermes_dump_axisymmetric.py` now
emits `G1`/`G3`, and the strict mesh bundle
`/Users/rogerio/local/jax_drb/examples/open_field_line/axisym_tokamak_bxcv_hermes_norm_parcurv_g22.npz`
now carries the stitched Hermes `G1`/`G3` arrays used by the strict parity
config.

The stitched vorticity fixture now also includes raw Hermes metric planes
(`G1`, `G3`, `g11`, `g13`, `g33`, `dx`, `dz`, `Bxy`, `zShift`) so the mirror
Laplacian can be checked directly against the saved BOUT guard-cell state. That
check now passes at operator level:

- local rank-0 `Delp2(phi)` vs raw BOUT single-index evaluation:
  correlation `0.9999999979364631`, diff RMS `6.903925415803028e-07`
- stitched global `Delp2(phi)` vs rank-stitched raw BOUT evaluation:
  correlation `0.9999988050053542`, diff RMS `3.9164034002630735e-05`

The first promotion attempt still failed because the runtime ExB transport was
carrying the `poisson_invert_set` auxiliary Dirichlet override into the
transport of `phi` / `phi + Pi_hat`. That is not what Hermes does: the
Dirichlet override belongs only in the `Delp2(phi)` construction. Once the
transport-side override was removed and the omega path was routed through the
same validated global mirror wrapper used by the density/pressure ExB channels,
the dump-backed full omega term moved to:

- full `term_Vort_exb` mirror correlation: `0.9286922397070627`
- full diff RMS: `9.242617198253543e-06`

The promoted strict audit
`/Users/rogerio/local/jax_drb/runs/audit_mirror_omega_transport_bc_fix_1step`
then reduced the live blocker:

- `omega advection/exb` weighted-array metric:
  `0.09741634145346564 -> 0.0035704721275969927`
- `omega advection/exb` correlation:
  `-0.6627029835778587 -> 0.9286922397070773`

With the omega-side blocker structurally closed, the remaining strict leaders
are back in the density/pressure ExB and parallel channels.

## References

- Dudson et al., Hermes-3 code and documentation:
  [Hermes repository](https://github.com/boutproject/hermes-3)
- BOUT++ field and boundary implementation:
  [BOUT++ repository](https://github.com/boutproject/BOUT-dev)
- Hermes solver and model overview:
  [Hermes solver numerics](https://hermes3.readthedocs.io/en/stable/solver_numerics.html)
