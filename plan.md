# jax_drb Agent Handoff + Master Plan

Last update: 2026-03-06
Owner: `jax_drb` core team
Primary workspace: `/Users/rogerio/local/jax_drb`

---

## 0) Copy/Paste Prompt for a New Agent

Use the block below as the takeover prompt.

```text
You are taking over development of jax_drb in /Users/rogerio/local/jax_drb.

Project mission:
- Build a research-grade drift-reduced Braginskii (DRB) solver in JAX that is:
  - easy to use (CLI + TOML + Python API),
  - easy to maintain,
  - fully validated and tested,
  - extensively documented,
  - end-to-end differentiable,
  - high-performance and memory-efficient on CPU and GPU.

Immediate strategy reset:
- Stop trying to close Milestone A by patching the existing Hermes-alignment branches in place.
- Build a temporary Hermes-mirror JAX path that is a literal transliteration of the specific Hermes/BOUT baseline operator stack needed for the strict open-field electrostatic baseline.
- Use that mirror path to reach strict early parity first.
- Once it passes the strict gates, promote its logic into the unified core and delete the superseded patched operator code.

Non-negotiables:
- No proxy/testbed equations in production paths.
- Keep production runtime operators pure JAX.
- Keep differentiability in all production solver paths.
- Keep performance and memory usage as first-class acceptance criteria.
- Validation-first: unit + primitive + term + regression + benchmark gates.

Current strategy rules:
1) Implement one Hermes primitive or operator at a time in a new mirror path.
2) Translate from Hermes source directly, with the source file/function named in comments and tests.
3) Add two tests before promotion:
   - a tiny synthetic fixture,
   - a Hermes dump-backed fixture.
4) Only then wire the primitive into the mirror engine.
5) Re-run strict term audit, short-window gate, and full CI.
6) Update docs/benchmarks/open_field_alignment.md and /Users/rogerio/local/jax_drb/plan.md.
7) Commit and push only green changes.

Do not:
- keep tuning the current patched ExB/boundary code path for Milestone A,
- mix new mirror logic into old operator helpers before the mirror primitive is validated,
- promote longer windows while short-window strict parity still fails.

Paths and context:
- jax_drb repo: /Users/rogerio/local/jax_drb
- Hermes-3 repo: /Users/rogerio/local/hermes-3
- Hermes-2 repo: /Users/rogerio/local/hermes-2
- Literature folder: /Users/rogerio/local/tests/drb_literature
- GBS refs: /Users/rogerio/local/tests/GBS_ISTTOK

Deliverables expected from you:
- code changes,
- tests,
- docs updates,
- reproducible benchmark artifacts,
- updated plan.md progress status.
```

---

## 1) North Star

Build `jax_drb` into a production-quality SOL turbulence code that:

1. solves DRB equations in a single maintainable core with toggles,
2. reproduces Hermes-class baseline workflows in JAX,
3. then extends to conservative DRB with strict energy diagnostics,
4. supports multiple geometry paradigms,
5. remains differentiable, fast, memory-efficient, and reproducible.

---

## 2) Strategy Reset

### Decision

Milestone A work is now a staged Hermes-mirror rewrite, not an in-place patching program.

### Why

- The remaining strict-window mismatch is localized but structurally tangled.
- Repeated local edits in the current operator stack are no longer converging reliably.
- The fastest path to parity is to mirror the Hermes source stack directly, function by function.

### What changes

- New Stage 1 baseline work goes into a new mirror path under:
  - `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror`
- The mirror path is allowed to duplicate limited Stage 1 logic temporarily.
- The current patched Hermes-specific logic in:
  - `/Users/rogerio/local/jax_drb/src/jaxdrb/core/geometry_field_aligned.py`
  - `/Users/rogerio/local/jax_drb/src/jaxdrb/core/terms/advection.py`
  - `/Users/rogerio/local/jax_drb/src/jaxdrb/core/terms/parallel.py`
  is frozen for Milestone A parity work and should not receive more speculative parity patches.
- After the mirror path passes Milestone A and Milestone B, fold its logic back into the unified core and delete the superseded patched code.

### Temporary architecture allowance

The mirror path may be introduced as a temporary engine mode, for example:

- `engine = "hermes_mirror"`

This is allowed only as a parity-integration scaffold. It is not the final architecture target.

---

## 3) Keep vs Replace

### Keep

- CLI, TOML loader, Python API
- audit and benchmark tooling
- geometry conversion scripts
- current normalization infrastructure
- current Poisson infrastructure until the mirror path needs to replace it
- CI workflow and local validation commands
- legacy paths for traceability

### Replace for Stage 1 parity

- patched Hermes-specific ExB path in the existing field-aligned geometry adapter
- patched open-field boundary helper logic built incrementally in the unified core
- continued term-by-term gap-closing inside the old operator path

### Delete later

Once the mirror path is promoted:

- remove superseded Hermes-specific branches from the old operator path,
- delete dead configs/tests that only supported the abandoned patch-by-patch strategy,
- reduce `drb_fv` and `legacy_v1` to traceability roles only.

---

## 4) Operational Map

### Core repos

- `jax_drb`: `/Users/rogerio/local/jax_drb`
- `Hermes-3`: `/Users/rogerio/local/hermes-3`
- `Hermes-2`: `/Users/rogerio/local/hermes-2`

### Literature and references

- DRB literature: `/Users/rogerio/local/tests/drb_literature`
- GBS references: `/Users/rogerio/local/tests/GBS_ISTTOK`

### jax_drb key directories

- Source: `/Users/rogerio/local/jax_drb/src/jaxdrb`
- New mirror path: `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror`
- Unified core: `/Users/rogerio/local/jax_drb/src/jaxdrb/core`
- Legacy path: `/Users/rogerio/local/jax_drb/src/jaxdrb/legacy_v1`
- Docs: `/Users/rogerio/local/jax_drb/docs`
- Tests: `/Users/rogerio/local/jax_drb/tests`
- Tools: `/Users/rogerio/local/jax_drb/tools`
- Runs: `/Users/rogerio/local/jax_drb/runs`

---

## 5) Hermes Source-of-Truth Map

The functions below define the mirror scope. New JAX functions must cite the Hermes source they mirror.

| Hermes source | Hermes function / logic | New JAX target | Purpose |
| --- | --- | --- | --- |
| `/Users/rogerio/local/hermes-3/src/sheath_boundary_simple.cxx` | `limitFree` | `hermes_mirror/primitives.py::limit_free` | sheath guard construction |
| `/Users/rogerio/local/hermes-3/src/evolve_density.cxx` | `transform_impl` | `hermes_mirror/species.py::density_transform_impl` | x-boundary average-z ghost setup |
| `/Users/rogerio/local/hermes-3/src/evolve_pressure.cxx` | `transform_impl` | `hermes_mirror/species.py::pressure_transform_impl` | pressure x-boundary average-z ghost setup |
| `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/field/field3d.cxx` | `Field3D::setBoundaryTo` | `hermes_mirror/boundary.py::set_boundary_to_midpoint` | boundary cell overwrite semantics |
| `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/boundary_standard.cxx` | `BoundaryNeumann::apply(Field3D&)` | `hermes_mirror/boundary.py::apply_neumann_field3d` | cell-centred/staggered Neumann boundary application |
| `/Users/rogerio/local/hermes-3/src/div_ops.cxx` | `Stencil1D`, `MC` | `hermes_mirror/primitives.py::stencil1d`, `mc_limiter` | limited face reconstruction |
| `/Users/rogerio/local/hermes-3/src/div_ops.cxx` | `Div_n_bxGrad_f_B_XPPM` | `hermes_mirror/exb.py::div_n_bxgrad_f_b_xppm` | Hermes ExB conservative advection |
| `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/coordinates.cxx` | `toFieldAligned(..., "RGN_NOX")`, `fromFieldAligned(..., "RGN_NOBNDRY")` | `hermes_mirror/transform.py::to_field_aligned_nox`, `from_field_aligned_nobndry` | shifted-field transport semantics |
| `/Users/rogerio/local/hermes-3/src/evolve_density.cxx` | `finally` density ExB / parallel ordering | `hermes_mirror/rhs.py::density_rhs_terms` | density RHS order |
| `/Users/rogerio/local/hermes-3/src/evolve_pressure.cxx` | `finally` pressure ExB / parallel ordering | `hermes_mirror/rhs.py::pressure_rhs_terms` | pressure RHS order |
| `/Users/rogerio/local/hermes-3/src/div_ops.cxx` and `/Users/rogerio/local/hermes-3/external/BOUT-dev/include/bout/fv_ops.hxx` | `FV::Div_par_mod<hermes::Limiter>` and limiter behavior | `hermes_mirror/parallel.py::div_par_mod` | parallel transport parity |

### Explicit non-goals for the mirror rewrite

Do not copy the full BOUT++ object model, component permissions system, or MPI layer.

Instead:

- preserve only the numerical semantics,
- flatten all required state into immutable JAX-friendly data structures,
- emulate the final single-device numerical result.

---

## 6) New JAX Mirror Layout

Create these files first:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/__init__.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/types.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/primitives.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/boundary.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/transform.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/exb.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/parallel.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/species.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/rhs.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_mirror/engine.py`

### Intended roles

- `types.py`
  - immutable `eqx.Module` containers for geometry arrays, masks, transform weights, and runtime options
- `primitives.py`
  - exact scalar/face primitives: `limit_free`, `mc_limiter`, face reconstruction helpers
- `boundary.py`
  - x and y boundary updates: average-z ghosts, Neumann apply, midpoint overwrite
- `transform.py`
  - field-aligned transforms driven by precomputed weights and region masks
- `exb.py`
  - literal Hermes ExB operator path
- `parallel.py`
  - literal Hermes parallel FV operator path
- `species.py`
  - density and pressure state-preparation order
- `rhs.py`
  - density and pressure RHS builders using the mirror operators
- `engine.py`
  - strict parity engine wrapper

---

## 7) Performance and Differentiability Rules

### Runtime rules

- Production mirror operators must be pure JAX:
  - `jax.numpy`
  - `jax.lax`
  - `jax.vmap`
  - `equinox`
- No NumPy in production runtime operators.
- No host callbacks.
- No hidden Python control flow over runtime-dependent shapes.

### Build-time preprocessing rules

Build-time geometry ingestion may use Python or NumPy only for static preprocessing:

- loading coefficients,
- assembling interpolation weights,
- constructing masks and integer index maps.

All runtime-used results must be stored as JAX arrays in the mirror geometry object.

### Reference-first pattern

For every nontrivial operator:

1. implement `*_ref` as a literal transliteration with `lax.fori_loop` or straightforward JAX loops,
2. implement `*_prod` as the fused/vectorized equivalent,
3. test `*_prod == *_ref`,
4. run Hermes dump-backed validation,
5. use `*_prod` in the mirror engine.

This preserves clarity first and performance second without changing the mathematics.

### Differentiability rules

- Use `jnp.where`, `jnp.maximum`, `jnp.minimum`, and JAX algebra for all branchy Hermes logic.
- Piecewise operations are acceptable if they are part of the real Hermes baseline.
- Keep all operator outputs differentiable with respect to the evolving state arrays.
- If later geometry parameters need gradients, replace any build-time-only weight construction that blocks that use case.

---

## 8) Fixture and Test Strategy

### New fixture tooling

Create:

- `/Users/rogerio/local/jax_drb/tools/build_hermes_mirror_fixture.py`

This script should extract compact primitive-level fixtures from Hermes dumps:

- x-boundary ghost/cell slices
- y-boundary sheath guard slices
- `DDX(phi)` and `DDY(phi)` boundary slices
- field-aligned transform inputs/outputs
- operator input/output slices for `Ne`, `Pe`, `phi`, and `Vort`

### Test layers

1. Primitive tests
   - tiny synthetic fixtures
   - no driver or full geometry build required

2. Dump-backed primitive tests
   - compare single functions against Hermes-derived arrays

3. Operator tests
   - compare full `Div_n_bxGrad_f_B_XPPM`
   - compare `Div_par_mod`

4. RHS tests
   - density and pressure term-by-term

5. Engine regression tests
   - strict 1-step
   - strict 3-step
   - short-window `t <= 0.1`

---

## 9) Implementation Program

### Phase 0: Scaffolding

- [x] Add `src/jaxdrb/hermes_mirror`
- [ ] Add temporary engine selection:
  - `engine = "hermes_mirror"`
- [ ] Add a strict mirror config:
  - `examples/open_field_line/input_tokamak_bxcv_alignment_strict_mirror.toml`
- [x] Add fixture builder script
- [x] Add initial test package:
  - `tests/hermes_mirror/`

### Phase 1: Boundary and primitive transliterations

- [x] `limit_free`
- [x] `mc_limiter`
- [x] `set_boundary_to_midpoint`
- [x] `apply_neumann_field3d`
- [x] `apply_neumann_boundary_average_z`

Acceptance:

- [x] primitive unit tests pass
- [ ] dump-backed boundary fixtures match Hermes

### Phase 2: Field-aligned transform transliterations

- [x] precompute transform weights and masks from existing geometry ingestion
- [x] `to_field_aligned_nox_ref`
- [x] `from_field_aligned_nobndry_ref`
- [x] `to_field_aligned_nox`
- [x] `from_field_aligned_nobndry`
- [x] `to_field_aligned_all` / `from_field_aligned_all`
- [x] local guard-aware `DDX` mirror helper exists

Acceptance:

- [x] fused transform equals reference transform
- [ ] dump-backed transform fixtures match Hermes region semantics

### Phase 3: Hermes ExB operator mirror

Create in this order:

- [x] `div_n_bxgrad_f_b_xppm_xz_ref`
- [x] local `div_n_bxgrad_f_b_xppm_xy_x_local_ref`
- [x] local field-aligned `div_n_bxgrad_f_b_xppm_xy_y_local_ref`
- [x] local assembled `div_n_bxgrad_f_b_xppm_local_ref`
- [ ] runtime-facing fused production version

Wire first for:

- [ ] density ExB term
- [ ] pressure ExB term

Then:

- [ ] temperature ExB term
- [ ] vorticity ExB term

Acceptance:

- [x] dump-backed local `Ne` / `Pe` ExB terms match Hermes on interior cells
- [ ] lower-open-boundary guard cells match Hermes diagnostic semantics
- [ ] `Ne exb` strict 1-step leader removed
- [ ] `Pe exb` strict 1-step leader removed
- [ ] 3-step term audit stays improved

### Phase 4: Species state-preparation mirror

- [x] local `DDX -> applyBoundary("neumann") -> toFieldAligned` prep helper exists
- [x] mirror density `transform_impl`
- [ ] mirror density `finally`
- [x] mirror pressure `transform_impl`
- [ ] mirror pressure `finally`
- [ ] mirror sheath guard preparation order used by Stage 1 baseline

Acceptance:

- [x] boundary-state dump fixtures match Hermes at transform-helper level
- [ ] mirror ExB operator uses the same prepared states as Hermes

### Phase 5: Parallel FV mirror

Create in this order:

- [ ] limiter and reconstruction helpers from Hermes FV path
- [ ] `div_par_mod_ref`
- [ ] fused `div_par_mod`
- [ ] density parallel term wiring
- [ ] pressure parallel term wiring
- [ ] later momentum and energy channels as needed

Acceptance:

- [ ] `n parallel/par` below threshold
- [ ] `Pe parallel/par_total` below threshold
- [ ] no regression in already-closed `omega parallel/jpar`

### Phase 6: Mirror engine promotion

- [ ] strict configs switched to `engine = "hermes_mirror"`
- [ ] short-window `t <= 0.1` RMS/PSD gate passes
- [ ] docs updated with mirror architecture and source map
- [ ] old patched Hermes operator path marked for deletion

### Phase 7: Fold-back into unified core

- [ ] identify mirror functions that should become shared production core operators
- [ ] port validated logic into unified core without changing numerics
- [ ] delete superseded patched Hermes-specific code
- [ ] keep mirror tests as regression gates

---

## 10) Immediate Next Steps

Start here, in this exact order:

1. Create `src/jaxdrb/hermes_mirror/` and `tests/hermes_mirror/`.
2. Implement `limit_free`.
3. Implement `apply_neumann_boundary_average_z`.
4. Implement `set_boundary_to_midpoint`.
5. Build compact dump-backed fixtures for those functions.
6. Add mirror geometry container with precomputed transform weights.
7. Implement `to_field_aligned_nox_ref` and `from_field_aligned_nobndry_ref`.
8. Only then start `Div_n_bxGrad_f_B_XPPM`.

Do not start by wiring a full engine before the first three boundary primitives are tested.

---

## 11) Cycle Loop

For every implementation cycle:

1. Pick the next unchecked primitive or operator from Section 9.
2. Read the Hermes source function directly.
3. Translate it into the mirror path with a source citation comment.
4. Add:
   - one tiny synthetic test,
   - one Hermes dump-backed test.
5. If it is an operator:
   - add a `*_ref` form first,
   - then a fused production form,
   - test equality.
6. Wire it into the mirror engine or mirror RHS path.
7. Re-run:
   - strict 1-step audit,
   - strict 3-step audit,
   - short-window gate when relevant,
   - `ruff`,
   - `black --check`,
   - `pytest`.
8. Update:
   - `docs/benchmarks/open_field_alignment.md`
   - this file
9. Commit and push only green changes.

---

## 12) Minimal Commands

### Install and checks

```bash
cd /Users/rogerio/local/jax_drb
python -m pip install -e ".[dev]"
ruff check src tests
black --check src tests
python -m pytest -q
```

### Current strict audit baseline

```bash
python /Users/rogerio/local/jax_drb/tools/audit_term_alignment.py \
  --jax-config /Users/rogerio/local/jax_drb/examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml \
  --hermes-data-dir /Users/rogerio/local/jax_drb/runs/hermes_open_field_terms_t01_vortterms/data \
  --out-dir /Users/rogerio/local/jax_drb/runs/audit_latest \
  --nsteps 3 --match-hermes-dt --strict-axis --use-hermes-state --use-hermes-phi-in-terms --start-index 1
```

### Mirror fixture build

```bash
python /Users/rogerio/local/jax_drb/tools/build_hermes_mirror_fixture.py
```

### Planned mirror strict run

```bash
jaxdrb /Users/rogerio/local/jax_drb/examples/open_field_line/input_tokamak_bxcv_alignment_strict_mirror.toml --run --output /tmp/jax_mirror_short.npz
```

---

## 13) Progress Tracker

### Milestone A0: strategy reset

- [x] Decide to stop patching the old Hermes parity path in place.
- [x] Define the Hermes source-of-truth function map.
- [x] Rewrite this plan around the mirror strategy.

### Milestone A1: mirror primitives

- [x] Mirror boundary primitives exist.
- [x] Primitive dump-backed fixtures exist.
- [ ] Primitive tests are green in CI.

### Milestone A2: mirror transforms

- [x] `to_field_aligned_nox` and `from_field_aligned_nobndry` mirror implementations exist.
- [x] Transform weights are precomputed from existing geometry ingestion and reusable.
- [x] Transform tests match Hermes.

### Milestone A3: mirror ExB parity

- [x] X-Z `Div_n_bxGrad_f_B_XPPM` slice exists with fused/reference + autodiff tests.
- [x] local X-flux slice exists with fused/reference + dump-backed tests.
- [x] local field-aligned Y-flux slice exists with fused/reference + dump-backed tests.
- [x] local assembled full ExB mirror exists with fused/reference + dump-backed tests.
- [ ] `Ne exb` strict 1-step mismatch reduced below threshold.
- [ ] `Pe exb` strict 1-step mismatch reduced below threshold.
- [ ] 3-step audit remains green for ExB.

### Milestone A4: mirror parallel parity

- [ ] `n parallel/par` reduced below threshold.
- [ ] `Pe parallel/par_total` reduced below threshold.
- [ ] No regression in previously closed terms.

### Milestone A5: strict short-window parity (`t <= 0.1`)

- [ ] strict term leaders all below acceptance threshold
- [ ] fluctuation RMS within target band
- [ ] PSD within target band
- [ ] strict short-window gate promoted to required CI check

### Milestone B: longer baseline windows

- [ ] promote to `t <= 0.5`
- [ ] promote to `t <= 1.0`
- [ ] generate runtime and memory comparison table

### Milestone C: fold-back and cleanup

- [ ] validated mirror operators moved into unified core
- [ ] superseded patched Hermes code deleted
- [ ] mirror tests retained as regression gates

### Milestone D: beyond baseline

- [ ] conservative DRB path
- [ ] energy diagnostics in CI
- [ ] broader geometry and physics matrix

---

## 14) Dated Notes

### 2026-03-06

- Strategy reset adopted.
- Patch-based Milestone A closure on the old ExB/boundary path is suspended.
- New work will proceed through a Hermes-mirror JAX implementation with primitive-level validation.
- `src/jaxdrb/hermes_mirror/` created with Phase 1 boundary primitives:
  `limit_free`, `mc_limiter`, `apply_neumann_boundary_average_z`,
  `set_boundary_to_midpoint`.
- `tests/hermes_mirror/test_primitives.py` added with unit and autodiff checks.
- `tools/build_hermes_mirror_fixture.py` added for compact `.npz` fixture slicing.
- First dump-backed primitive fixture added:
  `tests/fixtures/hermes_mirror_ne_local_rank0_t1.npz`, derived from
  `runs/hermes_open_field_terms_t01_vortterms/data/BOUT.dmp.0.nc`
  at `t=0.01` for the local-rank `Ne` field.
- Mirror architecture and source citations documented in `docs/hermes_mirror.md`.
- Local validation is green after the scaffold landing:
  `ruff check src tests tools/build_hermes_mirror_fixture.py`,
  `black --check src tests tools/build_hermes_mirror_fixture.py`,
  `python -m pytest -q`.
- Phase 2 started with precomputed shifted-transform weights and both reference
  and fused implementations of:
  `to_field_aligned_nox` and `from_field_aligned_nobndry`.
- Transform validation is currently against the existing JAX shifted-transform
  path in the overlap region; Hermes dump-backed transform fixtures are still
  pending.
- The true Hermes source path for the current benchmark uses FFT-based
  `ShiftedMetric`; the mirror transform work therefore needs both:
  an overlap-checked linear path and a source-true FFT path.
- A stitched global transform fixture now exists at
  `tests/fixtures/hermes_mirror_shiftedmetric_global_t1.npz`, produced by
  `tools/build_hermes_mirror_transform_fixture.py` from the Hermes dump set.
- Phase 3 started with the first mirrored ExB X-Z slice in
  `src/jaxdrb/hermes_mirror/exb.py`:
  `div_n_bxgrad_f_b_xppm_xz` and `div_n_bxgrad_f_b_xppm_xz_ref`.
- `tests/hermes_mirror/test_exb.py` now covers fused/reference equality,
  equality with the current unified `hermes_xppm` X-Z path when
  `exb_poloidal_flows = false`, and autodiff.
- Strict Hermes example configs now pin `parallel_shift_interp = "spectral"`
  so the active strict path uses the same shifted-metric interpolation family
  as Hermes.
- Added the missing `RGN_ALL` linear and FFT shifted-transform helpers to the
  mirror layer and validated them against the current geometry adapter
  `to_field_aligned(...)` / `from_field_aligned(...)` paths.
- Switched the active poloidal Y-flux branch to full-region shifted transforms
  to match the Hermes source signature. The 1-step strict audit at
  `runs/audit_phase3_yregion_probe` showed no change in the fail-fast leader,
  so transform-region selection is not the dominant remaining
  `Pe advection/exb` mismatch.
- Added a local guard-inclusive derivative helper
  `src/jaxdrb/hermes_mirror/derivs.py::ddx_centered_guarded` and dump-backed
  fixture `tests/fixtures/hermes_mirror_phi_metric_local_rank0_t1.npz`.
- A direct production-path attempt to drop a ghost-centred `DDX(phi)` boundary
  formula into the active Y-flux path was rejected after the 1-step strict
  audit at `runs/audit_phase3_ddxghost_probe` regressed badly, especially in
  `n advection/exb` and `Pe advection/exb`.
- That rejection narrows the next target to the full local preparation chain:
  `DDX(phi)` + communication + Neumann guard application + shifted transform.
- The first helper for that local preparation chain is now landed in
  `src/jaxdrb/hermes_mirror/species.py`:
  `prepare_poloidal_y_dfdx_local_ref`.
- A new dump-backed local field-aligned fixture exists at
  `tests/fixtures/hermes_mirror_phi_field_aligned_local_rank0_t1.npz`.
- The new local prep regression in `tests/hermes_mirror/test_species.py`
  shows that the literal local prep path differs materially from a guardless
  approximation, which justifies continuing the mirror rewrite there instead of
  making more production-path guesses.
- Added the next local field-aligned Phase 3 slice in
  `src/jaxdrb/hermes_mirror/exb.py`:
  `div_n_bxgrad_f_b_xppm_xy_y_local_ref`,
  `div_n_bxgrad_f_b_xppm_xy_y_local`, and the corresponding `*_from_fields`
  wrappers.
- Added dump-backed regression coverage in
  `tests/hermes_mirror/test_exb_y_local.py` using
  `tests/fixtures/hermes_mirror_exb_local_rank0_t1.npz`.
- The fused and reference local Y-flux operators now match exactly on that
  fixture for both `Ne` and `Pe`, but there is still no strict-audit delta
  because the assembled runtime mirror ExB operator is not landed yet.
- Added the local X-flux preparation helper
  `prepare_poloidal_x_dfdy_local_ref` in
  `src/jaxdrb/hermes_mirror/species.py`.
- Added the local X-flux mirror operator slice in
  `src/jaxdrb/hermes_mirror/exb.py`:
  `div_n_bxgrad_f_b_xppm_xy_x_local_ref`,
  `div_n_bxgrad_f_b_xppm_xy_x_local`, and the corresponding `*_from_fields`
  wrappers.
- Added dump-backed regression coverage in
  `tests/hermes_mirror/test_exb_x_local.py`.
- Added the first assembled local full mirror ExB operator in
  `src/jaxdrb/hermes_mirror/exb.py`:
  `div_n_bxgrad_f_b_xppm_local_ref` and `div_n_bxgrad_f_b_xppm_local`.
- Added dump-backed fused/reference and autodiff coverage in
  `tests/hermes_mirror/test_exb_local_full.py`.
- Added a second dump-backed parity fixture
  `tests/fixtures/hermes_mirror_exb_term_local_rank0_t1.npz` containing the raw
  Hermes `term_Ne_exb` and `term_Pe_exb` arrays from the same local dump.
- The assembled mirror local ExB operator now matches Hermes on the physical
  interior cells:
  `Ne` interior diff RMS `2.8867991448834276e-05`,
  `Pe` interior diff RMS `1.2432835191026055e-05`,
  with interior correlations above `0.9998`.
- The remaining ExB mismatch is now localized to the lower open-boundary guard
  cells, especially the lower-x guard, lower-y guard, and lower-left corner of
  the X-flux path. The next Phase 3 target is guard/boundary diagnostic
  semantics, not interior operator algebra.
- Added the first Phase 4 species state-preparation helpers in
  `src/jaxdrb/hermes_mirror/species.py`:
  `density_transform_impl` and `pressure_transform_impl`.
- Added dump-backed regression coverage in
  `tests/hermes_mirror/test_transform_impl.py`.
- Transform-helper-level dump fixtures now match the Hermes-prepared state
  semantics for x-guard reconstruction and pressure/temperature consistency.
- The next remaining Phase 4 gap is the `finally` ordering and routing those
  prepared states into the strict runtime path.
- The centred-field `apply_neumann_field3d` branch is now landed.
- The remaining follow-up is to pin its named axis/region wiring directly
  against Hermes/BOUT when the mirror geometry/runtime path is connected.

---

## 15) Risks and Mitigations

1. Risk: the mirror path duplicates too much code.
   Mitigation: keep duplication temporary and limited to Stage 1 baseline operators only.

2. Risk: literal translation becomes too slow.
   Mitigation: require a reference implementation and a fused production implementation for each major operator.

3. Risk: geometry transforms become the new hidden mismatch.
   Mitigation: give transform functions their own milestone and dump-backed fixtures before wiring ExB.

4. Risk: documentation drifts from the new strategy.
   Mitigation: every mirror operator landing must update `open_field_alignment.md` and this file.

5. Risk: parity succeeds in the mirror engine but is hard to fold back.
   Mitigation: use clean module boundaries and keep tests at the operator level, not only at the engine level.

---

## 16) External References

### Hermes / BOUT++

- Hermes repository: <https://github.com/boutproject/hermes-3>
- Hermes docs (equations): <https://hermes3.readthedocs.io/en/latest/equations.html>
- Hermes docs (boundary conditions): <https://hermes3.readthedocs.io/en/latest/boundary_conditions.html>
- Hermes docs (solver numerics): <https://hermes3.readthedocs.io/en/stable/solver_numerics.html>

### Other context

- de Lucca conserving DRB references in local literature folder
- GBS / GRILLIX / Gkeyll comparison context retained for later stages

---

## 17) Plan Maintenance Rule

When a step is completed:

1. mark the checkbox in this file,
2. add a dated note under Section 14,
3. record the commit hash and affected files,
4. update associated docs/tests references,
5. keep this file as the single source of truth for handoff continuity.
