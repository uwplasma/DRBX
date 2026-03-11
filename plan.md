# jax_drb Agent Handoff + Master Plan

Last update: 2026-03-09
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
- Stop using the hybrid Hermes-alignment path as the active Milestone A vehicle.
- Freeze the current hybrid translation under a legacy namespace for traceability only.
- Build a fresh `hermes_literal` JAX path from the Hermes/BOUT component execution model outward.
- Reach strict early parity only through `engine = "hermes_literal"`.
- Once the literal engine passes the strict gates, promote its logic into the unified core and delete the superseded hybrid code.

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

Milestone A work is now a staged Hermes-literal rewrite, not a hybrid mirror or
an in-place patching program.

### Why

- The remaining strict-window mismatch is localized but structurally tangled.
- Repeated local edits in the current operator stack are no longer converging reliably.
- The fastest path to parity is to mirror the Hermes source stack directly,
  component by component, with the same state contract and execution order.

### What changes

- New Stage 1 baseline work goes into a fresh literal path under:
  - `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal`
- The previous hybrid translation path is frozen under:
  - `/Users/rogerio/local/jax_drb/src/jaxdrb/legacy_hermes`
- The literal path is allowed to duplicate limited Stage 1 logic temporarily.
- The current Hermes-specific logic in:
  - `/Users/rogerio/local/jax_drb/src/jaxdrb/core/geometry_field_aligned.py`
  - `/Users/rogerio/local/jax_drb/src/jaxdrb/core/terms/advection.py`
  - `/Users/rogerio/local/jax_drb/src/jaxdrb/core/terms/parallel.py`
  is frozen for Milestone A parity work and should not receive more speculative
  parity patches.
- After the literal path passes Milestone A and Milestone B, fold its logic
  back into the unified core and delete the superseded hybrid code.

### Historical note

The implementation log later in this file still contains detailed dated notes
from the earlier `hermes_mirror` phase. Those notes are retained for
traceability, but they now refer to the frozen hybrid path under
`/Users/rogerio/local/jax_drb/src/jaxdrb/legacy_hermes`.

### Temporary architecture allowance

The literal path may be introduced as a temporary engine mode, for example:

- `engine = "hermes_literal"`

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

- continued term-by-term gap-closing inside the old operator path
- using the hybrid mirror path as the active parity vehicle

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
- Active literal path: `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal`
- Frozen hybrid parity path: `/Users/rogerio/local/jax_drb/src/jaxdrb/legacy_hermes`
- Unified core: `/Users/rogerio/local/jax_drb/src/jaxdrb/core`
- Legacy path: `/Users/rogerio/local/jax_drb/src/jaxdrb/legacy_v1`
- Docs: `/Users/rogerio/local/jax_drb/docs`
- Tests: `/Users/rogerio/local/jax_drb/tests`
- Tools: `/Users/rogerio/local/jax_drb/tools`
- Runs: `/Users/rogerio/local/jax_drb/runs`

---

## 5) Hermes Source-of-Truth Map

The functions below define the literal scope. New JAX functions must cite the
Hermes source they mirror.

| Hermes source | Hermes function / logic | New JAX target | Purpose |
| --- | --- | --- | --- |
| `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/field/field3d.cxx` | guard-aware `Field3D` storage + `clearParallelSlices` + `setBoundaryTo` | `hermes_literal/field.py`, `hermes_literal/boundary_standard.py` | literal state and boundary contract |
| `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/boundary_standard.cxx` | `BoundaryNeumann::apply(Field3D&)` and related boundary ops | `hermes_literal/boundary_standard.py` | standard boundary semantics |
| `/Users/rogerio/local/hermes-3/src/sound_speed.cxx` | `SoundSpeed::transform_impl` | `hermes_literal/sound_speed.py::compute_fastest_wave` | shared fastest-wave state |
| `/Users/rogerio/local/hermes-3/src/evolve_density.cxx` | `transform_impl` / `finally` | `hermes_literal/evolve_density.py` | density component parity |
| `/Users/rogerio/local/hermes-3/src/evolve_pressure.cxx` | `transform_impl` / `finally` | `hermes_literal/evolve_pressure.py` | pressure component parity |
| `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/coordinates.cxx` | shifted-metric transforms | `hermes_literal/shifted_metric.py` | field-aligned communication semantics |
| `/Users/rogerio/local/hermes-3/src/div_ops.cxx` | `Div_n_bxGrad_f_B_XPPM`, `Div_par` | `hermes_literal/div_ops.py` | ExB and centered parallel operators |
| `/Users/rogerio/local/hermes-3/external/BOUT-dev/include/bout/fv_ops.hxx` | `FV::Div_par_mod`, `FV::Div_a_Grad_perp` | `hermes_literal/fv.py` | FV transport operators |
| `/Users/rogerio/local/hermes-3/src/vorticity.cxx` | vorticity component `finally` | `hermes_literal/vorticity.py` | literal omega path |
| `/Users/rogerio/local/hermes-3/src/*` | Stage 1 component order | `hermes_literal/engine.py` | strict parity engine |

### Explicit non-goals for the mirror rewrite

Do not copy the full BOUT++ object model, component permissions system, or MPI layer.

Instead:

- preserve only the numerical semantics,
- flatten all required state into immutable JAX-friendly data structures,
- emulate the final single-device numerical result.

---

## 6) New JAX Literal Layout

Create these files first:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/__init__.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/types.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/field.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/boundary_standard.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/state.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/sound_speed.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/evolve_density.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/evolve_pressure.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/shifted_metric.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/div_ops.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/fv.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/vorticity.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/engine.py`

### Intended roles

- `types.py`
  - immutable layout and runtime metadata used by literal guarded fields
- `field.py`
  - guard-aware storage helpers and interior/guard views
- `boundary_standard.py`
  - literal translations of BOUT standard boundary operators and
    `Field3D::setBoundaryTo`
- `state.py`
  - immutable literal Stage 1 state containers
- `sound_speed.py`
  - literal `SoundSpeed::transform_impl`
- `evolve_density.py`
  - literal `EvolveDensity::transform_impl` and then `finally`
- `evolve_pressure.py`
  - literal `EvolvePressure::transform_impl` and then `finally`
- `shifted_metric.py`
  - literal `toFieldAligned` / `fromFieldAligned` and communication helpers
- `div_ops.py`
  - literal `Div_n_bxGrad_f_B_XPPM` and centered `Div_par`
- `fv.py`
  - literal `FV::Div_par_mod` and `FV::Div_a_Grad_perp`
- `vorticity.py`
  - literal `Vorticity::finally`
- `engine.py`
  - strict Stage 1 component ordering and runtime assembly

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
- [x] runtime-facing fused production version

Wire first for:

- [x] density ExB term
- [x] pressure ExB term

Then:

- [x] temperature ExB term
- [x] vorticity ExB term

Acceptance:

- [x] dump-backed local `Ne` / `Pe` ExB terms match Hermes on interior cells
- [x] lower-open-boundary guard cells match Hermes diagnostic semantics
- [ ] `Ne exb` strict 1-step leader removed
- [ ] `Pe exb` strict 1-step leader removed
- [ ] 3-step term audit stays improved

### Phase 4: Species state-preparation mirror

- [x] local `DDX -> applyBoundary("neumann") -> toFieldAligned` prep helper exists
- [x] mirror density `transform_impl`
- [x] mirror density `finally`
- [x] mirror pressure `transform_impl`
- [x] mirror pressure `finally`
- [x] mirror sheath guard preparation order used by Stage 1 baseline

Acceptance:

- [x] boundary-state dump fixtures match Hermes at transform-helper level
- [ ] mirror ExB operator uses the same prepared states as Hermes

### Phase 5: Parallel FV mirror

Create in this order:

- [x] limiter and reconstruction helpers from Hermes FV path
- [ ] `div_par_mod_ref`
- [x] fused `div_par_mod`
- [x] density parallel term wiring
- [x] pressure parallel term wiring
- [x] centered `Div_par(jpar)` mirror wiring
- [ ] runtime sheath / guard / transform contract feeding the mirror operator
- [ ] later momentum and energy channels as needed

Acceptance:

- [ ] `n parallel/par` below threshold
- [ ] `Pe parallel/par_total` below threshold
- [ ] no regression in already-closed `omega parallel/jpar`
- [x] dump-backed local `term_Ne_par` / `term_Pe_par` / `term_Vort_jpar`
  mirror parity regression exists

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

### Planned literal strict run

```bash
jaxdrb /Users/rogerio/local/jax_drb/examples/open_field_line/input_tokamak_bxcv_alignment_strict_mirror.toml --run --output /tmp/jax_mirror_short.npz
```

---

## 13) Progress Tracker

### Milestone A0: strategy reset

- [x] Decide to stop patching the old Hermes parity path in place.
- [x] Define the Hermes source-of-truth function map.
- [x] Rewrite this plan around the literal Hermes strategy.
- [x] Freeze the hybrid implementation under `src/jaxdrb/legacy_hermes`.
- [x] Add the fresh `src/jaxdrb/hermes_literal` scaffold and config aliases.

### Milestone A1: literal primitives

- [x] Literal boundary primitives exist.
- [x] Primitive dump-backed fixtures exist.
- [x] Primitive tests are green in CI.

### Milestone A2: literal transforms

- [x] `to_field_aligned_nox` and `from_field_aligned_nobndry` mirror implementations exist.
- [x] Transform weights are precomputed from existing geometry ingestion and reusable.
- [x] Transform tests match Hermes.
- 2026-03-09: the fresh `hermes_literal/shifted_metric.py` implementation is
  now landed and validated independently of `legacy_hermes`, including linear
  and FFT paths plus a dump-backed roundtrip fixture in
  `tests/hermes_literal/test_shifted_metric.py`.

### Milestone A3: literal ExB parity

- [x] X-Z `Div_n_bxGrad_f_B_XPPM` slice exists with fused/reference + autodiff tests.
- [x] local X-flux slice exists with fused/reference + dump-backed tests.
- [x] local field-aligned Y-flux slice exists with fused/reference + dump-backed tests.
- [x] local assembled full ExB mirror exists with fused/reference + dump-backed tests.
- [x] local assembled full ExB mirror matches Hermes on all dump-backed cells.
- [ ] `Ne exb` strict 1-step mismatch reduced below threshold.
- [ ] `Pe exb` strict 1-step mismatch reduced below threshold.
- [ ] 3-step audit remains green for ExB.
- 2026-03-09: the fresh `hermes_literal/exb.py`, `hermes_literal/delp2.py`,
  and `hermes_literal/vorticity.py` modules are now landed and validated
  independently of `legacy_hermes`, including dump-backed runtime regressions in
  `tests/hermes_literal/test_literal_exb_runtime.py` and
  `tests/hermes_literal/test_literal_vorticity.py`.
- 2026-03-09: the active strict runtime now imports the fresh literal ExB and
  vorticity path from `src/jaxdrb/hermes_literal/`. The 1-step audit at
  `runs/audit_literal_runtime_promotion_1step` preserved the current fail-fast
  ordering, with `n advection/exb = 0.09623829491706752` and
  `Pe advection/exb = 0.0676385260919583` still below the parallel leaders.

### Milestone A4: literal parallel parity

- [x] dump-backed local mirror parallel operator matches Hermes terms.
- [ ] `n parallel/par` reduced below threshold.
- [ ] `Pe parallel/par_total` reduced below threshold.
- [ ] No regression in previously closed terms.
- 2026-03-09: the fresh `hermes_literal/fv.py::div_par_mod` and
  `hermes_literal/div_ops.py::div_par_centered` translations are now landed and
  validated independently of `legacy_hermes`, including dump-backed local
  regressions in `tests/hermes_literal/test_literal_parallel_dump.py`.
- 2026-03-09: the active strict runtime now imports the fresh literal parallel
  transport and mirror RHS cache path from `src/jaxdrb/hermes_literal/`. The
  1-step audit at `runs/audit_literal_runtime_promotion_1step` preserved the
  current leaders:
  `n parallel/par = 0.13383127252151306`,
  `omega parallel/jpar = 0.11697795624618619`,
  `Pe parallel/par_total = 0.1133024567583403`.
- 2026-03-09: the reduced density/pressure cache now pulls its live parallel
  runtime state from `src/jaxdrb/hermes_literal/parallel.py` rather than
  `src/jaxdrb/core/terms/parallel.py`. The rehome audit at
  `runs/audit_literal_parallel_runtime_rehome` preserves the literal-engine
  baseline and moves the top parallel rows slightly in the right direction:
  `Te parallel 0.1474904091090806 -> 0.14748382093236653`,
  `n parallel 0.13383127252151306 -> 0.1338298917677307`,
  `Pe parallel 0.1133024567583403 -> 0.11330241103262646`.

### Milestone A5: strict short-window parity (`t <= 0.1`)

- [x] strict early config runs through `engine = "hermes_literal"`
- [ ] strict term leaders all below acceptance threshold
- [ ] fluctuation RMS within target band
- [ ] PSD within target band
- [ ] strict short-window gate promoted to required CI check
- 2026-03-09: the strict early parity config
  `/Users/rogerio/local/jax_drb/examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml`
  now explicitly sets `engine = "hermes_literal"`. The smallest strict audit at
  `runs/audit_literal_engine_smoke` runs through that engine successfully and
  keeps the current 1-step leaders at:
  `Te parallel = 0.1474904091090806`,
  `n parallel = 0.13383127252151306`,
  `omega parallel = 0.11697795624618619`,
  `Pe parallel = 0.1133024567583403`,
  `n advection = 0.09623829491706752`,
  `Pe advection = 0.0676385260919583`.

### Milestone B: longer baseline windows

- [ ] promote to `t <= 0.5`
- [ ] promote to `t <= 1.0`
- [ ] generate runtime and memory comparison table

### Milestone C: fold-back and cleanup

- [ ] validated literal operators moved into unified core
- [ ] superseded patched Hermes code deleted
- [ ] literal and legacy regression tests retained as traceability gates

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
- The remaining ExB mismatch was then traced to the local `DDY(f)` preparation
  path for the X-flux branch: the mirror helper was missing the lower-open
  parallel `applyBoundary("neumann")` step that Hermes applies after
  `mesh->communicate(dfdy)`.
- That fix is now landed in `src/jaxdrb/hermes_mirror/species.py` and threaded
  through the local X-flux/operator entrypoints. Dump-backed full-term parity is
  now closed across all local cells:
  `Ne` all-cell diff RMS `3.072901445531812e-05`,
  `Pe` all-cell diff RMS `1.3376334360587529e-05`,
  with all-cell correlations above `0.99998`.
- The next remaining Phase 3 target is runtime promotion of that same mirror
  ExB path into the strict Hermes audit configs, not further local operator
  reconstruction.
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
- Added the opt-in runtime wrapper
  `src/jaxdrb/hermes_mirror/exb.py::div_n_bxgrad_f_b_xppm` plus dump-backed
  regression `tests/hermes_mirror/test_exb_runtime.py`. On the local-rank
  fixture interior, the wrapper reaches:
  `Ne` RMS `2.488462499110523e-04`,
  `Pe` RMS `2.6183313968993464e-04`,
  with correlations above `0.983`.
- Wired that wrapper into the active field-aligned geometry adapter behind
  `exb_flux_scheme = "hermes_mirror"` and added geometry-path coverage in
  `tests/test_exb_poloidal_flows.py`.
- Added a stitched global runtime fixture via
  `tools/build_hermes_mirror_runtime_fixture.py` and
  `tests/fixtures/hermes_mirror_exb_global_t1.npz`, plus the regression
  `tests/hermes_mirror/test_exb_runtime_global.py`.
- Added the hybrid open-boundary runtime knob
  `hermes_mirror_parallel_edge_block`. Re-evaluating only the first and last
  parallel edge blocks with the local guard-inclusive mirror operator improves
  the direct global Hermes term arrays from:
  `Ne RMS 9.281612304656274e-04 -> 2.7785371223075885e-04`,
  `Pe RMS 9.436398753984853e-04 -> 2.9023628701603716e-04`,
  with correlations above `0.996`.
- The smallest strict gate with that edge-block wrapper is
  `runs/audit_hermes_mirror_edge_block_1step`. The current scalar fail-fast
  metric only moves slightly:
  `omega advection/exb 0.06804918916596805 -> 0.06712108791244092`,
  `Pe advection/exb 0.038900114007649214 -> 0.03873682407548267`.
- The audit tool now also writes direct term-array mismatch metrics:
  `array_diff_rms`, `array_rel_diff`, `array_corr`, and
  `weighted_array_rel`. `first_failing_terms.csv` now defaults to ranking by
  the array metric (`--term-ranking-metric=array`) while preserving the older
  RMS-magnitude columns for continuity.
- The runtime mirror path now supports the same non-unit poloidal scaling
  contract as the legacy geometry path:
  `exb_poloidal_scale`, `exb_poloidal_x_scale`,
  `exb_poloidal_y_scale`. The strict Hermes baseline no longer uses the old
  pre-mirror Y-branch tuning; it is now fixed at `1.0`.
- The strict early parity config
  `examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml`
  is now promoted to:
  `exb_flux_scheme = "hermes_mirror"` and
  `hermes_mirror_parallel_edge_block = 8`.
- The promoted 1-step Hermes-state audit is
  `runs/audit_strict_early_mirror_promoted_1step`. Relative to the previous
  array-ranked strict baseline (`runs/audit_current_arraymetric_1step`), the
  dominant ExB transport channels improve materially:
  `n advection/exb 0.6415487257460786 -> 0.30603226941513645`,
  `Pe advection/exb 0.43066567430657776 -> 0.20417452847516265`,
  with correlations improving to `0.9947894182550701` and
  `0.9952771323120512`.
- The closed parallel channels remain unchanged in the promoted config:
  `omega parallel/jpar = 0.2107103945115671`,
  `n parallel/par = 0.16847301041461074`,
  `Pe parallel/par_total = 0.15454019751690204`
  in the weighted-array metric.
- Remaining blocker after promotion:
  `omega advection/exb` worsens from `0.007979974955211428` to
  `0.09741634145346564` in weighted-array metric even though the dominant
  density/pressure ExB channels improve strongly.
- Next target: decompose the promoted runtime vorticity ExB term into its
  three branches (`-Div(phi,0.5*omega)`, `vE·grad(pi_hat)`, and
  `-Div(phi+pi_hat, Delp2(phi)/(2B^2))`) and match the Hermes `term_Vort_exb`
  composition on the same promoted mirror path.
- 2026-03-07: landed the next literal vorticity-side primitives without
  promoting them into production omega ExB:
  `src/jaxdrb/hermes_mirror/boundary.py::apply_free_o2_field3d`,
  `src/jaxdrb/hermes_mirror/fv.py::div_a_grad_perp_local`,
  `src/jaxdrb/hermes_mirror/fv.py::div_a_grad_perp`,
  `src/jaxdrb/hermes_mirror/vorticity.py::full_omega_exb_advection`,
  plus the stitched fixture
  `tests/fixtures/hermes_mirror_vorticity_global_t1.npz` and builder
  `tools/build_hermes_mirror_vorticity_fixture.py`.
- 2026-03-07: these operators are now covered by synthetic, autodiff, and
  dump-backed regressions in
  `tests/hermes_mirror/test_primitives.py`,
  `tests/hermes_mirror/test_fv.py`,
  and `tests/hermes_mirror/test_vorticity.py`.
- 2026-03-07: landed the source-true literal `Delp2(phi)` path in
  `src/jaxdrb/hermes_mirror/delp2.py`, extended
  `src/jaxdrb/core/geometry_field_aligned.py` to ingest Hermes `G1`/`G3`/`d1_dx`,
  updated `tools/convert_hermes_dump_axisymmetric.py` to emit `G1`/`G3`, and
  stitched those coefficients into
  `examples/open_field_line/axisym_tokamak_bxcv_hermes_norm_parcurv_g22.npz`.
- 2026-03-07: the expanded stitched vorticity fixture
  `tests/fixtures/hermes_mirror_vorticity_global_t1.npz` now carries the raw
  Hermes `G1`, `G3`, `g11`, `g13`, `g33`, `dx`, `dz`, `Bxy`, and `zShift`
  planes. The literal Laplacian matches the raw BOUT single-index operator at
  both local and stitched-global level:
  local corr `0.9999999979364631`, local diff RMS `6.903925415803028e-07`;
  global corr `0.9999988050053542`, global diff RMS `3.9164034002630735e-05`.
- 2026-03-07: the remaining promoted omega mismatch turned out not to be
  `Delp2(phi)` itself but the transport-side boundary contract. Passing
  `poisson_invert_set` through the runtime mirror transport of `phi` /
  `phi + Pi_hat` was wrong. After removing that override and routing the omega
  path through the validated global mirror wrapper, the dump-backed full
  `term_Vort_exb` mirror moved to correlation `0.9286922397070627` with diff
  RMS `9.242617198253543e-06`.
- 2026-03-07: the promoted strict 1-step Hermes-state audit
  `runs/audit_mirror_omega_transport_bc_fix_1step` reduces
  `omega advection/exb` from weighted-array metric `0.09741634145346564` to
  `0.0035704721275969927`. The fail-fast leader is now back to
  `n advection/exb = 0.30603226941513645`, followed by
  `omega parallel/jpar = 0.2107103945115671`,
  `Pe advection/exb = 0.20417452847516265`,
  `n parallel/par = 0.16847301041461074`, and
  `Pe parallel/par_total = 0.15454019751690204`.
- 2026-03-08: the promoted mirror path was still carrying the old
  pre-mirror `exb_poloidal_y_scale = 1.24` knob. On the dump-backed global
  mirror fixture, that single multiplier reproduces the live promoted
  density/pressure overshoot almost exactly. Resetting the strict config to
  `exb_poloidal_y_scale = 1.0` moves the 1-step Hermes-state audit
  `runs/audit_poloidal_y_scale_1p0_1step` to:
  `n advection/exb = 0.09608755774957915`,
  `Pe advection/exb = 0.06745309373399326`,
  `omega advection/exb = 0.004178515908061414`,
  leaving `omega parallel/jpar = 0.2107106038839909` as the next true
  fail-fast leader.
- 2026-03-08: `omega parallel/jpar` was still going through the FV transport
  helper even though Hermes vorticity uses plain `Div_par(jpar)`. The
  `wave=None` current branch in `src/jaxdrb/core/terms/parallel.py` now uses a
  centered `Div_par`-style divergence with explicit sheath-face current values.
  The strict 1-step Hermes-state audit `runs/audit_jpar_centered_1step`
  improves `omega parallel/jpar` from
  `0.2107106038839909 -> 0.11715792736854537` in weighted-array metric, and
  the 3-step window `runs/audit_jpar_centered_3step` keeps it below the
  previous promoted baseline at `t=0.01..0.03`. The next fail-fast terms are
  now `n parallel/par`, `Te parallel/par_total`, and `Pe parallel/par_total`,
  with the audit-level `Te/Pe sheath source_residual_boundary` bookkeeping gap
  still pending as a separate boundary-energy follow-up.
- 2026-03-09: the next strict parallel slice tightened the open-field
  finite-volume density and pressure channels to use sheath ghost states in
  the boundary-adjacent limited reconstruction, not only in the explicit
  sheath face flux. The confirm audit
  `runs/audit_parallel_ghost_stencil_confirm_1step` gives a small but real
  reduction in the promoted parallel leaders at `t=0.01`:
  `n parallel/par 0.15689932456328756 -> 0.15650650752322878`,
  `Te parallel/par_total 0.15587502102513381 -> 0.1556861680908554`,
  `Pe parallel/par_total 0.15453748447303708 -> 0.154109603265596`,
  while `omega parallel/jpar` stays fixed at `0.11715792736854537`.
- 2026-03-09: a simpler sheath-energy hypothesis was explicitly rejected.
  Replacing the current electron-sheath energy closure with a constant
  Hermes-like `gamma_e = 3.5` contract made the audit-level boundary-energy
  rows diverge badly in `runs/audit_parallel_and_sheath_fix_1step`
  (`Te sheath/source_residual_boundary 0.26493593205386157 -> 8.4765`,
  `Pe sheath/source_residual_boundary 0.11172993716659292 -> 3.6046`).
  The remaining `Te/Pe sheath source_residual_boundary` gap is therefore in
  the bookkeeping contract, not a single gamma coefficient.
- 2026-03-09: the next strict parallel slice corrected the finite-wave
  sheath-face metric factor in `src/jaxdrb/core/terms/parallel.py`. Hermes
  `FV::Div_par_mod` uses the boundary-cell metric on the sheath face even for
  the finite-wave density/pressure channels; the JAX path had still been using
  the first interior-face factor there. The promoted 1-step audit
  `runs/audit_parallel_boundary_metric_retry_1step` improves
  `n parallel/par 0.15650650752322878 -> 0.13432807982024225`,
  `Te parallel/par_total 0.1556861680908554 -> 0.14849268403368665`,
  and `Pe parallel/par_total 0.154109603265596 -> 0.11330115527226602`,
  while `omega parallel/jpar` stays fixed at `0.11715751270556365`.
- 2026-03-09: the 3-step confirm window
  `runs/audit_parallel_boundary_metric_retry_3step` preserves that gain
  through `t=0.03` with correlations above `0.9990` in the parallel channels.
  The audit-level `Te/Pe sheath source_residual_boundary` rows are unchanged,
  reinforcing that the remaining gap there is a residual bookkeeping issue
  rather than the same parallel transport bug.
- 2026-03-09: the next audit-side slice now reconstructs the Hermes electron
  sheath pressure source directly from the raw BOUT dumps as a synthetic
  `term_Pe_sheath`, mirroring `sheath_boundary.cxx`, and the `Pe/Te sheath`
  mismatch rows now prefer that direct term before falling back to the mixed
  `source_residual_boundary` bucket. In
  `runs/audit_direct_sheath_mapping_1step`, this moves
  `Pe sheath/sheath` to `0.022641938293208385` weighted-array mismatch with
  correlation `1.0`, and `Te sheath/sheath` to
  `0.08160536527344078` with correlation `1.0`.
- 2026-03-09: `n sheath/source_residual_boundary` remains a bookkeeping row at
  `0.006210587208797745`. The direct `Pe/Te sheath` parity is now measured
  cleanly; the remaining `source_residual_boundary` discrepancy is explicitly
  an audit residual channel rather than the primary electron sheath term.
- 2026-03-09: the literal Hermes parallel FV mirror is now landed in
  `src/jaxdrb/hermes_mirror/parallel.py`, promoted in the strict configs via
  `parallel_flux_scheme = "hermes_mirror"`, and validated directly against the
  rank-0 raw dump fixture `tests/fixtures/hermes_mirror_parallel_local_rank0_t1.npz`.
  The regression `tests/hermes_mirror/test_parallel_dump.py` now matches
  `term_Ne_par`, `term_Pe_par`, and `term_Vort_jpar` at operator level.
- 2026-03-09: the promoted strict audits
  `runs/audit_parallel_mirror_with_dy_1step` and
  `runs/audit_parallel_mirror_with_dy_3step` are unchanged relative to the
  previous promoted live baseline. This isolates the remaining live
  `n/Pe/jpar` parallel mismatch to the runtime sheath / guard / transform
  contract feeding the operator, not to the mirrored FV operator itself. The
  next coherent refactor target is therefore a literal Hermes sheath-state
  transform feeding the mirror parallel path.
- 2026-03-09: the literal sheath-state transform is now landed in
  `src/jaxdrb/hermes_mirror/sheath.py`, with new dump-backed fixtures
  `tests/fixtures/hermes_mirror_parallel_local_rank0_t1.npz` and
  `tests/fixtures/hermes_mirror_parallel_local_rank5_t1.npz`, and regression
  coverage in `tests/hermes_mirror/test_sheath.py`. That closes the Stage 1
  mirror guard-preparation step directly against
  `/Users/rogerio/local/hermes-3/src/sheath_boundary.cxx`: the open-end guard
  reconstruction now reproduces Hermes closely enough to recover
  `term_Ne_par`, `term_Pe_par`, and `term_Vort_jpar` without feeding dumped
  guard cells into the operator.
- 2026-03-09: promoting the guard builder alone did not move the strict live
  audits, which exposed the missing shifted-transform contract. In
  `src/jaxdrb/core/terms/parallel.py`, the solver was already shifting the
  cell-centered fields and explicit boundary face fluxes into field-aligned
  coordinates, but not `ghost_low/high_f` or `ghost_low/high_v`. That runtime
  omission is now fixed and covered by
  `tests/test_parallel_shifted_boundary_flux.py`. The promoted audit
  `runs/audit_sheath_shifted_ghosts_1step` then gives a small real improvement:
  `n parallel/par 0.13448644700674087 -> 0.1338459414001929`,
  `omega parallel/jpar 0.1169915003671119 -> 0.11697747997572151`,
  `Pe parallel/par_total 0.11335202275260099 -> 0.11330118219042988`. The
  runtime sheath / guard / transform contract is therefore narrower now, but
  not closed yet. The 3-step confirm window
  `runs/audit_sheath_shifted_ghosts_3step` preserves the same sign of
  improvement through `t=0.03`.
- 2026-03-09: the next full-refactor slice closes the remaining density and
  pressure state-preparation stubs in the promoted mirror runtime. New global
  no-guard helpers now exist in `src/jaxdrb/hermes_mirror/species.py`:
  `density_final_global`, `pressure_final_global`, and
  `prepare_reduced_species_state_global`. `src/jaxdrb/core/terms/context.py`
  now constructs a prepared mirror species state once, and the live strict ExB
  and parallel paths in `src/jaxdrb/core/terms/advection.py` and
  `src/jaxdrb/core/terms/parallel.py` consume `ctx.n_prepared`,
  `ctx.pe_prepared`, `ctx.Te_prepared`, `ctx.pi_prepared`, and
  `ctx.Ti_prepared` instead of rebuilding transformed fields locally.
- 2026-03-09: the new architectural layer is covered by
  `tests/hermes_mirror/test_species.py`, and the promoted audits
  `runs/audit_full_species_prep_1step` and
  `runs/audit_full_species_prep_3step` are unchanged relative to
  `runs/audit_sheath_shifted_ghosts_1step` and
  `runs/audit_sheath_shifted_ghosts_3step` up to roundoff. That is still a
  valid narrowing result: the dominant remaining Milestone A gap is now more
  clearly below the density/pressure state-preparation layer, in the lower
  operator / communication contract.
- 2026-03-09: `src/jaxdrb/hermes_mirror/rhs.py` now contains the reduced-model
  density and pressure `finally()` assembly used by the strict mirror runtime:
  `density_rhs_terms`, `pressure_rhs_terms`, and
  `build_reduced_mirror_term_cache`. `src/jaxdrb/core/terms/registry.py` now
  routes the live `advection` and `parallel` term groups through that cache
  whenever `exb_flux_scheme = "hermes_mirror"` or
  `parallel_flux_scheme = "hermes_mirror"`. This closes the Phase 4
  density/pressure `finally()` item at the runtime assembly level.
- 2026-03-09: the promoted strict audit
  `runs/audit_mirror_rhs_cache_1step` is numerically identical to
  `runs/audit_full_species_prep_1step_rerun`. That rules out the scheduler/RHS
  assembly layer as the remaining Milestone A blocker.
- 2026-03-09: the next lower contract fix landed in
  `src/jaxdrb/core/terms/parallel.py`: shifted boundary ghost planes now use
  the same Hermes-mirror shifted-metric implementation as interior fields,
  including the spectral path selected by
  `parallel_shift_interp = "spectral"`. The new regression is
  `tests/test_parallel_shifted_boundary_flux.py`.
- 2026-03-09: the promoted strict audit
  `runs/audit_parallel_boundary_spectral_shift_1step` shows the expected
  lower-level effect: `n parallel/par`
  `0.1338459414001856 -> 0.13383127252151306`, while
  `omega parallel/jpar` and `Pe parallel/par_total` move only at the
  numerical-noise level. The remaining blocker is therefore still the runtime
  sheath / guard / transform contract feeding the mirror parallel operator as a
  whole, not just the term assembly or boundary-plane interpolation choice.
- 2026-03-09: `src/jaxdrb/hermes_literal/rhs.py` no longer imports the live
  advection or parallel term groups from `core/terms`. The strict literal cache
  now gets those groups from:
  `src/jaxdrb/hermes_literal/advection.py` and
  `src/jaxdrb/hermes_literal/parallel.py`.
- 2026-03-09: `src/jaxdrb/hermes_literal/advection.py` preserves the existing
  runtime switches `exb_advection_simplified` and
  `exb_advect_conservative`, with regressions in
  `tests/hermes_literal/test_literal_advection_runtime.py` and
  `tests/test_vorticity_alignment_switches.py`.
- 2026-03-09: the strict 1-step audit
  `runs/audit_literal_advection_parallel_rehome_1step_after_fix` is unchanged
  on the leading rows relative to the prior literal-engine baseline:
  `n advection/exb = 0.09623829491706752`,
  `Pe advection/exb = 0.0676385260919583`,
  `n parallel/par = 0.1338298917677307`,
  `Te parallel/par_total = 0.14748382093236653`.
  That is the expected result for this slice: it removes another hybrid
  dependency without changing the live parity ranking.
- 2026-03-09: `src/jaxdrb/hermes_literal/engine.py` now calls
  `src/jaxdrb/hermes_literal/context.py::build_context` instead of importing
  `core.terms.build_context`. The new regression is
  `tests/hermes_literal/test_literal_context.py`.
- 2026-03-09: the literal context rehome preserves the strict runtime contract
  for `n_phys`, `Te_phys`, solved `phi`, and the prepared density/pressure
  state. This does not change the audit ranking, but it removes another direct
  dependency from the literal engine onto `core/terms`.
- 2026-03-09: `src/jaxdrb/hermes_literal/engine.py` now imports its active
  schedule and dispatch table from `src/jaxdrb/hermes_literal/registry.py`
  rather than from `core.terms.registry`. The new regression is
  `tests/hermes_literal/test_literal_registry.py`.
- 2026-03-09: the registry rehome does not change parity, but it reduces the
  remaining direct ownership gaps inside the literal engine itself. The next
  structural target is replacing the shared registry contents with literal
  wrappers for the remaining Stage 1 term groups.
- 2026-03-09: the strict literal config now sets
  `hermes_mirror_parallel_subdomain_size = 8`, and
  `src/jaxdrb/hermes_literal/exb.py` evaluates the literal ExB runtime
  blockwise over Hermes-sized local parallel chunks.
- 2026-03-09: the stitched-global ExB regression in
  `tests/hermes_literal/test_literal_exb_runtime.py` confirms that the
  blockwise local runtime improves the raw Hermes `Ne/Pe` ExB term relative
  error from about `0.097/0.107` to about `0.061/0.066`.
- 2026-03-09: the strict 1-step audit
  `runs/audit_literal_subdomain_parallel_1step` shows the same live gain:
  `n advection/exb = 0.06021497597645309`,
  `Te advection/exb = 0.03175328243530484`,
  `Pe advection/exb = 0.0417892594173691`,
  while the leading parallel rows remain unchanged at about
  `0.113-0.147`. This is progress, not closure: Milestone A is still above the
  `1e-2` target and should not be promoted to longer benchmark windows yet.
- 2026-03-09: `src/jaxdrb/hermes_literal/communicate.py` now owns the local
  parallel-slab assembly used by the promoted literal ExB runtime. The new
  regression is `tests/hermes_literal/test_literal_communicate.py`.
- 2026-03-09: `src/jaxdrb/hermes_literal/exb.py` now builds its promoted
  `MYSUB`-sized local slabs through that helper, while preserving the current
  validated internal seam contract. The stitched global Hermes regression in
  `tests/hermes_literal/test_literal_exb_runtime.py` is unchanged at the
  improved `Ne/Pe` raw relative errors `0.06090172693816785 / 0.06601079736963186`.
- 2026-03-09: the strict 1-step audit
  `runs/audit_literal_comm_layer_1step` is numerically identical to
  `runs/audit_literal_subdomain_parallel_1step` on the leading rows. This
  slice is therefore architectural, not a new parity jump: it removes hidden
  ExB runtime assembly logic without changing the promoted literal baseline.
- 2026-03-09: `src/jaxdrb/hermes_literal/context.py` now resolves BCs and
  `is_2d` through local copies in
  `src/jaxdrb/hermes_literal/bcs.py` and
  `src/jaxdrb/hermes_literal/ops.py` rather than importing those helpers from
  `core.terms`. The remaining major shared Stage 1 context dependency is still
  the Poisson/field helper layer in `core.terms.fields`.
- 2026-03-10: the strict `hermes_literal` engine no longer imports any Stage 1
  schedule or term implementations from `core.terms`. The active literal
  package now owns local copies of:
  `fields.py`, `sol.py`, `bc_relaxation.py`, `braginskii.py`,
  `curvature.py`, `diamagnetic.py`, `diffusion.py`, `drive.py`,
  `extra_dissipation.py`, `line_bcs.py`, `neutrals.py`, `perp_bc.py`,
  `region_bc.py`, `sheath_terms.py`, `volume_source.py`, and `registry.py`.
- 2026-03-10: focused literal-engine regressions
  `tests/hermes_literal/test_literal_context.py`,
  `tests/hermes_literal/test_literal_parallel_runtime.py`,
  `tests/hermes_literal/test_literal_exb_runtime.py`,
  `tests/test_open_field_strict_config.py`, and
  `tests/test_drb_fv_engine.py`
  pass against the local term-layer rehome.
- 2026-03-10: the strict 1-step audit
  `runs/audit_literal_local_registry_1step`
  is numerically identical to
  `runs/audit_literal_comm_layer_1step`
  on the remaining leaders:
  `n parallel/par = 0.1338298917677307`,
  `omega parallel/jpar = 0.11697795624618619`,
  `Te parallel/par_total = 0.14748382093236653`,
  `Pe parallel/par_total = 0.11330241103262646`,
  `n advection/exb = 0.06021497597645309`,
  `Pe advection/exb = 0.0417892594173691`.
  That is the intended result for this slice: the parity vehicle is now more
  fully owned by `hermes_literal`, and the remaining mismatch is in the live
  runtime state/communication contract rather than in shared unified term code.
- 2026-03-10: `src/jaxdrb/hermes_literal/context.py` now builds an explicit
  prepared `LiteralStage1State` from
  `src/jaxdrb/hermes_literal/state.py`, including guarded density, pressure,
  temperature, velocity, `phi`, `sound_speed`, and `fastest_wave` fields plus
  shared `Field3DLayout` metadata.
- 2026-03-10: the new regressions
  `tests/hermes_literal/test_literal_context.py` and
  `tests/hermes_literal/test_literal_state.py`
  verify that the guarded literal state preserves the prepared interior fields
  exposed by the strict runtime.
- 2026-03-10: `src/jaxdrb/hermes_literal/parallel.py` now reads `fastest_wave`
  from that prepared literal state when available instead of recomputing it
  ad hoc, while remaining compatible with older core-term contexts used by
  comparison tests.
- 2026-03-10: the strict 1-step audit
  `runs/audit_literal_stage1_state_1step`
  is numerically identical to
  `runs/audit_literal_local_registry_1step`
  on the live leaders. This is again the intended hard-reset outcome:
  more runtime ownership moves into `hermes_literal` without perturbing the
  current promoted parity baseline.
- 2026-03-10: `src/jaxdrb/hermes_literal/engine.py` no longer stores a live
  unified `DRBSystem` as `base_system`. The literal engine now owns split
  application, physical/log conversion, FD/spectral operator dispatch,
  Poisson/polarization cache ownership, and energy diagnostics directly.
- 2026-03-10: focused engine regressions
  `tests/test_drb_fv_engine.py`,
  `tests/test_open_field_strict_config.py`,
  `tests/hermes_literal/test_literal_context.py`,
  `tests/hermes_literal/test_literal_parallel_runtime.py`, and
  `tests/hermes_literal/test_literal_exb_runtime.py`
  pass after removing the unified runtime object from the active literal path.
- 2026-03-10: the strict 1-step audit
  `runs/audit_literal_engine_no_base_1step`
  is numerically identical to
  `runs/audit_literal_stage1_state_1step`
  on the remaining leaders. This means the active parity vehicle now depends
  less on the unified runtime without changing the promoted baseline.
- 2026-03-10: `tools/audit_term_alignment.py` now reports true relative array
  errors. The previous `array_rel_diff` and `weighted_array_rel` columns were
  normalized by `0.1 * hermes_rms`, so the promoted literal-engine gaps were
  overstated by `10x`. The legacy stricter normalization is retained in the
  new `scaled_*` columns for historical tracking.
- 2026-03-10: with the corrected strict audit
  `runs/audit_literal_engine_true_rel_reverted_1step`,
  the live parallel channels are already near the `1e-2` target on the actual
  array metric:
  `Te parallel/par_total = 0.015436351897318895`,
  `n parallel/par = 0.013543302195114624`,
  `omega parallel/jpar = 0.011715130895629948`,
  `Pe parallel/par_total = 0.011330241103262645`.
- 2026-03-10: the remaining real live solver gap is concentrated in the
  stitched global ExB runtime:
  `n advection/exb = 0.06090172614132968`,
  `Pe advection/exb = 0.0660107963883212`.
  The local literal ExB operator itself continues to match Hermes closely at
  both sheath ends in
  `tests/hermes_literal/test_literal_exb_runtime.py` using
  `tests/fixtures/hermes_mirror_exb_local_rank0_t1.npz` and
  `tests/fixtures/hermes_mirror_exb_local_rank5_t1.npz`.
- 2026-03-10: a naive stitched-global seam fix that imported neighboring
  physical planes directly into every local parallel slab was tested and
  rejected. Raw Hermes dump comparison shows the local guards for `phi`,
  `J`, `g11`, `g23`, and `zShift` are not simple neighbor copies from the
  stitched global interior, so the remaining work is the full processor-local
  guard reconstruction contract, not another local operator rewrite.

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
