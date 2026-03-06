# jax_drb Agent Handoff + Master Plan

Last update: 2026-03-05  
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
  - easy to maintain (single modular equation core),
  - fully validated and tested,
  - extensively documented (equations, numerics, implementation choices, references),
  - end-to-end differentiable,
  - high-performance and memory-efficient on CPU and GPU.

Immediate strategy:
- First, match Hermes behavior as tightly as possible for electrostatic/cold-ion/Boussinesq open-field SOL with sheath and open/closed field lines.
- Match not only outputs, but the practical stack: equations, normalization, geometry ingestion, numerics, diagnostics, and benchmark presentation.
- Then extend jax_drb beyond Hermes:
  - conservative DRB formulation (de Lucca-style conserving system),
  - richer geometry options (s-alpha, Miller, tokamak separatrix/divertor, stellarator, FCI),
  - broader physics toggles (EM, hot ions, non-Boussinesq, neutrals, advanced sheath closures).

Non-negotiables:
- No proxy/testbed equations in production paths.
- One core equation system with toggles (physics + geometry + BC policy), not fragmented branch codes.
- Keep differentiability in all production solver paths.
- Keep performance and memory usage as first-class acceptance criteria.
- Validation-first: unit + physics + regression + benchmark gates.

Current architecture:
- New finite-volume rewrite path: src/jaxdrb/drb_fv (retained as rewrite/reference path for parity work and future promotion).
- Legacy implementation retained in src/jaxdrb/legacy_v1 for traceability.
- CLI supports selecting engine="drb_fv" or unified engine.
- Tooling exists for alignment audits, benchmark bundles, scans, and panel generation.
- Current strict Hermes-state baseline configs run the unified engine unless
  `engine = "drb_fv"` is explicitly set; unified is the present authoritative
  parity path for Milestone A until a deliberate promotion changes that.

Current tactical objective:
- Finish strict short-window parity in staged windows:
  1) term-level RHS consistency at very early times (few steps),
  2) normalization/time-unit consistency,
  3) Poisson/vorticity equivalence and sheath boundary consistency,
  4) short-window (t<=0.1) fluctuation RMS/PSD agreement,
  5) only then promote to longer windows and turbulence benchmarking.

What to do every cycle:
1) run the smallest targeted audit to isolate the first mismatch,
2) fix the structural cause (equation, discretization, normalization, geometry, BC semantics),
3) re-run strict gate and record delta in docs/benchmarks/open_field_alignment.md,
4) run full CI locally (ruff/black/pytest),
5) commit and push only green changes,
6) update /Users/rogerio/local/jax_drb/plan.md checkboxes and next actions.

Do not:
- rely on parameter fiddling before structural mismatch causes are understood,
- promote long runs while short-window gates fail.

Paths and context:
- jax_drb repo: /Users/rogerio/local/jax_drb
- Hermes-3 repo: /Users/rogerio/local/hermes-3
- Hermes-2 repo: /Users/rogerio/local/hermes-2
- Literature folder: /Users/rogerio/local/tests/drb_literature
- GBS and related refs: /Users/rogerio/local/tests/GBS_ISTTOK

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

1. solves DRB equations in a single modular core with toggles,
2. reproduces Hermes-class workflows for the baseline model in JAX,
3. then extends to the conservative DRB system with strict energy diagnostics,
4. supports multiple geometry paradigms (field-aligned, axisymmetric, FCI),
5. remains differentiable, fast, memory-efficient, and reproducible.

---

## 2) Project Scope and Physics Roadmap

### Stage 1 (current): Hermes-equivalent baseline
- Electrostatic
- Cold-ion
- Boussinesq
- Open-field + sheath + open/closed masks
- Tokamak metric ingestion from mesh coefficients (`bxcv`, Jacobian/metrics, masks)
- FV parallel transport with slope-limited/Lax-style fluxes

### Stage 2: Full DRB feature matrix in one core
- EM toggles
- Hot ions
- Non-Boussinesq polarization
- Neutrals
- Sheath closure variants
- Region-policy BCs for core/SOL/divertor legs

### Stage 3: Conservative DRB system
- Implement conserving DRB equations (`conserving_drb.pdf` + de Lucca line)
- Add energy-balance diagnostics and conservative residual gates
- Verify with advection-only and reduced-system invariants + full budget closure

---

## 3) Where Files Live (Operational Map)

### Core repos
- `jax_drb`: `/Users/rogerio/local/jax_drb`
- `Hermes-3`: `/Users/rogerio/local/hermes-3`
- `Hermes-2`: `/Users/rogerio/local/hermes-2`

### Literature and external references
- DRB literature: `/Users/rogerio/local/tests/drb_literature`
- GBS references/code dumps: `/Users/rogerio/local/tests/GBS_ISTTOK`

### jax_drb key directories
- Source: `/Users/rogerio/local/jax_drb/src/jaxdrb`
- Rewrite path: `/Users/rogerio/local/jax_drb/src/jaxdrb/drb_fv`
- Legacy path: `/Users/rogerio/local/jax_drb/src/jaxdrb/legacy_v1`
- Docs: `/Users/rogerio/local/jax_drb/docs`
- Examples: `/Users/rogerio/local/jax_drb/examples`
- Tests: `/Users/rogerio/local/jax_drb/tests`
- Tooling scripts: `/Users/rogerio/local/jax_drb/tools`
- CI workflow: `/Users/rogerio/local/jax_drb/.github/workflows/ci.yml`
- Runtime artifacts (local): `/Users/rogerio/local/jax_drb/runs`

---

## 4) Current Infrastructure (Already Built)

### Engines and driver
- CLI entrypoint `jaxdrb`.
- Engine selection via TOML:
  - `engine = "unified"`
  - `engine = "drb_fv"` (rewrite/reference track; not the default strict gate path)

### Time stepping
- Fixed-step: `rk4_scan`, `rk4_imex`, `rk4_imex_strang`
- Diffrax path: `dopri8`, `dopri5`, `tsit5`, `euler`
- Stiff-support hooks and IMEX/split infrastructure are present and actively tuned.

### Performance tooling
- JIT scan loops with warm starts and solver diagnostics.
- Profiling scripts and named-scope traces:
  - `tools/profile_jaxdrb.py`
  - `docs/profiling.md`

### Benchmark/alignment tooling
- Term audit:
  - `tools/audit_term_alignment.py`
  - `tools/trace_first_mismatch.py`
  - `tools/compare_term_arrays.py`
- Bundle and panel:
  - `tools/build_benchmark_bundle.py`
  - `tools/compare_benchmark_bundles.py`
  - `tools/plot_benchmark_panel.py`
  - `tools/run_tokamak_hermes_benchmark.py`

### Geometry conversion and checks
- BOUT/Hermes mesh converters:
  - `tools/convert_bout_grid_axisymmetric.py`
  - `tools/convert_bout_metrics_axisymmetric.py`
  - `tools/convert_hermes_dump_axisymmetric.py`
- Geometry consistency scripts:
  - `tools/compare_geometry_coeffs.py`
  - `tools/compare_geometry_metrics.py`

---

## 5) Work Done So Far (Condensed History)

### Earlier refactor phase (before rewrite)
- Unified/core moves for 2D/line/FCI and diagnostics centralization.
- Consolidation of wrappers into core calls.
- Added many benchmark and gate scripts for DRB2D/FCI.

### Clean rewrite direction
- Introduced `drb_fv` engine for strict structural alignment.
- Moved older broad core implementation to `legacy_v1` for traceability.
- Added strict one-step and short-window gates for Hermes-coupled alignment.
- Added compact reference fixtures and deterministic comparison harness.
- Added continuous targeted mismatch-audit loop (term-by-term).

### Current audit status
- Several structural mismatches have already been reduced.
- Dominant early-time mismatch classes are now tracked by strict term audits.
- The authoritative strict `t<=0.1` Hermes-state audits currently exercise the
  unified engine path; `drb_fv` remains available for focused rewrite checks but
  is not the promoted baseline gate yet.
- Remaining gap still requires structural closure (especially Poisson/vorticity/time-scale normalization consistency and selected sheath/parallel details).

---

## 6) Model and Algorithm Parity Requirements (Hermes-baseline)

The baseline path must structurally match Hermes logic where applicable:

1. **Parallel transport**  
   Finite-volume + second-order limited reconstruction + Lax/consistent boundary flux semantics.

2. **Boundary conditions / sheath**  
   Bohm/Chodura-style constraints and energy flux terms as component-level boundary physics; region-aware BC policy.

3. **Curvature and metric handling**  
   `bxcv` and metric-normalized operators consistent with mesh inputs.

4. **Poisson/vorticity path**  
   Matching guard-cell/boundary treatment and normalization factors.

5. **Normalization/time units**  
   Explicit mapping of `Nnorm`, `Tnorm`, `Bnorm`, `rho_s0`, `Omega_ci`, and timestep units.

---

## 7) Beyond Hermes: Conservative DRB Extension

Target after baseline closure:

- Add conservative DRB equation pathway (de Lucca/conserving DRB references).
- Preserve baseline path for compatibility and benchmarking.
- Add strict energy diagnostics:
  - total energy
  - relative drift
  - source/sink budget closure
  - per-term contribution checks
- Add CI gates that fail if conservative invariants regress beyond tolerance.

---

## 8) Geometry Roadmap (Short -> Long)

### Short-term (now)
- Tokamak field-aligned with open/closed masks and sheath boundaries.
- Coefficient file ingestion from Hermes/BOUT meshes.

### Medium-term
- Analytic axisymmetric adapters:
  - `s-alpha`
  - Miller
  - separatrix/X-point analytic options

### Long-term
- FCI for complex topologies (divertors, stellarator islands, non-axisymmetry):
  - field-line map approach per Hariri et al.
  - implementation style aligned with GRILLIX experience.

---

## 9) Time Integrators and Solver Policy

### Required options
- Keep Diffrax options available (`Dopri8`, adaptive) for flexible workflows.
- Keep custom steppers (explicit, semi-implicit, IMEX, implicit/split) for controlled stiff testing.
- Default baseline path should mimic Hermes numerics for parity runs.

### Performance constraints
- JIT full loops (`lax.scan`) with static shapes.
- Persistent compile cache.
- Minimized host-device transfer.
- Reused preconditioners and warm starts where valid.

### Differentiability constraints
- End-to-end differentiable state evolution.
- Differentiable linear solves and no hidden non-diff branches in main paths.

---

## 10) Testing and Validation Strategy (CI/CD)

### Unit tests
- Operators, reconstructions, BC application, Poisson invariants/SPD behavior, sheath sign conventions.

### Physics tests
- Linear proxies (DW/ballooning/Mosetto/Halpern classes as applicable).
- Open-field sheath/target flux sanity.
- Conservation gates for reduced and conservative systems.

### Regression tests
- One-step RHS/term audit gate (strict early mismatch detection).
- Short-window (`t<=0.1`) fluctuation RMS/PSD gate with finite-run rejection.
- Performance regression (time/step + memory profile threshold).

### CI workflow
- Current workflow: `/Users/rogerio/local/jax_drb/.github/workflows/ci.yml`
  - `ruff check src tests`
  - `black --check src tests`
  - focused one-step gate
  - full `pytest`
- Extend with:
  - term-specific mismatch caps
  - conservative-energy drift cap
  - optional/nightly longer turbulence windows

---

## 11) Documentation System and Update Rules

### Primary docs files
- `/Users/rogerio/local/jax_drb/README.md` (concise entrypoint only)
- `/Users/rogerio/local/jax_drb/docs/run.md`
- `/Users/rogerio/local/jax_drb/docs/inputs_outputs.md`
- `/Users/rogerio/local/jax_drb/docs/options.md`
- `/Users/rogerio/local/jax_drb/docs/normalization.md`
- `/Users/rogerio/local/jax_drb/docs/geometry_models.md`
- `/Users/rogerio/local/jax_drb/docs/geometry_compare.md`
- `/Users/rogerio/local/jax_drb/docs/diagnostics.md`
- `/Users/rogerio/local/jax_drb/docs/profiling.md`
- `/Users/rogerio/local/jax_drb/docs/validation.md`
- `/Users/rogerio/local/jax_drb/docs/benchmarks/open_field_alignment.md`
- `/Users/rogerio/local/jax_drb/docs/drb_fv.md`
- `/Users/rogerio/local/jax_drb/docs/figures.md`

### Doc policy
- README stays concise.
- Full equations/algorithms/validation live in docs pages.
- Every benchmark figure/movie must be reproducible from a script in `tools/` with explicit command lines.
- Every new model toggle requires:
  - equation statement,
  - discretization description,
  - normalization statement,
  - validation/gate link.

---

## 12) Competitor / Ecosystem Landscape and Market Pull (Online + Literature)

As of 2026-03-05, external landscape signals remain strong:

- Fusion programs are explicitly accelerating toward pilot plants, with high demand for predictive edge/SOL simulation and robust validation workflows.
- Major public strategy documents call out simulation/HPC/AI acceleration and industry-aligned R&D.
- Code ecosystem is actively expanding and publishing in CPC/JPP/PoP with stronger verification and CI practices.

### Active code ecosystem to track
- Hermes (BOUT++ multiphysics drift-fluid components)
- GBS (global two-fluid SOL turbulence)
- GRILLIX (FCI fluid edge/SOL; tokamak + stellarator direction)
- GDB (open-field fluid turbulence; used in GDB/Gkeyll comparisons)
- GENE-X / GENE-3D (full-f GK edge/SOL and stellarator-capable workflows)
- Gkeyll (continuum GK/DG with open-field-line sheath studies)
- Also relevant: STORM, SOLEDGE3X, TOKAM3X, FELTOR

### Why this matters for jax_drb
- There is clear pull for:
  1) reproducible validation,
  2) robust stiff solvers,
  3) realistic open/closed field topology handling,
  4) fast turnaround for scenario exploration,
  5) modern differentiable/HPC-ready implementations.

---

## 13) Short / Medium / Long Horizon Plan

### Short-term (now -> parity closure)
- [x] Create and stabilize strict finite-volume rewrite track (`drb_fv`).
- [x] Keep legacy implementation under `legacy_v1`.
- [x] Build one-step audit and short-window regression infrastructure.
- [ ] Close remaining structural mismatches in `t<=0.1` strict window:
  - [ ] Poisson/vorticity equivalence and normalization
  - [ ] Sheath-energy and boundary flux term equivalence
  - [ ] Parallel momentum/pressure flux edge semantics
- [ ] Reach fluctuation RMS/PSD agreement band (10-20%) at `t<=0.1`.

### Medium-term (post-short parity)
- [ ] Promote validated baseline to `t<=0.5` and `t<=1.0`.
- [ ] Generate canonical benchmark panels and movies for tokamak SOL with open/closed regions + sheath.
- [ ] Add robust diagnostics suite aligned with literature:
  - radial profiles
  - PSD(k,f)
  - PDFs/skewness/flatness
  - fluxes and sheath target channels
- [ ] Integrate conservative-energy diagnostics in baseline runs.

### Long-term (beyond Hermes baseline)
- [ ] Implement conservative DRB equation set as first-class engine mode.
- [ ] Expand geometry adapters:
  - `s-alpha`
  - Miller
  - separatrix/X-point analytic
  - stellarator + FCI map path
- [ ] Add broader physics matrix (EM/hot/non-Bouss/neutrals) with unified toggles.
- [ ] Add publication-grade validation book in docs, fully reproducible.

---

## 14) Concrete Next Iteration Loop (Do This Repeatedly)

1. Run smallest strict audit window (`nsteps <= 3`) for a single mismatch class.
2. Dump both code paths for only the failing term group.
3. Classify mismatch origin:
   - equation form,
   - discretization/stencil,
   - normalization/time scale,
   - geometry/metric mapping,
   - boundary/sheath semantics.
4. Apply one structural fix at a time.
5. Re-run:
   - strict term gate
   - short-window gate
   - full CI.
6. If green, update docs + this plan + commit/push.
7. Promote only passing configs to longer windows.

---

## 15) Minimal Commands Reference

### Install and checks
```bash
cd /Users/rogerio/local/jax_drb
python -m pip install -e ".[dev]"
ruff check src tests
black --check src tests
python -m pytest -q
```

### Run jax_drb
```bash
jaxdrb /Users/rogerio/local/jax_drb/examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml --run --output /tmp/jax_short.npz
```

### Strict audit
```bash
python /Users/rogerio/local/jax_drb/tools/audit_term_alignment.py \
  --jax-config /Users/rogerio/local/jax_drb/examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml \
  --hermes-data-dir /Users/rogerio/local/jax_drb/runs/hermes_open_field_terms_t01_vortterms/data \
  --out-dir /Users/rogerio/local/jax_drb/runs/audit_latest \
  --nsteps 3 --match-hermes-dt --strict-axis --use-hermes-state --use-hermes-phi-in-terms --start-index 1
```

### Canonical benchmark script
```bash
python /Users/rogerio/local/jax_drb/tools/run_tokamak_hermes_benchmark.py \
  --jax-config examples/open_field_line/input_tokamak_bxcv_benchmark_es_cold.toml \
  --hermes-data runs/hermes_open_field_short/data \
  --out-dir runs/tokamak_benchmark_latest \
  --fig-dir docs/figures
```

---

## 16) Progress Tracker (Keep This Updated)

### Milestone A: strict early equivalence (`t<=0.1`)
- [x] One-step audit infrastructure exists.
- [x] Compact reference fixture exists.
- [x] Finite-run gate exists.
- [ ] Remaining dominant term mismatches reduced below threshold.
- [ ] Fluctuation RMS/PSD within target band.
- [ ] Gate promoted to required CI check.

2026-03-05 note:
- Narrowed the `Div_par(jpar)` sheath-face metric in
  `src/jaxdrb/core/terms/parallel.py` so the `gpar`/`wave=None` boundary
  coefficient uses the boundary-cell metric instead of the first interior face.
- Strict Hermes-state audit delta (`runs/audit_takeover_20260305` ->
  `runs/audit_takeover_after_metric_fix_v2`): `omega parallel/jpar`
  weighted-rel improved from `0.01227` to `0.001995` at `t=0.01`; the
  fail-fast leader moved to `omega advection/exb` at `0.00703`.
- Tests/docs touched: `tests/test_parallel_sheath_targets.py`,
  `docs/benchmarks/open_field_alignment.md`.
- Commit: `e0a0502` (`src/jaxdrb/core/terms/parallel.py`,
  `tests/test_parallel_sheath_targets.py`,
  `docs/benchmarks/open_field_alignment.md`).

2026-03-05 follow-on note:
- Matched the Hermes dense-run setting `exb_advection_simplified = false` in
  the unified alignment path by adding the full vorticity ExB branch in
  `src/jaxdrb/core/terms/advection.py` and wiring the strict alignment configs
  to the same numerics toggle.
- The structural closure was the missing polarization-current form, not radial
  ghost semantics: the dominant fix was the metric `Delp2(phi)` branch with a
  zero-Dirichlet radial auxiliary BC under `poisson_invert_set`.
- Strict Hermes-state audit delta
  (`runs/audit_takeover_after_metric_fix_v2` ->
  `runs/audit_takeover_full_vort_exb_fix`): `omega advection/exb`
  weighted-rel improved from `0.00703` to `0.000701` at `t=0.01`; the
  fail-fast leader moved to `Pe parallel/par_total` at `0.00622`.
- Tests/docs touched: `tests/test_vorticity_alignment_switches.py`,
  `docs/benchmarks/open_field_alignment.md`.
- Commit: `c654950` (`src/jaxdrb/core/terms/advection.py`,
  `src/jaxdrb/core/params.py`,
  `src/jaxdrb/legacy_v1/core/params.py`,
  `examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml`,
  `examples/open_field_line/input_tokamak_bxcv_alignment_strict_early_tuned.toml`,
  `examples/open_field_line/input_tokamak_bxcv_benchmark_hermes_strict.toml`,
  `tests/test_vorticity_alignment_switches.py`,
  `docs/benchmarks/open_field_alignment.md`).

2026-03-05 planning note:
- Clarified that the authoritative strict Hermes-state parity path is the
  unified engine, not `drb_fv`, unless a config explicitly sets
  `engine = "drb_fv"`.
- The old fail-fast target (`Pe parallel/par_total`) was isolated to a
  limiter-stack mismatch between Hermes `FV::Div_par_mod` and `Div_par(jpar)`,
  not to pressure boundary-flux coefficients.

2026-03-05 limiter-split note:
- Added a separate `parallel_current_limiter` so open-field `Div_par(jpar)` no
  longer shares the finite-wave FV limiter used by density/pressure channels.
- Updated the strict Hermes configs to use `parallel_limiter = "mc"` together
  with `parallel_current_limiter = "none"`, matching Hermes’ build-time
  `MC` limiter for `FV::Div_par_mod` while preserving the already-aligned
  current-divergence path.
- Strict Hermes-state audit delta
  (`runs/audit_takeover_full_vort_exb_fix` ->
  `runs/audit_pe_parallel_split_limiter_3step`): `Pe parallel/par_total`
  weighted-rel improved from `0.00622` to `0.00258` at `t=0.01`, while
  `omega parallel/jpar` stayed at `0.001995`.
- The fail-fast leader is now `Pe advection/exb` at `0.00476`, followed by
  `n parallel/par` at `0.00298`, then `Pe parallel/par_total` at `0.00258`.
- Tests/docs touched: `tests/test_open_field_strict_config.py`,
  `docs/benchmarks/open_field_alignment.md`.
- Commit: `ba8e37c` (`src/jaxdrb/core/params.py`,
  `src/jaxdrb/core/terms/parallel.py`,
  `src/jaxdrb/legacy_v1/core/params.py`,
  `src/jaxdrb/legacy_v1/core/terms/parallel.py`,
  `examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml`,
  `examples/open_field_line/input_tokamak_bxcv_alignment_strict_early_tuned.toml`,
  `examples/open_field_line/input_tokamak_bxcv_benchmark_hermes_strict.toml`,
  `tests/test_open_field_strict_config.py`,
  `docs/benchmarks/open_field_alignment.md`).

2026-03-05 shift-region note:
- Tightened the unified shifted-transform usage toward Hermes
  `toFieldAligned(..., "RGN_NOX")` semantics by leaving non-periodic x-boundary
  cells unshifted in the open-field parallel FV path and the poloidal ExB
  Y-flux branch.
- Strict Hermes-state audit delta
  (`runs/audit_pe_parallel_split_limiter_3step` ->
  `runs/audit_shift_nox_fix_3step`): `n parallel/par` weighted-rel improved
  slightly from `0.00298` to `0.00296` at `t=0.01`, while
  `Pe advection/exb` stayed at `0.00476`.
- Result: the next priority is still `Pe advection/exb`, specifically the
  radial-boundary semantics of the metric-coupled poloidal ExB branch; the
  likely next term after that is still `n parallel/par`.
- Tests/docs touched: `tests/test_exb_poloidal_flows.py`,
  `docs/benchmarks/open_field_alignment.md`.
- Commit: `f73a610` (`src/jaxdrb/core/geometry_field_aligned.py`,
  `src/jaxdrb/core/terms/parallel.py`,
  `tests/test_exb_poloidal_flows.py`,
  `docs/benchmarks/open_field_alignment.md`,
  `plan.md`).

2026-03-05 poloidal X-face note:
- Tightened the unified poloidal ExB X-face boundary velocity to use
  Hermes-style ghost/cell metric averaging at nonperiodic radial faces, while
  leaving the Y-face boundary branch unchanged.
- Strict Hermes-state audit delta
  (`runs/audit_shift_nox_fix_3step` -> `runs/audit_pe_exb_xface_avg_3step`):
  `Pe advection/exb` improved across the 3-step strict window
  (`0.00476 -> 0.00360` at `t=0.01`,
  `0.00833 -> 0.00714` at `t=0.02`,
  `0.01381 -> 0.01261` at `t=0.03`).
- Side effect: `n advection/exb` increased at the first strict step
  (`0.00140 -> 0.00304`) but remained below the main fail-fast terms; the
  remaining parallel leader `n parallel/par` stayed at `0.00296`.
- Rejected probe: a follow-up Hermes-style boundary-face metric factor for the
  finite-wave parallel sheath flux regressed `n parallel/par` to `0.03019` and
  `Pe parallel/par_total` to `0.02395` at `t=0.01`
  (`runs/audit_xface_and_parbnd_3step`), so that path should not be retried
  without new evidence.
- Result: the next priority remains the open-field density/pressure
  sheath-target state construction in the parallel channel, with `n parallel/par`
  still the next distinct structural mismatch after the improved `Pe exb` path.
- Tests/docs touched: `tests/test_exb_poloidal_flows.py`,
  `docs/benchmarks/open_field_alignment.md`.

### Milestone B: short benchmark parity (`t<=0.5`)
- [ ] Stable matched runs generated for Hermes and jax_drb.
- [ ] Panel diagnostics agree in accepted tolerance.
- [ ] Runtime/memory comparison table generated.

### Milestone C: turbulence benchmark
- [ ] Long-window open-field turbulence run stable.
- [ ] Publication-grade figures/movies and diagnostics.
- [ ] Docs benchmark page finalized.

### Milestone D: conservative DRB
- [ ] Conservative equations implemented and documented.
- [ ] Energy conservation diagnostics and tests in CI.
- [ ] Extended physics/geometry matrix validated.

---

## 17) Key Risks and Mitigations

1. **Slow convergence of parity work**  
   Mitigation: always isolate one mismatch class with tiny windows and strict dumps.

2. **Numerical mismatch hidden by parameter tuning**  
   Mitigation: structural fix-first policy; no broad scans until root cause is known.

3. **Performance regressions during alignment edits**  
   Mitigation: maintain small performance gate and profile snapshots in CI/nightly.

4. **Documentation drift**  
   Mitigation: code change requires same-PR doc and test updates.

---

## 18) External References (Used for Planning Context)

### Hermes / BOUT++
- Hermes docs (solver numerics): <https://hermes3.readthedocs.io/en/stable/solver_numerics.html>  
- Hermes docs (boundary conditions): <https://hermes3.readthedocs.io/en/latest/boundary_conditions.html>  
- Hermes docs (equations): <https://hermes3.readthedocs.io/en/latest/equations.html>  
- Hermes repository: <https://github.com/boutproject/hermes-3>  
- Hermes CPC paper page: <https://www.sciencedirect.com/science/article/pii/S0010465523003363>

### FCI / GRILLIX
- Hariri et al., field-line map approach: <https://www.sciencedirect.com/science/article/abs/pii/S0010465515003641>  
- GRILLIX unified tokamak/stellarator CPC 2026 page: <https://www.sciencedirect.com/science/article/pii/S0010465525003765>  
- GRILLIX code page: <https://physik.uni-greifswald.de/en/ag-manz/translate-to-english-forschung/codes/grillix/>

### GBS
- GBS code paper (JCP 2016): <https://www.sciencedirect.com/science/article/pii/S0021999116001923>  
- GBS plasma+kinetic neutrals (JCP 2022 / arXiv 2112.03573): <https://arxiv.org/abs/2112.03573>

### GENE-X / GENE-3D / Gkeyll / GDB
- GENE-X code paper (CPC 2021): <https://www.sciencedirect.com/science/article/pii/S0010465521000989>  
- GENE-X spectral acceleration (CPC 2025): <https://www.osti.gov/pages/servlets/purl/2997736>  
- GENE-3D global stellarator code (JCP 2020): <https://www.sciencedirect.com/science/article/pii/S002199912030468X>  
- Gkeyll open-field-line turbulence (JPP 2017): <https://doi.org/10.1017/S002237781700037X>  
- Fluid vs GK open-field turbulence (GDB + Gkeyll, PoP 2020): <https://www.osti.gov/pages/biblio/1647927>

### Market pull / program demand
- DOE Fusion Energy Strategy 2024: <https://www.energy.gov/sites/default/files/2024-06/fusion-energy-strategy-2024.pdf>  
- DOE strategy executive summary: <https://www.energy.gov/doe-fusion-energy-strategy-2024-executive-summary>  
- DOE Fusion S&T Roadmap announcement (2025): <https://www.energy.gov/articles/energy-department-announces-fusion-science-and-technology-roadmap-accelerate-commercial>  
- GAO commercialization planning review (2025): <https://www.gao.gov/products/gao-25-107037>  
- EUROfusion roadmap page: <https://www.eurofusion.org/eurofusion/roadmap/>

---

## 19) Plan Maintenance Rule

When you complete a step:
1. mark checkbox status in this file,
2. add a dated note under the relevant milestone,
3. link the commit hash and affected files,
4. update associated docs/tests references.

This `plan.md` is the single source of truth for handoff continuity.
