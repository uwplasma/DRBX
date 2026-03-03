# Parity-FV Rewrite Track

This page documents the new `parity_fv` engine, which is the strict
Hermes-parity rewrite path.

## Scope (current)

Implemented modules:
- `src/jaxdrb/parity_fv/params.py`
- `src/jaxdrb/parity_fv/state.py`
- `src/jaxdrb/parity_fv/geometry.py`
- `src/jaxdrb/parity_fv/flux_reconstruct.py`
- `src/jaxdrb/parity_fv/flux_parallel.py`
- `src/jaxdrb/parity_fv/ops.py`
- `src/jaxdrb/parity_fv/terms_density.py`
- `src/jaxdrb/parity_fv/terms_pressure.py`
- `src/jaxdrb/parity_fv/terms_sheath.py`
- `src/jaxdrb/parity_fv/terms_vorticity.py`
- `src/jaxdrb/parity_fv/poisson_vorticity.py`
- `src/jaxdrb/parity_fv/rhs.py`
- `src/jaxdrb/parity_fv/system.py`

## Geometry ingestion (new)

`parity_fv` can now ingest metric/coefficient files through
`[geometry].coeff_path`. For the current rewrite stage, the loader supports
Hermes-style axisymmetric coefficient bundles and broadcasts them onto the
`(z, x, y)` parity layout:

- `J` -> `jacobian`
- `curv_x` or `bxcv` -> `bxcv`
- `gxx`, `gxy`, `gyy`
- `dpar_factor`

Supported source shapes are:

- scalar
- `(nz,)`, `(nx,)`, `(ny,)`
- `(nz, nx)`, `(nz, ny)`, `(nx, ny)`
- `(nz, nx, ny)`

If both config and coefficient file specify `nx`, `ny`, or `nz`, mismatches are
treated as an error. This is deliberate: the parity path should fail early on
geometry inconsistencies rather than silently reshape metrics.

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

### Density / Pressure / Vorticity term gates (new)

The parity engine now assembles explicit per-channel terms:

- `parallel`:
  - density: \(-\nabla_\parallel (n v_{\parallel e})\)
  - pressure transport in conservative form with configurable coefficients
    (`parallel_pressure_flux_coeff`, `parallel_pressure_work_coeff`)
  - vorticity parallel-current proxy:
    \[
    \partial_t \omega \sim -c_{\omega,\parallel}\,\partial_\parallel (v_{\parallel i}-v_{\parallel e})
    \]
- `curvature`:
  - vorticity drive from pressure gradient and curvature coefficient:
    \[
    \partial_t \omega \sim -c_{\kappa}\,b_{xcv}\,\partial_x p_e
    \]
- `volume_source`:
  - scalar density source (`source_n0`)

These are intentionally minimal but structurally separated to enable strict
term-by-term parity auditing before adding additional closures.

### Poisson / vorticity path (new)

`parity_fv` now supports:

- `parity_poisson_solver = "spectral_xy"` (default):
  - solves \(\nabla_\perp^2 \phi = \omega\) via FFT in x/y
  - applies gauge fixing at \(k_x=k_y=0\)
  - uses the same spectral operator for \(\omega(\phi)\) in reverse mapping
- `parity_poisson_solver = "identity"`:
  - debug/calibration path: \(\phi = s_\phi \omega\)

This makes the vorticity/phi mapping explicit and testable in CI while keeping
the parity path small.

### Sheath boundary component parity (new)

`parity_fv` now has explicit sheath boundary channels, active only when both:
- `geometry.open_field_line = true`
- sheath is enabled (`terms.sheath_on` or sheath closure toggle)

Implemented component channels:
- particle sink: boundary \(\Gamma = n c_s\)
- momentum relaxation: \(v_{\parallel e}, v_{\parallel i}\) to Bohm-like targets
- electron energy sink: \(q_e = \gamma_e n T_e c_s\)

These terms are emitted in the term map as `sheath`, so term-by-term audits can
compare them directly.

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
- `tests/test_parity_fv_scaffold.py`
- `tests/test_parity_fv_parallel_flux.py`
- `tests/test_parity_fv_poisson_vorticity_guards.py`
- `tests/test_parity_fv_engine.py`
- `tests/test_parity_fv_term_gates.py`
- `tests/test_parity_fv_poisson_solver.py`
- `tests/test_parity_fv_sheath.py`
- `tests/test_parity_fv_one_step_audit_gate.py`
- `tests/test_parity_fv_short_window_gate.py`

These tests verify reconstruction, FV boundary-flux balance, and guard-cell
rules mapped directly from Hermes vorticity solver behavior. The short-window
gate extends this to a deterministic `t<=0.1` integration window, checking:

- fluctuation RMS traces for `n`, `Te`, `omega`, `phi`
- finite-run gate (non-finite / growth / peak rejection)
- probe PSD in time
- final-plane `k_y` PSD
- final fluctuation slices (`n`, `phi`) at fixed `z`

## Next steps

- Pass staged parity windows (`t<=0.1`, `t<=0.5`) with finite-run gating.
- Promote parity diagnostics to benchmark panel and long-window runs once staged gates pass.

## Engine selection

Use top-level TOML key:

```toml
engine = "parity_fv"
```

Alias values are accepted by loader and normalized to `parity_fv`:
- `fv_parity`
- `parity-fv`

CLI metadata includes engine listing:

```bash
jax_drb --list-engines
```

## Audit compatibility

`tools/audit_term_parity.py` and `tools/trace_first_mismatch.py` now detect
`engine = "parity_fv"` and use parity-engine term assembly directly instead
of the legacy term-context path.
