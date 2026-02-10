# Validation & benchmarks

This page lists the **numerical** and **physics** checks used in `jaxdrb` to support confident use in research.
The intent is to make it straightforward to answer:

- What was checked?
- Where is it implemented (tests and examples)?
- Which references support the check?

## How to run the checks

- Unit/integration tests: `python -m pytest -q`
- Documentation build: `mkdocs build --strict`
- Selected physics validations (examples): see `examples/` and `docs/examples.md`

## Nonlinear HW2D (Hasegawa–Wakatani) validation

The 2D HW2D model is used as a fast nonlinear testbed for:

- conservative/adaptive time stepping in JAX,
- Poisson solves (polarization closure),
- Poisson brackets / advection kernels,
- end-to-end differentiability through a nonlinear evolution.

Key files:

- Model: `src/jaxdrb/nonlinear/hw2d.py`
- Conservative bracket (Arakawa): `src/jaxdrb/operators/brackets.py`
- Validation tests: `tests/test_hw2d_validation.py`
- Hard-gate invariant test: `tests/test_hw2d_conservative_gate.py`
- Validation example: `examples/08_nonlinear_hw2d/hw2d_camargo1995_validation.py`

### Energy functional and budget closure

For periodic domains with the Boussinesq polarization closure,

$$
\omega = \nabla_\perp^2 \phi,
$$

a standard HW energy functional is:

$$
E = \frac{1}{2}\left\langle n^2 + |\nabla \phi|^2 \right\rangle,
$$

and can be differentiated using the periodic identity

$$
\frac{d}{dt}\left(\frac{1}{2}\langle |\nabla \phi|^2\rangle\right) = -\langle \phi\,\partial_t \omega\rangle,
$$

so that:

$$
\dot E = \left\langle n\,\partial_t n - \phi\,\partial_t \omega \right\rangle.
$$

`jaxdrb` uses this to compute a term-by-term energy budget from the discrete RHS, and compares it to a finite-difference estimate of $dE/dt$.

Reference:

- S. J. Camargo, D. Biskamp, and B. D. Scott, *Resistive drift-wave turbulence*, **Phys. Plasmas** 2(1), 48 (1995). DOI: [`10.1063/1.871116`](https://doi.org/10.1063/1.871116).

Example output:

![HW2D energy budget closure](assets/images/hw2d_camargo1995_budget_closure.png)

### Quadratic invariant conservation (ideal subset)

In the ideal subset (no drive, no coupling, no diffusion),

$$
\partial_t n + [\phi,n] = 0,\qquad \partial_t \omega + [\phi,\omega] = 0,
$$

the continuous system conserves quadratic invariants. For reviewer-proof nonlinear runs, `jaxdrb` uses **Arakawa's conservative Jacobian** (Arakawa, 1966) by default on periodic grids, and includes tests that verify invariant conservation over time.

For additional qualitative turbulence diagnostics, the validation example also plots final-time isotropic spectra:

![HW2D final-time spectra](assets/images/hw2d_camargo1995_spectrum.png)

### Hard benchmark gate for nonlinear invariants

`jaxdrb` includes a dedicated regression gate that evolves the ideal periodic HW2D subset and
checks strict conservation of:

- energy proxy $E$,
- enstrophy proxy $Z$,
- mean density $\langle n\rangle$ (mass proxy),
- mean vorticity $\langle \omega\rangle$ (charge/current-balance proxy),
- mean $\mathbf{E}\times\mathbf{B}$ velocity (net momentum proxy).

Gate test:

- `tests/test_hw2d_conservative_gate.py`

## Method of manufactured solutions (MMS)

MMS tests are included to validate the implementation order and to catch sign/normalization mistakes.

- Example: `examples/08_nonlinear_hw2d/mms_hw2d_convergence.py`
- Tests: see `tests/` for MMS-based checks

Additional operator convergence tests:

- `tests/test_fd_1d_operators.py` (1D derivative operators used in field-line geometry discretizations)

This verification strategy (including MMS) is widely used in SOL turbulence codes. For example:

- F. D. Halpern et al., *The GBS code for tokamak scrape-off layer simulations*, **J. Comput. Phys.** 315 (2016) 388–408.
  DOI: [`10.1016/j.jcp.2016.03.040`](https://doi.org/10.1016/j.jcp.2016.03.040).

## Linear solver checks (matrix-free J·v)

The linear stability solvers are validated by internal consistency checks:

- growth rates from initial-value evolution vs leading eigenvalues from Arnoldi,
- Jacobian–vector products via `jax.linearize` / `jax.jvp` compared against finite-difference sanity checks,
- known qualitative limits in slab/s–alpha geometries.

See:

- Tests: `tests/test_growth_vs_eigs.py`, `tests/test_slab_dispersion.py`
- Solver docs: `docs/solvers/`
- Known limits: `docs/theory/known-limits.md`

## Sheath / MPSE quantitative gates

Open-field-line sheath closures are validated with explicit quantitative consistency checks:

- Loizu-2012 full-set MPSE constraints for constructed states that satisfy the enforced boundary targets:
  - `tests/test_mpse_loizu2012_consistency.py`
- EM current closure consistency at the sheath:
  - MPSE-induced $(\delta v_{\parallel i} - \delta v_{\parallel e})$ is checked against the implemented
    $\delta \psi$ update through Ampere closure.
  - `tests/test_sheath_quantitative_gate.py`
- Hot-ion sheath heat + SEE toggles:
  - enabling heat transmission/SEE must change $(dT_e, dT_i)$ exactly by the closure terms from
    `sheath_energy_losses`.
  - `tests/test_sheath_quantitative_gate.py`

Core implementation:

- `src/jaxdrb/models/sheath.py`
- `src/jaxdrb/models/em_drb.py`
- `src/jaxdrb/models/hot_ion_drb.py`

## Literature transition-boundary gates

In addition to qualitative trend scans, `jaxdrb` enforces quantitative threshold checks where reduced
models are available:

- Halpern (2013) ideal-ballooning threshold gate:
  - finite $\alpha_{\mathrm{crit}}$ at $\hat{s}=0$,
  - monotonic shear stabilization of $\alpha_{\mathrm{crit}}(\hat{s})$.
  - `tests/test_ideal_ballooning.py`
- Mosetto (2012)-style regime transition gate (workflow classifier):
  - analytic/calibrated threshold checks at $\gamma$-ratio $g=1$:
    - $R/L_n|_{\mathrm{RDW/RBM}}\approx 75.2$,
    - $R/L_n|_{\mathrm{InDW/InBM}}\approx 18.8$,
    - $d_{\mathrm{crit}}(\hat{s}=0)\approx 3.55,\ d_{\mathrm{crit}}(\hat{s}=5)\approx 1.12$.
  - explicit 4-regime anchor-point classification checks (InDW/RDW/InBM/RBM).
  - solver-ablation transition check retained as a secondary workflow consistency test.
  - `tests/test_mosetto_calibration.py`
  - `tests/test_mosetto_regime_quantitative_gate.py`

### Arnoldi vs dense Jacobian (tiny problem)

On very small problems it is feasible to explicitly form the dense Jacobian (by applying `J·v` to basis vectors)
and compare its eigenvalues to Arnoldi Ritz values. This validates the end-to-end workflow:

- `jax.linearize` for matrix-free `J·v`,
- Arnoldi implementation (including residual norms).

Test:

- [`tests/test_arnoldi_dense_compare.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_arnoldi_dense_compare.py)

Example:

- [`examples/10_verification/arnoldi_vs_dense_jacobian.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/10_verification/arnoldi_vs_dense_jacobian.py)

![Arnoldi vs dense eigenvalue spectrum](assets/images/arnoldi_vs_dense_spectrum.png)

## Geometry provider checks

Geometry is validated by:

- consistency checks on tabulated coefficients (shapes, positivity where required),
- reference analytic cases (slab / s–alpha),
- ESSOS-driven field-line workflows (VMEC / near-axis / Biot–Savart).

See:

- Geometry docs: `docs/geometry/`
- Examples: `examples/02_geometry/`, `examples/07_essos_geometries/`

## Elliptic (Poisson/polarization) solver verification

Elliptic solves are a central ingredient for nonlinear evolution (polarization closure).

`jaxdrb` includes:

- a spectral inverse Laplacian for periodic domains (exact up to roundoff),
- a matrix-free conjugate-gradient (CG) FD Poisson solver for Dirichlet/Neumann domains.

Verification tests:

- [`tests/test_fd_poisson_cg.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_fd_poisson_cg.py)

Example output (Dirichlet and Neumann cases):

![FD+CG Poisson verification](assets/images/poisson_cg_verification_panel.png)

## Verification problems inspired by GDB (Zhu et al. 2018)

The GDB code paper (Zhu et al., 2018, CPC) describes a suite of simplified verification tests:

- Kelvin–Helmholtz growth (Poisson bracket),
- shear-Alfvén wave propagation (parallel operator),
- resistive ballooning growth (curvature operator),
- convergence of turbulence statistics with resolution.

`jaxdrb` includes a directly comparable **shear-Alfvén dispersion** verification in `jaxdrb.verification`,
and uses separate operator/unit tests for Poisson brackets, parallel derivatives, and curvature coefficients.

Reference:

- B. Zhu et al., *GDB: A global 3D two-fluid model of plasma turbulence and transport in the tokamak edge*,
  **Computer Physics Communications** 232 (2018) 46–58. DOI: [`10.1016/j.cpc.2018.06.002`](https://doi.org/10.1016/j.cpc.2018.06.002).

Test:

- [`tests/test_gdb2018_saw.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_gdb2018_saw.py)

Example:

- [`examples/10_verification/saw_dispersion_gdb2018.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/10_verification/saw_dispersion_gdb2018.py)

Implementation:

- [`src/jaxdrb/verification/gdb2018.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/verification/gdb2018.py)

![SAW phase speed vs Te (GDB 2018 verification)](assets/images/saw_gdb2018_speed_vs_Te.png)

## FCI operator verification (preparation milestone)

Before adopting a full FCI nonlinear SOL model, the *geometry-agnostic* building blocks are validated:

- field-line maps + bilinear interpolation on structured planes,
- centered parallel derivative operator and MMS-style convergence checks.

See:

- Tests: `tests/test_fci_parallel.py`
- Docs: `docs/fci/index.md`
- Example: `examples/09_fci/fci_slab_parallel_derivative_mms.py`

## Neutral-model validation gates

The neutral milestone model is validated with explicit particle-balance and source/sink checks:

- isolated ionization conservation of total particles $\langle n + N \rangle$,
- isolated ionization+recombination conservation of total particles,
- uniform source/sink relaxation to the analytic equilibrium $N^\*=S_0/\nu_{\mathrm{sink}}$.
- exact enforcement of optional charge-exchange-like vorticity drag.

Tests:

- `tests/test_neutrals_exchange.py`

Implementation:

- `src/jaxdrb/nonlinear/neutrals.py`

## Nonlinear closure-toggle gates (field-line DRB)

For nonlinear-preparation workflows in the field-line DRB models, `jaxdrb` includes dedicated
toggle tests for:

- nonlinear non-Boussinesq polarization (`n_0 + Re[n]` denominator),
- state-dependent Braginskii coefficients (`T_{e0}+Re[Te]`, `T_{i0}+Re[Ti]`),
- finite-RHS sanity under combined state-dependent toggles.

Tests:

- `tests/test_nonlinear_fieldline_toggles.py`
- `tests/test_polarization_models.py`
- `tests/test_braginskii_scalings.py`

## DRB nonlinear conservative hard gate (field-line branch)

Beyond the HW2D testbed, `jaxdrb` now includes a **hard conservative gate on the actual cold-ion DRB model**
in a periodic conservative subset (`omega_n=omega_Te=0`, curvature/sinks/diffusion/sheath off, finite `me_hat`).

Tracked diagnostics:

- quadratic DRB energy functional
  $$
  E=\frac{1}{2}\left\langle |n|^2 + k_\perp^2 |\phi|^2 + \hat m_e |v_{\parallel e}|^2 + |v_{\parallel i}|^2
  + \frac{3}{2}\alpha_{Te}|T_e|^2 \right\rangle,
  $$
- mass proxy $\langle n\rangle$,
- charge proxy $\langle \Omega\rangle$,
- mean current $\langle j_\parallel \rangle$ with $j_\parallel=v_{\parallel i}-v_{\parallel e}$,
- momentum proxy $\langle v_{\parallel i}+\hat m_e v_{\parallel e}\rangle$.

Implementation:

- Functional/diagnostics: `src/jaxdrb/models/invariants.py`
- Hard-gate tests:
  - `tests/test_drb_nonlinear_conservative_gate.py`
  - `tests/test_drb_operator_rates.py`
- Reproducible examples:
  - `examples/10_verification/drb_cold_ion_conservative_gate.py`
  - `examples/10_verification/drb_cold_ion_operator_gate.py`
- CI physics benchmark gate:
  - `benchmarks/check_drb_conservative_gate.py`
  - `.github/workflows/ci.yml` (Ubuntu + Python 3.12)

### Operator-level residual gate

In addition to finite-time drift checks, `jaxdrb` now enforces a strict **operator-level** gate:

- compute `dy = rhs_nonlinear(y)` on random states and multiple `k_y`,
- evaluate instantaneous rates
  $$
  \dot{E},\ \frac{d}{dt}\langle n\rangle,\ \frac{d}{dt}\langle\Omega\rangle,\ \frac{d}{dt}\langle j_\parallel\rangle,\ \frac{d}{dt}\langle v_{\parallel i}+\hat m_e v_{\parallel e}\rangle,
  $$
- fail if rates exceed conservative roundoff-scale thresholds.

For Boussinesq periodic runs, the energy-rate diagnostic is evaluated using the exact discrete chain rule:
$$
\dot E = \Re\left\langle
n^*\,\dot n
-\phi^*\,\dot\Omega
+\hat m_e v_{\parallel e}^*\,\dot v_{\parallel e}
+v_{\parallel i}^*\,\dot v_{\parallel i}
+\frac{3}{2}\alpha_{Te}\,T_e^*\,\dot T_e
\right\rangle,
$$
which is the direct quadratic-form derivative of the implemented energy functional.

Example output:

![Cold-ion DRB conservative gate](assets/images/drb_cold_ion_conservative_gate.png)

![Cold-ion DRB operator gate](assets/images/drb_cold_ion_operator_gate.png)

## Performance regression gates

CI enforces a conservative core-kernel throughput gate on Ubuntu/Python 3.12:

- nonlinear HW2D RK4 stepping throughput (steps/s),
- linear matrix-free matvec throughput (matvec/s).

Gate script:

- `benchmarks/check_core_kernels.py`

CI workflow:

- `.github/workflows/ci.yml`

## Differentiability checks

One goal of `jaxdrb` is to keep verification and solver workflows **end-to-end differentiable** where feasible.
This is validated with small gradient checks through time integration and operator pipelines.

See:

- `tests/test_hw2d_validation.py` (gradient through nonlinear time stepping)
- `tests/test_fci_parallel.py` (gradient through FCI mapping and ∂_|| operator)
