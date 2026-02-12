# Examples

Runnable scripts live in:

`examples/`

and are organized by **topic** (geometry, sheath closures, literature workflows, nonlinear HW2D, …).
Each script prints progress and writes `.npz` + publication-friendly `.png` figures.

> Tip: most examples run faster if you reduce `nl` and use fewer `ky`/`kx` points.

## Linear basics

Slab ky scan:

```bash
python examples/01_linear_basics/slab_ky_scan.py
```

Circular tokamak ky scan:

```bash
python examples/01_linear_basics/circular_tokamak_ky_scan.py
```

Cyclone-like s–α ky scan:

```bash
python examples/01_linear_basics/salpha_cyclone_ky_scan.py
```

2D scan over `(kx, ky)` in s–α:

```bash
python examples/01_linear_basics/cyclone_kxky_scan.py
```

## Geometry I/O

Tabulated geometry round-trip:

```bash
python examples/02_geometry/tabulated_geometry_roundtrip.py
```

## Sheath / MPSE boundary conditions

Open field line + MPSE Bohm BCs:

```bash
python examples/03_sheath_mpse/open_slab_sheath_ky_scan.py
```

Loizu (2012) MPSE boundary condition comparison:

```bash
python examples/03_sheath_mpse/loizu2012_full_mpse_bc.py
```

Loizu (2012) MPSE “full set” in the hot-ion model (includes $\partial_\parallel T_i = 0$ at the MPSE nodes):

```bash
python examples/03_sheath_mpse/loizu2012_full_hot_ion_mpse_bc.py
```

Sheath heat transmission + SEE effects:

```bash
python examples/03_sheath_mpse/sheath_heat_see_effects.py --out out_sheath_heat
```

## Closures / transport

Parallel closures and sinks:

```bash
python examples/04_closures_transport/parallel_closures_effects.py
```

Braginskii/Spitzer transport scalings (equilibrium-based):

```bash
python examples/04_closures_transport/braginskii_closures_effects.py --out out_braginskii
```

Nonlinear field-line toggles (state-dependent non-Boussinesq + Braginskii):

```bash
python examples/04_closures_transport/nonlinear_flux_tube_toggles.py
```

## JAX autodiff

Autodiff optimization of $k_{y,*}$:

```bash
python examples/05_jax_autodiff/autodiff_optimize_ky_star.py
```

## Literature workflows

Mosetto (2012), Halpern (2013), Jorge (2016):

```bash
python examples/06_literature_tokamak_sol/mosetto2012_regime_map.py
python examples/06_literature_tokamak_sol/halpern2013_gradient_removal_lp.py
python examples/06_literature_tokamak_sol/jorge2016_isttok_linear_workflow.py
```

Optional slower calibrated-vs-ablation comparison for the Mosetto map:

```bash
python examples/06_literature_tokamak_sol/mosetto2012_regime_map.py --classifier both
```

More context and references: `docs/literature/`.

## ESSOS geometries (optional)

Near-axis/VMEC/Biot–Savart workflows:

```bash
python examples/07_essos_geometries/stellarator_nearaxis_essos.py
python examples/07_essos_geometries/essos_vmec_edge_s09.py
python examples/07_essos_geometries/essos_biotsavart_r14.py
```


## DRB2D conservative nonlinear gate

```bash
python examples/08_nonlinear_drb2d/drb2d_conservative_gate.py
python examples/08_nonlinear_drb2d/drb2d_energy_budget.py
python examples/08_nonlinear_drb2d/drb2d_energy_budget.py --model hot-ion
python examples/08_nonlinear_drb2d/drb2d_energy_budget.py --model em
python examples/08_nonlinear_drb2d/drb2d_nonbouss_gate.py
python examples/08_nonlinear_drb2d/drb2d_curvature_benchmarks.py
python examples/08_nonlinear_drb2d/drb2d_linear_phase_benchmark_em_hot_ion.py
python examples/08_nonlinear_drb2d/drb2d_movie.py
python examples/08_nonlinear_drb2d/drb2d_linear_phase_benchmark.py
python examples/10_verification/drb2d_solver_comparison.py
python examples/10_verification/poisson_preconditioner_bench.py
```

Produces a conservative energy time series figure and JSON metrics.

## Nonlinear HW2D milestone

Baseline turbulence run:

```bash
python examples/08_nonlinear_hw2d/hw2d_driftwave_turbulence.py
```

Neutrals effect:

```bash
python examples/08_nonlinear_hw2d/hw2d_neutrals_effect.py
```

Neutrals with optional charge-exchange-like vorticity drag:

```bash
python examples/08_nonlinear_hw2d/hw2d_neutrals_effect.py --nu-cx-omega 0.2
```

Movie (fast GIF):

```bash
python examples/08_nonlinear_hw2d/hw2d_movie.py
```

MMS convergence:

```bash
python examples/08_nonlinear_hw2d/mms_hw2d_convergence.py
```

## FCI preparation milestone

Slab MMS-style convergence for an FCI parallel derivative (field-line map + interpolation + centered difference):

```bash
python examples/09_fci/fci_slab_parallel_derivative_mms.py --out out_fci_mms
```

Hello-world FCI demo (parallel derivative + line-integral mapping):

```bash
python examples/09_fci/fci_hello_world.py --out out_fci_hello_world
```

Curved-map regression (spatially varying Bx):

```bash
python examples/09_fci/fci_curved_map_regression.py --out out_fci_curved_map
```

Minimal 3D DRB slab with sheath damping (energy/mass decay):

```bash
python examples/09_fci/fci_drb3d_sheath_budget.py --out out_fci_drb3d_sheath
```

## Verification (elliptic solves)

FD+CG Poisson solver verification (Dirichlet and Neumann cases):

```bash
python examples/10_verification/poisson_cg_verification.py --out out_poisson_cg_verify
```

Arnoldi vs dense Jacobian (tiny problem):

```bash
python examples/10_verification/arnoldi_vs_dense_jacobian.py --out out_arnoldi_dense_verify
```

Shear-Alfvén dispersion verification (Zhu et al. 2018 / GDB):

```bash
python examples/10_verification/saw_dispersion_gdb2018.py --out out_saw_gdb2018
```

Cold-ion DRB conservative hard-gate benchmark:

```bash
python examples/10_verification/drb_cold_ion_conservative_gate.py
```

Cold-ion DRB strict operator-gate benchmark:

```bash
python examples/10_verification/drb_cold_ion_operator_gate.py
```

Cold-ion DRB operator-splitting diagnostics:

```bash
python examples/10_verification/drb_operator_split_diagnostics.py
```
