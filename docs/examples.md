# Examples

The `examples/` tree is organized by complexity:

- `examples/1_simple/`: quick “hello world” runs (single ky scan, minimal knobs).
- `examples/2_intermediate/`: richer diagnostics (tabulated geometry round-trip, kx–ky scans, JAX workflows).
- `examples/3_advanced/`: literature-inspired workflows and stellarator (ESSOS) geometry.

All scripts:

- write an `out/...` folder with `.npz` data plus multiple publication-ready `.png` figures,
- print progress and key numbers so it does not look like the run is “hanging”.

> Tip: most examples will run faster if you reduce `nl` and use fewer `ky`/`kx` points.

## 1) Simple (fast)

Slab ky scan (curvature off, drift-wave-like):

```bash
python examples/1_simple/01_slab_ky_scan.py
```

Circular tokamak ky scan (curvature on):

```bash
python examples/1_simple/02_circular_tokamak_ky_scan.py
```

Cyclone-like s-alpha ky scan:

```bash
python examples/1_simple/03_salpha_cyclone_ky_scan.py
```

Open field line + sheath closures (MPSE Bohm-sheath BC enforcement + optional volumetric loss proxy):

```bash
python examples/1_simple/04_open_slab_sheath_ky_scan.py
```

## 2) Intermediate (diagnostics + workflows)

Tabulated geometry round-trip (export analytic coefficients → reload → verify results match):

```bash
python examples/2_intermediate/01_tabulated_geometry_roundtrip.py
```

2D (kx, ky) scan on Cyclone-like s-alpha geometry:

```bash
python examples/2_intermediate/02_cyclone_kxky_scan.py
```

JAX “advantage” demo: autodiff optimization of $k_{y,*}$ that maximizes $\max(\gamma,0)/k_y$:

```bash
python examples/2_intermediate/03_jax_autodiff_optimize_ky_star.py
```

Electromagnetic (finite-beta) workflow demo:

```bash
python examples/2_intermediate/04_em_beta_scan.py
```

Hot-ion (finite $\tau_i$) workflow demo:

```bash
python examples/2_intermediate/05_hot_ions_tau_scan.py
```

Parallel closures + sinks demo (parallel heat conduction / flow diffusion):

```bash
python examples/2_intermediate/06_parallel_closures_effects.py
```

## 3) Advanced (literature-inspired + stellarator)

Mosetto (2012) drift-wave branch workflow (curvature off):

```bash
python examples/3_advanced/01_mosetto2012_driftwave_branches.py
```

Mosetto (2012) ballooning-like branch + shear trends (curvature on):

```bash
python examples/3_advanced/02_mosetto2012_ballooning_branches.py
```

Halpern (2013) gradient removal + fixed-point $L_p$ workflow:

```bash
python examples/3_advanced/03_halpern2013_gradient_removal_lp.py
```

Near-axis stellarator geometry via ESSOS (optional dependency):

```bash
python examples/3_advanced/04_stellarator_nearaxis_essos.py
```

Additional advanced literature/geometry examples (optional and potentially slower) are listed in
`docs/literature/index.md`.

ISTTOK-inspired linear workflow (Jorge et al. 2016):

```bash
python examples/3_advanced/05_jorge2016_isttok_linear_workflow.py
```

Loizu (2012) MPSE boundary conditions (simple vs full-set linearized enforcement):

```bash
python examples/3_advanced/11_loizu2012_full_mpse_bc.py
```

Nonlinear milestone (2D periodic drift-wave turbulence, HW-like):

```bash
python examples/3_advanced/20_hw2d_driftwave_turbulence.py
```

Nonlinear + neutrals (minimal particle-exchange model):

```bash
python examples/3_advanced/21_hw2d_neutrals_effect.py
```
