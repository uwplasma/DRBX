# Nonlinear capabilities

`jaxdrb` includes a growing set of nonlinear models and validation gates designed to be
auditable and differentiable. The nonlinear stack is no longer a placeholder: it is a
first‑class capability that complements the linear field‑line solvers.

The nonlinear subsystem includes:

- **HW2D** (Hasegawa–Wakatani‑like) as a fast turbulence testbed for operator and solver validation,
- **DRB2D** (cold‑ion) with conservative operator splitting, energy budgets, and curvature benchmarks,
- **Hot‑ion** and **EM** DRB2D branches with parity tests and curvature‑drive comparisons,
- **Neutral coupling** and MMS convergence tests,
- **FCI preparation** with analytic slab maps, curved‑map regression, and minimal 3D slab operators
  with conservative + sheath budget gates.

These models are used to:

- test numerical kernels (operators and time stepping),
- validate invariants (where applicable),
- benchmark performance and solver choices,
- prepare the code structure for fully nonlinear 3D DRB with open‑field‑line physics.

In addition, `jaxdrb` includes a **periodic cold‑ion DRB conservative gate** on the actual field‑line
branch (energy/mass/charge/current/momentum invariants), with both finite‑time and operator‑level checks:

- test: `tests/test_drb_nonlinear_conservative_gate.py`
- test: `tests/test_drb_operator_rates.py`
- example: `examples/10_verification/drb_cold_ion_conservative_gate.py`
- example: `examples/10_verification/drb_cold_ion_operator_gate.py`
- CI benchmark: `benchmarks/check_drb_conservative_gate.py`

![Cold-ion DRB strict conservative operator gate](../assets/images/drb_cold_ion_operator_gate.png)

![Cold-ion DRB operator split diagnostics](../assets/images/drb_operator_split_diagnostics.png)

## Run the nonlinear examples

From the repository root:

```bash
python examples/08_nonlinear_hw2d/hw2d_driftwave_turbulence.py
python examples/08_nonlinear_hw2d/hw2d_neutrals_effect.py
python examples/08_nonlinear_hw2d/hw2d_movie.py
python examples/08_nonlinear_drb2d/drb2d_movie.py
python examples/08_nonlinear_drb2d/drb2d_kelvin_helmholtz.py
python examples/08_nonlinear_drb2d/drb2d_hot_ion_movie.py
python examples/08_nonlinear_drb2d/drb2d_energy_budget.py
python examples/08_nonlinear_drb2d/drb2d_energy_budget.py --model hot-ion
python examples/08_nonlinear_drb2d/drb2d_energy_budget.py --model em
python examples/08_nonlinear_drb2d/drb2d_nonbouss_gate.py
python examples/08_nonlinear_drb2d/drb2d_curvature_benchmarks.py
python examples/08_nonlinear_drb2d/drb2d_curvature_benchmarks_em_hot_proxy.py
python examples/08_nonlinear_drb2d/drb2d_linear_phase_benchmark_em_hot_ion.py
python examples/08_nonlinear_drb2d/drb2d_linear_phase_benchmark.py

python examples/08_nonlinear_drb2d/drb2d_conservative_gate.py
python examples/04_closures_transport/nonlinear_flux_tube_toggles.py
```

Both examples write results to small `out_*` folders with plots and `.npz` data.

![DRB2D energy budget](../assets/images/drb2d_energy_budget.png)
![DRB2D hot-ion energy budget](../assets/images/drb2d_energy_budget_hot_ion.png)
![DRB2D EM energy budget](../assets/images/drb2d_energy_budget_em.png)
![DRB2D curvature benchmark](../assets/images/drb2d_curvature_benchmarks.png)
![DRB2D EM/hot-ion curvature proxy](../assets/images/drb2d_curvature_benchmarks_em_hot.png)
The curvature benchmark now includes EM and hot-ion variants alongside the base DRB2D branch.
![DRB2D hot-ion linear-phase benchmark](../assets/images/drb2d_linear_phase_hot_ion.png)
![DRB2D EM linear-phase benchmark](../assets/images/drb2d_linear_phase_em.png)

## Notes on DRB2D turbulence movies

The DRB2D movie (`examples/08_nonlinear_drb2d/drb2d_movie.py`) is intended to be short and
reproducible, but 2D drift-wave/interchange-like systems can condense into a zonal/banded state
at long times. To keep the README movie visually informative on coarse grids, the default
parameters include:

- small biharmonic hyperdiffusion (`Dn4`, `DOmega4`, `DTe4`) and
- a weak zonal vorticity drag (`mu_zonal_omega`)

These are numerical control knobs and should be reported explicitly when used in studies.

The hot-ion DRB2D movie (`examples/08_nonlinear_drb2d/drb2d_hot_ion_movie.py`) is a companion
case that exercises the `Ti` extension while keeping the same conservative advection and Poisson
inversion choices. Its defaults are tuned slightly more dissipative to avoid non-finite values on
coarse grids.

The SOL proxy movie (`examples/08_nonlinear_drb2d/drb2d_sol_movie.py`) introduces a closed→open
radial mask (LCFS at $x=x_s$) to emulate SOL-like transport and outward blob propagation.

## DRB2D equations and numerics

See [`docs/nonlinear/drb2d.md`](drb2d.md) for the full 2D model equations, closures,
boundary conditions, and references.

## Kelvin–Helmholtz benchmark (2D vorticity limit)

`examples/08_nonlinear_drb2d/drb2d_kelvin_helmholtz.py` runs the DRB2D solver in the
incompressible vorticity limit (curvature/parallel/drives disabled) and initializes a
double shear-layer profile with a small sinusoidal perturbation. The resulting movie shows
KH roll-up, vortex merging, and secondary instabilities, providing a compact nonlinear
advection benchmark for the conservative bracket and Poisson solve.

## Hermes-2 blob2d proxy (open-field-line interchange)

`examples/08_nonlinear_drb2d/drb2d_hermes2_blob2d.py` reproduces the Hermes-2 `blob2d`
configuration in a 2D DRB2D setting: Gaussian density/pressure blob and curvature drive
(`bxcvz = 1/R^2`, $R=1.5$ m). The defaults normalize the domain so $L_{rad}=L_{pol}=0.3$ m
maps to $L_x=L_y=1$ and use periodic boundaries with linear damping to emulate open-field-line
losses on a reduced grid. The benchmark tracks outward blob motion and radial flux over a
short, reproducible run.
