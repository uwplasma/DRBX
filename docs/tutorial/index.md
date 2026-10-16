# Tutorial: start here

Eight short steps, from a first run to a stellarator simulation. Each step is
one script in [`examples/tutorial/`](../../examples/tutorial) and one page
here. Every script has the same four parts: imports, a block of input
parameters (edit these), a verbose run, and the results (printed, plotted to
`docs/media/tutorial_*.png`, and explained in comments).

## Setup

```bash
git clone https://github.com/uwplasma/DRBX && cd DRBX
pip install -e .            # installs the `drbx` command and the package
python -c "import drbx"     # check
```

Run every script from the repository root:

```bash
PYTHONPATH=src python examples/tutorial/01_first_run.py
```

All models run in float64 (`jax_enable_x64`) on CPU.

## The steps

| Step | What you learn | Model | Python | CLI | Laptop time |
|---|---|---|---|---|---|
| [01 First run](01_first_run.md) | input decks, `drbx inspect/run`, outputs, restart | radial anomalous diffusion | yes | yes | ~3 s |
| [02 Drift-wave turbulence](02_drift_wave_turbulence.md) | a 2-D turbulence run, adiabaticity, energy and flux | Hasegawa-Wakatani | yes | no | ~50 s |
| [03 Linear vs nonlinear](03_linear_vs_nonlinear.md) | dispersion relations with `drbx.linear`, growth rates | linearized Hasegawa-Wakatani | yes | no | ~6 s |
| [04 Open field lines](04_open_sol_sheath.md) | sheath boundaries, the two-point model | isothermal SOL flux tube | yes | no | ~3 s |
| [05 Neutrals and detachment](05_neutrals_and_detachment.md) | recycling, atomic rates, a divertor leg | recycling SOL, SD1D-matched leg | yes | no | ~15 s |
| [06 Geometries](06_geometries.md) | FCI stellarator geometry, limiter SOL, island divertor | 4-field interchange | yes | no | ~12 s |
| [07 Differentiable workflows](07_differentiable.md) | gradients through runs, inverse design, control | HW + SD1D leg | yes | no | ~25 s |
| [08 HSX stellarator](08_hsx_fci_braginskii.md) | the production 3-D drift-reduced Braginskii backend | FCI Braginskii | yes | yes | ~3 min |

Notes:

- The `drbx` command runs TOML decks for the native deck models (step 01)
  and for the HSX backend (step 08). Steps 02-07 are Python APIs only; the
  pages say so.
- Every model here is covered by tests in `tests/`, and
  `tests/docs/examples/test_tutorial_scripts.py` runs each tutorial script with
  reduced parameters.
- For the full equation set see [Models and Governing Equations](../models_and_equations.md)
  and the [Equation To Code Map](../equation_to_code_map.md). Longer
  walk-throughs of the flagship examples:
  [Turbulence From Zero](../tutorial_hasegawa_wakatani.md),
  [Open SOL, Neutrals, Detachment](../tutorial_open_sol.md),
  [Stellarator FCI Turbulence](../tutorial_stellarator_fci.md).
