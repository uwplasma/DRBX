# Camargo, Biskamp & Scott (1995): resistive drift-wave turbulence (HW)

This reference studies 2D resistive drift-wave turbulence with the Hasegawa–Wakatani model:

- S. J. Camargo, D. Biskamp, and B. D. Scott, *Resistive drift-wave turbulence*,
  **Physics of Plasmas** 2(1), 48 (1995). DOI: [`10.1063/1.871116`](https://doi.org/10.1063/1.871116).

`jaxdrb` uses the HW2D milestone to reproduce **validation checks** that are standard in this literature:

- quadratic energy functional and budget closure,
- conservative advection operator checks (quadratic invariant conservation),
- qualitative spectral diagnostics.

## What `jaxdrb` implements

See:

- Model equations and implementation notes: `docs/nonlinear/hw2d.md`
- Validation summary and plots: `docs/validation.md`

Key code:

- `src/jaxdrb/nonlinear/hw2d.py`
- `src/jaxdrb/operators/brackets.py` (Arakawa bracket)

## What to run

Run the validation script:

```bash
python examples/08_nonlinear_hw2d/hw2d_camargo1995_validation.py --out out_hw2d_camargo1995
```

It writes:

- `panel_budget.png`: energy/enstrophy time traces + budget closure and decomposition,
- `spectrum.png`: qualitative isotropic spectra at final time,
- `timeseries.npz`: saved time series of diagnostics and budget terms.

