# Fluid 1D MMS Convergence

This package promotes the existing fluid-1D manufactured-solution refinement
study from a standalone script into a reviewable validation bundle.

It serves two purposes:

- direct verification that the promoted 1D density, pressure, and momentum
  operators converge under refinement;
- a publication-ready convergence figure that can be reused in the docs and in
  the future manuscript instead of leaving the evidence hidden in a JSON file or
  a test assertion.

The generated package contains:

- a summary JSON with per-resolution errors and observed orders;
- a compact NPZ payload for downstream plotting and regression checks;
- a publication-grade convergence plot.

This is a verification surface, not a code-to-code parity surface. It belongs
to the same evidence family emphasized in the verification literature and in
the convergence sections of major edge/SOL code papers: refinement studies are
used to establish operator correctness before broader benchmark validation is
interpreted.

Run the demo with:

```bash
python examples/engineering/fluid_1d_mms_convergence_demo.py
```

Committed artifacts:

- `docs/data/fluid_1d_mms_convergence_artifacts/data/fluid_1d_mms_convergence.json`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__fluid_1d_mms_convergence_artifacts__data__fluid_1d_mms_convergence.npz`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__fluid_1d_mms_convergence_artifacts__images__fluid_1d_mms_convergence.png`
