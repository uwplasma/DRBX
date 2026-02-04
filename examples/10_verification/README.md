# Verification examples

This folder contains scripts that reproduce **verification** checks commonly used in edge/SOL turbulence codes:

- elliptic solver verification (Poisson/polarization),
- operator convergence checks (MMS),
- conservation/budget closure checks.

These scripts are meant to be quick to run and to produce reviewer-friendly plots that can be included in documentation.

Scripts:

- `poisson_cg_verification.py`: verify the FD+CG Poisson solver for Dirichlet and Neumann BCs against analytic solutions.

