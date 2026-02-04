# Verification examples

This folder contains scripts that reproduce **verification** checks commonly used in edge/SOL turbulence codes:

- elliptic solver verification (Poisson/polarization),
- operator convergence checks (MMS),
- conservation/budget closure checks.

These scripts are meant to be quick to run and to produce reviewer-friendly plots that can be included in documentation.

Scripts:

- `poisson_cg_verification.py`: verify the FD+CG Poisson solver for Dirichlet and Neumann BCs against analytic solutions.
- `arnoldi_vs_dense_jacobian.py`: verify matrix-free Arnoldi eigenvalues vs a dense Jacobian eigensolve (tiny system).
- `saw_dispersion_gdb2018.py`: verify a shear-Alfvén dispersion relation from the GDB code paper (Zhu et al. 2018).
