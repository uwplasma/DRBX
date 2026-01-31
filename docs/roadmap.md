# Roadmap

This is a v1 project with a focus on correctness and extensibility over completeness.

## Near-term

- Add shift-invert support (GMRES solve of `(J - σ I)x = b`) to better target unstable eigenvalues.
- Add better default equilibria (simple 1D profiles along `l`) and explicit equilibrium builders.
- Add more geometry sources (field-line traced from equilibria, diverted configurations).

## Medium-term

- Relax the Boussinesq closure to non-Boussinesq polarization.
- Add multi-mode perpendicular spectral support to enable nonlinear brackets.
- Add sheath boundary conditions and simple SOL closures.

## Long-term

- Coupling to equilibrium/field-line tracer pipelines and high-fidelity geometry.
- Implicit/IMEX time stepping and better stiffness handling in the initial-value solver.

