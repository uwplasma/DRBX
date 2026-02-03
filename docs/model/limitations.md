# Limitations and interpretation

This project is intentionally simplified.

## Physical scope

- The model is meant for **qualitative** drift-wave / ballooning-like stability behavior.
- The current equations are not a complete SOL model:
  - no full sheath boundary-condition implementation at the magnetic pre-sheath entrance,
  - no neutral physics or realistic sources/sinks,
  - simplified closures (a lightweight volumetric sheath-loss option exists for open field lines).
- The equilibrium is currently a simple local expansion (constant gradient drives via `omega_n`,
  `omega_Te`) rather than a self-consistent profile.

## Numerical scope

- The field-line coordinate `l` uses periodic finite differences by default.
- The eigenvalue solver is a basic, matrix-free Arnoldi implementation without implicit restarting.
- For difficult cases the CLI increases the Krylov dimension up to the full state dimension
  `N = 5 * nl`.

## Coordinate and operator conventions

Geometry is abstracted, so the meaning of `curvature0`, `omega_d`, and metric coefficients is tied
to the chosen geometry model. This is deliberate: it keeps the solver core geometry-agnostic.

If you intend to compare to a specific reference (e.g., a published dispersion relation), ensure
that:

- your geometry model matches the reference’s operator definitions,
- the parameter normalization matches the reference,
- the equilibrium drives (e.g. `omega_n`) map to the reference gradients.
