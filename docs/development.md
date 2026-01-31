# Development and contributing

## Repository structure

Source code lives in `src/jaxdrb/`:

- `geometry/`: geometry providers (analytic and tabulated)
- `models/`: the drift-reduced model RHS and parameter definitions
- `linear/`: matrix-free linearization, Arnoldi, growth-rate estimation
- `cli/`: command-line entry point
- `operators/`: small numerical operators (finite differences, placeholders)

## Local workflow

Run tests:

```bash
make test
```

Run the quick example:

```bash
make examples
```

Build docs (requires `.[docs]` installed):

```bash
make docs
```

## Coding principles (v1)

- Keep geometry isolated behind the `Geometry` interface.
- Prefer matrix-free linear operators (`matvec`) over dense matrices.
- Avoid global Poisson solves in v1; keep the Fourier polarization closure fast.
- Keep allocations small; use JAX primitives where appropriate.

## Adding a new geometry provider

Create a new `eqx.Module` implementing:

- `kperp2(kx, ky) -> k_perp^2(l)`
- `dpar(f) -> ∇_|| f`
- `curvature(kx, ky, f) -> C(f)`

Then:

- add it to `src/jaxdrb/geometry/__init__.py`,
- add a CLI option (optional),
- add an example and (ideally) a small test.

## Contributing

Contributions are welcome, especially:

- improved physical closures (non-Boussinesq polarization, sheath models),
- better eigenvalue solvers (implicit restart, shift-invert),
- robust geometry importers from equilibrium tools,
- documentation improvements.

Before opening a PR:

- ensure `python -m pytest -q` passes,
- keep changes focused (separate PRs for unrelated refactors).

