# Boundary conditions

`jaxdrb` currently uses two different notions of “boundary conditions”, depending on the problem:

1. **Field-line (flux-tube) linear models** are 1D in the parallel coordinate $l$ (with perpendicular directions treated in Fourier space).
2. **Nonlinear milestone models** (HW2D) are 2D in $(x,y)$ on a rectangular domain.

This page documents both the *physics-motivated* sheath/MPSE closures and the *numerical* BC hooks that are useful for benchmarking and for the nonlinear transition.

## Field-line (1D in $l$)

### Default behavior

The field-line geometries provided by `jaxdrb.geometry.*` choose one of:

- **Periodic** field lines (closed topology): periodic finite differences are used for $\nabla_\parallel$.
- **Open** field lines (SOL-like): open-grid finite differences are used for $\nabla_\parallel$ and *optional* sheath/MPSE closures can be enabled.

### MPSE / sheath entrance boundary conditions (physics closure)

For open field lines, `jaxdrb` can apply Loizu-style magnetic-pre-sheath entrance (MPSE) boundary conditions at the two ends of the field line.

These are controlled by `DRBParams.sheath_bc_*` and documented in:

- `docs/model/extensions.md` (overview and toggles)
- `docs/literature/index.md` (literature-aligned workflows)

The code includes:

- a **simple** MPSE mode (velocity-focused closure), and
- a **Loizu (2012) “full set”** *linearized* enforcement mode used for SOL linear studies.

In the **hot-ion** model, the Loizu2012 full-set option also enforces a matching
ion-temperature entrance constraint $\partial_\parallel T_i = 0$ (Neumann at the MPSE nodes).

For `*-open` geometries in the CLI, MPSE/Bohm sheath entrance boundary conditions are enabled by default
(disable with `--no-sheath-bc`).

`jaxdrb` also provides an optional **sheath heat transmission / energy-loss** closure localized at the
MPSE nodes. This is controlled by `DRBParams.sheath_heat_on` (and related `sheath_gamma_*` / SEE knobs)
and is documented in `docs/model/extensions.md`.

### User-defined BCs (numerical hook)

For benchmarking and nonlinear-preparation work, `jaxdrb` also supports **user-defined** BCs that can be applied to the *evolving perturbation fields* at the ends of the field line:

- periodic
- Dirichlet
- Neumann

These are enforced weakly as *relaxation/SAT* terms added to the RHS at the boundary nodes.

Implementation:

- `jaxdrb.bc.BC1D` stores a BC type and parameters.
- `jaxdrb.models.bcs.LineBCs` stores optional per-field BCs.
- Each RHS adds a term of the form

$$
\partial_t f \;\leftarrow\; \partial_t f - \nu\,\chi_{\partial\Omega}\,(f - f_{\text{target}}),
$$

where $\chi_{\partial\Omega}$ is a mask that is nonzero only at the two ends of the grid, and $f_{\text{target}}$ is computed from the requested BC (value or implied value from a one-sided derivative relation).

CLI shortcut (applied to all fields uniformly):

```bash
jaxdrb-scan --geom slab-open --line-bc dirichlet --line-bc-value 0 --line-bc-nu 5.0 ...
```

These user BCs are not meant to replace MPSE/sheath models; they are meant to make it easy to compare how sensitive results are to end conditions in controlled tests.

## Nonlinear (2D in $x,y$)

The nonlinear HW2D milestone assumes periodicity by default (FFT operators and pseudo-spectral bracket).
For experimentation and preparation work, it also supports Dirichlet/Neumann BCs in $(x,y)$ with a finite-difference Laplacian and a CG Poisson solve.

Key points:

- **Fast path (default)**: periodic in $x$ and $y$ → FFT Poisson solve and spectral bracket.
- **Non-periodic path (experimental)**: Dirichlet/Neumann in $x$ and $y$ → FD operators + matrix-free CG Poisson solve.

The HW2D CLI exposes these options:

```bash
jaxdrb-hw2d --poisson cg_fd --bracket centered --bc-x dirichlet --bc-y dirichlet --bc-enforce-nu 10.0
```

Because the non-periodic path is intended for development/benchmarking, it is currently more limited than the periodic path (and slower).
