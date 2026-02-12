# Boundary conditions

`jaxdrb` currently uses two different notions of “boundary conditions”, depending on the problem:

1. **Field-line (flux-tube) models** are 1D in the parallel coordinate $\ell$ (with perpendicular directions treated in Fourier space).
2. **Nonlinear 2D testbeds** (HW2D/DRB2D) evolve fields on a perpendicular box $(x,y)$.
3. **FCI/3D milestone models** evolve fields on a plane stack and impose plate/target interactions through
   target-aware parallel operators.

This page documents both the *physics-motivated* sheath/MPSE closures and the *numerical* BC hooks that are useful for benchmarking and for the nonlinear transition.

For coordinate conventions (what “$x,y,\ell$” mean in each branch), see:

- `geometry/conventions.md`

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

The nonlinear 2D testbeds assume periodicity by default (FFT Poisson solve; spectral or Arakawa bracket).
This is the fast path used in most regression gates and README movies.

**Important:** in the 2D perpendicular-box models there is *no explicit parallel coordinate*, so physical
sheath closures are not applied in 2D. The 2D systems are used to validate conservative operators and
time stepping before turning on the full 3D open-field-line problem.

Key points:

- **Fast path (default)**: periodic in $x$ and $y$ → FFT Poisson solve and spectral bracket.
- **Non-periodic path (experimental)**: Dirichlet/Neumann in $x$ and $y$ → FD operators + matrix-free CG Poisson solve.

The HW2D CLI exposes these options:

```bash
jaxdrb-hw2d --poisson cg_fd --bracket centered --bc-x dirichlet --bc-y dirichlet --bc-enforce-nu 10.0
```

Because the non-periodic path is intended for development/benchmarking, it is currently more limited than the periodic path (and slower).

## FCI / 3D (planes with targets/plates)

In the FCI/3D milestone branch, the parallel direction is realized by **field-line maps** between planes.
This enables open-field-line physics with **plate/target intersections**:

- the parallel derivative becomes one-sided/non-uniform near targets,
- sheath/plate closures become localized sink/source terms at hit points,
- particle and energy budgets can be tracked explicitly as diagnostic channels.

The FCI operator takes:

- forward/backward maps (`map_fwd`, `map_bwd`),
- a `BC1D` specification for the target model (Dirichlet/Neumann/Periodic),
- optional target-handling modes (Appendix-B style B/C/X classification).

See:

- `fci/index.md` (concept + references)
- `fci/maps.md` (map file format + metadata)
- `src/jaxdrb/fci/parallel.py` (target-aware ∂|| implementation)
