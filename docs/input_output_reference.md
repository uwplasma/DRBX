# Input And Output Reference

This page collects the current public input-deck, command-line, Python-driver,
and output-artifact conventions. It is the practical reference for users who
want to go beyond the README quick start.

## TOML Decks

The native runtime reads TOML decks. The most common top-level sections are:

| Section | Purpose |
| --- | --- |
| `[time]` | Output count and timestep. |
| `[runtime]` | Precision, backend-sensitive runtime options, and solver mode switches. |
| `[runtime.logging]` | Terminal verbosity and quiet mode. |
| `[mesh]` | Mesh size, spacing, Jacobian, and metric-like inputs. |
| `[solver]` | Native solver options such as step limits. |
| `[model]` | Model component list. |
| `[species.<name>]` | Species charge, mass, active equations, and transport coefficients. |
| `[fields.<name>]` | Initial fields and boundary conditions. |
| `[output]` | Artifact directory and write toggles. |
| `[restart]` | Optional restart input and resume length. |

Minimal example:

```toml
[time]
nout = 3
timestep = 5.0

[runtime]
precision = "float64"

[mesh]
nx = 16
ny = 24
nz = 1
dx = { expr = "0.0075 + 0.005*x" }
dy = 0.01
dz = 0.01
J = 1

[solver]
mxstep = 1000

[model]
components = ["h"]

[species.h]
type = ["evolve_density", "evolve_pressure", "anomalous_diffusion"]
AA = 1
charge = 1
anomalous_D = 2.0
thermal_conduction = false

[fields.Nh]
function = { expr = "1 + H(x - 0.25) * H(0.75-x) * exp(-(y-pi)^2)" }
bndry_all = "neumann"

[fields.Ph]
function = { ref = "Nh:function" }
bndry_all = "neumann"

[output]
directory = "output/restartable_diffusion"
write_summary = true
write_arrays = true
write_restart = true
write_log = true
```

The committed deck is
[`examples/inputs/restartable_diffusion.toml`](https://github.com/uwplasma/jax_drb/blob/main/examples/inputs/restartable_diffusion.toml).

## Expressions

Expression-valued inputs use `{ expr = "..." }`. The expression evaluator is
documented through examples rather than a separate symbolic language manual.
The current examples use coordinate names such as `x` and `y`, constants such
as `pi`, and helper functions such as `H(...)` for a Heaviside-style switch.

Reference-valued fields use `{ ref = "OtherField:function" }` when one field
should reuse another field definition.

## Boundary Conditions

The public TOML examples use field-level boundary keys such as:

```toml
[fields.Nh]
bndry_all = "neumann"
```

The higher-fidelity reference-backed and curated campaigns additionally stage
target/sheath/recycling boundary behavior from their validated setup paths. The
current capability tier for those cases is shown in the run summary and in the
validation matrix.

## CLI Commands

Run a deck:

```bash
jax_drb path/to/input.toml
```

Equivalent explicit form:

```bash
jax_drb run path/to/input.toml
```

Inspect without advancing:

```bash
jax_drb inspect path/to/input.toml
```

Resume from a restart bundle:

```bash
jax_drb run path/to/input.toml \
  --output-dir output/resumed_case \
  --restart-in output/base_case/base_restart.npz \
  --resume-steps 2
```

Use detailed runtime progress:

```bash
jax_drb run path/to/input.toml --verbose
```

## Python Driver

```python
from jax_drb.native import run_input_case

result = run_input_case(
    "examples/inputs/restartable_diffusion.toml",
    case_name="diffusion_driver",
    parity_mode="run",
    verbose=True,
)

print(result.time_points[-1])
print(sorted(result.variables))
```

Curated cases can be launched through:

```python
from pathlib import Path
from jax_drb.native import run_curated_case

result = run_curated_case(
    "tokamak_isothermal_one_step",
    reference_root=Path("/path/to/reference-suite"),
)
print(result.payload["capability_tier"])
```

## Output Artifacts

When `[output]` enables every public artifact, the runtime writes:

| Artifact | Contents |
| --- | --- |
| `<case>_summary.json` | Scalar metadata, capability tier, runtime configuration, and variable summaries. |
| `<case>_arrays.npz` | Field arrays and time histories for downstream plotting. |
| `<case>_restart.npz` | Restart payload for resumed runs. |
| `<case>_run_log.json` | Ordered runtime events, progress messages, output paths, and sanitized machine/runtime metadata. |

The run log is designed for long calculations. On live native implicit
recycling lanes it records interval progress, accepted timestep, simulated
time, elapsed wall time, and estimated remaining wall time.

## Capability Tiers

Every promoted or curated result should be interpreted through its tier:

| Tier | Meaning |
| --- | --- |
| `native_exact` | Strong enough for the main promoted native benchmark surface. |
| `native_operational` | Native and useful, but still carrying bounded residuals. |
| `scaffolded_reference_backed` | Useful for diagnostics or geometry staging, but not counted as native closure. |

The tier is written to summaries, run logs, and validation artifacts so docs
figures and paper figures do not overstate the solver claim.

## Artifact Locations In Docs

Large figures, movies, and NPZ bundles are release-hosted to keep the Git
repository lightweight. The docs link those assets directly from GitHub
Releases while keeping the code, JSON reports, and tests in the repository.
The full gallery is [Validation Gallery](validation_gallery.md).
