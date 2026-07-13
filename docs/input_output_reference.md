# Input And Output Reference

This page is the practical reference for every public input and output surface.
Use it after the [Feature Reference](feature_reference.md) when you know which
workflow you want to run and need the exact files, command-line switches,
Python entry points, and artifacts.

## Input Families

| Input family | Used by | What to edit |
| --- | --- | --- |
| TOML native decks | `jax_drb run`, `jax_drb inspect`, `jax_drb run-case` staging | Time, runtime precision, mesh, solver settings, model components, species, fields, output, restart |
| Python example constants | scripts under `examples/` | Constants near the top of the file: grid sizes, timestep, output directory, model flags, plotting toggles |
| Release-backed artifacts | movies, docs galleries, cached user examples, cached parity checks | Restored by `scripts/fetch_example_artifacts.py`; do not edit by hand |
| Developer reference roots | live parity refresh and heavy external-geometry regeneration | `--reference-root`, `JAX_DRB_REFERENCE_ROOT`, or geometry-specific environment variables |
| Imported geometry payloads | ESSOS/VMEC/VMEC-extender workflows | Field-line arrays, VMEC-coordinate maps, compact NetCDF edge-field grids, metadata JSON |

Most user examples are self-contained after artifact restoration and do not
require external plasma-code installations. Developer/live-reference commands
are documented separately and require explicit local reference inputs.

## TOML Decks

The native runtime reads TOML decks. A minimal promoted deck is committed at
[`examples/inputs/restartable_diffusion.toml`](https://github.com/uwplasma/jax_drb/blob/main/examples/inputs/restartable_diffusion.toml).

Common top-level sections:

| Section | Keys and meaning |
| --- | --- |
| `[time]` | `nout` output intervals, `timestep` physical or normalized interval length |
| `[runtime]` | `precision` (`float32` or `float64`) and backend-sensitive runtime options |
| `[runtime.logging]` | `verbosity`, `verbose`, `quiet`; controls terminal progress and log detail |
| `[mesh]` | `nx`, `ny`, `nz`, `dx`, `dy`, `dz`, `J`, metric-like factors or expression-valued spacing |
| `[solver]` | native solver selectors, step limits, and opt-in research backend switches |
| `[model]` | active component list and model-family selectors |
| `[species.<name>]` | species type/equations, charge, mass number, transport coefficients, closure toggles |
| `[fields.<name>]` | initial field formula or reference, boundary conditions, guard behavior |
| `[output]` | output directory and write toggles for summary, arrays, restart, and run log |
| `[restart]` | optional restart input and resume length |

Minimal example:

```toml
[time]
nout = 3
timestep = 5.0

[runtime]
precision = "float64"

[runtime.logging]
verbosity = "detailed"
verbose = true
quiet = false

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

### Expressions And References

Expression-valued inputs use `{ expr = "..." }`. Current examples use
coordinate names such as `x`, `y`, and `z`, constants such as `pi`, elementary
functions such as `exp`, and helper functions such as `H(...)` for a
Heaviside-style switch.

Reference-valued fields use `{ ref = "OtherField:function" }` when one field
should reuse another field definition.

### Boundary Conditions

The public TOML examples use field-level boundary keys such as:

```toml
[fields.Nh]
bndry_all = "neumann"
```

Higher-fidelity recycling, sheath, neutral, and target behavior is usually
staged by curated campaigns or Python examples, because those paths need
species coupling, target masks, guard-cell sequencing, and source accounting.
Their capability tier is written to the run summary and validation report.

## CLI Commands

The executable is `jax_drb`. Running a TOML path directly is equivalent to
`jax_drb run`.

| Command | Purpose | Typical inputs | Typical outputs |
| --- | --- | --- | --- |
| `jax_drb run input.toml` | Run a native input deck | TOML deck, optional restart | summary JSON, arrays NPZ, restart NPZ, run log |
| `jax_drb inspect input.toml` | Parse and print the resolved plan without advancing | TOML deck | terminal plan summary |
| `jax_drb reference-cases` | List curated reference-backed cases | optional reference root | terminal or JSON case list |
| `jax_drb run-case` | Run a curated case through native JAXDRB | reference root or fixture deck, case name | portable native summary |
| `jax_drb compare-summary` | Compare portable summary JSON files | expected and actual JSON | scalar diff report |
| `jax_drb compare-arrays` | Compare portable NPZ payloads | expected and actual NPZ | array diff report |
| `jax_drb compare-recycling` | Localize recycling parity differences | compact reference/native artifacts | worst-variable/cell report |

Common commands:

```bash
jax_drb inspect examples/inputs/restartable_diffusion.toml
jax_drb run examples/inputs/restartable_diffusion.toml --verbose
jax_drb run examples/inputs/restartable_diffusion.toml \
  --output-dir output/resumed_case \
  --restart-in output/base_case/base_restart.npz \
  --resume-steps 2
```

## Python Entry Points

Use the CLI for ordinary runs. Use the Python API when embedding JAXDRB in
analysis scripts or examples.

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

Common Python entry points:

| API | Source | Purpose |
| --- | --- | --- |
| `run_input_case` | [`src/jax_drb/native/runner.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/runner.py) | Run a TOML deck and return structured native output |
| `run_curated_case` | [`src/jax_drb/native/runner.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/runner.py) | Run curated validation cases through native JAXDRB |
| `load_run_config` | [`src/jax_drb/runtime/run_config.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/runtime/run_config.py) | Parse runtime configuration |
| `write_run_outputs` and output helpers | [`src/jax_drb/runtime/output.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/runtime/output.py) | Write summary, arrays, restart, and run-log artifacts |
| `compute_fci_drb_rhs` | [`src/jax_drb/native/fci_drb_rhs.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/fci_drb_rhs.py) | Evaluate compact 3D FCI DRB RHS terms |
| `build_*_campaign` functions | [`src/jax_drb/validation`](https://github.com/uwplasma/jax_drb/tree/main/src/jax_drb/validation) | Generate validation reports, figures, and publication artifacts |

## Output Artifacts

When `[output]` enables every public artifact, the runtime writes:

| Artifact | Format | Contents | Typical use |
| --- | --- | --- | --- |
| `<case>_summary.json` | JSON | scalar metadata, capability tier, runtime configuration, variable summaries, output manifest | quick inspection, docs tables, CI checks |
| `<case>_arrays.npz` | NPZ | field arrays and time histories | plotting, parity, analysis scripts |
| `<case>_restart.npz` | NPZ | restart payload with fields and metadata | resume native runs |
| `<case>_run_log.json` | JSON | ordered runtime events, progress messages, artifact paths, sanitized runtime metadata | long-run monitoring and reproducibility |
| validation report | JSON | campaign settings, metrics, thresholds, capability tier, literature/reference notes | docs and paper figures |
| validation arrays | NPZ | campaign arrays used to make plots | regeneration and review |
| validation figures | PNG/GIF | publication-style plots and movies | docs, README, manuscript |

The run log is designed for long calculations. On live native implicit
recycling lanes it records interval progress, accepted timestep, simulated
time, elapsed wall time, and estimated remaining wall time.

## Where Outputs Go

Most examples write to either `output/<case>/` or `docs/data/<case>_artifacts/`.
The first location is for user runs. The second location is used by examples
that regenerate documentation figures. Large files in `docs/data/` are either
ignored by git or restored from the release artifact manifest.

Use these conventions:

| Situation | Recommended output directory |
| --- | --- |
| Trying a tutorial locally | `output/<case_name>/` |
| Regenerating a documented example figure | the script default under `docs/data/<case>_artifacts/` |
| Running heavy profiling | `tmp/profiles/<profile_name>/` |
| Running a live reference campaign | explicit scratch directory outside the repo or under ignored `tmp/` |

## Artifact Restore And No-Bloat Policy

The repository tracks source code, documentation, small fixtures, and small
machine-readable reports. Large media and baselines are release-hosted.

Restore only user-facing media:

```bash
python scripts/fetch_example_artifacts.py --skip-baselines
```

Restore media plus cached baselines:

```bash
python scripts/fetch_example_artifacts.py
```

Use a shared cache:

```bash
export JAX_DRB_ARTIFACT_CACHE_DIR=/path/to/cache
python scripts/fetch_example_artifacts.py --skip-baselines
```

For private-repository users, authenticate with `gh auth login --hostname
github.com` or set `GH_TOKEN`/`GITHUB_TOKEN`.

## Capability Tiers In Outputs

Every promoted or curated result should be interpreted through its tier:

| Tier | Meaning |
| --- | --- |
| `native_exact` | Strong enough for the main promoted native benchmark surface. |
| `native_operational` | Native and useful, but still carrying bounded residuals or reduced fidelity. |
| `scaffolded_reference_backed` | Useful for diagnostics or geometry staging, but not counted as native closure. |

The tier is written to summaries, run logs, and validation artifacts so docs
figures and paper figures do not overstate the solver claim.

## Source Links

For implementation details, use:

| Topic | Source |
| --- | --- |
| CLI parsing and command dispatch | [`src/jax_drb/cli.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/cli.py) |
| Runtime state and output writing | [`src/jax_drb/runtime`](https://github.com/uwplasma/jax_drb/tree/main/src/jax_drb/runtime) |
| Native run orchestration | [`src/jax_drb/native/runner.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/runner.py) |
| Fixed-layout recycling residuals | [`src/jax_drb/native/recycling_fixed_residual.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/recycling_fixed_residual.py) |
| 3D FCI geometry and RHS terms | [`src/jax_drb/native/fci.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/fci.py), [`src/jax_drb/native/fci_drb_rhs.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/fci_drb_rhs.py) |
| Validation campaigns and plotting | [`src/jax_drb/validation`](https://github.com/uwplasma/jax_drb/tree/main/src/jax_drb/validation) |
