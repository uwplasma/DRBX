# Native Runtime CLI

`dkx` now has a standalone native runtime surface for supported inputs. After an editable install,

```bash
pip install -e .
```

both of these console commands are available:

```bash
dkx path/to/input.toml
dkx path/to/input.toml
```

The bare input-file form is equivalent to:

```bash
dkx run path/to/input.toml
```

## Input Layout

The native CLI accepts organized TOML decks and also keeps compatibility with legacy `.inp` decks. The intended TOML layout is:

- `[time]`
- `[runtime]`
- `[runtime.logging]`
- `[mesh]`
- `[solver]`
- `[model]`
- `[output]`
- `[restart]`
- `[species.<name>]`
- `[fields.<name>]`

Expression-valued entries can be written explicitly as wrappers:

```toml
[mesh]
dx = { expr = "0.0075 + 0.005*x" }

[fields.Nh]
function = { expr = "1 + H(x - 0.25) * H(0.75-x)" }
```

List-valued component/type entries use standard TOML arrays:

```toml
[model]
components = ["h"]

[species.h]
type = ["evolve_density", "evolve_pressure", "anomalous_diffusion"]
```

## Precision Selection

Precision can be chosen in the input file:

```toml
[runtime]
precision = "float64"

[runtime.logging]
verbosity = "detailed"
verbose = true
quiet = false

[output]
directory = "output/my_case"
write_summary = true
write_arrays = true
write_restart = true
write_log = true
```

or overridden at the terminal:

```bash
dkx input.toml --precision float32
```

The terminal logging mode can also be controlled directly:

```bash
dkx input.toml --verbose
```

The logging rules are:

- `[runtime.logging].verbose = true` means detailed staged event output
- `[runtime.logging].verbose = false` means the concise summary path
- `[runtime.logging].verbosity = "summary"` or `"detailed"` pins the level explicitly
- `[runtime.logging].quiet = true` suppresses terminal output entirely
- `--verbose` overrides the deck for a one-off detailed run

Current status:

- `float64` is the default and the most complete runtime mode.
- `float32` now runs cleanly on the simple diffusion/restart tutorial path and no longer emits the old internal dtype-truncation warnings there.
- broader native paths still contain explicit `float64` requests internally, so `float32` should currently be treated as an opt-in performance experiment, not a default production mode.

For a concrete measurement workflow, use:

```bash
PYTHONPATH=src .venv/bin/python examples/diffusion_precision_benchmark.py
```

The committed example benchmark artifacts are in:

- [docs/runtime_precision_benchmark/data/diffusion_precision_analysis.json](docs/runtime_precision_benchmark/data/diffusion_precision_analysis.json)
- [https://github.com/uwplasma/dkx/releases/download/validation-artifacts-2026-04-28/docs__runtime_precision_benchmark__images__diffusion_precision_elapsed.png](https://github.com/uwplasma/dkx/releases/download/validation-artifacts-2026-04-28/docs__runtime_precision_benchmark__images__diffusion_precision_elapsed.png)

On the current machine, the warm second-run `float32` diffusion path is about `1.23x` faster than `float64` (`2.096s` vs `2.584s`) on the same input.

## Runtime Output

A native run can write four main artifact types:

- summary JSON
- arrays NPZ
- restart NPZ
- verbose run-log JSON

Each emitted payload is also expected to carry the current capability-tier label for that run:

- `native_exact`
- `native_operational`

For direct deck-driven native runs, the current default is `native_exact`.

Example:

```bash
dkx examples/inputs/restartable_diffusion.toml \
  --precision float32
```

If the deck includes an `[output]` section, the bare `dkx input.toml` form is enough. CLI flags still override the deck when you need an ad hoc run location. A typical deck-controlled output block is:

```toml
[output]
directory = "output/restartable_diffusion"
write_summary = true
write_arrays = true
write_restart = true
write_log = true
```

This writes:

- `<output-dir>/<case>_summary.json`
- `<output-dir>/<case>_arrays.npz`
- `<output-dir>/<case>_restart.npz`
- `<output-dir>/<case>_run_log.json`

The terminal output is rich-formatted when `rich` is available and falls back to plain text otherwise. It now has two layers:

- event-style run messages while the simulation is being configured, restarted, launched, and written out
- the final run summary table

For Python driver scripts, the same native entry point now exposes a matching verbose switch:

```python
from dkx.native import run_input_case

result = run_input_case(
    "examples/inputs/restartable_diffusion.toml",
    case_name="diffusion_driver",
    parity_mode="run",
    verbose=True,
)
```

`verbose=True` emits the same staged event stream through the native runner, and `event_logger=` can be supplied if a script wants to capture those events instead of printing them.

Both versions report the same core metadata:

- input file
- case name
- runtime precision
- runtime backend/device/cache
- runtime library and machine metadata (`jax_version`, `python_version`, platform, process id)
- time/mesh/solver settings
- scheduled components
- compare variables
- capability tier
- restart provenance
- output artifact paths
- variable min/max/mean/delta summaries

The verbose run-log JSON now also stores the ordered event stream, so a downstream plotting or workflow script can reconstruct what happened during the run.
The same JSON also stores sanitized working-directory and machine/runtime metadata so a saved run can be audited later without leaking workstation-specific absolute paths.
It now also carries `event_count` and `event_stages`.

In practice, the detailed runtime stream now covers:

- configuration loading
- restart loading
- native run launch/completion
- artifact destination resolution
- per-artifact write completion
- final run summary

## Restart / Resume

To resume from a saved restart bundle from the CLI:

```bash
dkx input.toml \
  --output-dir /tmp/dkx_run_resume \
  --restart-in /tmp/dkx_run/<case>_restart.npz \
  --resume-steps 2
```

The same workflow can also be encoded in the deck:

```toml
[restart]
input = "output/restartable_diffusion/restartable_diffusion_restart.npz"
resume_steps = 2
```

The runnable tutorial for the full flow is:

- [examples/restartable_diffusion_tutorial.py](examples/restartable_diffusion_tutorial.py)

And the simplest shipped example deck is:

- [examples/inputs/restartable_diffusion.toml](examples/inputs/restartable_diffusion.toml)
