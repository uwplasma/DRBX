# Native Runtime CLI

`jax_drb` now has a standalone native runtime surface for supported inputs. After an editable install,

```bash
pip install -e .[dev,validation]
```

both of these console commands are available:

```bash
jax_drb path/to/input.toml
jax-drb path/to/input.toml
```

The bare input-file form is equivalent to:

```bash
jax_drb run path/to/input.toml
```

## Input Layout

The native CLI accepts the original BOUT-style `.inp` files and now also accepts organized TOML decks. The intended TOML layout is:

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
jax_drb input.toml --precision float32
```

Current status:

- `float64` is the default and the most complete runtime mode.
- `float32` now runs cleanly on the simple diffusion/restart tutorial path and no longer emits the old internal dtype-truncation warnings there.
- broader native paths still contain explicit `float64` requests internally, so `float32` should currently be treated as an opt-in performance experiment, not a parity-default production mode.

For a concrete measurement workflow, use:

```bash
PYTHONPATH=src .venv/bin/python examples/diffusion_precision_benchmark.py
```

The committed example benchmark artifacts are in:

- [docs/runtime_precision_benchmark/data/diffusion_precision_analysis.json](/Users/rogerio/local/jax_drb/docs/runtime_precision_benchmark/data/diffusion_precision_analysis.json)
- [docs/runtime_precision_benchmark/images/diffusion_precision_elapsed.png](/Users/rogerio/local/jax_drb/docs/runtime_precision_benchmark/images/diffusion_precision_elapsed.png)

On the current machine, the warm second-run `float32` diffusion path is about `1.23x` faster than `float64` (`2.096s` vs `2.584s`) on the same input.

## Runtime Output

A native run can write four main artifact types:

- summary JSON
- arrays NPZ
- restart NPZ
- verbose run-log JSON

Example:

```bash
jax_drb examples/inputs/restartable_diffusion.toml \
  --precision float32
```

If the deck includes an `[output]` section, the bare `jax_drb input.toml` form is enough. CLI flags still override the deck when you need an ad hoc run location. A typical deck-controlled output block is:

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

Both versions report the same core metadata:

- input file
- case name
- runtime precision
- runtime backend/device/cache
- runtime library and machine metadata (`jax_version`, `python_version`, platform, process id)
- time/mesh/solver settings
- scheduled components
- compare variables
- restart provenance
- output artifact paths
- variable min/max/mean/delta summaries

The verbose run-log JSON now also stores the ordered event stream, so a downstream plotting or workflow script can reconstruct what happened during the run.
The same JSON also stores the working directory and machine/runtime metadata so a saved run can be audited later without guessing the execution environment.

## Restart / Resume

To resume from a saved restart bundle from the CLI:

```bash
jax_drb input.toml \
  --output-dir /tmp/jax_drb_run_resume \
  --restart-in /tmp/jax_drb_run/<case>_restart.npz \
  --resume-steps 2
```

The same workflow can also be encoded in the deck:

```toml
[restart]
input = "output/restartable_diffusion/restartable_diffusion_restart.npz"
resume_steps = 2
```

The runnable tutorial for the full flow is:

- [examples/restartable_diffusion_tutorial.py](/Users/rogerio/local/jax_drb/examples/restartable_diffusion_tutorial.py)

And the simplest shipped example deck is:

- [examples/inputs/restartable_diffusion.toml](/Users/rogerio/local/jax_drb/examples/inputs/restartable_diffusion.toml)
