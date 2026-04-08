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
- `[mesh]`
- `[solver]`
- `[model]`
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
  --output-dir /tmp/jax_drb_run
```

This writes:

- `/tmp/jax_drb_run/<case>_summary.json`
- `/tmp/jax_drb_run/<case>_arrays.npz`
- `/tmp/jax_drb_run/<case>_restart.npz`
- `/tmp/jax_drb_run/<case>_run_log.json`

The terminal summary is rich-formatted when `rich` is available and falls back to plain text otherwise. Both versions report the same core metadata:

- input file
- case name
- runtime precision
- time/mesh/solver settings
- scheduled components
- compare variables
- restart provenance
- output artifact paths

## Restart / Resume

To resume from a saved restart bundle:

```bash
jax_drb input.toml \
  --output-dir /tmp/jax_drb_run_resume \
  --restart-in /tmp/jax_drb_run/<case>_restart.npz \
  --resume-steps 2
```

The runnable tutorial for the full flow is:

- [examples/restartable_diffusion_tutorial.py](/Users/rogerio/local/jax_drb/examples/restartable_diffusion_tutorial.py)

And the simplest shipped example deck is:

- [examples/inputs/restartable_diffusion.toml](/Users/rogerio/local/jax_drb/examples/inputs/restartable_diffusion.toml)
