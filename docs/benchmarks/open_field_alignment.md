# Open-Field Short-Window Alignment

This benchmark aligns `jax_drb` and Hermes on a short open-field window before
extending to longer turbulence runs.

## 1) Run jax_drb (normalization-enabled)

```bash
cd <repo-root>
PYTHONPATH=src python -m jaxdrb.cli.main \
  examples/open_field_line/input_tokamak_bxcv_short_align_norm.toml \
  --run \
  --output <run-dir>/jaxdrb_open_field_tokamak_bxcv_short_align_norm.npz
```

## 2) Extract Hermes RMS + fluctuation RMS

```bash
cd <repo-root>
PYTHONPATH=src python tools/extract_hermes_rms.py \
  --data-dir <run-dir>/hermes_tokamak_turb_short/data \
  --out <run-dir>/hermes_tokamak_turb_short_rms.npz
```

## 3) Compare fluctuation RMS (default)

```bash
cd <repo-root>
PYTHONPATH=src python tools/compare_short_rms.py \
  --hermes <run-dir>/hermes_tokamak_turb_short_rms.npz \
  --jax <run-dir>/jaxdrb_open_field_tokamak_bxcv_short_align_norm.npz \
  --metric fluct \
  --out-plot <run-dir>/hermes_jax_short_rms_fluct.png \
  --out-csv <run-dir>/hermes_jax_short_rms_fluct.csv
```

## 4) Constrained poisson_scale scan (finite + spike gate)

```bash
cd <repo-root>
PYTHONPATH=src python tools/scan_poisson_scale.py \
  --config examples/open_field_line/input_tokamak_bxcv_short_align_norm.toml \
  --scales "1e-5,3e-5,1e-4,3e-4,1e-3" \
  --target-rms <run-dir>/hermes_tokamak_turb_short_rms.npz \
  --dt 5e-5 \
  --nsteps 300 \
  --max-growth-factor 200 \
  --max-rms-abs 20 \
  --out-csv <run-dir>/poisson_scale_scan_short.csv
```

Only scales that are finite and below spike thresholds should be used to
extend runs beyond `t=1.0`.

Growth gating uses the second saved sample as the reference level to avoid
false rejections from channels that start at exactly zero by construction.

## Current Short-Window Status

- Constrained scan in the current aligned setup selects `poisson_scale = 1e-3`
  as the best finite candidate under short-window gates.
- Comparison metric is fluctuation RMS (`*_fluct`) rather than total RMS.
- The benchmark figure is tracked in:
  - `docs/figures/tokamak_sol_benchmark_fluct_rms.png`
  - `docs/figures/tokamak_sol_benchmark_fluct_rms.csv`

This short-window gate must pass before extending to longer nonlinear
open-field runs.

## 5) Long-Window Extension (`t=1.0`)

```bash
cd <repo-root>
PYTHONPATH=src python -m jaxdrb.cli.main \
  examples/open_field_line/input_tokamak_bxcv_t1_align_norm.toml \
  --run \
  --output <run-dir>/jaxdrb_open_field_tokamak_bxcv_t1_align_norm.npz
```

If comparing against historical outputs produced before fluctuation channels
were exported directly, append `*_fluct` diagnostics with:

```bash
python - <<'PY'
import numpy as np
src = np.load("<run-dir>/jaxdrb_open_field_tokamak_bxcv_t1_align.npz", allow_pickle=True)
obj = {k: src[k] for k in src.files}
for fld in ("n", "Te", "omega", "phi"):
    s = np.asarray(obj[f"snapshots_{fld}"])
    eq = s[0]
    d = s - eq[None, ...]
    obj[f"equilibrium_{fld}"] = eq
    obj[f"rms_{fld}_fluct"] = np.sqrt(np.mean(d * d, axis=tuple(range(1, d.ndim))))
np.savez("<run-dir>/jaxdrb_open_field_tokamak_bxcv_t1_align_with_fluct.npz", **obj)
PY
```
