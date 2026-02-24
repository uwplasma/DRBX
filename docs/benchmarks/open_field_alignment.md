# Open-Field Hermes Alignment (Staged)

This workflow aligns `jax_drb` and Hermes on the same open-field tokamak case,
using the same normalization convention and fluctuation diagnostics.

Base alignment config:
- `examples/open_field_line/input_tokamak_bxcv_benchmark_alignment.toml`

## 1) Run staged windows with finite-run gating

```bash
cd <repo>
PYTHONPATH=src python tools/run_staged_benchmark.py \
  --config examples/open_field_line/input_tokamak_bxcv_benchmark_alignment.toml \
  --stages short:0.5,onset:1.0,saturated:3.0 \
  --max-growth-factor 200 \
  --max-rms-abs 20 \
  --out-dir runs/staged_open_field
```

Outputs per stage:
- `runs/staged_open_field/jax_<stage>.npz`
- `runs/staged_open_field/bundle_jax_<stage>.npz`

## 2) Build Hermes bundle (same normalization metadata)

```bash
cd <repo>
PYTHONPATH=src python tools/build_benchmark_bundle.py \
  --code hermes \
  --input <hermes-run>/data \
  --output runs/staged_open_field/bundle_hermes_short.npz \
  --geometry tokamak_open_field
```

## 3) Canonical side-by-side panel (shared axes/colormap)

```bash
cd <repo>
PYTHONPATH=src python tools/plot_benchmark_panel.py \
  --hermes runs/staged_open_field/bundle_hermes_short.npz \
  --jax runs/staged_open_field/bundle_jax_short.npz \
  --out docs/figures/tokamak_sol_benchmark_panel.png \
  --summary-csv docs/figures/tokamak_sol_benchmark_panel.csv
```

The panel includes:
- side-by-side fluctuation snapshots (shared colormap range)
- fluctuation RMS overlays (`n, Te, omega, phi`)
- `k_y` spectrum, frequency spectrum
- PDFs, cross-coherence/phase, radial particle flux profile

## 4) Constrained `poisson_scale` scan before longer runs

```bash
cd <repo>
PYTHONPATH=src python tools/scan_poisson_scale.py \
  --config examples/open_field_line/input_tokamak_bxcv_benchmark_alignment.toml \
  --scales "1e-5,3e-5,1e-4,3e-4,1e-3" \
  --target-rms <hermes-rms>.npz \
  --dt 5e-5 \
  --nsteps 300 \
  --max-growth-factor 200 \
  --max-rms-abs 20 \
  --out-csv runs/staged_open_field/poisson_scale_scan.csv
```

Only finite, non-spiking candidates should be used for `t > 1.0` runs.

## Notes on Physics Alignment

- Open-field + sheath (`bohm_current`) enabled in the benchmark config.
- Curvature is read from the `bxcv` tokamak grid (not a proxy field).
- Parallel transport uses conservative + limiter options (`parallel_flux_conservative=true`,
  `parallel_limiter="mc"`).
- Fluctuation diagnostics are computed against equilibrium (`t0`) in both code paths.

## Related docs

- `docs/diagnostics.md`
- `docs/normalization.md`
- `docs/validation.md`
