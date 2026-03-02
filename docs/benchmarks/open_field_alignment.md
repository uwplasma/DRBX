# Open-Field Hermes Alignment (Staged)

This workflow aligns `jax_drb` and Hermes on the same open-field tokamak case,
using the same normalization convention and fluctuation diagnostics.

## One-Command Workflow

```bash
cd <repo>
PYTHONPATH=src python tools/run_tokamak_hermes_benchmark.py \
  --jax-config examples/open_field_line/input_tokamak_bxcv_benchmark_es_cold.toml \
  --hermes-data runs/hermes_open_field_short/data \
  --out-dir runs/tokamak_benchmark_latest \
  --fig-dir docs/figures \
  --t-end-short 0.1 \
  --t-end-visual 0.12 \
  --field n
```

Outputs:
- `runs/tokamak_benchmark_latest/jax_short.npz`
- `runs/tokamak_benchmark_latest/bundle_jax_short.npz`
- `runs/tokamak_benchmark_latest/bundle_hermes_short.npz`
- `docs/figures/tokamak_sol_benchmark_panel.png`
- `docs/figures/tokamak_sol_poloidal_fluct.png`
- `docs/figures/tokamak_sol_movie.gif`
- `docs/figures/tokamak_sol_3d_movie.gif`

The benchmark panel uses the **poloidal (`x-z`) plane** with tokamak
`Rxy/Zxy` geometry mapping. If the coefficient file includes `mask_open`,
the open/closed boundary is overlaid on the snapshot row.

Base alignment config:
- `examples/open_field_line/input_tokamak_bxcv_benchmark_alignment.toml`
- Calibrated short-window config:
  `examples/open_field_line/input_tokamak_bxcv_benchmark_alignment_calibrated.toml`
- Hermes-like initial-perturbation variant:
  `examples/open_field_line/input_tokamak_bxcv_benchmark_alignment_hermes_init.toml`
- Hermes-equivalent exact IC variant (`n` mixmode + pressure-consistent `Te`):
  `examples/open_field_line/input_tokamak_bxcv_benchmark_alignment_hermes_exact_ic.toml`

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

Latest short-loop scan (calibrated config) selected:
- `poisson_scale = 2e-4`
- score `1.348` (fluctuation RMS mismatch score)
- finite gate: passed (`growth=2.01`, `peak=0.295`)

## 5) Multi-parameter parity loop (rtol target)

Use the calibration loop for staged, finite-gated scans and an explicit
`rtol` target on fluctuation RMS mismatch:

```bash
cd <repo>
PYTHONPATH=src python tools/calibrate_parity_loop.py \
  --config examples/open_field_line/input_tokamak_bxcv_benchmark_alignment_hermes_exact_ic.toml \
  --hermes-rms <hermes-rms>.npz \
  --stages 0.1,0.5,1.0 \
  --grid-short 24,32,24 \
  --omega-mults 1.0,1.1 \
  --source-mults 1.0,1.2 \
  --dn-mults 1.0 \
  --domega-mults 0.8 \
  --poisson-scales 2e-4 \
  --phi-dissipation-on 0,1 \
  --phi-sheath-dissipation-on 0,1 \
  --core-vorticity-damping-on 0,1 \
  --promote-top-k 8 \
  --rtol-target 1e-1 \
  --out-csv runs/staged_open_field/parity_scan_t01.csv
```

Recommended staged workflow:
- run `t_end=0.1` on reduced grid (`24x32x24`) to reject unstable candidates
- keep only finite candidates with lowest score
- rerun selected candidates at `t_end=0.5` on full grid
- extend to `t_end=1.0` only after `t_end=0.5` is finite and non-spiking

Current status for the best finite full-grid candidate (`t_end=0.5`):
- `Te` and `phi` fluctuation RMS are near the `rtol=1e-1` target
- `n` and `omega` fluctuation RMS remain under-predicted and require further
  term-level alignment

## Notes on Physics Alignment

- Open-field + sheath (`bohm_current`) enabled in the benchmark config.
- Curvature is read from the `bxcv` tokamak grid (not a proxy field).
- Parallel transport uses conservative + limiter options (`parallel_flux_conservative=true`,
  `parallel_limiter="mc"`).
- Fluctuation diagnostics are computed against equilibrium (`t0`) in both code paths.
- Initialization supports deterministic Hermes-style density perturbations
  (`n_mixmode_amp`, `n_mixmode_terms`) in addition to stochastic seeds.
- Short-loop calibration that reduced mismatch used:
  - radial BC: `bc_x = neumann` (geometry + perpendicular BC policy)
  - normalization-enabled physical inputs for drives/sources
  - reduced transport (`Dn=1e-3`, `DOmega=1e-4`, `DTe=1e-3`)

## Related docs

- `docs/diagnostics.md`
- `docs/normalization.md`
- `docs/validation.md`
