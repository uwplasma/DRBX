# Diagnostics

`jax_drb` ships **built‑in diagnostics** for spectra, PDFs, and zonal averages.
These are used by the plotting scripts in `tools/` and are fully reproducible.

## Spectra

The 2D power spectrum uses FFTs with optional detrending and windowing. For
isotropic spectra we bin in |k| shells.

Python:

```python
from jaxdrb.diagnostics import isotropic_spectrum

spec = isotropic_spectrum(field, dx=dx, dy=dy)
```

## PDFs

PDFs are computed from mean‑subtracted fluctuations:

```python
from jaxdrb.diagnostics import pdf_1d

centers, hist = pdf_1d(field - field.mean(), bins=80)
```

## Zonal Averages

Zonal means are defined by averaging over the binormal axis:

```python
from jaxdrb.diagnostics import zonal_mean

zonal = zonal_mean(field, axis=1)
```

## Plotting Scripts

The public examples call these utilities via the following scripts:

- `tools/plot_spectra.py`
- `tools/plot_pdf.py`
- `tools/plot_zonal_profile.py`
- `tools/plot_zonal_flow.py`
- `tools/plot_poloidal_plane.py`
- `tools/plot_3d_slices.py`
- `tools/make_movie.py`

These scripts read `.npz` output and generate the figures included in the
documentation.
