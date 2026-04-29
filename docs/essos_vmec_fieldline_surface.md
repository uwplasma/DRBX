# ESSOS Field-Line And VMEC Surface Registration

This gate independently traces Landreman-Paul QA coil field lines from scaled
VMEC seed surfaces and overlays the resulting Poincare points on the same
scaled VMEC Fourier surfaces used by the imported FCI geometry and movie
campaigns.

Regenerate it with:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/vmec_fieldline_surface_campaign.py
```

The diagnostic deliberately separates two questions. The first is whether the
adapter can reproduce a non-axisymmetric Landreman-Paul QA VMEC boundary and
trace finite coil-field Poincare points from that seed shell. That gate passes.
The second is whether the coil-field Poincare points remain on the seeded VMEC
surface. That stricter closed-surface match is reported separately and is
currently false for this imported coil file: the field-line points leave the
seed shell by order-unity fractions of the outer-surface minor extent over the
long trace. The movie therefore uses the VMEC Fourier surface as the visual
boundary and the imported coil traces as FCI maps, without claiming that the
coil field is an exact VMEC-surface-preserving equilibrium.

Current report highlights:

- non-axisymmetric VMEC major-radius RMS: about `0.116`;
- Poincare point count: `1024`;
- same-seeded-surface 95th-percentile distance: about `1.84` outer-surface minor extents;
- nearest-reference-surface 95th-percentile distance: about `1.35` outer-surface minor extents;
- strict field-line/surface match flag: `false`;
- diagnostic pass flag: `true`.

![ESSOS field-line/VMEC surface registration](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_vmec_fieldline_surface_artifacts__images__essos_vmec_fieldline_surface_campaign.png)

## Artifact Files

- `docs/data/essos_vmec_fieldline_surface_artifacts/data/essos_vmec_fieldline_surface_campaign.json`
- `docs/data/essos_vmec_fieldline_surface_artifacts/data/essos_vmec_fieldline_surface_campaign.npz`
- `docs/data/essos_vmec_fieldline_surface_artifacts/images/essos_vmec_fieldline_surface_campaign.png`
