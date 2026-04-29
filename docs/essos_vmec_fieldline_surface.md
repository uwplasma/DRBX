# ESSOS Field-Line And VMEC Surface Registration

This gate independently traces Landreman-Paul QA field lines and overlays the
resulting Poincare points on the same VMEC Fourier surfaces used by the
imported FCI geometry and movie campaigns. It has two modes. The default
`coil` mode traces the imported coil field from scaled VMEC seed surfaces. The
`vmec` mode traces the VMEC-coordinate equilibrium field and is the
surface-preserving registration check used to verify that the rendered
Landreman-Paul QA boundary is non-axisymmetric and consistent with the
`vmec_jax --plot` convention.

Regenerate it with:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/vmec_fieldline_surface_campaign.py
```

Regenerate the VMEC-coordinate control gate with:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/vmec_fieldline_surface_campaign.py \
  --field-source vmec \
  --output-root docs/data/essos_vmec_equilibrium_fieldline_surface_artifacts \
  --case-label essos_vmec_equilibrium_fieldline_surface_campaign
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

The VMEC-coordinate control mode closes the separate geometry-registration
question. It traces field lines in `(s, theta, phi)` coordinates, where the
VMEC contravariant field has `B^s = 0`, maps section crossings back to the
sampled Fourier surface, and confirms that the field-line hits remain on the
same seeded surface. This mode is not used to claim that the imported coil
field is VMEC-surface-preserving; it verifies that the movie boundary itself is
the expected non-axisymmetric Landreman-Paul QA surface.

Current coil-trace report highlights:

- non-axisymmetric VMEC major-radius RMS: about `0.116`;
- Poincare point count: `1024`;
- same-seeded-surface 95th-percentile distance: about `1.84` outer-surface minor extents;
- nearest-reference-surface 95th-percentile distance: about `1.35` outer-surface minor extents;
- strict field-line/surface match flag: `false`;
- diagnostic pass flag: `true`.

![ESSOS field-line/VMEC surface registration](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_vmec_fieldline_surface_artifacts__images__essos_vmec_fieldline_surface_campaign.png)

Current VMEC-coordinate control report highlights:

- Poincare point count: `2171`;
- same-seeded-surface 95th-percentile distance: about `7.6e-3` outer-surface
  minor extents;
- nearest-reference-surface 95th-percentile distance: about `7.6e-3`
  outer-surface minor extents;
- VMEC radial-coordinate drift: `0`;
- strict field-line/surface match flag: `true`;
- diagnostic pass flag: `true`.

![ESSOS VMEC equilibrium field-line/VMEC surface registration](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_vmec_equilibrium_fieldline_surface_artifacts__images__essos_vmec_equilibrium_fieldline_surface_campaign.png)

## Artifact Files

- `docs/data/essos_vmec_fieldline_surface_artifacts/data/essos_vmec_fieldline_surface_campaign.json`
- `docs/data/essos_vmec_fieldline_surface_artifacts/data/essos_vmec_fieldline_surface_campaign.npz`
- `docs/data/essos_vmec_fieldline_surface_artifacts/images/essos_vmec_fieldline_surface_campaign.png`
- `docs/data/essos_vmec_equilibrium_fieldline_surface_artifacts/data/essos_vmec_equilibrium_fieldline_surface_campaign.json`
- `docs/data/essos_vmec_equilibrium_fieldline_surface_artifacts/data/essos_vmec_equilibrium_fieldline_surface_campaign.npz`
- `docs/data/essos_vmec_equilibrium_fieldline_surface_artifacts/images/essos_vmec_equilibrium_fieldline_surface_campaign.png`
