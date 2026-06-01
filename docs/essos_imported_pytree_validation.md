# ESSOS Imported PyTree/JVP Validation

This page documents the first fixed-layout PyTree/JVP gate driven by imported
Landreman-Paul QA field-line maps. The external geometry adapter can supply
coil-traced maps, VMEC-coordinate maps, or a hybrid map that uses VMEC
coordinates with coil endpoint masks. `jax_drb` imports those maps, builds a
compact metric from the scaled VMEC QA surface coordinates, initializes ion,
electron, and neutral fields on the imported logical grid, and advances the
JAX-native drift-reduced Braginskii RHS through a short transformable
transient.

Regenerate the campaign with:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_pytree_campaign.py
```

Set `MAP_SOURCE = "coil"`, `"vmec"`, or `"hybrid"` near the top of
`examples/geometry-3D/essos-field-lines/imported_pytree_campaign.py` to choose
the map semantics. The `coil` source is the open-field endpoint path for
sheath/recycling masks from external coil traces. The `vmec` source is a
closed-field coordinate-map control for differentiability checks when target
losses vanish. The `hybrid` source combines VMEC-coordinate interpolation with
coil endpoint masks and is the preferred bridge for non-axisymmetric SOL
closure development.

This is a differentiability and software-architecture gate, not yet a
wall-resolved stellarator edge prediction. Its purpose is to prove that the
imported non-axisymmetric FCI maps feed the same fixed-layout PyTree state
used by the native 3D RHS, and that the resulting objective is compatible
with `jax.jvp` and `jax.vmap`.

## Model Path

The imported state contains the fixed component fields
\((N_i,N_e,N_n,P_i,P_e,P_n,M_i,M_n,\Omega)\). The RHS assembly uses the same
JAX kernels as the synthetic non-axisymmetric PyTree gate:

- target endpoint masks from the imported FCI maps;
- Bohm sheath losses and exact recycled neutral accounting;
- FCI neutral diffusion, perpendicular metric diffusion, ionisation,
  recombination, and charge exchange;
- metric-weighted vorticity diffusion and compact potential inversion;
- a clipped explicit short transient used only for transformability and
  regression diagnostics.

The derivative gate defines a scalar objective from the final ion density,
neutral density, vorticity RMS, and potential residual. It compares
`jax.jvp` with a centered finite-difference derivative and checks that
batched `vmap` evaluations agree with serial evaluations on the same drive
parameter.

## Current Artifacts

![ESSOS imported PyTree/JVP coil validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_pytree_artifacts__images__essos_imported_pytree_campaign.png)

![ESSOS imported PyTree/JVP VMEC-coordinate validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_pytree_vmec_artifacts__images__essos_imported_pytree_vmec_campaign.png)

![ESSOS imported PyTree/JVP hybrid validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_pytree_hybrid_artifacts__images__essos_imported_pytree_hybrid_campaign.png)

The figures show the short imported-map transient, endpoint distribution,
final ion and neutral sections, serial-versus-batched objective parity, and
JVP derivative comparison for `coil`, `vmec`, and `hybrid` maps. The closed
VMEC-coordinate control has zero endpoint fraction and a much smaller compact
potential residual; the hybrid map keeps the open-field endpoint fraction and
passes the same JVP and `vmap` gates as the coil-traced map. The current
reports record imported map resolution, endpoint fraction, magnetic-field
modulation, warm execution time, JVP relative error, `vmap` serial mismatch,
and final density/vorticity diagnostics.

## Artifact Files

- `docs/data/essos_imported_pytree_artifacts/data/essos_imported_pytree_campaign.json`
- `docs/data/essos_imported_pytree_artifacts/data/essos_imported_pytree_campaign.npz`
- `docs/data/essos_imported_pytree_artifacts/images/essos_imported_pytree_campaign.png`
- `docs/data/essos_imported_pytree_vmec_artifacts/data/essos_imported_pytree_vmec_campaign.json`
- `docs/data/essos_imported_pytree_vmec_artifacts/data/essos_imported_pytree_vmec_campaign.npz`
- `docs/data/essos_imported_pytree_vmec_artifacts/images/essos_imported_pytree_vmec_campaign.png`
- `docs/data/essos_imported_pytree_hybrid_artifacts/data/essos_imported_pytree_hybrid_campaign.json`
- `docs/data/essos_imported_pytree_hybrid_artifacts/data/essos_imported_pytree_hybrid_campaign.npz`
- `docs/data/essos_imported_pytree_hybrid_artifacts/images/essos_imported_pytree_hybrid_campaign.png`
