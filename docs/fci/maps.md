# FCI maps: data structures and file format

This page documents how `jaxdrb` represents **field-line maps** for flux-coordinate independent (FCI)
operators, and how those maps are stored on disk.

FCI requires (at minimum):

- an interpolation operator that maps a field on plane $k\pm 1$ to the plane-$k$ grid points,
- a distance-to-plane $\Delta l$ along the field line (optionally spatially varying),
- for open field lines: masks and distances to the **target intersection**.

The goal is to keep the runtime operator compact, differentiable, and JAX-friendly, while allowing
external tools to precompute maps (VMEC/ESSOS, field-line tracers, etc.).

## Runtime map object

`jaxdrb` uses a bilinear structured-plane map:

- [`src/jaxdrb/fci/map.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/map.py)
  (`FCIBilinearMap`)

Each map stores indices and weights for bilinear interpolation on a structured $(x,y)$ grid:

$$
f_{k+1}(x^+,y^+) \;\approx\; \sum_{m=1}^4 w_m\, f_{k+1}(i_m, j_m),
$$

where $(x^+,y^+)$ is the forward footpoint and $(i_m,j_m)$ are the four surrounding cell corners.

### Arrays

For a plane stack of shape `(nz, nx, ny)`:

- `ix, iy, w`: `(nz, nx, ny, 4)`
- `dl`: `(nz, nx, ny)` (distance between planes along the field line)

For plane-independent slab maps, the same arrays can be stored without the leading `nz` dimension
and broadcasted at runtime.

### Open-field-line metadata

When a field line hits a target **before** reaching the next/previous plane, the map can encode:

- `hit`: boolean mask with shape `(nz, nx, ny)`
- `dl_hit`: distance-to-target along the field line at hit points

This is used by the **target-aware** parallel derivative operator:

- [`src/jaxdrb/fci/parallel.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/parallel.py)
  (`parallel_derivative_target_aware_3d`)

which switches to a non-uniform second-order stencil near targets.

## On-disk format (`.npz`)

Maps are stored as a single compressed file containing both forward and backward maps:

- save: `jaxdrb.fci.save_fci_maps_npz`
- load: `jaxdrb.fci.load_fci_maps_npz`

Implementation:

- [`src/jaxdrb/fci/io.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/io.py)

### Keys

Required arrays:

- `fwd_ix, fwd_iy, fwd_w, fwd_dl`
- `bwd_ix, bwd_iy, bwd_w, bwd_dl`

Optional arrays:

- `fwd_hit, fwd_dl_hit`
- `bwd_hit, bwd_dl_hit`

Optional metadata:

- `meta_json`: a JSON string with provenance information (source geometry, trace settings, etc.).

The format includes `format_version=2` for forward compatibility.

Version `2` adds optional target-intersection metadata:

- `*_hit_R, *_hit_Z, *_hit_phi` (intersection coordinates)
- `*_hit_target` (integer target ID mask)

## Map builders (current milestones)

`jaxdrb` currently includes two builder paths:

### A) Cartesian z-plane builder (JAX-native)

As an early-stage, JAX-native builder, `jaxdrb` provides a **z-plane** map construction routine:

- [`src/jaxdrb/fci/builder.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/builder.py)
  (`build_fci_maps_zplanes`)

This integrates field lines using $z$ as the independent variable:

$$
\frac{dx}{dz} = \frac{B_x}{B_z},\qquad
\frac{dy}{dz} = \frac{B_y}{B_z},
$$

on a periodic $(x,y)$ plane, returning `ix/iy/w` and `dl`.

This builder is useful for:

- MMS tests that isolate target handling and convergence,
- regression tests on curved maps in periodic boxes,
- ESSOS Biot–Savart maps in a local Cartesian patch (stepping-stone toward full diverted geometries).

### B) ESSOS toroidal-plane builder (R–Z at fixed toroidal angle)

For realistic magnetic fields exposed through ESSOS `field.B(xyz)`, `jaxdrb` now provides:

- [`src/jaxdrb/fci/builder.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/builder.py)
  (`build_fci_maps_essos_toroidal_planes`, `EssosToroidalFCIConfig`)

This builder traces between toroidal planes `\phi_k -> \phi_{k+1}` using cylindrical field-line equations:

$$
\frac{dR}{d\phi} = R\,\frac{B_R}{B_\phi},\qquad
\frac{dZ}{d\phi} = R\,\frac{B_Z}{B_\phi}.
$$

It records interpolation stencils plus open-field-line metadata:

- `hit`, `dl_hit`
- `hit_R`, `hit_Z`, `hit_phi`
- `hit_target`

where a rectangular limiter/target window can be configured for intersection detection.

### Example (build + save)

```python
import jax
import jax.numpy as jnp

from jaxdrb.fci import ZPlaneFCIConfig, build_fci_maps_zplanes, save_fci_maps_npz

cfg = ZPlaneFCIConfig(
    x0=0.0, y0=0.0, dx=0.1, dy=0.1, nx=64, ny=64,
    z0=0.0, dz=0.2, nz=32,
    periodic_z=False,
)

def B(points):
    # points shape (...,3) -> B shape (...,3)
    return jnp.broadcast_to(jnp.array([0.2, 0.0, 1.0]), points.shape)

map_fwd, map_bwd = build_fci_maps_zplanes(cfg, B=B, nsub=8)
save_fci_maps_npz("my_maps.npz", map_fwd=map_fwd, map_bwd=map_bwd, meta={"builder": "zplanes"})
```

ESSOS toroidal-plane example (with target metadata) is covered by:

- test: `tests/test_fci_essos_toroidal_builder.py`

## Next steps (toward diverted tokamaks / islands)

The full FCI pipeline for X-point and island-divertor geometries requires:

- plane definition in curvilinear coordinates (often toroidal angle $\varphi$),
- robust intersection detection with wall/plates,
- target-aware interpolation/extrapolation consistent with sheath BCs,
- regression gates on interpolation error and long-time turbulence statistics.

See:

- [`docs/fci/requirements.md`](requirements.md)
- Hariri et al. (2014), DOI: [`10.1063/1.4892405`](https://doi.org/10.1063/1.4892405)
- Stegmeir et al. (2018), DOI: [`10.1088/1361-6587/aaa373`](https://doi.org/10.1088/1361-6587/aaa373)
