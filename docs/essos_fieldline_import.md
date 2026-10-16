# ESSOS Coil Fields

DRBX evaluates stellarator vacuum fields directly from coils with the
installed [ESSOS](https://github.com/uwplasma/ESSOS) package
(`pip install 'drbx[essos]'`). ESSOS owns the coil representation, the
Biot--Savart field and the adaptive field-line tracer; DRBX owns the
conversion of those fields into FCI maps, masks and metrics.

## Field providers

`drbx.geometry.field_provider` defines one protocol for every magnetic-field
source:

```python
class MagneticFieldProvider(Protocol):
    nfp: int                 # field periods
    stellsym: bool
    provenance: dict         # source file name, kind, currents
    def B(self, xyz): ...    # Cartesian B [T] at points (..., 3) [m]
    def inside(self, xyz): ...  # validity mask of the field
```

Providers are JAX pytrees whose leaves are their parameters, so
`jax.jvp`/`jax.grad` through `provider.B` differentiate with respect to coil
shapes and currents, and `grad_B(provider, xyz)` gives \(\partial B_i/\partial x_j\)
by forward-mode AD.

`EssosCoilField.from_json(path)` wraps an ESSOS `Coils` JSON and evaluates
`essos.fields.BiotSavart`. The Landreman--Paul QA coil set is bundled with
DRBX (`resolve_essos_landreman_qa_json()`), so no ESSOS checkout is needed.

```python
from drbx.geometry import EssosCoilField, resolve_essos_landreman_qa_json
field = EssosCoilField.from_json(resolve_essos_landreman_qa_json())
B = field.B(xyz)
```

## Field lines and FCI maps

`trace_essos_coil_field_lines(...)` traces coil field lines with
`essos.dynamics.Tracing` and returns trajectories, Poincare points and field
samples. `build_essos_imported_fci_geometry(map_source=...)` builds FCI maps
on scaled Landreman--Paul QA VMEC surfaces, with three map sources:

- `coil`: maps traced in the coil field; field lines that leave the shell
  give open endpoints with target-exit lengths.
- `vmec`: closed, surface-preserving maps from the VMEC equilibrium.
- `hybrid`: VMEC map coordinates with the coil-derived endpoint masks.

The VMEC-surface grid needs the QA `wout` file, which is not bundled: set
`DRBX_ESSOS_ROOT` to an ESSOS checkout (the checkout is used for input files
only; the installed `essos` package is always imported when present).
`essos_imported_geometry_to_fci(...)` converts the result to a native
`FciGeometry3D`, optionally opening a scrape-off layer with a limiter radius.

```bash
PYTHONPATH=src python examples/geometry-3D/essos-field-lines/landreman_paul_qa_import.py
```

## Checks

`tests/test_essos_coil_field.py` asserts, on the installed ESSOS:
divergence- and curl-free coil field (relative to \(|\nabla B|\), below
\(10^{-8}\)); field-period and stellarator symmetry to \(10^{-10}\); the
JVP with respect to all coil degrees of freedom against a central finite
difference; finite traced field lines with Poincare crossings; and the
coil/VMEC/hybrid map semantics. `tests/test_landreman_paul_turbulence.py`
checks the traced rotational transform (about 0.42) and a short four-field
run with a sheath sink on the imported geometry.
