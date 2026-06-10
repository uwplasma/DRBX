# Connection Length

Connection length is one of the basic geometric quantities in scrape-off-layer
(SOL) physics. The SOL is the open-field-line region outside the closed
confinement volume: particles and energy leave the confined plasma mostly by
rapid transport along magnetic field lines and are intercepted by a limiter,
divertor plate, first wall, or another plasma-facing component. This makes the
distance measured along the magnetic field, not the straight-line Euclidean
distance, the relevant length scale for parallel heat conduction, sheath
closure, neutral recycling, and divertor heat-load interpretation. Standard SOL
reviews emphasize that open field lines connect the edge plasma to material
surfaces and set the exhaust channel; see Fundamenski's JET SOL review and the
IPP overview of SOL heat-exhaust physics:
[Fundamenski 2007](https://scientific-publications.ukaea.uk/wp-content/uploads/Published/FusionSTVOL53p1023.pdf),
[IPP SOL overview](https://www.ipp.mpg.de/5458169/pds). A concise definition
is also given in the CIEMAT FusionWiki entry:
[Connection length](https://wiki.fusion.ciemat.es/wiki/Connection_length).

## Definitions

Let \(\mathbf{x}(\ell)\) be a magnetic field line parameterized by arc length
\(\ell\), with

\[
\frac{d\mathbf{x}}{d\ell} = \mathbf{b}(\mathbf{x})
  = \frac{\mathbf{B}(\mathbf{x})}{|\mathbf{B}(\mathbf{x})|}.
\]

For a starting point \(\mathbf{x}_0\), the one-sided forward and backward
connection lengths are

\[
L_+(\mathbf{x}_0)
  = \int_0^{\ell_+} d\ell,
\qquad
L_-(\mathbf{x}_0)
  = \int_0^{\ell_-} d\ell,
\]

where \(\ell_+\) and \(\ell_-\) are the first positive arc lengths at which the
forward and backward field-line integrations intersect a material boundary or
leave the computational SOL domain. A target-to-target connection length is the
distance along the same field line between two material intersections,

\[
L_{ct} = L_+ + L_-.
\]

Many reduced SOL closures use \(L_\parallel\) for the effective parallel
distance between the modeled perpendicular plane and a sheath/target. Depending
on the model, \(L_\parallel\) may be one-sided, half of a target-to-target
length, or a prescribed constant. For example, the sheath-connected blob
example in the reference fluid-model documentation uses
\(\nabla\cdot\mathbf{j}_{sh}=n_e\phi/L_{||}\) with a configurable
`connection_length`:
[Hermes-3 blob example](https://hermes3.readthedocs.io/en/latest/examples.html#blob2d).
In axisymmetric tokamak estimates near a poloidal limiter or divertor,
\(L_\parallel\) often scales like the distance traveled along a helical field
line, approximately \(qR\) times an order-one poloidal-angle factor. In shaped,
diverted, or non-axisymmetric geometry, this approximation is replaced by field
line tracing.

When toroidal angle \(\phi\) is used as the integration coordinate, a field
line in cylindrical coordinates satisfies

\[
\frac{dR}{d\phi} = R\frac{B_R}{B_\phi},
\qquad
\frac{dZ}{d\phi} = R\frac{B_Z}{B_\phi},
\qquad
\frac{d\ell}{d\phi} = R\frac{|\mathbf{B}|}{|B_\phi|}.
\]

In flux coordinates, the same idea is written with contravariant components.
For a VMEC-like fixed-flux-surface map with \(s\) held constant,

\[
\frac{d\theta}{d\phi} = \frac{B^\theta}{B^\phi},
\]

and the arc length over a toroidal step is computed from the metric along the
mapped curve. The important point is that \(L_c\) is a property of field-line
topology and wall intersection, while \(L_\parallel\) in a reduced model is the
particular effective length inserted into a closure.

## Why It Matters

Connection length controls parallel transport time scales. A short connection
length tends to couple a point strongly to a nearby sheath or wall, while a long
connection length gives more distance for parallel conduction, radiation,
ionization, charge exchange, and cross-field diffusion to modify the plasma
before it reaches a target. W7-X connection-length studies explicitly connect
\(L_c\) to SOL temperature profiles, high-recycling access, and heat-flux
spreading:
[W7-X connection-length study](https://scipub.euro-fusion.org/wp-content/uploads/eurofusion/WPS1PR17_18106_submitted.pdf).
Recent stellarator divertor optimization work uses target-to-target connection
length as a heat-load design metric and notes that long island-divertor
connection lengths help spread heat flux:
[stellarator island-divertor optimization](https://arxiv.org/html/2602.24049v1).

Historically, tokamak SOL modeling often reduced the parallel direction to a
one-dimensional or two-point model along a representative field line. That is
still useful for closures and scaling arguments. Stellarator and resonant
magnetic perturbation geometries made the full three-dimensional connection
map unavoidable: islands, stochastic layers, and localized wall intersections
produce large spatial variation in \(L_c\). This is why modern 3D boundary
workflows use field-line tracing, magnetic meshes, and repeated map
reconstruction. FLARE and magnetic-mesh work for EMC3-EIRENE describe this
field-line reconstruction viewpoint for stellarator SOL/divertor modeling:
[FLARE field-line analysis](https://arxiv.org/html/2402.05225v1),
[magnetic mesh generation for stellarators](https://arxiv.org/html/2410.01139v1).

## Calculation In JAXDRB

JAXDRB keeps the connection-length calculation separate from the drift-reduced
Braginskii RHS. The RHS consumes an array called `connection_length`; geometry
adapters decide how that array was produced and document whether it is a true
wall-hit length, an adjacent-plane arc length, or a proxy.

The synthetic self-contained stellarator lane uses a controlled proxy in
[`src/jax_drb/geometry/stellarator.py`](../src/jax_drb/geometry/stellarator.py).
`_estimate_connection_length(...)` repeatedly applies the forward FCI map,
counts how many toroidal-plane steps remain inside the radial domain, and
multiplies the bounded step count by an average metric step length. This is not
a wall-hit trace. It is a deterministic non-axisymmetric proxy used for
clean-clone examples, metric/operator tests, and qualitative turbulence
weighting.

The imported-field lane in
[`src/jax_drb/geometry/essos_import.py`](../src/jax_drb/geometry/essos_import.py)
has three map-source semantics:

- `coil`: field lines are traced with the external coil-field runtime. JAXDRB
  interpolates forward/backward endpoints at adjacent toroidal planes and uses
  the traced exit length to the boundary when that exit length is finite.
  Otherwise it falls back to the bidirectional adjacent-plane arc length.
- `vmec`: JAXDRB integrates a VMEC-coordinate map using
  \(d\theta/d\phi=B^\theta/B^\phi\). Because this map stays on closed VMEC
  surfaces, it stores a bidirectional adjacent-plane arc length and has no
  sheath endpoint masks.
- `hybrid`: JAXDRB uses the smooth VMEC map coordinates but keeps the
  coil-derived endpoint masks, \(|B|\), and connection-length array. This is
  the current bridge for open-field SOL closure tests on a smooth
  non-axisymmetric interpolation map.

The imported validation campaign in
[`src/jax_drb/validation/essos_imported_fci_campaign.py`](../src/jax_drb/validation/essos_imported_fci_campaign.py)
checks the connection-length arrays before they are used for physics claims.
`build_essos_imported_fci_map_diagnostics(...)` verifies finite and
nonnegative values, records radial means, and computes single-grid neighbor
jump diagnostics. `build_essos_imported_connection_length_refinement_diagnostics(...)`
then compares nested grids by conservatively restricting each fine grid to the
coarser grid and reporting normalized RMS, 95th-percentile, and
\(L_\infty\) errors plus an observed order when three or more levels are
available. The clean-clone example
[`examples/geometry-3D/essos-field-lines/imported_connection_length_refinement_demo.py`](../examples/geometry-3D/essos-field-lines/imported_connection_length_refinement_demo.py)
exercises that exact refinement path on manufactured non-axisymmetric data.

## Current Promotion Rule

Connection-length arrays are accepted for publication-grade imported-field
turbulence only when all of the following are true:

- Values are finite and nonnegative on the consumed FCI grid.
- Endpoint masks used by sheath/recycling exactly match the imported
  forward/backward map boundary masks.
- Single-grid roughness diagnostics do not show unresolved grid-scale jumps.
- A live multi-grid coil, VMEC, or hybrid refinement campaign passes the same
  conservative restriction test currently demonstrated by the manufactured
  clean-clone example.
- The turbulence figure or movie reports which connection-length definition was
  used: wall-hit/exit length, adjacent-plane arc length, or synthetic proxy.

Until the live multi-grid imported run passes, synthetic and manufactured
connection-length figures are validation evidence for the algorithms, not a
claim that a particular stellarator wall/target design has been resolved.
