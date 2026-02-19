# Geometry Models (Analytic)

This document covers the **analytic axisymmetric geometry models** that feed the unified DRB system.
These models produce the field-aligned coefficients
`curv_x`, `curv_y`, `dpar_factor`, and `B` used by the core equations.

The guiding principle is **no proxy physics**: the analytic models implement the same curvature
and parallel-derivative structure used by the unified DRB system and are compatible with
metric-derived coefficients from external grid tools.

**Source code (clickable)**
- `/Users/rogerio/local/jax_drb/src/jaxdrb/core/geometry_axisymmetric_analytic.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/core/geometry_field_aligned.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/core/geometry_logb.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/core/geometry_registry.py`

**Model List**
- `salpha` (analytic s-alpha, with optional log-B curvature)
- `miller` (Miller surface parameterization)
- `xpoint_psi76` (analytic X-point field line from Eq. 76)

---

**S-Alpha Log-B Curvature Model**

This model is intended to match the curvature derived from **metric-based grids** that store
`logB` in ballooning coordinates. The core definition is:

\[
\log B(\theta) = -\epsilon \cos(\theta_{\mathrm{eff}})
\]

\[
\theta_{\mathrm{eff}} = \theta \, \frac{q}{q_{\mathrm{eff}}}
\]

\[
\frac{\partial \log B}{\partial z} = \frac{\partial \log B}{\partial \theta}\,\frac{\partial \theta}{\partial z}
\]

\[
\kappa_x = -B\,\partial_z \log B, \quad \kappa_y = B\,\partial_x \log B
\]

By default, the parallel derivative factor is
\[
\nabla_\parallel = \mathrm{dpar\_factor}\,\partial_z, \quad \mathrm{dpar\_factor} = \frac{\theta_\mathrm{scale}}{q R_0}
\]

**Ballooning (theta) correction**

When `theta_ballooning_on = true`, a local ballooning correction is applied by using
an effective safety factor at a reference radius `r_ref`:

\[
q_{\mathrm{eff}} = q + \hat{s}\,\frac{r_{\mathrm{ref}} - r_0}{r_0}
\]

This changes the effective field-line angle via \(\theta_{\mathrm{eff}} = \theta (q/q_{\mathrm{eff}})\).
The reference radius is set by `theta_ballooning_r` if provided, or defaults to `r0`.

**Linear shear correction**

When `linear_shear_on = true`, the radial dependence of \(q\) enters the x-derivative of
\(\theta\):

\[
\frac{\partial \theta}{\partial x} = -\frac{\theta_{\mathrm{eff}}}{q_{\mathrm{eff}}}\,\frac{\partial q}{\partial x},
\quad \frac{\partial q}{\partial x} = \frac{\hat{s}}{r_0}
\]

This correction adds the shear contribution to \(\partial_x \log B\) and therefore to \(\kappa_y\).

**Optional epsilon gradient**

If `epsilon_x_grad` is provided, it sets the explicit radial gradient used in
\(\partial_x \log B\). Otherwise the default is \(\epsilon_x\_\mathrm{grad} = 1/R_0\).

**Parameters**
- `theta_ballooning_on`: enable the \(q_{\mathrm{eff}}\) correction
- `theta_ballooning_r`: reference radius for the correction
- `linear_shear_on`: enable the \(\partial_x \theta\) term
- `epsilon_x_grad`: explicit radial gradient for \(\log B\)
- `r0`: reference minor radius (default `epsilon * R0`)
- `B0`: constant field strength (used in log-B model)

---

**Canonical Mapping (Axis Conventions)**

For comparison against metric-derived grids, the canonical mapping assumes:
- `x`: radial coordinate
- `y`: binormal (poloidal) coordinate
- `z`: field-aligned (ballooning) coordinate

Curvature mapping:
\[
\kappa_x = -B\,\partial_z \log B, \quad \kappa_y = B\,\partial_x \log B
\]

This is exposed through the comparison scripts as a **canonical mapping preset**:

```
--mapping canonical
```

This is the recommended default for s-alpha + log-B comparisons.

---

**Metric-Derived Curvature and Parallel Factor**

When comparing against metric-derived grids (e.g., Hermes/BOUT-style files), the curvature and
parallel derivative factor used in the comparison tools follow:

\[
\kappa_x = -B\,\partial_z \log B,\quad \kappa_y = B\,\partial_x \log B
\]

with optional perpendicular metric scaling:

\[
\kappa_x \leftarrow -B\,\sqrt{g_\perp}\,\partial_z \log B,\quad
\kappa_y \leftarrow B\,\sqrt{g_{xx}}\,\partial_x \log B
\]

where \(g_\perp = g_{yy} - g_{xy}^2 / g_{xx}\). The parallel derivative factor is derived from
metric data as:

\[
\mathrm{dpar\_factor} = \frac{B_{p}}{B\,h_\theta}
\]

These definitions align the analytic coefficients with metric outputs used in
GBS/Hermes-style grids. See the local references below for geometry conventions and
ballooning coordinate context.

---

**Design Decisions**

- Analytic models are **pure coefficient generators**: once coefficients are built, the unified DRB
  equations do not depend on which model produced them.
- The log-B curvature option is included specifically to match **metric-derived** curvature
  from ballooning coordinate grids without introducing proxy equations.
- Ballooning and linear-shear corrections are optional toggles; turning them on/off produces
  physically meaningful subsets without branching the model equations.

---

**Boundary Policy Windows (Region Masks)**

Analytic axisymmetric geometries support **region masks** based on the field-line angle
\(\theta\), which are used to tag core/SOL/divertor legs without splitting the equations.
Regions can specify either a single window or multiple windows:

- `theta_window = [theta_min, theta_max]`
- `theta_windows = [[theta_min, theta_max], [theta_min2, theta_max2], ...]`

These masks are consumed by the SOL source/sink logic and by region‑policy BCs, enabling
open/closed‑field‑line mixes in a single simulation.

**Numerics & Performance**

- All coefficient builders are written in JAX and operate on full arrays; they are JIT-safe
  and do not include Python loops over grid points.
- Coefficients are computed once and reused; downstream operators are vectorized over
  the perpendicular plane.

**Implementation Notes**

- The analytic models are **pure coefficient generators**. Once `curv_x`, `curv_y`, `dpar_factor`, and `B`
  are computed, the unified DRB system uses the same operators in 1D/2D/3D.
- The geometry code is written in JAX and vectorized; all formulas are array-based with no Python loops.
- These coefficients can be used directly or compared against metric-derived grids from external tools.

---

**References (Local PDFs)**
- [Ricci 2012, PPCF (GBS geometry and s-alpha)](/Users/rogerio/local/tests/drb_literature/Ricci_2012_Plasma_Phys._Control._Fusion_54_124047.pdf)
- [Halpern 2013, Nuclear Fusion (ballooning / s-alpha context)](/Users/rogerio/local/tests/drb_literature/Halpern_2013_Nucl._Fusion_53_122001.pdf)
- [Stegmeir 2018, PPCF (FCI / limiter context)](/Users/rogerio/local/tests/drb_literature/Stegmeir_2018_Plasma_Phys._Control._Fusion_60_035005.pdf)
- [2112.03573v1 (analytic X-point flux function, Eq. 76)](/Users/rogerio/local/tests/drb_literature/2112.03573v1.pdf)
