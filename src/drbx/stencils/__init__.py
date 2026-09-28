"""One-time stencil setup for the perpendicular operators, built after geometry.

Field-independent preparation that turns a prepared FCI geometry into the stencils
the perpendicular operators apply: the face census, geometry coefficients at the
stencil nodes, and the per-grid row artifact (donors and weights for every point,
side, Neumann and integrated-face row). Nothing here depends on field values;
runtime application lives in ``drbx.native``. See the P08 execution plan in
``dev_docs/perpendicular_second_order_roadmap.md``.
"""
