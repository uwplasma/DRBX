# Flux-coordinate independent (FCI) preparation

These scripts are a *preparatory milestone* for using the flux-coordinate independent (FCI) approach
to model X-points and island divertors.

They focus on a **structured perpendicular grid** and a **field-line map + interpolation** approach
to build parallel derivatives without relying on flux coordinates in the perpendicular plane.

References (see `drb_literature/fci_approach/`):

- Hariri et al. (2014), *The flux-coordinate independent approach applied to X-point geometries*
- Stegmeir et al. (2018), *GRILLIX: a 3D turbulence code based on the flux-coordinate independent approach*

Scripts:

- `fci_slab_parallel_derivative_mms.py`: MMS-style convergence test for the parallel derivative in a slab,
  using an analytic constant-$B$ map.

