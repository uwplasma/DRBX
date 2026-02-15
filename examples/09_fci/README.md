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
- `fci_target_aware_parallel_derivative_mms.py`: MMS-style convergence test for the **target-aware**
  parallel derivative on open field lines with Dirichlet plates (cell-centered planes).
- `fci_hello_world.py`: simple visualization of the FCI map, centered parallel derivative, and line-integral
  mapping on a constant-$B$ slab.
- `fci_curved_map_regression.py`: convergence check for a spatially varying in-plane field (Bx varies with y).
- `fci_drb3d_sheath_budget.py`: minimal 3D DRB slab operator with sheath damping (energy/mass decay).
- `fci_drb3d_full_operator_wallbc_stats.py`: full 3D DRB milestone diagnostics for FD/FV wall-BC behavior
  and long-time turbulence regression metrics (zonal fraction + fluctuation levels).
- `fci_drb3d_full_multiphysics_sheath.py`: full DRB3D branch with target/sheath coupling plus hot-ion,
  EM, and neutrals toggles; reports explicit sheath and particle-budget diagnostics.
- `fci_drb3d_full_essos_biotsavart.py`: full DRB3D run on ESSOS Landreman-Paul QA coils using a local
  toroidal-plane FCI map; includes a practical axis estimate, target-hit metadata, and edge/SOL patch diagnostics.
- `fci_drb3d_full_movie_open_closed.py`: open-field-line slab with a smooth LCFS mask that gates
  sheath losses (open SOL vs closed core) to explore blob-like intermittency in 3D.
