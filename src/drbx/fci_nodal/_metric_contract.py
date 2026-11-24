"""Fixed spline fit and exact level-set plane settings for the backend metric."""

from types import MappingProxyType


SPLINE_COARSE_FIT = MappingProxyType(dict(h=0.02, nphi=45, maxiter=600))
SPLINE_FIT = MappingProxyType(dict(
    p=5, h=0.01, nphi=90, gauss=3, rho_data=1.02, lam=1.0e-2, pad=0.02,
    atol=1.0e-10, btol=1.0e-10, conlim=1.0e12, maxiter=5000,
    monitor_every=100, monitor_rtol=2.0e-3, monitor_min_iterations=200, monitor_sample=65536,
))
SPLINE_KNOT_SNAP = 0.01
SPLINE_PHI_ORIGIN = 0.0
SPLINE_POLOIDAL_CURRENT = 0.0
BASE_MESH_SHAPE = (64, 64, 64)
FILTERED_MAP_CONSTRUCTION = "eta_filtered_series"
FILTERED_MAP_CUTOFF_KEY = "map_eta_cutoff"
BASE_RADIAL_DEGREE = 17
BASE_POLOIDAL_MODES = 15
BASE_TOROIDAL_MODES = 15
BASE_PROJECTIONS = 3
PHI_NEWTON_ITERATIONS = 8
LEVEL_SET_TOLERANCE = 1.0e-9


def spline_cache_settings():
    """Explicit tags and fixed fit settings already used by the cold spline case."""
    return dict(eta_representation="spline", plane_construction="exact",
                spline_eta=dict(SPLINE_FIT, coarse=dict(SPLINE_COARSE_FIT)))
