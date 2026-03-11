import jax
import jax.numpy as jnp

from jaxdrb.core.params import DRBSystemParams, NumericsParams
from jaxdrb.core.geometry_field_aligned import FieldAlignedGeometryAdapter


def _make_geom(mode: str) -> FieldAlignedGeometryAdapter:
    params = DRBSystemParams(
        numerics=NumericsParams(parallel_z_mode=mode),
    )
    return FieldAlignedGeometryAdapter.make_salpha(
        params=params,
        nx=6,
        ny=6,
        nz=4,
        Lx=1.0,
        Ly=1.0,
        Lz=2.0 * jnp.pi,
        bc_x="periodic",
        bc_y="periodic",
        curvature_model="vector_xy",
    )


def test_parallel_z_mode_scan_matches_vmap():
    geom_v = _make_geom("vmap")
    geom_s = _make_geom("scan")

    key = jax.random.PRNGKey(0)
    f = jax.random.normal(key, geom_v.shape())
    phi = jax.random.normal(jax.random.PRNGKey(1), geom_v.shape())

    ddx_v = geom_v.ddx(f)
    ddx_s = geom_s.ddx(f)
    lap_v = geom_v.laplacian(f)
    lap_s = geom_s.laplacian(f)
    br_v = geom_v.bracket(phi, f)
    br_s = geom_s.bracket(phi, f)

    assert jnp.allclose(ddx_v, ddx_s, atol=1e-10, rtol=1e-10)
    assert jnp.allclose(lap_v, lap_s, atol=1e-10, rtol=1e-10)
    assert jnp.allclose(br_v, br_s, atol=1e-10, rtol=1e-10)
