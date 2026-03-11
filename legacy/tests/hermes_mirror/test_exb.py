from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.core.geometry_field_aligned import FieldAlignedGeometryAdapter, FieldAlignedGrid
from jaxdrb.core.params import DRBSystemParams, NumericsParams
from jaxdrb.legacy_hermes import div_n_bxgrad_f_b_xppm_xz, div_n_bxgrad_f_b_xppm_xz_ref


def _make_geom(*, bc_x: str, scheme: str) -> FieldAlignedGeometryAdapter:
    params = DRBSystemParams(
        numerics=NumericsParams(
            exb_flux_scheme=scheme,
            exb_poloidal_flows=False,
            parallel_transform="none",
            perp_operator="fd",
            bracket="centered",
            neumann_boundary_average_y=True,
        )
    )
    grid = FieldAlignedGrid.make(
        nx=6,
        ny=7,
        nz=4,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        bc_x=bc_x,
        bc_y="periodic",
        dealias=False,
        open_field_line=True,
    )
    jacobian = jnp.broadcast_to(
        1.0 + 0.1 * jnp.arange(6, dtype=jnp.float64)[None, :, None],
        (4, 6, 7),
    )
    return FieldAlignedGeometryAdapter.from_coefficients(
        params=params,
        grid=grid,
        curv_x=0.0,
        curv_y=0.0,
        dpar_factor=1.0,
        B=1.0,
        jacobian=jacobian,
        gxx=1.0,
        gxy=0.0,
        gyy=1.0,
        g23=1.0,
    )


def test_exb_xz_ref_matches_fused_mc_periodic() -> None:
    key_phi, key_adv = jax.random.split(jax.random.PRNGKey(20))
    phi = jax.random.normal(key_phi, (4, 6, 7), dtype=jnp.float64)
    adv = jax.random.normal(key_adv, (4, 6, 7), dtype=jnp.float64)
    J = jnp.ones_like(phi)
    dx = jnp.ones_like(phi)
    dz = jnp.ones_like(phi)

    ref = div_n_bxgrad_f_b_xppm_xz_ref(
        adv,
        phi,
        jacobian=J,
        dx=dx,
        dz=dz,
        periodic_x=True,
        periodic_z=True,
        bndry_flux=True,
        use_mc=True,
    )
    fused = div_n_bxgrad_f_b_xppm_xz(
        adv,
        phi,
        jacobian=J,
        dx=dx,
        dz=dz,
        periodic_x=True,
        periodic_z=True,
        bndry_flux=True,
        use_mc=True,
    )
    np.testing.assert_allclose(np.asarray(fused), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_exb_xz_ref_matches_fused_fromm_neumann() -> None:
    key_phi, key_adv = jax.random.split(jax.random.PRNGKey(21))
    phi = jax.random.normal(key_phi, (4, 6, 7), dtype=jnp.float64)
    adv = jax.random.normal(key_adv, (4, 6, 7), dtype=jnp.float64)
    J = jnp.ones_like(phi)
    dx = jnp.ones_like(phi)
    dz = jnp.ones_like(phi)

    ref = div_n_bxgrad_f_b_xppm_xz_ref(
        adv,
        phi,
        jacobian=J,
        dx=dx,
        dz=dz,
        periodic_x=False,
        periodic_z=True,
        bndry_flux=True,
        use_mc=False,
        bc_kind_x=2,
        neumann_boundary_average_z=True,
    )
    fused = div_n_bxgrad_f_b_xppm_xz(
        adv,
        phi,
        jacobian=J,
        dx=dx,
        dz=dz,
        periodic_x=False,
        periodic_z=True,
        bndry_flux=True,
        use_mc=False,
        bc_kind_x=2,
        neumann_boundary_average_z=True,
    )
    np.testing.assert_allclose(np.asarray(fused), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_exb_xz_fused_matches_current_geometry_hermes_xppm_when_poloidal_off() -> None:
    geom = _make_geom(bc_x="neumann", scheme="hermes_xppm")
    key_phi, key_adv = jax.random.split(jax.random.PRNGKey(22))
    phi = jax.random.normal(key_phi, geom.shape(), dtype=jnp.float64)
    adv = jax.random.normal(key_adv, geom.shape(), dtype=jnp.float64)

    mirror = div_n_bxgrad_f_b_xppm_xz(
        adv,
        phi,
        jacobian=jnp.asarray(geom.jacobian, dtype=jnp.float64),
        dx=jnp.ones_like(phi) * float(geom.grid.perp.dx),
        dz=jnp.ones_like(phi) * float(geom.grid.perp.dy),
        periodic_x=False,
        periodic_z=True,
        bndry_flux=True,
        use_mc=True,
        bc_kind_x=int(geom.grid.perp.bc.kind_x),
        bc_value_x=float(geom.grid.perp.bc.x_value),
        bc_grad_x=float(geom.grid.perp.bc.x_grad),
        neumann_boundary_average_z=True,
    )
    current = geom.exb_flux_divergence(phi, adv)
    np.testing.assert_allclose(np.asarray(mirror), np.asarray(current), rtol=1e-12, atol=1e-12)


def test_exb_xz_is_differentiable() -> None:
    key_phi, key_adv = jax.random.split(jax.random.PRNGKey(23))
    phi = jax.random.normal(key_phi, (4, 6, 7), dtype=jnp.float64)
    adv = jax.random.normal(key_adv, (4, 6, 7), dtype=jnp.float64)
    J = jnp.ones_like(phi)
    dx = jnp.ones_like(phi)
    dz = jnp.ones_like(phi)

    grad = jax.grad(
        lambda arr: jnp.sum(
            div_n_bxgrad_f_b_xppm_xz(
                adv,
                arr,
                jacobian=J,
                dx=dx,
                dz=dz,
                periodic_x=False,
                periodic_z=True,
                bndry_flux=True,
                use_mc=True,
                bc_kind_x=2,
                neumann_boundary_average_z=True,
            )
        )
    )(phi)

    assert np.isfinite(np.asarray(grad)).all()
