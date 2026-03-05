from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.core.geometry_field_aligned import FieldAlignedGeometryAdapter, FieldAlignedGrid
from jaxdrb.core.params import DRBSystemParams, NumericsParams


def _make_geom(
    *,
    poloidal_on: bool,
    with_g23: bool,
    neumann_avg_y: bool = False,
    bc_x: str = "periodic",
    exb_flux_scheme: str = "centered",
    exb_poloidal_x_scale: float = 1.0,
    exb_poloidal_y_scale: float = 1.0,
) -> FieldAlignedGeometryAdapter:
    params = DRBSystemParams(
        numerics=NumericsParams(
            exb_poloidal_flows=poloidal_on,
            exb_poloidal_scale=1.0,
            exb_poloidal_x_scale=exb_poloidal_x_scale,
            exb_poloidal_y_scale=exb_poloidal_y_scale,
            exb_flux_scheme=exb_flux_scheme,
            neumann_boundary_average_y=neumann_avg_y,
            perp_operator="fd",
            bracket="centered",
            parallel_transform="none",
        )
    )
    grid = FieldAlignedGrid.make(
        nx=8,
        ny=8,
        nz=6,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        bc_x=bc_x,
        bc_y="periodic",
        dealias=False,
        open_field_line=False,
    )
    return FieldAlignedGeometryAdapter.from_coefficients(
        params=params,
        grid=grid,
        curv_x=0.0,
        curv_y=0.0,
        dpar_factor=1.0,
        B=1.0,
        jacobian=1.0,
        gxx=1.0,
        gxy=0.0,
        gyy=1.0,
        g23=1.0 if with_g23 else None,
    )


def test_exb_poloidal_flows_toggle_changes_flux_divergence() -> None:
    key_phi, key_adv = jax.random.split(jax.random.PRNGKey(0))
    phi = jax.random.normal(key_phi, (6, 8, 8), dtype=jnp.float64)
    adv = jax.random.normal(key_adv, (6, 8, 8), dtype=jnp.float64)

    geom_off = _make_geom(poloidal_on=False, with_g23=True)
    geom_on = _make_geom(poloidal_on=True, with_g23=True)
    div_off = geom_off.exb_flux_divergence(phi, adv)
    div_on = geom_on.exb_flux_divergence(phi, adv)

    assert not jnp.allclose(div_on, div_off, atol=1e-12, rtol=1e-12)


def test_exb_poloidal_flows_requires_g23_metric() -> None:
    key_phi, key_adv = jax.random.split(jax.random.PRNGKey(1))
    phi = jax.random.normal(key_phi, (6, 8, 8), dtype=jnp.float64)
    adv = jax.random.normal(key_adv, (6, 8, 8), dtype=jnp.float64)

    geom_off = _make_geom(poloidal_on=False, with_g23=False)
    geom_on_missing = _make_geom(poloidal_on=True, with_g23=False)
    div_off = geom_off.exb_flux_divergence(phi, adv)
    div_on_missing = geom_on_missing.exb_flux_divergence(phi, adv)

    assert jnp.allclose(div_on_missing, div_off, atol=1e-12, rtol=1e-12)


def test_exb_neumann_boundary_average_y_only_affects_neumann_x() -> None:
    key_phi, key_adv = jax.random.split(jax.random.PRNGKey(2))
    phi = jax.random.normal(key_phi, (6, 8, 8), dtype=jnp.float64)
    adv = jax.random.normal(key_adv, (6, 8, 8), dtype=jnp.float64)

    # Emphasize x-boundary variation across y where Hermes/BOUT applies
    # neumann_boundary_average_z.
    yvec = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float64)
    adv = adv.at[:, 0, :].set(adv[:, 0, :] + 4.0 * yvec[None, :])
    adv = adv.at[:, -1, :].set(adv[:, -1, :] - 3.0 * yvec[None, :])

    geom_neu_off = _make_geom(poloidal_on=True, with_g23=True, neumann_avg_y=False, bc_x="neumann")
    geom_neu_on = _make_geom(poloidal_on=True, with_g23=True, neumann_avg_y=True, bc_x="neumann")
    div_neu_off = geom_neu_off.exb_flux_divergence(phi, adv)
    div_neu_on = geom_neu_on.exb_flux_divergence(phi, adv)
    assert not jnp.allclose(div_neu_on, div_neu_off, atol=1e-12, rtol=1e-12)

    geom_per_off = _make_geom(poloidal_on=True, with_g23=True, neumann_avg_y=False, bc_x="periodic")
    geom_per_on = _make_geom(poloidal_on=True, with_g23=True, neumann_avg_y=True, bc_x="periodic")
    div_per_off = geom_per_off.exb_flux_divergence(phi, adv)
    div_per_on = geom_per_on.exb_flux_divergence(phi, adv)
    assert jnp.allclose(div_per_on, div_per_off, atol=1e-12, rtol=1e-12)


def test_exb_flux_scheme_hermes_fromm_differs_from_centered() -> None:
    key_phi, key_adv = jax.random.split(jax.random.PRNGKey(3))
    phi = jax.random.normal(key_phi, (6, 8, 8), dtype=jnp.float64)
    adv = jax.random.normal(key_adv, (6, 8, 8), dtype=jnp.float64)

    geom_centered = _make_geom(
        poloidal_on=True, with_g23=True, exb_flux_scheme="centered", bc_x="neumann"
    )
    geom_fromm = _make_geom(
        poloidal_on=True, with_g23=True, exb_flux_scheme="hermes_fromm", bc_x="neumann"
    )
    div_centered = geom_centered.exb_flux_divergence(phi, adv)
    div_fromm = geom_fromm.exb_flux_divergence(phi, adv)

    assert not jnp.allclose(div_fromm, div_centered, atol=1e-12, rtol=1e-12)


def test_exb_poloidal_branch_scales_split_x_and_y_contributions() -> None:
    key_phi, key_adv = jax.random.split(jax.random.PRNGKey(4))
    phi = jax.random.normal(key_phi, (6, 8, 8), dtype=jnp.float64)
    adv = jax.random.normal(key_adv, (6, 8, 8), dtype=jnp.float64)

    geom_off = _make_geom(poloidal_on=False, with_g23=True, exb_flux_scheme="hermes_xppm")
    geom_xy0 = _make_geom(
        poloidal_on=True,
        with_g23=True,
        exb_flux_scheme="hermes_xppm",
        exb_poloidal_x_scale=0.0,
        exb_poloidal_y_scale=0.0,
    )
    geom_xonly = _make_geom(
        poloidal_on=True,
        with_g23=True,
        exb_flux_scheme="hermes_xppm",
        exb_poloidal_x_scale=1.0,
        exb_poloidal_y_scale=0.0,
    )
    geom_yonly = _make_geom(
        poloidal_on=True,
        with_g23=True,
        exb_flux_scheme="hermes_xppm",
        exb_poloidal_x_scale=0.0,
        exb_poloidal_y_scale=1.0,
    )
    geom_on = _make_geom(poloidal_on=True, with_g23=True, exb_flux_scheme="hermes_xppm")

    div_off = geom_off.exb_flux_divergence(phi, adv)
    div_xy0 = geom_xy0.exb_flux_divergence(phi, adv)
    div_x = geom_xonly.exb_flux_divergence(phi, adv)
    div_y = geom_yonly.exb_flux_divergence(phi, adv)
    div_on = geom_on.exb_flux_divergence(phi, adv)

    assert jnp.allclose(div_xy0, div_off, atol=1e-12, rtol=1e-12)
    assert not jnp.allclose(div_x, div_y, atol=1e-12, rtol=1e-12)
    assert jnp.allclose(div_on, div_x + div_y - div_off, atol=1e-9, rtol=1e-9)


def test_metric_open_ddy_c2_uses_local_cell_spacing() -> None:
    geom = _make_geom(poloidal_on=True, with_g23=True)
    f = jnp.arange(5, dtype=jnp.float64)[:, None, None] * jnp.ones((1, 2, 3), dtype=jnp.float64)
    ds = jnp.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=jnp.float64)[:, None, None]

    ddy = geom._ddy_open_c2_metric(f, ds)
    expected = jnp.broadcast_to((1.0 / ds), f.shape)

    assert jnp.allclose(ddy, expected, atol=1e-12, rtol=1e-12)


def test_shifted_transform_nox_leaves_x_boundaries_unshifted() -> None:
    params = DRBSystemParams(
        numerics=NumericsParams(
            parallel_transform="shifted",
            parallel_shift_interp="linear",
            perp_operator="fd",
            bracket="centered",
        )
    )
    grid = FieldAlignedGrid.make(
        nx=6,
        ny=8,
        nz=4,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        bc_x="neumann",
        bc_y="periodic",
        dealias=False,
        open_field_line=True,
    )
    geom = FieldAlignedGeometryAdapter.from_coefficients(
        params=params,
        grid=grid,
        curv_x=0.0,
        curv_y=0.0,
        dpar_factor=1.0,
        B=1.0,
        jacobian=1.0,
        gxx=1.0,
        gxy=0.0,
        gyy=1.0,
        g23=1.0,
        z_shift=jnp.ones((4, 6), dtype=jnp.float64) * 0.3,
    )
    field = jax.random.normal(jax.random.PRNGKey(5), (4, 6, 8), dtype=jnp.float64)

    shifted = geom.to_field_aligned(field)
    shifted_nox = geom.to_field_aligned_nox(field)

    assert not jnp.allclose(shifted_nox[:, 0, :], shifted[:, 0, :], atol=1e-12, rtol=1e-12)
    assert not jnp.allclose(shifted_nox[:, -1, :], shifted[:, -1, :], atol=1e-12, rtol=1e-12)
    assert jnp.allclose(shifted_nox[:, 0, :], field[:, 0, :], atol=1e-12, rtol=1e-12)
    assert jnp.allclose(shifted_nox[:, -1, :], field[:, -1, :], atol=1e-12, rtol=1e-12)
    assert jnp.allclose(shifted_nox[:, 1:-1, :], shifted[:, 1:-1, :], atol=1e-12, rtol=1e-12)
