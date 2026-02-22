import jax.numpy as jnp
import numpy as np

from jaxdrb.core.geometry_field_aligned import FieldAlignedGrid, FieldAlignedGeometryAdapter
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.region_bcs import RegionBC, RegionBCField
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.context import build_context
from jaxdrb.core.terms.region_bc import region_bc_relaxation


def _make_geom(params, region_masks, region_bcs):
    grid = FieldAlignedGrid.make(
        nx=1,
        ny=1,
        nz=3,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        bc_x="periodic",
        bc_y="periodic",
        dealias=False,
        open_field_line=True,
        region_masks=region_masks,
        region_bcs=region_bcs,
    )
    return FieldAlignedGeometryAdapter.from_coefficients(
        params=params,
        grid=grid,
        curv_x=0.0,
        curv_y=0.0,
        dpar_factor=1.0,
        B=1.0,
    )


def test_region_bc_dirichlet_log_and_linear():
    mask = jnp.asarray([1.0, 0.0, 1.0])
    region_masks = {"target": mask}
    region_bcs = (
        RegionBC(
            name="target",
            mask=mask,
            n=RegionBCField(kind="dirichlet", value=1.5, nu=2.0),
        ),
    )

    n_phys = jnp.asarray([1.0, 2.0, 4.0])
    shape = (3, 1, 1)

    # Log-form case
    params = update_params_from_dict(DRBSystemParams(), {"physics": {"log_n": True}})
    geom = _make_geom(params, region_masks, region_bcs)
    y = DRBSystemState(
        n=jnp.log(n_phys).reshape(shape),
        omega=jnp.zeros(shape),
        vpar_e=jnp.zeros(shape),
        vpar_i=jnp.zeros(shape),
        Te=jnp.zeros(shape),
        Ti=None,
        psi=None,
        N=None,
    )
    ctx = build_context(params, geom, y)
    rhs = region_bc_relaxation(ctx, y)

    expected = -2.0 * mask.reshape(shape) * (n_phys.reshape(shape) - 1.5) / n_phys.reshape(shape)
    np.testing.assert_allclose(rhs.n, expected, rtol=1e-6, atol=1e-6)

    # Linear-form case
    params_lin = update_params_from_dict(DRBSystemParams(), {"physics": {"log_n": False}})
    geom_lin = _make_geom(params_lin, region_masks, region_bcs)
    y_lin = DRBSystemState(
        n=n_phys.reshape(shape),
        omega=jnp.zeros(shape),
        vpar_e=jnp.zeros(shape),
        vpar_i=jnp.zeros(shape),
        Te=jnp.zeros(shape),
        Ti=None,
        psi=None,
        N=None,
    )
    ctx_lin = build_context(params_lin, geom_lin, y_lin)
    rhs_lin = region_bc_relaxation(ctx_lin, y_lin)

    expected_lin = -2.0 * mask.reshape(shape) * (n_phys.reshape(shape) - 1.5)
    np.testing.assert_allclose(rhs_lin.n, expected_lin, rtol=1e-6, atol=1e-6)


def test_region_bc_neumann_target():
    mask = jnp.asarray([1.0, 0.0, 1.0])
    region_masks = {"leg": mask}
    region_bcs = (
        RegionBC(
            name="leg",
            mask=mask,
            vpar_e=RegionBCField(kind="neumann", grad=0.5, nu=3.0),
        ),
    )
    params = DRBSystemParams()
    geom = _make_geom(params, region_masks, region_bcs)

    shape = (3, 1, 1)
    z = jnp.linspace(0.0, 1.0, shape[0])
    vpar_e = z.reshape(shape)
    y = DRBSystemState(
        n=jnp.zeros(shape),
        omega=jnp.zeros(shape),
        vpar_e=vpar_e,
        vpar_i=jnp.zeros(shape),
        Te=jnp.zeros(shape),
        Ti=None,
        psi=None,
        N=None,
    )
    ctx = build_context(params, geom, y)
    rhs = region_bc_relaxation(ctx, y)

    dpar = geom.dpar(vpar_e)
    expected = -3.0 * mask.reshape(shape) * (dpar - 0.5)
    np.testing.assert_allclose(rhs.vpar_e, expected, rtol=1e-6, atol=1e-6)


def test_region_bc_dirichlet_log_Te():
    mask = jnp.asarray([1.0, 0.0, 1.0])
    region_masks = {"target": mask}
    region_bcs = (
        RegionBC(
            name="target",
            mask=mask,
            Te=RegionBCField(kind="dirichlet", value=2.5, nu=1.5),
        ),
    )

    Te_phys = jnp.asarray([2.0, 4.0, 8.0])
    shape = (3, 1, 1)

    params = update_params_from_dict(DRBSystemParams(), {"physics": {"log_Te": True}})
    geom = _make_geom(params, region_masks, region_bcs)
    y = DRBSystemState(
        n=jnp.zeros(shape),
        omega=jnp.zeros(shape),
        vpar_e=jnp.zeros(shape),
        vpar_i=jnp.zeros(shape),
        Te=jnp.log(Te_phys).reshape(shape),
        Ti=None,
        psi=None,
        N=None,
    )
    ctx = build_context(params, geom, y)
    rhs = region_bc_relaxation(ctx, y)

    expected = -1.5 * mask.reshape(shape) * (Te_phys.reshape(shape) - 2.5) / Te_phys.reshape(shape)
    np.testing.assert_allclose(rhs.Te, expected, rtol=1e-6, atol=1e-6)


def test_region_bc_neumann_log_n():
    mask = jnp.asarray([1.0, 0.0, 1.0])
    region_masks = {"leg": mask}
    region_bcs = (
        RegionBC(
            name="leg",
            mask=mask,
            n=RegionBCField(kind="neumann", grad=0.2, nu=2.0),
        ),
    )
    params = update_params_from_dict(DRBSystemParams(), {"physics": {"log_n": True}})
    geom = _make_geom(params, region_masks, region_bcs)

    shape = (3, 1, 1)
    z = jnp.linspace(0.0, 1.0, shape[0])
    n_phys = (1.0 + z).reshape(shape)
    y = DRBSystemState(
        n=jnp.log(n_phys),
        omega=jnp.zeros(shape),
        vpar_e=jnp.zeros(shape),
        vpar_i=jnp.zeros(shape),
        Te=jnp.zeros(shape),
        Ti=None,
        psi=None,
        N=None,
    )
    ctx = build_context(params, geom, y)
    rhs = region_bc_relaxation(ctx, y)

    dpar = geom.dpar(n_phys)
    expected = -2.0 * mask.reshape(shape) * (dpar - 0.2) / n_phys
    np.testing.assert_allclose(rhs.n, expected, rtol=1e-6, atol=1e-6)


def test_region_bc_dirichlet_vpar_i():
    mask = jnp.asarray([1.0, 0.0, 1.0])
    region_masks = {"target": mask}
    region_bcs = (
        RegionBC(
            name="target",
            mask=mask,
            vpar_i=RegionBCField(kind="dirichlet", value=0.4, nu=1.0),
        ),
    )
    params = DRBSystemParams()
    geom = _make_geom(params, region_masks, region_bcs)

    shape = (3, 1, 1)
    vpar_i = jnp.asarray([0.0, 0.5, 1.0]).reshape(shape)
    y = DRBSystemState(
        n=jnp.zeros(shape),
        omega=jnp.zeros(shape),
        vpar_e=jnp.zeros(shape),
        vpar_i=vpar_i,
        Te=jnp.zeros(shape),
        Ti=None,
        psi=None,
        N=None,
    )
    ctx = build_context(params, geom, y)
    rhs = region_bc_relaxation(ctx, y)

    expected = -1.0 * mask.reshape(shape) * (vpar_i - 0.4)
    np.testing.assert_allclose(rhs.vpar_i, expected, rtol=1e-6, atol=1e-6)
