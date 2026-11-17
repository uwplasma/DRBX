"""Property tests for the shard-local FCI operators of the fci_braginskii backend.

All cases run on one-device open-slab geometries (identity metric, B along z).
The slab is periodic in y and z; x is either walled (homogeneous Dirichlet) or
periodic.  Only the x-y plane is exercised by the perpendicular operators.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.fci_braginskii.geometry.fci_geometry import (  # noqa: E402
    FciGeometry3D,
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
    build_local_curvature_face_coefficients,
)
from drbx.fci_braginskii.native import fci_operators as ops  # noqa: E402
from drbx.fci_braginskii.native.fci_boundaries import (  # noqa: E402
    BC_DIRICHLET,
    FaceFluxStencil3D,
    LocalBoundaryFaceBC3D,
    LocalControlVolumeFluxStencil3D,
)
from drbx.fci_braginskii.native.fci_halo import (  # noqa: E402
    GhostFillWeights1D,
    LocalHaloClosure3D,
    PhysicalGhostCellFiller3D,
    make_default_topology_halo_filler_3d,
)
from drbx.fci_braginskii.native.fci_model import inject_owned_field_to_halo  # noqa: E402
from drbx.fci_braginskii.native.fci_sharding import (  # noqa: E402
    assemble_single_device_local_fci_geometry,
    build_local_fci_geometries,
)
from drbx.geometry.open_slab import build_open_slab_geometry  # noqa: E402

NZ = 2
GRIDS = (8, 16)


@lru_cache(maxsize=None)
def _slab(n: int, x_periodic: bool = False):
    """One-device slab with ``n x n x NZ`` cells; x is walled unless ``x_periodic``."""
    legacy = build_open_slab_geometry((n, n, NZ))
    glob = FciGeometry3D(**{f.name: getattr(legacy, f.name) for f in dataclasses.fields(legacy)})
    sharded = build_local_fci_geometries(glob, (1, 1, 1), periodic_axes=(x_periodic, True, True))
    geometry = assemble_single_device_local_fci_geometry(sharded)
    domain = dataclasses.replace(sharded.domain, mesh_axis_names=(None, None, None))
    return geometry, domain


def _nonuniform_b(geometry, slope=0.3):
    """Same slab with B_z = 1 + slope*x, so the curvature coefficients are nonzero."""

    def with_b(field, x):
        bmag = (1.0 + slope * x) * jnp.ones(field.Bmag_halo.shape)
        contra = jnp.zeros(field.B_contra_halo.shape).at[..., 2].set(bmag)
        return dataclasses.replace(field, B_contra_halo=contra, Bmag_halo=bmag)

    xc = geometry.grid.x.centers_halo[:, None, None]
    xf = geometry.grid.x.faces_halo[:, None, None]
    fb = geometry.face_bfield
    faces = dataclasses.replace(fb, x=with_b(fb.x, xf), y=with_b(fb.y, xc), z=with_b(fb.z, xc))
    return dataclasses.replace(geometry, cell_bfield=with_b(geometry.cell_bfield, xc), face_bfield=faces)


def _dirichlet_bc(layout, value=0.0):
    bc = LocalBoundaryFaceBC3D.empty(layout)
    return dataclasses.replace(
        bc,
        kind_x=bc.kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        value_x=bc.value_x.at[0].set(value).at[-1].set(value),
        mask_x=bc.mask_x.at[0].set(True).at[-1].set(True),
    )


def _ghost_filler(halo_width):
    # Linear extrapolation to the wall value: ghost = 2*wall - owned.
    dirichlet = GhostFillWeights1D(
        owned_weights=jnp.full((halo_width, 1), -1.0), bc_weights=jnp.full((halo_width,), 2.0)
    )
    neutral = GhostFillWeights1D(
        owned_weights=jnp.ones((halo_width, 1)), bc_weights=jnp.zeros((halo_width,))
    )
    return PhysicalGhostCellFiller3D(
        dirichlet=(dirichlet, neutral, neutral),
        neumann_lower=(neutral, neutral, neutral),
        neumann_upper=(neutral, neutral, neutral),
    )


def _stencil(geometry, domain, field, bc=None):
    """Conservative stencil of an owned field; walled x uses ``bc`` (Dirichlet) ghosts."""
    halo = inject_owned_field_to_halo(field, domain.layout)
    walled = not domain.periodic_axes[0]
    topology = make_default_topology_halo_filler_3d(
        radial_axis_lower_regular=False, fill_periodic_axes=(not walled, True, True)
    )
    if walled:
        halo = LocalHaloClosure3D(
            physical_ghost_filler=_ghost_filler(domain.layout.halo_width),
            halo_exchange=None,
            topology_filler=topology,
        )(halo, domain, bc)
    else:
        halo = topology(halo, domain)
    return build_local_conservative_stencil_from_field(
        halo, geometry, StencilBuilderContext(layout=domain.layout, domain=domain)
    )


def _xy(geometry):
    return geometry.grid.x.centers_owned[:, None, None], geometry.grid.y.centers_owned[None, :, None]


def _order(errors):
    return float(np.log2(errors[0] / errors[1]))


def test_divergence_sums_to_boundary_flux():
    geometry, _ = _slab(8)
    rng = np.random.default_rng(0)
    nx, ny, nz = geometry.owned_shape
    flux = FaceFluxStencil3D(
        jnp.asarray(rng.standard_normal((nx + 1, ny, nz))),
        jnp.asarray(rng.standard_normal((nx, ny + 1, nz))),
        jnp.asarray(rng.standard_normal((nx, ny, nz + 1))),
    )
    cv = LocalControlVolumeFluxStencil3D(
        regular_flux=flux,
        regular_face_geometry=geometry.regular_face_geometry,
        cell_volume=geometry.cell_volume_geometry,
    )
    div = ops.local_divergence_conservative_op(cv, geometry)
    volume = geometry.cell_volume_geometry.volume * geometry.cell_volume_geometry.volume_fraction
    h = [float(s.flatten()[0]) for s in (geometry.spacing.dx_owned, geometry.spacing.dy_owned, geometry.spacing.dz_owned)]
    # Unit face areas: the cell sum telescopes to the flux through the six outer faces.
    boundary = (
        (flux.x[-1].sum() - flux.x[0].sum()) / h[0]
        + (flux.y[:, -1].sum() - flux.y[:, 0].sum()) / h[1]
        + (flux.z[:, :, -1].sum() - flux.z[:, :, 0].sum()) / h[2]
    )
    np.testing.assert_allclose(float(jnp.sum(div * volume)), float(boundary), rtol=1e-12)
    with pytest.raises(ValueError, match="cv_flux must have shape"):
        ops.local_divergence_conservative_op(cv, _slab(16)[0])


def test_perp_laplacian_matches_manufactured_solution_at_second_order():
    errors = []
    for n in GRIDS:
        geometry, domain = _slab(n)
        bc = _dirichlet_bc(domain.layout)
        x, y = _xy(geometry)
        u = jnp.sin(jnp.pi * x) * jnp.cos(2 * jnp.pi * y) * jnp.ones((1, 1, NZ))
        lap = ops.local_perp_laplacian_conservative_op(_stencil(geometry, domain, u, bc), geometry, domain, face_bc=bc)
        errors.append(float(jnp.max(jnp.abs(lap + 5 * jnp.pi**2 * u))))
    assert _order(errors) > 1.7


def test_perp_laplacian_is_symmetric_on_a_periodic_slab_and_annihilates_constants():
    geometry, domain = _slab(8, x_periodic=True)
    rng = np.random.default_rng(1)
    u, v = (jnp.asarray(rng.standard_normal(geometry.owned_shape)) for _ in range(2))

    def apply(field):
        return ops.local_perp_laplacian_conservative_op(_stencil(geometry, domain, field), geometry, domain)

    # Uniform cells: the volume weight is constant, so <Au, v> = <u, Av> is plain symmetry.
    np.testing.assert_allclose(float(jnp.sum(apply(u) * v)), float(jnp.sum(u * apply(v))), rtol=1e-12)
    assert float(jnp.max(jnp.abs(apply(jnp.full(geometry.owned_shape, 3.0))))) < 1e-10


def test_bracket_is_antisymmetric_vanishes_on_equal_arguments_and_conserves_sum():
    geometry, domain = _slab(8, x_periodic=True)
    rng = np.random.default_rng(2)
    f, g = (jnp.asarray(rng.standard_normal(geometry.owned_shape)) for _ in range(2))
    sf, sg = _stencil(geometry, domain, f), _stencil(geometry, domain, g)

    def bracket(a, b):
        return ops.local_poisson_bracket_compatible_flux_op(a, b, geometry, domain=domain)

    fg = bracket(sf, sg)
    assert float(jnp.max(jnp.abs(fg))) > 1.0
    np.testing.assert_allclose(np.asarray(fg), -np.asarray(bracket(sg, sf)), atol=1e-12)
    assert float(jnp.max(jnp.abs(bracket(sf, sf)))) < 1e-12
    assert abs(float(jnp.sum(fg))) < 1e-10  # compatible-face fluxes telescope on a periodic grid


def test_bracket_matches_manufactured_solution_at_second_order():
    errors = []
    for n in GRIDS:
        geometry, domain = _slab(n)
        bc = _dirichlet_bc(domain.layout)
        x, y = _xy(geometry)
        one = jnp.ones((1, 1, NZ))
        f = jnp.sin(jnp.pi * x) * jnp.cos(2 * jnp.pi * y) * one
        g = jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y) * one
        # {f, g} = f_x g_y - f_y g_x for B = e_z (unit B), up to the stencil's own sign convention.
        exact = (
            jnp.pi * jnp.cos(jnp.pi * x) * jnp.cos(2 * jnp.pi * y) * 2 * jnp.pi * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
            + 2 * jnp.pi * jnp.sin(jnp.pi * x) * jnp.sin(2 * jnp.pi * y) * 2 * jnp.pi * jnp.cos(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
        ) * one
        value = ops.local_poisson_bracket_compatible_flux_op(
            _stencil(geometry, domain, f, bc), _stencil(geometry, domain, g, bc), geometry, domain=domain
        )
        errors.append(float(jnp.max(jnp.abs(value - exact))))
    assert _order(errors) > 1.4  # 8 -> 16 is pre-asymptotic (1.5); 16 -> 32 gives 1.9


def test_curvature_conservative_op_annihilates_constants_and_converges():
    errors = []
    for n in GRIDS:
        geometry, domain = _slab(n)
        geometry = _nonuniform_b(geometry)
        coefficients = build_local_curvature_face_coefficients(geometry, domain)
        assert float(jnp.max(jnp.abs(coefficients.y))) > 0.1
        bc = _dirichlet_bc(domain.layout)
        x, y = _xy(geometry)
        u = jnp.sin(jnp.pi * x) * jnp.cos(2 * jnp.pi * y) * jnp.ones((1, 1, NZ))
        curv = ops.local_curvature_conservative_op(
            _stencil(geometry, domain, u, bc), geometry, coefficients, domain=domain, face_bc=bc
        )
        # C(u) = B K^y u_y with K^y = B'/(2 B^2) for A = e_z / B(x).
        exact = 0.5 * 0.3 / (1.0 + 0.3 * x) * (-2 * jnp.pi) * jnp.sin(jnp.pi * x) * jnp.sin(2 * jnp.pi * y) * jnp.ones((1, 1, NZ))
        errors.append(float(jnp.max(jnp.abs(curv - exact))))
        if n == GRIDS[0]:
            bc2 = _dirichlet_bc(domain.layout, 2.0)
            const = _stencil(geometry, domain, jnp.full(geometry.owned_shape, 2.0), bc2)
            zero = ops.local_curvature_conservative_op(const, geometry, coefficients, domain=domain, face_bc=bc2)
            assert float(jnp.max(jnp.abs(zero))) < 1e-12
    assert _order(errors) > 1.7


def test_curvature_production_path_is_zero_at_the_equilibrium_state():
    geometry, domain = _slab(8)
    geometry = _nonuniform_b(geometry)
    coefficients = build_local_curvature_face_coefficients(geometry, domain)
    equilibrium = (1.0, 1.0, 1.0, 0.0)
    stencils = tuple(
        _stencil(geometry, domain, jnp.full(geometry.owned_shape, v), _dirichlet_bc(domain.layout, v))
        for v in equilibrium
    )
    residual = ops.local_curvature_production_path_op(stencils, geometry, coefficients, tau=1.0)
    assert residual.shape == geometry.owned_shape + (4,)
    assert float(jnp.max(jnp.abs(residual))) < 1e-12
    x, y = _xy(geometry)
    bump = 0.1 * jnp.sin(jnp.pi * x) * jnp.cos(2 * jnp.pi * y) * jnp.ones((1, 1, NZ))
    bc = _dirichlet_bc(domain.layout)
    perturbed = tuple(_stencil(geometry, domain, v + bump, bc) for v in (1.0, 1.0, 1.0, 0.0))
    assert float(jnp.max(jnp.abs(ops.local_curvature_production_path_op(perturbed, geometry, coefficients, tau=1.0, domain=domain)))) > 1e-3


def test_inverse_solver_returns_a_right_inverse_of_the_negative_laplacian():
    geometry, domain = _slab(8)
    bc = _dirichlet_bc(domain.layout)
    solver = ops.LocalPerpLaplacianInverseSolver(
        geometry,
        domain,
        topology_filler=make_default_topology_halo_filler_3d(
            radial_axis_lower_regular=False, fill_periodic_axes=(False, True, True)
        ),
        physical_ghost_filler=_ghost_filler(domain.layout.halo_width),
        face_bc=bc,
    )
    b = jnp.asarray(np.random.default_rng(3).standard_normal(geometry.owned_shape))
    solution = solver(b)
    applied = -ops.local_perp_laplacian_conservative_op(
        _stencil(geometry, domain, solution, bc), geometry, domain, face_bc=bc
    )
    assert float(jnp.max(jnp.abs(applied - b))) < 1e-5 * float(jnp.max(jnp.abs(b)))
