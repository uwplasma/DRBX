"""Fully periodic Cartesian slab around the full ``LocalFciDrbEBRhs`` (test helper).

Uniform ``B`` along ``z``, identity metric, constant artificial curvature ``Q^y = kappa``
(``C(f) = kappa d_y f``), all axes periodic, no physical walls.  A uniform state
(``n = Te = Ti = 1``, everything else zero) is an exact steady state, which makes the
linearization about it exact under ``jax.jvp``.  Used by the tau p_i stage-2 RHS tests.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from drbx.geometry.fci_geometry import (
    BFieldGeometry,
    CellCenteredGrid3D,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    FciGeometry3D,
    FciMaps3D,
    Grid1D,
    LocalCurvatureFaceCoefficients3D,
    MetricGeometry,
    Spacing3D,
    build_fci_maps_from_b_contravariant,
)
from drbx.native import (
    FciDrbEBRhsParameters,
    FciDrbEBState,
    GhostFillWeights1D,
    HaloExchange3D,
    LocalBoundaryFaceBC3D,
    LocalFciDrbEBFaceBCBundle,
    LocalFciDrbEBRhs,
    LocalPeriodicTopologyRule3D,
    PhysicalGhostCellFiller3D,
    SolvaxGmresConfig,
    TopologyHaloFiller3D,
    assemble_local_fci_geometry,
    build_local_fci_geometries,
    build_local_perp_laplacian_face_projectors,
    make_shard_mesh,
)
from drbx.native.fci_drb_EB_rhs import RHS_TERM_FIELD_NAMES, RHS_TERM_NAMES

HALO = 2
PERIODIC = (True, True, True)
FIELDS7 = ("density", "Te", "Ti", "Vi", "Ve", "vorticity", "phi")

#: the two parallel configurations: the coordinate path and the production FCI path
PARALLEL_KWARGS = {
    "coord": ({}, False),
    "prod": (
        dict(
            parallel_operator_scheme="fci",
            parallel_material_scheme="production-path",
            parallel_flux_pairing="support-core",
            parallel_boundary_pairing="characteristic-sat",
        ),
        True,
    ),
}


def term_slot(field: str, term: str) -> tuple[int, int]:
    index = RHS_TERM_FIELD_NAMES.index(field)
    return index, RHS_TERM_NAMES[index].index(term)


def slab_geometry(shape, lengths, *, fci_maps=False):
    nx, ny, nz = shape
    centers = [(np.arange(n) + 0.5) * length / n for n, length in zip(shape, lengths)]
    grid = CellCenteredGrid3D(
        x=Grid1D.from_centers(jnp.asarray(centers[0])),
        y=Grid1D.from_centers(jnp.asarray(centers[1])),
        z=Grid1D.from_centers(jnp.asarray(centers[2])),
    )

    def metric(s):
        one, zero = jnp.ones(s), jnp.zeros(s)
        return MetricGeometry(
            J=one, g11=one, g22=one, g33=one, g12=zero, g13=zero, g23=zero,
            g_11=one, g_22=one, g_33=one, g_12=zero, g_13=zero, g_23=zero,
        )

    def bfield(s):
        B = jnp.stack((jnp.zeros(s), jnp.zeros(s), jnp.ones(s)), axis=-1)
        return BFieldGeometry(B_contra=B, Bmag=jnp.ones(s))

    cs = tuple(shape)
    fx, fy, fz = (nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1)
    zeros = jnp.zeros(cs)
    maps = FciMaps3D(
        forward_x=zeros, forward_y=zeros, backward_x=zeros, backward_y=zeros,
        forward_endpoint_x=zeros, forward_endpoint_y=zeros, forward_endpoint_z=zeros,
        backward_endpoint_x=zeros, backward_endpoint_y=zeros, backward_endpoint_z=zeros,
        forward_length=jnp.ones(cs), backward_length=jnp.ones(cs),
        forward_boundary=zeros.astype(bool), backward_boundary=zeros.astype(bool),
    )
    if fci_maps:
        bc = bfield(cs)
        mf = build_fci_maps_from_b_contravariant(
            grid, bc.B_contra, bc.Bmag, periodic_axes=PERIODIC
        )
        maps = FciMaps3D(**{k: mf[k] for k in (
            "forward_x", "forward_y", "backward_x", "backward_y", "forward_endpoint_x",
            "forward_endpoint_y", "forward_endpoint_z", "backward_endpoint_x",
            "backward_endpoint_y", "backward_endpoint_z", "forward_length",
            "backward_length", "forward_boundary", "backward_boundary")})
    spacing = Spacing3D(
        dx=jnp.broadcast_to(grid.x.widths[:, None, None], cs),
        dy=jnp.broadcast_to(grid.y.widths[None, :, None], cs),
        dz=jnp.broadcast_to(grid.z.widths[None, None, :], cs),
    )
    return FciGeometry3D(
        grid=grid, maps=maps, spacing=spacing, cell_metric=metric(cs),
        face_metric=FaceMetricGeometry(x=metric(fx), y=metric(fy), z=metric(fz)),
        cell_bfield=bfield(cs),
        face_bfield=FaceBFieldGeometry(x=bfield(fx), y=bfield(fy), z=bfield(fz)),
    )


def _ghost_filler():
    d = GhostFillWeights1D(owned_weights=-jnp.ones((HALO, 1)), bc_weights=2.0 * jnp.ones((HALO,)))
    n = GhostFillWeights1D(owned_weights=jnp.ones((HALO, 1)), bc_weights=jnp.zeros((HALO,)))
    return PhysicalGhostCellFiller3D(
        dirichlet=(d, d, d), neumann_lower=(n, n, n), neumann_upper=(n, n, n)
    )


def _empty_face_bcs(state, geometry, domain, parameters):
    e = LocalBoundaryFaceBC3D.empty(geometry.layout)
    return LocalFciDrbEBFaceBCBundle(density=e, phi=e, Te=e, Ti=e, Vi=e, Ve=e, vorticity=e)


def make_parameters(tau, polarization_variable, *, mi_over_me=1836.0):
    return FciDrbEBRhsParameters(
        n0=1.0, Te0=1.0, Ti0=1.0, tau=float(tau), mi_over_me=float(mi_over_me),
        rho_star=1.0, phi_inversion_regularization=0.0,
        polarization_variable=polarization_variable,
    )


class SlabModel:
    """Jitted shard_map kernels (value and jvp of ``evaluate_stage`` term fields)."""

    def __init__(self, shape, lengths, parameters, *, path="coord", kappa=0.2):
        rhs_kwargs, fci = PARALLEL_KWARGS[path]
        self.shape = tuple(shape)
        self.parameters = parameters
        self.kappa = float(kappa)
        self.fci = fci
        self.rhs_kwargs = dict(rhs_kwargs)
        self.geometry = slab_geometry(shape, lengths, fci_maps=fci)
        self.mesh = make_shard_mesh((1, 1, 1))
        self.local = build_local_fci_geometries(
            self.geometry, (1, 1, 1), halo_width=HALO, periodic_axes=PERIODIC
        )
        self.part = P("x", "y", "z")
        self.sharding = NamedSharding(self.mesh, self.part)
        self.cell_fields = jax.device_put(self.local.cell_fields, self.sharding)
        mf = self.local.map_fields if fci else np.zeros(self.shape + (1,))
        self.map_fields = jax.device_put(jnp.asarray(mf), self.sharding)
        self._terms = None
        self._jvp_terms = None

    def rhs(self, cell_fields, map_fields):
        geom = assemble_local_fci_geometry(
            self.local, cell_fields, map_fields if self.fci else None
        )
        domain = self.local.domain
        layout = geom.layout
        curvature = LocalCurvatureFaceCoefficients3D(
            layout=layout,
            x=jnp.zeros(layout.face_control_shape(0)),
            y=jnp.full(layout.face_control_shape(1), self.kappa),
            z=jnp.zeros(layout.face_control_shape(2)),
        )
        return LocalFciDrbEBRhs(
            geometry=geom, domain=domain, halo_exchange=HaloExchange3D(),
            topology_filler=TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),)),
            physical_ghost_filler=_ghost_filler(), parameters=self.parameters,
            curvature_face_coefficients=curvature,
            face_projectors=build_local_perp_laplacian_face_projectors(
                geom, domain, axis_regular_axes=(False, False, False)
            ),
            gmres_config=SolvaxGmresConfig(
                tol=1e-12, atol=1e-12, maxiter=200, restart=50,
                acceptance_tol=1e-10, acceptance_atol=1e-10,
                project_mean_zero=False, regularization_epsilon=0.0,
            ),
            face_bc_builder=_empty_face_bcs,
            **self.rhs_kwargs,
        )

    def _term_fields(self, rhs, f7):
        n, Te, Ti, Vi, Ve, w, phi = f7
        state = FciDrbEBState(
            density=n, phi=phi, Te=Te, Ti=Ti, Vi=Vi, Ve=Ve, vorticity=w
        )
        out = rhs.evaluate_stage(state, phi_owned=phi, return_rhs_term_fields=True)
        return out[1]  # (6 fields, term slots, nx, ny, nz)

    def terms(self, f7):
        """``(6, slots, nx, ny, nz)`` RHS term fields of the state ``f7`` (n, Te, Ti, Vi, Ve, w, phi)."""
        if self._terms is None:
            def kernel(*args):
                *f, cf, mf = args
                return self._term_fields(self.rhs(cf, mf), tuple(f))

            self._terms = jax.jit(jax.shard_map(
                kernel, mesh=self.mesh, in_specs=(self.part,) * 9,
                out_specs=P(None, None, "x", "y", "z"), check_vma=False))
        return np.asarray(self._terms(*self._put(f7), self.cell_fields, self.map_fields))

    def jvp_terms(self, f7, t7):
        """Tangent of :meth:`terms` along ``t7`` (exact linearization)."""
        if self._jvp_terms is None:
            def kernel(*args):
                f7_, t7_ = args[:7], args[7:14]
                cf, mf = args[14], args[15]
                rhs = self.rhs(cf, mf)
                return jax.jvp(lambda *x: self._term_fields(rhs, x), f7_, t7_)[1]

            self._jvp_terms = jax.jit(jax.shard_map(
                kernel, mesh=self.mesh, in_specs=(self.part,) * 16,
                out_specs=P(None, None, "x", "y", "z"), check_vma=False))
        return np.asarray(self._jvp_terms(
            *self._put(f7), *self._put(t7), self.cell_fields, self.map_fields))

    def _put(self, f7):
        return [
            jax.device_put(jnp.asarray(f, dtype=jnp.float64), self.sharding) for f in f7
        ]

    def uniform(self):
        one, zero = np.ones(self.shape), np.zeros(self.shape)
        return [one, one, one, zero, zero, zero, zero]


def replace_variable(parameters, variable):
    return replace(parameters, polarization_variable=variable)
