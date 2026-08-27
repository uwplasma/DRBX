"""Focused tests for the standalone mapped-FCI parallel operator family."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax.numpy as jnp
import jax
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))
_TEST_PATH = Path(__file__).resolve().parent
if str(_TEST_PATH) not in sys.path:
    sys.path.insert(0, str(_TEST_PATH))

from drbx.geometry import (  # noqa: E402
    FCI_DEP_CUT_WALL,
    FCI_DEP_PHYSICAL_BOUNDARY,
    HaloLayout3D,
    LocalFciDirectionMap,
    LocalFciLocalDependencyTable,
    LocalFciMaps3D,
    LocalControlVolumeCellGeometry3D,
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
)
from drbx.native.fci_operators import (  # noqa: E402
    local_center_to_outgoing_face_average_fci_op,
    local_center_to_outgoing_face_grad_parallel_fci_op,
    local_grad_parallel_op_fci_compatible_from_q,
    local_grad_parallel_op_fci_compatible_from_q_components,
    local_outgoing_face_to_center_average_fci_op,
    local_outgoing_face_to_center_div_parallel_fci_op,
    local_parallel_diffusion_fci_op,
    local_parallel_div_b_fci_from_q_op,
    local_parallel_laplacian_fci_op,
    local_parallel_q_flux_div_fci_op,
    local_perp_laplacian_conservative_op,
    aggregate_local_control_volume_average,
    expand_local_control_volume_owner_field,
)
from drbx.native.fci_boundaries import BC_NEUMANN, LocalBoundaryFaceBC3D  # noqa: E402
from drbx.native.fci_halo import (  # noqa: E402
    HaloExchange3D,
    LocalPeriodicTopologyRule3D,
    TopologyHaloFiller3D,
    LocalHaloClosure3D,
)
from drbx.native.fci_model import inject_owned_field_to_halo  # noqa: E402

from test_fci_operators_domain_decomp import (  # noqa: E402
    _build_local_geometry,
    _build_domain,
    _build_ghost_filler,
    _prepare_scalar_field_halo,
    _single_fci_local_row,
)


def _geometry(*, dm: float = 1.0, dp: float = 1.0, wall: bool = False):
    layout = HaloLayout3D((1, 1, 1), 1)
    kind = FCI_DEP_CUT_WALL if wall else None
    forward = LocalFciDirectionMap(
        layout=layout,
        local=_single_fci_local_row(
            source=(2, 1, 1),
            dependency_kind=kind,
        ),
        connection_length=jnp.full(layout.owned_shape, dp),
    )
    backward = LocalFciDirectionMap(
        layout=layout,
        local=_single_fci_local_row(
            source=(0, 1, 1),
            dependency_kind=kind,
        ),
        connection_length=jnp.full(layout.owned_shape, dm),
    )
    maps = LocalFciMaps3D(
        layout=layout,
        forward=forward,
        backward=backward,
        mode="local_halo_only",
    )
    geometry = _build_local_geometry(
        layout.owned_shape,
        layout.halo_width,
        global_shape=layout.owned_shape,
        construct_fci_maps=True,
        traced_maps=maps,
    )
    # Make the field-line magnitude one on owned and endpoint cells.  The
    # low-level q tests then have a simple analytic interpretation.
    bfield = replace(
        geometry.cell_bfield,
        B_contra_halo=jnp.broadcast_to(
            jnp.array([0.0, 0.0, 1.0]), geometry.cell_bfield.B_contra_halo.shape
        ),
        Bmag_halo=jnp.ones(geometry.cell_bfield.halo_shape),
    )
    return replace(geometry, cell_bfield=bfield)


def _field_with_legs(dm: float, dp: float, *, constant: float = 0.0):
    field = jnp.zeros((3, 3, 3), dtype=jnp.float64)
    field = field.at[1, 1, 1].set(constant)
    return field


def _context(geometry):
    return StencilBuilderContext(layout=geometry.layout)


def _periodic_source_edge_geometry(*, shifted: bool, nz: int = 4):
    """Small periodic straight/shifted source-edge FCI map with full legs."""

    layout = HaloLayout3D((1, 4, nz), 1)
    ii, jj, kk = np.meshgrid(
        np.arange(1), np.arange(4), np.arange(nz), indexing="ij"
    )
    target = np.ravel_multi_index((ii.ravel(), jj.ravel(), kk.ravel()), layout.owned_shape)
    transverse_shift = 1 if shifted else 0

    def direction(sign: int):
        source_j = (jj.ravel() + sign * transverse_shift) % 4
        source_k = (kk.ravel() + sign) % nz
        return LocalFciDirectionMap(
            layout=layout,
            local=LocalFciLocalDependencyTable(
                target_flat=jnp.asarray(target, dtype=jnp.int32),
                source_i=jnp.full(target.size, 1, dtype=jnp.int32),
                source_j=jnp.asarray(source_j + 1, dtype=jnp.int32),
                source_k=jnp.asarray(source_k + 1, dtype=jnp.int32),
                weight=jnp.ones(target.size, dtype=jnp.float64),
                active=jnp.ones(target.size, dtype=bool),
            ),
            connection_length=jnp.ones(layout.owned_shape, dtype=jnp.float64),
        )

    maps = LocalFciMaps3D(
        layout=layout, forward=direction(+1), backward=direction(-1),
        mode="local_halo_only",
    )
    return _build_local_geometry(
        layout.owned_shape, layout.halo_width, global_shape=layout.owned_shape,
        construct_fci_maps=True, traced_maps=maps,
    )


def _owned_in_halo(values, geometry):
    halo = jnp.zeros(geometry.halo_shape, dtype=jnp.float64)
    return halo.at[geometry.layout.owned_slices_cell].set(values)


@pytest.mark.parametrize("shifted", (False, True), ids=("straight", "shifted"))
def test_source_edge_fci_primitives_are_uniform_grid_exact_and_keep_nyquist(shifted):
    geometry = _periodic_source_edge_geometry(shifted=shifted)
    context = _context(geometry)
    constant = _owned_in_halo(3.0 * jnp.ones(geometry.owned_shape), geometry)
    assert np.allclose(
        local_center_to_outgoing_face_average_fci_op(constant, geometry, context=context), 3.0
    )
    assert np.allclose(
        local_outgoing_face_to_center_average_fci_op(constant, geometry, context=context), 3.0
    )
    assert np.allclose(
        local_center_to_outgoing_face_grad_parallel_fci_op(constant, geometry, context=context), 0.0
    )
    assert np.allclose(
        local_outgoing_face_to_center_div_parallel_fci_op(constant, geometry, context=context), 0.0
    )

    # The z-Nyquist mode is exactly the failure case for a centered
    # cell-to-cell derivative, but the staggered full-leg G and D retain it.
    values = jnp.broadcast_to(
        jnp.where(jnp.arange(geometry.owned_shape[2]) % 2, -1.0, 1.0),
        geometry.owned_shape,
    )
    halo = _owned_in_halo(values, geometry)
    expected_plus = jnp.roll(jnp.roll(values, -1, axis=2), -int(shifted), axis=1)
    expected_minus = jnp.roll(jnp.roll(values, 1, axis=2), int(shifted), axis=1)
    assert np.allclose(
        local_center_to_outgoing_face_average_fci_op(halo, geometry, context=context),
        0.5 * (values + expected_plus),
    )
    assert np.allclose(
        local_center_to_outgoing_face_grad_parallel_fci_op(halo, geometry, context=context),
        expected_plus - values,
    )
    assert np.allclose(
        local_outgoing_face_to_center_div_parallel_fci_op(halo, geometry, context=context),
        values - expected_minus,
    )
    assert np.max(np.abs(np.asarray(local_center_to_outgoing_face_grad_parallel_fci_op(halo, geometry, context=context)))) > 0.0
    assert np.max(np.abs(np.asarray(local_outgoing_face_to_center_div_parallel_fci_op(halo, geometry, context=context)))) > 0.0


@pytest.mark.parametrize("B_value", (0.8, 1.7))
def test_staggered_b_compatible_flux_reduces_to_bare_uniform_b(B_value):
    """B D(F/B) exactly preserves the uniform-B outgoing-face stencil."""
    geometry = _periodic_source_edge_geometry(shifted=True)
    context = _context(geometry)
    values = jnp.broadcast_to(
        jnp.sin(2.0 * jnp.pi * jnp.arange(geometry.owned_shape[2]) / 4.0),
        geometry.owned_shape,
    )
    field_halo = _owned_in_halo(values, geometry)
    inverse_b_halo = _owned_in_halo(
        jnp.full(geometry.owned_shape, 1.0 / B_value), geometry
    )
    inverse_b_face = local_center_to_outgoing_face_average_fci_op(
        inverse_b_halo, geometry, context=context
    )
    inverse_b_face_halo = _owned_in_halo(inverse_b_face, geometry)
    bare_div = local_outgoing_face_to_center_div_parallel_fci_op(
        field_halo, geometry, context=context
    )
    compatible_div = B_value * local_outgoing_face_to_center_div_parallel_fci_op(
        field_halo * inverse_b_face_halo, geometry, context=context
    )
    np.testing.assert_allclose(compatible_div, bare_div, rtol=1.0e-13, atol=1.0e-13)


def test_staggered_b_compatible_flux_keeps_variable_b_divergence_of_constant_flux():
    """A constant F has B*D(1/B), not the incorrect zero bare D(F)."""
    geometry = _periodic_source_edge_geometry(shifted=False)
    context = _context(geometry)
    flux_halo = _owned_in_halo(jnp.ones(geometry.owned_shape), geometry)
    inverse_b = jnp.broadcast_to(
        jnp.asarray((0.75, 1.25, 0.5, 1.0)), geometry.owned_shape
    )
    inverse_b_halo = _owned_in_halo(inverse_b, geometry)
    inverse_b_face = local_center_to_outgoing_face_average_fci_op(
        inverse_b_halo, geometry, context=context
    )
    inverse_b_face_halo = _owned_in_halo(inverse_b_face, geometry)
    B_center = 1.0 / inverse_b
    compatible = B_center * local_outgoing_face_to_center_div_parallel_fci_op(
        flux_halo * inverse_b_face_halo, geometry, context=context
    )
    expected = B_center * local_outgoing_face_to_center_div_parallel_fci_op(
        inverse_b_face_halo, geometry, context=context
    )
    assert np.max(np.abs(np.asarray(compatible))) > 0.0
    np.testing.assert_allclose(compatible, expected, rtol=1.0e-13, atol=1.0e-13)


def test_source_edge_face_to_center_average_uses_leg_length_weights():
    geometry = _geometry(dm=1.0, dp=2.0)
    faces = jnp.zeros(geometry.halo_shape, dtype=jnp.float64)
    faces = faces.at[0, 1, 1].set(-1.0)  # incoming edge
    faces = faces.at[1, 1, 1].set(2.0)   # outgoing edge
    average = local_outgoing_face_to_center_average_fci_op(
        faces, geometry, context=_context(geometry)
    )
    # At the center, q_out and q_in are weighted by their opposite distances.
    assert np.allclose(average, 0.0)


def test_mapped_div_b_and_compatible_gradient_annihilate_constants():
    geometry = _geometry()
    q_halo = jnp.ones(geometry.halo_shape, dtype=jnp.float64)
    div_b = local_parallel_div_b_fci_from_q_op(
        q_halo,
        geometry,
        context=_context(geometry),
    )
    gradient = local_grad_parallel_op_fci_compatible_from_q(
        q_halo,
        geometry,
        context=_context(geometry),
        field_owned=jnp.ones(geometry.owned_shape),
        div_b=div_b,
    )
    assert np.allclose(div_b, 0.0)
    assert np.allclose(gradient, 0.0)


def test_prepared_q_gradient_matches_analytic_unequal_leg_derivative():
    dm, dp = 1.0, 2.0
    geometry = _geometry(dm=dm, dp=dp)
    q_halo = jnp.zeros(geometry.halo_shape, dtype=jnp.float64)
    q_halo = q_halo.at[0, 1, 1].set(-dm + dm * dm)
    q_halo = q_halo.at[1, 1, 1].set(0.0)
    q_halo = q_halo.at[2, 1, 1].set(dp + dp * dp)
    gradient = local_grad_parallel_op_fci_compatible_from_q(
        q_halo,
        geometry,
        context=_context(geometry),
        field_halo_full=q_halo,
        div_b=jnp.zeros(geometry.owned_shape),
    )
    assert np.allclose(gradient, 1.0)


def test_compatible_gradient_components_reconstruct_production_gradient():
    dm, dp = 1.0, 2.0
    geometry = _geometry(dm=dm, dp=dp)
    q_halo = jnp.zeros(geometry.halo_shape, dtype=jnp.float64)
    q_halo = q_halo.at[0, 1, 1].set(-dm + dm * dm)
    q_halo = q_halo.at[1, 1, 1].set(0.0)
    q_halo = q_halo.at[2, 1, 1].set(dp + dp * dp)
    inverse_b_halo = jnp.ones(geometry.halo_shape, dtype=jnp.float64)
    components, endpoints = (
        local_grad_parallel_op_fci_compatible_from_q_components(
            q_halo,
            inverse_b_halo,
            geometry,
            context=_context(geometry),
            field_owned=jnp.zeros(geometry.owned_shape),
        )
    )
    gradient = local_grad_parallel_op_fci_compatible_from_q(
        q_halo,
        geometry,
        context=_context(geometry),
        field_owned=jnp.zeros(geometry.owned_shape),
        div_b=jnp.zeros(geometry.owned_shape),
    )
    np.testing.assert_allclose(np.sum(components, axis=0), gradient)
    np.testing.assert_allclose(endpoints[:, 0, 0, 0], (0.0, 6.0))


def test_mapped_laplacian_and_diffusion_are_quadratic_exact_on_unequal_legs():
    dm, dp = 1.0, 2.0
    geometry = _geometry(dm=dm, dp=dp)
    field_halo = jnp.zeros(geometry.halo_shape, dtype=jnp.float64)
    field_halo = field_halo.at[0, 1, 1].set(-dm + dm * dm)
    field_halo = field_halo.at[1, 1, 1].set(0.0)
    field_halo = field_halo.at[2, 1, 1].set(dp + dp * dp)
    laplacian = local_parallel_laplacian_fci_op(
        field_halo,
        geometry,
        context=_context(geometry),
    )
    diffusion = local_parallel_diffusion_fci_op(
        field_halo,
        geometry,
        context=_context(geometry),
        diffusivity_halo_full=2.0 * jnp.ones_like(field_halo),
        inverse_b_halo_full=jnp.ones_like(field_halo),
    )
    assert np.allclose(laplacian, 2.0)
    assert np.allclose(diffusion, 4.0)


def test_direction_aware_supplied_q_wall_endpoints_override_field_halo():
    geometry = _geometry(dm=1.0, dp=2.0, wall=True)
    q_halo = jnp.zeros(geometry.halo_shape, dtype=jnp.float64)
    divergence = local_parallel_q_flux_div_fci_op(
        q_halo,
        geometry,
        context=_context(geometry),
        forward_cut_wall_q_values=jnp.array([2.0]),
        backward_cut_wall_q_values=jnp.array([-3.0]),
    )
    # Unequal-leg first derivative weights for dm=1, dp=2.
    assert np.allclose(divergence, 7.0 / 3.0)


def test_low_level_q_operator_does_not_read_physical_b_ghosts():
    geometry = _geometry()
    bfield = replace(
        geometry.cell_bfield,
        Bmag_halo=geometry.cell_bfield.Bmag_halo.at[0].set(jnp.nan),
    )
    geometry = replace(geometry, cell_bfield=bfield)
    q_halo = jnp.zeros(geometry.halo_shape, dtype=jnp.float64)
    q_halo = q_halo.at[0, 1, 1].set(-1.0)
    q_halo = q_halo.at[2, 1, 1].set(1.0)
    result = local_parallel_q_flux_div_fci_op(
        q_halo,
        geometry,
        context=_context(geometry),
    )
    assert np.allclose(result, 1.0)
