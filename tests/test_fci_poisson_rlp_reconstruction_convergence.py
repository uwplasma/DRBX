"""Continuum and algebraic guards for bracket-specific RLP reconstruction."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from axis_regular_operator_support import polar_fixture
from drbx.geometry import StencilBuilderContext, build_local_conservative_stencil_from_field
from drbx.geometry.fci_control_volumes import build_polar_angular_agglomeration_geometry
from drbx.native.fci_angular_agglomeration import lower_polar_angular_agglomeration_geometry
from drbx.native.fci_boundaries import BC_NEUMANN, LocalBoundaryFaceBC3D
from drbx.native.fci_drb_EB_rhs import LocalFciDrbEBRhs
from drbx.native.fci_halo import (
    GhostFillWeights1D,
    LocalHaloClosure3D,
    PhysicalGhostCellFiller3D,
)
from drbx.native.fci_model import inject_owned_field_to_halo
from drbx.native.fci_operators import (
    aggregate_local_control_volume_average,
    expand_local_control_volume_owner_field,
    local_control_volume_projected_fine_cell_volume,
    local_poisson_bracket_compatible_flux_op,
    reconstruct_local_control_volume_poisson_field,
)


def _neumann_filler(layout):
    neutral = GhostFillWeights1D(
        owned_weights=jnp.ones((layout.halo_width, 1), dtype=jnp.float64),
        bc_weights=jnp.zeros((layout.halo_width,), dtype=jnp.float64),
    )
    return PhysicalGhostCellFiller3D(
        dirichlet=(neutral, neutral, neutral),
        neumann_lower=(neutral, neutral, neutral),
        neumann_upper=(neutral, neutral, neutral),
    )


@lru_cache(maxsize=None)
def _setup(ntheta):
    nr = ntheta // 2
    shape = (nr, ntheta, 4)
    geometry, domain, context, coords, exchange, scalar_filler, *_ = polar_fixture(
        shape=shape, halo_width=2
    )
    host = build_polar_angular_agglomeration_geometry(
        np.linspace(0.0, 1.0, nr + 1),
        np.linspace(0.0, 2.0 * np.pi, ntheta + 1),
        np.linspace(0.0, 2.0 * np.pi, shape[2] + 1),
        lambda points: np.maximum(np.asarray(points)[..., 0], 1.0e-14),
        quadrature_order=2,
    )
    control_volume = lower_polar_angular_agglomeration_geometry(host, geometry)
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    face_bc = replace(
        face_bc,
        kind_x=face_bc.kind_x.at[-1].set(BC_NEUMANN),
        mask_x=face_bc.mask_x.at[-1].set(True),
    )
    closure = LocalHaloClosure3D(
        physical_ghost_filler=_neumann_filler(geometry.layout),
        halo_exchange=exchange,
        topology_filler=scalar_filler,
    )
    return geometry, domain, context, coords, control_volume, face_bc, closure


def _fields(coords):
    r, theta, _eta = coords
    F = r**2 * (1.0 - r**2) ** 2
    f = F * jnp.cos(2.0 * theta)
    g = F * jnp.sin(2.0 * theta)
    exact = 4.0 * r**2 * (1.0 - r**2) ** 3 * (1.0 - 3.0 * r**2)
    return f, g, exact


def _relative_l2(actual, exact, weight, active):
    error2 = jnp.sum(jnp.where(active, weight * (actual - exact) ** 2, 0.0))
    exact2 = jnp.sum(jnp.where(active, weight * exact**2, 0.0))
    return float(jnp.sqrt(error2 / exact2))


def _case(ntheta, characteristic_scheme="centered"):
    geometry, domain, context, coords, cv, face_bc, closure = _setup(ntheta)
    cells = cv.cells
    f_halo, g_halo, exact_halo = _fields(coords)
    owned = geometry.layout.owned_slices_cell
    f_owner = aggregate_local_control_volume_average(f_halo[owned], cells, domain)
    g_owner = aggregate_local_control_volume_average(g_halo[owned], cells, domain)
    exact = aggregate_local_control_volume_average(exact_halo[owned], cells, domain)

    def close(values):
        return closure(inject_owned_field_to_halo(values, geometry.layout), domain, face_bc)

    def action_from_closed(f_closed, g_closed):
        upwind = characteristic_scheme != "centered"
        fine = local_poisson_bracket_compatible_flux_op(
            build_local_conservative_stencil_from_field(f_closed, geometry, context),
            build_local_conservative_stencil_from_field(g_closed, geometry, context),
            geometry,
            domain=domain,
            axis_regular_axes=(True, False, False),
            characteristic_scheme=characteristic_scheme,
            g_field_halo=g_closed if upwind else None,
            cell_volume=local_control_volume_projected_fine_cell_volume(geometry, cv),
        )
        return aggregate_local_control_volume_average(fine, cells, domain)

    def action(f_fine, g_fine):
        return action_from_closed(close(f_fine), close(g_fine))

    smooth = action(f_halo[owned], g_halo[owned])
    piecewise = action(
        expand_local_control_volume_owner_field(f_owner, cells),
        expand_local_control_volume_owner_field(g_owner, cells),
    )

    class Harness:
        _owner_field = LocalFciDrbEBRhs._owner_field

    harness = Harness()
    harness.geometry = geometry
    harness.domain = domain
    harness.control_volume_geometry = cv
    harness.halo_exchange = closure.halo_exchange
    harness.topology_filler = closure.topology_filler
    harness.physical_ghost_filler = closure.physical_ghost_filler
    reconstructed = action_from_closed(
        LocalFciDrbEBRhs._prepare_poisson_bracket_halo(
            harness, f_owner, face_bc
        ),
        LocalFciDrbEBRhs._prepare_poisson_bracket_halo(
            harness, g_owner, face_bc
        ),
    )
    active = cells.is_active_owner
    weight = cells.aggregate_volume
    return {
        "continuum_error": _relative_l2(reconstructed, exact, weight, active),
        "piecewise_action_error": _relative_l2(piecewise, smooth, weight, active),
        "reconstructed_action_error": _relative_l2(reconstructed, smooth, weight, active),
    }


def test_bracket_specific_reconstruction_removes_transition_order_loss():
    cases = [_case(n) for n in (16, 32, 64)]
    errors = [case["continuum_error"] for case in cases]
    orders = [math.log(a / b, 2.0) for a, b in zip(errors[:-1], errors[1:])]
    assert min(orders) > 1.8
    # At the hardest level the former piecewise representation error is the
    # transition-limited contribution; H reduces that action defect by well
    # over an order of magnitude without changing the bracket algebra.
    assert cases[-1]["reconstructed_action_error"] < 0.1 * cases[-1]["piecewise_action_error"]


def test_production_scalar_upwind_bracket_is_second_order_through_rlp_transitions():
    cases = [
        _case(n, characteristic_scheme="scalar-third-order-upwind")
        for n in (16, 32, 64)
    ]
    errors = [case["continuum_error"] for case in cases]
    orders = [math.log(a / b, 2.0) for a, b in zip(errors[:-1], errors[1:])]
    assert min(orders) > 1.8, (errors, orders)


def test_rhs_prepared_centered_reconstruction_is_antisymmetric_and_constant_exact():
    geometry, domain, context, coords, cv, face_bc, closure = _setup(32)
    f_halo, g_halo, _ = _fields(coords)
    owned = geometry.layout.owned_slices_cell
    cells = cv.cells
    f_owner = aggregate_local_control_volume_average(f_halo[owned], cells, domain)
    g_owner = aggregate_local_control_volume_average(g_halo[owned], cells, domain)
    one_owner = aggregate_local_control_volume_average(jnp.ones_like(f_halo[owned]), cells, domain)

    class Harness:
        _owner_field = LocalFciDrbEBRhs._owner_field

    harness = Harness()
    harness.geometry = geometry
    harness.domain = domain
    harness.control_volume_geometry = cv
    harness.halo_exchange = closure.halo_exchange
    harness.topology_filler = closure.topology_filler
    harness.physical_ghost_filler = closure.physical_ghost_filler

    def prepared(values):
        return LocalFciDrbEBRhs._prepare_poisson_bracket_halo(
            harness, values, face_bc
        )

    f_prepared, g_prepared, one_prepared = map(prepared, (f_owner, g_owner, one_owner))
    stencils = [
        build_local_conservative_stencil_from_field(value, geometry, context)
        for value in (f_prepared, g_prepared, one_prepared)
    ]
    common = dict(
        domain=domain,
        axis_regular_axes=(True, False, False),
        characteristic_scheme="centered",
        cell_volume=local_control_volume_projected_fine_cell_volume(geometry, cv),
    )
    fg = local_poisson_bracket_compatible_flux_op(stencils[0], stencils[1], geometry, **common)
    gf = local_poisson_bracket_compatible_flux_op(stencils[1], stencils[0], geometry, **common)
    fc = local_poisson_bracket_compatible_flux_op(stencils[0], stencils[2], geometry, **common)
    cf = local_poisson_bracket_compatible_flux_op(stencils[2], stencils[0], geometry, **common)
    np.testing.assert_allclose(fg + gf, 0.0, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(fc, 0.0, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(cf, 0.0, rtol=0.0, atol=2.0e-13)


def test_poisson_reconstruction_is_linear_and_preserves_owner_averages():
    _geometry, domain, _context, _coords, cv, _face_bc, _closure = _setup(16)
    cells = cv.cells
    active = np.asarray(cells.is_active_owner, dtype=bool)
    indices = np.indices(cells.shape)
    left = np.zeros(cells.shape, dtype=float)
    right = np.zeros(cells.shape, dtype=float)
    left[active] = (
        0.3
        + 0.07 * indices[0][active]
        - 0.02 * indices[1][active]
    )
    right[active] = (
        -0.4
        + 0.03 * indices[0][active]
        + 0.05 * indices[1][active]
    )
    left_fine = reconstruct_local_control_volume_poisson_field(
        jnp.asarray(left), cv, domain
    )
    right_fine = reconstruct_local_control_volume_poisson_field(
        jnp.asarray(right), cv, domain
    )
    combined_fine = reconstruct_local_control_volume_poisson_field(
        jnp.asarray(1.7 * left - 0.4 * right), cv, domain
    )
    np.testing.assert_allclose(
        np.asarray(combined_fine),
        1.7 * np.asarray(left_fine) - 0.4 * np.asarray(right_fine),
        rtol=0.0,
        atol=2.0e-13,
    )
    restricted = aggregate_local_control_volume_average(
        left_fine, cells, domain
    )
    np.testing.assert_allclose(
        np.asarray(restricted)[active], left[active], rtol=0.0, atol=2.0e-12
    )
