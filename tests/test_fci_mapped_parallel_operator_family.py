"""Focused tests for the standalone mapped-FCI parallel operator family."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))
_TEST_PATH = Path(__file__).resolve().parent
if str(_TEST_PATH) not in sys.path:
    sys.path.insert(0, str(_TEST_PATH))

from drbx.geometry import (  # noqa: E402
    FCI_DEP_CUT_WALL,
    HaloLayout3D,
    LocalFciDirectionMap,
    LocalFciLocalDependencyTable,
    LocalFciMaps3D,
    StencilBuilderContext,
)
from drbx.native.fci_operators import (  # noqa: E402
    local_grad_parallel_op_fci_compatible_from_q,
    local_parallel_diffusion_fci_op,
    local_parallel_div_b_fci_from_q_op,
    local_parallel_laplacian_fci_op,
    local_parallel_q_flux_div_fci_op,
)

from test_fci_operators_domain_decomp import (  # noqa: E402
    _build_local_geometry,
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
