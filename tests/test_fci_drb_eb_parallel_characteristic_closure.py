"""Focused tests for the five-field local parallel wall closure."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace
from pathlib import Path
import sys
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from drbx.geometry import StencilBuilderContext
from drbx.native.fci_boundaries import LocalBoundaryFaceTrace3D
from drbx.native.fci_drb_EB_rhs import (
    FciDrbEBRhsParameters,
    FciDrbEBState,
    LocalFciDrbEBOperatorBoundaryBundle,
    LocalFciDrbEBRhs,
)
from test_fci_geometry_axis_regular_curvature import _build_polar_geometry

from drbx.native.fci_drb_EB_rhs import (
    parallel_characteristic_matrix,
    parallel_characteristic_wall_state,
    parallel_derived_state_traces,
    parallel_equilibrium_characteristic_wall_state,
    parallel_incoming_projector,
)


RHS_SOURCE = Path(__file__).resolve().parents[1] / "src" / "drbx" / "native" / "fci_drb_EB_rhs.py"
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from jax.sharding import PartitionSpec as P  # noqa: E402
from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_sharding import assemble_local_fci_geometry  # noqa: E402
from fci_drb_eb_test_helpers import (  # noqa: E402
    _build_rhs,
    _context_and_sharded_inputs,
)


def _state():
    return dict(
        density=jnp.asarray(2.0),
        Te=jnp.asarray(3.0),
        Ti=jnp.asarray(5.0),
        Vi=jnp.asarray(0.7),
        Ve=jnp.asarray(-0.2),
    )


def test_analytic_parallel_matrix_entries():
    state = _state()
    tau = 4.0
    mu = 10.0
    matrix = np.asarray(parallel_characteristic_matrix(**state, tau=tau, mu=mu))
    dV = state["Vi"] - state["Ve"]
    expected = np.array([
        [state["Ve"], 0, 0, 0, state["density"]],
        [-1.42 * state["Te"] * dV / (3 * state["density"]), state["Ve"], 0,
         -1.42 * state["Te"] / 3, 3.42 * state["Te"] / 3],
        [-2 * state["Ti"] * dV / (3 * state["density"]), 0, state["Vi"], 0,
         2 * state["Ti"] / 3],
        [(state["Te"] + tau * state["Ti"]) / state["density"], 1, tau,
         state["Vi"], 0],
        [mu * state["Te"] / state["density"], 1.71 * mu, 0, 0, state["Ve"]],
    ], dtype=float)
    np.testing.assert_allclose(matrix, expected)


def test_incoming_modes_only_are_replaced():
    matrix = jnp.diag(jnp.asarray([1.0, -2.0, 0.0, 3.0, -4.0]))
    owner = jnp.zeros(5)
    candidate = jnp.ones(5)
    state = parallel_characteristic_wall_state(owner, candidate, matrix, 1.0)
    np.testing.assert_allclose(state, [0, 1, 0, 0, 1])


def test_normal_reversal_changes_incoming_selection():
    matrix = jnp.diag(jnp.asarray([1.0, -2.0, 0.0, 3.0, -4.0]))
    state = parallel_characteristic_wall_state(
        jnp.zeros(5), jnp.ones(5), matrix, -1.0
    )
    np.testing.assert_allclose(state, [1, 0, 0, 1, 0])


def test_identity_candidate_and_tangent_face_are_owner_state():
    matrix = jnp.diag(jnp.asarray([1.0, -2.0, 0.0, 3.0, -4.0]))
    owner = jnp.asarray([2.0, 3.0, 5.0, 0.7, -0.2])
    np.testing.assert_allclose(
        parallel_characteristic_wall_state(owner, owner, matrix, 1.0), owner
    )
    np.testing.assert_allclose(
        parallel_characteristic_wall_state(owner, jnp.ones(5), matrix, 0.0), owner
    )


def test_equilibrium_closure_zeroes_only_incoming_perturbations():
    matrix = jnp.diag(jnp.asarray([1.0, -2.0, 0.0, 3.0, -4.0]))
    equilibrium = jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0])
    owner = jnp.asarray([2.0, 3.0, 5.0, 0.7, -0.2])
    state = parallel_equilibrium_characteristic_wall_state(
        owner, equilibrium, matrix, 1.0
    )
    np.testing.assert_allclose(state, [2.0, 1.0, 5.0, 0.7, 0.0])
    projector = parallel_incoming_projector(matrix, jnp.asarray(1.0))
    np.testing.assert_allclose(
        projector @ (state - equilibrium), jnp.zeros(5), atol=1.0e-12
    )


def test_equilibrium_closure_preserves_equilibrium_and_tangent_owner():
    matrix = jnp.diag(jnp.asarray([1.0, -2.0, 0.0, 3.0, -4.0]))
    equilibrium = jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0])
    owner = jnp.asarray([2.0, 3.0, 5.0, 0.7, -0.2])
    np.testing.assert_allclose(
        parallel_equilibrium_characteristic_wall_state(
            equilibrium, equilibrium, matrix, 1.0
        ),
        equilibrium,
    )
    np.testing.assert_allclose(
        parallel_equilibrium_characteristic_wall_state(
            owner, equilibrium, matrix, 0.0
        ),
        owner,
    )


def test_derived_traces_use_the_same_projected_state():
    state = jnp.asarray([2.0, 3.0, 5.0, 0.7, -0.2])
    density_flux, current, Pe, pressure = parallel_derived_state_traces(state, 4.0)
    np.testing.assert_allclose(
        [density_flux, current, Pe, pressure],
        [-0.4, 1.8, 6.0, 46.0],
    )


def test_eig_projector_has_finite_stopped_jvp():
    state = _state()
    matrix = parallel_characteristic_matrix(**state, tau=4.0, mu=10.0)

    def closure(x):
        return parallel_characteristic_wall_state(
            x, jnp.ones(5), matrix, 1.0
        )

    _, tangent = jax.jvp(closure, (jnp.zeros(5),), (jnp.ones(5),))
    assert bool(jnp.all(jnp.isfinite(tangent)))


def test_projector_is_five_field_only():
    matrix = parallel_characteristic_matrix(**_state(), tau=4.0, mu=10.0)
    assert matrix.shape[-2:] == (5, 5)
    projector = parallel_incoming_projector(matrix, jnp.asarray(1.0))
    assert projector.shape[-2:] == (5, 5)


def test_source_contract_preserves_phi_vorticity_and_physical_masks():
    source = RHS_SOURCE.read_text()
    method = source[source.index("    def _parallel_operator_boundary("):]
    method = method[:method.index("    def _coordinate_stage_parallel_terms(")]
    assert "if side == 0 and self.axis_regular_axes[axis]" in method
    assert "mask = jnp.asarray(getattr(base_traces[0], f\"mask_{name}\")[sl]" in method
    assert "phi=operator_boundary.phi" not in method
    assert "vorticity=operator_boundary.vorticity" not in method
    assert "return replace(" in method
    assert "density=density, Te=Te, Ti=Ti, Vi=Vi, Ve=Ve" in method
    assert "face_bfield.B_contra_owned" in method
    assert "face_bfield.Bmag_owned" in method
    assert "face_metric.g_contra_owned" in method
    assert "face_bfield.B_contra[..., axis]" not in method
    assert "face_bfield.Bmag," not in method
    coordinate = source[source.index("    def _coordinate_stage_parallel_terms("):]
    assert "vorticity_current_flux_div = local_parallel_flux_div_op(" in coordinate
    vorticity_flux = coordinate.index("vorticity_current_flux_div = local_parallel_flux_div_op(")
    vorticity_trace = coordinate[vorticity_flux:vorticity_flux + 600]
    assert 'self.vorticity_current_inflow_trace == "parallel-characteristic"' in vorticity_trace
    assert "parallel_boundary.current" in vorticity_trace
    assert "operator_boundary.current" in vorticity_trace
    assert "* vorticity_current_flux_divergence" in source


def test_parallel_closure_has_static_central_default_and_characteristic_options():
    source = RHS_SOURCE.read_text()
    assert 'parallel_inflow_closure: str = "central"' in source
    assert 'vorticity_current_inflow_trace: str = "operator"' in source
    assert '"local-characteristic"' in source
    assert '"equilibrium-characteristic"' in source


@pytest.mark.parametrize(
    ("parallel_inflow_closure", "vorticity_current_inflow_trace"),
    (
        ("local-characteristic", "operator"),
        ("equilibrium-characteristic", "operator"),
        ("equilibrium-characteristic", "parallel-characteristic"),
    ),
)
def test_characteristic_coordinate_production_path_is_finite_and_additive(
    parallel_inflow_closure,
    vorticity_current_inflow_trace,
):
    """Compile the real sharded RHS with owned-face characteristic traces."""

    context, mesh, local, partition, fields, cell_fields = (
        _context_and_sharded_inputs()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, packed):
        geometry = assemble_local_fci_geometry(local, packed)
        rhs = _build_rhs(context, local, geometry)
        rhs = replace(
            rhs,
            parallel_inflow_closure=parallel_inflow_closure,
            vorticity_current_inflow_trace=vorticity_current_inflow_trace,
            parallel_operator_scheme="coordinate",
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        full = rhs.evaluate_stage(state, phi_owned=phi)
        finite_values = tuple(
            getattr(full, name)
            for name in full.field_names()
        )
        def stage_from_density(value):
            perturbed = state.replace(density=value)
            return rhs.evaluate_stage(perturbed, phi_owned=phi)

        tangent = jax.jvp(
            stage_from_density,
            (density,),
            (jnp.ones_like(density),),
        )[1]
        return jnp.asarray([
            jnp.max(jnp.stack(tuple(jnp.max(jnp.abs(value)) for value in finite_values))),
            jnp.asarray(0.0),
            jnp.max(jnp.stack(tuple(
                jnp.max(jnp.abs(getattr(tangent, name)))
                for name in tangent.field_names()
            ))),
        ])

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 8,
            out_specs=P(),
            check_vma=False,
        )
    )
    result = np.asarray(jax.block_until_ready(compiled(*fields, cell_fields)))
    assert np.all(np.isfinite(result)), result
    assert result[0] < np.inf
    assert result[1] == 0.0, result
    assert result[2] < np.inf


def test_parallel_boundary_real_local_geometry_is_jittable_and_owned_face_sized():
    """Exercise the production boundary builder with real local face shapes.

    In particular, this catches accidentally using halo face B-field/metric
    arrays: their shapes differ from the owned coordinate-face traces at a
    physical boundary.
    """

    geometry, domain = _build_polar_geometry(shape=(4, 8, 2), halo_width=1)
    layout = geometry.layout

    def wall_trace(value):
        shapes = tuple(layout.face_control_shape(axis) for axis in range(3))
        mask_x = jnp.zeros(shapes[0], dtype=bool)
        mask_x = mask_x.at[0].set(True)
        mask_x = mask_x.at[-1].set(True)
        return LocalBoundaryFaceTrace3D(
            value_x=jnp.full(shapes[0], value, dtype=jnp.float64),
            value_y=jnp.zeros(shapes[1], dtype=jnp.float64),
            value_z=jnp.zeros(shapes[2], dtype=jnp.float64),
            mask_x=mask_x,
            mask_y=jnp.zeros(shapes[1], dtype=bool),
            mask_z=jnp.zeros(shapes[2], dtype=bool),
            layout=layout,
        )

    base = LocalFciDrbEBOperatorBoundaryBundle(
        density=wall_trace(2.0),
        phi=wall_trace(0.0),
        Te=wall_trace(3.0),
        Ti=wall_trace(5.0),
        Vi=wall_trace(0.7),
        Ve=wall_trace(-0.2),
        vorticity=wall_trace(0.0),
        density_flux=wall_trace(0.0),
        current=wall_trace(0.0),
        Pe=wall_trace(0.0),
        pressure=wall_trace(0.0),
        Ti_squared=wall_trace(0.0),
    )
    halo_shape = geometry.cell_bfield.shape
    state_halo = FciDrbEBState(*(
        jnp.full(halo_shape, value, dtype=jnp.float64)
        for value in (2.0, 0.0, 3.0, 5.0, 0.7, -0.2, 0.0)
    ))

    rhs = object.__new__(LocalFciDrbEBRhs)
    object.__setattr__(rhs, "geometry", geometry)
    object.__setattr__(rhs, "domain", domain)
    object.__setattr__(rhs, "parameters", FciDrbEBRhsParameters(tau=4.0, mi_over_me=10.0))
    object.__setattr__(rhs, "axis_regular_axes", (True, False, False))
    object.__setattr__(rhs, "parallel_inflow_closure", "local-characteristic")
    @jax.jit
    def build_boundary(state, boundary):
        return rhs._parallel_operator_boundary(
            state_halo=state,
            operator_boundary=boundary,
        )

    result = build_boundary(state_halo, base)
    assert result.density.value_x.shape == layout.face_control_shape(0)
    assert result.Ve.value_y.shape == layout.face_control_shape(1)
    assert bool(jnp.all(jnp.isfinite(result.current.value_x)))
    # The base phi/vorticity traces are not part of the returned five-field
    # replacement and must remain unchanged.
    np.testing.assert_allclose(result.phi.value_x, base.phi.value_x)
    np.testing.assert_allclose(result.vorticity.value_x, base.vorticity.value_x)
