from __future__ import annotations

import numpy as np
import pytest

from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.recycling_fixed_residual import (
    RecyclingFixedState,
    build_fixed_array_rhs,
    build_fixed_backward_euler_residual,
    fixed_state_from_fields,
    pack_fixed_state,
    unpack_fixed_state,
)
from jax_drb.native.recycling_layout import (
    build_recycling_packed_state_layout,
    pack_recycling_active_state,
    recycling_active_domain_slices,
    recycling_active_field_size,
    recycling_active_shape,
    unpack_recycling_active_state,
)


def _sample_mesh() -> StructuredMesh:
    return StructuredMesh(
        nx=2,
        ny=2,
        nz=1,
        mxg=0,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=0,
        jyseps1_2=0,
        jyseps2_2=0,
        ny_inner=2,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=np.array([0.0, 1.0], dtype=np.float64),
        y=np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )


def test_recycling_active_domain_helpers_match_mesh_core() -> None:
    mesh = _sample_mesh()

    assert recycling_active_domain_slices(mesh) == (slice(0, 2), slice(1, 3), slice(None))
    assert recycling_active_shape(mesh) == (2, 2, 1)
    assert recycling_active_field_size(mesh) == 4


def test_pack_and_unpack_recycling_state_round_trip() -> None:
    mesh = _sample_mesh()
    field_names = ("Nd", "Pd")
    feedback_names = ("d_feedback",)
    fields = {
        "Nd": np.array([[[10.0], [11.0], [12.0], [13.0]], [[20.0], [21.0], [22.0], [23.0]]], dtype=np.float64),
        "Pd": np.array([[[30.0], [31.0], [32.0], [33.0]], [[40.0], [41.0], [42.0], [43.0]]], dtype=np.float64),
    }
    feedback_integrals = {"d_feedback": 1.25, "unused": 9.0}

    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
    )
    packed = pack_recycling_active_state(
        fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
        layout=layout,
    )

    assert packed.shape == (9,)
    np.testing.assert_allclose(packed[:4], np.array([11.0, 12.0, 21.0, 22.0]))
    np.testing.assert_allclose(packed[4:8], np.array([31.0, 32.0, 41.0, 42.0]))
    np.testing.assert_allclose(packed[8:], np.array([1.25]))

    restored_fields, restored_integrals = unpack_recycling_active_state(
        packed,
        field_templates=fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
        layout=layout,
    )

    np.testing.assert_allclose(restored_fields["Nd"], fields["Nd"])
    np.testing.assert_allclose(restored_fields["Pd"], fields["Pd"])
    assert restored_integrals["d_feedback"] == 1.25
    assert restored_integrals["unused"] == 9.0


def test_pack_recycling_state_without_layout_matches_layout_path() -> None:
    mesh = _sample_mesh()
    field_names = ("Nd",)
    feedback_names = ()
    fields = {
        "Nd": np.array([[[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]]], dtype=np.float64),
    }
    feedback_integrals: dict[str, float] = {}

    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
    )

    packed_with_layout = pack_recycling_active_state(
        fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
        layout=layout,
    )
    packed_without_layout = pack_recycling_active_state(
        fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
        layout=None,
    )

    np.testing.assert_allclose(packed_with_layout, packed_without_layout)


def test_recycling_active_pack_unpack_preserves_jax_tracers() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh = _sample_mesh()
    field_names = ("Nd",)
    feedback_names = ("controller",)
    fields = {
        "Nd": jnp.asarray(
            [[[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]]],
            dtype=jnp.float64,
        ),
    }
    layout = build_recycling_packed_state_layout(
        fields={"Nd": np.asarray(fields["Nd"], dtype=np.float64)},
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
    )

    def qoi(scale):
        scaled = {"Nd": fields["Nd"] * scale}
        packed = pack_recycling_active_state(
            scaled,
            feedback_integrals={"controller": scale},
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
            layout=layout,
        )
        restored, integrals = unpack_recycling_active_state(
            packed,
            field_templates=fields,
            feedback_integrals={"controller": 0.0},
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
            layout=layout,
        )
        return jnp.sum(restored["Nd"][layout.active_slices]) + integrals["controller"]

    value, tangent = jax.jvp(qoi, (jnp.array(2.0),), (jnp.array(1.0),))

    assert value == pytest.approx(2.0 * (2.0 + 3.0 + 6.0 + 7.0) + 2.0)
    assert tangent == pytest.approx((2.0 + 3.0 + 6.0 + 7.0) + 1.0)


def test_fixed_recycling_backward_euler_residual_is_jax_linearizable() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh = _sample_mesh()
    fields = {
        "Nd": np.array([[[10.0], [11.0], [12.0], [13.0]], [[20.0], [21.0], [22.0], [23.0]]], dtype=np.float64),
        "Pd": np.array([[[30.0], [31.0], [32.0], [33.0]], [[40.0], [41.0], [42.0], [43.0]]], dtype=np.float64),
    }
    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=("Nd", "Pd"),
        feedback_names=("controller",),
        mesh=mesh,
    )
    previous_state = fixed_state_from_fields(fields, feedback_integrals={"controller": 0.5}, layout=layout)
    previous_packed = pack_fixed_state(previous_state)

    def rhs(state: RecyclingFixedState) -> RecyclingFixedState:
        density, pressure = state.field_values
        return RecyclingFixedState(
            field_values=(-0.25 * density + 0.1 * pressure, 0.2 * density - 0.5 * pressure),
            feedback_values=-0.1 * state.feedback_values,
        )

    residual = build_fixed_backward_euler_residual(
        rhs,
        layout=layout,
        previous_packed_state=previous_packed,
        timestep=0.25,
    )
    candidate = previous_packed + 0.01
    direction = jnp.ones_like(candidate)
    value, tangent = jax.jvp(residual, (candidate,), (direction,))
    unpacked = unpack_fixed_state(candidate, layout=layout)

    assert value.shape == candidate.shape
    assert tangent.shape == candidate.shape
    assert len(unpacked.field_values) == 2
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(jax.jacfwd(residual)(candidate) @ direction))


def test_fixed_array_rhs_builds_transformable_term_by_term_residual() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh = _sample_mesh()
    fields = {
        "Nd": np.array([[[10.0], [11.0], [12.0], [13.0]], [[20.0], [21.0], [22.0], [23.0]]], dtype=np.float64),
        "Pd": np.array([[[30.0], [31.0], [32.0], [33.0]], [[40.0], [41.0], [42.0], [43.0]]], dtype=np.float64),
    }
    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=("Nd", "Pd"),
        feedback_names=("controller",),
        mesh=mesh,
    )
    previous_state = fixed_state_from_fields(fields, feedback_integrals={"controller": 0.5}, layout=layout)
    previous_packed = pack_fixed_state(previous_state)

    def field_rhs(active_fields: dict[str, object], feedback_values: object) -> dict[str, object]:
        density = jnp.asarray(active_fields["Nd"])
        pressure = jnp.asarray(active_fields["Pd"])
        controller = jnp.asarray(feedback_values)[0]
        return {
            "Nd": -0.1 * density + 0.02 * pressure + controller,
            "Pd": 0.05 * density - 0.2 * pressure,
        }

    def feedback_rhs(active_fields: dict[str, object], feedback_values: object) -> object:
        return jnp.asarray([-0.5 * jnp.asarray(feedback_values)[0] + 0.01 * jnp.mean(active_fields["Nd"])])

    fixed_rhs = build_fixed_array_rhs(field_rhs, layout=layout, feedback_rhs_function=feedback_rhs)
    residual = build_fixed_backward_euler_residual(
        fixed_rhs,
        layout=layout,
        previous_packed_state=previous_packed,
        timestep=0.25,
    )

    candidate = previous_packed + 0.02
    value, tangent = jax.jvp(residual, (candidate,), (jnp.ones_like(candidate),))

    assert value.shape == candidate.shape
    assert tangent.shape == candidate.shape
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(jax.jacfwd(residual)(candidate) @ jnp.ones_like(candidate)))
