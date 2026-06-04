from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.open_field import compute_target_recycling_sources
from jax_drb.native.recycling_collision_closure import apply_collision_closure
from jax_drb.native.recycling_fixed_residual import (
    RecyclingFixedState,
    build_fixed_array_rhs,
    build_fixed_backward_euler_residual,
    build_fixed_bdf2_residual,
    build_fixed_full_field_array_rhs,
    build_fixed_host_rhs_bridge,
    fixed_residual_jvp_action,
    fixed_state_from_fields,
    fixed_state_to_feedback_integrals,
    fixed_state_to_full_fields,
    linearize_fixed_residual_action,
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
from jax_drb.native.recycling_neutral_diffusion import apply_neutral_parallel_diffusion


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


def _sample_metrics(mesh: StructuredMesh) -> StructuredMetrics:
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    ones = np.ones(shape, dtype=np.float64)
    return StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g22=ones,
        g33=ones,
        g_22=ones,
        g23=np.zeros(shape, dtype=np.float64),
        Bxy=ones,
    )


class _MiniConfig:
    def __init__(self, sections: dict[str, dict[str, object]]) -> None:
        self._sections = sections

    def has_section(self, section: str) -> bool:
        return section in self._sections

    def has_option(self, section: str, key: str) -> bool:
        return key in self._sections.get(section, {})

    def parsed(self, section: str, key: str) -> object:
        return self._sections[section][key]


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


def test_fixed_state_pytree_pack_and_numpy_restore_edge_cases() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh = _sample_mesh()
    fields = {
        "Nd": np.array([[[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]]], dtype=np.float64),
    }
    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=("Nd",),
        feedback_names=("controller",),
        mesh=mesh,
    )
    active = fields["Nd"][layout.active_slices]
    state = RecyclingFixedState(
        field_values=(jnp.asarray(active, dtype=jnp.float64),),
        feedback_values=jnp.asarray([0.25], dtype=jnp.float64),
    )

    leaves, aux = state.tree_flatten()
    rebuilt = RecyclingFixedState.tree_unflatten(aux, leaves)
    packed_without_feedback = pack_fixed_state(
        RecyclingFixedState(field_values=state.field_values, feedback_values=jnp.asarray([], dtype=jnp.float64))
    )
    packed_feedback_only = pack_fixed_state(
        RecyclingFixedState(field_values=(), feedback_values=jnp.asarray([1.0, 2.0], dtype=jnp.float64))
    )
    packed_numpy_without_feedback = pack_fixed_state(
        RecyclingFixedState(field_values=(active,), feedback_values=np.asarray([], dtype=np.float64))
    )
    packed_empty_numpy = pack_fixed_state(
        RecyclingFixedState(field_values=(), feedback_values=np.asarray([], dtype=np.float64))
    )

    assert len(rebuilt.field_values) == 1
    np.testing.assert_allclose(np.asarray(packed_without_feedback), np.ravel(active))
    np.testing.assert_allclose(np.asarray(packed_feedback_only), np.array([1.0, 2.0]))
    np.testing.assert_allclose(packed_numpy_without_feedback, np.ravel(active))
    assert isinstance(packed_empty_numpy, np.ndarray)
    assert packed_empty_numpy.size == 0
    scaled = jax.tree_util.tree_map(lambda value: value + 1.0, state)
    np.testing.assert_allclose(np.asarray(scaled.field_values[0]), active + 1.0)

    numpy_state = RecyclingFixedState(field_values=(active + 10.0,), feedback_values=np.asarray([1.5], dtype=np.float64))
    packed_numpy = pack_fixed_state(numpy_state)
    restored = unpack_fixed_state(packed_numpy, layout=layout)
    full_fields = fixed_state_to_full_fields(restored, layout=layout)
    integrals = fixed_state_to_feedback_integrals(
        restored,
        layout=layout,
        base_feedback_integrals={"unused": 9.0},
    )

    assert isinstance(packed_numpy, np.ndarray)
    assert isinstance(full_fields["Nd"], np.ndarray)
    np.testing.assert_allclose(full_fields["Nd"][layout.active_slices], active + 10.0)
    assert integrals["controller"] == pytest.approx(1.5)
    assert integrals["unused"] == pytest.approx(9.0)


def test_fixed_host_bridge_and_bdf2_residual_use_static_layout_without_reference_deck() -> None:
    mesh = _sample_mesh()
    fields = {
        "Nd": np.array([[[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]]], dtype=np.float64),
        "Pd": np.array([[[2.0], [4.0], [6.0], [8.0]], [[10.0], [12.0], [14.0], [16.0]]], dtype=np.float64),
    }
    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=("Nd", "Pd"),
        feedback_names=("controller",),
        mesh=mesh,
    )
    previous_state = fixed_state_from_fields(fields, feedback_integrals={"controller": 0.5}, layout=layout)
    previous = np.asarray(pack_fixed_state(previous_state), dtype=np.float64)
    previous_previous = previous - 0.1

    def packed_rhs(full_fields: dict[str, object], feedback_integrals: dict[str, object]) -> object:
        density = np.asarray(full_fields["Nd"], dtype=np.float64)
        pressure = np.asarray(full_fields["Pd"], dtype=np.float64)
        assert feedback_integrals["unused"] == pytest.approx(7.0)
        rhs_fields = {
            "Nd": 0.2 * density + 0.1 * pressure,
            "Pd": -0.05 * density + 0.3 * pressure,
        }
        rhs_integrals = {"controller": 0.25 * float(feedback_integrals["controller"])}
        return pack_recycling_active_state(
            rhs_fields,
            feedback_integrals=rhs_integrals,
            field_names=layout.field_names,
            feedback_names=layout.feedback_names,
            mesh=mesh,
            layout=layout,
        )

    bridge = build_fixed_host_rhs_bridge(
        packed_rhs,
        layout=layout,
        base_feedback_integrals={"unused": 7.0},
    )
    residual = build_fixed_bdf2_residual(
        bridge,
        layout=layout,
        previous_packed_state=previous,
        previous_previous_packed_state=previous_previous,
        timestep=0.2,
        previous_timestep=0.1,
    )
    candidate = previous + 0.03
    rhs_state = bridge(unpack_fixed_state(candidate, layout=layout))
    rhs_packed = np.asarray(pack_fixed_state(rhs_state), dtype=np.float64)
    step_ratio = 2.0
    previous_coefficient = ((step_ratio + 1.0) ** 2) / (2.0 * step_ratio + 1.0)
    previous_previous_coefficient = (step_ratio**2) / (2.0 * step_ratio + 1.0)
    rhs_coefficient = 0.2 * (step_ratio + 1.0) / (2.0 * step_ratio + 1.0)
    expected = (
        candidate
        - previous_coefficient * previous
        + previous_previous_coefficient * previous_previous
        - rhs_coefficient * rhs_packed
    )

    np.testing.assert_allclose(np.asarray(residual(candidate)), expected, rtol=1.0e-12, atol=1.0e-12)


def test_fixed_bdf2_residual_rejects_nonpositive_previous_timestep() -> None:
    mesh = _sample_mesh()
    layout = build_recycling_packed_state_layout(
        fields={"N": np.ones((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)},
        field_names=("N",),
        feedback_names=(),
        mesh=mesh,
    )

    def rhs_function(state: dict[str, object]) -> dict[str, object]:
        return {"N": np.asarray(state["N"], dtype=np.float64) * 0.0}

    active_size = int(np.prod(layout.active_shape))
    with pytest.raises(ValueError, match="previous_timestep must be positive"):
        build_fixed_bdf2_residual(
            rhs_function,
            layout=layout,
            previous_packed_state=np.ones(active_size, dtype=np.float64),
            previous_previous_packed_state=np.full(active_size, 0.9, dtype=np.float64),
            timestep=0.1,
            previous_timestep=0.0,
        )


def test_fixed_residual_linearize_and_jvp_actions_match_dense_jacobian() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh = _sample_mesh()
    fields = {
        "Nd": np.array([[[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]]], dtype=np.float64),
        "Pd": np.array([[[2.0], [4.0], [6.0], [8.0]], [[10.0], [12.0], [14.0], [16.0]]], dtype=np.float64),
    }
    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=("Nd", "Pd"),
        feedback_names=(),
        mesh=mesh,
    )
    previous_state = fixed_state_from_fields(fields, feedback_integrals={}, layout=layout)
    previous = pack_fixed_state(previous_state)

    def rhs(state: RecyclingFixedState) -> RecyclingFixedState:
        density, pressure = state.field_values
        return RecyclingFixedState(
            field_values=(0.2 * density + 0.1 * pressure, -0.05 * density + 0.3 * pressure),
            feedback_values=state.feedback_values,
        )

    residual = build_fixed_backward_euler_residual(
        rhs,
        layout=layout,
        previous_packed_state=previous,
        timestep=0.2,
    )
    candidate = jnp.asarray(previous, dtype=jnp.float64) + 0.03
    direction = jnp.linspace(0.1, 0.8, candidate.size, dtype=jnp.float64)
    residual_value, action = linearize_fixed_residual_action(residual, candidate)
    action_value = action(direction)
    jvp_value = fixed_residual_jvp_action(residual, candidate, direction)
    dense_value = jax.jacfwd(residual)(candidate) @ direction

    assert residual_value.shape == candidate.shape
    np.testing.assert_allclose(np.asarray(action_value), np.asarray(dense_value), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(jvp_value), np.asarray(dense_value), rtol=1.0e-12, atol=1.0e-12)


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


def test_fixed_full_field_array_rhs_stages_target_recycling_kernel() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh = _sample_mesh()
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    fields = {
        "Nd": np.ones(shape, dtype=np.float64),
        "Vd": np.zeros(shape, dtype=np.float64),
        "Td": 2.0 * np.ones(shape, dtype=np.float64),
        "SNd": np.zeros(shape, dtype=np.float64),
        "SPd": np.zeros(shape, dtype=np.float64),
    }
    fields["Vd"][:, mesh.ystart - 1, :] = -3.0
    fields["Vd"][:, mesh.ystart, :] = -1.0
    fields["Vd"][:, mesh.yend, :] = 2.0
    fields["Vd"][:, mesh.yend + 1, :] = 4.0
    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=("Nd", "Vd", "Td", "SNd", "SPd"),
        feedback_names=(),
        mesh=mesh,
    )
    state = fixed_state_from_fields(fields, feedback_integrals={}, layout=layout)
    unit_metric = jnp.ones(shape, dtype=jnp.float64)
    dy = 2.0 * unit_metric
    dx = 3.0 * unit_metric
    dz = 5.0 * unit_metric
    g_22 = 4.0 * unit_metric

    def full_field_rhs(full_fields: dict[str, object], _feedback_values: object) -> dict[str, object]:
        sources = compute_target_recycling_sources(
            full_fields["Nd"],
            full_fields["Vd"],
            full_fields["Td"],
            mesh=mesh,
            J=unit_metric,
            dy=dy,
            dx=dx,
            dz=dz,
            g_22=g_22,
            target_multiplier=0.5,
            target_energy=3.0,
            gamma_i=3.5,
        )
        return {
            "SNd": sources.density_source,
            "SPd": sources.energy_source,
        }

    fixed_rhs = build_fixed_full_field_array_rhs(full_field_rhs, layout=layout)
    rhs_state = fixed_rhs(state)
    rhs_fields = {name: value for name, value in zip(layout.field_names, rhs_state.field_values, strict=True)}
    direct = compute_target_recycling_sources(
        fields["Nd"],
        fields["Vd"],
        fields["Td"],
        mesh=mesh,
        J=np.ones(shape, dtype=np.float64),
        dy=2.0 * np.ones(shape, dtype=np.float64),
        dx=3.0 * np.ones(shape, dtype=np.float64),
        dz=5.0 * np.ones(shape, dtype=np.float64),
        g_22=4.0 * np.ones(shape, dtype=np.float64),
        target_multiplier=0.5,
        target_energy=3.0,
        gamma_i=3.5,
    )

    np.testing.assert_allclose(np.asarray(rhs_fields["Nd"]), np.zeros(layout.active_shape), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(rhs_fields["SNd"]),
        direct.density_source[layout.active_slices],
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        np.asarray(rhs_fields["SPd"]),
        direct.energy_source[layout.active_slices],
        rtol=1.0e-6,
        atol=1.0e-8,
    )

    def qoi(scale: object) -> object:
        scaled_state = RecyclingFixedState(
            field_values=(
                state.field_values[0],
                state.field_values[1] * scale,
                state.field_values[2],
                state.field_values[3],
                state.field_values[4],
            ),
            feedback_values=state.feedback_values,
        )
        result = fixed_rhs(scaled_state)
        return jnp.sum(result.field_values[3]) + jnp.sum(result.field_values[4])

    _, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-3
    finite_difference = (qoi(jnp.array(1.0 + eps)) - qoi(jnp.array(1.0 - eps))) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=1.0e-4, atol=1.0e-6)


def test_fixed_full_field_array_rhs_stages_collision_closure_kernel() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh = _sample_mesh()
    metrics = _sample_metrics(mesh)
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    fields = {
        "Nd": np.ones(shape, dtype=np.float64),
        "Pd": np.ones(shape, dtype=np.float64),
        "Td": np.ones(shape, dtype=np.float64),
        "Vd": np.zeros(shape, dtype=np.float64),
        "Md": np.zeros(shape, dtype=np.float64),
        "Nt": np.ones(shape, dtype=np.float64),
        "Pt": 2.0 * np.ones(shape, dtype=np.float64),
        "Tt": 2.0 * np.ones(shape, dtype=np.float64),
        "Vt": np.ones(shape, dtype=np.float64),
        "Mt": np.ones(shape, dtype=np.float64),
        "FNd": np.zeros(shape, dtype=np.float64),
        "EPd": np.zeros(shape, dtype=np.float64),
    }
    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=("Nd", "Pd", "Td", "Vd", "Md", "Nt", "Pt", "Tt", "Vt", "Mt", "FNd", "EPd"),
        feedback_names=(),
        mesh=mesh,
    )
    state = fixed_state_from_fields(fields, feedback_integrals={}, layout=layout)
    config = _MiniConfig({"model": {"components": ("braginskii_friction", "braginskii_heat_exchange")}})
    rates = {("d+", "t+"): jnp.ones(shape, dtype=jnp.float64) * 0.25}

    def full_field_rhs(full_fields: dict[str, object], _feedback_values: object) -> dict[str, object]:
        species = {
            "d+": SimpleNamespace(
                name="d+",
                charge=1.0,
                atomic_mass=2.0,
                has_pressure=True,
                has_momentum=True,
                density=full_fields["Nd"],
            ),
            "t+": SimpleNamespace(
                name="t+",
                charge=1.0,
                atomic_mass=3.0,
                has_pressure=True,
                has_momentum=True,
                density=full_fields["Nt"],
            ),
        }
        prepared = {
            "d+": SimpleNamespace(
                density=full_fields["Nd"],
                pressure=full_fields["Pd"],
                temperature=full_fields["Td"],
                velocity=full_fields["Vd"],
                momentum=full_fields["Md"],
            ),
            "t+": SimpleNamespace(
                density=full_fields["Nt"],
                pressure=full_fields["Pt"],
                temperature=full_fields["Tt"],
                velocity=full_fields["Vt"],
                momentum=full_fields["Mt"],
            ),
        }
        terms = apply_collision_closure(
            config,
            species,
            prepared,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars={},
            collision_rates=rates,
            cx_rates={},
        )
        return {
            "FNd": terms.momentum_source["d+"],
            "EPd": terms.energy_source["d+"],
        }

    fixed_rhs = build_fixed_full_field_array_rhs(full_field_rhs, layout=layout)

    def qoi(scale: object) -> object:
        scaled_state = RecyclingFixedState(
            field_values=(
                state.field_values[0],
                state.field_values[1],
                state.field_values[2],
                state.field_values[3],
                state.field_values[4],
                state.field_values[5],
                state.field_values[6],
                state.field_values[7],
                state.field_values[8] * scale,
                state.field_values[9] * scale,
                state.field_values[10],
                state.field_values[11],
            ),
            feedback_values=state.feedback_values,
        )
        result = fixed_rhs(scaled_state)
        return jnp.sum(result.field_values[10]) + 0.1 * jnp.sum(result.field_values[11])

    value, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-3
    finite_difference = (qoi(jnp.array(1.0 + eps)) - qoi(jnp.array(1.0 - eps))) / (2.0 * eps)
    assert np.isfinite(float(value))
    assert abs(float(tangent)) > 0.0
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=1.0e-4, atol=1.0e-6)


def test_fixed_full_field_array_rhs_stages_neutral_parallel_diffusion_kernel() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh = _sample_mesh()
    metrics = _sample_metrics(mesh)
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    fields = {
        "Nd": np.ones(shape, dtype=np.float64),
        "Pd": np.linspace(1.0, 1.4, num=np.prod(shape), dtype=np.float64).reshape(shape),
        "Td": np.linspace(1.0, 1.4, num=np.prod(shape), dtype=np.float64).reshape(shape),
        "Vd": np.zeros(shape, dtype=np.float64),
        "Md": np.zeros(shape, dtype=np.float64),
        "Ni": np.ones(shape, dtype=np.float64),
        "SNd": np.zeros(shape, dtype=np.float64),
        "EPd": np.zeros(shape, dtype=np.float64),
        "FNd": np.zeros(shape, dtype=np.float64),
    }
    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=("Nd", "Pd", "Td", "Vd", "Md", "Ni", "SNd", "EPd", "FNd"),
        feedback_names=(),
        mesh=mesh,
    )
    state = fixed_state_from_fields(fields, feedback_integrals={}, layout=layout)
    config = _MiniConfig(
        {
            "model": {"components": ("neutral_parallel_diffusion",)},
            "neutral_parallel_diffusion": {
                "dneut": 1.0,
                "diffusion_collisions_mode": "multispecies",
                "diagnose": True,
            },
        }
    )
    collision_rates = {("d", "d+"): jnp.ones(shape, dtype=jnp.float64) * 0.5}

    def full_field_rhs(full_fields: dict[str, object], _feedback_values: object) -> dict[str, object]:
        species = {
            "d": SimpleNamespace(
                name="d",
                charge=0.0,
                atomic_mass=2.0,
                has_pressure=True,
                has_momentum=True,
                density=full_fields["Nd"],
                noflow_lower_y=True,
                noflow_upper_y=True,
            ),
            "d+": SimpleNamespace(
                name="d+",
                charge=1.0,
                atomic_mass=2.0,
                has_pressure=True,
                has_momentum=True,
                density=full_fields["Ni"],
                noflow_lower_y=True,
                noflow_upper_y=True,
            ),
        }
        prepared = {
            "d": SimpleNamespace(
                density=full_fields["Nd"],
                pressure=full_fields["Pd"],
                temperature=full_fields["Td"],
                velocity=full_fields["Vd"],
                momentum=full_fields["Md"],
            ),
            "d+": SimpleNamespace(
                density=full_fields["Ni"],
                pressure=full_fields["Ni"],
                temperature=full_fields["Ni"],
                velocity=jnp.zeros_like(jnp.asarray(full_fields["Ni"], dtype=jnp.float64)),
                momentum=jnp.zeros_like(jnp.asarray(full_fields["Ni"], dtype=jnp.float64)),
            ),
        }
        terms = apply_neutral_parallel_diffusion(
            config,
            species=species,
            prepared=prepared,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars={},
            collision_rates=collision_rates,
            ionisation_rates={},
            charge_exchange_rates={},
        )
        return {
            "SNd": terms.density_source["d"],
            "EPd": terms.energy_source["d"],
            "FNd": terms.momentum_source["d"],
        }

    fixed_rhs = build_fixed_full_field_array_rhs(full_field_rhs, layout=layout)

    def qoi(scale: object) -> object:
        scaled_state = RecyclingFixedState(
            field_values=(
                state.field_values[0],
                state.field_values[1] * scale,
                state.field_values[2] * scale,
                state.field_values[3],
                state.field_values[4],
                state.field_values[5],
                state.field_values[6],
                state.field_values[7],
                state.field_values[8],
            ),
            feedback_values=state.feedback_values,
        )
        result = fixed_rhs(scaled_state)
        return jnp.sum(result.field_values[6]) + 0.1 * jnp.sum(result.field_values[7]) + 0.01 * jnp.sum(result.field_values[8])

    value, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-3
    finite_difference = (qoi(jnp.array(1.0 + eps)) - qoi(jnp.array(1.0 - eps))) / (2.0 * eps)
    assert np.isfinite(float(value))
    assert np.isfinite(float(tangent))
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=1.0e-4, atol=1.0e-6)
