from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.recycling_rhs_terms import (
    ElectronPressureRhsTerms,
    IonRhsTerms,
    NeutralRhsTerms,
    assemble_electron_pressure_rhs_terms,
    assemble_ion_rhs_terms,
    assemble_neutral_rhs_terms,
)


def _mesh_and_metrics() -> tuple[StructuredMesh, StructuredMetrics]:
    mesh = StructuredMesh(
        nx=1,
        ny=1,
        nz=1,
        mxg=0,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=0,
        jyseps1_2=0,
        jyseps2_2=0,
        ny_inner=1,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=np.array([0.0], dtype=np.float64),
        y=np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )
    ones = np.ones((1, 3, 1), dtype=np.float64)
    return mesh, StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g22=ones,
        g33=ones,
        g_22=ones,
        g23=np.zeros_like(ones),
        Bxy=ones,
    )


def test_electron_pressure_rhs_terms_sum_to_total() -> None:
    mesh, metrics = _mesh_and_metrics()
    explicit = np.full((1, 3, 1), 2.0, dtype=np.float64)
    pressure = np.array([[[3.0], [4.0], [5.0]]], dtype=np.float64)
    velocity = np.array([[[-1.0], [-0.5], [0.0]]], dtype=np.float64)
    fastest_wave = np.full((1, 3, 1), 1.5, dtype=np.float64)
    energy_source = np.full((1, 3, 1), 0.75, dtype=np.float64)

    terms = assemble_electron_pressure_rhs_terms(
        explicit_pressure_source=explicit,
        electron_pressure=pressure,
        electron_velocity=velocity,
        electron_fastest_wave=fastest_wave,
        electron_energy_source=energy_source,
        mesh=mesh,
        metrics=metrics,
    )

    assert isinstance(terms, ElectronPressureRhsTerms)
    np.testing.assert_allclose(
        terms.total,
        terms.explicit_pressure_source + terms.parallel_divergence + terms.parallel_advection + terms.energy_source,
    )


def test_ion_rhs_terms_sum_to_total() -> None:
    mesh, metrics = _mesh_and_metrics()
    ion_state = SimpleNamespace(
        density=np.array([[[2.0], [2.5], [3.0]]], dtype=np.float64),
        pressure=np.array([[[4.0], [5.0], [6.0]]], dtype=np.float64),
        momentum_error=np.full((1, 3, 1), 0.125, dtype=np.float64),
    )
    terms = assemble_ion_rhs_terms(
        density_source=np.full((1, 3, 1), 1.0, dtype=np.float64),
        explicit_pressure_source=np.full((1, 3, 1), 2.0, dtype=np.float64),
        momentum_source=np.full((1, 3, 1), -0.75, dtype=np.float64),
        atomic_mass=2.0,
        density_floor=1.0e-6,
        ion_state=ion_state,
        ion_velocity=np.array([[[-1.0], [-0.25], [0.5]]], dtype=np.float64),
        fastest_wave=np.full((1, 3, 1), 1.5, dtype=np.float64),
        mesh=mesh,
        metrics=metrics,
        energy_source=np.full((1, 3, 1), 0.9, dtype=np.float64),
    )

    assert isinstance(terms, IonRhsTerms)
    np.testing.assert_allclose(
        terms.density_total,
        terms.density_source + terms.density_transport,
    )
    np.testing.assert_allclose(
        terms.pressure_total,
        terms.explicit_pressure_source + terms.parallel_divergence + terms.parallel_advection + terms.energy_source,
    )
    np.testing.assert_allclose(
        terms.momentum_total,
        terms.momentum_advection + terms.pressure_gradient + terms.momentum_source + terms.momentum_error,
    )


def test_neutral_rhs_terms_sum_to_total_and_respect_total_pressure_source_override() -> None:
    mesh, metrics = _mesh_and_metrics()
    neutral_state = SimpleNamespace(
        density=np.array([[[2.0], [2.5], [3.0]]], dtype=np.float64),
        pressure=np.array([[[4.0], [5.0], [6.0]]], dtype=np.float64),
        momentum_error=np.full((1, 3, 1), -0.25, dtype=np.float64),
    )
    terms = assemble_neutral_rhs_terms(
        density_source=np.full((1, 3, 1), 1.0, dtype=np.float64),
        explicit_pressure_source=np.full((1, 3, 1), 2.0, dtype=np.float64),
        momentum_source=np.full((1, 3, 1), -0.75, dtype=np.float64),
        atomic_mass=2.0,
        density_floor=1.0e-6,
        neutral_state=neutral_state,
        neutral_velocity=np.array([[[-1.0], [-0.25], [0.5]]], dtype=np.float64),
        fastest_wave=np.full((1, 3, 1), 1.5, dtype=np.float64),
        mesh=mesh,
        metrics=metrics,
        energy_source=np.full((1, 3, 1), 0.9, dtype=np.float64),
        include_energy_source=False,
    )

    assert isinstance(terms, NeutralRhsTerms)
    np.testing.assert_allclose(
        terms.density_total,
        terms.density_source + terms.density_transport,
    )
    np.testing.assert_allclose(
        terms.pressure_total,
        terms.explicit_pressure_source + terms.parallel_divergence + terms.parallel_advection,
    )
    np.testing.assert_allclose(terms.energy_source, 0.0)
    np.testing.assert_allclose(
        terms.momentum_total,
        terms.momentum_advection + terms.pressure_gradient + terms.momentum_source + terms.momentum_error,
    )


def test_electron_pressure_rhs_terms_are_jax_jvp_transformable() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh, metrics = _mesh_and_metrics()
    explicit = np.full((1, 3, 1), 2.0, dtype=np.float64)
    pressure = np.array([[[3.0], [4.0], [5.0]]], dtype=np.float64)
    velocity = np.array([[[-1.0], [-0.5], [0.0]]], dtype=np.float64)
    fastest_wave = np.full((1, 3, 1), 1.5, dtype=np.float64)
    energy_source = np.full((1, 3, 1), 0.75, dtype=np.float64)

    def qoi(scale):
        terms = assemble_electron_pressure_rhs_terms(
            explicit_pressure_source=jnp.asarray(explicit),
            electron_pressure=jnp.asarray(pressure) * scale,
            electron_velocity=jnp.asarray(velocity),
            electron_fastest_wave=jnp.asarray(fastest_wave),
            electron_energy_source=jnp.asarray(energy_source),
            mesh=mesh,
            metrics=metrics,
        )
        return jnp.sum(terms.total * jnp.asarray([[[0.2], [0.7], [1.1]]], dtype=jnp.float64))

    _, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-5
    finite_difference = (qoi(1.0 + eps) - qoi(1.0 - eps)) / (2.0 * eps)

    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=2.0e-6, atol=2.0e-8)


def test_neutral_rhs_terms_jax_branch_matches_numpy_and_jvp_finite_difference() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh, metrics = _mesh_and_metrics()
    neutral_state = SimpleNamespace(
        density=np.array([[[2.0], [2.5], [3.0]]], dtype=np.float64),
        pressure=np.array([[[4.0], [5.0], [6.0]]], dtype=np.float64),
        momentum_error=np.full((1, 3, 1), -0.25, dtype=np.float64),
    )
    common_kwargs = {
        "density_source": np.full((1, 3, 1), 1.0, dtype=np.float64),
        "explicit_pressure_source": np.full((1, 3, 1), 2.0, dtype=np.float64),
        "momentum_source": np.full((1, 3, 1), -0.75, dtype=np.float64),
        "atomic_mass": 2.0,
        "density_floor": 1.0e-6,
        "neutral_velocity": np.array([[[-1.0], [-0.25], [0.5]]], dtype=np.float64),
        "fastest_wave": np.full((1, 3, 1), 1.5, dtype=np.float64),
        "mesh": mesh,
        "metrics": metrics,
        "energy_source": np.full((1, 3, 1), 0.9, dtype=np.float64),
    }
    numpy_terms = assemble_neutral_rhs_terms(neutral_state=neutral_state, **common_kwargs)
    jax_terms = assemble_neutral_rhs_terms(
        neutral_state=SimpleNamespace(
            density=jnp.asarray(neutral_state.density),
            pressure=jnp.asarray(neutral_state.pressure),
            momentum_error=jnp.asarray(neutral_state.momentum_error),
        ),
        **common_kwargs,
    )

    np.testing.assert_allclose(np.asarray(jax_terms.density_total), numpy_terms.density_total, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(jax_terms.pressure_total), numpy_terms.pressure_total, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(jax_terms.momentum_total), numpy_terms.momentum_total, rtol=1.0e-12, atol=1.0e-12)

    weights = jnp.asarray([[[0.3], [0.8], [1.4]]], dtype=jnp.float64)

    def qoi(scale):
        terms = assemble_neutral_rhs_terms(
            neutral_state=SimpleNamespace(
                density=jnp.asarray(neutral_state.density) * scale,
                pressure=jnp.asarray(neutral_state.pressure),
                momentum_error=jnp.asarray(neutral_state.momentum_error),
            ),
            density_source=jnp.asarray(common_kwargs["density_source"]),
            explicit_pressure_source=jnp.asarray(common_kwargs["explicit_pressure_source"]),
            momentum_source=jnp.asarray(common_kwargs["momentum_source"]),
            atomic_mass=2.0,
            density_floor=1.0e-6,
            neutral_velocity=jnp.asarray(common_kwargs["neutral_velocity"]),
            fastest_wave=jnp.asarray(common_kwargs["fastest_wave"]),
            mesh=mesh,
            metrics=metrics,
            energy_source=jnp.asarray(common_kwargs["energy_source"]),
        )
        return jnp.sum(terms.momentum_total * weights)

    _, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-5
    finite_difference = (qoi(1.0 + eps) - qoi(1.0 - eps)) / (2.0 * eps)

    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=2.0e-6, atol=2.0e-8)


def test_ion_rhs_terms_jax_branch_matches_numpy_and_jvp_finite_difference() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh, metrics = _mesh_and_metrics()
    ion_state = SimpleNamespace(
        density=np.array([[[2.0], [2.5], [3.0]]], dtype=np.float64),
        pressure=np.array([[[4.0], [5.0], [6.0]]], dtype=np.float64),
        momentum_error=np.full((1, 3, 1), 0.125, dtype=np.float64),
    )
    common_kwargs = {
        "density_source": np.full((1, 3, 1), 1.0, dtype=np.float64),
        "explicit_pressure_source": np.full((1, 3, 1), 2.0, dtype=np.float64),
        "momentum_source": np.full((1, 3, 1), -0.75, dtype=np.float64),
        "atomic_mass": 2.0,
        "density_floor": 1.0e-6,
        "ion_velocity": np.array([[[-1.0], [-0.25], [0.5]]], dtype=np.float64),
        "fastest_wave": np.full((1, 3, 1), 1.5, dtype=np.float64),
        "mesh": mesh,
        "metrics": metrics,
        "energy_source": np.full((1, 3, 1), 0.9, dtype=np.float64),
    }
    numpy_terms = assemble_ion_rhs_terms(ion_state=ion_state, **common_kwargs)
    jax_terms = assemble_ion_rhs_terms(
        ion_state=SimpleNamespace(
            density=jnp.asarray(ion_state.density),
            pressure=jnp.asarray(ion_state.pressure),
            momentum_error=jnp.asarray(ion_state.momentum_error),
        ),
        **common_kwargs,
    )

    np.testing.assert_allclose(np.asarray(jax_terms.density_total), numpy_terms.density_total, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(jax_terms.pressure_total), numpy_terms.pressure_total, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(jax_terms.momentum_total), numpy_terms.momentum_total, rtol=1.0e-12, atol=1.0e-12)

    weights = jnp.asarray([[[0.3], [0.8], [1.4]]], dtype=jnp.float64)

    def qoi(scale):
        terms = assemble_ion_rhs_terms(
            ion_state=SimpleNamespace(
                density=jnp.asarray(ion_state.density) * scale,
                pressure=jnp.asarray(ion_state.pressure),
                momentum_error=jnp.asarray(ion_state.momentum_error),
            ),
            density_source=jnp.asarray(common_kwargs["density_source"]),
            explicit_pressure_source=jnp.asarray(common_kwargs["explicit_pressure_source"]),
            momentum_source=jnp.asarray(common_kwargs["momentum_source"]),
            atomic_mass=2.0,
            density_floor=1.0e-6,
            ion_velocity=jnp.asarray(common_kwargs["ion_velocity"]),
            fastest_wave=jnp.asarray(common_kwargs["fastest_wave"]),
            mesh=mesh,
            metrics=metrics,
            energy_source=jnp.asarray(common_kwargs["energy_source"]),
        )
        return jnp.sum(terms.momentum_total * weights)

    _, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-5
    finite_difference = (qoi(1.0 + eps) - qoi(1.0 - eps)) / (2.0 * eps)

    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=2.0e-6, atol=2.0e-8)
