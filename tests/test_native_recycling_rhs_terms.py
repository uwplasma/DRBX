from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import jax_drb.native.recycling_rhs_terms as rhs_terms_mod
from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.recycling_rhs_terms import (
    ElectronParallelForceTerms,
    ElectronPressureRhsTerms,
    IonRhsTerms,
    NeutralRhsTerms,
    assemble_electron_parallel_force_active_terms,
    assemble_electron_parallel_force_terms,
    assemble_electron_pressure_active_rhs_terms,
    assemble_electron_pressure_rhs_terms,
    assemble_ion_active_rhs_terms,
    assemble_ion_rhs_terms,
    assemble_neutral_active_rhs_terms,
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


def _multi_y_mesh_and_metrics() -> tuple[StructuredMesh, StructuredMetrics]:
    mesh = StructuredMesh(
        nx=1,
        ny=4,
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
        x=np.array([0.0], dtype=np.float64),
        y=np.linspace(-1.0, 1.0, 6, dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )
    shape = (1, 6, 1)
    grid = np.linspace(0.0, 1.0, 6, dtype=np.float64).reshape(shape)
    return mesh, StructuredMetrics(
        dx=np.ones(shape, dtype=np.float64),
        dy=1.0 + 0.1 * grid,
        dz=np.ones(shape, dtype=np.float64),
        J=1.0 + 0.2 * grid,
        g11=np.ones(shape, dtype=np.float64),
        g22=1.0 + 0.05 * grid,
        g33=np.ones(shape, dtype=np.float64),
        g_22=1.0 + 0.15 * grid,
        g23=np.zeros(shape, dtype=np.float64),
        Bxy=np.ones(shape, dtype=np.float64),
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


def test_electron_parallel_force_terms_add_ion_electric_source() -> None:
    mesh, metrics = _mesh_and_metrics()
    pressure = np.array([[[3.0], [4.0], [6.0]]], dtype=np.float64)
    electron_density = np.array([[[2.0], [2.5], [3.0]]], dtype=np.float64)
    electron_momentum_source = np.full((1, 3, 1), 0.25, dtype=np.float64)
    ion_density = {"d+": np.array([[[1.0], [1.5], [2.0]]], dtype=np.float64)}
    ion_momentum_source = {"d+": np.full((1, 3, 1), -0.5, dtype=np.float64)}

    terms = assemble_electron_parallel_force_terms(
        electron_pressure=pressure,
        electron_density=electron_density,
        electron_momentum_source=electron_momentum_source,
        ion_density=ion_density,
        ion_charge={"d+": 1.0},
        ion_momentum_source=ion_momentum_source,
        mesh=mesh,
        metrics=metrics,
    )

    assert isinstance(terms, ElectronParallelForceTerms)
    np.testing.assert_allclose(
        terms.ion_momentum_source["d+"],
        ion_momentum_source["d+"] + ion_density["d+"] * terms.epar,
    )
    np.testing.assert_allclose(terms.epar, terms.force_density / np.maximum(electron_density, 1.0e-5))


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


def test_ion_rhs_terms_use_numpy_for_numpy_state_with_jax_metrics() -> None:
    jnp = pytest.importorskip("jax.numpy")
    mesh, metrics = _mesh_and_metrics()
    jax_metrics = replace(
        metrics,
        dx=jnp.asarray(metrics.dx),
        dy=jnp.asarray(metrics.dy),
        dz=jnp.asarray(metrics.dz),
        J=jnp.asarray(metrics.J),
        g11=jnp.asarray(metrics.g11),
        g22=jnp.asarray(metrics.g22),
        g33=jnp.asarray(metrics.g33),
        g_22=jnp.asarray(metrics.g_22),
        g23=jnp.asarray(metrics.g23),
        Bxy=jnp.asarray(metrics.Bxy),
    )
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
        metrics=jax_metrics,
        energy_source=np.full((1, 3, 1), 0.9, dtype=np.float64),
    )

    assert isinstance(terms.density_total, np.ndarray)
    assert isinstance(terms.pressure_total, np.ndarray)
    assert isinstance(terms.momentum_total, np.ndarray)


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


def test_absent_sources_match_full_zero_sources() -> None:
    mesh, metrics = _mesh_and_metrics()
    shape = (1, 3, 1)
    zero = np.zeros(shape, dtype=np.float64)
    pressure = np.array([[[3.0], [4.0], [5.0]]], dtype=np.float64)
    velocity = np.array([[[-1.0], [-0.5], [0.0]]], dtype=np.float64)
    fastest_wave = np.full(shape, 1.5, dtype=np.float64)
    state = SimpleNamespace(
        density=np.array([[[2.0], [2.5], [3.0]]], dtype=np.float64),
        pressure=np.array([[[4.0], [5.0], [6.0]]], dtype=np.float64),
        momentum_error=np.full(shape, 0.125, dtype=np.float64),
    )

    electron_none = assemble_electron_pressure_rhs_terms(
        explicit_pressure_source=None,
        electron_pressure=pressure,
        electron_velocity=velocity,
        electron_fastest_wave=fastest_wave,
        electron_energy_source=None,
        mesh=mesh,
        metrics=metrics,
    )
    electron_zero = assemble_electron_pressure_rhs_terms(
        explicit_pressure_source=zero,
        electron_pressure=pressure,
        electron_velocity=velocity,
        electron_fastest_wave=fastest_wave,
        electron_energy_source=zero,
        mesh=mesh,
        metrics=metrics,
    )
    np.testing.assert_allclose(electron_none.total, electron_zero.total)

    ion_none = assemble_ion_rhs_terms(
        density_source=None,
        explicit_pressure_source=None,
        momentum_source=None,
        atomic_mass=2.0,
        density_floor=1.0e-6,
        ion_state=state,
        ion_velocity=velocity,
        fastest_wave=fastest_wave,
        mesh=mesh,
        metrics=metrics,
        energy_source=None,
    )
    ion_zero = assemble_ion_rhs_terms(
        density_source=zero,
        explicit_pressure_source=zero,
        momentum_source=zero,
        atomic_mass=2.0,
        density_floor=1.0e-6,
        ion_state=state,
        ion_velocity=velocity,
        fastest_wave=fastest_wave,
        mesh=mesh,
        metrics=metrics,
        energy_source=zero,
    )
    np.testing.assert_allclose(ion_none.density_total, ion_zero.density_total)
    np.testing.assert_allclose(ion_none.pressure_total, ion_zero.pressure_total)
    np.testing.assert_allclose(ion_none.momentum_total, ion_zero.momentum_total)

    neutral_none = assemble_neutral_rhs_terms(
        density_source=None,
        explicit_pressure_source=None,
        momentum_source=None,
        atomic_mass=2.0,
        density_floor=1.0e-6,
        neutral_state=state,
        neutral_velocity=velocity,
        fastest_wave=fastest_wave,
        mesh=mesh,
        metrics=metrics,
        energy_source=None,
    )
    neutral_zero = assemble_neutral_rhs_terms(
        density_source=zero,
        explicit_pressure_source=zero,
        momentum_source=zero,
        atomic_mass=2.0,
        density_floor=1.0e-6,
        neutral_state=state,
        neutral_velocity=velocity,
        fastest_wave=fastest_wave,
        mesh=mesh,
        metrics=metrics,
        energy_source=zero,
    )
    np.testing.assert_allclose(neutral_none.density_total, neutral_zero.density_total)
    np.testing.assert_allclose(neutral_none.pressure_total, neutral_zero.pressure_total)
    np.testing.assert_allclose(neutral_none.momentum_total, neutral_zero.momentum_total)


def test_active_rhs_terms_match_full_field_slices() -> None:
    mesh, metrics = _multi_y_mesh_and_metrics()
    active = (slice(mesh.xstart, mesh.xend + 1), slice(mesh.ystart, mesh.yend + 1), slice(None))
    shape = (1, mesh.local_ny, 1)
    y = np.linspace(0.0, 1.0, mesh.local_ny, dtype=np.float64).reshape(shape)
    density = 2.0 + 0.4 * y
    pressure = 3.0 + 0.7 * y + 0.05 * y * y
    velocity = -0.3 + 0.15 * y
    fastest_wave = 1.2 + 0.2 * y
    source = 0.1 + 0.03 * y
    momentum_error = -0.02 + 0.01 * y
    state = SimpleNamespace(
        density=density,
        pressure=pressure,
        momentum_error=momentum_error,
    )

    full_electron = assemble_electron_pressure_rhs_terms(
        explicit_pressure_source=source,
        electron_pressure=pressure,
        electron_velocity=velocity,
        electron_fastest_wave=fastest_wave,
        electron_energy_source=2.0 * source,
        mesh=mesh,
        metrics=metrics,
    )
    active_electron = assemble_electron_pressure_active_rhs_terms(
        explicit_pressure_source=source,
        electron_pressure=pressure,
        electron_velocity=velocity,
        electron_fastest_wave=fastest_wave,
        electron_energy_source=2.0 * source,
        mesh=mesh,
        metrics=metrics,
    )
    np.testing.assert_allclose(active_electron.total, full_electron.total[active])

    full_force = assemble_electron_parallel_force_terms(
        electron_pressure=pressure,
        electron_density=density,
        electron_momentum_source=source,
        ion_density={"d+": 1.4 * density},
        ion_charge={"d+": 1.0},
        ion_momentum_source={"d+": -source},
        mesh=mesh,
        metrics=metrics,
    )
    active_force = assemble_electron_parallel_force_active_terms(
        electron_pressure=pressure,
        electron_density=density,
        electron_momentum_source=source,
        ion_density={"d+": 1.4 * density},
        ion_charge={"d+": 1.0},
        ion_momentum_source={"d+": -source},
        mesh=mesh,
        metrics=metrics,
    )
    np.testing.assert_allclose(active_force.force_density, full_force.force_density[active])
    np.testing.assert_allclose(active_force.epar, full_force.epar[active])
    np.testing.assert_allclose(
        active_force.ion_momentum_source["d+"],
        full_force.ion_momentum_source["d+"][active],
    )

    full_ion = assemble_ion_rhs_terms(
        density_source=source,
        explicit_pressure_source=2.0 * source,
        momentum_source=-0.5 * source,
        atomic_mass=2.0,
        density_floor=1.0e-6,
        ion_state=state,
        ion_velocity=velocity,
        fastest_wave=fastest_wave,
        mesh=mesh,
        metrics=metrics,
        energy_source=3.0 * source,
    )
    active_ion = assemble_ion_active_rhs_terms(
        density_source=source,
        explicit_pressure_source=2.0 * source,
        momentum_source=-0.5 * source,
        atomic_mass=2.0,
        density_floor=1.0e-6,
        ion_state=state,
        ion_velocity=velocity,
        fastest_wave=fastest_wave,
        mesh=mesh,
        metrics=metrics,
        energy_source=3.0 * source,
    )
    np.testing.assert_allclose(active_ion.density_total, full_ion.density_total[active])
    np.testing.assert_allclose(active_ion.pressure_total, full_ion.pressure_total[active])
    np.testing.assert_allclose(active_ion.momentum_total, full_ion.momentum_total[active])

    full_neutral = assemble_neutral_rhs_terms(
        density_source=source,
        explicit_pressure_source=2.0 * source,
        momentum_source=-0.5 * source,
        atomic_mass=2.0,
        density_floor=1.0e-6,
        neutral_state=state,
        neutral_velocity=velocity,
        fastest_wave=fastest_wave,
        mesh=mesh,
        metrics=metrics,
        energy_source=3.0 * source,
    )
    active_neutral = assemble_neutral_active_rhs_terms(
        density_source=source,
        explicit_pressure_source=2.0 * source,
        momentum_source=-0.5 * source,
        atomic_mass=2.0,
        density_floor=1.0e-6,
        neutral_state=state,
        neutral_velocity=velocity,
        fastest_wave=fastest_wave,
        mesh=mesh,
        metrics=metrics,
        energy_source=3.0 * source,
    )
    np.testing.assert_allclose(
        active_neutral.density_total,
        full_neutral.density_total[active],
    )
    np.testing.assert_allclose(
        active_neutral.pressure_total,
        full_neutral.pressure_total[active],
    )
    np.testing.assert_allclose(
        active_neutral.momentum_total,
        full_neutral.momentum_total[active],
    )


def test_ion_and_neutral_rhs_terms_reuse_pressure_gradient_stencil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, metrics = _multi_y_mesh_and_metrics()
    active = (
        slice(mesh.xstart, mesh.xend + 1),
        slice(mesh.ystart, mesh.yend + 1),
        slice(None),
    )
    shape = (1, mesh.local_ny, 1)
    state = SimpleNamespace(
        density=np.full(shape, 2.0, dtype=np.float64),
        pressure=np.linspace(3.0, 4.0, mesh.local_ny, dtype=np.float64).reshape(shape),
        momentum_error=np.zeros(shape, dtype=np.float64),
    )
    velocity = np.full(shape, 0.25, dtype=np.float64)
    fastest_wave = np.full(shape, 1.5, dtype=np.float64)
    call_counts = {"full": 0, "active": 0}

    def fake_grad_par_open(pressure, *, mesh, metrics):
        del mesh, metrics
        call_counts["full"] += 1
        return np.ones_like(pressure, dtype=np.float64)

    def fake_grad_par_open_active(pressure, *, mesh, metrics):
        del mesh, metrics
        call_counts["active"] += 1
        return np.ones_like(pressure[active], dtype=np.float64)

    monkeypatch.setattr(rhs_terms_mod, "_grad_par_open", fake_grad_par_open)
    monkeypatch.setattr(
        rhs_terms_mod,
        "_grad_par_open_active",
        fake_grad_par_open_active,
    )

    assemble_ion_rhs_terms(
        density_source=None,
        explicit_pressure_source=None,
        momentum_source=None,
        atomic_mass=2.0,
        density_floor=1.0e-6,
        ion_state=state,
        ion_velocity=velocity,
        fastest_wave=fastest_wave,
        mesh=mesh,
        metrics=metrics,
        energy_source=None,
    )
    assert call_counts["full"] == 1

    assemble_neutral_rhs_terms(
        density_source=None,
        explicit_pressure_source=None,
        momentum_source=None,
        atomic_mass=2.0,
        density_floor=1.0e-6,
        neutral_state=state,
        neutral_velocity=velocity,
        fastest_wave=fastest_wave,
        mesh=mesh,
        metrics=metrics,
        energy_source=None,
    )
    assert call_counts["full"] == 2

    assemble_ion_active_rhs_terms(
        density_source=None,
        explicit_pressure_source=None,
        momentum_source=None,
        atomic_mass=2.0,
        density_floor=1.0e-6,
        ion_state=state,
        ion_velocity=velocity,
        fastest_wave=fastest_wave,
        mesh=mesh,
        metrics=metrics,
        energy_source=None,
    )
    assert call_counts["active"] == 1

    assemble_neutral_active_rhs_terms(
        density_source=None,
        explicit_pressure_source=None,
        momentum_source=None,
        atomic_mass=2.0,
        density_floor=1.0e-6,
        neutral_state=state,
        neutral_velocity=velocity,
        fastest_wave=fastest_wave,
        mesh=mesh,
        metrics=metrics,
        energy_source=None,
    )
    assert call_counts["active"] == 2


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


def test_electron_parallel_force_terms_jax_branch_matches_numpy_and_jvp() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh, metrics = _mesh_and_metrics()
    pressure = np.array([[[3.0], [4.0], [6.0]]], dtype=np.float64)
    electron_density = np.array([[[2.0], [2.5], [3.0]]], dtype=np.float64)
    electron_momentum_source = np.full((1, 3, 1), 0.25, dtype=np.float64)
    ion_density = {"d+": np.array([[[1.0], [1.5], [2.0]]], dtype=np.float64)}
    ion_momentum_source = {"d+": np.full((1, 3, 1), -0.5, dtype=np.float64)}
    numpy_terms = assemble_electron_parallel_force_terms(
        electron_pressure=pressure,
        electron_density=electron_density,
        electron_momentum_source=electron_momentum_source,
        ion_density=ion_density,
        ion_charge={"d+": 1.0},
        ion_momentum_source=ion_momentum_source,
        mesh=mesh,
        metrics=metrics,
    )
    jax_terms = assemble_electron_parallel_force_terms(
        electron_pressure=jnp.asarray(pressure),
        electron_density=jnp.asarray(electron_density),
        electron_momentum_source=jnp.asarray(electron_momentum_source),
        ion_density={"d+": jnp.asarray(ion_density["d+"])},
        ion_charge={"d+": 1.0},
        ion_momentum_source={"d+": jnp.asarray(ion_momentum_source["d+"])},
        mesh=mesh,
        metrics=metrics,
    )

    np.testing.assert_allclose(np.asarray(jax_terms.epar), numpy_terms.epar, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(jax_terms.ion_momentum_source["d+"]),
        numpy_terms.ion_momentum_source["d+"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    weights = jnp.asarray([[[0.3], [0.8], [1.4]]], dtype=jnp.float64)

    def qoi(scale):
        terms = assemble_electron_parallel_force_terms(
            electron_pressure=scale * jnp.asarray(pressure),
            electron_density=jnp.asarray(electron_density),
            electron_momentum_source=jnp.asarray(electron_momentum_source),
            ion_density={"d+": jnp.asarray(ion_density["d+"])},
            ion_charge={"d+": 1.0},
            ion_momentum_source={"d+": jnp.asarray(ion_momentum_source["d+"])},
            mesh=mesh,
            metrics=metrics,
        )
        return jnp.sum(terms.epar * weights) + 0.1 * jnp.sum(terms.ion_momentum_source["d+"] * weights)

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
