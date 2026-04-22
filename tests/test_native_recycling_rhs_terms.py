from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.recycling_rhs_terms import (
    ElectronPressureRhsTerms,
    IonRhsTerms,
    assemble_electron_pressure_rhs_terms,
    assemble_ion_rhs_terms,
)


def test_electron_pressure_rhs_terms_sum_to_total() -> None:
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
    metrics = StructuredMetrics(
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
    metrics = StructuredMetrics(
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
