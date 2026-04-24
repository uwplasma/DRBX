from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
from jax_drb.native.mesh import StructuredMesh, build_structured_mesh
from jax_drb.native.metrics import StructuredMetrics, build_structured_metrics
from jax_drb.native.neutral_mixed import _grad_par_open
from jax_drb.native.recycling_collision_closure import (
    apply_collision_closure,
    conduction_collision_time,
    conduction_kappa_coefficient,
    ion_thermal_force_pair,
    momentum_coefficient,
    parallel_ion_viscous_stress_open,
    thermal_force_enabled,
)
from jax_drb.native.recycling_collisions import compute_collision_frequencies
from jax_drb.native.recycling_reactions import charge_exchange_collision_rates
from jax_drb.native.recycling_setup import initialize_species
from jax_drb.native.recycling_1d import _prepare_open_field_states
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars


_DTHE_INPUT = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling-dthe/data/BOUT.inp")


class _MiniConfig:
    def __init__(self, sections: dict[str, dict[str, object]] | None = None) -> None:
        self._sections = sections or {}

    def has_section(self, section: str) -> bool:
        return section in self._sections

    def has_option(self, section: str, key: str) -> bool:
        return key in self._sections.get(section, {})

    def parsed(self, section: str, key: str) -> object:
        return self._sections[section][key]


def _field(value: float) -> np.ndarray:
    return np.asarray([[[value]]], dtype=np.float64)


def _species(
    name: str,
    *,
    charge: float,
    atomic_mass: float,
    has_pressure: bool = True,
    has_momentum: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        charge=charge,
        atomic_mass=atomic_mass,
        has_pressure=has_pressure,
        has_momentum=has_momentum,
        density=_field(1.0),
    )


def _prepared(
    *,
    density: float = 1.0,
    pressure: float = 1.0,
    temperature: float = 1.0,
    velocity: float = 0.0,
    momentum: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        density=_field(density),
        pressure=_field(pressure),
        temperature=_field(temperature),
        velocity=_field(velocity),
        momentum=_field(momentum),
    )


def _line_field(values: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(1, 3, 1)


def _line_prepared(*, temperature: tuple[float, float, float] = (1.0, 2.0, 4.0)) -> SimpleNamespace:
    return SimpleNamespace(
        density=_line_field((1.0, 1.0, 1.0)),
        pressure=_line_field(temperature),
        temperature=_line_field(temperature),
        velocity=_line_field((0.0, 0.0, 0.0)),
        momentum=_line_field((0.0, 0.0, 0.0)),
    )


def _line_mesh_and_metrics() -> tuple[StructuredMesh, StructuredMetrics]:
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
    return mesh, metrics


def test_momentum_coefficient_matches_braginskii_charge_branches() -> None:
    assert momentum_coefficient("e", -1.0, "d+", 1.0) == pytest.approx(0.51)
    assert momentum_coefficient("e", -1.0, "he+", 2.0) == pytest.approx(0.44)
    assert momentum_coefficient("ne+++", 3.0, "e", -1.0) == pytest.approx(0.40)
    assert momentum_coefficient("c++++", 4.0, "e", -1.0) == pytest.approx(0.38)
    assert momentum_coefficient("d+", 1.0, "t+", 1.0) == pytest.approx(1.0)


def test_ion_thermal_force_pair_covers_mass_ordering_and_skip_rules() -> None:
    mesh, metrics = _line_mesh_and_metrics()
    species = {
        "e": _species("e", charge=-1.0, atomic_mass=1.0),
        "d": _species("d", charge=0.0, atomic_mass=2.0),
        "d+": _species("d+", charge=1.0, atomic_mass=2.0),
        "t+": _species("t+", charge=1.0, atomic_mass=3.0),
        "ne+": _species("ne+", charge=1.0, atomic_mass=20.0),
    }
    prepared = {name: _line_prepared() for name in species}

    assert (
        ion_thermal_force_pair(
            "e",
            "d+",
            species=species,
            prepared=prepared,
            mesh=mesh,
            metrics=metrics,
            override_mass_restrictions=True,
        )
        is None
    )
    assert (
        ion_thermal_force_pair(
            "d",
            "d+",
            species=species,
            prepared=prepared,
            mesh=mesh,
            metrics=metrics,
            override_mass_restrictions=True,
        )
        is None
    )

    light_name, heavy_name, heavy_force = ion_thermal_force_pair(
        "d+",
        "ne+",
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        override_mass_restrictions=False,
    )
    assert (light_name, heavy_name) == ("d+", "ne+")
    assert heavy_force.shape == prepared["ne+"].density.shape

    light_name, heavy_name, _ = ion_thermal_force_pair(
        "ne+",
        "d+",
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        override_mass_restrictions=False,
    )
    assert (light_name, heavy_name) == ("d+", "ne+")

    assert (
        ion_thermal_force_pair(
            "d+",
            "t+",
            species=species,
            prepared=prepared,
            mesh=mesh,
            metrics=metrics,
            override_mass_restrictions=False,
        )
        is None
    )
    light_name, heavy_name, _ = ion_thermal_force_pair(
        "t+",
        "d+",
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        override_mass_restrictions=True,
    )
    assert (light_name, heavy_name) == ("d+", "t+")


def test_thermal_force_enabled_uses_default_when_option_missing() -> None:
    assert thermal_force_enabled(_MiniConfig(), "electron_ion", True) is True
    assert thermal_force_enabled(_MiniConfig({"braginskii_thermal_force": {}}), "ion_ion", False) is False
    assert (
        thermal_force_enabled(
            _MiniConfig({"braginskii_thermal_force": {"electron_ion": False}}),
            "electron_ion",
            True,
        )
        is False
    )


def test_conduction_collision_time_covers_braginskii_multispecies_and_afn_modes() -> None:
    species = {
        "e": _species("e", charge=-1.0, atomic_mass=1.0),
        "d+": _species("d+", charge=1.0, atomic_mass=2.0),
        "d": _species("d", charge=0.0, atomic_mass=2.0),
    }
    prepared = {name: _prepared(density=1.0) for name in species}
    rates = {
        ("e", "e"): _field(2.0),
        ("d+", "d+"): _field(4.0),
        ("d", "e"): _field(5.0),
        ("d", "d+"): _field(7.0),
        ("d", "d"): _field(11.0),
    }
    cx_rates = {"d": _field(3.0), "d+": _field(13.0)}

    electron_tau = conduction_collision_time(
        _MiniConfig({"e": {"conduction_collisions_mode": "braginskii"}}),
        species=species,
        prepared=prepared,
        collision_rates=rates,
        cx_rates=cx_rates,
        species_name="e",
    )
    ion_tau = conduction_collision_time(
        _MiniConfig({"d+": {"conduction_collisions_mode": "braginskii"}}),
        species=species,
        prepared=prepared,
        collision_rates=rates,
        cx_rates=cx_rates,
        species_name="d+",
    )
    neutral_afn_tau = conduction_collision_time(
        _MiniConfig({"d": {"conduction_collisions_mode": "afn"}}),
        species=species,
        prepared=prepared,
        collision_rates=rates,
        cx_rates=cx_rates,
        species_name="d",
    )
    default_multispecies_tau = conduction_collision_time(
        _MiniConfig(),
        species=species,
        prepared=prepared,
        collision_rates=rates,
        cx_rates=cx_rates,
        species_name="d+",
    )

    np.testing.assert_allclose(electron_tau, _field(0.5))
    np.testing.assert_allclose(ion_tau, _field(0.25))
    np.testing.assert_allclose(neutral_afn_tau, _field(1.0 / (5.0 + 7.0 + 3.0)))
    np.testing.assert_allclose(default_multispecies_tau, _field(1.0 / (4.0 + 13.0)))

    with pytest.raises(NotImplementedError, match="Neutral conduction_collisions_mode='braginskii'"):
        conduction_collision_time(
            _MiniConfig({"d": {"conduction_collisions_mode": "braginskii"}}),
            species=species,
            prepared=prepared,
            collision_rates=rates,
            cx_rates=cx_rates,
            species_name="d",
        )
    with pytest.raises(NotImplementedError, match="only supported for neutrals"):
        conduction_collision_time(
            _MiniConfig({"d+": {"conduction_collisions_mode": "afn"}}),
            species=species,
            prepared=prepared,
            collision_rates=rates,
            cx_rates=cx_rates,
            species_name="d+",
        )
    with pytest.raises(NotImplementedError, match="Unsupported conduction_collisions_mode"):
        conduction_collision_time(
            _MiniConfig({"d": {"conduction_collisions_mode": "other"}}),
            species=species,
            prepared=prepared,
            collision_rates=rates,
            cx_rates=cx_rates,
            species_name="d",
        )


def test_parallel_ion_viscous_stress_matches_braginskii_formula() -> None:
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
        Bxy=np.array([[[2.0], [4.0], [8.0]]], dtype=np.float64),
    )
    pressure = np.array([[[3.0], [5.0], [7.0]]], dtype=np.float64)
    tau = np.array([[[0.5], [0.25], [0.125]]], dtype=np.float64)
    velocity = np.array([[[1.0], [2.0], [4.0]]], dtype=np.float64)

    stress = parallel_ion_viscous_stress_open(
        pressure,
        tau,
        velocity,
        mesh=mesh,
        metrics=metrics,
    )

    expected = -0.96 * pressure * tau * (
        2.0 * _grad_par_open(velocity, mesh=mesh, metrics=metrics)
        + velocity * _grad_par_open(np.log(metrics.Bxy), mesh=mesh, metrics=metrics)
    )
    np.testing.assert_allclose(stress, expected)


def test_ion_thermal_force_pair_is_enabled_for_dt_when_mass_override_is_set() -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    species = initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    pair = ion_thermal_force_pair(
        "d+",
        "t+",
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        override_mass_restrictions=True,
    )

    assert pair is not None
    light_name, heavy_name, heavy_force = pair
    active = (mesh.xstart, mesh.yend, 0)

    assert light_name == "d+"
    assert heavy_name == "t+"
    assert np.isfinite(float(heavy_force[active]))
    assert heavy_force.shape == species["t+"].density.shape


def test_conduction_kappa_coefficient_uses_species_defaults_and_override() -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    species = initialize_species(config, mesh=mesh)

    assert conduction_kappa_coefficient(config, species["e"]) == pytest.approx(3.16 / math.sqrt(2.0))
    assert conduction_kappa_coefficient(config, species["d+"]) == pytest.approx(3.9)
    assert conduction_kappa_coefficient(config, species["d"]) == pytest.approx(2.5)

    override_config = apply_bout_overrides(load_bout_input(_DTHE_INPUT), ("d:kappa_coefficient=7.25",))
    assert conduction_kappa_coefficient(override_config, species["d"]) == pytest.approx(7.25)


def test_collision_closure_accepts_precomputed_collision_and_cx_rates() -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    species = initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )

    collision_rates = compute_collision_frequencies(config, species, prepared, dataset_scalars=scalars)
    cx_rates = charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=scalars,
    )

    baseline = apply_collision_closure(
        config,
        species,
        prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    reused = apply_collision_closure(
        config,
        species,
        prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        collision_rates=collision_rates,
        cx_rates=cx_rates,
    )

    for name in baseline.energy_source:
        np.testing.assert_allclose(reused.energy_source[name], baseline.energy_source[name])
    for name in baseline.momentum_source:
        np.testing.assert_allclose(reused.momentum_source[name], baseline.momentum_source[name])


def test_collision_closure_friction_lane_is_jax_jvp_transformable() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh, metrics = _line_mesh_and_metrics()
    config = _MiniConfig({"model": {"components": ("braginskii_friction", "braginskii_heat_exchange")}})
    species = {
        "d+": _species("d+", charge=1.0, atomic_mass=2.0),
        "t+": _species("t+", charge=1.0, atomic_mass=3.0),
    }
    species = {
        name: SimpleNamespace(**{**sp.__dict__, "density": jnp.asarray(sp.density, dtype=jnp.float64)})
        for name, sp in species.items()
    }
    rates = {("d+", "t+"): jnp.ones((1, 3, 1), dtype=jnp.float64) * 0.25}
    cx_rates: dict[str, object] = {}

    def qoi(scale):
        prepared = {
            "d+": SimpleNamespace(
                density=jnp.ones((1, 3, 1), dtype=jnp.float64),
                pressure=jnp.ones((1, 3, 1), dtype=jnp.float64),
                temperature=jnp.ones((1, 3, 1), dtype=jnp.float64),
                velocity=jnp.zeros((1, 3, 1), dtype=jnp.float64),
                momentum=jnp.zeros((1, 3, 1), dtype=jnp.float64),
            ),
            "t+": SimpleNamespace(
                density=jnp.ones((1, 3, 1), dtype=jnp.float64),
                pressure=2.0 * jnp.ones((1, 3, 1), dtype=jnp.float64),
                temperature=2.0 * jnp.ones((1, 3, 1), dtype=jnp.float64),
                velocity=scale * jnp.ones((1, 3, 1), dtype=jnp.float64),
                momentum=scale * jnp.ones((1, 3, 1), dtype=jnp.float64),
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
            cx_rates=cx_rates,
        )
        return jnp.sum(terms.momentum_source["d+"]) + 0.1 * jnp.sum(terms.energy_source["d+"])

    value, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))

    assert np.isfinite(float(value))
    assert np.isfinite(float(tangent))
    assert abs(float(tangent)) > 0.0
