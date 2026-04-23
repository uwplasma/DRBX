from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import jax_drb.native.runner as native_runner


class _FakeConfig:
    def __init__(self, sections: dict[str, dict[str, object]]) -> None:
        self.sections = sections

    def has_section(self, section: str) -> bool:
        return section in self.sections

    def has_option(self, section: str, key: str) -> bool:
        return section in self.sections and key in self.sections[section]

    def parsed(self, section: str, key: str) -> object:
        return self.sections[section][key]


def _component(section: str, implementation: str):
    return SimpleNamespace(section=section, implementation=implementation)


def _metrics(*, j: float = 1.0, g22: float = 1.0, g23: float = 0.0, dy: np.ndarray | None = None):
    dy_array = np.ones((1, 4, 1), dtype=np.float64) if dy is None else dy
    return SimpleNamespace(
        J=np.full((1, 4, 1), j, dtype=np.float64),
        g22=np.full((1, 4, 1), g22, dtype=np.float64),
        g23=np.full((1, 4, 1), g23, dtype=np.float64),
        dy=dy_array,
    )


def _mesh(**overrides: object):
    values = {
        "nx": 1,
        "ny": 4,
        "nz": 1,
        "mxg": 2,
        "myg": 2,
        "xstart": 2,
        "xend": 2,
        "ystart": 1,
        "yend": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _mms_config(**section_overrides: dict[str, object]) -> _FakeConfig:
    sections = {
        "i": {"thermal_conduction": False, "p_div_v": False},
        "Ni": {"solution": "N", "source": "SN"},
        "Pi": {"solution": "P", "source": "SP"},
        "NVi": {"solution": "NV", "source": "SNV"},
    }
    for section, values in section_overrides.items():
        sections[section] = values
    return _FakeConfig(sections)


def test_uniform_identity_parallel_metric_checks_each_metric() -> None:
    mesh = _mesh()
    assert native_runner._uniform_identity_parallel_metric(mesh, metrics=_metrics()) is True
    assert native_runner._uniform_identity_parallel_metric(mesh, metrics=_metrics(j=2.0)) is False
    assert native_runner._uniform_identity_parallel_metric(mesh, metrics=_metrics(g22=2.0)) is False
    assert native_runner._uniform_identity_parallel_metric(mesh, metrics=_metrics(g23=0.1)) is False
    assert native_runner._uniform_identity_parallel_metric(
        mesh,
        metrics=_metrics(dy=np.array([[[1.0], [1.0], [2.0], [1.0]]], dtype=np.float64)),
    ) is False


def test_periodic_fluid_mms_support_accepts_only_exact_supported_contract() -> None:
    run_config = SimpleNamespace(
        components=(
            _component("i", "evolve_density"),
            _component("i", "evolve_pressure"),
            _component("i", "evolve_momentum"),
        ),
        mesh=SimpleNamespace(nx=1, nz=1),
        solver=SimpleNamespace(mms=True),
    )
    mesh = _mesh(nx=1, nz=1)
    assert native_runner._is_supported_periodic_fluid_mms_case(_mms_config(), run_config, mesh, _metrics()) is True

    bad_components = SimpleNamespace(
        components=(_component("i", "evolve_density"),),
        mesh=run_config.mesh,
        solver=run_config.solver,
    )
    assert native_runner._is_supported_periodic_fluid_mms_case(_mms_config(), bad_components, mesh, _metrics()) is False

    split_sections = SimpleNamespace(
        components=(
            _component("i", "evolve_density"),
            _component("e", "evolve_pressure"),
            _component("i", "evolve_momentum"),
        ),
        mesh=run_config.mesh,
        solver=run_config.solver,
    )
    assert native_runner._is_supported_periodic_fluid_mms_case(_mms_config(), split_sections, mesh, _metrics()) is False
    assert native_runner._is_supported_periodic_fluid_mms_case(
        _mms_config(),
        SimpleNamespace(components=run_config.components, mesh=SimpleNamespace(nx=2, nz=1), solver=run_config.solver),
        mesh,
        _metrics(),
    ) is False
    assert native_runner._is_supported_periodic_fluid_mms_case(
        _mms_config(),
        SimpleNamespace(components=run_config.components, mesh=run_config.mesh, solver=SimpleNamespace(mms=False)),
        mesh,
        _metrics(),
    ) is False
    assert native_runner._is_supported_periodic_fluid_mms_case(
        _mms_config(i={"thermal_conduction": True, "p_div_v": False}),
        run_config,
        mesh,
        _metrics(),
    ) is False
    assert native_runner._is_supported_periodic_fluid_mms_case(
        _mms_config(i={"thermal_conduction": False, "p_div_v": True}),
        run_config,
        mesh,
        _metrics(),
    ) is False
    assert native_runner._is_supported_periodic_fluid_mms_case(_mms_config(Ni={}), run_config, mesh, _metrics()) is False


def test_electrostatic_vorticity_support_rejects_unsupported_options() -> None:
    config = _FakeConfig(
        {
            "Vort": {"function": "vort"},
            "vorticity": {
                "diamagnetic": False,
                "diamagnetic_polarisation": False,
                "bndry_flux": False,
                "poloidal_flows": False,
                "split_n0": False,
                "phi_dissipation": False,
                "vort_dissipation": False,
                "collisional_friction": False,
                "phi_boundary_relax": False,
                "phi_sheath_dissipation": False,
                "damp_core_vorticity": False,
                "exb_advection": True,
                "exb_advection_simplified": True,
            },
        }
    )
    run_config = SimpleNamespace(components=(_component("vorticity", "vorticity"),), mesh=SimpleNamespace(ny=1, myg=0))
    mesh = _mesh(mxg=2)
    assert native_runner._is_supported_electrostatic_vorticity_case(config, run_config, mesh, _metrics()) is True

    assert native_runner._is_supported_electrostatic_vorticity_case(
        config,
        SimpleNamespace(components=(_component("i", "evolve_density"),), mesh=run_config.mesh),
        mesh,
        _metrics(),
    ) is False
    assert native_runner._is_supported_electrostatic_vorticity_case(
        config,
        SimpleNamespace(components=run_config.components, mesh=SimpleNamespace(ny=2, myg=0)),
        mesh,
        _metrics(),
    ) is False
    assert native_runner._is_supported_electrostatic_vorticity_case(config, run_config, _mesh(mxg=1), _metrics()) is False
    assert native_runner._is_supported_electrostatic_vorticity_case(_FakeConfig({"vorticity": {}}), run_config, mesh, _metrics()) is False
    bad_option = _FakeConfig({**config.sections, "vorticity": {**config.sections["vorticity"], "diamagnetic": True}})
    assert native_runner._is_supported_electrostatic_vorticity_case(bad_option, run_config, mesh, _metrics()) is False
    bad_exb = _FakeConfig({**config.sections, "vorticity": {**config.sections["vorticity"], "exb_advection": False}})
    assert native_runner._is_supported_electrostatic_vorticity_case(bad_exb, run_config, mesh, _metrics()) is False
    assert native_runner._is_supported_electrostatic_vorticity_case(config, run_config, mesh, _metrics(g23=0.1)) is False


def test_neutral_mixed_support_requires_single_neutral_component() -> None:
    assert native_runner._is_supported_neutral_mixed_case(
        SimpleNamespace(components=(_component("h", "neutral_mixed"),))
    ) is True
    assert native_runner._is_supported_neutral_mixed_case(
        SimpleNamespace(components=(_component("h", "neutral_mixed"), _component("d", "neutral_mixed")))
    ) is False
    assert native_runner._is_supported_neutral_mixed_case(
        SimpleNamespace(components=(_component("h", "evolve_density"),))
    ) is False


def test_drift_wave_support_checks_component_sequence_geometry_and_options() -> None:
    expected = (
        ("i", "evolve_density"),
        ("i", "fixed_velocity"),
        ("i", "fixed_temperature"),
        ("e", "quasineutral"),
        ("e", "evolve_momentum"),
        ("e", "fixed_temperature"),
        ("vorticity", "vorticity"),
        ("sound_speed", "sound_speed"),
        ("braginskii_collisions", "braginskii_collisions"),
        ("braginskii_friction", "braginskii_friction"),
        ("braginskii_heat_exchange", "braginskii_heat_exchange"),
    )
    run_config = SimpleNamespace(components=tuple(_component(section, impl) for section, impl in expected))
    config = _FakeConfig({"i": {"charge": 1.0}, "e": {"charge": -1.0}, "Ni": {"function": "Ni"}})
    mesh = _mesh(mxg=2, myg=2, xstart=2, xend=2)
    assert native_runner._is_supported_drift_wave_case(config, run_config, mesh, _metrics()) is True

    assert native_runner._is_supported_drift_wave_case(config, SimpleNamespace(components=run_config.components[:-1]), mesh, _metrics()) is False
    wrong_section = SimpleNamespace(components=(_component("d", "evolve_density"), *run_config.components[1:]))
    assert native_runner._is_supported_drift_wave_case(config, wrong_section, mesh, _metrics()) is False
    assert native_runner._is_supported_drift_wave_case(config, run_config, _mesh(mxg=1, myg=2), _metrics()) is False
    assert native_runner._is_supported_drift_wave_case(config, run_config, _mesh(mxg=2, myg=2, xstart=2, xend=3), _metrics()) is False
    assert native_runner._is_supported_drift_wave_case(
        _FakeConfig({"i": {"charge": 2.0}, "e": {"charge": -1.0}, "Ni": {"function": "Ni"}}),
        run_config,
        mesh,
        _metrics(),
    ) is False
    assert native_runner._is_supported_drift_wave_case(_FakeConfig({"i": {"charge": 1.0}, "e": {"charge": -1.0}}), run_config, mesh, _metrics()) is False
    assert native_runner._is_supported_drift_wave_case(config, run_config, mesh, _metrics(g23=0.1)) is False


def test_blob2d_support_checks_component_sequence_geometry_and_options() -> None:
    run_config = SimpleNamespace(
        components=(
            _component("e", "evolve_density"),
            _component("e", "isothermal"),
            _component("vorticity", "vorticity"),
            _component("sheath_closure", "sheath_closure"),
        )
    )
    config = _FakeConfig(
        {
            "Ne": {"function": "Ne"},
            "vorticity": {
                "diamagnetic": True,
                "diamagnetic_polarisation": False,
                "bndry_flux": False,
                "poloidal_flows": False,
                "split_n0": False,
                "phi_dissipation": False,
            },
        }
    )
    mesh = _mesh(myg=0, nz=2)
    assert native_runner._is_supported_blob2d_case(config, run_config, mesh, _metrics()) is True

    assert native_runner._is_supported_blob2d_case(config, SimpleNamespace(components=run_config.components[:-1]), mesh, _metrics()) is False
    assert native_runner._is_supported_blob2d_case(
        config,
        SimpleNamespace(components=(_component("i", "evolve_density"), *run_config.components[1:])),
        mesh,
        _metrics(),
    ) is False
    assert native_runner._is_supported_blob2d_case(config, run_config, _mesh(myg=1, nz=2), _metrics()) is False
    assert native_runner._is_supported_blob2d_case(config, run_config, _mesh(myg=0, nz=1), _metrics()) is False
    assert native_runner._is_supported_blob2d_case(_FakeConfig({"vorticity": config.sections["vorticity"]}), run_config, mesh, _metrics()) is False
    bad_option = _FakeConfig({**config.sections, "vorticity": {**config.sections["vorticity"], "bndry_flux": True}})
    assert native_runner._is_supported_blob2d_case(bad_option, run_config, mesh, _metrics()) is False
    assert native_runner._is_supported_blob2d_case(config, run_config, mesh, _metrics(g23=0.1)) is False
