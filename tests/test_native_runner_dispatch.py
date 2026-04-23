from __future__ import annotations

from pathlib import Path

import pytest

import jax_drb.parity.reference as reference_module
from jax_drb.native import runner as native_runner
from jax_drb.reference.cases import ReferenceCase


_CURATED_DISPATCH_CASES = (
    ("alfven_wave_rhs", "_run_alfven_wave_rhs_case"),
    ("alfven_wave_one_step", "_run_alfven_wave_one_step_case"),
    ("alfven_wave_short_window", "_run_alfven_wave_short_window_case"),
    ("alfven_wave_medium_window", "_run_alfven_wave_medium_window_case"),
    ("annulus_he_emag_rhs", "_run_annulus_he_emag_rhs_case"),
    ("annulus_he_emag_one_step", "_run_annulus_he_emag_one_step_case"),
    ("annulus_he_emag_short_window", "_run_annulus_he_emag_short_window_case"),
    ("tokamak_diffusion_flow_one_step", "_run_tokamak_diffusion_flow_one_step_case"),
    ("tokamak_diffusion_one_step", "_run_tokamak_diffusion_one_step_case"),
    ("tokamak_diffusion_transport_one_step", "_run_tokamak_diffusion_transport_one_step_case"),
    ("tokamak_diffusion_transport_short_window", "_run_tokamak_diffusion_transport_short_window_case"),
    ("tokamak_heat_transport_one_step", "_run_tokamak_heat_transport_one_step_case"),
    ("tokamak_heat_transport_short_window", "_run_tokamak_heat_transport_short_window_case"),
    ("tokamak_diffusion_conduction_one_step", "_run_tokamak_diffusion_conduction_one_step_case"),
    ("tokamak_diffusion_conduction_short_window", "_run_tokamak_diffusion_conduction_short_window_case"),
    ("tokamak_linear_transport_one_step", "_run_tokamak_linear_transport_one_step_case"),
    ("tokamak_linear_transport_short_window", "_run_tokamak_linear_transport_short_window_case"),
    ("tokamak_isothermal_rhs", "_run_tokamak_isothermal_rhs_case"),
    ("tokamak_isothermal_one_step", "_run_tokamak_isothermal_one_step_case"),
    ("tokamak_isothermal_short_window", "_run_tokamak_isothermal_short_window_case"),
    ("tokamak_isothermal_medium_window", "_run_tokamak_isothermal_medium_window_case"),
    ("tokamak_turbulence_rhs", "_run_tokamak_turbulence_rhs_case"),
    ("tokamak_turbulence_one_step", "_run_tokamak_turbulence_one_step_case"),
    ("tokamak_turbulence_short_window", "_run_tokamak_turbulence_short_window_case"),
    ("integrated_2d_recycling_rhs", "_run_integrated_2d_recycling_rhs_case"),
    ("tokamak_recycling_rhs", "_run_tokamak_recycling_rhs_case"),
    ("tokamak_recycling_dthe_rhs", "_run_tokamak_recycling_rhs_case"),
    ("tokamak_recycling_dthe_drifts_rhs", "_run_tokamak_recycling_rhs_case"),
    ("tokamak_recycling_dthene_rhs", "_run_tokamak_recycling_rhs_case"),
    ("integrated_2d_production_rhs", "_run_integrated_2d_recycling_rhs_case"),
    ("tokamak_recycling_one_step", "_run_tokamak_recycling_one_step_case"),
    ("tokamak_recycling_dthe_one_step", "_run_tokamak_recycling_one_step_case"),
    ("tokamak_recycling_dthe_drifts_one_step", "_run_tokamak_recycling_one_step_case"),
    ("tokamak_recycling_dthene_one_step", "_run_tokamak_recycling_one_step_case"),
    ("integrated_2d_production_one_step", "_run_integrated_2d_recycling_one_step_case"),
    ("integrated_2d_production_short_window", "_run_integrated_2d_recycling_short_window_case"),
    ("integrated_2d_production_medium_window", "_run_integrated_2d_recycling_medium_window_case"),
    ("integrated_2d_recycling_one_step", "_run_integrated_2d_recycling_one_step_case"),
    ("integrated_2d_recycling_short_window", "_run_integrated_2d_recycling_short_window_case"),
    ("integrated_2d_recycling_medium_window", "_run_integrated_2d_recycling_medium_window_case"),
)


def _case(name: str) -> ReferenceCase:
    return ReferenceCase(
        name=name,
        stage="unit",
        reference_path=f"cases/{name}/BOUT.inp",
        parity_mode="one_step",
        rationale="Unit-test curated dispatch target.",
        compare_variables=("Ne",),
    )


@pytest.mark.parametrize(("case_name", "helper_name"), _CURATED_DISPATCH_CASES)
def test_run_curated_case_dispatches_each_curated_case(
    case_name: str,
    helper_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(case_name)
    input_path = tmp_path / case.reference_path
    reference_root = tmp_path / "reference-root"
    manifest_path = tmp_path / "manifest.toml"
    sentinel = object()
    calls: list[tuple[ReferenceCase, Path, Path]] = []

    def fake_resolve_reference_case(
        requested_case_name: str,
        *,
        reference_root: Path,
        manifest_path: Path | None = None,
    ) -> tuple[ReferenceCase, Path]:
        assert requested_case_name == case_name
        assert reference_root == reference_root_path
        assert manifest_path == manifest_path_arg
        return case, input_path

    def fake_helper(case_arg: ReferenceCase, *, input_path: Path, reference_root: Path) -> object:
        calls.append((case_arg, input_path, reference_root))
        return sentinel

    reference_root_path = reference_root
    manifest_path_arg = manifest_path
    monkeypatch.setattr(reference_module, "resolve_reference_case", fake_resolve_reference_case)
    monkeypatch.setattr(native_runner, helper_name, fake_helper)

    result = native_runner.run_curated_case(case_name, reference_root=reference_root, manifest_path=manifest_path)

    assert result is sentinel
    assert calls == [(case, input_path, reference_root)]


def test_run_curated_case_falls_back_to_generic_config_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case("evolve_density_rhs")
    input_path = tmp_path / case.reference_path
    reference_root = tmp_path / "reference-root"
    config = object()
    sentinel = object()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        reference_module,
        "resolve_reference_case",
        lambda *args, **kwargs: (case, input_path),
    )
    monkeypatch.setattr(native_runner, "_load_curated_case_config", lambda *args, **kwargs: config)

    def fake_run_config_case(config_arg: object, **kwargs: object) -> object:
        calls["config"] = config_arg
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(native_runner, "run_config_case", fake_run_config_case)

    result = native_runner.run_curated_case(case.name, reference_root=reference_root)

    assert result is sentinel
    assert calls["config"] is config
    assert calls["case_name"] == "evolve_density_rhs"
    assert calls["parity_mode"] == "one_step"
    assert calls["compare_variables"] == ("Ne",)
    assert calls["reference_case"] is case


def test_runner_cache_path_wrappers_use_committed_snapshot_cache_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_runner, "_REFERENCE_SNAPSHOT_CACHE_DIR", tmp_path)

    assert native_runner._integrated_2d_snapshot_cache_path("case") == tmp_path / "case_snapshot.npz"
    assert native_runner._integrated_2d_optional_history_cache_path("case") == tmp_path / "case_optional_history.npz"
    assert native_runner._open_field_snapshot_cache_path("case") == tmp_path / "case_snapshot.npz"
    assert native_runner._tokamak_snapshot_cache_path("case") == tmp_path / "case_snapshot.npz"
    assert native_runner._tokamak_field_history_cache_path("case") == tmp_path / "case_field_history.npz"


_THIN_WRAPPER_CASES = (
    (
        "_run_alfven_wave_rhs_case",
        "_run_alfven_wave_dump_case",
        {"time_indices": (0,), "field_names": ("Apar", "phi", "Vort", "NVe", "Ne", "Ni"), "optional_field_names": ("ddt(NVe)", "ddt(Vort)")},
    ),
    (
        "_run_alfven_wave_one_step_case",
        "_run_alfven_wave_dump_case",
        {"time_indices": (0, 1), "field_names": ("Apar", "phi", "Vort", "NVe", "Ne", "Ni"), "optional_field_names": ()},
    ),
    (
        "_run_alfven_wave_short_window_case",
        "_run_alfven_wave_dump_case",
        {"time_indices": None, "field_names": ("Apar", "phi", "Vort", "NVe", "Ne", "Ni"), "optional_field_names": ()},
    ),
    (
        "_run_alfven_wave_medium_window_case",
        "_run_alfven_wave_dump_case",
        {"time_indices": None, "field_names": ("Apar", "phi", "Vort", "NVe", "Ne", "Ni"), "optional_field_names": ()},
    ),
    (
        "_run_annulus_he_emag_rhs_case",
        "_run_annulus_he_emag_dump_case",
        {
            "time_indices": (0,),
            "field_names": ("Apar", "Ne", "Nhe+", "NVe", "NVhe+"),
            "optional_field_names": ("ddt(Ne)", "ddt(NVe)", "ddt(Vort)"),
        },
    ),
    (
        "_run_annulus_he_emag_one_step_case",
        "_run_annulus_he_emag_dump_case",
        {"time_indices": (0, 1), "field_names": ("Apar", "Ne", "Nhe+", "NVe", "NVhe+", "phi", "Vort"), "optional_field_names": ()},
    ),
    (
        "_run_annulus_he_emag_short_window_case",
        "_run_annulus_he_emag_dump_case",
        {"time_indices": None, "field_names": ("Apar", "Ne", "Nhe+", "NVe", "NVhe+", "phi", "Vort"), "optional_field_names": ()},
    ),
    ("_run_tokamak_diffusion_flow_one_step_case", "_run_tokamak_dump_case", {"time_indices": (0, 1), "field_names": ("Nh", "Ph", "NVh")}),
    ("_run_tokamak_diffusion_one_step_case", "_run_tokamak_dump_case", {"time_indices": (0, 1), "field_names": ("Nh",)}),
    (
        "_run_tokamak_diffusion_transport_one_step_case",
        "_run_tokamak_dump_case",
        {"time_indices": (0, 1), "field_names": ("Nh+", "Ph+", "NVh+", "Pe")},
    ),
    (
        "_run_tokamak_diffusion_transport_short_window_case",
        "_run_tokamak_dump_case",
        {"time_indices": None, "field_names": ("Nh+", "Ph+", "NVh+", "Pe")},
    ),
    ("_run_tokamak_heat_transport_one_step_case", "_run_tokamak_dump_case", {"time_indices": (0, 1), "field_names": ("Pe",)}),
    ("_run_tokamak_heat_transport_short_window_case", "_run_tokamak_dump_case", {"time_indices": None, "field_names": ("Pe",)}),
    (
        "_run_tokamak_diffusion_conduction_one_step_case",
        "_run_tokamak_dump_case",
        {"time_indices": (0, 1), "field_names": ("Nh+", "Ph+", "Pe")},
    ),
    (
        "_run_tokamak_diffusion_conduction_short_window_case",
        "_run_tokamak_dump_case",
        {"time_indices": None, "field_names": ("Nh+", "Ph+", "Pe")},
    ),
    ("_run_tokamak_linear_transport_one_step_case", "_run_tokamak_dump_case", {"time_indices": (0, 1), "field_names": ("Pe",)}),
    ("_run_tokamak_linear_transport_short_window_case", "_run_tokamak_dump_case", {"time_indices": None, "field_names": ("Pe",)}),
    (
        "_run_tokamak_isothermal_rhs_case",
        "_run_tokamak_dump_case",
        {"time_indices": (0,), "field_names": ("Ne", "Ni", "NVe", "NVi", "phi", "Vort", "ddt(Ne)", "ddt(NVe)", "ddt(NVi)", "ddt(Vort)")},
    ),
    (
        "_run_tokamak_isothermal_one_step_case",
        "_run_tokamak_dump_case",
        {"time_indices": (0, 1), "field_names": ("Ne", "Ni", "NVe", "NVi", "phi", "Vort")},
    ),
    (
        "_run_tokamak_isothermal_short_window_case",
        "_run_tokamak_dump_case",
        {"time_indices": None, "field_names": ("Ne", "Ni", "NVe", "NVi", "phi", "Vort")},
    ),
    (
        "_run_tokamak_isothermal_medium_window_case",
        "_run_tokamak_dump_case",
        {"time_indices": None, "field_names": ("Ne", "Ni", "NVe", "NVi", "phi", "Vort")},
    ),
    (
        "_run_tokamak_turbulence_one_step_case",
        "_run_tokamak_dump_case",
        {"time_indices": (0, 1), "field_names": ("Ne", "Nd+", "NVe", "NVd+", "Pe", "Pd+", "phi", "Vort")},
    ),
    (
        "_run_tokamak_turbulence_rhs_case",
        "_run_tokamak_dump_case",
        {"time_indices": (0,), "field_names": ("Ne", "Nd+", "NVe", "NVd+", "Pe", "Pd+", "phi", "Vort", "ddt(Ne)", "ddt(NVe)", "ddt(Pe)")},
    ),
    (
        "_run_tokamak_turbulence_short_window_case",
        "_run_tokamak_dump_case",
        {"time_indices": None, "field_names": ("Ne", "Nd+", "NVe", "NVd+", "Pe", "Pd+", "phi", "Vort")},
    ),
)


@pytest.mark.parametrize(("wrapper_name", "delegate_name", "expected_kwargs"), _THIN_WRAPPER_CASES)
def test_thin_runner_wrappers_forward_expected_dump_requests(
    wrapper_name: str,
    delegate_name: str,
    expected_kwargs: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(wrapper_name)
    input_path = tmp_path / "BOUT.inp"
    reference_root = tmp_path / "reference"
    sentinel = object()
    calls: list[tuple[ReferenceCase, dict[str, object]]] = []

    def fake_delegate(case_arg: ReferenceCase, **kwargs: object) -> object:
        calls.append((case_arg, kwargs))
        return sentinel

    monkeypatch.setattr(native_runner, delegate_name, fake_delegate)

    result = getattr(native_runner, wrapper_name)(case, input_path=input_path, reference_root=reference_root)

    assert result is sentinel
    assert calls[0][0] is case
    assert calls[0][1]["input_path"] == input_path
    assert calls[0][1]["reference_root"] == reference_root
    for key, value in expected_kwargs.items():
        assert calls[0][1][key] == value


def test_tokamak_recycling_rhs_wrapper_reuses_integrated_recycling_rhs_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case("tokamak_recycling_rhs")
    input_path = tmp_path / "BOUT.inp"
    reference_root = tmp_path / "reference"
    sentinel = object()
    calls: list[tuple[ReferenceCase, Path, Path]] = []

    def fake_integrated(case_arg: ReferenceCase, *, input_path: Path, reference_root: Path) -> object:
        calls.append((case_arg, input_path, reference_root))
        return sentinel

    monkeypatch.setattr(native_runner, "_run_integrated_2d_recycling_rhs_case", fake_integrated)

    result = native_runner._run_tokamak_recycling_rhs_case(case, input_path=input_path, reference_root=reference_root)

    assert result is sentinel
    assert calls == [(case, input_path, reference_root)]
