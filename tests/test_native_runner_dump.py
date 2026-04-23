from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import jax_drb.native.runner as native_runner
from jax_drb.reference.cases import ReferenceCase


def _mesh() -> SimpleNamespace:
    return SimpleNamespace(
        nx=4,
        local_ny=4,
        nz=1,
        mxg=1,
        myg=1,
        xstart=1,
        xend=2,
        ystart=1,
        yend=2,
    )


def _snapshot(*, offset: float = 0.0, optional: bool = True) -> SimpleNamespace:
    shape = (4, 4, 1)
    base = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + offset
    fields = {
        "Ne": base + 1.0,
        "Ni": base + 2.0,
        "Nhe+": base + 3.0,
        "NVe": base + 4.0,
        "NVhe+": base + 5.0,
        "Vort": base + 6.0,
        "Pe": base + 7.0,
    }
    optional_fields = {}
    if optional:
        optional_fields = {
            "ddt(NVe)": base + 8.0,
            "ddt(Vort)": base + 9.0,
            "Vort": base + 10.0,
        }
    return SimpleNamespace(
        fields=fields,
        optional_fields=optional_fields,
        scalar_values={"Nnorm": 1.0, "Tnorm": 2.0, "Bnorm": 3.0, "Cs0": 4.0, "Omega_ci": 5.0, "rho_s0": 6.0},
        mesh=_mesh(),
        metrics=SimpleNamespace(name="metrics"),
    )


def _run_config(*, timestep: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        time=SimpleNamespace(nout=3, timestep=timestep),
        components=(SimpleNamespace(label="e:evolve_density"),),
    )


def _summary(*, time_points: tuple[float, ...] = (0.0, 0.5, 1.0)) -> SimpleNamespace:
    return SimpleNamespace(
        artifacts={"BOUT.dmp.0.nc": "/tmp/fake-dump.nc"},
        time_points=time_points,
        overrides=("nout=3",),
    )


def _case(name: str, *, compare_variables: tuple[str, ...], parity_mode: str = "one_step") -> ReferenceCase:
    return ReferenceCase(
        name=name,
        stage="unit",
        reference_path=f"cases/{name}/BOUT.inp",
        parity_mode=parity_mode,
        rationale="Unit-test dump helper.",
        capability_tier="scaffolded_reference_backed",
        compare_variables=compare_variables,
        trim_x_guards=False,
        trim_y_guards=False,
    )


def _patch_common_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native_runner, "load_bout_input", lambda path: object())
    monkeypatch.setattr(native_runner, "_load_curated_case_config", lambda case, input_path: object())
    monkeypatch.setattr(native_runner.RunConfiguration, "from_config", lambda config: _run_config())
    monkeypatch.setattr(native_runner, "resolved_dataset_scalars", lambda run_config: {"Nnorm": 10.0})
    monkeypatch.setattr(native_runner, "_prepare_compare_variables", lambda variables, mesh, **kwargs: variables)


def test_tokamak_dump_case_uses_committed_snapshot_and_history_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_config(monkeypatch)
    snapshot_cache = tmp_path / "tokamak_snapshot.npz"
    history_cache = tmp_path / "tokamak_history.npz"
    snapshot_cache.write_text("snapshot", encoding="utf-8")
    history_cache.write_text("history", encoding="utf-8")
    history = {
        "Ne": np.arange(12, dtype=np.float64).reshape(3, 4, 1, 1),
        "Pe": np.arange(12, 24, dtype=np.float64).reshape(3, 4, 1, 1),
    }
    monkeypatch.setattr(native_runner, "_tokamak_snapshot_cache_path", lambda case_name: snapshot_cache)
    monkeypatch.setattr(native_runner, "_tokamak_field_history_cache_path", lambda case_name: history_cache)
    monkeypatch.setattr(native_runner, "_uses_tokamak_snapshot_cache", lambda case_name: True)
    monkeypatch.setattr(native_runner, "_uses_tokamak_field_history_cache", lambda case_name: True)
    monkeypatch.setattr(native_runner, "load_local_reference_snapshot_cache", lambda *args, **kwargs: _snapshot())
    monkeypatch.setattr(native_runner, "load_optional_field_history_cache", lambda *args, **kwargs: history)

    result = native_runner._run_tokamak_dump_case(
        _case("tokamak_isothermal_one_step", compare_variables=("Ne",)),
        input_path=tmp_path / "BOUT.inp",
        reference_root=tmp_path,
        time_indices=(0, 2),
        field_names=("Ne", "Pe"),
    )

    assert result.time_points == (0.0, 1.0)
    np.testing.assert_allclose(result.variables["Ne"], history["Ne"][(0, 2), ...])
    assert result.payload["overrides"] == ["nout=1"]
    assert result.payload["dataset_scalars"]["Nnorm"] == 1.0


def test_tokamak_dump_case_rejects_empty_history_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_config(monkeypatch)
    snapshot_cache = tmp_path / "tokamak_snapshot.npz"
    history_cache = tmp_path / "tokamak_history.npz"
    snapshot_cache.write_text("snapshot", encoding="utf-8")
    history_cache.write_text("history", encoding="utf-8")
    monkeypatch.setattr(native_runner, "_tokamak_snapshot_cache_path", lambda case_name: snapshot_cache)
    monkeypatch.setattr(native_runner, "_tokamak_field_history_cache_path", lambda case_name: history_cache)
    monkeypatch.setattr(native_runner, "_uses_tokamak_snapshot_cache", lambda case_name: True)
    monkeypatch.setattr(native_runner, "_uses_tokamak_field_history_cache", lambda case_name: True)
    monkeypatch.setattr(native_runner, "load_local_reference_snapshot_cache", lambda *args, **kwargs: _snapshot())
    monkeypatch.setattr(native_runner, "load_optional_field_history_cache", lambda *args, **kwargs: {})

    with pytest.raises(ValueError, match="does not contain requested fields"):
        native_runner._run_tokamak_dump_case(
            _case("tokamak_isothermal_one_step", compare_variables=("Ne",)),
            input_path=tmp_path / "BOUT.inp",
            reference_root=tmp_path,
            time_indices=None,
            field_names=("Ne",),
        )


def test_tokamak_dump_case_falls_back_to_reference_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_config(monkeypatch)
    monkeypatch.setattr(native_runner, "_uses_tokamak_snapshot_cache", lambda case_name: False)
    monkeypatch.setattr(native_runner, "_uses_tokamak_field_history_cache", lambda case_name: False)
    monkeypatch.setattr(native_runner, "run_reference_case", lambda *args, **kwargs: SimpleNamespace(summary=_summary()))
    monkeypatch.setattr(
        native_runner,
        "load_local_reference_snapshot",
        lambda *args, time_index, **kwargs: _snapshot(offset=float(time_index)),
    )

    result = native_runner._run_tokamak_dump_case(
        _case("tokamak_turbulence_short_window", compare_variables=("Ne", "Pe"), parity_mode="short_window"),
        input_path=tmp_path / "BOUT.inp",
        reference_root=tmp_path,
        time_indices=None,
        field_names=("Ne", "Pe"),
    )

    assert result.time_points == (0.0, 0.5, 1.0)
    assert result.variables["Ne"].shape == (3, 4, 4, 1)
    assert result.payload["overrides"] == ["nout=3"]


def test_annulus_he_emag_dump_case_builds_native_em_quantities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_config(monkeypatch)
    monkeypatch.setattr(native_runner, "run_reference_case", lambda *args, **kwargs: SimpleNamespace(summary=_summary()))
    monkeypatch.setattr(
        native_runner,
        "load_local_reference_snapshot",
        lambda *args, time_index, **kwargs: _snapshot(offset=float(time_index)),
    )
    monkeypatch.setattr(
        native_runner,
        "extract_charged_species_metadata",
        lambda config: (SimpleNamespace(section="e"), SimpleNamespace(section="he+")),
    )
    monkeypatch.setattr(
        native_runner,
        "compute_parallel_current_density",
        lambda fields, species: np.full_like(next(iter(fields.values())), 11.0),
    )
    monkeypatch.setattr(
        native_runner,
        "compute_alpha_em",
        lambda fields, species: np.full_like(next(iter(fields.values())), 12.0),
    )

    result = native_runner._run_annulus_he_emag_dump_case(
        _case("annulus_he_emag_short_window", compare_variables=("Ajpar", "alpha_em"), parity_mode="short_window"),
        input_path=tmp_path / "BOUT.inp",
        reference_root=tmp_path,
        time_indices=(0, 2),
        field_names=("Ne", "Nhe+", "NVe", "NVhe+"),
        optional_field_names=("Vort",),
    )

    assert result.time_points == (0.0, 1.0)
    np.testing.assert_allclose(result.variables["Ajpar"], 11.0)
    np.testing.assert_allclose(result.variables["alpha_em"], 12.0)
    assert "Vort" in result.variables


def test_alfven_wave_dump_case_reconstructs_electromagnetic_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_config(monkeypatch)
    monkeypatch.setattr(native_runner, "run_reference_case", lambda *args, **kwargs: SimpleNamespace(summary=_summary()))
    monkeypatch.setattr(
        native_runner,
        "load_local_reference_snapshot",
        lambda *args, time_index, **kwargs: _snapshot(offset=float(time_index)),
    )
    monkeypatch.setattr(
        native_runner,
        "extract_charged_species_metadata",
        lambda config: (SimpleNamespace(section="e", current_factor=-2.0), SimpleNamespace(section="i", current_factor=1.0)),
    )
    monkeypatch.setattr(
        native_runner,
        "compute_parallel_current_density",
        lambda fields, species: np.full_like(next(iter(fields.values())), 2.0),
    )
    monkeypatch.setattr(native_runner, "compute_beta_em", lambda **kwargs: 1.0)
    monkeypatch.setattr(native_runner, "solve_slab_neumann_apar", lambda *args, **kwargs: np.full((4, 4, 1), 4.0))
    monkeypatch.setattr(
        native_runner,
        "invert_slab_neumann_apar_to_current_density",
        lambda *args, **kwargs: np.full((4, 4, 1), 6.0),
    )
    monkeypatch.setattr(native_runner, "compute_alfven_wave_ddt_nve_core", lambda *args, **kwargs: np.full((4, 4, 1), 8.0))
    monkeypatch.setattr(native_runner, "compute_alfven_wave_ddt_vort_core", lambda *args, **kwargs: np.full((4, 4, 1), 9.0))

    result = native_runner._run_alfven_wave_dump_case(
        _case(
            "alfven_wave_short_window",
            compare_variables=("Ajpar", "Apar", "NVe", "ddt(NVe)", "ddt(Vort)"),
            parity_mode="short_window",
        ),
        input_path=tmp_path / "BOUT.inp",
        reference_root=tmp_path,
        time_indices=(0, 1),
        field_names=("Ne", "Ni", "NVe", "Vort"),
        optional_field_names=("ddt(NVe)", "ddt(Vort)"),
    )

    assert result.time_points == (0.0, 0.5)
    assert "Ne" not in result.variables
    assert "Ni" not in result.variables
    np.testing.assert_allclose(result.variables["Apar"], 4.0)
    np.testing.assert_allclose(result.variables["NVe"][:, 1:3, 1:3, :], -3.0)
    np.testing.assert_allclose(result.variables["Ajpar"][:, 1:3, 1:3, :], 6.0)
    np.testing.assert_allclose(result.variables["ddt(NVe)"][:, 1:3, 1:3, :], 8.0)
    np.testing.assert_allclose(result.variables["ddt(Vort)"][:, 0, 1:3, :], 9.0)
    np.testing.assert_allclose(result.variables["ddt(Vort)"][:, 2, 1:3, :], 9.0)
