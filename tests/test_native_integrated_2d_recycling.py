from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from jax import grad
from jax.errors import TracerArrayConversionError
import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.config.boutinp import load_bout_input, parse_bout_input
from jax_drb.native import runner as native_runner
from jax_drb.parity.arrays import build_array_payload_from_summary_payload, load_portable_array_payload
from jax_drb.parity.diff import build_scaled_array_diff_entries
from jax_drb.parity import reference as parity_reference
from jax_drb.native.reference_dump import (
    LocalReferenceSnapshot,
    save_local_reference_snapshot_cache,
    save_optional_field_history_cache,
)
from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.recycling_1d import (
    OpenFieldSpecies,
    _PreparedSpeciesState,
    _SimpleSheathSettings,
    _apply_electron_simple_sheath_boundary,
    _initialize_species,
    _prepare_open_field_states,
)
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.reference.cases import ReferenceCase, load_reference_cases

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_INPUT = Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-recycling/data/BOUT.inp")
_REFERENCE_ROOT = Path("/Users/rogerio/local/hermes-3")
_BASELINE_ARRAY_DIR = _REPO_ROOT / "references/baselines/reference_arrays"


def test_run_curated_case_dispatches_tokamak_recycling_rhs_to_dedicated_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = ReferenceCase(
        name="tokamak_recycling_dthe_rhs",
        stage="stage7",
        reference_path="examples/tokamak-2D/recycling-dthe/BOUT.inp",
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+"),
    )
    sentinel = object()

    monkeypatch.setattr(parity_reference, "resolve_reference_case", lambda *args, **kwargs: (case, _REFERENCE_INPUT))
    monkeypatch.setattr(native_runner, "_run_tokamak_recycling_rhs_case", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(
        native_runner,
        "_run_integrated_2d_recycling_rhs_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("integrated helper should not be dispatched")),
    )

    assert native_runner.run_curated_case(case.name, reference_root=_REFERENCE_ROOT) is sentinel


def test_run_curated_case_dispatches_tokamak_recycling_one_step_to_dedicated_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = ReferenceCase(
        name="tokamak_recycling_dthe_one_step",
        stage="stage7",
        reference_path="examples/tokamak-2D/recycling-dthe/BOUT.inp",
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+"),
    )
    sentinel = object()

    monkeypatch.setattr(parity_reference, "resolve_reference_case", lambda *args, **kwargs: (case, _REFERENCE_INPUT))
    monkeypatch.setattr(native_runner, "_run_tokamak_recycling_one_step_case", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(
        native_runner,
        "_run_integrated_2d_recycling_one_step_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("integrated helper should not be dispatched")),
    )

    assert native_runner.run_curated_case(case.name, reference_root=_REFERENCE_ROOT) is sentinel


def _reference_case_by_name(name: str) -> ReferenceCase:
    return next(case for case in load_reference_cases() if case.name == name)


def _run_integrated_2d_case_against_committed_baseline(case_name: str):
    case = _reference_case_by_name(case_name)
    input_path = _REFERENCE_ROOT / case.reference_path
    if case_name.endswith("_rhs"):
        result = native_runner._run_integrated_2d_recycling_rhs_case(
            case,
            input_path=input_path,
            reference_root=_REFERENCE_ROOT,
        )
    elif case_name.endswith("_one_step"):
        result = native_runner._run_integrated_2d_recycling_transient_case(
            case,
            input_path=input_path,
            reference_root=_REFERENCE_ROOT,
            steps=1,
        )
    elif case_name.endswith("_short_window"):
        result = native_runner._run_integrated_2d_recycling_transient_case(
            case,
            input_path=input_path,
            reference_root=_REFERENCE_ROOT,
            steps=5,
        )
    elif case_name.endswith("_medium_window"):
        result = native_runner._run_integrated_2d_recycling_medium_window_case(
            case,
            input_path=input_path,
            reference_root=_REFERENCE_ROOT,
        )
    else:
        raise ValueError(f"Unsupported committed-baseline test case {case_name!r}")

    expected = load_portable_array_payload(_BASELINE_ARRAY_DIR / f"{case_name}.npz")
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    entries = build_scaled_array_diff_entries(
        expected["variables"],
        actual["variables"],
        compare_variables=case.compare_variables,
    )
    return {entry.field: entry for entry in entries}


def _run_direct_tokamak_case_against_committed_baseline(case_name: str):
    case = _reference_case_by_name(case_name)
    input_path = _REFERENCE_ROOT / case.reference_path
    result = native_runner.run_curated_case(case_name, reference_root=_REFERENCE_ROOT)
    expected = load_portable_array_payload(_BASELINE_ARRAY_DIR / f"{case_name}.npz")
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    entries = build_scaled_array_diff_entries(
        expected["variables"],
        actual["variables"],
        compare_variables=case.compare_variables,
    )
    return {entry.field: entry for entry in entries}


def test_integrated_2d_production_one_step_prefers_bdf_solver() -> None:
    config = parse_bout_input(
        """
        [mesh]
        nx = 4
        ny = 4
        nz = 1

        [model]
        components = e, d+

        [e]
        type = quasineutral
        charge = -1

        [d+]
        type = evolve_density, evolve_pressure, evolve_momentum
        charge = 1
        """
    )
    assert native_runner._select_integrated_2d_transient_solver_mode(
        "integrated_2d_production_one_step",
        config=config,
        parity_mode="one_step",
    ) == "bdf"


@dataclass(frozen=True)
class _FakeSummary:
    artifacts: dict[str, str]
    time_points: tuple[float, ...] = ()


@dataclass(frozen=True)
class _FakeExecution:
    summary: _FakeSummary


def test_integrated_2d_recycling_rhs_uses_local_reference_dump(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("integrated 2D recycling reference input is unavailable")

    netcdf4 = pytest.importorskip("netCDF4")
    dump_path = tmp_path / "BOUT.dmp.0.nc"
    with netcdf4.Dataset(dump_path, "w") as dataset:
        dataset.createDimension("x", 4)
        dataset.createDimension("y", 5)
        dataset.createDimension("z", 1)
        dataset.createDimension("t", 1)

        for name, value in {
            "MXG": 1,
            "MYG": 1,
            "jyseps1_1": 0,
            "jyseps2_1": 2,
            "jyseps1_2": 2,
            "jyseps2_2": 2,
            "ny_inner": 3,
            "PE_YIND": 0,
            "NYPE": 2,
        }.items():
            variable = dataset.createVariable(name, "i4")
            variable.assignValue(value)

        field2d = np.ones((4, 5), dtype=np.float64)
        for name in ("dx", "dy", "J", "g11", "g22", "g_22", "g33", "g23", "Bxy"):
            variable = dataset.createVariable(name, "f8", ("x", "y"))
            variable[:] = field2d

        state_fields = {
            "Nd+": 2.0 * field2d,
            "Pd+": 3.0 * field2d,
            "NVd+": np.zeros_like(field2d),
            "Nd": np.zeros_like(field2d),
            "Pd": np.zeros_like(field2d),
            "NVd": np.zeros_like(field2d),
            "Pe": 3.0 * field2d,
        }
        for name, value in state_fields.items():
            variable = dataset.createVariable(name, "f8", ("t", "x", "y", "z"))
            variable[:] = value.reshape(1, 4, 5, 1)

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(summary=_FakeSummary(artifacts={"BOUT.dmp.0.nc": str(dump_path)})),
    )

    case = ReferenceCase(
        name="integrated_2d_recycling_rhs",
        stage="stage7",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "ddt(Nd+)", "ddt(Pe)"),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    result = native_runner._run_integrated_2d_recycling_rhs_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert result.payload["dimensions"] == {"t": 1, "x": 4, "y": 5, "z": 1}
    assert "ddt(Nd+)" in result.payload["variable_summaries"]
    assert result.variables["Nd+"].shape == (1, 2, 3, 1)


def test_integrated_2d_recycling_rhs_requests_auxiliary_dump_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("integrated 2D recycling reference input is unavailable")

    captured: dict[str, tuple[str, ...]] = {}

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=ones,
        Bxy=ones,
    )
    fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": np.ones((4, 5, 1), dtype=np.float64),
    }

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(summary=_FakeSummary(artifacts={"BOUT.dmp.0.nc": "/tmp/fake-dump.nc"})),
    )

    def fake_load_snapshot(dump_path, *, field_names, optional_field_names=(), scalar_names=(), time_index=0):
        del dump_path, time_index
        captured["field_names"] = tuple(field_names)
        captured["optional_field_names"] = tuple(optional_field_names)
        captured["scalar_names"] = tuple(scalar_names)
        return LocalReferenceSnapshot(
            mesh=mesh,
            metrics=metrics,
            fields=fields,
            optional_fields={
                "SNd+": np.zeros((4, 5, 1), dtype=np.float64),
                "SNVd+": np.zeros((4, 5, 1), dtype=np.float64),
                "SPd+": np.zeros((4, 5, 1), dtype=np.float64),
                "SNd": np.zeros((4, 5, 1), dtype=np.float64),
                "SNVd": np.zeros((4, 5, 1), dtype=np.float64),
                "SPd": np.zeros((4, 5, 1), dtype=np.float64),
                "SPe": np.zeros((4, 5, 1), dtype=np.float64),
                "Sd_target_recycle": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_target_recycle": np.zeros((4, 5, 1), dtype=np.float64),
                "Sd_wall_recycle": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_wall_recycle": np.zeros((4, 5, 1), dtype=np.float64),
                "Sd_pump": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_pump": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_target_refl": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_wall_refl": np.zeros((4, 5, 1), dtype=np.float64),
                "is_pump": np.zeros((4, 5, 1), dtype=np.float64),
            },
            scalar_values={"Nnorm": 1.0e17},
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)

    case = ReferenceCase(
        name="integrated_2d_recycling_rhs",
        stage="stage7",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "ddt(Nd+)"),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    native_runner._run_integrated_2d_recycling_rhs_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured["field_names"] == ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe")
    assert captured["optional_field_names"] == (
        "Ne",
        "Vd+",
        "Vd",
        "SNd+",
        "SNVd+",
        "SPd+",
        "SNd",
        "SNVd",
        "SPd",
        "SPe",
        "Sd_target_recycle",
        "Ed_target_recycle",
        "Sd_wall_recycle",
        "Ed_wall_recycle",
        "Sd_pump",
        "Ed_pump",
        "Ed_target_refl",
        "Ed_wall_refl",
        "is_pump",
        "anomalous_D_d+",
        "anomalous_Chi_d+",
        "anomalous_nu_d+",
        "anomalous_D_e",
        "anomalous_Chi_e",
        "anomalous_nu_e",
    )
    assert captured["scalar_names"] == ("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0")


def test_integrated_2d_production_rhs_preserves_only_ion_target_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_input = Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-production/data/BOUT.inp")
    if not production_input.exists():
        pytest.skip("integrated 2D production reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=ones,
        Bxy=ones,
    )
    fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": np.ones((4, 5, 1), dtype=np.float64),
    }

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(summary=_FakeSummary(artifacts={"BOUT.dmp.0.nc": "/tmp/fake-dump.nc"})),
    )
    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_snapshot_cache_path",
        lambda case_name: Path("/tmp") / f"{case_name}.missing",
    )

    monkeypatch.setattr(
        native_runner,
        "load_local_reference_snapshot",
        lambda *args, **kwargs: LocalReferenceSnapshot(
            mesh=mesh,
            metrics=metrics,
            fields=fields,
            optional_fields={
                "Vd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
                "Vd": np.full((4, 5, 1), 3.0, dtype=np.float64),
                "SNd+": np.zeros((4, 5, 1), dtype=np.float64),
                "SNVd+": np.zeros((4, 5, 1), dtype=np.float64),
                "SPd+": np.zeros((4, 5, 1), dtype=np.float64),
                "SNd": np.zeros((4, 5, 1), dtype=np.float64),
                "SNVd": np.zeros((4, 5, 1), dtype=np.float64),
                "SPd": np.zeros((4, 5, 1), dtype=np.float64),
                "SPe": np.zeros((4, 5, 1), dtype=np.float64),
                "Sd_target_recycle": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_target_recycle": np.zeros((4, 5, 1), dtype=np.float64),
                "Sd_wall_recycle": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_wall_recycle": np.zeros((4, 5, 1), dtype=np.float64),
                "Sd_pump": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_pump": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_target_refl": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_wall_refl": np.zeros((4, 5, 1), dtype=np.float64),
                "is_pump": np.zeros((4, 5, 1), dtype=np.float64),
            },
            scalar_values={"Nnorm": 1.0e17},
        ),
    )

    captured: dict[str, object] = {}

    def fake_rhs(*args, **kwargs):
        captured["preserve_dump_ion_target_state_only"] = kwargs["preserve_dump_ion_target_state_only"]
        captured["field_overrides"] = kwargs["field_overrides"]
        return SimpleNamespace(
            variables={
                "Nd+": fields["Nd+"][None, ...],
                "Pd+": fields["Pd+"][None, ...],
                "NVd+": fields["NVd+"][None, ...],
                "Nd": fields["Nd"][None, ...],
                "Pd": fields["Pd"][None, ...],
                "NVd": fields["NVd"][None, ...],
                "Pe": fields["Pe"][None, ...],
                "ddt(Nd+)": np.zeros((1, 4, 5, 1), dtype=np.float64),
                "ddt(Pd+)": np.zeros((1, 4, 5, 1), dtype=np.float64),
                "ddt(NVd+)": np.zeros((1, 4, 5, 1), dtype=np.float64),
                "ddt(Nd)": np.zeros((1, 4, 5, 1), dtype=np.float64),
                "ddt(Pd)": np.zeros((1, 4, 5, 1), dtype=np.float64),
                "ddt(NVd)": np.zeros((1, 4, 5, 1), dtype=np.float64),
                "ddt(Pe)": np.zeros((1, 4, 5, 1), dtype=np.float64),
                "Sd_target_recycle": np.zeros((1, 4, 5, 1), dtype=np.float64),
                "Ed_target_recycle": np.zeros((1, 4, 5, 1), dtype=np.float64),
            }
        )

    monkeypatch.setattr(native_runner, "compute_recycling_1d_rhs", fake_rhs)

    case = ReferenceCase(
        name="integrated_2d_production_rhs",
        stage="stage7",
        reference_path=str(production_input),
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "ddt(Nd+)", "ddt(Pe)"),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    native_runner._run_integrated_2d_recycling_rhs_case(
        case,
        input_path=production_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured["preserve_dump_ion_target_state_only"] is True
    expected_fields = native_runner._apply_species_velocity_overrides(
        load_bout_input(production_input),
        field_overrides=fields,
        velocity_field_overrides={"d+": np.full((4, 5, 1), 2.0, dtype=np.float64), "d": np.full((4, 5, 1), 3.0, dtype=np.float64)},
    )
    np.testing.assert_allclose(captured["field_overrides"]["NVd+"], expected_fields["NVd+"])


def test_integrated_2d_production_rhs_requests_anomalous_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_input = Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-production/data/BOUT.inp")
    if not production_input.exists():
        pytest.skip("integrated 2D production reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": np.ones((4, 5, 1), dtype=np.float64),
    }
    captured: dict[str, tuple[str, ...]] = {}

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(summary=_FakeSummary(artifacts={"BOUT.dmp.0.nc": "/tmp/fake-dump.nc"})),
    )
    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_snapshot_cache_path",
        lambda case_name: Path("/tmp") / f"{case_name}.missing",
    )

    def fake_load_snapshot(dump_path, *, field_names, optional_field_names=(), scalar_names=(), time_index=0):
        del dump_path, time_index
        captured["field_names"] = tuple(field_names)
        captured["optional_field_names"] = tuple(optional_field_names)
        captured["scalar_names"] = tuple(scalar_names)
        return LocalReferenceSnapshot(
            mesh=mesh,
            metrics=metrics,
            fields=fields,
            optional_fields={
                "Vd+": np.zeros((4, 5, 1), dtype=np.float64),
                "Vd": np.zeros((4, 5, 1), dtype=np.float64),
                "SNd+": np.zeros((4, 5, 1), dtype=np.float64),
                "SNVd+": np.zeros((4, 5, 1), dtype=np.float64),
                "SPd+": np.zeros((4, 5, 1), dtype=np.float64),
                "SNd": np.zeros((4, 5, 1), dtype=np.float64),
                "SNVd": np.zeros((4, 5, 1), dtype=np.float64),
                "SPd": np.zeros((4, 5, 1), dtype=np.float64),
                "SPe": np.zeros((4, 5, 1), dtype=np.float64),
                "Sd_target_recycle": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_target_recycle": np.zeros((4, 5, 1), dtype=np.float64),
                "Sd_wall_recycle": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_wall_recycle": np.zeros((4, 5, 1), dtype=np.float64),
                "Sd_pump": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_pump": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_target_refl": np.zeros((4, 5, 1), dtype=np.float64),
                "Ed_wall_refl": np.zeros((4, 5, 1), dtype=np.float64),
                "is_pump": np.zeros((4, 5, 1), dtype=np.float64),
                "anomalous_D_d+": np.full((4, 5, 1), 1.0e-2, dtype=np.float64),
                "anomalous_Chi_d+": np.full((4, 5, 1), 3.0e-2, dtype=np.float64),
                "anomalous_nu_d+": np.full((4, 5, 1), 2.0e-3, dtype=np.float64),
                "anomalous_D_e": np.full((4, 5, 1), 1.0e-2, dtype=np.float64),
                "anomalous_Chi_e": np.full((4, 5, 1), 3.0e-2, dtype=np.float64),
                "anomalous_nu_e": np.full((4, 5, 1), 2.0e-3, dtype=np.float64),
            },
            scalar_values={"Nnorm": 1.0e17, "Tnorm": 1.0, "Bnorm": 1.0, "Cs0": 1.0, "Omega_ci": 1.0, "rho_s0": 1.0},
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)

    case = ReferenceCase(
        name="integrated_2d_production_rhs",
        stage="stage7",
        reference_path=str(production_input),
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "ddt(Nd+)", "ddt(Pe)"),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    native_runner._run_integrated_2d_recycling_rhs_case(
        case,
        input_path=production_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured["field_names"] == ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe")
    assert captured["scalar_names"] == ("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0")
    assert "anomalous_D_d+" in captured["optional_field_names"]
    assert "anomalous_Chi_d+" in captured["optional_field_names"]
    assert "anomalous_nu_d+" in captured["optional_field_names"]
    assert "anomalous_D_e" in captured["optional_field_names"]
    assert "anomalous_Chi_e" in captured["optional_field_names"]
    assert "anomalous_nu_e" in captured["optional_field_names"]


def test_integrated_2d_recycling_rhs_preserves_dump_sheath_state(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("integrated 2D recycling reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": np.ones((4, 5, 1), dtype=np.float64),
    }

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(summary=_FakeSummary(artifacts={"BOUT.dmp.0.nc": "/tmp/fake-dump.nc"})),
    )
    monkeypatch.setattr(
        native_runner,
        "load_local_reference_snapshot",
        lambda *args, **kwargs: LocalReferenceSnapshot(
            mesh=mesh,
            metrics=metrics,
            fields=fields,
            optional_fields={
                "SNd+": np.ones((4, 5, 1), dtype=np.float64),
                "SNVd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
                "SPd+": np.full((4, 5, 1), 3.0, dtype=np.float64),
                "SNd": np.full((4, 5, 1), 4.0, dtype=np.float64),
                "SNVd": np.full((4, 5, 1), 5.0, dtype=np.float64),
                "SPd": np.full((4, 5, 1), 6.0, dtype=np.float64),
                "SPe": np.full((4, 5, 1), 7.0, dtype=np.float64),
            },
            scalar_values={"Nnorm": 1.0e17},
        ),
    )

    captured: dict[str, bool] = {}
    original = native_runner.compute_recycling_1d_rhs

    def wrapper(*args, **kwargs):
        captured["apply_sheath_boundaries"] = kwargs["apply_sheath_boundaries"]
        captured["preserve_dump_target_state"] = kwargs["preserve_dump_target_state"]
        captured["density_source_overrides"] = kwargs["density_source_overrides"]
        captured["pressure_source_overrides"] = kwargs["pressure_source_overrides"]
        return original(*args, **kwargs)

    monkeypatch.setattr(native_runner, "compute_recycling_1d_rhs", wrapper)

    case = ReferenceCase(
        name="integrated_2d_recycling_rhs",
        stage="stage7",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Nd+", "ddt(Nd+)"),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    native_runner._run_integrated_2d_recycling_rhs_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured["apply_sheath_boundaries"] is True
    assert captured["preserve_dump_target_state"] is True
    assert tuple(captured["density_source_overrides"]) == ("d+", "d")
    assert tuple(captured["pressure_source_overrides"]) == ("d+", "d")


def test_integrated_2d_simple_sheath_preserve_mode_keeps_simple_guard_cells() -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("integrated 2D recycling reference input is unavailable")

    config = load_bout_input(_REFERENCE_INPUT)
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": np.ones((4, 5, 1), dtype=np.float64),
    }
    species = _initialize_species(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=fields,
    )

    prepared_free, _, boundary_free = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        apply_sheath_boundaries=True,
        preserve_dump_target_state=False,
    )
    prepared_preserve, _, boundary_preserve = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        apply_sheath_boundaries=True,
        preserve_dump_target_state=True,
    )

    ghost = mesh.ystart - 1
    np.testing.assert_allclose(boundary_preserve.density[:, ghost, :], boundary_free.density[:, ghost, :])
    np.testing.assert_allclose(boundary_preserve.pressure[:, ghost, :], boundary_free.pressure[:, ghost, :])
    np.testing.assert_allclose(
        prepared_preserve["e"].density[:, mesh.ystart, :],
        prepared_free["e"].density[:, mesh.ystart, :],
    )


def test_integrated_2d_simple_sheath_ion_only_preserve_uses_sheath_electron_state() -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("integrated 2D recycling reference input is unavailable")

    config = load_bout_input(_REFERENCE_INPUT)
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": np.ones((4, 5, 1), dtype=np.float64),
    }
    species = _initialize_species(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=fields,
    )

    prepared_free, ion_free, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        apply_sheath_boundaries=True,
        preserve_dump_target_state=False,
    )
    prepared_preserve, ion_preserve, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        apply_sheath_boundaries=True,
        preserve_dump_target_state=True,
    )
    prepared_ion_only, ion_ion_only, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        apply_sheath_boundaries=True,
        preserve_dump_target_state=True,
        preserve_dump_ion_target_state_only=True,
    )

    np.testing.assert_allclose(
        prepared_ion_only["e"].pressure[:, mesh.ystart, :],
        prepared_free["e"].pressure[:, mesh.ystart, :],
    )
    np.testing.assert_allclose(
        ion_ion_only.pressure["d+"][:, mesh.ystart, :],
        ion_preserve.pressure["d+"][:, mesh.ystart, :],
    )
    ghost = mesh.ystart - 1
    np.testing.assert_allclose(
        ion_ion_only.pressure["d+"][:, ghost, :],
        prepared_ion_only["d+"].pressure[:, ghost, :],
    )
    np.testing.assert_allclose(
        ion_ion_only.velocity["d+"][:, ghost, :],
        prepared_ion_only["d+"].velocity[:, ghost, :],
    )
    np.testing.assert_allclose(
        ion_ion_only.energy_source["d+"][:, mesh.ystart, :],
        0.0,
    )
    assert np.any(np.abs(ion_free.energy_source["d+"][:, mesh.ystart, :]) > 0.0)


def test_integrated_2d_simple_sheath_electron_energy_source_matches_hermes_formula() -> None:
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
        has_upper_y_target=False,
        x=jnp.arange(1, dtype=jnp.float64),
        y=jnp.arange(3, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((1, 3, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    electron_density = 2.0 * np.ones((1, 3, 1), dtype=np.float64)
    electron_pressure = 4.0 * np.ones((1, 3, 1), dtype=np.float64)
    electron_velocity = np.zeros((1, 3, 1), dtype=np.float64)
    ion_density = 2.0 * np.ones((1, 3, 1), dtype=np.float64)
    ion_temperature = np.ones((1, 3, 1), dtype=np.float64)
    ion_velocity = -3.0 * np.ones((1, 3, 1), dtype=np.float64)
    zero = np.zeros((1, 3, 1), dtype=np.float64)

    ion = OpenFieldSpecies(
        name="d+",
        density=ion_density,
        pressure=ion_density * ion_temperature,
        momentum=2.0 * ion_density * ion_velocity,
        charge=1.0,
        atomic_mass=2.0,
        density_floor=1.0e-8,
        has_pressure=True,
        has_momentum=True,
        noflow_lower_y=False,
        noflow_upper_y=False,
        target_recycle=False,
        recycle_as=None,
        target_recycle_multiplier=0.0,
        target_recycle_energy=0.0,
        target_fast_recycle_fraction=0.0,
        target_fast_recycle_energy_factor=1.0,
    )
    prepared_ions = {
        "d+": _PreparedSpeciesState(
            density=ion_density,
            pressure=ion_density * ion_temperature,
            temperature=ion_temperature,
            velocity=ion_velocity,
            momentum=2.0 * ion_density * ion_velocity,
            momentum_error=zero,
        )
    }
    settings = _SimpleSheathSettings(
        gamma_e=4.5,
        gamma_i=3.5,
        secondary_electron_coef=0.0,
        sheath_ion_polytropic=1.0,
        lower_y=True,
        upper_y=False,
        no_flow=False,
        density_boundary_mode=1.0,
        pressure_boundary_mode=1.0,
        temperature_boundary_mode=1.0,
        wall_potential=np.zeros((1, 3, 1), dtype=np.float64),
    )

    result = _apply_electron_simple_sheath_boundary(
        electron_pressure=electron_pressure,
        electron_density=electron_density,
        electron_velocity=electron_velocity,
        electron_mass=1.0,
        electron_density_floor=1.0e-8,
        ion_velocity={"d+": ion_velocity},
        ions=(ion,),
        prepared_ions=prepared_ions,
        mesh=mesh,
        metrics=metrics,
        settings=settings,
    )

    nesheath = 2.0
    tesheath = 2.0
    ion_sum = 6.0
    phi_boundary = tesheath * np.log(
        np.sqrt(tesheath / (1.0 * (2.0 * np.pi)))
        * (1.0 - settings.secondary_electron_coef)
        * nesheath
        / ion_sum
    )
    phisheath = max(phi_boundary, 0.0)
    vesheath = -np.sqrt(tesheath / (2.0 * np.pi * 1.0)) * (1.0 - settings.secondary_electron_coef) * np.exp(
        -(phisheath - 0.0) / tesheath
    )
    expected_q = settings.gamma_e * tesheath * nesheath * vesheath
    expected_q -= (2.5 * tesheath + 0.5 * 1.0 * vesheath * vesheath) * nesheath * vesheath

    assert result.energy_source[0, mesh.ystart, 0] == pytest.approx(expected_q)
    assert result.velocity[0, mesh.ystart - 1, 0] == pytest.approx(2.0 * vesheath)
@pytest.mark.xfail(
    raises=TracerArrayConversionError,
    reason="The staged integrated 2D recycling RHS still materializes NumPy arrays in _initialize_species.",
    strict=False,
)
def test_integrated_2d_recycling_rhs_is_not_yet_differentiable() -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("integrated 2D recycling reference input is unavailable")

    config = load_bout_input(_REFERENCE_INPUT)
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    base_fields = {
        "Nd+": jnp.ones((4, 5, 1), dtype=jnp.float64) * 2.0,
        "Pd+": jnp.ones((4, 5, 1), dtype=jnp.float64) * 3.0,
        "NVd+": jnp.zeros((4, 5, 1), dtype=jnp.float64),
        "Nd": jnp.ones((4, 5, 1), dtype=jnp.float64) * 0.2,
        "Pd": jnp.ones((4, 5, 1), dtype=jnp.float64) * 0.1,
        "NVd": jnp.zeros((4, 5, 1), dtype=jnp.float64),
        "Pe": jnp.ones((4, 5, 1), dtype=jnp.float64) * 3.0,
    }

    def loss(scale: jnp.ndarray) -> jnp.ndarray:
        fields = dict(base_fields)
        fields["Pd+"] = fields["Pd+"] * scale
        result = native_runner.compute_recycling_1d_rhs(
            config,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            field_overrides=fields,
            apply_sheath_boundaries=True,
            preserve_dump_target_state=True,
        )
        return jnp.sum(jnp.asarray(result.variables["ddt(Pd+)"]))

    grad(loss)(jnp.array(1.0, dtype=jnp.float64))


def test_integrated_2d_recycling_one_step_uses_rhs_snapshot_start(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("integrated 2D recycling reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    initial_fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": 3.0 * np.ones((4, 5, 1), dtype=np.float64),
    }
    evolved_history = {name: np.stack([value, value + 1.0], axis=0) for name, value in initial_fields.items()}

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-dump.nc"},
                time_points=(0.0, 1.0e-4),
            )
        ),
    )

    def fake_snapshot_loader(*args, **kwargs):
        time_index = int(kwargs.get("time_index", 0))
        return LocalReferenceSnapshot(
            mesh=mesh,
            metrics=metrics,
            fields=initial_fields,
            optional_fields={
                "SNd+": np.full((4, 5, 1), 1.0, dtype=np.float64),
                "SNVd+": np.full((4, 5, 1), 1.5, dtype=np.float64),
                "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
                "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
                "SNVd": np.full((4, 5, 1), 3.5, dtype=np.float64),
                "SPd": np.full((4, 5, 1), 4.0, dtype=np.float64),
                "Sd_target_recycle": np.full((4, 5, 1), 5.0 + time_index, dtype=np.float64),
                "Ed_target_recycle": np.full((4, 5, 1), 6.0 + time_index, dtype=np.float64),
                "Vd+": np.full((4, 5, 1), 3.0 + time_index, dtype=np.float64),
            },
            scalar_values={"Nnorm": 1.0e17},
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_snapshot_loader)
    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_snapshot_cache_path",
        lambda case_name: Path("/tmp") / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_optional_history_cache_path",
        lambda case_name: Path("/tmp") / f"{case_name}.missing",
    )

    captured: dict[str, object] = {}

    def fake_history(*args, **kwargs):
        captured["initial_fields"] = kwargs["initial_fields"]
        captured["density_source_overrides"] = kwargs["density_source_overrides"]
        captured["pressure_source_overrides"] = kwargs["pressure_source_overrides"]
        captured["momentum_source_overrides"] = kwargs["momentum_source_overrides"]
        captured["preserve_dump_target_state"] = kwargs["preserve_dump_target_state"]
        captured["preserve_dump_ion_target_state_only"] = kwargs["preserve_dump_ion_target_state_only"]
        return SimpleNamespace(variable_history=evolved_history, feedback_integral_history={})

    monkeypatch.setattr(native_runner, "advance_recycling_1d_implicit_history", fake_history)

    def fake_rhs(*args, **kwargs):
        captured["diagnostic_preserve_dump_ion_target_state_only"] = kwargs["preserve_dump_ion_target_state_only"]
        fields = kwargs["field_overrides"]
        density = np.asarray(fields["Nd+"], dtype=np.float64)
        pressure = np.asarray(fields["Pd+"], dtype=np.float64)
        return SimpleNamespace(
            variables={
                "Sd_target_recycle": density[None, ...],
                "Ed_target_recycle": pressure[None, ...],
            }
        )

    monkeypatch.setattr(native_runner, "compute_recycling_1d_rhs", fake_rhs)

    case = ReferenceCase(
        name="integrated_2d_recycling_one_step",
        stage="stage7",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "Sd_target_recycle", "Ed_target_recycle"),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    result = native_runner._run_integrated_2d_recycling_one_step_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert tuple(captured["initial_fields"]) == tuple(initial_fields)
    assert captured["density_source_overrides"] is None
    assert captured["pressure_source_overrides"] is None
    assert captured["momentum_source_overrides"] is None
    assert captured["preserve_dump_target_state"] is False
    assert captured["preserve_dump_ion_target_state_only"] is False
    assert captured["diagnostic_preserve_dump_ion_target_state_only"] is False
    assert result.time_points == (0.0, 0.0001)
    assert result.variables["Nd+"].shape == (2, 2, 3, 1)
    assert result.variables["Sd_target_recycle"].shape == (2, 2, 3, 1)
    np.testing.assert_allclose(result.variables["Sd_target_recycle"][0], 1.0)
    np.testing.assert_allclose(result.variables["Ed_target_recycle"][0], 2.0)


def test_integrated_2d_recycling_short_window_reuses_staged_transient_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("integrated 2D recycling reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    initial_fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": 3.0 * np.ones((4, 5, 1), dtype=np.float64),
    }
    evolved_history = {name: np.stack([value + float(index) for index in range(6)], axis=0) for name, value in initial_fields.items()}

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-dump.nc"},
                time_points=(0.0, 1.0e-4),
            )
        ),
    )

    def fake_snapshot_loader(*args, **kwargs):
        time_index = int(kwargs.get("time_index", 0))
        return LocalReferenceSnapshot(
            mesh=mesh,
            metrics=metrics,
            fields=initial_fields,
            optional_fields={
                "SNd+": np.full((4, 5, 1), 1.0, dtype=np.float64),
                "SNVd+": np.full((4, 5, 1), 1.5, dtype=np.float64),
                "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
                "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
                "SNVd": np.full((4, 5, 1), 3.5, dtype=np.float64),
                "SPd": np.full((4, 5, 1), 4.0, dtype=np.float64),
                "Sd_target_recycle": np.full((4, 5, 1), 5.0 + time_index, dtype=np.float64),
                "Ed_target_recycle": np.full((4, 5, 1), 6.0 + time_index, dtype=np.float64),
                "Vd+": np.full((4, 5, 1), 3.0 + time_index, dtype=np.float64),
            },
            scalar_values={"Nnorm": 1.0e17},
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_snapshot_loader)
    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_snapshot_cache_path",
        lambda case_name: tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_optional_history_cache_path",
        lambda case_name: tmp_path / f"{case_name}.missing",
    )

    captured: dict[str, object] = {}

    def fake_history(*args, **kwargs):
        captured["steps"] = kwargs["steps"]
        captured["preserve_dump_target_state"] = kwargs["preserve_dump_target_state"]
        captured["preserve_dump_ion_target_state_only"] = kwargs["preserve_dump_ion_target_state_only"]
        return SimpleNamespace(variable_history=evolved_history, feedback_integral_history={})

    monkeypatch.setattr(native_runner, "advance_recycling_1d_implicit_history", fake_history)

    def fake_rhs(*args, **kwargs):
        captured["diagnostic_preserve_dump_ion_target_state_only"] = kwargs["preserve_dump_ion_target_state_only"]
        fields = kwargs["field_overrides"]
        density = np.asarray(fields["Nd+"], dtype=np.float64)
        pressure = np.asarray(fields["Pd+"], dtype=np.float64)
        return SimpleNamespace(
            variables={
                "Sd_target_recycle": density[None, ...],
                "Ed_target_recycle": pressure[None, ...],
            }
        )

    monkeypatch.setattr(native_runner, "compute_recycling_1d_rhs", fake_rhs)

    case = ReferenceCase(
        name="integrated_2d_recycling_short_window",
        stage="stage7",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="short_window",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "Sd_target_recycle", "Ed_target_recycle"),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    result = native_runner._run_integrated_2d_recycling_short_window_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured["steps"] == 5
    assert captured["preserve_dump_target_state"] is False
    assert captured["preserve_dump_ion_target_state_only"] is False
    assert captured["diagnostic_preserve_dump_ion_target_state_only"] is False
    assert result.time_points == (0.0, 0.0001, 0.0002, 0.00030000000000000003, 0.0004, 0.0005)
    assert result.variables["Nd+"].shape == (6, 2, 3, 1)
    assert result.variables["Sd_target_recycle"].shape == (6, 2, 3, 1)
    np.testing.assert_allclose(result.variables["Sd_target_recycle"][0], 1.0)
    np.testing.assert_allclose(result.variables["Ed_target_recycle"][0], 2.0)


def test_integrated_2d_production_one_step_preserves_only_ion_target_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    production_input = Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-production/data/BOUT.inp")
    if not production_input.exists():
        pytest.skip("integrated 2D production reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    initial_fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": 3.0 * np.ones((4, 5, 1), dtype=np.float64),
    }
    evolved_history = {name: np.stack([value, value + 1.0], axis=0) for name, value in initial_fields.items()}

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-dump.nc"},
                time_points=(0.0, 1.0e-4),
            )
        ),
    )

    def fake_snapshot_loader(*args, **kwargs):
        time_index = int(kwargs.get("time_index", 0))
        return LocalReferenceSnapshot(
            mesh=mesh,
            metrics=metrics,
            fields=initial_fields,
            optional_fields={
                "SNd+": np.full((4, 5, 1), 1.0, dtype=np.float64),
                "SNVd+": np.full((4, 5, 1), 1.5, dtype=np.float64),
                "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
                "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
                "SNVd": np.full((4, 5, 1), 3.5, dtype=np.float64),
                "SPd": np.full((4, 5, 1), 4.0, dtype=np.float64),
                "Sd_target_recycle": np.full((4, 5, 1), 5.0 + time_index, dtype=np.float64),
                "Ed_target_recycle": np.full((4, 5, 1), 6.0 + time_index, dtype=np.float64),
                "Vd+": np.full((4, 5, 1), 3.0 + time_index, dtype=np.float64),
            },
            scalar_values={"Nnorm": 1.0e17},
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_snapshot_loader)
    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_snapshot_cache_path",
        lambda case_name: tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_optional_history_cache_path",
        lambda case_name: tmp_path / f"{case_name}.missing",
    )

    captured: dict[str, object] = {}

    def fake_history(*args, **kwargs):
        captured["preserve_dump_target_state"] = kwargs["preserve_dump_target_state"]
        captured["preserve_dump_ion_target_state_only"] = kwargs["preserve_dump_ion_target_state_only"]
        captured["pressure_source_overrides"] = kwargs["pressure_source_overrides"]
        captured["initial_fields"] = kwargs["initial_fields"]
        return SimpleNamespace(variable_history=evolved_history, feedback_integral_history={})

    monkeypatch.setattr(native_runner, "advance_recycling_1d_implicit_history", fake_history)
    def fake_rhs(*args, **kwargs):
        captured["diagnostic_preserve_dump_ion_target_state_only"] = kwargs["preserve_dump_ion_target_state_only"]
        captured["diagnostic_nvdp"] = np.asarray(kwargs["field_overrides"]["NVd+"], dtype=np.float64)
        return SimpleNamespace(
            variables={
                "Sd_target_recycle": np.ones((1, 4, 5, 1), dtype=np.float64),
                "Ed_target_recycle": np.ones((1, 4, 5, 1), dtype=np.float64),
            }
        )

    monkeypatch.setattr(native_runner, "compute_recycling_1d_rhs", fake_rhs)

    case = ReferenceCase(
        name="integrated_2d_production_one_step",
        stage="stage7",
        reference_path=str(production_input),
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "Sd_target_recycle", "Ed_target_recycle"),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    native_runner._run_integrated_2d_recycling_one_step_case(
        case,
        input_path=production_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured["preserve_dump_target_state"] is True
    assert captured["preserve_dump_ion_target_state_only"] is True
    assert tuple(captured["pressure_source_overrides"]) == ("d+", "d")
    expected_initial_fields = native_runner._apply_species_velocity_overrides(
        load_bout_input(production_input),
        field_overrides=initial_fields,
        velocity_field_overrides={"d+": np.full((4, 5, 1), 3.0, dtype=np.float64)},
    )
    np.testing.assert_allclose(captured["initial_fields"]["NVd+"], expected_initial_fields["NVd+"])
    assert captured["diagnostic_preserve_dump_ion_target_state_only"] is True
    expected_fields = native_runner._apply_species_velocity_overrides(
        load_bout_input(production_input),
        field_overrides={name: value[1] for name, value in evolved_history.items()},
        velocity_field_overrides={"d+": np.full((4, 5, 1), 4.0, dtype=np.float64)},
    )
    np.testing.assert_allclose(captured["diagnostic_nvdp"], expected_fields["NVd+"])


def test_integrated_2d_production_one_step_uses_committed_snapshot_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    production_input = Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-production/data/BOUT.inp")
    if not production_input.exists():
        pytest.skip("integrated 2D production reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    initial_fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": 3.0 * np.ones((4, 5, 1), dtype=np.float64),
    }
    snapshot = LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields=initial_fields,
        optional_fields={
            "SNd+": np.full((4, 5, 1), 1.0, dtype=np.float64),
            "SNVd+": np.full((4, 5, 1), 1.5, dtype=np.float64),
            "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
            "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
            "SNVd": np.full((4, 5, 1), 3.5, dtype=np.float64),
            "SPd": np.full((4, 5, 1), 4.0, dtype=np.float64),
            "Sd_target_recycle": np.full((4, 5, 1), 5.0, dtype=np.float64),
            "Ed_target_recycle": np.full((4, 5, 1), 6.0, dtype=np.float64),
        },
        scalar_values={"Nnorm": 1.0e17, "Tnorm": 1.0, "Bnorm": 1.0, "Cs0": 1.0, "Omega_ci": 1.0, "rho_s0": 1.0},
    )
    snapshot_cache = tmp_path / "integrated_2d_production_rhs_snapshot.npz"
    save_local_reference_snapshot_cache(snapshot, snapshot_cache)
    history_cache = tmp_path / "integrated_2d_production_one_step_optional_history.npz"
    save_optional_field_history_cache(
        {
            "Vd+": np.stack(
                [
                    np.full((4, 5, 1), 3.0, dtype=np.float64),
                    np.full((4, 5, 1), 4.0, dtype=np.float64),
                ],
                axis=0,
            ),
            "Sd_target_recycle": np.stack(
                [
                    np.full((4, 5, 1), 5.0, dtype=np.float64),
                    np.full((4, 5, 1), 6.0, dtype=np.float64),
                ],
                axis=0,
            ),
            "Ed_target_recycle": np.stack(
                [
                    np.full((4, 5, 1), 7.0, dtype=np.float64),
                    np.full((4, 5, 1), 8.0, dtype=np.float64),
                ],
                axis=0,
            ),
        },
        history_cache,
    )

    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_snapshot_cache_path",
        lambda case_name: snapshot_cache if case_name == "integrated_2d_production_rhs" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_optional_history_cache_path",
        lambda case_name: history_cache if case_name == "integrated_2d_production_one_step" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference run should not be used when caches are present")),
    )

    evolved_history = {name: np.stack([value, value + 1.0], axis=0) for name, value in initial_fields.items()}
    captured: dict[str, object] = {}

    def fake_history(*args, **kwargs):
        captured["initial_fields"] = kwargs["initial_fields"]
        captured["solver_mode"] = kwargs["solver_mode"]
        captured["preserve_dump_ion_target_state_only"] = kwargs["preserve_dump_ion_target_state_only"]
        return SimpleNamespace(variable_history=evolved_history, feedback_integral_history={})

    monkeypatch.setattr(native_runner, "advance_recycling_1d_implicit_history", fake_history)

    def fake_rhs(*args, **kwargs):
        captured["diagnostic_nvdp"] = np.asarray(kwargs["field_overrides"]["NVd+"], dtype=np.float64)
        return SimpleNamespace(
            variables={
                "Sd_target_recycle": np.ones((1, 4, 5, 1), dtype=np.float64),
                "Ed_target_recycle": np.ones((1, 4, 5, 1), dtype=np.float64),
            }
        )

    monkeypatch.setattr(native_runner, "compute_recycling_1d_rhs", fake_rhs)

    case = ReferenceCase(
        name="integrated_2d_production_one_step",
        stage="stage7",
        reference_path=str(production_input),
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "Sd_target_recycle", "Ed_target_recycle"),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    result = native_runner._run_integrated_2d_recycling_one_step_case(
        case,
        input_path=production_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    expected_initial_fields = native_runner._apply_species_velocity_overrides(
        load_bout_input(production_input),
        field_overrides=initial_fields,
        velocity_field_overrides={"d+": np.full((4, 5, 1), 3.0, dtype=np.float64)},
    )
    np.testing.assert_allclose(captured["initial_fields"]["NVd+"], expected_initial_fields["NVd+"])
    expected_fields = native_runner._apply_species_velocity_overrides(
        load_bout_input(production_input),
        field_overrides={name: value[1] for name, value in evolved_history.items()},
        velocity_field_overrides={"d+": np.full((4, 5, 1), 4.0, dtype=np.float64)},
    )
    np.testing.assert_allclose(captured["diagnostic_nvdp"], expected_fields["NVd+"])
    np.testing.assert_allclose(result.variables["Sd_target_recycle"][0], 5.0)
    np.testing.assert_allclose(result.variables["Ed_target_recycle"][0], 7.0)


def test_tokamak_recycling_rhs_uses_committed_snapshot_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokamak_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/recycling/BOUT.inp")
    if not tokamak_input.exists():
        pytest.skip("tokamak recycling reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    snapshot = LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields={
            "Nd+": np.ones((4, 5, 1), dtype=np.float64),
            "Pd+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
            "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
            "Nd": np.ones((4, 5, 1), dtype=np.float64),
            "Pd": np.ones((4, 5, 1), dtype=np.float64),
            "NVd": np.zeros((4, 5, 1), dtype=np.float64),
            "Pe": 3.0 * np.ones((4, 5, 1), dtype=np.float64),
        },
        optional_fields={
            "SNd+": np.full((4, 5, 1), 1.0, dtype=np.float64),
            "SNVd+": np.full((4, 5, 1), 1.5, dtype=np.float64),
            "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
            "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
            "SNVd": np.full((4, 5, 1), 3.5, dtype=np.float64),
            "SPd": np.full((4, 5, 1), 4.0, dtype=np.float64),
            "Sd_target_recycle": np.full((4, 5, 1), 5.0, dtype=np.float64),
            "Ed_target_recycle": np.full((4, 5, 1), 6.0, dtype=np.float64),
        },
        scalar_values={"Nnorm": 1.0e17, "Tnorm": 1.0, "Bnorm": 1.0, "Cs0": 1.0, "Omega_ci": 1.0, "rho_s0": 1.0},
    )
    snapshot_cache = tmp_path / "tokamak_recycling_rhs_snapshot.npz"
    save_local_reference_snapshot_cache(snapshot, snapshot_cache)

    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_snapshot_cache_path",
        lambda case_name: snapshot_cache if case_name == "tokamak_recycling_rhs" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference run should not be used when tokamak cache is present")),
    )

    result = native_runner.run_curated_case("tokamak_recycling_rhs", reference_root=_REFERENCE_ROOT)

    assert result.time_points == (0.0,)
    assert np.asarray(result.variables["ddt(Nd+)"]).shape == (1, 2, 3, 1)


def test_tokamak_recycling_dthe_rhs_uses_committed_snapshot_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokamak_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/recycling-dthe/BOUT.inp")
    if not tokamak_input.exists():
        pytest.skip("tokamak recycling dthe reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=ones,
        Bxy=ones,
    )
    snapshot = LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields={
            "Nd+": np.ones((4, 5, 1), dtype=np.float64),
            "Pd+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
            "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
            "Nd": np.ones((4, 5, 1), dtype=np.float64),
            "Pd": np.ones((4, 5, 1), dtype=np.float64),
            "NVd": np.zeros((4, 5, 1), dtype=np.float64),
            "Nt+": np.ones((4, 5, 1), dtype=np.float64),
            "Pt+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
            "NVt+": np.zeros((4, 5, 1), dtype=np.float64),
            "Nt": np.ones((4, 5, 1), dtype=np.float64),
            "Pt": np.ones((4, 5, 1), dtype=np.float64),
            "NVt": np.zeros((4, 5, 1), dtype=np.float64),
            "Nhe+": 0.01 * np.ones((4, 5, 1), dtype=np.float64),
            "Phe+": 0.02 * np.ones((4, 5, 1), dtype=np.float64),
            "NVhe+": np.zeros((4, 5, 1), dtype=np.float64),
            "Nhe": np.ones((4, 5, 1), dtype=np.float64),
            "Phe": np.ones((4, 5, 1), dtype=np.float64),
            "NVhe": np.zeros((4, 5, 1), dtype=np.float64),
            "Pe": 3.0 * np.ones((4, 5, 1), dtype=np.float64),
        },
        optional_fields={
            "SNd+": np.full((4, 5, 1), 1.0, dtype=np.float64),
            "SNVd+": np.full((4, 5, 1), 1.5, dtype=np.float64),
            "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
            "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
            "SNVd": np.full((4, 5, 1), 3.5, dtype=np.float64),
            "SPd": np.full((4, 5, 1), 4.0, dtype=np.float64),
            "SNt+": np.full((4, 5, 1), 1.1, dtype=np.float64),
            "SNVt+": np.full((4, 5, 1), 1.6, dtype=np.float64),
            "SPt+": np.full((4, 5, 1), 2.1, dtype=np.float64),
            "SNt": np.full((4, 5, 1), 3.1, dtype=np.float64),
            "SNVt": np.full((4, 5, 1), 3.6, dtype=np.float64),
            "SPt": np.full((4, 5, 1), 4.1, dtype=np.float64),
            "SNhe+": np.full((4, 5, 1), 1.2, dtype=np.float64),
            "SNVhe+": np.full((4, 5, 1), 1.7, dtype=np.float64),
            "SPhe+": np.full((4, 5, 1), 2.2, dtype=np.float64),
            "SNhe": np.full((4, 5, 1), 3.2, dtype=np.float64),
            "SNVhe": np.full((4, 5, 1), 3.7, dtype=np.float64),
            "SPhe": np.full((4, 5, 1), 4.2, dtype=np.float64),
            "Sd_target_recycle": np.full((4, 5, 1), 5.0, dtype=np.float64),
            "Ed_target_recycle": np.full((4, 5, 1), 6.0, dtype=np.float64),
            "anomalous_D_d+": np.zeros((4, 5, 1), dtype=np.float64),
            "anomalous_Chi_d+": np.zeros((4, 5, 1), dtype=np.float64),
            "anomalous_nu_d+": np.zeros((4, 5, 1), dtype=np.float64),
            "anomalous_D_e": np.zeros((4, 5, 1), dtype=np.float64),
            "anomalous_Chi_e": np.zeros((4, 5, 1), dtype=np.float64),
            "anomalous_nu_e": np.zeros((4, 5, 1), dtype=np.float64),
        },
        scalar_values={"Nnorm": 1.0e17, "Tnorm": 1.0, "Bnorm": 1.0, "Cs0": 1.0, "Omega_ci": 1.0, "rho_s0": 1.0},
    )
    snapshot_cache = tmp_path / "tokamak_recycling_dthe_rhs_snapshot.npz"
    save_local_reference_snapshot_cache(snapshot, snapshot_cache)

    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_snapshot_cache_path",
        lambda case_name: snapshot_cache if case_name == "tokamak_recycling_dthe_rhs" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference run should not be used when tokamak dthe cache is present")),
    )

    result = native_runner.run_curated_case("tokamak_recycling_dthe_rhs", reference_root=_REFERENCE_ROOT)

    assert result.time_points == (0.0,)
    assert np.asarray(result.variables["Nd+"]).shape == (1, 2, 3, 1)
    assert np.asarray(result.variables["Pe"]).shape == (1, 2, 3, 1)


def test_tokamak_recycling_dthene_rhs_uses_committed_snapshot_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokamak_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/recycling-dthene/BOUT.inp")
    if not tokamak_input.exists():
        pytest.skip("tokamak recycling dthene reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=ones,
        Bxy=ones,
    )
    snapshot = LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields={
            "Nd+": np.ones((4, 5, 1), dtype=np.float64),
            "Pd+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
            "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
            "Nd": np.ones((4, 5, 1), dtype=np.float64),
            "Pd": np.ones((4, 5, 1), dtype=np.float64),
            "NVd": np.zeros((4, 5, 1), dtype=np.float64),
            "Nt+": np.ones((4, 5, 1), dtype=np.float64),
            "Pt+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
            "NVt+": np.zeros((4, 5, 1), dtype=np.float64),
            "Nt": np.ones((4, 5, 1), dtype=np.float64),
            "Pt": np.ones((4, 5, 1), dtype=np.float64),
            "NVt": np.zeros((4, 5, 1), dtype=np.float64),
            "Nhe+": 0.01 * np.ones((4, 5, 1), dtype=np.float64),
            "Phe+": 0.02 * np.ones((4, 5, 1), dtype=np.float64),
            "NVhe+": np.zeros((4, 5, 1), dtype=np.float64),
            "Nhe": np.ones((4, 5, 1), dtype=np.float64),
            "Phe": np.ones((4, 5, 1), dtype=np.float64),
            "NVhe": np.zeros((4, 5, 1), dtype=np.float64),
            "Nne+": 0.005 * np.ones((4, 5, 1), dtype=np.float64),
            "Pne+": 0.01 * np.ones((4, 5, 1), dtype=np.float64),
            "NVne+": np.zeros((4, 5, 1), dtype=np.float64),
            "Nne": np.ones((4, 5, 1), dtype=np.float64),
            "Pne": np.ones((4, 5, 1), dtype=np.float64),
            "NVne": np.zeros((4, 5, 1), dtype=np.float64),
            "Pe": 3.0 * np.ones((4, 5, 1), dtype=np.float64),
        },
        optional_fields={
            "SNd+": np.full((4, 5, 1), 1.0, dtype=np.float64),
            "SNVd+": np.full((4, 5, 1), 1.5, dtype=np.float64),
            "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
            "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
            "SNVd": np.full((4, 5, 1), 3.5, dtype=np.float64),
            "SPd": np.full((4, 5, 1), 4.0, dtype=np.float64),
            "SNt+": np.full((4, 5, 1), 1.1, dtype=np.float64),
            "SNVt+": np.full((4, 5, 1), 1.6, dtype=np.float64),
            "SPt+": np.full((4, 5, 1), 2.1, dtype=np.float64),
            "SNt": np.full((4, 5, 1), 3.1, dtype=np.float64),
            "SNVt": np.full((4, 5, 1), 3.6, dtype=np.float64),
            "SPt": np.full((4, 5, 1), 4.1, dtype=np.float64),
            "SNhe+": np.full((4, 5, 1), 1.2, dtype=np.float64),
            "SNVhe+": np.full((4, 5, 1), 1.7, dtype=np.float64),
            "SPhe+": np.full((4, 5, 1), 2.2, dtype=np.float64),
            "SNhe": np.full((4, 5, 1), 3.2, dtype=np.float64),
            "SNVhe": np.full((4, 5, 1), 3.7, dtype=np.float64),
            "SPhe": np.full((4, 5, 1), 4.2, dtype=np.float64),
            "SNne+": np.full((4, 5, 1), 1.25, dtype=np.float64),
            "SNVne+": np.full((4, 5, 1), 1.75, dtype=np.float64),
            "SPne+": np.full((4, 5, 1), 2.25, dtype=np.float64),
            "SNne": np.full((4, 5, 1), 3.25, dtype=np.float64),
            "SNVne": np.full((4, 5, 1), 3.75, dtype=np.float64),
            "SPne": np.full((4, 5, 1), 4.25, dtype=np.float64),
            "Sd_target_recycle": np.full((4, 5, 1), 5.0, dtype=np.float64),
            "Ed_target_recycle": np.full((4, 5, 1), 6.0, dtype=np.float64),
            "anomalous_D_d+": np.zeros((4, 5, 1), dtype=np.float64),
            "anomalous_Chi_d+": np.zeros((4, 5, 1), dtype=np.float64),
            "anomalous_nu_d+": np.zeros((4, 5, 1), dtype=np.float64),
            "anomalous_D_e": np.zeros((4, 5, 1), dtype=np.float64),
            "anomalous_Chi_e": np.zeros((4, 5, 1), dtype=np.float64),
            "anomalous_nu_e": np.zeros((4, 5, 1), dtype=np.float64),
        },
        scalar_values={"Nnorm": 1.0e17, "Tnorm": 1.0, "Bnorm": 1.0, "Cs0": 1.0, "Omega_ci": 1.0, "rho_s0": 1.0},
    )
    snapshot_cache = tmp_path / "tokamak_recycling_dthene_rhs_snapshot.npz"
    save_local_reference_snapshot_cache(snapshot, snapshot_cache)

    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_snapshot_cache_path",
        lambda case_name: snapshot_cache if case_name == "tokamak_recycling_dthene_rhs" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference run should not be used when tokamak dthene cache is present")),
    )

    result = native_runner.run_curated_case("tokamak_recycling_dthene_rhs", reference_root=_REFERENCE_ROOT)

    assert result.time_points == (0.0,)
    assert np.asarray(result.variables["Nd+"]).shape == (1, 2, 3, 1)
    assert np.asarray(result.variables["Nne+"]).shape == (1, 2, 3, 1)
    assert np.asarray(result.variables["Pe"]).shape == (1, 2, 3, 1)


def test_tokamak_recycling_dthene_one_step_uses_committed_optional_history_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokamak_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/recycling-dthene/BOUT.inp")
    if not tokamak_input.exists():
        pytest.skip("tokamak recycling dthene reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    initial_fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.ones((4, 5, 1), dtype=np.float64),
        "Pd": np.ones((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Nt+": np.ones((4, 5, 1), dtype=np.float64),
        "Pt+": 2.5 * np.ones((4, 5, 1), dtype=np.float64),
        "NVt+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nt": np.ones((4, 5, 1), dtype=np.float64),
        "Pt": np.ones((4, 5, 1), dtype=np.float64),
        "NVt": np.zeros((4, 5, 1), dtype=np.float64),
        "Nhe+": 0.1 * np.ones((4, 5, 1), dtype=np.float64),
        "Phe+": 0.2 * np.ones((4, 5, 1), dtype=np.float64),
        "NVhe+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nhe": 0.1 * np.ones((4, 5, 1), dtype=np.float64),
        "Phe": 0.1 * np.ones((4, 5, 1), dtype=np.float64),
        "NVhe": np.zeros((4, 5, 1), dtype=np.float64),
        "Nne+": 0.01 * np.ones((4, 5, 1), dtype=np.float64),
        "Pne+": 0.02 * np.ones((4, 5, 1), dtype=np.float64),
        "NVne+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nne": 0.01 * np.ones((4, 5, 1), dtype=np.float64),
        "Pne": 0.01 * np.ones((4, 5, 1), dtype=np.float64),
        "NVne": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": 3.0 * np.ones((4, 5, 1), dtype=np.float64),
    }
    snapshot = LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields=initial_fields,
        optional_fields={
            "Vd+": np.full((4, 5, 1), 3.0, dtype=np.float64),
            "Vd": np.full((4, 5, 1), 1.5, dtype=np.float64),
            "Vt+": np.full((4, 5, 1), 2.0, dtype=np.float64),
            "Vt": np.full((4, 5, 1), 1.0, dtype=np.float64),
            "Vhe+": np.full((4, 5, 1), 0.3, dtype=np.float64),
            "Vhe": np.full((4, 5, 1), 0.2, dtype=np.float64),
            "Vne+": np.full((4, 5, 1), 0.03, dtype=np.float64),
            "Vne": np.full((4, 5, 1), 0.02, dtype=np.float64),
            "SNd+": np.full((4, 5, 1), 1.0, dtype=np.float64),
            "SNVd+": np.full((4, 5, 1), 1.5, dtype=np.float64),
            "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
            "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
            "SNVd": np.full((4, 5, 1), 3.5, dtype=np.float64),
            "SPd": np.full((4, 5, 1), 4.0, dtype=np.float64),
            "SNt+": np.full((4, 5, 1), 1.1, dtype=np.float64),
            "SNVt+": np.full((4, 5, 1), 1.6, dtype=np.float64),
            "SPt+": np.full((4, 5, 1), 2.1, dtype=np.float64),
            "SNt": np.full((4, 5, 1), 3.1, dtype=np.float64),
            "SNVt": np.full((4, 5, 1), 3.6, dtype=np.float64),
            "SPt": np.full((4, 5, 1), 4.1, dtype=np.float64),
            "SNhe+": np.full((4, 5, 1), 1.2, dtype=np.float64),
            "SNVhe+": np.full((4, 5, 1), 1.7, dtype=np.float64),
            "SPhe+": np.full((4, 5, 1), 2.2, dtype=np.float64),
            "SNhe": np.full((4, 5, 1), 3.2, dtype=np.float64),
            "SNVhe": np.full((4, 5, 1), 3.7, dtype=np.float64),
            "SPhe": np.full((4, 5, 1), 4.2, dtype=np.float64),
            "SNne+": np.full((4, 5, 1), 1.3, dtype=np.float64),
            "SNVne+": np.full((4, 5, 1), 1.8, dtype=np.float64),
            "SPne+": np.full((4, 5, 1), 2.3, dtype=np.float64),
            "SNne": np.full((4, 5, 1), 3.3, dtype=np.float64),
            "SNVne": np.full((4, 5, 1), 3.8, dtype=np.float64),
            "SPne": np.full((4, 5, 1), 4.3, dtype=np.float64),
            "Sd_target_recycle": np.full((4, 5, 1), 5.0, dtype=np.float64),
            "Ed_target_recycle": np.full((4, 5, 1), 6.0, dtype=np.float64),
        },
        scalar_values={"Nnorm": 1.0e17, "Tnorm": 1.0, "Bnorm": 1.0, "Cs0": 1.0, "Omega_ci": 1.0, "rho_s0": 1.0},
    )
    snapshot_cache = tmp_path / "tokamak_recycling_dthene_rhs_snapshot.npz"
    save_local_reference_snapshot_cache(snapshot, snapshot_cache)
    history_cache = tmp_path / "tokamak_recycling_dthene_one_step_optional_history.npz"
    save_optional_field_history_cache(
        {
            **{name: np.stack([value, value + 1.0], axis=0) for name, value in initial_fields.items()},
            "Vd+": np.stack([np.full((4, 5, 1), 3.0), np.full((4, 5, 1), 4.0)], axis=0),
            "Vd": np.stack([np.full((4, 5, 1), 1.5), np.full((4, 5, 1), 2.5)], axis=0),
            "Vt+": np.stack([np.full((4, 5, 1), 2.0), np.full((4, 5, 1), 3.0)], axis=0),
            "Vt": np.stack([np.full((4, 5, 1), 1.0), np.full((4, 5, 1), 2.0)], axis=0),
            "Vhe+": np.stack([np.full((4, 5, 1), 0.3), np.full((4, 5, 1), 0.4)], axis=0),
            "Vhe": np.stack([np.full((4, 5, 1), 0.2), np.full((4, 5, 1), 0.3)], axis=0),
            "Vne+": np.stack([np.full((4, 5, 1), 0.03), np.full((4, 5, 1), 0.04)], axis=0),
            "Vne": np.stack([np.full((4, 5, 1), 0.02), np.full((4, 5, 1), 0.03)], axis=0),
            "Sd_target_recycle": np.stack([np.full((4, 5, 1), 5.0), np.full((4, 5, 1), 6.0)], axis=0),
            "Ed_target_recycle": np.stack([np.full((4, 5, 1), 7.0), np.full((4, 5, 1), 8.0)], axis=0),
        },
        history_cache,
    )

    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_snapshot_cache_path",
        lambda case_name: snapshot_cache if case_name == "tokamak_recycling_dthene_rhs" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_optional_history_cache_path",
        lambda case_name: history_cache if case_name == "tokamak_recycling_dthene_one_step" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference run should not be used when tokamak caches are present")),
    )

    evolved_history = {name: np.stack([value, value + 1.0], axis=0) for name, value in initial_fields.items()}
    captured: dict[str, object] = {}

    def fake_history(*args, **kwargs):
        captured["initial_fields"] = kwargs["initial_fields"]
        captured["field_template_overrides"] = kwargs["field_template_overrides"]
        captured["solver_mode"] = kwargs["solver_mode"]
        captured["preserve_dump_ion_target_state_only"] = kwargs["preserve_dump_ion_target_state_only"]
        return SimpleNamespace(variable_history=evolved_history, feedback_integral_history={})

    monkeypatch.setattr(native_runner, "advance_recycling_1d_implicit_history", fake_history)

    def fake_rhs(*args, **kwargs):
        captured["diagnostic_nvene"] = np.asarray(kwargs["field_overrides"]["NVne+"], dtype=np.float64)
        return SimpleNamespace(
            variables={
                "Sd_target_recycle": np.ones((1, 4, 5, 1), dtype=np.float64),
                "Ed_target_recycle": np.ones((1, 4, 5, 1), dtype=np.float64),
            }
        )

    monkeypatch.setattr(native_runner, "compute_recycling_1d_rhs", fake_rhs)

    case = ReferenceCase(
        name="tokamak_recycling_dthene_one_step",
        stage="stage7",
        reference_path="examples/tokamak-2D/recycling-dthene/BOUT.inp",
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Nt+", "Pt+", "NVt+", "Nt", "Pt", "NVt", "Nhe+", "Phe+", "NVhe+", "Nhe", "Phe", "NVhe", "Nne+", "Pne+", "NVne+", "Nne", "Pne", "NVne", "Pe"),
        extra_overrides=(
            "timestep=0.1",
            "mesh:file={reference_root}/examples/tokamak-2D/tokamak.nc",
            "json_database_dir={reference_root}/json_database",
            "he+:diagnose=false",
            "ne+:diagnose=false",
            "input:error_on_unused_options=false",
        ),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_integrated_2d_recycling_one_step_case(
        case,
        input_path=tokamak_input,
        reference_root=_REFERENCE_ROOT,
    )

    expected_initial_fields = native_runner._apply_species_velocity_overrides(
        native_runner._load_curated_case_config(case, tokamak_input),
        field_overrides=initial_fields,
        velocity_field_overrides={
            "d+": np.full((4, 5, 1), 3.0, dtype=np.float64),
            "d": np.full((4, 5, 1), 1.5, dtype=np.float64),
            "t+": np.full((4, 5, 1), 2.0, dtype=np.float64),
            "t": np.full((4, 5, 1), 1.0, dtype=np.float64),
            "he+": np.full((4, 5, 1), 0.3, dtype=np.float64),
            "he": np.full((4, 5, 1), 0.2, dtype=np.float64),
            "ne+": np.full((4, 5, 1), 0.03, dtype=np.float64),
            "ne": np.full((4, 5, 1), 0.02, dtype=np.float64),
        },
    )
    np.testing.assert_allclose(captured["initial_fields"]["NVd+"], expected_initial_fields["NVd+"])
    np.testing.assert_allclose(captured["initial_fields"]["NVt+"], expected_initial_fields["NVt+"])
    np.testing.assert_allclose(captured["initial_fields"]["NVhe+"], expected_initial_fields["NVhe+"])
    np.testing.assert_allclose(captured["initial_fields"]["NVne+"], expected_initial_fields["NVne+"])
    assert captured["field_template_overrides"] is None
    assert captured["solver_mode"] == "bdf"
    assert captured["preserve_dump_ion_target_state_only"] is False

    expected_fields = native_runner._apply_species_velocity_overrides(
        native_runner._load_curated_case_config(case, tokamak_input),
        field_overrides={name: value[1] for name, value in evolved_history.items()},
        velocity_field_overrides={
            "d+": np.full((4, 5, 1), 4.0, dtype=np.float64),
            "d": np.full((4, 5, 1), 2.5, dtype=np.float64),
            "t+": np.full((4, 5, 1), 3.0, dtype=np.float64),
            "t": np.full((4, 5, 1), 2.0, dtype=np.float64),
            "he+": np.full((4, 5, 1), 0.4, dtype=np.float64),
            "he": np.full((4, 5, 1), 0.3, dtype=np.float64),
            "ne+": np.full((4, 5, 1), 0.04, dtype=np.float64),
            "ne": np.full((4, 5, 1), 0.03, dtype=np.float64),
        },
    )
    np.testing.assert_allclose(captured["diagnostic_nvene"], expected_fields["NVne+"])
    assert result.time_points == (0.0, 0.1)
    assert np.asarray(result.variables["Nne+"]).shape == (2, 2, 3, 1)
    assert np.asarray(result.variables["Pe"]).shape == (2, 2, 3, 1)


def test_tokamak_recycling_one_step_uses_committed_optional_history_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokamak_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/recycling/BOUT.inp")
    if not tokamak_input.exists():
        pytest.skip("tokamak recycling reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    initial_fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.ones((4, 5, 1), dtype=np.float64),
        "Pd": np.ones((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": 3.0 * np.ones((4, 5, 1), dtype=np.float64),
    }
    snapshot = LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields=initial_fields,
        optional_fields={
            "SNd+": np.full((4, 5, 1), 1.0, dtype=np.float64),
            "SNVd+": np.full((4, 5, 1), 1.5, dtype=np.float64),
            "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
            "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
            "SNVd": np.full((4, 5, 1), 3.5, dtype=np.float64),
            "SPd": np.full((4, 5, 1), 4.0, dtype=np.float64),
            "Sd_target_recycle": np.full((4, 5, 1), 5.0, dtype=np.float64),
            "Ed_target_recycle": np.full((4, 5, 1), 6.0, dtype=np.float64),
        },
        scalar_values={"Nnorm": 1.0e17, "Tnorm": 1.0, "Bnorm": 1.0, "Cs0": 1.0, "Omega_ci": 1.0, "rho_s0": 1.0},
    )
    snapshot_cache = tmp_path / "tokamak_recycling_rhs_snapshot.npz"
    save_local_reference_snapshot_cache(snapshot, snapshot_cache)
    history_cache = tmp_path / "tokamak_recycling_one_step_optional_history.npz"
    save_optional_field_history_cache(
        {
            **{name: np.stack([value, value + 1.0], axis=0) for name, value in initial_fields.items()},
            "Vd+": np.stack(
                [
                    np.full((4, 5, 1), 3.0, dtype=np.float64),
                    np.full((4, 5, 1), 4.0, dtype=np.float64),
                ],
                axis=0,
            ),
            "Vd": np.stack(
                [
                    np.full((4, 5, 1), 1.5, dtype=np.float64),
                    np.full((4, 5, 1), 2.5, dtype=np.float64),
                ],
                axis=0,
            ),
            "Sd_target_recycle": np.stack(
                [
                    np.full((4, 5, 1), 5.0, dtype=np.float64),
                    np.full((4, 5, 1), 6.0, dtype=np.float64),
                ],
                axis=0,
            ),
            "Ed_target_recycle": np.stack(
                [
                    np.full((4, 5, 1), 7.0, dtype=np.float64),
                    np.full((4, 5, 1), 8.0, dtype=np.float64),
                ],
                axis=0,
            ),
        },
        history_cache,
    )

    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_snapshot_cache_path",
        lambda case_name: snapshot_cache if case_name == "tokamak_recycling_rhs" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_optional_history_cache_path",
        lambda case_name: history_cache if case_name == "tokamak_recycling_one_step" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference run should not be used when tokamak caches are present")),
    )

    evolved_history = {name: np.stack([value, value + 1.0], axis=0) for name, value in initial_fields.items()}
    captured: dict[str, object] = {}

    def fake_history(*args, **kwargs):
        captured["initial_fields"] = kwargs["initial_fields"]
        captured["field_template_overrides"] = kwargs["field_template_overrides"]
        captured["solver_mode"] = kwargs["solver_mode"]
        captured["preserve_dump_ion_target_state_only"] = kwargs["preserve_dump_ion_target_state_only"]
        return SimpleNamespace(variable_history=evolved_history, feedback_integral_history={})

    monkeypatch.setattr(native_runner, "advance_recycling_1d_implicit_history", fake_history)

    def fake_rhs(*args, **kwargs):
        captured["diagnostic_nvdp"] = np.asarray(kwargs["field_overrides"]["NVd+"], dtype=np.float64)
        return SimpleNamespace(
            variables={
                "Sd_target_recycle": np.ones((1, 4, 5, 1), dtype=np.float64),
                "Ed_target_recycle": np.ones((1, 4, 5, 1), dtype=np.float64),
            }
        )

    monkeypatch.setattr(native_runner, "compute_recycling_1d_rhs", fake_rhs)

    case = ReferenceCase(
        name="tokamak_recycling_one_step",
        stage="stage7",
        reference_path="examples/tokamak-2D/recycling/BOUT.inp",
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"),
        extra_overrides=(
            "timestep=1",
            "mesh:file={reference_root}/examples/tokamak-2D/recycling/tokamak.nc",
            "hermes:components=(d+, d, e, sheath_boundary_simple, braginskii_collisions, braginskii_friction, braginskii_heat_exchange, sound_speed, reactions, electron_force_balance, braginskii_conduction, recycling)",
        ),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_integrated_2d_recycling_one_step_case(
        case,
        input_path=tokamak_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    expected_initial_fields = native_runner._apply_species_velocity_overrides(
        native_runner._load_curated_case_config(case, tokamak_input),
        field_overrides=initial_fields,
        velocity_field_overrides={
            "d+": np.full((4, 5, 1), 3.0, dtype=np.float64),
            "d": np.full((4, 5, 1), 1.5, dtype=np.float64),
        },
    )
    np.testing.assert_allclose(captured["initial_fields"]["NVd+"], expected_initial_fields["NVd+"])
    np.testing.assert_allclose(captured["initial_fields"]["NVd"], expected_initial_fields["NVd"])
    assert captured["field_template_overrides"] is None
    assert captured["solver_mode"] == "bdf"
    assert captured["preserve_dump_ion_target_state_only"] is False

    expected_fields = native_runner._apply_species_velocity_overrides(
        native_runner._load_curated_case_config(case, tokamak_input),
        field_overrides={name: value[1] for name, value in evolved_history.items()},
        velocity_field_overrides={
            "d+": np.full((4, 5, 1), 4.0, dtype=np.float64),
            "d": np.full((4, 5, 1), 2.5, dtype=np.float64),
        },
    )
    np.testing.assert_allclose(captured["diagnostic_nvdp"], expected_fields["NVd+"])
    np.testing.assert_allclose(result.variables["Sd_target_recycle"][0], 5.0)
    np.testing.assert_allclose(result.variables["Ed_target_recycle"][0], 7.0)


def test_integrated_2d_production_short_window_uses_committed_history_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    production_input = Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-production/data/BOUT.inp")
    if not production_input.exists():
        pytest.skip("integrated 2D production reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    initial_fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": 3.0 * np.ones((4, 5, 1), dtype=np.float64),
    }
    snapshot = LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields=initial_fields,
        optional_fields={
            "SNd+": np.full((4, 5, 1), 1.0, dtype=np.float64),
            "SNVd+": np.full((4, 5, 1), 1.5, dtype=np.float64),
            "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
            "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
            "SNVd": np.full((4, 5, 1), 3.5, dtype=np.float64),
            "SPd": np.full((4, 5, 1), 4.0, dtype=np.float64),
            "Sd_target_recycle": np.full((4, 5, 1), 5.0, dtype=np.float64),
            "Ed_target_recycle": np.full((4, 5, 1), 6.0, dtype=np.float64),
        },
        scalar_values={"Nnorm": 1.0e17, "Tnorm": 1.0, "Bnorm": 1.0, "Cs0": 1.0, "Omega_ci": 1.0, "rho_s0": 1.0},
    )
    snapshot_cache = tmp_path / "integrated_2d_production_rhs_snapshot.npz"
    save_local_reference_snapshot_cache(snapshot, snapshot_cache)
    history_cache = tmp_path / "integrated_2d_production_short_window_optional_history.npz"
    save_optional_field_history_cache(
        {
            "Vd+": np.stack(
                [
                    np.full((4, 5, 1), 3.0 + i, dtype=np.float64)
                    for i in range(6)
                ],
                axis=0,
            ),
            "Sd_target_recycle": np.stack(
                [
                    np.full((4, 5, 1), 5.0 + i, dtype=np.float64)
                    for i in range(6)
                ],
                axis=0,
            ),
            "Ed_target_recycle": np.stack(
                [
                    np.full((4, 5, 1), 7.0 + i, dtype=np.float64)
                    for i in range(6)
                ],
                axis=0,
            ),
        },
        history_cache,
    )

    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_snapshot_cache_path",
        lambda case_name: snapshot_cache if case_name == "integrated_2d_production_rhs" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "_integrated_2d_optional_history_cache_path",
        lambda case_name: history_cache if case_name == "integrated_2d_production_short_window" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference run should not be used when caches are present")),
    )

    evolved_history = {name: np.stack([value + float(i) for i in range(6)], axis=0) for name, value in initial_fields.items()}
    captured: dict[str, object] = {}

    def fake_history(*args, **kwargs):
        captured["initial_fields"] = kwargs["initial_fields"]
        return SimpleNamespace(variable_history=evolved_history, feedback_integral_history={})

    monkeypatch.setattr(native_runner, "advance_recycling_1d_implicit_history", fake_history)

    def fake_rhs(*args, **kwargs):
        captured.setdefault("diagnostic_nvdp_history", []).append(
            np.asarray(kwargs["field_overrides"]["NVd+"], dtype=np.float64)
        )
        return SimpleNamespace(
            variables={
                "Sd_target_recycle": np.ones((1, 4, 5, 1), dtype=np.float64),
                "Ed_target_recycle": np.ones((1, 4, 5, 1), dtype=np.float64),
            }
        )

    monkeypatch.setattr(native_runner, "compute_recycling_1d_rhs", fake_rhs)

    case = ReferenceCase(
        name="integrated_2d_production_short_window",
        stage="stage7",
        reference_path=str(production_input),
        parity_mode="short_window",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "Sd_target_recycle", "Ed_target_recycle"),
        trim_x_guards=True,
        trim_y_guards=True,
        extra_overrides=("nout=5",),
    )

    result = native_runner._run_integrated_2d_recycling_short_window_case(
        case,
        input_path=production_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    expected_initial_fields = native_runner._apply_species_velocity_overrides(
        load_bout_input(production_input),
        field_overrides=initial_fields,
        velocity_field_overrides={"d+": np.full((4, 5, 1), 3.0, dtype=np.float64)},
    )
    np.testing.assert_allclose(captured["initial_fields"]["NVd+"], expected_initial_fields["NVd+"])
    assert len(captured["diagnostic_nvdp_history"]) == 6
    expected_fields = native_runner._apply_species_velocity_overrides(
        load_bout_input(production_input),
        field_overrides={name: value[1] for name, value in evolved_history.items()},
        velocity_field_overrides={"d+": np.full((4, 5, 1), 4.0, dtype=np.float64)},
    )
    np.testing.assert_allclose(captured["diagnostic_nvdp_history"][1], expected_fields["NVd+"])
    np.testing.assert_allclose(result.variables["Sd_target_recycle"][0], 5.0)
    np.testing.assert_allclose(result.variables["Sd_target_recycle"][-1], 1.0)
    np.testing.assert_allclose(result.variables["Ed_target_recycle"][0], 7.0)


def test_integrated_2d_recycling_medium_window_honors_manifest_nout_override(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("integrated 2D recycling reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    initial_fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": 2.0 * np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": 3.0 * np.ones((4, 5, 1), dtype=np.float64),
    }
    evolved_history = {name: np.stack([value + float(index) for index in range(21)], axis=0) for name, value in initial_fields.items()}

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(summary=_FakeSummary(artifacts={"BOUT.dmp.0.nc": "/tmp/fake-dump.nc"})),
    )
    monkeypatch.setattr(
        native_runner,
        "load_local_reference_snapshot",
        lambda *args, **kwargs: LocalReferenceSnapshot(
            mesh=mesh,
            metrics=metrics,
            fields=initial_fields,
            optional_fields={
                "SNd+": np.full((4, 5, 1), 1.0, dtype=np.float64),
                "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
                "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
                "SPd": np.full((4, 5, 1), 4.0, dtype=np.float64),
                "Sd_target_recycle": np.full((4, 5, 1), 5.0, dtype=np.float64),
                "Ed_target_recycle": np.full((4, 5, 1), 6.0, dtype=np.float64),
            },
            scalar_values={"Nnorm": 1.0e17},
        ),
    )

    captured: dict[str, object] = {}

    def fake_history(*args, **kwargs):
        captured["steps"] = kwargs["steps"]
        return SimpleNamespace(variable_history=evolved_history, feedback_integral_history={})

    monkeypatch.setattr(native_runner, "advance_recycling_1d_implicit_history", fake_history)

    def fake_rhs(*args, **kwargs):
        fields = kwargs["field_overrides"]
        density = np.asarray(fields["Nd+"], dtype=np.float64)
        pressure = np.asarray(fields["Pd+"], dtype=np.float64)
        return SimpleNamespace(
            variables={
                "Sd_target_recycle": density[None, ...],
                "Ed_target_recycle": pressure[None, ...],
            }
        )

    monkeypatch.setattr(native_runner, "compute_recycling_1d_rhs", fake_rhs)

    case = ReferenceCase(
        name="integrated_2d_recycling_medium_window",
        stage="stage7",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="short_window",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "Sd_target_recycle", "Ed_target_recycle"),
        extra_overrides=("nout=20",),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    result = native_runner._run_integrated_2d_recycling_medium_window_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured["steps"] == 20
    assert len(result.time_points) == 21
    assert result.time_points[-1] == pytest.approx(0.002)
    assert result.variables["Nd+"].shape == (21, 2, 3, 1)
    assert result.variables["Ed_target_recycle"].shape == (21, 2, 3, 1)
    np.testing.assert_allclose(result.variables["Sd_target_recycle"][0], 1.0)
    np.testing.assert_allclose(result.variables["Ed_target_recycle"][0], 2.0)


def test_integrated_2d_production_one_step_stays_within_operational_target_band() -> None:
    entries = _run_integrated_2d_case_against_committed_baseline("integrated_2d_production_one_step")

    assert entries["Pe"].max_abs_diff < 1.7e-1
    assert entries["Nd"].max_abs_diff < 1.2e-2
    assert entries["Pd+"].max_abs_diff < 6.0e-3
    assert entries["Nd+"].max_abs_diff < 5.0e-3
    assert entries["Sd_target_recycle"].max_abs_diff < 2.0e-3
    assert entries["NVd+"].max_abs_diff < 1.0e-3
    assert entries["Ed_target_recycle"].max_abs_diff < 4.0e-4


def test_integrated_2d_recycling_one_step_stays_within_operational_target_band() -> None:
    entries = _run_integrated_2d_case_against_committed_baseline("integrated_2d_recycling_one_step")

    assert entries["Sd_target_recycle"].max_abs_diff < 1.1
    assert entries["Ed_target_recycle"].max_abs_diff < 1.0e-2
    assert entries["Pd+"].max_abs_diff < 3.0e-3
    assert entries["NVd+"].max_abs_diff < 2.0e-3
    assert entries["Nd+"].max_abs_diff < 2.0e-4
    assert entries["Pe"].max_abs_diff < 1.0e-4
    assert entries["Pd"].max_abs_diff < 2.0e-5
    assert entries["Nd"].max_abs_diff < 1.2e-4
    assert entries["NVd"].max_abs_diff < 1.0e-9


def test_integrated_2d_recycling_short_window_stays_within_operational_target_band() -> None:
    entries = _run_integrated_2d_case_against_committed_baseline("integrated_2d_recycling_short_window")

    assert entries["Sd_target_recycle"].max_abs_diff < 1.1
    assert entries["Ed_target_recycle"].max_abs_diff < 1.0e-2
    assert entries["Pd+"].max_abs_diff < 1.5e-2
    assert entries["NVd+"].max_abs_diff < 6.0e-3
    assert entries["Nd+"].max_abs_diff < 6.0e-4
    assert entries["Pe"].max_abs_diff < 2.0e-3
    assert entries["Pd"].max_abs_diff < 1.0e-4
    assert entries["Nd"].max_abs_diff < 6.0e-4
    assert entries["NVd"].max_abs_diff < 1.0e-8


def test_integrated_2d_recycling_medium_window_stays_within_operational_target_band() -> None:
    entries = _run_integrated_2d_case_against_committed_baseline("integrated_2d_recycling_medium_window")

    assert entries["Sd_target_recycle"].max_abs_diff < 1.1
    assert entries["Ed_target_recycle"].max_abs_diff < 1.0e-2
    assert entries["Pd+"].max_abs_diff < 5.0e-2
    assert entries["NVd+"].max_abs_diff < 2.5e-2
    assert entries["Nd+"].max_abs_diff < 3.0e-3
    assert entries["Pe"].max_abs_diff < 1.5e-2
    assert entries["Pd"].max_abs_diff < 3.0e-4
    assert entries["Nd"].max_abs_diff < 2.5e-3
    assert entries["NVd"].max_abs_diff < 1.0e-8


def test_integrated_2d_production_short_window_stays_within_operational_target_band() -> None:
    entries = _run_integrated_2d_case_against_committed_baseline("integrated_2d_production_short_window")

    assert entries["Pe"].max_abs_diff < 1.5
    assert entries["NVd+"].max_abs_diff < 5.5e-1
    assert entries["Nd"].max_abs_diff < 3.0e-1
    assert entries["Nd+"].max_abs_diff < 9.0e-2
    assert entries["Pd+"].max_abs_diff < 8.0e-2
    assert entries["Pd"].max_abs_diff < 7.0e-2
    assert entries["Sd_target_recycle"].max_abs_diff < 7.0e-3
    assert entries["Ed_target_recycle"].max_abs_diff < 4.0e-4


def test_integrated_2d_production_rhs_stays_within_operational_summary_band() -> None:
    entries = _run_integrated_2d_case_against_committed_baseline("integrated_2d_production_rhs")

    assert entries["ddt(Pe)"].max_abs_diff < 9.0e-2
    assert entries["ddt(Pd)"].max_abs_diff < 2.0e-3
    assert entries["ddt(NVd+)"].max_abs_diff == pytest.approx(0.0)


def test_tokamak_recycling_one_step_stays_within_operational_target_band() -> None:
    entries = _run_direct_tokamak_case_against_committed_baseline("tokamak_recycling_one_step")

    assert entries["Pe"].max_abs_diff < 7.0e-3
    assert entries["Pd+"].max_abs_diff < 5.0e-5
    assert entries["Nd+"].max_abs_diff < 3.0e-5
    assert entries["NVd+"].max_abs_diff < 5.0e-5
    assert entries["Nd"].max_abs_diff < 1.1e-4
    assert entries["Pd"].max_abs_diff < 4.0e-6
    assert entries["NVd"].max_abs_diff < 3.5e-7


def test_tokamak_recycling_dthe_rhs_matches_committed_baseline_exactly() -> None:
    entries = _run_direct_tokamak_case_against_committed_baseline("tokamak_recycling_dthe_rhs")

    for entry in entries.values():
        assert entry.max_abs_diff == pytest.approx(0.0)


def test_tokamak_recycling_dthene_rhs_matches_committed_baseline_exactly() -> None:
    entries = _run_direct_tokamak_case_against_committed_baseline("tokamak_recycling_dthene_rhs")

    for entry in entries.values():
        assert entry.max_abs_diff == pytest.approx(0.0)


def test_tokamak_recycling_dthe_drifts_rhs_matches_committed_baseline_exactly() -> None:
    entries = _run_direct_tokamak_case_against_committed_baseline("tokamak_recycling_dthe_drifts_rhs")

    for entry in entries.values():
        assert entry.max_abs_diff == pytest.approx(0.0)


def test_tokamak_recycling_dthe_one_step_stays_within_operational_target_band() -> None:
    entries = _run_direct_tokamak_case_against_committed_baseline("tokamak_recycling_dthe_one_step")

    assert entries["Pe"].max_abs_diff < 3.0e-4
    assert entries["Pd+"].max_abs_diff < 1.5e-2
    assert entries["NVd+"].max_abs_diff < 5.0e-2
    assert entries["Pt+"].max_abs_diff < 1.5e-2
    assert entries["NVt+"].max_abs_diff < 7.0e-2
    assert entries["Phe+"].max_abs_diff < 5.0e-5
    assert entries["NVhe+"].max_abs_diff < 1.0e-4
    assert entries["Nd+"].max_abs_diff < 5.0e-5
    assert entries["Nt+"].max_abs_diff < 5.0e-5
    assert entries["Nd"].max_abs_diff < 5.0e-6
    assert entries["Nt"].max_abs_diff < 5.0e-6


def test_tokamak_recycling_dthe_drifts_one_step_stays_within_operational_target_band() -> None:
    entries = _run_direct_tokamak_case_against_committed_baseline("tokamak_recycling_dthe_drifts_one_step")

    assert entries["Pe"].max_abs_diff < 1.0e-2
    assert entries["Pd+"].max_abs_diff < 1.0e-2
    assert entries["NVd+"].max_abs_diff < 1.0e-2
    assert entries["Pt+"].max_abs_diff < 1.0e-2
    assert entries["NVt+"].max_abs_diff < 1.0e-2
    assert entries["Phe+"].max_abs_diff < 1.0e-4
    assert entries["NVhe+"].max_abs_diff < 1.0e-4
    assert "phi" not in entries
    assert "Vort" not in entries


def test_tokamak_recycling_dthene_one_step_stays_within_operational_target_band() -> None:
    entries = _run_direct_tokamak_case_against_committed_baseline("tokamak_recycling_dthene_one_step")

    assert entries["Pe"].max_abs_diff < 3.0e-3
    assert entries["Pd+"].max_abs_diff < 3.5e-3
    assert entries["NVd+"].max_abs_diff < 1.0e-5
    assert entries["Nd+"].max_abs_diff < 3.0e-3
    assert entries["Pd"].max_abs_diff < 1.0e-4
    assert entries["Nd"].max_abs_diff < 5.0e-7
    assert entries["Pt+"].max_abs_diff < 3.5e-3
    assert entries["NVt+"].max_abs_diff < 1.0e-5
    assert entries["Nt+"].max_abs_diff < 3.0e-3
    assert entries["Pt"].max_abs_diff < 1.0e-4
    assert entries["Nt"].max_abs_diff < 5.0e-7
    assert entries["Phe+"].max_abs_diff < 5.0e-8
    assert entries["NVhe+"].max_abs_diff < 1.0e-8
    assert entries["Nhe+"].max_abs_diff < 2.0e-8
    assert entries["Pne+"].max_abs_diff < 1.0e-9
    assert entries["NVne+"].max_abs_diff < 1.0e-9
    assert entries["Nne+"].max_abs_diff < 1.0e-9
