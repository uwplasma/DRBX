from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from jax import grad
from jax.errors import TracerArrayConversionError
import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native import runner as native_runner
from jax_drb.native.reference_dump import LocalReferenceSnapshot
from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.recycling_1d import _initialize_species, _prepare_open_field_states
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.reference.cases import ReferenceCase

_REFERENCE_INPUT = Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-recycling/data/BOUT.inp")


@dataclass(frozen=True)
class _FakeSummary:
    artifacts: dict[str, str]


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
    )
    assert captured["scalar_names"] == ("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0")


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
                "SNVd+": np.full((4, 5, 1), 1.5, dtype=np.float64),
                "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
                "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
                "SNVd": np.full((4, 5, 1), 3.5, dtype=np.float64),
                "SPd": np.full((4, 5, 1), 4.0, dtype=np.float64),
                "Sd_target_recycle": np.full((4, 5, 1), 5.0, dtype=np.float64),
                "Ed_target_recycle": np.full((4, 5, 1), 6.0, dtype=np.float64),
            },
            scalar_values={"Nnorm": 1.0e17},
        ),
    )

    captured: dict[str, object] = {}

    def fake_history(*args, **kwargs):
        captured["initial_fields"] = kwargs["initial_fields"]
        captured["density_source_overrides"] = kwargs["density_source_overrides"]
        captured["pressure_source_overrides"] = kwargs["pressure_source_overrides"]
        captured["momentum_source_overrides"] = kwargs["momentum_source_overrides"]
        captured["preserve_dump_target_state"] = kwargs["preserve_dump_target_state"]
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
    assert tuple(captured["density_source_overrides"]) == ("d+", "d")
    assert tuple(captured["pressure_source_overrides"]) == ("d+", "d")
    assert tuple(captured["momentum_source_overrides"]) == ("d+", "d")
    assert captured["preserve_dump_target_state"] is True
    assert result.time_points == (0.0, 0.0001)
    assert result.variables["Nd+"].shape == (2, 2, 3, 1)
    assert result.variables["Sd_target_recycle"].shape == (2, 2, 3, 1)
    np.testing.assert_allclose(result.variables["Sd_target_recycle"][0], 5.0)
    np.testing.assert_allclose(result.variables["Ed_target_recycle"][0], 6.0)


def test_integrated_2d_recycling_short_window_reuses_staged_transient_path(monkeypatch: pytest.MonkeyPatch) -> None:
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
                "SNVd+": np.full((4, 5, 1), 1.5, dtype=np.float64),
                "SPd+": np.full((4, 5, 1), 2.0, dtype=np.float64),
                "SNd": np.full((4, 5, 1), 3.0, dtype=np.float64),
                "SNVd": np.full((4, 5, 1), 3.5, dtype=np.float64),
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
        captured["preserve_dump_target_state"] = kwargs["preserve_dump_target_state"]
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
    assert captured["preserve_dump_target_state"] is True
    assert result.time_points == (0.0, 0.0001, 0.0002, 0.00030000000000000003, 0.0004, 0.0005)
    assert result.variables["Nd+"].shape == (6, 2, 3, 1)
    assert result.variables["Sd_target_recycle"].shape == (6, 2, 3, 1)
    np.testing.assert_allclose(result.variables["Sd_target_recycle"][0], 5.0)
    np.testing.assert_allclose(result.variables["Ed_target_recycle"][0], 6.0)


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
    np.testing.assert_allclose(result.variables["Sd_target_recycle"][0], 5.0)
    np.testing.assert_allclose(result.variables["Ed_target_recycle"][0], 6.0)
