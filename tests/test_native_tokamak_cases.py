from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.native import runner as native_runner
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.reference_dump import LocalReferenceSnapshot
from jax_drb.reference.cases import ReferenceCase

_REFERENCE_INPUT = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/diffusion-flow-evolveT/BOUT.inp")


@dataclass(frozen=True)
class _FakeSummary:
    artifacts: dict[str, str]
    time_points: tuple[float, ...]
    overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FakeExecution:
    summary: _FakeSummary


def _tokamak_snapshot(*, time_index: int) -> LocalReferenceSnapshot:
    mesh = StructuredMesh(
        nx=6,
        ny=8,
        nz=1,
        mxg=2,
        myg=2,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=-1,
        jyseps2_1=7,
        jyseps1_2=7,
        jyseps2_2=7,
        ny_inner=8,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=jnp.arange(6, dtype=jnp.float64),
        y=jnp.arange(12, dtype=jnp.float64) - 2.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((6, 12, 1), dtype=jnp.float64)
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
    scale = float(time_index + 1)
    return LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields={
            "Nh": np.full((6, 12, 1), 1.0 * scale, dtype=np.float64),
            "Ph": np.full((6, 12, 1), 2.0 * scale, dtype=np.float64),
            "NVh": np.full((6, 12, 1), 3.0 * scale, dtype=np.float64),
        },
        optional_fields={},
        scalar_values={"Nnorm": 1.0e19, "Tnorm": 10.0, "Bnorm": 1.0, "Cs0": 2.0, "Omega_ci": 3.0, "rho_s0": 4.0},
    )


def test_tokamak_diffusion_flow_one_step_stacks_initial_and_final_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("tokamak diffusion-flow reference input is unavailable")

    captured_time_indices: list[int] = []

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-tokamak-diffusion-flow.nc"},
                time_points=(0.0, 50.0),
                overrides=("nout=1",),
            )
        ),
    )

    def fake_load_snapshot(*args, **kwargs):
        time_index = kwargs["time_index"]
        captured_time_indices.append(time_index)
        return _tokamak_snapshot(time_index=time_index)

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)

    case = ReferenceCase(
        name="tokamak_diffusion_flow_one_step",
        stage="stage7",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Nh", "Ph", "NVh"),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_tokamak_diffusion_flow_one_step_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured_time_indices == [0, 1]
    assert result.time_points == (0.0, 50.0)
    assert result.variables["Nh"].shape == (2, 2, 8, 1)
    assert result.variables["Ph"].shape == (2, 2, 8, 1)
    assert result.variables["NVh"].shape == (2, 2, 8, 1)
    np.testing.assert_allclose(result.variables["Nh"][0], 1.0)
    np.testing.assert_allclose(result.variables["Nh"][1], 2.0)
    np.testing.assert_allclose(result.variables["Ph"][0], 2.0)
    np.testing.assert_allclose(result.variables["Ph"][1], 4.0)
    np.testing.assert_allclose(result.variables["NVh"][0], 3.0)
    np.testing.assert_allclose(result.variables["NVh"][1], 6.0)
    assert result.payload["compare_variables"] == ["Nh", "Ph", "NVh"]
