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

_REFERENCE_INPUT = Path("/Users/rogerio/local/hermes-3/examples/other/linear/annulus-isothermal-he-emag/BOUT.inp")


@dataclass(frozen=True)
class _FakeSummary:
    artifacts: dict[str, str]
    time_points: tuple[float, ...]
    overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FakeExecution:
    summary: _FakeSummary


def _annulus_snapshot() -> LocalReferenceSnapshot:
    mesh = StructuredMesh(
        nx=8,
        ny=4,
        nz=6,
        mxg=2,
        myg=2,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=-1,
        jyseps2_1=3,
        jyseps1_2=3,
        jyseps2_2=3,
        ny_inner=4,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=jnp.arange(8, dtype=jnp.float64),
        y=jnp.arange(8, dtype=jnp.float64) - 2.0,
        z=jnp.arange(6, dtype=jnp.float64),
    )
    ones = jnp.ones((8, 8, 6), dtype=jnp.float64)
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
    ne = np.full((8, 8, 6), 0.4, dtype=np.float64)
    nhe = np.full((8, 8, 6), 0.4, dtype=np.float64)
    nve = np.full((8, 8, 6), -2.0 / 1836.0, dtype=np.float64)
    nvhe = np.full((8, 8, 6), 0.5, dtype=np.float64)
    fields = {
        "Apar": np.full((8, 8, 6), 1.5e-3, dtype=np.float64),
        "Ne": ne,
        "Nhe+": nhe,
        "NVe": nve,
        "NVhe+": nvhe,
    }
    optional_fields = {
        "ddt(Ne)": np.full((8, 8, 6), 0.125, dtype=np.float64),
        "ddt(NVe)": np.full((8, 8, 6), -0.25, dtype=np.float64),
        "ddt(Vort)": np.full((8, 8, 6), 0.5, dtype=np.float64),
    }
    return LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields=fields,
        optional_fields=optional_fields,
        scalar_values={"Nnorm": 1.0e18, "Tnorm": 1.0, "Bnorm": 1.0, "Cs0": 1.0, "Omega_ci": 1.0, "rho_s0": 1.0},
    )


def test_annulus_he_emag_rhs_uses_em_native_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("annulus electromagnetic reference input is unavailable")

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-annulus-he-emag.nc"},
                time_points=(0.0,),
                overrides=("nout=0", "e:diagnose=true"),
            )
        ),
    )
    monkeypatch.setattr(
        native_runner,
        "load_local_reference_snapshot",
        lambda *args, **kwargs: _annulus_snapshot(),
    )

    case = ReferenceCase(
        name="annulus_he_emag_rhs",
        stage="stage8",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Apar", "alpha_em", "ddt(Ne)", "ddt(NVe)", "ddt(Vort)"),
        extra_overrides=("nout=0", "e:diagnose=true", "vorticity:diagnose=true", "electromagnetic:diagnose=true"),
    )

    result = native_runner._run_annulus_he_emag_rhs_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert result.time_points == (0.0,)
    assert result.variables["Apar"].shape == (1, 8, 8, 6)
    np.testing.assert_allclose(result.variables["Apar"][0], 1.5e-3)
    np.testing.assert_allclose(result.variables["ddt(Ne)"][0], 0.125)
    np.testing.assert_allclose(result.variables["ddt(NVe)"][0], -0.25)
    np.testing.assert_allclose(result.variables["ddt(Vort)"][0], 0.5)
    np.testing.assert_allclose(result.variables["Ajpar"][0], (0.5 / 4.0) + ((2.0 / 1836.0) * (1836.0 / 60.0)))
    np.testing.assert_allclose(result.variables["alpha_em"][0], 0.4 * ((1836.0 / 60.0) + 0.25))
    assert result.payload["compare_variables"] == [
        "Apar",
        "alpha_em",
        "ddt(Ne)",
        "ddt(NVe)",
        "ddt(Vort)",
    ]
