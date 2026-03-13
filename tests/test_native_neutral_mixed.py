from __future__ import annotations

import numpy as np
import pytest

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.neutral_mixed import (
    build_neutral_mixed_transport_operators,
    compute_neutral_mixed_rhs,
    initialize_neutral_mixed_state,
)
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars


_NEUTRAL_MIXED_INPUT = """
nout = 15
timestep = 20

[mesh]
nx = 10
ny = 10
nz = 10

dx = 1e-3
dy = 1e-3
dz = 1e-3

yn = y / (2π)
zn = z / (2π)

J = 1

[solver]
mxstep = 1000

[model]
components = h

[h]
type = neutral_mixed

[Nh]
function = exp(-(x - 0.5)^2 - (mesh:yn - 0.5)^2 - (mesh:zn - 0.5)^2)

[Ph]
function = 0.1 * Nh:function
"""


def _build_case():
    config = parse_bout_input(_NEUTRAL_MIXED_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    state = initialize_neutral_mixed_state(config, section="h", mesh=mesh)
    scalars = resolved_dataset_scalars(run_config)
    rhs = compute_neutral_mixed_rhs(
        config,
        state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
    )
    return config, run_config, mesh, metrics, state, rhs


def test_neutral_mixed_diffusion_matches_known_case_values() -> None:
    _, _, _, _, _, rhs = _build_case()
    interior = rhs.diffusion[2:8, 2:12, :]

    assert interior.min() == pytest.approx(0.4208780853793485, rel=1e-12, abs=1e-12)
    assert interior.max() == pytest.approx(1.4907267304514251, rel=1e-12, abs=1e-12)
    assert rhs.diffusion[5, 5, 5] == pytest.approx(1.170446626082471, rel=1e-12, abs=1e-12)


def test_neutral_mixed_rhs_tracks_reference_case_center_values() -> None:
    _, _, _, _, state, rhs = _build_case()

    assert state.density[5, 5, 5] == pytest.approx(0.9709848197438855, rel=1e-12, abs=1e-12)
    assert state.pressure[5, 5, 5] == pytest.approx(0.09709848197438856, rel=1e-12, abs=1e-12)
    assert rhs.density[5, 5, 5] == pytest.approx(-0.07201691370446264, rel=1e-12, abs=1e-12)
    assert rhs.pressure[5, 5, 5] == pytest.approx(-0.012002818950743778, rel=1e-12, abs=1e-12)
    assert rhs.momentum[5, 5, 5] == pytest.approx(-0.002947131977001758, rel=1e-12, abs=1e-12)


def test_neutral_mixed_rhs_keeps_guard_derivatives_zero_in_x() -> None:
    _, _, _, _, _, rhs = _build_case()

    assert rhs.density[:2].max() == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert rhs.density[8:].max() == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert rhs.pressure[:2].max() == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert rhs.pressure[8:].max() == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert rhs.momentum[:2].max() == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert rhs.momentum[8:].max() == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_neutral_mixed_transport_operators_reproduce_frozen_density_rhs() -> None:
    _, _, mesh, metrics, state, rhs = _build_case()
    density = state.density
    pressure = state.pressure
    density_limited = density.clip(min=1.0e-8)
    pressure_limited = pressure.clip(min=1.0e-11)
    log_pressure = np.log(pressure_limited)
    operators = build_neutral_mixed_transport_operators(
        rhs.diffusion,
        log_pressure,
        mesh=mesh,
        metrics=metrics,
    )

    active_density_rhs = rhs.density[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :]
    for j_offset, operator in enumerate(operators):
        density_slice = density_limited[mesh.xstart : mesh.xend + 1, mesh.ystart + j_offset, :]
        actual = (operator @ density_slice.reshape(-1)).reshape(density_slice.shape)
        expected = active_density_rhs[:, j_offset, :]
        error = actual - expected
        assert np.max(np.abs(error)) < 3.5e-3
        assert np.sqrt(np.mean(np.square(error))) < 1.0e-3
