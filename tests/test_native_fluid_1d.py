from __future__ import annotations

import numpy as np

from drbx.config.boutinp import parse_bout_input
from drbx.native.fluid_1d import advance_mms_history, compute_mms_rhs, initialize_mms_state
from drbx.native.mesh import build_structured_mesh
from drbx.native.metrics import build_structured_metrics
from drbx.runtime.run_config import RunConfiguration

_FLUID_1D_MMS_INPUT = """
nout = 50
timestep = 0.1

MXG = 0

[mesh]
nx = 1
ny = 128
nz = 1
Ly = 10
dy = Ly / ny
J = 1

[solver]
mxstep = 10000
rtol = 1e-7
mms = true

[model]
components = i
normalise_metric = false
Nnorm = 1e18
Bnorm = 1
Tnorm = 5

[i]
type = evolve_density, evolve_pressure, evolve_momentum
charge = 1.0
AA = 2.0
thermal_conduction = false

[Ni]
solution = 1 - 0.1*sin(t - 2.0*y)
source = -0.1*cos(t - 2.0*y) + 0.0628318530717959*cos(2*t + y)

[Pi]
solution = 0.1*cos(t + 3.0*y) + 1
source = (0.0628318530717959*cos(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.0125663706143592*sin(2*t + y)*cos(t - 2.0*y)/(1 - 0.1*sin(t - 2.0*y))^2)*(0.0666666666666667*cos(t + 3.0*y) + 0.666666666666667) - 0.1*sin(t + 3.0*y) + 0.0628318530717959*(0.1*cos(t + 3.0*y) + 1)*cos(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.0188495559215388*sin(t + 3.0*y)*sin(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.0125663706143592*(0.1*cos(t + 3.0*y) + 1)*sin(2*t + y)*cos(t - 2.0*y)/(1 - 0.1*sin(t - 2.0*y))^2

[NVi]
solution = 0.2*sin(2*t + y)
source = -0.188495559215388*sin(t + 3.0*y) + 0.4*cos(2*t + y) + 0.0251327412287183*sin(2*t + y)*cos(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.00251327412287184*sin(2*t + y)^2*cos(t - 2.0*y)/(1 - 0.1*sin(t - 2.0*y))^2
"""


def test_initialize_mms_state_wraps_periodic_y_guards() -> None:
    config = parse_bout_input(_FLUID_1D_MMS_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)

    state = initialize_mms_state(config, section="i", mesh=mesh)
    density = np.asarray(state.density[0, :, 0])

    np.testing.assert_allclose(density[: mesh.myg], density[mesh.yend - mesh.ystart + 1 : mesh.yend - mesh.ystart + 1 + mesh.myg])
    np.testing.assert_allclose(density[-mesh.myg :], density[mesh.ystart : mesh.ystart + mesh.myg])


def test_compute_mms_rhs_matches_expected_discrete_rhs_sample() -> None:
    config = parse_bout_input(_FLUID_1D_MMS_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    state = initialize_mms_state(config, section="i", mesh=mesh)

    rhs = compute_mms_rhs(config, state, section="i", mesh=mesh, metrics=metrics, atomic_mass=2.0, time=0.0)
    interior = slice(mesh.ystart, mesh.ystart + 8)

    np.testing.assert_allclose(
        np.asarray(rhs.density[0, interior, 0]),
        np.array([-0.09988946, -0.09892711, -0.09701223, -0.09416325, -0.0904076, -0.08578143, -0.08032928, -0.07410364]),
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(rhs.pressure[0, interior, 0]),
        np.array([-0.01175049, -0.0177226, -0.03607704, -0.04936326, -0.06158035, -0.07246401, -0.08177884, -0.0893234]),
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(rhs.momentum[0, interior, 0]),
        np.array([0.39982405, 0.39876197, 0.3967422, 0.39377142, 0.3898585, 0.38501444, 0.37925232, 0.37258722]),
        rtol=1e-7,
        atol=1e-9,
    )


def test_advance_mms_history_produces_periodic_short_window_state_history() -> None:
    config = parse_bout_input(_FLUID_1D_MMS_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)

    history = advance_mms_history(
        config,
        section="i",
        mesh=mesh,
        metrics=metrics,
        atomic_mass=2.0,
        timestep=0.1,
        steps=2,
        substeps=20,
    )

    density = np.asarray(history.density_history)
    assert density.shape == (3, 1, mesh.local_ny, 1)
    assert np.all(np.isfinite(density))
    np.testing.assert_allclose(density[1, 0, : mesh.myg, 0], density[1, 0, -2 * mesh.myg : -mesh.myg, 0])
    np.testing.assert_allclose(density[1, 0, -mesh.myg :, 0], density[1, 0, mesh.myg : 2 * mesh.myg, 0])
