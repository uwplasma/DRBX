from __future__ import annotations

import math

from drbx.config.boutinp import parse_bout_input
from drbx.runtime.run_config import RunConfiguration


def test_run_configuration_resolves_time_mesh_solver_and_scalars() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 95788 * 0.05
        MZ = 81
        zperiod = 5
        MXG = 0

        tnorm_setting = 100

        [mesh]
        file = "grid.nc"
        nx = 68
        ny = 32
        nz = 64
        Bxy = 0.35
        Rxy = 1.5
        dz = 2 * pi / nz
        extrapolate_y = false

        [mesh:paralleltransform]
        type = shifted

        [solver]
        type = pvode
        mxstep = 1e9
        use_precon = true
        cvode_max_order = 3

        [model]
        components = e, vorticity
        Nnorm = 1e19
        Tnorm = tnorm_setting
        Bnorm = mesh:Bxy

        [e]
        type = evolve_density, evolve_pressure
        """
    )

    run_config = RunConfiguration.from_config(config)

    assert run_config.time.nout == 1
    assert math.isclose(run_config.time.timestep, 95788.0 * 0.05)
    assert run_config.mesh.file == "grid.nc"
    assert run_config.mesh.nx == 68
    assert run_config.mesh.myg == 2
    assert run_config.mesh.mxg == 0
    assert run_config.mesh.mz == 81
    assert math.isclose(run_config.mesh.zperiod or 0.0, 5.0)
    assert run_config.mesh.parallel_transform.type == "shifted"
    assert math.isclose(run_config.mesh.resolved_scalars["dz"], 2.0 * math.pi / 64.0)
    assert run_config.solver.type == "pvode"
    assert run_config.solver.mxstep == 1_000_000_000
    assert run_config.solver.use_precon is True
    assert run_config.components[0].label == "e:evolve_density"
    assert run_config.components[1].label == "e:evolve_pressure"
    assert run_config.components[2].label == "vorticity"
    assert math.isclose(run_config.model_scalars["Tnorm"], 100.0)
    assert math.isclose(run_config.normalization.Bnorm, 0.35)
