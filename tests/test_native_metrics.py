from __future__ import annotations

import numpy as np

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.runtime.run_config import RunConfiguration


def test_structured_metrics_match_normalized_diffusion_scalars() -> None:
    config = parse_bout_input(
        """
        nout = 5
        timestep = 1000

        [mesh]
        nx = 10
        ny = 10
        nz = 1
        dx = 0.0075 + 0.005*x
        dy = 0.01
        dz = 0.01
        J = 1

        [model]
        components = h

        [h]
        type = evolve_density, evolve_pressure, anomalous_diffusion
        anomalous_D = 2
        thermal_conduction = false

        [Nh]
        function = 1
        bndry_all = neumann

        [Ph]
        function = 1
        bndry_all = neumann
        """
    )
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)

    np.testing.assert_allclose(
        np.asarray(metrics.dx[:, mesh.ystart, 0]),
        np.array(
            [
                5986.770814416301,
                6785.006922672475,
                7583.243030928649,
                8381.479139184821,
                9179.715247440996,
                9977.95135569717,
                10776.187463953344,
                11574.423572209517,
                12372.659680465691,
                13170.895788721864,
            ]
        ),
        rtol=1e-9,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(metrics.dy[:, mesh.ystart, 0]), 0.01)
    np.testing.assert_allclose(np.asarray(metrics.dz[:, mesh.ystart, 0]), 0.01)
    np.testing.assert_allclose(np.asarray(metrics.J[:, mesh.ystart, 0]), 978.7151425755138, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(metrics.g11[:, mesh.ystart, 0]),
        957883.3303066083,
        rtol=1e-12,
        atol=1e-9,
    )
    np.testing.assert_allclose(np.asarray(metrics.g23[:, mesh.ystart, 0]), 0.0)
    np.testing.assert_allclose(np.asarray(metrics.Bxy[:, mesh.ystart, 0]), 1.0)
