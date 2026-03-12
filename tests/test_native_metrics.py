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
    np.testing.assert_allclose(
        np.asarray(metrics.g33[:, mesh.ystart, 0]),
        1.0439684845591577e-06,
        rtol=1e-8,
        atol=1e-15,
    )
    np.testing.assert_allclose(np.asarray(metrics.g22[:, mesh.ystart, 0]), 1.0)
    np.testing.assert_allclose(np.asarray(metrics.g23[:, mesh.ystart, 0]), 0.0)
    np.testing.assert_allclose(np.asarray(metrics.Bxy[:, mesh.ystart, 0]), 1.0)


def test_structured_metrics_respect_normalise_metric_false() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 0.1
        MXG = 0

        [mesh]
        nx = 1
        ny = 8
        nz = 1
        dy = 10 / ny
        J = 1

        [solver]
        mms = true

        [model]
        components = i
        normalise_metric = false
        Nnorm = 1e18
        Tnorm = 5
        Bnorm = 1

        [i]
        type = evolve_density, evolve_pressure, evolve_momentum
        thermal_conduction = false
        AA = 2

        [Ni]
        solution = 1
        source = 0

        [Pi]
        solution = 1
        source = 0

        [NVi]
        solution = 0
        source = 0
        """
    )
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)

    np.testing.assert_allclose(np.asarray(metrics.dx[:, mesh.ystart, 0]), 1.0)
    np.testing.assert_allclose(np.asarray(metrics.dy[:, mesh.ystart, 0]), 1.25)
    np.testing.assert_allclose(np.asarray(metrics.J[:, mesh.ystart, 0]), 1.0)
    np.testing.assert_allclose(np.asarray(metrics.g11[:, mesh.ystart, 0]), 1.0)
    np.testing.assert_allclose(np.asarray(metrics.g33[:, mesh.ystart, 0]), 1.0)
    np.testing.assert_allclose(np.asarray(metrics.g22[:, mesh.ystart, 0]), 1.0)


def test_structured_metrics_default_periodic_dz_matches_reference_convention() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 1

        [mesh]
        nx = 6
        ny = 1
        nz = 8
        J = 1

        [model]
        components = e
        Nnorm = 1e19
        Tnorm = 100
        Bnorm = 1

        [e]
        type = evolve_density

        [Ne]
        function = 1
        """
    )
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)

    np.testing.assert_allclose(np.asarray(metrics.dz[:, mesh.ystart, 0]), np.pi / 4.0, rtol=1e-12, atol=1e-12)


def test_structured_metrics_recalculate_metric_matches_blob_reference_geometry() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 1
        MYG = 0

        [mesh]
        nx = 260
        ny = 1
        nz = 256
        Lrad = 0.05
        Lpol = 0.05
        Bpxy = 0.35
        Rxy = 1.5
        dx = Lrad * Rxy * Bpxy / (nx - 4)
        dz = Lpol / Rxy / nz
        hthe = 1
        sinty = 0
        Bxy = Bpxy
        Btxy = 0

        [mesh:paralleltransform]
        type = identity

        [model]
        components = e, vorticity, sheath_closure
        recalculate_metric = true
        Nnorm = 2e18
        Bnorm = mesh:Bxy
        Tnorm = 5

        [e]
        type = evolve_density, isothermal
        charge = -1
        AA = 1./1836
        temperature = 5

        [Ne]
        function = 1

        [vorticity]
        diamagnetic = true
        diamagnetic_polarisation = false
        average_atomic_mass = 1
        bndry_flux = false
        poloidal_flows = false
        phi_dissipation = false

        [sheath_closure]
        connection_length = 10
        """
    )
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)

    np.testing.assert_allclose(np.asarray(metrics.dx[:, mesh.ystart, 0]), 687.5432107181219, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(np.asarray(metrics.dy[:, mesh.ystart, 0]), 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(metrics.dz[:, mesh.ystart, 0]), 1.3020833333333333e-4, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.asarray(metrics.J[:, mesh.ystart, 0]), 1531.931512585073, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(metrics.g11[:, mesh.ystart, 0]), 5280331.858315176, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(np.asarray(metrics.g22[:, mesh.ystart, 0]), 4.2610958181668514e-7, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(np.asarray(metrics.g33[:, mesh.ystart, 0]), 1.893820363629712e-7, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(np.asarray(metrics.g23[:, mesh.ystart, 0]), 0.0, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(np.asarray(metrics.Bxy[:, mesh.ystart, 0]), 1.0, rtol=1e-12, atol=1e-12)
