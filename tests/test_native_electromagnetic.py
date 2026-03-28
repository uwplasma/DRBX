from __future__ import annotations

import numpy as np

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.electromagnetic import (
    ALFVEN_WAVE_DDT_NVE_DDY_COEF,
    ALFVEN_WAVE_DDT_NVE_DDZ_COEF,
    ChargedSpeciesMetadata,
    apply_canonical_momentum_correction,
    compute_alfven_wave_ddt_nve_core,
    compute_alpha_em,
    compute_apar_flutter,
    compute_beta_em,
    compute_parallel_current_density,
    invert_slab_neumann_apar_to_current_density,
    solve_slab_neumann_apar,
    extract_charged_species_metadata,
)
from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.metrics import StructuredMetrics


def test_compute_beta_em_matches_reference_formula() -> None:
    expected = 4.0e-7 * np.pi * 1.602176634e-19 * 100.0 * 1.0e19 / (0.2 * 0.2)
    assert compute_beta_em(Nnorm=1.0e19, Tnorm=100.0, Bnorm=0.2) == expected


def test_extract_charged_species_metadata_reads_alfven_input() -> None:
    config = load_bout_input("/Users/rogerio/local/hermes-3/tests/integrated/alfven-wave/data/BOUT.inp")
    metadata = extract_charged_species_metadata(config)

    assert tuple(species.section for species in metadata) == ("i", "e")
    assert metadata[0].charge == 1.0
    assert metadata[0].atomic_mass == 1.0
    assert metadata[1].charge == -1.0
    assert metadata[1].atomic_mass == 1.0 / 1836.0


def test_compute_parallel_current_density_matches_species_sum() -> None:
    config = load_bout_input("/Users/rogerio/local/hermes-3/tests/integrated/alfven-wave/data/BOUT.inp")
    metadata = extract_charged_species_metadata(config)
    momenta = {
        "NVi": np.full((2, 3), 0.25, dtype=np.float64),
        "NVe": np.full((2, 3), -2.0 / 1836.0, dtype=np.float64),
    }

    current = compute_parallel_current_density(momenta, metadata)

    expected = np.full((2, 3), 0.25 + 2.0, dtype=np.float64)
    np.testing.assert_allclose(current, expected)


def test_compute_alpha_em_uses_density_floor() -> None:
    config = load_bout_input("/Users/rogerio/local/hermes-3/tests/integrated/alfven-wave/data/BOUT.inp")
    metadata = extract_charged_species_metadata(config)
    densities = {
        "Ni": np.zeros((2, 3), dtype=np.float64),
        "Ne": np.zeros((2, 3), dtype=np.float64),
    }

    alpha = compute_alpha_em(densities, metadata, density_floor=1.0e-5)

    expected = np.full((2, 3), 1.0e-5 * (1.0 + 1836.0), dtype=np.float64)
    np.testing.assert_allclose(alpha, expected)


def test_apply_canonical_momentum_correction_matches_reference_formula() -> None:
    density = np.full((2, 3), 4.0, dtype=np.float64)
    momentum = np.full((2, 3), 7.0, dtype=np.float64)
    velocity = np.full((2, 3), 5.0, dtype=np.float64)
    apar = np.full((2, 3), 0.25, dtype=np.float64)

    corrected_momentum, corrected_velocity = apply_canonical_momentum_correction(
        density=density,
        momentum=momentum,
        velocity=velocity,
        apar=apar,
        charge=-1.0,
        atomic_mass=1.0 / 1836.0,
    )

    np.testing.assert_allclose(corrected_momentum, momentum + density * apar)
    np.testing.assert_allclose(corrected_velocity, velocity + 1836.0 * apar)


def test_compute_apar_flutter_returns_zero_for_constant_field() -> None:
    apar = np.full((5, 36, 27), 3.0, dtype=np.float64)
    flutter = compute_apar_flutter(apar, axis=1)
    np.testing.assert_allclose(flutter, 0.0)


def test_solve_slab_neumann_apar_matches_single_mode_analytic_solution() -> None:
    mesh = StructuredMesh(
        nx=5,
        ny=4,
        nz=8,
        mxg=2,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=True,
        jyseps1_1=-1,
        jyseps2_1=3,
        jyseps1_2=3,
        jyseps2_2=3,
        ny_inner=4,
        has_lower_y_target=False,
        has_upper_y_target=False,
        x=np.arange(5, dtype=np.float64),
        y=np.arange(6, dtype=np.float64),
        z=np.arange(8, dtype=np.float64),
    )
    ones = np.ones((5, 6, 8), dtype=np.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=0.25 * ones,
        J=ones,
        g11=ones,
        g33=2.0 * ones,
        g22=ones,
        g_22=ones,
        g23=np.zeros_like(ones),
        Bxy=ones,
    )
    z = np.arange(mesh.nz, dtype=np.float64)
    mode = np.sin((2.0 * np.pi * z) / float(mesh.nz))
    current = np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    current[mesh.xstart, mesh.ystart : mesh.yend + 1, :] = mode[None, :]
    density = np.zeros_like(current)
    density[:, :, :] = 1.0
    species = (ChargedSpeciesMetadata(section="e", charge=-1.0, atomic_mass=1.0 / 1836.0),)
    beta_em = 0.2

    apar = solve_slab_neumann_apar(
        current,
        density_fields={"Ne": density},
        species_metadata=species,
        mesh=mesh,
        metrics=metrics,
        beta_em=beta_em,
    )

    k = 2.0 * np.pi / (float(mesh.nz) * 0.25)
    alpha = 1836.0
    expected_interior = ((-beta_em) / (-(k * k) * 2.0 - beta_em * alpha)) * mode
    expected_rows = np.broadcast_to(expected_interior[None, :], (mesh.ny, mesh.nz))
    np.testing.assert_allclose(apar[mesh.xstart, mesh.ystart : mesh.yend + 1, :], expected_rows, atol=1.0e-18)
    np.testing.assert_allclose(apar[mesh.xstart - 1, mesh.ystart : mesh.yend + 1, :], expected_rows, atol=1.0e-18)
    np.testing.assert_allclose(apar[mesh.xend + 1, mesh.ystart : mesh.yend + 1, :], expected_rows, atol=1.0e-18)
    np.testing.assert_allclose(apar[:, mesh.ystart - 1, :], apar[:, mesh.yend, :], atol=1.0e-18)
    np.testing.assert_allclose(apar[:, mesh.yend + 1, :], apar[:, mesh.ystart, :], atol=1.0e-18)


def test_invert_slab_neumann_apar_to_current_density_recovers_current_on_physical_planes() -> None:
    mesh = StructuredMesh(
        nx=5,
        ny=4,
        nz=8,
        mxg=2,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=True,
        jyseps1_1=-1,
        jyseps2_1=3,
        jyseps1_2=3,
        jyseps2_2=3,
        ny_inner=4,
        has_lower_y_target=False,
        has_upper_y_target=False,
        x=np.arange(5, dtype=np.float64),
        y=np.arange(6, dtype=np.float64),
        z=np.arange(8, dtype=np.float64),
    )
    ones = np.ones((5, 6, 8), dtype=np.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=0.25 * ones,
        J=ones,
        g11=ones,
        g33=2.0 * ones,
        g22=ones,
        g_22=ones,
        g23=np.zeros_like(ones),
        Bxy=ones,
    )
    z = np.arange(mesh.nz, dtype=np.float64)
    mode = np.sin((2.0 * np.pi * z) / float(mesh.nz))
    current = np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    current[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :] = mode[None, None, :]
    density = np.ones_like(current)
    species = (ChargedSpeciesMetadata(section="e", charge=-1.0, atomic_mass=1.0 / 1836.0),)
    beta_em = 0.2

    apar = solve_slab_neumann_apar(
        current,
        density_fields={"Ne": density},
        species_metadata=species,
        mesh=mesh,
        metrics=metrics,
        beta_em=beta_em,
    )
    recovered = invert_slab_neumann_apar_to_current_density(
        apar,
        density_fields={"Ne": density},
        species_metadata=species,
        mesh=mesh,
        metrics=metrics,
        beta_em=beta_em,
    )

    np.testing.assert_allclose(
        recovered[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :],
        current[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :],
        atol=1.0e-15,
    )
    np.testing.assert_allclose(recovered[:, mesh.ystart - 1, :], recovered[:, mesh.yend, :], atol=1.0e-18)
    np.testing.assert_allclose(recovered[:, mesh.yend + 1, :], recovered[:, mesh.ystart, :], atol=1.0e-18)


def test_compute_alfven_wave_ddt_nve_core_matches_periodic_vorticity_derivatives() -> None:
    mesh = StructuredMesh(
        nx=5,
        ny=4,
        nz=8,
        mxg=2,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=True,
        jyseps1_1=-1,
        jyseps2_1=3,
        jyseps1_2=3,
        jyseps2_2=3,
        ny_inner=4,
        has_lower_y_target=False,
        has_upper_y_target=False,
        x=np.arange(5, dtype=np.float64),
        y=np.arange(6, dtype=np.float64),
        z=np.arange(8, dtype=np.float64),
    )
    y_index = np.arange(mesh.ny, dtype=np.float64)[:, None]
    z_index = np.arange(mesh.nz, dtype=np.float64)[None, :]
    core = 1.0e-3 * np.sin(z_index - y_index)
    vorticity = np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    vorticity[mesh.xstart, mesh.ystart : mesh.yend + 1, :] = core

    ddt_nve = compute_alfven_wave_ddt_nve_core(vorticity, mesh=mesh)

    ddy = 0.5 * (np.roll(core, -1, axis=0) - np.roll(core, 1, axis=0))
    ddz = 0.5 * (np.roll(core, -1, axis=1) - np.roll(core, 1, axis=1))
    expected = ALFVEN_WAVE_DDT_NVE_DDY_COEF * ddy + ALFVEN_WAVE_DDT_NVE_DDZ_COEF * ddz
    np.testing.assert_allclose(
        ddt_nve[mesh.xstart, mesh.ystart : mesh.yend + 1, :],
        expected,
        atol=1.0e-18,
    )
    np.testing.assert_allclose(ddt_nve[: mesh.xstart, :, :], 0.0, atol=1.0e-18)
    np.testing.assert_allclose(ddt_nve[mesh.xend + 1 :, :, :], 0.0, atol=1.0e-18)
