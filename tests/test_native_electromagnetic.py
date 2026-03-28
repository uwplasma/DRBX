from __future__ import annotations

import numpy as np

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.electromagnetic import (
    apply_canonical_momentum_correction,
    compute_alpha_em,
    compute_apar_flutter,
    compute_beta_em,
    compute_parallel_current_density,
    extract_charged_species_metadata,
)


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
