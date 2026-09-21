import numpy as np

from scripts import audit_hsx_owner_face_derivative_adaptive as adaptive


def _displacement(angle: np.ndarray) -> np.ndarray:
    return np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def test_perlmutter_pi_over_four_roundoff_enters_counterclockwise_sector() -> None:
    angle = np.asarray([0.7853981633974474], dtype=np.float64)
    np.testing.assert_array_equal(
        adaptive._planar_sector(_displacement(angle)), np.asarray([1], dtype=np.int8)
    )


def test_all_sector_boundaries_have_stable_roundoff_ties() -> None:
    epsilon = np.finfo(np.float64).eps
    boundary = np.arange(adaptive.PLANAR_SECTOR_COUNT, dtype=np.float64)
    within = 8.0 * epsilon
    radians_per_sector = 2.0 * np.pi / adaptive.PLANAR_SECTOR_COUNT

    below = (boundary - within) * radians_per_sector
    exact = boundary * radians_per_sector
    above = (boundary + within) * radians_per_sector
    expected = boundary.astype(np.int8)

    np.testing.assert_array_equal(adaptive._planar_sector(_displacement(below)), expected)
    np.testing.assert_array_equal(adaptive._planar_sector(_displacement(exact)), expected)
    np.testing.assert_array_equal(adaptive._planar_sector(_displacement(above)), expected)


def test_angles_outside_boundary_tolerance_keep_their_original_sectors() -> None:
    epsilon = np.finfo(np.float64).eps
    boundary = np.arange(adaptive.PLANAR_SECTOR_COUNT, dtype=np.float64)
    outside = 128.0 * epsilon
    radians_per_sector = 2.0 * np.pi / adaptive.PLANAR_SECTOR_COUNT

    below = (boundary - outside) * radians_per_sector
    above = (boundary + outside) * radians_per_sector

    np.testing.assert_array_equal(
        adaptive._planar_sector(_displacement(below)),
        ((boundary.astype(np.int16) - 1) % adaptive.PLANAR_SECTOR_COUNT).astype(np.int8),
    )
    np.testing.assert_array_equal(
        adaptive._planar_sector(_displacement(above)), boundary.astype(np.int8)
    )
