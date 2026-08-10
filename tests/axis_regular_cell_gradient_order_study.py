"""Small, standalone order study for an axis-regular cell gradient fit.

The fit is deliberately expressed as the same three objects used by the
cell-gradient path: observation values, precomputed coefficients, and a
precomputed coefficient-to-gradient target.  No DRBX implementation is
imported, so this file can also be used as a numerical design note.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEGREES = tuple(range(2, 9))
MESH_SIZE = 32
RADIAL_RINGS = (np.arange(MESH_SIZE, dtype=float) + 0.5) / MESH_SIZE
TARGET_RING_INDICES = (0, 1, 2)
TRANSITION_RING_INDEX = 3
OBSERVATION_RING_INDICES = (0, 1, 2, 3, 4, 5)  # 192 observations, 45 unknowns at p=8


@dataclass(frozen=True)
class Fit:
    degree: int
    observation_ring_count: int
    condition: float
    coefficients: np.ndarray
    gradient_target: np.ndarray


def _powers(degree: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, j) for total in range(degree + 1) for i in range(total + 1) for j in (total - i,))


def _design(x: np.ndarray, y: np.ndarray, powers: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.column_stack([(x**i) * (y**j) for i, j in powers])


def _gradient_target(x: np.ndarray, y: np.ndarray, powers: tuple[tuple[int, int], ...]) -> np.ndarray:
    dx = [i * x ** (i - 1) * y**j if i else np.zeros_like(x) for i, j in powers]
    dy = [j * x**i * y ** (j - 1) if j else np.zeros_like(x) for i, j in powers]
    return np.stack((np.asarray(dx).T, np.asarray(dy).T))


def _mesh(radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, MESH_SIZE, endpoint=False)
    radius, angle = np.meshgrid(radii, theta, indexing="ij")
    return (radius * np.cos(angle)).ravel(), (radius * np.sin(angle)).ravel()


def _observations() -> tuple[np.ndarray, np.ndarray]:
    return _mesh(RADIAL_RINGS[list(OBSERVATION_RING_INDICES)])


def fit_order(degree: int, values: np.ndarray, target_x: np.ndarray, target_y: np.ndarray) -> Fit:
    powers = _powers(degree)
    ox, oy = _observations()
    matrix = _design(ox, oy, powers)
    column_norm = np.linalg.norm(matrix, axis=0)
    normalized = matrix / column_norm
    condition = float(np.linalg.cond(normalized))
    scaled_coefficients = np.linalg.lstsq(normalized, values, rcond=None)[0]
    coefficients = scaled_coefficients / column_norm
    target = _gradient_target(target_x, target_y, powers)
    return Fit(degree, len(OBSERVATION_RING_INDICES), condition, coefficients, target)


def _regular(m: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = x + 1j * y
    value = np.real(z**m)
    if m == 0:
        return value, np.zeros((2, x.size))
    grad = np.stack((m * np.real(z ** (m - 1)), -m * np.imag(z ** (m - 1))))
    return value, grad


def _nonpolynomial(x: np.ndarray, y: np.ndarray, z_amplitude: float = 1.17) -> tuple[np.ndarray, np.ndarray]:
    value = np.exp(0.72 * x - 0.41 * y) * z_amplitude
    return value, np.stack((0.72 * value, -0.41 * value))


def _nonregular(m: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.hypot(x, y)
    angle = np.arctan2(y, x)
    value = np.cos(m * angle)
    grad = np.stack((m * np.sin(m * angle) * y / radius**2, -m * np.sin(m * angle) * x / radius**2))
    return value, grad


def _gradient_error(fit: Fit, truth: np.ndarray) -> float:
    prediction = np.einsum("tck,k->tc", fit.gradient_target, fit.coefficients)
    scale = max(float(np.linalg.norm(truth)), 1.0e-14)
    return float(np.linalg.norm(prediction - truth) / scale)


def run_study() -> dict[str, object]:
    target_x, target_y = _mesh(RADIAL_RINGS[list(TARGET_RING_INDICES)])
    transition_x, transition_y = _mesh(np.asarray([RADIAL_RINGS[TRANSITION_RING_INDEX]]))
    ox, oy = _observations()
    result: dict[str, object] = {
        "observation_ring_count": len(OBSERVATION_RING_INDICES),
        "orders": {}, "regular": {}, "nonpolynomial": {}, "nonregular": {},
    }

    zero_values = np.zeros_like(ox)
    result["orders"] = {p: fit_order(p, zero_values, target_x, target_y).condition for p in DEGREES}

    regular: dict[int, dict[int, float]] = {}
    for m in range(1, 13):
        values, truth = _regular(m, target_x, target_y)
        regular[m] = {}
        for p in DEGREES:
            fit = fit_order(p, _regular(m, ox, oy)[0], target_x, target_y)
            regular[m][p] = _gradient_error(fit, truth)
    result["regular"] = regular

    nonpoly_errors: dict[int, float] = {}
    for p in DEGREES:
        fit = fit_order(p, _nonpolynomial(ox, oy)[0], target_x, target_y)
        nonpoly_errors[p] = _gradient_error(fit, _nonpolynomial(target_x, target_y)[1])
    result["nonpolynomial"] = nonpoly_errors

    contamination: dict[int, dict[str, float]] = {}
    m = 8
    desired_values, _ = _regular(3, ox, oy)
    _, desired_target_gradient = _regular(3, target_x, target_y)
    contaminated_values = desired_values + 0.35 * _nonregular(m, ox, oy)[0]
    for p in DEGREES:
        fit = fit_order(p, contaminated_values, target_x, target_y)
        transition_fit = fit_order(p, contaminated_values, transition_x, transition_y)
        contamination[p] = {
            "core_error": _gradient_error(fit, desired_target_gradient),
            "transition_error": _gradient_error(
                transition_fit,
                _regular(3, transition_x, transition_y)[1],
            ),
            "contamination_fit": _gradient_error(
                fit_order(p, 0.35 * _nonregular(m, ox, oy)[0], target_x, target_y),
                0.35 * _nonregular(m, target_x, target_y)[1],
            ),
        }
    result["nonregular"] = contamination
    return result


def test_order_study_conditioning_and_regular_mode_cutoff() -> None:
    report = run_study()
    condition = report["orders"]
    assert all(np.isfinite(condition[p]) and condition[p] > 1.0 for p in DEGREES)
    assert condition[8] > condition[2]
    regular = report["regular"]
    assert regular[8][2] > 0.99  # an unrepresented regular mode is rejected
    assert regular[8][8] < 1.0e-8  # p=m reproduces its regular harmonic gradient


def test_order_study_shows_p_convergence_and_unknown_mode_risk() -> None:
    report = run_study()
    errors = report["nonpolynomial"]
    assert errors[8] < errors[2]
    contamination = report["nonregular"]
    assert contamination[8]["contamination_fit"] < contamination[2]["contamination_fit"]
    # p=3 is the clean regular-signal choice here; p=8 (the contaminant's
    # angular order) gives the fit enough freedom to retain the rejected mode.
    assert contamination[8]["core_error"] > contamination[3]["core_error"]
    assert contamination[8]["transition_error"] > contamination[3]["transition_error"]


def _print_table(report: dict[str, object]) -> None:
    print(f"target rings={TARGET_RING_INDICES}, transition ring={TRANSITION_RING_INDEX}, "
          f"observation rings={OBSERVATION_RING_INDICES}")
    print("p  obs-rings  cond(A_norm)  exp-gradient-error  contam-core  contam-transition")
    for p in DEGREES:
        print(f"{p:1d}  {report['observation_ring_count']:9d}  {report['orders'][p]:11.3e}  {report['nonpolynomial'][p]:17.3e}  "
              f"{report['nonregular'][p]['core_error']:11.3e}  {report['nonregular'][p]['transition_error']:17.3e}")
    print("\nregular-mode relative gradient errors (rows m, columns p):")
    for m in range(1, 13):
        print(m, " ".join(f"{report['regular'][m][p]:.2e}" for p in DEGREES))


if __name__ == "__main__":
    _print_table(run_study())
