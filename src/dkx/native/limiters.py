from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def minmod3_jax(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    same_sign = (a * b > 0.0) & (a * c > 0.0)
    magnitude = jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c)))
    return jnp.where(same_sign, jnp.sign(a) * magnitude, 0.0)


def monotonic_centered_edges_jax(
    center: jnp.ndarray,
    minus: jnp.ndarray,
    plus: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    slope = minmod3_jax(2.0 * (plus - center), 0.5 * (plus - minus), 2.0 * (center - minus))
    return center - 0.5 * slope, center + 0.5 * slope


def periodic_monotonic_centered_edges_jax(field: jnp.ndarray, *, axis: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    center = jnp.asarray(field, dtype=jnp.float64)
    minus = jnp.roll(center, shift=1, axis=axis)
    plus = jnp.roll(center, shift=-1, axis=axis)
    return monotonic_centered_edges_jax(center, minus, plus)


def minmod3_numpy(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    same_sign = (a * b > 0.0) & (a * c > 0.0)
    magnitude = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(same_sign, np.sign(a) * magnitude, 0.0)


def monotonic_centered_edges_numpy(
    center: np.ndarray,
    minus: np.ndarray,
    plus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    slope = minmod3_numpy(2.0 * (plus - center), 0.5 * (plus - minus), 2.0 * (center - minus))
    return center - 0.5 * slope, center + 0.5 * slope


def minmod3_scalar(a: float, b: float, c: float) -> float:
    if (a * b <= 0.0) or (a * c <= 0.0):
        return 0.0
    magnitude = min(abs(a), abs(b), abs(c))
    return float(np.sign(a) * magnitude)


def monotonic_centered_edges_scalar(center: float, minus: float, plus: float) -> tuple[float, float]:
    slope = minmod3_scalar(
        2.0 * (plus - center),
        0.5 * (plus - minus),
        2.0 * (center - minus),
    )
    return center - 0.5 * slope, center + 0.5 * slope
