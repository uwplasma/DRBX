"""Compact periodic Hermite interpolation of raw toroidal sample planes.

C2 uses endpoint jets through order two from centered five-point formulas;
C3 uses jets through order three from centered seven-point formulas. Adjacent
intervals share the same nodal jets, including the periodic seam. The union
of the two endpoint supports contains six/eight planes, respectively. No
inverse filter or solve couples source planes during field preparation.

Weights act on dimensionless grid coordinates, so physical spacing factors
in the jets cancel. These are interpolants, not divergence-free projections.
"""
from functools import lru_cache
from math import factorial

import numpy as np


@lru_cache(maxsize=2)
def hermite_coefficients(continuity: int) -> np.ndarray:
    """Power coefficients [power, donor] on the unit interval (host setup)."""
    if continuity not in (2, 3):
        raise ValueError("continuity must be 2 or 3")
    m = continuity
    nodes = np.arange(-m, m + 1, dtype=float)
    vandermonde = nodes[None, :] ** np.arange(2 * m + 1)[:, None]
    jets = []
    for derivative in range(m + 1):
        target = np.zeros(2 * m + 1)
        target[derivative] = factorial(derivative)
        jets.append(np.linalg.solve(vandermonde, target))
    size = 2 * m + 2
    constraints = np.zeros((size, size))
    rhs = np.zeros_like(constraints)
    for side in (0, 1):
        for derivative in range(m + 1):
            row = side * (m + 1) + derivative
            for power in range(derivative, size):
                constraints[row, power] = (
                    factorial(power) / factorial(power - derivative)
                    * side ** (power - derivative)
                )
            rhs[row, side:side + 2 * m + 1] = jets[derivative]
    result = np.linalg.solve(constraints, rhs)
    result.setflags(write=False)
    return result


def compact_weights(fraction, continuity, xp=np):
    """Evaluate the fixed donor weights with NumPy or JAX primitives."""
    coefficients = xp.asarray(hermite_coefficients(continuity))
    weights = xp.zeros(fraction.shape + (coefficients.shape[1],))
    for row in coefficients[::-1]:
        weights = weights * fraction[..., None] + row
    return weights


def interpolate_compact(coordinates, coefficients, continuity, xp=np):
    """Tensor evaluation: compact phi, unchanged cubic B-splines in Z/R."""
    base = xp.floor(coordinates).astype(xp.int32)
    t = coordinates - base
    phi_i = (base[:, 0, None] + xp.arange(-continuity, continuity + 2)) % coefficients[0].shape[0]
    z_i = (base[:, 1, None] + xp.arange(-1, 3)) % coefficients[0].shape[1]
    r_i = (base[:, 2, None] + xp.arange(-1, 3)) % coefficients[0].shape[2]
    weights = xp.stack(((1-t)**3/6, (3*t**3-6*t**2+4)/6,
                        (-3*t**3+3*t**2+3*t+1)/6, t**3/6), axis=-1)
    wp = compact_weights(t[:, 0], continuity, xp)
    result = []
    for component in coefficients:
        values = component[phi_i[:, :, None, None], z_i[:, None, :, None], r_i[:, None, None, :]]
        result.append(xp.einsum('nabc,na,nb,nc->n', values, wp, weights[:, 1], weights[:, 2]))
    return xp.stack(result, axis=-1)
