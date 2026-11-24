"""Low-pass the solved map in eta and fit its filtered positions."""

import numpy as np

from .._metric_contract import BASE_MESH_SHAPE
from .eta_filtered_field import EtaFilterSpec, _harmonic_coefficients, _synthesize
from .metric_evaluator import MetricEvaluator, validate_fitted_map


def filter_map_samples(position, eta, spec):
    """Apply the column filter to R, Z and phi minus eta."""
    x, y, z = np.moveaxis(position, -1, 0)
    delta = (np.arctan2(y, x) - eta + np.pi) % (2 * np.pi) - np.pi
    samples = np.stack((np.hypot(x, y), z, delta), axis=-1)
    coef = _harmonic_coefficients(samples, spec)
    coef = np.repeat(coef.reshape(-1, spec.n_coefficients, 3), eta.size, axis=0)
    tiled_eta = np.tile(eta, position[..., 0].size // eta.size)
    channels = _synthesize(coef, tiled_eta, spec).reshape(position.shape)
    R, Z, delta = np.moveaxis(channels, -1, 0)
    phi = eta + delta
    return np.stack((R * np.cos(phi), R * np.sin(phi), Z), axis=-1)


def filter_solved_map(metric, field_spec):
    nfp, cutoff = field_spec.nfp, field_spec.max_harmonic_per_period
    if metric.nfp != nfp:
        raise ValueError("eta-filtered map periods must match the solved map")
    # Keep both signs of the highest Fourier mode, including for large cutoffs.
    count = max(BASE_MESH_SHAPE[-1], 2 * cutoff + 1)
    spec = EtaFilterSpec(max_harmonic_per_period=cutoff, nfp=nfp, samples_per_period=count)
    u, theta = metric.u, metric.v
    eta = np.arange(count) * (spec.period / count)
    points = np.stack(np.meshgrid(u, theta, eta, indexing="ij"), axis=-1)
    positions = filter_map_samples(metric.position(points), eta, spec)
    fitted = MetricEvaluator(u, theta, eta, positions, nfp=nfp, topology="toroidal",
                             radial_degree=metric.radial_degree, poloidal_modes=metric.poloidal_modes,
                             toroidal_modes=cutoff)
    validate_fitted_map(fitted)
    error = np.concatenate([
        np.linalg.norm(fitted.position(q) - target, axis=-1) for q, target in zip(points, positions)
    ])
    fitted._filtered_map_cutoff = cutoff
    return fitted, dict(rms=float(np.sqrt(np.mean(error**2))), max=float(np.max(error)))
