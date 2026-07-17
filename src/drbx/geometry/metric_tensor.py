from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class MetricTensor3D:
    """Complete 3D metric payload used by non-axisymmetric geometry kernels."""

    dx: jnp.ndarray
    dy: jnp.ndarray
    dz: jnp.ndarray
    J: jnp.ndarray
    Bxy: jnp.ndarray
    g11: jnp.ndarray
    g22: jnp.ndarray
    g33: jnp.ndarray
    g12: jnp.ndarray
    g13: jnp.ndarray
    g23: jnp.ndarray
    g_11: jnp.ndarray
    g_22: jnp.ndarray
    g_33: jnp.ndarray
    g_12: jnp.ndarray
    g_13: jnp.ndarray
    g_23: jnp.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.J.shape)


def metric_inverse_residual(metric: MetricTensor3D) -> jnp.ndarray:
    """Return `max(abs(g^ik g_kj - delta^i_j))` over the grid."""

    contravariant = jnp.stack(
        [
            jnp.stack([metric.g11, metric.g12, metric.g13], axis=-1),
            jnp.stack([metric.g12, metric.g22, metric.g23], axis=-1),
            jnp.stack([metric.g13, metric.g23, metric.g33], axis=-1),
        ],
        axis=-2,
    )
    covariant = jnp.stack(
        [
            jnp.stack([metric.g_11, metric.g_12, metric.g_13], axis=-1),
            jnp.stack([metric.g_12, metric.g_22, metric.g_23], axis=-1),
            jnp.stack([metric.g_13, metric.g_23, metric.g_33], axis=-1),
        ],
        axis=-2,
    )
    product = jnp.einsum("...ik,...kj->...ij", contravariant, covariant)
    identity = jnp.eye(3, dtype=product.dtype)
    return jnp.max(jnp.abs(product - identity))


def build_metric_report(metric: MetricTensor3D) -> dict[str, object]:
    """Build finite/positive/inverse-consistency diagnostics for a metric payload."""

    report: dict[str, object] = {
        "shape": list(metric.shape),
        "inverse_residual_linf": float(metric_inverse_residual(metric)),
        "fields": {},
    }
    for name in (
        "J",
        "Bxy",
        "g11",
        "g22",
        "g33",
        "g12",
        "g13",
        "g23",
        "g_11",
        "g_22",
        "g_33",
        "g_12",
        "g_13",
        "g_23",
    ):
        values = np.asarray(getattr(metric, name), dtype=np.float64)
        report["fields"][name] = {
            "finite": bool(np.all(np.isfinite(values))),
            "minimum": float(np.nanmin(values)),
            "maximum": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
    report["passed"] = (
        bool(report["fields"]["J"]["finite"])
        and bool(report["fields"]["Bxy"]["finite"])
        and float(report["fields"]["J"]["minimum"]) > 0.0
        and float(report["fields"]["Bxy"]["minimum"]) > 0.0
        and float(report["inverse_residual_linf"]) < 1.0e-8
    )
    return report
