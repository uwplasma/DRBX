"""Field-provider adapter for the HSX geometry builder.

The builder historically read a MAKEGRID ``mgrid`` file through
:class:`~.Bfield_evaluator.ComponentSplineBFieldEvaluator`. This adapter lets
it consume any :class:`drbx.geometry.field_provider.MagneticFieldProvider`
(ESSOS coils, MAKEGRID filaments, later VMEX + vmec-extender) instead.

``R``/``phi``/``Z`` are not a sampling grid here: the builder uses only
their end points, as the box in which the scalar-potential coordinate is
fitted, and ``phi[0]`` as the period origin.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .Bfield_evaluator import BFieldEvaluator


@jax.tree_util.register_pytree_node_class
class JaxProviderBFieldEvaluator:
    """JAX view used by the compiled tracer (``evaluate_cartesian`` only)."""

    def __init__(self, provider: Any):
        self.provider = provider

    def tree_flatten(self):
        return (self.provider,), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)

    def evaluate_cartesian(self, points_xyz: Any) -> jax.Array:
        return self.provider.B(jnp.asarray(points_xyz, dtype=jnp.float64))

    __call__ = evaluate_cartesian


class ProviderBFieldEvaluator(BFieldEvaluator):
    """NumPy-facing :class:`BFieldEvaluator` backed by a field provider."""

    method = "provider"
    extrapolate = False

    def __init__(
        self,
        provider: Any,
        *,
        R_bounds: tuple[float, float],
        Z_bounds: tuple[float, float],
        phi0: float = 0.0,
    ):
        self.provider = provider
        self._nfp = int(provider.nfp)
        self._period = 2.0 * np.pi / self._nfp
        self._R = np.asarray(R_bounds, dtype=np.float64)
        self._Z = np.asarray(Z_bounds, dtype=np.float64)
        self._phi = np.array([phi0, phi0 + self._period])
        self._B = jax.jit(lambda p, x: p.B(x))

    @property
    def R(self) -> np.ndarray:
        return self._R

    @property
    def phi(self) -> np.ndarray:
        return self._phi

    @property
    def Z(self) -> np.ndarray:
        return self._Z

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def period(self) -> float:
        return self._period

    @property
    def currents(self) -> np.ndarray:
        return np.asarray(getattr(self.provider, "currents", np.zeros(0)), dtype=np.float64)

    def evaluate_cartesian(self, points_xyz: Any) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float64)
        return np.asarray(self._B(self.provider, jnp.asarray(points)), dtype=np.float64)

    def evaluate_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        points = np.asarray(points_rphiz, dtype=np.float64)
        R, phi, Z = points[..., 0], points[..., 1], points[..., 2]
        b = self.evaluate_cartesian(np.stack((R * np.cos(phi), R * np.sin(phi), Z), axis=-1))
        c, s = np.cos(phi), np.sin(phi)
        return np.stack((b[..., 0] * c + b[..., 1] * s, -b[..., 0] * s + b[..., 1] * c, b[..., 2]), axis=-1)

    def jax_evaluator(self) -> JaxProviderBFieldEvaluator:
        return JaxProviderBFieldEvaluator(self.provider)

    def content_identity(self) -> dict[str, object]:
        """Cache identity: provenance plus a hash of every parameter leaf."""

        digest = hashlib.sha256(b"drbx-field-provider-v1\0")
        for leaf in jax.tree_util.tree_leaves(self.provider):
            array = np.ascontiguousarray(np.asarray(leaf, dtype=np.float64))
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        provenance = json.loads(json.dumps(getattr(self.provider, "provenance", {}), default=str))
        return {"scheme": "field-provider-v1", "sha256": digest.hexdigest(), "provenance": provenance}


def provider_bfield_evaluator(
    provider: Any,
    *,
    R_bounds: tuple[float, float],
    Z_bounds: tuple[float, float],
) -> ProviderBFieldEvaluator:
    return ProviderBFieldEvaluator(provider, R_bounds=R_bounds, Z_bounds=Z_bounds)


__all__ = ["JaxProviderBFieldEvaluator", "ProviderBFieldEvaluator", "provider_bfield_evaluator"]
