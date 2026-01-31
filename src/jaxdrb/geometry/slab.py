from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.operators.fd import d1_periodic


class SlabGeometry(eqx.Module):
    """Simple shear-slab / s-alpha-like geometry for benchmarking.

    Coordinate choice:
      - l ~ theta in [-pi, pi)
      - periodic boundary conditions
    """

    l: jnp.ndarray
    dl: float = eqx.field(static=True)
    shat: float = 0.0
    curvature0: float = 0.0
    dpar_factor: jnp.ndarray | None = None

    @classmethod
    def make(cls, nl: int = 64, length: float = 2 * jnp.pi, **kwargs) -> "SlabGeometry":
        l = jnp.linspace(-0.5 * length, 0.5 * length, nl, endpoint=False)
        dl = float(length / nl)
        return cls(l=l, dl=dl, **kwargs)

    def metric_components(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        theta = self.l
        gxx = jnp.ones_like(theta)
        gxy = self.shat * theta
        gyy = 1.0 + (self.shat * theta) ** 2
        return gxx, gxy, gyy

    def _metric(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Backwards-compatible alias (used in early examples).
        return self.metric_components()

    def kperp2(self, kx: float, ky: float) -> jnp.ndarray:
        gxx, gxy, gyy = self.metric_components()
        return (kx**2) * gxx + 2.0 * kx * ky * gxy + (ky**2) * gyy

    def dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        df = d1_periodic(f, self.dl)
        if self.dpar_factor is None:
            return df
        return self.dpar_factor * df

    def curvature(self, kx: float, ky: float, f: jnp.ndarray) -> jnp.ndarray:
        theta = self.l
        curv_x = jnp.zeros_like(theta)
        curv_y = self.curvature0 * jnp.cos(theta)
        omega_d = kx * curv_x + ky * curv_y
        return 1j * omega_d * f
