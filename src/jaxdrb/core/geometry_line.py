from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.core.geometry import GeometryBase
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.geometry.base import Geometry


class LineGeometryAdapter(GeometryBase):
    """Geometry adapter for 1D field-line / flux-tube models.

    Uses a fixed (kx, ky) pair and delegates parallel derivatives and curvature
    to the underlying Geometry object.
    """

    geom: Geometry
    params: DRBSystemParams
    kx: float = 0.0
    ky: float = 0.0
    name: ClassVar[str] = "line"
    ndim: ClassVar[int] = 1

    def shape(self) -> tuple[int]:
        return (int(self.geom.l.size),)

    def _kperp2(self) -> jnp.ndarray:
        return jnp.asarray(self.geom.kperp2(float(self.kx), float(self.ky)))

    def ddx(self, f: jnp.ndarray) -> jnp.ndarray:
        return 1j * float(self.kx) * f

    def ddy(self, f: jnp.ndarray) -> jnp.ndarray:
        return 1j * float(self.ky) * f

    def laplacian(self, f: jnp.ndarray) -> jnp.ndarray:
        return -self._kperp2() * f

    def biharmonic(self, f: jnp.ndarray) -> jnp.ndarray:
        k2 = self._kperp2()
        return (k2**2) * f

    def inv_laplacian(self, f: jnp.ndarray, *, x0: jnp.ndarray | None = None) -> jnp.ndarray:
        _ = x0
        k2 = jnp.maximum(self._kperp2(), float(self.params.kperp2_min))
        return -f / k2

    def inv_div_n_grad(
        self, n_eff: jnp.ndarray, f: jnp.ndarray, *, x0: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        _ = x0
        k2 = jnp.maximum(self._kperp2(), float(self.params.kperp2_min))
        n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
        return -f / (k2 * n_eff)

    def bracket(self, phi: jnp.ndarray, f: jnp.ndarray, *, bc_phi=None, bc_f=None) -> jnp.ndarray:
        _ = (phi, f, bc_phi, bc_f)
        return jnp.zeros_like(f)

    def dpar(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        _ = bc_kind
        return self.geom.dpar(f)

    def d2par(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        return self.geom.dpar(self.geom.dpar(f))

    def curvature(self, f: jnp.ndarray) -> jnp.ndarray:
        return self.geom.curvature(float(self.kx), float(self.ky), f)

    def kappa_profile(self) -> float:
        return 0.0

    def sheath_mask_sign(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        nl = int(self.geom.l.size)
        mask = jnp.zeros((nl,), dtype=jnp.float64).at[0].set(1.0).at[-1].set(1.0)
        sign = jnp.zeros_like(mask).at[0].set(-1.0).at[-1].set(1.0)
        return mask, sign
