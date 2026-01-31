from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from jaxdrb.operators.fd import d1_periodic


class TabulatedGeometry(eqx.Module):
    """Geometry loaded from an .npz file containing coefficient arrays along `l`.

    Required keys:
      - l, gxx, gxy, gyy
    Optional keys:
      - curv_x, curv_y
      - dpar_factor
    """

    l: jnp.ndarray
    dl: float = eqx.field(static=True)
    gxx: jnp.ndarray
    gxy: jnp.ndarray
    gyy: jnp.ndarray
    curv_x: jnp.ndarray
    curv_y: jnp.ndarray
    dpar_factor: jnp.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> "TabulatedGeometry":
        data = np.load(path)
        l = np.asarray(data["l"])
        if l.ndim != 1:
            raise ValueError("Expected 1D l array in geometry file.")
        dl = float(l[1] - l[0])
        if not np.allclose(np.diff(l), dl, rtol=1e-6, atol=1e-12):
            raise ValueError("TabulatedGeometry currently requires a uniform l grid.")

        def get(name: str, default: float = 0.0) -> np.ndarray:
            if name in data:
                return np.asarray(data[name])
            return np.full_like(l, default, dtype=float)

        return cls(
            l=jnp.asarray(l),
            dl=dl,
            gxx=jnp.asarray(data["gxx"]),
            gxy=jnp.asarray(data["gxy"]),
            gyy=jnp.asarray(data["gyy"]),
            curv_x=jnp.asarray(get("curv_x", 0.0)),
            curv_y=jnp.asarray(get("curv_y", 0.0)),
            dpar_factor=jnp.asarray(get("dpar_factor", 1.0)),
        )

    def kperp2(self, kx: float, ky: float) -> jnp.ndarray:
        return (kx**2) * self.gxx + 2.0 * kx * ky * self.gxy + (ky**2) * self.gyy

    def dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        return self.dpar_factor * d1_periodic(f, self.dl)

    def curvature(self, kx: float, ky: float, f: jnp.ndarray) -> jnp.ndarray:
        omega_d = kx * self.curv_x + ky * self.curv_y
        return 1j * omega_d * f
