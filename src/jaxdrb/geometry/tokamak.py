from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.operators.fd import d1_periodic


class CircularTokamakGeometry(eqx.Module):
    """Large-aspect-ratio circular tokamak flux-tube geometry (field-line model).

    This is a lightweight analytic geometry intended for qualitative linear studies and for
    generating tabulated geometry files.

    Conventions:
      - Parallel coordinate `l` is taken to be the poloidal angle `theta` in [-pi, pi) by default.
      - Perpendicular Fourier dependence is exp(i kx psi + i ky alpha).
      - k_perp^2(l) is built from simple field-aligned metric coefficients.
      - ∇_|| is approximated as (1 / (q R0)) d/dtheta.

    Notes:
      - For v1 we keep the metric minimal: g_xx = 1, g_xy = s_hat * theta, g_yy = 1 + g_xy^2.
      - The curvature operator is implemented as C(f) = i (kx curv_x + ky curv_y) f with
        curv_y(theta) ~ curvature0 * cos(theta) * B(theta).
    """

    l: jnp.ndarray
    dl: float = eqx.field(static=True)

    shat: float = 0.0
    q: float = 1.4
    R0: float = 1.0
    epsilon: float = 0.18  # inverse aspect ratio r/R0

    curvature0: float = 0.18
    b_min: float = 0.05

    @classmethod
    def make(
        cls,
        *,
        nl: int = 64,
        length: float = float(2 * jnp.pi),
        shat: float = 0.0,
        q: float = 1.4,
        R0: float = 1.0,
        epsilon: float = 0.18,
        curvature0: float | None = None,
    ) -> "CircularTokamakGeometry":
        l = jnp.linspace(-0.5 * length, 0.5 * length, nl, endpoint=False)
        dl = float(length / nl)
        if curvature0 is None:
            curvature0 = float(epsilon)
        return cls(
            l=l,
            dl=dl,
            shat=shat,
            q=q,
            R0=R0,
            epsilon=epsilon,
            curvature0=curvature0,
        )

    def B(self) -> jnp.ndarray:
        theta = self.l
        return 1.0 / jnp.maximum(1.0 + self.epsilon * jnp.cos(theta), self.b_min)

    def metric_components(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        theta = self.l
        gxx = jnp.ones_like(theta)
        gxy = self.shat * theta
        gyy = 1.0 + gxy**2
        return gxx, gxy, gyy

    def kperp2(self, kx: float, ky: float) -> jnp.ndarray:
        gxx, gxy, gyy = self.metric_components()
        return (kx**2) * gxx + 2.0 * kx * ky * gxy + (ky**2) * gyy

    def dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        return d1_periodic(f, self.dl) / (self.q * self.R0)

    def curvature(self, kx: float, ky: float, f: jnp.ndarray) -> jnp.ndarray:
        theta = self.l
        curv_x = jnp.zeros_like(theta)
        curv_y = self.curvature0 * jnp.cos(theta) * self.B()
        omega_d = kx * curv_x + ky * curv_y
        return 1j * omega_d * f

    def coefficients(self) -> dict[str, jnp.ndarray]:
        gxx, gxy, gyy = self.metric_components()
        theta = self.l
        curv_x = jnp.zeros_like(theta)
        curv_y = self.curvature0 * jnp.cos(theta) * self.B()
        dpar_factor = jnp.ones_like(theta) / (self.q * self.R0)
        return {
            "l": self.l,
            "gxx": gxx,
            "gxy": gxy,
            "gyy": gyy,
            "curv_x": curv_x,
            "curv_y": curv_y,
            "dpar_factor": dpar_factor,
            "B": self.B(),
        }


class SAlphaGeometry(eqx.Module):
    """s-alpha tokamak flux-tube geometry (ballooning representation).

    This is a standard analytic model used in many gyrokinetic/ballooning benchmarks. In this
    v1 implementation, the primary effect of `alpha` is to modify the local radial wavevector
    along the field line through the metric cross term:

      g_xy(theta) = s_hat * theta - alpha * sin(theta)

    which yields a ballooning-like effective k_perp^2(theta).
    """

    l: jnp.ndarray
    dl: float = eqx.field(static=True)

    shat: float = 0.796
    alpha: float = 0.0
    q: float = 1.4
    R0: float = 1.0
    epsilon: float = 0.18
    curvature0: float = 0.18
    b_min: float = 0.05

    @classmethod
    def make(
        cls,
        *,
        nl: int = 64,
        length: float = float(2 * jnp.pi),
        shat: float = 0.796,
        alpha: float = 0.0,
        q: float = 1.4,
        R0: float = 1.0,
        epsilon: float = 0.18,
        curvature0: float | None = None,
    ) -> "SAlphaGeometry":
        l = jnp.linspace(-0.5 * length, 0.5 * length, nl, endpoint=False)
        dl = float(length / nl)
        if curvature0 is None:
            curvature0 = float(epsilon)
        return cls(
            l=l,
            dl=dl,
            shat=shat,
            alpha=alpha,
            q=q,
            R0=R0,
            epsilon=epsilon,
            curvature0=curvature0,
        )

    @classmethod
    def cyclone_base_case(cls, *, nl: int = 64) -> "SAlphaGeometry":
        # Cyclone Base Case (CBC) is typically alpha=0 (electrostatic, zero-beta).
        # We keep only the geometry parameters here.
        return cls.make(nl=nl, shat=0.796, alpha=0.0, q=1.4, R0=1.0, epsilon=0.18, curvature0=0.18)

    def B(self) -> jnp.ndarray:
        theta = self.l
        return 1.0 / jnp.maximum(1.0 + self.epsilon * jnp.cos(theta), self.b_min)

    def metric_components(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        theta = self.l
        gxx = jnp.ones_like(theta)
        gxy = self.shat * theta - self.alpha * jnp.sin(theta)
        gyy = 1.0 + gxy**2
        return gxx, gxy, gyy

    def kperp2(self, kx: float, ky: float) -> jnp.ndarray:
        gxx, gxy, gyy = self.metric_components()
        return (kx**2) * gxx + 2.0 * kx * ky * gxy + (ky**2) * gyy

    def dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        return d1_periodic(f, self.dl) / (self.q * self.R0)

    def curvature(self, kx: float, ky: float, f: jnp.ndarray) -> jnp.ndarray:
        theta = self.l
        curv_x = jnp.zeros_like(theta)
        curv_y = self.curvature0 * jnp.cos(theta) * self.B()
        omega_d = kx * curv_x + ky * curv_y
        return 1j * omega_d * f

    def coefficients(self) -> dict[str, jnp.ndarray]:
        gxx, gxy, gyy = self.metric_components()
        theta = self.l
        curv_x = jnp.zeros_like(theta)
        curv_y = self.curvature0 * jnp.cos(theta) * self.B()
        dpar_factor = jnp.ones_like(theta) / (self.q * self.R0)
        return {
            "l": self.l,
            "gxx": gxx,
            "gxy": gxy,
            "gyy": gyy,
            "curv_x": curv_x,
            "curv_y": curv_y,
            "dpar_factor": dpar_factor,
            "B": self.B(),
        }

