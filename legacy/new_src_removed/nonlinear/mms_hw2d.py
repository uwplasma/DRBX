from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .hw2d import HW2DState


class MMSHW2D(eqx.Module):
    """Manufactured solution for the HW2D system on a periodic domain.

    We choose analytic fields (resolved Fourier modes) and derive forcing terms
    S_n, S_ω so that the continuum PDE is satisfied exactly.

    The PDE solved by HW2D (without forcing) is:

      ∂t n + [φ,n] = -κ ∂y φ + α(φ-n) + Dn ∇² n
      ∂t ω + [φ,ω] = -κ ∂y n + α(φ-n) + Dω ∇² ω
      ω = ∇² φ

    The manufactured forcing is then:

      S_n = ∂t n - RHS_n
      S_ω = ∂t ω - RHS_ω

    where RHS_* are the continuum spatial operators evaluated at (x,y,t).
    """

    kx: int = 3
    ky: int = 2
    sigma: float = 0.1  # exponential-in-time factor
    Aphi: float = 0.3
    An: float = 0.2
    phase: float = 0.1

    def phi(self, x: jnp.ndarray, y: jnp.ndarray, t: float, *, Lx: float, Ly: float) -> jnp.ndarray:
        kx = 2.0 * jnp.pi * self.kx / Lx
        ky = 2.0 * jnp.pi * self.ky / Ly
        return self.Aphi * jnp.sin(kx * x) * jnp.cos(ky * y + self.phase) * jnp.exp(self.sigma * t)

    def n(self, x: jnp.ndarray, y: jnp.ndarray, t: float, *, Lx: float, Ly: float) -> jnp.ndarray:
        kx = 2.0 * jnp.pi * self.kx / Lx
        ky = 2.0 * jnp.pi * self.ky / Ly
        return self.An * jnp.cos(kx * x) * jnp.sin(ky * y - self.phase) * jnp.exp(self.sigma * t)

    def omega(
        self, x: jnp.ndarray, y: jnp.ndarray, t: float, *, Lx: float, Ly: float
    ) -> jnp.ndarray:
        kx = 2.0 * jnp.pi * self.kx / Lx
        ky = 2.0 * jnp.pi * self.ky / Ly
        k2 = kx**2 + ky**2
        return -k2 * self.phi(x, y, t, Lx=Lx, Ly=Ly)

    def state(self, x: jnp.ndarray, y: jnp.ndarray, t: float, *, Lx: float, Ly: float) -> HW2DState:
        return HW2DState(
            n=self.n(x, y, t, Lx=Lx, Ly=Ly),
            omega=self.omega(x, y, t, Lx=Lx, Ly=Ly),
            N=None,
        )

    def forcing(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        t: float,
        *,
        Lx: float,
        Ly: float,
        kappa: float,
        alpha: float,
        Dn: float,
        DOmega: float,
    ) -> HW2DState:
        """Continuum forcing for the manufactured fields."""

        kx = 2.0 * jnp.pi * self.kx / Lx
        ky = 2.0 * jnp.pi * self.ky / Ly
        k2 = kx**2 + ky**2

        phi = self.phi(x, y, t, Lx=Lx, Ly=Ly)
        n = self.n(x, y, t, Lx=Lx, Ly=Ly)
        omega = -k2 * phi

        # Continuum derivatives.
        dphi_dx = (
            self.Aphi
            * kx
            * jnp.cos(kx * x)
            * jnp.cos(ky * y + self.phase)
            * jnp.exp(self.sigma * t)
        )
        dphi_dy = (
            -self.Aphi
            * ky
            * jnp.sin(kx * x)
            * jnp.sin(ky * y + self.phase)
            * jnp.exp(self.sigma * t)
        )
        dn_dx = (
            -self.An * kx * jnp.sin(kx * x) * jnp.sin(ky * y - self.phase) * jnp.exp(self.sigma * t)
        )
        dn_dy = (
            self.An * ky * jnp.cos(kx * x) * jnp.cos(ky * y - self.phase) * jnp.exp(self.sigma * t)
        )

        bracket_phi_n = dphi_dx * dn_dy - dphi_dy * dn_dx

        # Laplacians.
        lap_n = -k2 * n
        lap_omega = -k2 * omega

        # Time derivatives.
        dt_n = self.sigma * n
        dt_omega = self.sigma * omega

        rhs_n = -bracket_phi_n - kappa * dphi_dy + alpha * (phi - n) + Dn * lap_n

        # For ω equation we need [φ,ω]. Since ω is proportional to φ, [φ,ω]=0 in continuum.
        bracket_phi_omega = jnp.zeros_like(omega)
        rhs_w = -bracket_phi_omega - kappa * dn_dy + alpha * (phi - n) + DOmega * lap_omega

        Sn = dt_n - rhs_n
        Sw = dt_omega - rhs_w
        return HW2DState(n=Sn, omega=Sw, N=None)
