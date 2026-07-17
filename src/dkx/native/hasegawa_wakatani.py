"""JAX-native 2-D Hasegawa-Wakatani drift-wave turbulence model.

This is the closed-field-line (periodic flux-tube) drift-wave turbulence
flagship: a pseudo-spectral solver for the two-field Hasegawa-Wakatani system
in the perpendicular plane,

    d/dt zeta = -{phi, zeta} + alpha (phi - n) - nu * lap^2 zeta - mu * zeta
    d/dt n    = -{phi, n} - kappa d/dy phi + alpha (phi - n) - nu * lap^2 n - mu * n

with vorticity ``zeta = lap phi`` (so ``phi_k = -zeta_k / k^2``), adiabaticity
``alpha``, background density gradient ``kappa``, hyperviscosity ``nu``, and an
optional scale-independent friction ``mu`` (default 0). The friction models
large-scale drag (e.g. sheath or neutral damping in the tokamak edge) and, by
absorbing the 2-D inverse cascade at the box scale, lets a fixed-step run reach
a statistically steady saturated state.
``{a, b} = da/dx db/dy - da/dy db/dx`` is the E x B Poisson bracket, evaluated
pseudo-spectrally with 2/3-rule dealiasing.

The whole right-hand side is written in JAX, so a run is ``jit``-compiled and
differentiable. Its single-mode linear growth rate reproduces the eigenvalue of
:func:`dkx.linear.resistive_drift_wave_operator` (benchmark B2); at finite
amplitude it develops nonlinear E x B transport with an outward particle flux
that saturates statistically when ``mu > 0`` absorbs the inverse cascade.

Initial spectra must be Hermitian (e.g. the FFT of a real field). The solver
evolves the complex spectral state as given; a non-Hermitian state corresponds
to complex-valued fields, i.e. an unphysical complexified system that does not
obey the real system's energy balance and can blow up nonlinearly.

References: Hasegawa & Wakatani, Phys. Rev. Lett. 50, 682 (1983); Numata
et al., Phys. Plasmas 14, 102312 (2007).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = [
    "HasegawaWakataniParameters",
    "HasegawaWakataniGrid",
    "hw_grid",
    "potential_from_vorticity",
    "hw_rhs",
    "hw_step",
    "hw_run",
    "hw_run_flux_history",
    "particle_flux",
]


@dataclass(frozen=True)
class HasegawaWakataniParameters:
    """Physical parameters of the Hasegawa-Wakatani system."""

    adiabaticity: float = 1.0        # alpha: parallel coupling
    gradient: float = 1.0            # kappa: background density gradient drive
    hyperviscosity: float = 1.0e-3   # nu: lap^2 damping of the grid scale
    friction: float = 0.0            # mu: linear drag absorbing the inverse cascade


@dataclass(frozen=True)
class HasegawaWakataniGrid:
    """Periodic perpendicular grid and its spectral wavenumbers."""

    n: int
    length: float
    kx: jnp.ndarray
    ky: jnp.ndarray
    k2: jnp.ndarray
    inv_k2: jnp.ndarray
    dealias: jnp.ndarray


def hw_grid(n: int, length: float) -> HasegawaWakataniGrid:
    """Build an ``n x n`` periodic grid of side ``length`` with spectral operators."""

    modes = jnp.fft.fftfreq(n, d=length / n) * 2.0 * jnp.pi
    kx = modes[:, None] * jnp.ones((1, n))
    ky = jnp.ones((n, 1)) * modes[None, :]
    k2 = kx**2 + ky**2
    inv_k2 = jnp.where(k2 > 0, 1.0 / jnp.where(k2 > 0, k2, 1.0), 0.0)
    # 2/3-rule dealiasing mask.
    kmax = jnp.max(jnp.abs(modes)) * (2.0 / 3.0)
    dealias = (jnp.abs(kx) <= kmax) & (jnp.abs(ky) <= kmax)
    return HasegawaWakataniGrid(
        n=n, length=length, kx=kx, ky=ky, k2=k2, inv_k2=inv_k2,
        dealias=dealias.astype(jnp.float64),
    )


def potential_from_vorticity(zeta_hat: jnp.ndarray, grid: HasegawaWakataniGrid) -> jnp.ndarray:
    """Invert ``lap phi = zeta`` in Fourier space: ``phi_k = -zeta_k / k^2``."""

    return -zeta_hat * grid.inv_k2


def _bracket_hat(phi_hat, field_hat, grid: HasegawaWakataniGrid) -> jnp.ndarray:
    """Poisson bracket ``{phi, field}`` returned in Fourier space (dealiased)."""

    dphi_dx = jnp.fft.ifft2(1j * grid.kx * phi_hat)
    dphi_dy = jnp.fft.ifft2(1j * grid.ky * phi_hat)
    df_dx = jnp.fft.ifft2(1j * grid.kx * field_hat)
    df_dy = jnp.fft.ifft2(1j * grid.ky * field_hat)
    bracket = dphi_dx * df_dy - dphi_dy * df_dx
    return jnp.fft.fft2(bracket) * grid.dealias


def hw_rhs(zeta_hat, n_hat, grid: HasegawaWakataniGrid, params: HasegawaWakataniParameters):
    """Spectral right-hand side ``(d zeta_hat/dt, d n_hat/dt)``."""

    phi_hat = potential_from_vorticity(zeta_hat, grid)
    coupling = params.adiabaticity * (phi_hat - n_hat)
    damping_zeta = params.hyperviscosity * grid.k2**2 * zeta_hat + params.friction * zeta_hat
    damping_n = params.hyperviscosity * grid.k2**2 * n_hat + params.friction * n_hat
    d_zeta = -_bracket_hat(phi_hat, zeta_hat, grid) + coupling - damping_zeta
    d_n = (
        -_bracket_hat(phi_hat, n_hat, grid)
        - params.gradient * (1j * grid.ky) * phi_hat
        + coupling
        - damping_n
    )
    return d_zeta, d_n


def hw_step(zeta_hat, n_hat, grid, params, dt):
    """One classical RK4 step of the spectral system."""

    def rhs(z, m):
        return hw_rhs(z, m, grid, params)

    k1z, k1n = rhs(zeta_hat, n_hat)
    k2z, k2n = rhs(zeta_hat + 0.5 * dt * k1z, n_hat + 0.5 * dt * k1n)
    k3z, k3n = rhs(zeta_hat + 0.5 * dt * k2z, n_hat + 0.5 * dt * k2n)
    k4z, k4n = rhs(zeta_hat + dt * k3z, n_hat + dt * k3n)
    zeta_next = zeta_hat + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
    n_next = n_hat + (dt / 6.0) * (k1n + 2 * k2n + 2 * k3n + k4n)
    return zeta_next, n_next


def hw_run(zeta_hat, n_hat, grid, params, *, dt, steps):
    """Advance ``steps`` RK4 steps with a jitted ``lax.scan``; return final spectra.

    ``grid``, ``params``, ``dt``, and ``steps`` are captured as constants, so the
    compiled function depends only on the two spectral state arrays.
    """

    @jax.jit
    def _run(zeta0, n0):
        def body(carry, _):
            z, m = carry
            return hw_step(z, m, grid, params, dt), None

        (zeta_final, n_final), _ = jax.lax.scan(body, (zeta0, n0), None, length=steps)
        return zeta_final, n_final

    return _run(zeta_hat, n_hat)


def hw_run_flux_history(zeta_hat, n_hat, grid, params, *, dt, steps, sample_every=1):
    """Advance ``steps`` RK4 steps; return final spectra and the sampled flux history.

    Like :func:`hw_run`, but the jitted ``lax.scan`` also records the
    domain-averaged radial particle flux every ``sample_every`` steps (an array
    of length ``steps // sample_every``). Because the whole scan is JAX, any
    reduction of the history (e.g. its time average over a saturated window) is
    differentiable with respect to the physical parameters.
    """

    outer = steps // sample_every

    @jax.jit
    def _run(zeta0, n0):
        def inner(carry, _):
            z, m = carry
            return hw_step(z, m, grid, params, dt), None

        def body(carry, _):
            carry, _ = jax.lax.scan(inner, carry, None, length=sample_every)
            return carry, particle_flux(carry[0], carry[1], grid)

        (zeta_final, n_final), fluxes = jax.lax.scan(body, (zeta0, n0), None, length=outer)
        return zeta_final, n_final, fluxes

    return _run(zeta_hat, n_hat)


def particle_flux(zeta_hat, n_hat, grid: HasegawaWakataniGrid) -> jnp.ndarray:
    """Domain-averaged radial E x B particle flux ``<n v_x> = <n * (-d phi/dy)>``."""

    phi_hat = potential_from_vorticity(zeta_hat, grid)
    v_x = jnp.real(jnp.fft.ifft2(-1j * grid.ky * phi_hat))
    density = jnp.real(jnp.fft.ifft2(n_hat))
    return jnp.mean(density * v_x)
