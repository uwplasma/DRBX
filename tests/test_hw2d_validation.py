from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, HW2DState
from jaxdrb.nonlinear.stepper import rk4_step


def _real_field_from_single_mode(
    *,
    nx: int,
    ny: int,
    i: int,
    j: int,
    amp: complex,
) -> jnp.ndarray:
    """Create a real field with a single complex Fourier mode (i,j) and its conjugate."""

    hat = jnp.zeros((nx, ny), dtype=jnp.complex128)
    hat = hat.at[i, j].set(jnp.asarray(amp, dtype=jnp.complex128))
    hat = hat.at[(-i) % nx, (-j) % ny].set(jnp.conj(hat[i, j]))
    return jnp.fft.ifft2(hat).real


def test_hw2d_single_mode_linear_rhs_matches_analytic() -> None:
    """Check the linearized (single-mode) HW2D operator against the analytic Fourier form.

    This is anchored in the standard HW formulation used in Camargo et al. (1995):
    when linearized about (n,omega)=(0,0), each Fourier mode evolves independently.
    """

    nx = 32
    ny = 32
    grid = Grid2D.make(nx=nx, ny=ny, Lx=2 * math.pi, Ly=2 * math.pi, dealias=False)
    params = HW2DParams(
        kappa=1.0,
        alpha=0.7,
        Dn=3e-3,
        DOmega=2e-3,
        nu4_n=4e-6,
        nu4_omega=7e-6,
        bracket="spectral",
        poisson="spectral",
        dealias_on=False,
        alpha_nonzonal_only=False,
    )
    model = HW2DModel(params=params, grid=grid)

    # Pick a nontrivial Fourier mode (avoid k=0).
    i = 3
    j = 5
    n_hat = 1.2 + 0.4j
    w_hat = -0.7 + 0.3j

    n = _real_field_from_single_mode(nx=nx, ny=ny, i=i, j=j, amp=n_hat)
    omega = _real_field_from_single_mode(nx=nx, ny=ny, i=i, j=j, amp=w_hat)
    y = HW2DState(n=n, omega=omega)

    dy = model.rhs(0.0, y)
    dn_hat_num = jnp.fft.fft2(dy.n)[i, j]
    dw_hat_num = jnp.fft.fft2(dy.omega)[i, j]

    ky = jnp.asarray(grid.ky[i, j])
    k2 = jnp.asarray(grid.k2[i, j])
    phi_hat = -w_hat / k2

    dn_hat_expected = (
        -params.kappa * (1j * ky) * phi_hat
        + params.alpha * (phi_hat - n_hat)
        - params.Dn * k2 * n_hat
        - params.nu4_n * (k2**2) * n_hat
    )
    dw_hat_expected = (
        -params.kappa * (1j * ky) * n_hat
        + params.alpha * (phi_hat - n_hat)
        - params.DOmega * k2 * w_hat
        - params.nu4_omega * (k2**2) * w_hat
    )

    assert jnp.allclose(dn_hat_num, dn_hat_expected, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(dw_hat_num, dw_hat_expected, rtol=1e-10, atol=1e-10)


def test_hw2d_advection_energy_conserving_budget() -> None:
    """For kappa=alpha=diffusion=0, the nonlinear subset should conserve key invariants.

    - For the Arakawa Jacobian (recommended for conservation), the HW energy budget should close.
    - For the pseudo-spectral bracket, the L2 norms of advected scalars should be conserved.
    """

    nx = 48
    ny = 48
    grid = Grid2D.make(nx=nx, ny=ny, Lx=2 * math.pi, Ly=2 * math.pi, dealias=True)
    key = jax.random.key(0)
    # Start in the dealiased spectral subspace to avoid artifacts from high-k noise.
    from jaxdrb.nonlinear.spectral import dealias as dealias_spec

    phi = dealias_spec(jax.random.normal(key, (nx, ny)), grid.dealias_mask)
    n = dealias_spec(jax.random.normal(jax.random.split(key, 2)[1], (nx, ny)), grid.dealias_mask)

    # Build omega from phi so that the elliptic inversion is consistent.
    omega = jnp.fft.ifft2(-grid.k2 * jnp.fft.fft2(phi)).real

    y = HW2DState(n=n, omega=omega)

    # 1) Arakawa: energy-conserving nonlinear subset.
    params_a = HW2DParams(
        kappa=0.0,
        alpha=0.0,
        Dn=0.0,
        DOmega=0.0,
        nu4_n=0.0,
        nu4_omega=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
    )
    model_a = HW2DModel(params=params_a, grid=grid)
    budget = model_a.energy_budget(y)
    assert float(jnp.abs(budget["E_dot_adv"])) < 5e-10
    assert float(jnp.abs(budget["E_dot_total"])) < 5e-10

    # 2) Pseudo-spectral: L2 invariants for advected scalars.
    from jaxdrb.nonlinear.spectral import poisson_bracket_spectral

    Jn = poisson_bracket_spectral(
        phi=model_a.phi_from_omega(y.omega),
        f=y.n,
        kx=grid.kx,
        ky=grid.ky,
        dealias_mask=grid.dealias_mask,
    )
    Jw = poisson_bracket_spectral(
        phi=model_a.phi_from_omega(y.omega),
        f=y.omega,
        kx=grid.kx,
        ky=grid.ky,
        dealias_mask=grid.dealias_mask,
    )
    assert float(jnp.abs(jnp.mean(y.n * Jn))) < 5e-10
    assert float(jnp.abs(jnp.mean(y.omega * Jw))) < 5e-10


def test_hw2d_stability_no_nans_short_run() -> None:
    """A short HW2D run with hyperdiffusion should remain finite (no NaNs/Infs)."""

    nx = 48
    ny = 48
    grid = Grid2D.make(nx=nx, ny=ny, Lx=2 * math.pi, Ly=2 * math.pi, dealias=True)
    params = HW2DParams(
        kappa=1.0,
        alpha=0.5,
        Dn=1e-3,
        DOmega=1e-3,
        nu4_n=1e-6,
        nu4_omega=1e-6,
        bracket="spectral",
        poisson="spectral",
        dealias_on=True,
    )
    model = HW2DModel(params=params, grid=grid)

    key = jax.random.key(1)
    y = HW2DState(
        n=1e-3 * jax.random.normal(key, (nx, ny)),
        omega=1e-3 * jax.random.normal(jax.random.split(key, 2)[1], (nx, ny)),
    )

    dt = 0.01
    nsteps = 800  # t=8

    @jax.jit
    def integrate(y0: HW2DState) -> HW2DState:
        def body(i, carry):
            t, y_ = carry
            y_next = rk4_step(y_, t, dt, model.rhs)
            return (t + dt, y_next)

        _, y_end = jax.lax.fori_loop(0, nsteps, body, (jnp.asarray(0.0), y0))
        return y_end

    y_end = integrate(y)
    diag = model.diagnostics(y_end)
    assert bool(jnp.isfinite(diag["E"]))
    assert bool(jnp.isfinite(diag["Z"]))


def test_hw2d_ideal_invariants_conserved_over_time_arakawa() -> None:
    """Arakawa + periodic Poisson should conserve E and Z in the ideal limit.

    This is a standard nonlinear verification for vorticity-based 2D models (e.g. HW subsets)
    and is emphasized in the drift-wave turbulence literature as a check on the advection kernel.
    """

    nx = 48
    ny = 48
    grid = Grid2D.make(nx=nx, ny=ny, Lx=2 * math.pi, Ly=2 * math.pi, dealias=True)
    params = HW2DParams(
        kappa=0.0,
        alpha=0.0,
        Dn=0.0,
        DOmega=0.0,
        nu4_n=0.0,
        nu4_omega=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
    )
    model = HW2DModel(params=params, grid=grid)

    key = jax.random.key(2)
    from jaxdrb.nonlinear.spectral import dealias as dealias_spec

    n0 = dealias_spec(1e-2 * jax.random.normal(key, (nx, ny)), grid.dealias_mask)
    omega0 = dealias_spec(
        1e-2 * jax.random.normal(jax.random.split(key, 2)[1], (nx, ny)), grid.dealias_mask
    )
    y0 = HW2DState(n=n0, omega=omega0)

    dt = 0.01
    nsteps = 600  # t=6

    @jax.jit
    def integrate(y: HW2DState) -> HW2DState:
        def body(i, carry):
            t, y_ = carry
            y_next = rk4_step(y_, t, dt, model.rhs)
            return (t + dt, y_next)

        _, y_end = jax.lax.fori_loop(0, nsteps, body, (jnp.asarray(0.0), y))
        return y_end

    diag0 = model.diagnostics(y0)
    y_end = integrate(y0)
    diag1 = model.diagnostics(y_end)

    relE = float(jnp.abs(diag1["E"] - diag0["E"]) / jnp.maximum(diag0["E"], 1e-30))
    relZ = float(jnp.abs(diag1["Z"] - diag0["Z"]) / jnp.maximum(diag0["Z"], 1e-30))
    assert relE < 5e-5
    assert relZ < 5e-5


def test_hw2d_end_to_end_grad_is_finite() -> None:
    """Check end-to-end differentiability through time integration and the Poisson solve."""

    nx = 16
    ny = 16
    grid = Grid2D.make(nx=nx, ny=ny, Lx=2 * math.pi, Ly=2 * math.pi, dealias=True)
    key = jax.random.key(3)
    y0 = HW2DState(
        n=1e-3 * jax.random.normal(key, (nx, ny)),
        omega=1e-3 * jax.random.normal(jax.random.split(key, 2)[1], (nx, ny)),
    )

    dt = 0.02
    nsteps = 20

    def final_energy(alpha: float) -> jnp.ndarray:
        params = HW2DParams(
            kappa=1.0,
            alpha=alpha,
            Dn=1e-3,
            DOmega=1e-3,
            nu4_n=1e-6,
            nu4_omega=1e-6,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=True,
        )
        model = HW2DModel(params=params, grid=grid)

        def body(i, carry):
            t, y = carry
            return (t + dt, rk4_step(y, t, dt, model.rhs))

        _, y_end = jax.lax.fori_loop(0, nsteps, body, (jnp.asarray(0.0), y0))
        return model.diagnostics(y_end)["E"]

    g = jax.grad(final_energy)(0.7)
    assert bool(jnp.isfinite(g))
