"""Tokamak closed-field-line flagship: 2-D Hasegawa-Wakatani drift-wave turbulence.

The nonlinear pseudo-spectral model is cross-checked against the linear
dispersion solver (B2): a single Fourier mode has zero self-bracket, so it
evolves purely linearly and its growth rate must equal the eigenvalue of
``jax_drb.linear.resistive_drift_wave_operator``. Further tests cover the ideal
invariant, the transport direction, and end-to-end differentiability.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.linear import eigenmodes, resistive_drift_wave_operator
from jax_drb.native.hasegawa_wakatani import (
    HasegawaWakataniParameters,
    hw_grid,
    hw_run,
    particle_flux,
    potential_from_vorticity,
)

_N = 32
_LENGTH = 2.0 * np.pi * 4.0


def _seed_eigenmode(mx, my, alpha, kappa):
    """Seed the dominant linear eigenmode of a single Fourier component."""

    kx = 2.0 * np.pi * mx / _LENGTH
    ky = 2.0 * np.pi * my / _LENGTH
    k2 = kx * kx + ky * ky
    operator = np.asarray(resistive_drift_wave_operator(ky, k2, alpha, kappa))
    modes = eigenmodes(operator)
    eigenvalue = complex(np.asarray(modes.eigenvalues)[0])
    vector = np.asarray(modes.eigenvectors)[:, 0]
    zeta = np.zeros((_N, _N), dtype=complex)
    density = np.zeros((_N, _N), dtype=complex)
    zeta[mx, my] = -k2 * vector[0] * 1.0e-6
    density[mx, my] = vector[1] * 1.0e-6
    return jnp.array(zeta), jnp.array(density), eigenvalue, (mx, my)


@pytest.mark.parametrize("mx,my", [(1, 1), (1, 2), (2, 1)])
def test_linear_growth_matches_drift_wave_eigenvalue(mx, my) -> None:
    # A single mode carries zero self-bracket, so the nonlinear model reproduces
    # the linear operator's growth rate exactly (B2 cross-check).
    grid = hw_grid(_N, _LENGTH)
    params = HasegawaWakataniParameters(adiabaticity=1.0, gradient=1.0, hyperviscosity=0.0)
    zeta, density, eigenvalue, (i, j) = _seed_eigenmode(mx, my, 1.0, 1.0)
    dt, steps = 1.0e-3, 300
    zeta_f, _ = hw_run(zeta, density, grid, params, dt=dt, steps=steps)
    amplitude_ratio = abs(complex(np.asarray(zeta_f)[i, j])) / abs(complex(np.asarray(zeta)[i, j]))
    growth = math.log(amplitude_ratio) / (dt * steps)
    assert growth == pytest.approx(eigenvalue.real, rel=1e-6)


def test_single_mode_has_zero_self_bracket() -> None:
    # The Poisson bracket of a single Fourier mode with itself vanishes, which is
    # why the linear cross-check above is exact rather than approximate.
    from jax_drb.native.hasegawa_wakatani import _bracket_hat

    grid = hw_grid(_N, _LENGTH)
    field = jnp.zeros((_N, _N), dtype=complex).at[2, 3].set(1.0 + 0.5j)
    phi = potential_from_vorticity(field, grid)
    bracket = _bracket_hat(phi, field, grid)
    assert float(jnp.max(jnp.abs(bracket))) < 1e-12


def test_ideal_system_conserves_energy() -> None:
    # With no drive, coupling, or dissipation the model is 2-D advection, whose
    # energy E = <|grad phi|^2 + n^2> is an ideal invariant.
    grid = hw_grid(_N, _LENGTH)
    params = HasegawaWakataniParameters(adiabaticity=0.0, gradient=0.0, hyperviscosity=0.0)
    rng = np.random.default_rng(1)
    # Hermitian seed (FFT of a real field): a non-Hermitian spectrum evolves the
    # unphysical complexified system, which does not conserve the energy.
    field = np.fft.fft2(rng.standard_normal((_N, _N))) * (1.0e-2 / _N)
    field[0, 0] = 0.0
    zeta = jnp.array(field)
    density = jnp.array(field * 0.5)

    def energy(z, n):
        phi = potential_from_vorticity(z, grid)
        return float(jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(n) ** 2))

    e0 = energy(zeta, density)
    zeta_f, density_f = hw_run(zeta, density, grid, params, dt=1.0e-3, steps=400)
    e1 = energy(zeta_f, density_f)
    assert e1 == pytest.approx(e0, rel=1e-4)


def test_particle_flux_is_outward_under_positive_gradient() -> None:
    # A growing drift wave transports density down the background gradient
    # (outward), giving a positive radial E x B particle flux.
    grid = hw_grid(_N, _LENGTH)
    params = HasegawaWakataniParameters(adiabaticity=1.0, gradient=1.0, hyperviscosity=1.0e-3)
    zeta, density, _, _ = _seed_eigenmode(1, 2, 1.0, 1.0)
    zeta = zeta * 1.0e3  # lift into a measurable amplitude
    density = density * 1.0e3
    zeta_f, density_f = hw_run(zeta, density, grid, params, dt=1.0e-2, steps=300)
    assert float(particle_flux(zeta_f, density_f, grid)) > 0.0


def test_inverse_design_recovers_a_parameter_through_turbulence() -> None:
    # Gradient descent on the density-gradient drive, differentiating through the
    # whole nonlinear turbulence run, recovers the drive that produced a target
    # fluctuation energy. This is the differentiable inverse-design capability.
    grid = hw_grid(24, 2.0 * np.pi * 5.0)
    rng = np.random.default_rng(3)
    # Hermitian seed (FFT of a real field) so the run is the physical system.
    field = np.fft.fft2(rng.standard_normal((24, 24))) * (1.0e-2 / 24.0)
    field[0, 0] = 0.0
    zeta0, density0 = jnp.array(field), jnp.array(field * 0.7)

    def final_energy(kappa):
        params = HasegawaWakataniParameters(adiabaticity=1.0, gradient=kappa, hyperviscosity=3.0e-2)
        zeta, density = hw_run(zeta0, density0, grid, params, dt=5.0e-3, steps=150)
        phi = potential_from_vorticity(zeta, grid)
        return jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2)

    target_kappa = 1.3
    target = float(final_energy(target_kappa))

    def loss(kappa):
        return (jnp.log(final_energy(kappa)) - jnp.log(target)) ** 2

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    kappa = 0.6
    first_loss = float(loss(kappa))
    for _ in range(40):
        _, grad = value_and_grad(kappa)
        kappa = float(np.clip(kappa - 0.15 * grad, 0.05, 3.0))
    final_loss = float(loss(kappa))
    assert final_loss < 1e-2 * first_loss  # optimization drives the loss down
    assert kappa == pytest.approx(target_kappa, abs=0.08)  # and recovers the drive


# ---------------------------------------------------------------------------
# Example smoke coverage. The tokamak examples are flat pedagogical scripts
# (no main()), so importing one executes a multi-minute turbulence run --
# far over the smoke budget. Instead each example is (a) statically checked
# (it must parse and only use the public API exercised below) and (b) its
# exact src-level code path is exercised at tiny size, so a break in the API
# the scripts rely on fails these tests immediately.
# ---------------------------------------------------------------------------

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "tokamak"
_EXAMPLE_FILES = [
    "drift_wave_turbulence.py",
    "drift_wave_inverse_design.py",
    "hasegawa_wakatani_optimization.py",
]


@pytest.mark.parametrize("filename", _EXAMPLE_FILES)
def test_example_scripts_parse_and_stay_flat(filename) -> None:
    # The examples are flat top-to-bottom scripts: they must parse, import only
    # public names from the HW module, and contain no main()/argparse plumbing.
    import ast

    source = (_EXAMPLES_DIR / filename).read_text()
    tree = ast.parse(source)
    assert "argparse" not in source and "__main__" not in source
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "jax_drb.native.hasegawa_wakatani"
        for alias in node.names
    ]
    assert imported, f"{filename} must use the public Hasegawa-Wakatani API"
    import jax_drb.native.hasegawa_wakatani as hw

    assert all(name in hw.__all__ for name in imported)


def _hermitian_noise(grid, n, seed, amplitude):
    # The examples' initial condition: FFT of a real field (Hermitian spectrum,
    # so the evolved fields stay real), mean removed, low-k weighted.
    rng = np.random.default_rng(seed)
    noise_hat = np.fft.fft2(rng.standard_normal((n, n)))
    noise_hat[0, 0] = 0.0
    noise_hat *= np.exp(-np.asarray(grid.k2)) * np.asarray(grid.dealias)
    noise_hat *= amplitude / np.sqrt(np.mean(np.real(np.fft.ifft2(noise_hat)) ** 2))
    return jnp.array(noise_hat)


def test_turbulence_example_api_path() -> None:
    # Tiny-size version of examples/tokamak/drift_wave_turbulence.py: seeded
    # noise, friction-regularized run, energy and flux diagnostics stay finite
    # and the density field stays real (Hermitian spectrum preserved).
    from jax_drb.native.hasegawa_wakatani import hw_run

    n = 16
    grid = hw_grid(n, 2.0 * np.pi * 4.0)
    params = HasegawaWakataniParameters(
        adiabaticity=1.0, gradient=1.0, hyperviscosity=1.0e-2, friction=3.0e-2
    )
    zeta = _hermitian_noise(grid, n, seed=0, amplitude=5.0e-2)
    density = zeta
    for _ in range(3):
        zeta, density = hw_run(zeta, density, grid, params, dt=5.0e-3, steps=20)
        phi = potential_from_vorticity(zeta, grid)
        energy = float(jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2))
        assert np.isfinite(energy)
        assert np.isfinite(float(particle_flux(zeta, density, grid)))
    density_real = np.asarray(jnp.fft.ifft2(density))
    assert np.max(np.abs(density_real.imag)) < 1e-10 * max(np.max(np.abs(density_real)), 1e-30)


def test_inverse_design_example_api_path() -> None:
    # Tiny-size version of examples/tokamak/drift_wave_inverse_design.py:
    # a few gradient-descent steps on kappa reduce the energy-matching loss.
    from jax_drb.native.hasegawa_wakatani import hw_run

    n = 16
    grid = hw_grid(n, 2.0 * np.pi * 5.0)
    zeta0 = _hermitian_noise(grid, n, seed=3, amplitude=1.0e-2)
    density0 = zeta0 * 0.7

    def final_energy(kappa):
        params = HasegawaWakataniParameters(
            adiabaticity=1.0, gradient=kappa, hyperviscosity=3.0e-2
        )
        zeta, density = hw_run(zeta0, density0, grid, params, dt=5.0e-3, steps=60)
        phi = potential_from_vorticity(zeta, grid)
        return jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2)

    target = float(final_energy(1.3))

    def loss(kappa):
        return (jnp.log(final_energy(kappa)) - jnp.log(target)) ** 2

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    kappa = 0.6
    first_loss = float(loss(kappa))
    for _ in range(5):
        _, grad = value_and_grad(kappa)
        kappa = float(np.clip(kappa - 0.15 * float(grad), 0.05, 3.0))
    assert float(loss(kappa)) < first_loss


def test_optimization_example_api_path() -> None:
    # Tiny-size version of examples/tokamak/hasegawa_wakatani_optimization.py:
    # windowed mean-flux objective via hw_run_flux_history, forward-mode
    # derivative with respect to ln(alpha) via jax.jvp, and one safeguarded
    # damped-Newton step that stays finite and inside the trust region.
    from jax_drb.native.hasegawa_wakatani import hw_run, hw_run_flux_history

    n = 16
    grid = hw_grid(n, 2.0 * np.pi * 4.0)
    zeta_seed = _hermitian_noise(grid, n, seed=7, amplitude=0.5)
    density_seed = zeta_seed * 0.9

    def make_params(alpha):
        return HasegawaWakataniParameters(
            adiabaticity=alpha, gradient=1.0, hyperviscosity=1.0e-2, friction=3.0e-2
        )

    zeta_base, density_base = hw_run(
        zeta_seed, density_seed, grid, make_params(0.3), dt=5.0e-3, steps=40
    )

    def windowed_ln_flux(ln_alpha):
        _, _, fluxes = hw_run_flux_history(
            zeta_base, density_base, grid, make_params(jnp.exp(ln_alpha)),
            dt=5.0e-3, steps=40, sample_every=5,
        )
        return jnp.log(jnp.mean(jnp.abs(fluxes[-4:]) + 1e-30))

    ln_alpha = float(np.log(0.3))
    value, slope = jax.jvp(windowed_ln_flux, (ln_alpha,), (1.0,))
    assert np.isfinite(float(value)) and np.isfinite(float(slope))
    residual = float(value) - np.log(1e-3)
    safe_slope = min(float(slope), -0.3)
    step = float(np.clip(-residual / safe_slope, -0.5, 0.5))
    assert np.isfinite(step) and abs(step) <= 0.5


def test_flux_history_matches_plain_run() -> None:
    # hw_run_flux_history is the reusable helper behind the optimization
    # example: same integrator, so its final state must equal hw_run's, and
    # its last flux sample must equal the flux of that final state.
    from jax_drb.native.hasegawa_wakatani import hw_run, hw_run_flux_history

    n = 16
    grid = hw_grid(n, 2.0 * np.pi * 4.0)
    zeta0 = _hermitian_noise(grid, n, seed=5, amplitude=0.1)
    density0 = zeta0 * 0.8
    params = HasegawaWakataniParameters(
        adiabaticity=0.7, gradient=1.0, hyperviscosity=1.0e-2, friction=2.0e-2
    )
    zeta_a, density_a = hw_run(zeta0, density0, grid, params, dt=5.0e-3, steps=40)
    zeta_b, density_b, fluxes = hw_run_flux_history(
        zeta0, density0, grid, params, dt=5.0e-3, steps=40, sample_every=10
    )
    # XLA may fuse the nested (sampled) scan differently from the flat one, so
    # agreement is to roundoff, not bitwise.
    np.testing.assert_allclose(
        np.asarray(zeta_a), np.asarray(zeta_b), rtol=1e-8, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(density_a), np.asarray(density_b), rtol=1e-8, atol=1e-14
    )
    assert fluxes.shape == (4,)
    assert float(fluxes[-1]) == pytest.approx(
        float(particle_flux(zeta_b, density_b, grid)), rel=1e-12
    )


def test_friction_damps_a_single_mode_at_the_exact_rate() -> None:
    # The optional friction parameter is a scale-independent linear drag: with
    # all other physics off, a single mode must decay at exactly exp(-mu t).
    from jax_drb.native.hasegawa_wakatani import hw_run

    grid = hw_grid(_N, _LENGTH)
    mu = 0.35
    params = HasegawaWakataniParameters(
        adiabaticity=0.0, gradient=0.0, hyperviscosity=0.0, friction=mu
    )
    zeta = jnp.zeros((_N, _N), dtype=complex).at[2, 3].set(1.0)
    density = jnp.zeros((_N, _N), dtype=complex).at[2, 3].set(0.5)
    dt, steps = 1.0e-2, 200
    zeta_f, density_f = hw_run(zeta, density, grid, params, dt=dt, steps=steps)
    expected = np.exp(-mu * dt * steps)
    assert abs(complex(np.asarray(zeta_f)[2, 3])) == pytest.approx(expected, rel=1e-8)
    assert abs(complex(np.asarray(density_f)[2, 3])) == pytest.approx(0.5 * expected, rel=1e-8)


def test_run_is_differentiable_end_to_end() -> None:
    # The whole spectral run is JAX: the saturated fluctuation energy has a
    # finite gradient with respect to the adiabaticity, verified vs finite
    # differences. This is the differentiable-turbulence capability.
    grid = hw_grid(_N, _LENGTH)
    rng = np.random.default_rng(2)
    # Hermitian seed (FFT of a real field) so the run is the physical system.
    field = np.fft.fft2(rng.standard_normal((_N, _N))) * (1.0e-3 / _N)
    field[0, 0] = 0.0
    zeta0 = jnp.array(field)
    density0 = jnp.array(field * 0.7)

    def final_energy(alpha):
        params = HasegawaWakataniParameters(adiabaticity=alpha, gradient=1.0, hyperviscosity=5.0e-3)
        zeta_f, density_f = hw_run(zeta0, density0, grid, params, dt=1.0e-2, steps=200)
        phi = potential_from_vorticity(zeta_f, grid)
        return jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density_f) ** 2)

    grad = float(jax.grad(final_energy)(1.0))
    step = 1.0e-3
    fd = (float(final_energy(1.0 + step)) - float(final_energy(1.0 - step))) / (2.0 * step)
    assert np.isfinite(grad)
    assert grad == pytest.approx(fd, rel=1e-3, abs=1e-6)
