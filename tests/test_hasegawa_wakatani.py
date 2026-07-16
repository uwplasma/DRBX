"""Tokamak closed-field-line flagship: 2-D Hasegawa-Wakatani drift-wave turbulence.

The nonlinear pseudo-spectral model is cross-checked against the linear
dispersion solver (B2): a single Fourier mode has zero self-bracket, so it
evolves purely linearly and its growth rate must equal the eigenvalue of
``jax_drb.linear.resistive_drift_wave_operator``. Further tests cover the ideal
invariant, the transport direction, and end-to-end differentiability.
"""

from __future__ import annotations

import math

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
    field = 1.0e-2 * (rng.standard_normal((_N, _N)) + 1j * rng.standard_normal((_N, _N)))
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
    field = 1.0e-2 * (rng.standard_normal((24, 24)) + 1j * rng.standard_normal((24, 24)))
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


def test_turbulence_example_runs(tmp_path, monkeypatch) -> None:
    # Smoke-run the flagship example at tiny resolution so it stays alive fast.
    import importlib.util
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "tokamak" / "drift_wave_turbulence_demo.py"
    spec = importlib.util.spec_from_file_location("_drift_wave_turbulence_demo", example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "N", 16)
    monkeypatch.setattr(module, "STEPS_PER_BLOCK", 20)
    monkeypatch.setattr(module, "BLOCKS", 3)
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "drift_wave_turbulence")
    module.main()
    assert (tmp_path / "drift_wave_turbulence" / "drift_wave_turbulence.png").exists()


def test_inverse_design_example_runs(tmp_path, monkeypatch) -> None:
    import importlib.util
    from pathlib import Path

    example = (
        Path(__file__).resolve().parents[1]
        / "examples" / "tokamak" / "drift_wave_inverse_design_demo.py"
    )
    spec = importlib.util.spec_from_file_location("_drift_wave_inverse_design_demo", example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "N", 16)
    monkeypatch.setattr(module, "STEPS", 60)
    monkeypatch.setattr(module, "ITERATIONS", 5)
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "inverse_design")
    module.main()
    assert (tmp_path / "inverse_design" / "drift_wave_inverse_design.png").exists()


def test_run_is_differentiable_end_to_end() -> None:
    # The whole spectral run is JAX: the saturated fluctuation energy has a
    # finite gradient with respect to the adiabaticity, verified vs finite
    # differences. This is the differentiable-turbulence capability.
    grid = hw_grid(_N, _LENGTH)
    rng = np.random.default_rng(2)
    field = 1.0e-3 * (rng.standard_normal((_N, _N)) + 1j * rng.standard_normal((_N, _N)))
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
