"""Linear dispersion benchmarks B2 (resistive drift wave) and B3 (shear Alfven).

These verify that the linear operators in :mod:`jax_drb.linear`, assembled
directly from the reduced model equations, reproduce the analytic dispersion
relations from the literature when diagonalized -- and that the general
Jacobian-based engine recovers a known operator.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.linear import (
    drift_wave_adiabatic_frequency,
    eigenmodes,
    jacobian_operator,
    resistive_drift_wave_operator,
    shear_alfven_frequency,
    shear_alfven_operator,
)


def _mode_frequency(operator) -> float:
    """Physical angular frequency |Im(lambda)| of the fastest eigenmode."""

    modes = eigenmodes(operator)
    return float(jnp.max(jnp.abs(modes.frequencies)))


def _mode_growth(operator) -> float:
    modes = eigenmodes(operator)
    return float(modes.dominant_growth_rate)


# --- B3: shear-Alfven wave with electron inertia --------------------------------

_ALFVEN_SPEED = 3.2e6      # m/s
_SKIN_DEPTH = 1.1e-3       # m


@pytest.mark.parametrize("k_par", [10.0, 50.0, 200.0])
@pytest.mark.parametrize("k_perp", [0.0, 100.0, 800.0])
def test_shear_alfven_operator_matches_analytic_dispersion(k_par, k_perp) -> None:
    operator = shear_alfven_operator(k_par, k_perp, _ALFVEN_SPEED, _SKIN_DEPTH)
    numeric = _mode_frequency(operator)
    analytic = float(shear_alfven_frequency(k_par, k_perp, _ALFVEN_SPEED, _SKIN_DEPTH))
    assert numeric == pytest.approx(analytic, rel=1e-10)
    # The ideal shear-Alfven wave is undamped: growth rate is zero.
    assert abs(_mode_growth(operator)) < 1e-6 * analytic


def test_shear_alfven_electron_inertia_lowers_frequency() -> None:
    # Finite k_perp * d_e reduces the phase velocity below v_A * k_par.
    ideal = float(shear_alfven_frequency(120.0, 0.0, _ALFVEN_SPEED, _SKIN_DEPTH))
    inertial = float(shear_alfven_frequency(120.0, 900.0, _ALFVEN_SPEED, _SKIN_DEPTH))
    assert inertial < ideal
    assert _mode_frequency(
        shear_alfven_operator(120.0, 900.0, _ALFVEN_SPEED, _SKIN_DEPTH)
    ) == pytest.approx(inertial, rel=1e-10)


def test_shear_alfven_matches_committed_benchmark_scalars() -> None:
    # Reproduce the analytic omega of the code's own Alfven-wave benchmark deck
    # from the linear operator, so B3 is anchored to the documented convention.
    from jax_drb.config.boutinp import parse_bout_input
    from jax_drb.native.units import resolved_dataset_scalars
    from jax_drb.runtime.run_config import RunConfiguration
    from jax_drb.validation import compute_alfven_wave_benchmark_scalars

    deck = """
nout = 10
timestep = 10

[mesh]
nx = 5
ny = 32
nz = 27
Lx = 0.1
Ly = 10
Lz = 1
B = 0.2
dx = Lx / (nx - 4)
dy = Ly / ny
dz = Lz / nz
g11 = 1
g22 = 1
g33 = 1
J = 1

[mesh:paralleltransform]
type = identity

[model]
components = (e, i, electromagnetic, vorticity)
Nnorm = 1e19
Tnorm = 100
Bnorm = 0.2

[e]
AA = 1/1836

[i]
AA = 2
density = 1e19
"""
    config = parse_bout_input(deck)
    scalars = resolved_dataset_scalars(RunConfiguration.from_config(config))
    benchmark = compute_alfven_wave_benchmark_scalars(config, dataset_scalars=scalars)
    numeric = _mode_frequency(
        shear_alfven_operator(
            benchmark.kpar,
            benchmark.kperp,
            benchmark.alfven_speed,
            benchmark.electron_skin_depth,
        )
    )
    assert numeric == pytest.approx(benchmark.analytic_omega, rel=1e-10)


# --- B2: resistive drift wave (Hasegawa-Wakatani) -------------------------------

_KY = 0.5
_KPERP2 = 0.5**2 + 0.3**2
_KAPPA = 1.0


def test_drift_wave_adiabatic_limit_recovers_drift_frequency() -> None:
    # As alpha -> infinity the mode becomes adiabatic (phi ~ n) and oscillates
    # at the diamagnetic drift frequency omega_star = kappa k_y / (1 + kperp^2).
    operator = resistive_drift_wave_operator(_KY, _KPERP2, 1.0e6, _KAPPA)
    omega_star = float(drift_wave_adiabatic_frequency(_KY, _KPERP2, _KAPPA))
    assert _mode_frequency(operator) == pytest.approx(omega_star, rel=1e-3)
    assert abs(_mode_growth(operator)) < 1e-3 * omega_star


def test_drift_wave_is_unstable_at_finite_adiabaticity() -> None:
    # Finite parallel resistivity (finite alpha) destabilizes the drift wave:
    # the dominant eigenmode has a positive growth rate.
    for alpha in (0.5, 1.0, 2.0):
        growth = _mode_growth(resistive_drift_wave_operator(_KY, _KPERP2, alpha, _KAPPA))
        assert growth > 0.0


def test_drift_wave_growth_increases_toward_hydrodynamic_regime() -> None:
    # Growth rate rises as alpha decreases from the adiabatic toward the
    # hydrodynamic regime, a signature of the resistive drift instability.
    growths = [
        _mode_growth(resistive_drift_wave_operator(_KY, _KPERP2, alpha, _KAPPA))
        for alpha in (10.0, 2.0, 0.5)
    ]
    assert growths[0] < growths[1] < growths[2]


def test_drift_wave_needs_a_density_gradient() -> None:
    # With zero gradient drive there is no instability and no real frequency.
    operator = resistive_drift_wave_operator(_KY, _KPERP2, 1.0, 0.0)
    assert _mode_growth(operator) <= 1e-12
    assert _mode_frequency(operator) < 1e-12


def test_drift_wave_eigenvalues_match_characteristic_polynomial() -> None:
    # Independent check of the eigensolver: the eigenvalues must be the roots of
    # lambda^2 - tr(A) lambda + det(A) = 0 for the 2x2 operator.
    operator = np.asarray(resistive_drift_wave_operator(_KY, _KPERP2, 1.5, _KAPPA))
    trace = operator[0, 0] + operator[1, 1]
    determinant = operator[0, 0] * operator[1, 1] - operator[0, 1] * operator[1, 0]
    roots = np.roots([1.0, -trace, determinant])
    modes = eigenmodes(operator)
    numeric = np.sort_complex(np.asarray(modes.eigenvalues))
    assert np.allclose(numeric, np.sort_complex(roots), rtol=1e-10, atol=1e-12)


# --- general engine -------------------------------------------------------------

def test_linear_dispersion_example_runs(tmp_path, monkeypatch) -> None:
    # Smoke-run the benchmark example at reduced resolution so CI keeps it alive.
    import importlib.util

    example = Path(__file__).resolve().parents[1] / "examples" / "benchmarks" / "linear_dispersion_demo.py"
    spec = importlib.util.spec_from_file_location("_linear_dispersion_demo", example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "linear_dispersion")
    monkeypatch.setattr(module, "K_PERP_SCAN", np.linspace(0.0, 1500.0, 5))
    monkeypatch.setattr(module, "ADIABATICITY_SCAN", np.geomspace(0.1, 100.0, 5))
    module.main()
    assert (tmp_path / "linear_dispersion" / "linear_dispersion.png").exists()
    assert (tmp_path / "linear_dispersion" / "linear_dispersion.json").exists()


def test_jacobian_operator_recovers_a_known_linear_system() -> None:
    # A manufactured 3x3 system with prescribed eigenvalues: the Jacobian-based
    # engine must recover the growth rates and frequencies of a linear rhs.
    matrix = jnp.array(
        [[-0.2, 3.0, 0.0], [-3.0, -0.2, 0.0], [0.0, 0.0, 0.5]],
        dtype=jnp.float64,
    )

    def rhs(state):
        return matrix @ state

    modes = eigenmodes(jacobian_operator(rhs, jnp.zeros(3)))
    growths = np.sort(np.asarray(modes.growth_rates))
    freqs = np.sort(np.abs(np.asarray(modes.frequencies)))
    # eigenvalues are 0.5 and -0.2 +/- 3i
    assert np.allclose(growths, [-0.2, -0.2, 0.5], atol=1e-10)
    assert np.allclose(freqs, [0.0, 3.0, 3.0], atol=1e-10)
