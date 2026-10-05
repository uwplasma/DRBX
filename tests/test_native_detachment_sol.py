"""Qualification of the SD1D-matched 1D detachment model (B6).

Reference: SD1D ``13.6eV`` scan of Dudson et al., PPCF 61, 065008 (2019) --
hydrogen only, fixed 13.6 eV ionisation cost, no excitation, no impurity,
area expansion 2, 30 m leg, 5e7 W/m^2, recycling 0.99, ny = 800. The reference
target values are the final SD1D states of the published dataset; the two
profile sets in ``tests/data/sd1d_13p6eV_profiles.npz`` are the SD1D restart
fields (Ne, NVi, P, Nn, NVn, Pn, normalised) at n_up = 2e19 and 5e19.

Default suite: Jacobian structure, steady ledgers, implicit gradient vs finite
differences, and one SD1D point, all at cheap resolution or a single solve.
``slow``: the full 19-point SD1D comparison, profiles, and grid convergence.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native.neutrals import (
    DetachmentSolParameters,
    detachment_diagnostics,
    detachment_ledger,
    detachment_sol_run,
    detachment_target_outputs,
)
from drbx.native.neutrals import detachment_sol_model as dsm

jax.config.update("jax_enable_x64", True)

DATA = Path(__file__).parent / "data" / "sd1d_13p6eV_profiles.npz"

# SD1D 13.6eV scan: (achieved upstream density [m^-3], last-cell T_t [eV],
# Gamma_t * A_t [m^-2 s^-1, per unit upstream area]).
SD1D_13P6 = [
    (1.699818e19, 29.35, 1.640e24), (1.799896e19, 26.32, 1.822e24), (1.897272e19, 23.73, 2.015e24),
    (2.000355e19, 21.30, 2.231e24), (2.099797e19, 19.30, 2.452e24), (2.200214e19, 17.52, 2.688e24),
    (2.297850e19, 16.07, 2.915e24), (2.401330e19, 14.65, 3.176e24), (2.501205e19, 13.50, 3.427e24),
    (2.999828e19, 9.58, 4.691e24), (3.499920e19, 7.45, 5.870e24), (3.992420e19, 6.22, 6.861e24),
    (4.500669e19, 5.37, 7.746e24), (5.000658e19, 4.80, 8.460e24), (5.499997e19, 4.39, 9.067e24),
    (5.999908e19, 4.07, 9.577e24), (6.500008e19, 3.82, 1.002e25), (6.999388e19, 3.62, 1.041e25),
]
# SD1D's n_up = 7.5e19 restart is excluded from the tolerance: it carries no PI
# integral, its time (2.633e7) is earlier than the 7e19 run it continues from,
# and its state is not steady -- this module's residual of that state is
# ~100-500 against <= 7 for every other case (and its last-cell Mach jumps
# from 0.38 to 0.43 against the trend). DRBX gives T_t = 3.43 eV, 1.075e25.
SD1D_13P6_UNSTEADY = (7.521918e19, 3.13, 1.102e25)
# Observed agreement at 800 cells: |dT_t| <= 0.35 %, |d(Gamma_t A)| <= 0.15 %
# over the 18 steady points. Tolerance 1 %: SD1D's saved states are CVODE
# solutions at rtol 1e-5 stopped at a finite time with the PI integral still
# settling, and T_t is tabulated to 0.01 eV.
T_TOL = 0.01
FLUX_TOL = 0.01


def _solve(ny=100, n_up=3.0e19, **kw):
    params = DetachmentSolParameters(ny=ny, upstream_density=n_up, **kw)
    result = detachment_sol_run(params)
    assert result.converged, result.residual
    return params, result


def test_colored_block_jacobian_is_exact() -> None:
    params = DetachmentSolParameters(ny=20)
    grid = dsm._grid(params)
    u = jnp.stack(dsm.detachment_initial_state(params), axis=1)
    s, theta = jnp.asarray(0.1), dsm._theta(params)
    dense = jax.jacfwd(lambda uu: dsm._terms(uu, s, theta, params, grid).ddt.reshape(-1))(u).reshape(120, 120)
    lower, diag, upper, _ = dsm._jacobian_blocks(u, s, theta, params, grid)
    assembled = np.zeros((120, 120))
    for k in range(10):
        assembled[12 * k:12 * k + 12, 12 * k:12 * k + 12] = diag[k]
        if k > 0:
            assembled[12 * k:12 * k + 12, 12 * k - 12:12 * k] = lower[k]
        if k < 9:
            assembled[12 * k:12 * k + 12, 12 * k + 12:12 * k + 24] = upper[k]
    np.testing.assert_allclose(assembled, np.asarray(dense), rtol=0, atol=1e-14 * float(jnp.max(jnp.abs(dense))))


def test_charge_exchange_fit_matches_double_sum() -> None:
    temperatures = jnp.asarray([0.5, 3.0, 30.0, 300.0])
    log_t, log_e = np.log(np.asarray(temperatures)), np.log(10.0)
    direct = np.exp(sum(dsm._CX[i, j] * log_t**i * log_e**j for i in range(9) for j in range(9))) * 1e-6
    np.testing.assert_allclose(np.asarray(dsm.sd1d_charge_exchange_rate(temperatures)), direct, rtol=1e-12)


def test_sd1d_final_state_is_a_root_of_the_discrete_equations() -> None:
    # SD1D's own n_up = 2e19 final state (with its PI source amplitude) nearly
    # zeroes this module's residual; changing any one modelled ingredient
    # (ionisation cost, charge exchange, recycling fraction, neutral diffusion)
    # raises the residual by orders of magnitude.
    ref = np.load(DATA)
    u = jnp.stack([jnp.asarray(ref[f"n2e19_{k}"]) for k in ("Ne", "NVi", "P", "Nn", "NVn", "Pn")], axis=1)
    n_up = 2.0e19
    s = 5e-2 * (n_up / 1e20 - float(u[0, 0])) + 1e-3 * float(ref["n2e19_density_error_integral"])
    base = DetachmentSolParameters(ny=800, upstream_density=n_up)

    def residual(params):
        ddt = dsm._terms(u, s, dsm._theta(params), params, dsm._grid(params)).ddt
        return float(dsm._residual_norm(ddt, u, params))

    matched = residual(base)
    assert matched < 0.1
    for change in (dict(ionisation_energy=30.0), dict(charge_exchange=False),
                   dict(recycling_fraction=0.98), dict(neutral_diffusion=5.0)):
        assert residual(replace(base, **change)) > 300.0 * matched


def test_steady_state_ledgers_close() -> None:
    params, result = _solve()
    assert result.residual < 1e-8
    ledger = detachment_ledger(result, params)
    # Particles: source + net ionisation = target flux (plasma), and the
    # external source replaces exactly the unrecycled fraction (plasma + neutrals).
    assert abs(ledger["particle_imbalance"]) < 1e-8
    assert abs(ledger["global_particle_imbalance"]) < 1e-8
    # Power: input = (3/2) P V A at the target face + radiation/ionisation cost
    # + net transfer to neutrals + compression work (all from the discrete terms).
    assert abs(ledger["power_imbalance"]) < 1e-8
    assert ledger["radiation"] > 0.0 and ledger["target_advected_power"] > 0.0


def test_one_sd1d_point_at_full_resolution() -> None:
    n_up, t_ref, flux_ref = SD1D_13P6[9]
    params, result = _solve(ny=800, n_up=n_up)
    diag = detachment_diagnostics(result.state, params, result.source_scale)
    assert float(diag.target_temperature_ev) == pytest.approx(t_ref, rel=T_TOL)
    assert float(diag.target_ion_flux_area) == pytest.approx(flux_ref, rel=FLUX_TOL)


def test_implicit_gradient_matches_central_differences() -> None:
    params, result = _solve(ny=100)
    theta = jnp.asarray([3.0e19, 5.0e7])

    def outputs(th):
        return detachment_target_outputs(th, params, result)

    jac = np.asarray(jax.jacfwd(outputs)(theta))
    grad_t = np.asarray(jax.grad(lambda th: outputs(th)[0])(theta))
    np.testing.assert_allclose(grad_t, jac[0], rtol=1e-10)
    errors = []
    for h in (1.0e17, 3.0e16):
        e = np.array([h, 0.0])
        fd = (np.asarray(outputs(theta + e)) - np.asarray(outputs(theta - e))) / (2 * h)
        errors.append(np.max(np.abs(fd / jac[:, 0] - 1.0)))
    assert errors[-1] < 1e-4
    assert errors[-1] < errors[0]  # second-order approach
    e = np.array([0.0, 1.5e5])
    fd = (np.asarray(outputs(theta + e)) - np.asarray(outputs(theta - e))) / 3.0e5
    np.testing.assert_allclose(fd, jac[:, 1], rtol=1e-4)
    assert jac[0, 0] < 0.0 and jac[0, 1] > 0.0  # denser -> colder, more power -> hotter


@pytest.mark.slow
def test_sd1d_13p6ev_scan_target_values() -> None:
    previous = None
    for n_up, t_ref, flux_ref in SD1D_13P6:
        params = DetachmentSolParameters(ny=800, upstream_density=n_up)
        result = detachment_sol_run(params, previous.state if previous else None,
                                    source_scale=previous.source_scale if previous else 0.1)
        assert result.converged
        previous = result
        diag = detachment_diagnostics(result.state, params, result.source_scale)
        assert float(diag.target_temperature_ev) == pytest.approx(t_ref, rel=T_TOL)
        assert float(diag.target_ion_flux_area) == pytest.approx(flux_ref, rel=FLUX_TOL)


@pytest.mark.slow
@pytest.mark.parametrize("tag,n_up", [("n2e19", 2.000355e19), ("n5e19", 5.000658e19)])
def test_sd1d_profiles(tag, n_up) -> None:
    ref = np.load(DATA)
    params, result = _solve(ny=800, n_up=n_up)
    st = [np.asarray(f) for f in result.state]
    ne, nvi, p, nn = (ref[f"{tag}_{k}"] for k in ("Ne", "NVi", "P", "Nn"))
    np.testing.assert_allclose(st[0], ne, rtol=0.02)
    np.testing.assert_allclose(0.5 * st[2] / st[0], 0.5 * p / ne, rtol=0.02)
    mach, mach_ref = st[1] / st[0] / np.sqrt(st[2] / st[0]), nvi / ne / np.sqrt(p / ne)
    np.testing.assert_allclose(mach, mach_ref, atol=0.01)
    # neutral density: compare where it matters (above 1e-3 of its peak)
    mask = nn > 1e-3 * nn.max()
    np.testing.assert_allclose(st[3][mask], nn[mask], rtol=0.05)


@pytest.mark.slow
def test_grid_convergence_of_target_values() -> None:
    for n_up in (2.0e19, 5.0e19):
        previous, prev_params, temps = None, None, []
        for ny in (100, 200, 400, 800):
            params = DetachmentSolParameters(ny=ny, upstream_density=n_up)
            result = detachment_sol_run(params, previous.state if previous else None,
                                        source_scale=previous.source_scale if previous else 0.1,
                                        source_params=prev_params)
            assert result.converged
            previous, prev_params = result, params
            temps.append(float(detachment_diagnostics(result.state, params).target_temperature_ev))
        steps = np.abs(np.diff(temps))
        assert np.all(steps[1:] < steps[:-1] * 1.01) or steps[-1] < 0.05 * temps[-1]
        assert steps[-1] / temps[-1] < 0.10
