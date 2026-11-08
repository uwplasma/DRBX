"""Hydrogenic atomic reaction-rate coefficients (self-contained, pure JAX).

Ionization and recombination rate coefficients ``<sigma v>(T_e, n_e)`` come from
the packaged AMJUEL double-polynomial fits; the charge-exchange coefficient
``<sigma v>(T_eff)`` comes from the AMJUEL H.2 3.1.8 one-dimensional polynomial
(inlined here, as hermes-3 does, since it carries no tabulated file). Every
routine is pure ``jax.numpy`` and therefore ``jit``/``grad``/``vmap``
transparent; the coefficient tables ship with the package under
``drbx.data.atomic_rates`` (real published AMJUEL fits, verified against the
hermes-3 database provenance), so there is no runtime external dependency.

Temperatures are in eV and densities in m^-3; rate coefficients are returned in
m^3 / s. The hydrogen fits (H.4/H.10 2.1.5 and 2.1.8, H.2 3.1.8) are those of
the AMJUEL report version of January 13, 2020; the 2011 version differs by up
to ~5% in recombination. Inputs are clamped to ``T in [0.1, 1e4] eV`` and
``n in [1e14, 1e22] m^-3`` (the 2020 ionisation/recombination fits extend to
2e4 eV; the density range is the report's 1e8-1e16 cm^-3).
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

import jax.numpy as jnp
import numpy as np

__all__ = [
    "rate_coefficient",
    "energy_loss_coefficient",
    "charge_exchange_rate_coefficient",
    "eval_amjuel_fit",
    "load_amjuel_coefficients",
    "sd1d_charge_exchange_rate",
    "sd1d_ionisation_rate",
    "sd1d_recombination_rate",
]

# AMJUEL fit files, keyed by (species, reaction). Deuterium and tritium share the
# hydrogen fits; helium has its own.
_AMJUEL_FILENAMES = {
    ("d", "iz"): "iz_AMJUEL_H.x_2.1.5.json",
    ("d", "rec"): "rec_AMJUEL_H.x_2.1.8.json",
    ("t", "iz"): "iz_AMJUEL_H.x_2.1.5.json",
    ("t", "rec"): "rec_AMJUEL_H.x_2.1.8.json",
    ("he", "iz"): "iz_AMJUEL_H.x_2.3.9a.json",
    ("he", "rec"): "rec_AMJUEL_H.x_2.3.13a.json",
}

# AMJUEL H.2 3.1.8 charge-exchange polynomial: ln(sigma v [cm^3/s]) as a degree-8
# polynomial in ln(T_eff [eV]) (density independent), as inlined in hermes-3's
# hydrogen_charge_exchange.cxx.
_CX_CONSTANT = -18.5028
_CX_COEFFICIENTS = (
    0.3708409,
    7.949876e-3,
    -6.143769e-4,
    -4.698969e-4,
    -4.096807e-4,
    1.440382e-4,
    -1.514243e-5,
    5.122435e-7,
)


@lru_cache(maxsize=None)
def load_amjuel_coefficients(species: str, reaction: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load one AMJUEL fit: ``(sigma_v_coeffs, sigma_v_E_coeffs, electron_heating)``.

    ``sigma_v_coeffs`` and ``sigma_v_E_coeffs`` are 9x9 log-log polynomial
    coefficient matrices; ``electron_heating`` is the potential energy [eV]
    returned to the electrons (e.g. 13.6 eV on recombination).
    """

    filename = _AMJUEL_FILENAMES[(species, reaction)]
    payload = json.loads(
        resources.files("drbx.data.atomic_rates").joinpath(filename).read_text(encoding="utf-8")
    )
    return (
        np.asarray(payload["sigma_v_coeffs"], dtype=np.float64),
        np.asarray(payload["sigma_v_E_coeffs"], dtype=np.float64),
        float(payload["electron_heating"]),
    )


def eval_amjuel_fit(temperature_ev, density_m3, coefficients) -> jnp.ndarray:
    """Evaluate an AMJUEL double-polynomial fit, returning ``<sigma v>`` in m^3/s.

    ``<sigma v> = 1e-6 * exp( sum_ij c_ij (ln T)^i (ln(n/1e14))^j )`` with ``T``
    and ``n`` clamped to the fit's validity range.
    """

    temperature = jnp.clip(jnp.asarray(temperature_ev, dtype=jnp.float64), 0.1, 1.0e4)
    density = jnp.clip(jnp.asarray(density_m3, dtype=jnp.float64), 1.0e14, 1.0e22)
    log_t = jnp.log(temperature)
    log_n = jnp.log(density / 1.0e14)

    coeffs = jnp.asarray(coefficients, dtype=jnp.float64)
    result = jnp.zeros_like(log_t)
    for row in coeffs[::-1]:
        row_result = jnp.zeros_like(log_n)
        for coefficient in row[::-1]:
            row_result = row_result * log_n + coefficient
        result = result * log_t + row_result
    return jnp.exp(result) * 1.0e-6


def rate_coefficient(species: str, reaction: str, electron_temperature, electron_density) -> jnp.ndarray:
    """Ionization (``reaction="iz"``) or recombination (``"rec"``) ``<sigma v>`` [m^3/s]."""

    sigma_v_coeffs, _, _ = load_amjuel_coefficients(species, reaction)
    return eval_amjuel_fit(electron_temperature, electron_density, sigma_v_coeffs)


def energy_loss_coefficient(species: str, reaction: str, electron_temperature, electron_density) -> jnp.ndarray:
    """Electron energy-loss coefficient ``<sigma v E>`` [eV m^3/s] for the reaction.

    The recombination channel returns the ``electron_heating`` potential energy
    (13.6 eV for hydrogen) as a heating term, so the net electron energy sink is
    ``<sigma v E> - electron_heating * <sigma v>``; ionization has no potential
    add-back (``electron_heating = 0``).
    """

    sigma_v_coeffs, sigma_v_E_coeffs, electron_heating = load_amjuel_coefficients(species, reaction)
    energy_loss = eval_amjuel_fit(electron_temperature, electron_density, sigma_v_E_coeffs)
    rate = eval_amjuel_fit(electron_temperature, electron_density, sigma_v_coeffs)
    return energy_loss - float(electron_heating) * rate


def charge_exchange_rate_coefficient(effective_temperature_ev) -> jnp.ndarray:
    """Hydrogenic charge-exchange ``<sigma v>(T_eff)`` [m^3/s] (AMJUEL H.2 3.1.8)."""

    log_t = jnp.log(jnp.clip(jnp.asarray(effective_temperature_ev, dtype=jnp.float64), 0.1, 1.0e4))
    log_sigma_v = jnp.full_like(log_t, _CX_CONSTANT)
    power = jnp.ones_like(log_t)
    for coefficient in _CX_COEFFICIENTS:
        power = power * log_t
        log_sigma_v = log_sigma_v + coefficient * power
    return jnp.exp(log_sigma_v) * 1.0e-6


# ----------------------------------------------------------------------------
# SD1D ``UpdatedRadiatedPower`` hydrogen rates [m^3/s] (radiation.cxx): the
# parity set of the 1D detachment model (drbx.native.neutrals.detachment_sol_model).
# They differ from AMJUEL H.4 2.1.5/2.1.8 (ionisation 0.3-0.8x, recombination
# 0.7-1.1x, CX at fixed E = 10 eV vs H.2 3.1.8), so the SD1D 1 % comparison is
# pinned to them; the FCI path uses the AMJUEL functions above.

_IZ = (-3.271397e1, 1.353656e1, -5.739329, 1.563155, -2.877056e-1, 3.482560e-2,
       -2.631976e-3, 1.119544e-4, -2.039150e-6)

_REC = np.array([
    [-2.855728479302E+01, -7.664042607917E-01, -4.930424003280E-03, -5.386830982777E-03, -1.626039237665E-04, 6.080907650243E-06, 2.101102051942E-05, -2.770717597683E-06, 1.038235939800E-07],
    [3.488563234375E-02, -3.583233366133E-03, -3.620245352252E-03, -9.532840484460E-04, 1.888048628708E-04, -1.014890683861E-05, 2.245676563601E-05, -4.695982369246E-06, 2.523166611507E-07],
    [-2.799644392058E-02, -7.452514292790E-03, 6.958711963182E-03, 4.631753807534E-04, 1.288577690147E-04, -1.145028889459E-04, -2.245624273814E-06, 3.250878872873E-06, -2.145390398476E-07],
    [1.209545317879E-02, 2.709299760454E-03, -2.139257298118E-03, -5.371179699661E-04, -1.634580516353E-05, 5.942193980802E-05, -2.944873763540E-06, -9.387290785993E-07, 7.381435237585E-08],
    [-2.436630799820E-03, -7.745129766167E-04, 4.603883706734E-04, 1.543350502150E-04, -9.601036952725E-06, -1.211851723717E-05, 1.002105099354E-06, 1.392391630459E-07, -1.299713684966E-08],
    [2.837893719800E-04, 1.142444698207E-04, -5.991636837395E-05, -2.257565836876E-05, 3.425262385387E-06, 1.118965496365E-06, -1.291320799814E-07, -1.139093288575E-08, 1.265189576423E-09],
    [-1.886511169084E-05, -9.382783518064E-06, 4.729262545726E-06, 1.730782954588E-06, -4.077019941998E-07, -4.275321573501E-08, 7.786155463269E-09, 5.178505597480E-10, -6.854203970018E-11],
    [6.752155602894E-07, 3.902800099653E-07, -1.993485395689E-07, -6.618240780594E-08, 2.042041097083E-08, 3.708616111085E-10, -2.441127783437E-10, -9.452402157390E-12, 1.836615031798E-12],
    [-1.005893858779E-08, -6.387411585521E-09, 3.352589865190E-09, 1.013364275013E-09, -3.707977721109E-10, 7.068450112690E-12, 3.773208484020E-12, -4.672724022059E-14, -1.640492364811E-14],
])  # MATA[j][i]: j = density power, i = temperature power

_CX = np.array([
    [-1.829079582E1, 1.640252721E-1, 3.364564509E-2, 9.530225559E-3, -8.519413900E-4, -1.247583861E-3, 3.014307546E-4, -2.499323170E-5, 6.932627238E-7],
    [2.169137616E-1, -1.106722014E-1, -1.382158680E-3, 7.348786287E-3, -6.343059502E-4, -1.919569450E-4, 4.075019352E-5, -2.850044983E-6, 6.966822400E-8],
    [4.307131244E-2, 8.948693625E-3, -1.209480567E-2, -3.675019470E-4, 1.039643391E-3, -1.553840718E-4, 2.670827249E-6, 7.695300598E-7, -3.783302282E-8],
    [-5.754895093E-4, 6.062141761E-3, 1.075907882E-3, -8.119301728E-4, 8.911036876E-6, 3.175388950E-5, -4.515123642E-6, 2.187439284E-7, -2.911233952E-9],
    [-1.552077120E-3, -1.210431588E-3, 8.297212634E-4, 1.361661817E-4, -1.008928628E-4, 1.080693990E-5, 5.106059414E-7, -1.299275586E-7, 5.117133050E-9],
    [-1.876800283E-4, -4.052878752E-5, -1.907025663E-4, 1.141663042E-5, 1.775681984E-5, -3.149286924E-6, 3.105491555E-8, 2.274394089E-8, -1.130988251E-9],
    [1.125490271E-4, 2.875900436E-5, 1.338839629E-5, -4.340802793E-6, -7.003521917E-7, 2.318308730E-7, -6.030983538E-9, -1.755944926E-9, 1.005189187E-10],
    [-1.238982763E-5, -2.616998140E-6, -1.171762874E-7, 3.517971869E-7, -4.928692833E-8, 1.756388999E-10, -1.446756796E-10, 7.143183138E-11, -3.989884106E-12],
    [4.163596197E-7, 7.558092849E-8, -1.328404104E-8, -9.170850254E-9, 3.208853884E-9, -3.952740759E-10, 2.739558476E-11, -1.693040209E-12, 6.388219930E-14],
])  # cxcoeffs[i][j]: i = ln T power, j = ln E power
_CX_T = tuple(float(c) for c in _CX @ (np.log(10.0) ** np.arange(9)))  # E = 10 eV


def _poly(coeffs, x):
    out = jnp.zeros_like(x)
    for c in reversed(coeffs):
        out = out * x + c
    return out


def sd1d_ionisation_rate(temperature_ev):
    """SD1D ionisation ``<sigma v>`` [m^3/s] (T floored at 0.025 eV)."""
    t = jnp.maximum(temperature_ev, 0.025)
    return jnp.exp(_poly(_IZ, jnp.log(t))) * 1.0e-6


def sd1d_charge_exchange_rate(temperature_ev):
    """SD1D charge-exchange ``<sigma v>`` [m^3/s] at E = 10 eV."""
    t = jnp.maximum(temperature_ev, 0.025)
    return jnp.exp(_poly(_CX_T, jnp.log(t))) * 1.0e-6


def sd1d_recombination_rate(density_m3, temperature_ev):
    """SD1D recombination ``<sigma v>`` [m^3/s] (zero for n < 1e3 m^-3)."""
    t = jnp.maximum(temperature_ev, 0.025)
    n = jnp.maximum(density_m3, 1.0e3)
    rn = jnp.log(n * 1.0e-14)
    rt = jnp.log(t)
    tpows = [jnp.ones_like(rt)]
    for _ in range(8):
        tpows.append(tpows[-1] * rt)
    suma = jnp.zeros_like(rt)
    npow = jnp.ones_like(rn)
    for j in range(9):
        row = sum(float(_REC[j, i]) * tpows[i] for i in range(9))
        suma = suma + row * npow
        npow = npow * rn
    hav = jnp.exp(suma) * 1.0e-6 / (1.0 + 0.125 * t)
    ry, chi, a = 13.60569, 0.35, 3.92e-20
    b = 3.0e-124 * 1.6e-19 ** (-4.5)
    rad = a * ry**1.5 / (jnp.sqrt(t) * (ry + chi * t))
    return jnp.where(density_m3 < 1.0e3, 0.0, hav + rad + b * n * t ** (-5.0))
