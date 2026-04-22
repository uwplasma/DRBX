from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from ..config.boutinp import BoutConfig


_QE = 1.602176634e-19
_EPS0 = 8.8541878128e-12
_MP = 1.67262192369e-27
_ME = 9.1093837015e-31


@dataclass(frozen=True)
class IonParallelViscosityInputs:
    total_collisionality: np.ndarray
    tau: np.ndarray
    eta: np.ndarray


def electron_density(ions: tuple[Any, ...]) -> np.ndarray:
    density = np.zeros_like(ions[0].density, dtype=np.float64)
    for ion in ions:
        density = density + ion.charge * ion.density
    return density


def prepared_electron_density(
    ions: tuple[Any, ...],
    prepared: Mapping[str, Any],
) -> np.ndarray:
    density = np.zeros_like(prepared[ions[0].name].density, dtype=np.float64)
    for ion in ions:
        density = density + ion.charge * np.asarray(prepared[ion.name].density, dtype=np.float64)
    return density


def compute_collision_frequencies(
    config: BoutConfig,
    species: Mapping[str, Any],
    prepared: Mapping[str, Any],
    *,
    dataset_scalars: Mapping[str, float],
) -> dict[tuple[str, str], np.ndarray]:
    collision_rates: dict[tuple[str, str], np.ndarray] = {}
    nnorm = float(dataset_scalars["Nnorm"])
    tnorm = float(dataset_scalars["Tnorm"])
    rho_s0 = float(dataset_scalars["rho_s0"])
    omega_ci = float(dataset_scalars["Omega_ci"])
    electron_ion = bool(config.parsed("braginskii_collisions", "electron_ion")) if config.has_option("braginskii_collisions", "electron_ion") else True
    electron_electron = bool(config.parsed("braginskii_collisions", "electron_electron")) if config.has_option("braginskii_collisions", "electron_electron") else True
    electron_neutral = bool(config.parsed("braginskii_collisions", "electron_neutral")) if config.has_option("braginskii_collisions", "electron_neutral") else False
    ion_ion = bool(config.parsed("braginskii_collisions", "ion_ion")) if config.has_option("braginskii_collisions", "ion_ion") else True
    ion_neutral = bool(config.parsed("braginskii_collisions", "ion_neutral")) if config.has_option("braginskii_collisions", "ion_neutral") else False
    neutral_neutral = bool(config.parsed("braginskii_collisions", "neutral_neutral")) if config.has_option("braginskii_collisions", "neutral_neutral") else True

    electron = species["e"]
    electron_state = prepared["e"]
    te_ev = electron_state.temperature * tnorm
    ne_m3 = electron_state.density * nnorm

    if electron_electron:
        te_limited = np.maximum(te_ev, 0.1)
        ne_limited = np.maximum(ne_m3, 1.0e10)
        log_te = np.log(te_limited)
        coulomb_log = 30.4 - 0.5 * np.log(ne_limited) + 1.25 * log_te - np.sqrt(1.0e-5 + np.square(log_te - 2.0) / 16.0)
        v1sq = 2.0 * te_limited * _QE / _ME
        nu_ee = (
            (_QE**4)
            * np.maximum(ne_m3, 0.0)
            * np.maximum(coulomb_log, 1.0)
            * 2.0
            / (3.0 * np.power(math.pi * 2.0 * v1sq, 1.5) * ((_EPS0 * _ME) ** 2))
        )
        collision_rates[("e", "e")] = np.asarray(nu_ee / omega_ci, dtype=np.float64)

    for species_name, sp in species.items():
        if not electron_ion or species_name == "e" or sp.charge <= 0.0:
            continue
        state = prepared[species_name]
        ti_ev = state.temperature * tnorm
        ni_m3 = state.density * nnorm
        zi = sp.charge
        ai = sp.atomic_mass
        me_mi = _ME / (_MP * ai)

        te_limited = np.maximum(te_ev, 0.1)
        ti_limited = np.maximum(ti_ev, 0.1)
        ne_limited = np.maximum(ne_m3, 1.0e10)
        ni_limited = np.maximum(ni_m3, 1.0e10)
        mask_very_low = (te_ev < 0.1) | (ni_m3 < 1.0e10) | (ne_m3 < 1.0e10)
        mask_low_te = te_ev < (ti_ev * me_mi)
        mask_mid_te = te_ev < (math.exp(2.0) * zi * zi)
        coulomb_log = np.where(
            mask_very_low,
            10.0,
            np.where(
                mask_low_te,
                23.0 - 0.5 * np.log(ni_limited) + 1.5 * np.log(ti_limited) - np.log((zi * zi) * ai),
                np.where(
                    mask_mid_te,
                    30.0 - 0.5 * np.log(ne_limited) - np.log(zi) + 1.5 * np.log(te_limited),
                    31.0 - 0.5 * np.log(ne_limited) + np.log(te_limited),
                ),
            ),
        )
        vesq = 2.0 * te_limited * _QE / _ME
        visq = 2.0 * ti_limited * _QE / (_MP * ai)
        nu_ei = (
            (((_QE * _QE) * zi) ** 2)
            * np.maximum(ni_m3, 0.0)
            * np.maximum(coulomb_log, 1.0)
            * (1.0 + me_mi)
            / (3.0 * np.power(math.pi * (vesq + visq), 1.5) * ((_EPS0 * _ME) ** 2))
        )
        nu_ei = np.asarray(nu_ei / omega_ci, dtype=np.float64)
        collision_rates[("e", species_name)] = nu_ei
        collision_rates[(species_name, "e")] = (
            nu_ei
            * (electron.atomic_mass / sp.atomic_mass)
            * prepared["e"].density
            / np.maximum(state.density, 1.0e-5)
        )

    for neutral_name, neutral_species in species.items():
        if not electron_neutral or neutral_name == "e" or neutral_species.charge != 0.0:
            continue
        neutral_state = prepared[neutral_name]
        vth_e = np.sqrt((_MP / _ME) * np.maximum(prepared["e"].temperature, 0.0))
        nu_en = vth_e * nnorm * neutral_state.density * 5.0e-19 * rho_s0
        nu_en = np.asarray(nu_en, dtype=np.float64)
        collision_rates[("e", neutral_name)] = nu_en
        collision_rates[(neutral_name, "e")] = (
            nu_en
            * (electron.atomic_mass / neutral_species.atomic_mass)
            * prepared["e"].density
            / np.maximum(neutral_state.density, 1.0e-5)
        )

    names = tuple(sorted(name for name in species if name != "e"))

    def collide(name1: str, name2: str, nu_12: np.ndarray) -> None:
        first_species = species[name1]
        second_species = species[name2]
        first_state = prepared[name1]
        second_state = prepared[name2]
        nu_12 = np.asarray(nu_12, dtype=np.float64)
        collision_rates[(name1, name2)] = nu_12
        if name1 == name2:
            return
        collision_rates[(name2, name1)] = (
            nu_12
            * (first_species.atomic_mass / second_species.atomic_mass)
            * first_state.density
            / np.maximum(second_state.density, 1.0e-5)
        )

    for index, first_name in enumerate(names):
        first_species = species[first_name]
        first_state = prepared[first_name]
        t1_ev = first_state.temperature * tnorm
        n1_m3 = first_state.density * nnorm
        first_charged = first_species.charge != 0.0

        for second_name in names[index:]:
            second_species = species[second_name]
            second_state = prepared[second_name]
            t2_ev = second_state.temperature * tnorm
            n2_m3 = second_state.density * nnorm
            second_charged = second_species.charge != 0.0

            if first_charged:
                if second_charged:
                    if not ion_ion:
                        continue
                    z1 = first_species.charge
                    z2 = second_species.charge
                    a1 = first_species.atomic_mass
                    a2 = second_species.atomic_mass
                    m1 = a1 * _MP
                    m2 = a2 * _MP
                    t1_limited = np.maximum(t1_ev, 0.1)
                    t2_limited = np.maximum(t2_ev, 0.1)
                    n1_limited = np.maximum(n1_m3, 1.0e10)
                    n2_limited = np.maximum(n2_m3, 1.0e10)
                    coulomb_log = 29.91 - np.log(
                        ((z1 * z2 * (a1 + a2)) / (a1 * t2_limited + a2 * t1_limited))
                        * np.sqrt(n1_limited * (z1 * z1) / t1_limited + n2_limited * (z2 * z2) / t2_limited)
                    )
                    v1sq = 2.0 * t1_limited * _QE / m1
                    v2sq = 2.0 * t2_limited * _QE / m2
                    nu_12 = (
                        (((z1 * _QE) * (z2 * _QE)) ** 2)
                        * n2_limited
                        * np.maximum(coulomb_log, 1.0)
                        * (1.0 + (m1 / m2))
                        / (3.0 * np.power(math.pi * (v1sq + v2sq), 1.5) * ((_EPS0 * m1) ** 2))
                    )
                    collide(first_name, second_name, nu_12 / omega_ci)
                else:
                    if not ion_neutral:
                        continue
                    vrel = np.sqrt(
                        np.maximum(
                            first_state.temperature / first_species.atomic_mass
                            + second_state.temperature / second_species.atomic_mass,
                            0.0,
                        )
                    )
                    collide(first_name, second_name, vrel * second_state.density * nnorm * 5.0e-19 * rho_s0)
            else:
                if second_charged:
                    if not ion_neutral:
                        continue
                    vrel = np.sqrt(
                        np.maximum(
                            first_state.temperature / first_species.atomic_mass
                            + second_state.temperature / second_species.atomic_mass,
                            0.0,
                        )
                    )
                    collide(first_name, second_name, vrel * second_state.density * nnorm * 5.0e-19 * rho_s0)
                else:
                    if not neutral_neutral:
                        continue
                    vrel = np.sqrt(
                        np.maximum(
                            first_state.temperature / first_species.atomic_mass
                            + second_state.temperature / second_species.atomic_mass,
                            0.0,
                        )
                    )
                    collide(first_name, second_name, vrel * second_state.density * nnorm * (math.pi * (2.8e-10**2)) * rho_s0)
    return collision_rates


def ion_parallel_viscosity_inputs(
    *,
    species_name: str,
    species: Mapping[str, Any],
    prepared: Mapping[str, Any],
    collision_rates: Mapping[tuple[str, str], np.ndarray],
    cx_rates: Mapping[str, np.ndarray],
) -> IonParallelViscosityInputs:
    total_collisionality = np.zeros_like(prepared[species_name].density, dtype=np.float64)
    for other_name in species:
        rate = collision_rates.get((species_name, other_name))
        if rate is not None:
            total_collisionality = total_collisionality + rate
    if species_name in cx_rates:
        total_collisionality = total_collisionality + cx_rates[species_name]
    total_collisionality = np.maximum(total_collisionality, 1.0e-12)
    tau = 1.0 / total_collisionality
    eta = 1.28 * np.asarray(prepared[species_name].pressure, dtype=np.float64) * tau
    return IonParallelViscosityInputs(
        total_collisionality=total_collisionality,
        tau=tau,
        eta=eta,
    )
