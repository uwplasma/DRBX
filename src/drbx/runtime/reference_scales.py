"""Physical reference scales of the single-length rho-star normalization.

The FCI drift-reduced Braginskii right-hand sides carry one dimensionless
parameter, ``rho_star = rho_s0 / L_ref``, with

    rho_s0 = sqrt(m_i T_e0) / (e B0)        [m]
    c_s0   = sqrt(e T_e0 / m_i)             [m/s]
    t_ref  = L_ref / c_s0                   [s]

(``T_e0`` in eV, ``B0`` in tesla).  The HSX geometry artifacts are in metres
(``L_ref = 1 m``) and normalize ``B`` by ``B0``, the field the producer
records as ``reference_magnetic_field_tesla``.

This module is pure Python (no JAX) and sits in ``drbx.runtime`` rather than
next to ``FciDrbEBRhsParameters`` so that drivers, analysis scripts and tests
can derive the scales without importing the solver stack, and so that the
parameter module stays free of unit conversions.  Constants are CODATA 2018.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

#: Elementary charge [C] (exact, SI 2019).
ELEMENTARY_CHARGE_C = 1.602176634e-19
#: Unified atomic mass unit [kg] (CODATA 2018).
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
#: Proton mass in unified atomic mass units (CODATA 2018, rounded to the
#: precision used by the HSX studies); the default ion is hydrogen.
HYDROGEN_ION_MASS_AMU = 1.00728
#: Deuteron mass in unified atomic mass units (CODATA 2018).
DEUTERIUM_ION_MASS_AMU = 2.013553212745


@dataclass(frozen=True)
class ReferenceScales:
    """Inputs and derived physical reference scales (SI units)."""

    te_ev: float
    b0_tesla: float
    ion_mass_amu: float
    l_ref_m: float
    ion_mass_kg: float
    rho_s0_m: float
    c_s0: float
    t_ref: float
    rho_star: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serializable mapping of every field."""

        return {key: float(value) for key, value in asdict(self).items()}


def _positive_finite(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real number, got {value!r}") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return number


def reference_rho_star(
    te_ev: float,
    b0_tesla: float,
    ion_mass_amu: float = HYDROGEN_ION_MASS_AMU,
    l_ref_m: float = 1.0,
) -> ReferenceScales:
    """Return ``rho_star = rho_s0 / L_ref`` with ``c_s0`` and ``t_ref``.

    Args:
        te_ev: Reference electron temperature ``T_e0`` [eV].
        b0_tesla: Reference magnetic field ``B0`` [T].
        ion_mass_amu: Ion mass [u]; the default is hydrogen (the proton).
        l_ref_m: Reference length ``L_ref`` [m]; the HSX geometry is in
            metres, so the default is 1 m.

    Returns:
        A :class:`ReferenceScales` with ``rho_star`` (dimensionless),
        ``c_s0`` [m/s], ``t_ref`` [s] and ``rho_s0_m`` [m].
    """

    te = _positive_finite("te_ev", te_ev)
    b0 = _positive_finite("b0_tesla", b0_tesla)
    amu = _positive_finite("ion_mass_amu", ion_mass_amu)
    length = _positive_finite("l_ref_m", l_ref_m)
    mass = amu * ATOMIC_MASS_UNIT_KG
    c_s0 = math.sqrt(ELEMENTARY_CHARGE_C * te / mass)
    rho_s0 = mass * c_s0 / (ELEMENTARY_CHARGE_C * b0)
    return ReferenceScales(
        te_ev=te,
        b0_tesla=b0,
        ion_mass_amu=amu,
        l_ref_m=length,
        ion_mass_kg=mass,
        rho_s0_m=rho_s0,
        c_s0=c_s0,
        t_ref=length / c_s0,
        rho_star=rho_s0 / length,
    )


__all__ = [
    "ATOMIC_MASS_UNIT_KG",
    "DEUTERIUM_ION_MASS_AMU",
    "ELEMENTARY_CHARGE_C",
    "HYDROGEN_ION_MASS_AMU",
    "ReferenceScales",
    "reference_rho_star",
]
