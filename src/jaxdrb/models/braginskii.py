from __future__ import annotations

import jax.numpy as jnp
from jax import nn


def smooth_floor(x: jnp.ndarray, *, floor: float, width: float) -> jnp.ndarray:
    """Return a smooth, positive-clamped version of x.

    Uses a softplus transition so gradients remain well-defined.
    """

    x = jnp.asarray(x)
    w = jnp.asarray(width, dtype=x.dtype)
    f = jnp.asarray(floor, dtype=x.dtype)
    w = jnp.maximum(w, jnp.asarray(1e-30, dtype=x.dtype))
    return f + w * nn.softplus((x - f) / w)


def _Te0_eff(params, eq) -> jnp.ndarray:
    Te0 = jnp.asarray(eq.Te0, dtype=jnp.float64)
    floor = float(getattr(params, "braginskii_T_floor", 1e-3))
    width = float(getattr(params, "braginskii_T_smooth", 1e-3))
    return smooth_floor(Te0, floor=floor, width=width)


def _Ti0_eff(params, eq) -> jnp.ndarray:
    # Hot-ion model currently uses Ti0 = tau_i * Te0 as the equilibrium ion temperature.
    tau_i = float(getattr(params, "tau_i", 0.0))
    return tau_i * _Te0_eff(params, eq)


def eta_parallel(params, eq) -> jnp.ndarray:
    """Parallel resistivity η (constant or Spitzer-like scaling)."""

    eta0 = jnp.asarray(getattr(params, "eta", 0.0), dtype=jnp.float64)
    if not bool(getattr(params, "braginskii_on", False)):
        return eta0
    if not bool(getattr(params, "braginskii_eta_on", True)):
        return eta0

    Te0 = _Te0_eff(params, eq)
    Tref = jnp.asarray(getattr(params, "braginskii_Tref", 1.0), dtype=jnp.float64)
    # Spitzer resistivity scaling: η ∝ T_e^{-3/2}. (Zeff, lnΛ can be folded into eta0.)
    return eta0 * (Tref / Te0) ** 1.5


def chi_par_Te(params, eq) -> jnp.ndarray:
    """Parallel electron heat conduction coefficient χ_||,e.

    In Braginskii/Spitzer-Härm scaling, κ_||,e ∝ T_e^{5/2}. In the reduced model we represent
    this by scaling a reference χ_||,e by (Te0/Tref)^{5/2}.
    """

    chi0 = jnp.asarray(getattr(params, "chi_par_Te", 0.0), dtype=jnp.float64)
    if not bool(getattr(params, "braginskii_on", False)):
        return chi0
    if not bool(getattr(params, "braginskii_kappa_e_on", True)):
        return chi0

    Te0 = _Te0_eff(params, eq)
    Tref = jnp.asarray(getattr(params, "braginskii_Tref", 1.0), dtype=jnp.float64)
    return chi0 * (Te0 / Tref) ** 2.5


def chi_par_Ti(params, eq) -> jnp.ndarray:
    """Parallel ion heat conduction coefficient χ_||,i (hot-ion model)."""

    chi0 = jnp.asarray(getattr(params, "chi_par_Ti", 0.0), dtype=jnp.float64)
    if not bool(getattr(params, "braginskii_on", False)):
        return chi0
    if not bool(getattr(params, "braginskii_kappa_i_on", True)):
        return chi0

    Ti0 = _Ti0_eff(params, eq)
    Tref = jnp.asarray(getattr(params, "braginskii_Tref", 1.0), dtype=jnp.float64)
    return chi0 * (Ti0 / Tref) ** 2.5


def nu_par_e(params, eq) -> jnp.ndarray:
    """Parallel electron flow diffusion/viscosity coefficient (placeholder scaling)."""

    nu0 = jnp.asarray(getattr(params, "nu_par_e", 0.0), dtype=jnp.float64)
    if not bool(getattr(params, "braginskii_on", False)):
        return nu0
    if not bool(getattr(params, "braginskii_visc_e_on", True)):
        return nu0

    Te0 = _Te0_eff(params, eq)
    Tref = jnp.asarray(getattr(params, "braginskii_Tref", 1.0), dtype=jnp.float64)
    return nu0 * (Te0 / Tref) ** 2.5


def nu_par_i(params, eq) -> jnp.ndarray:
    """Parallel ion flow diffusion/viscosity coefficient (placeholder scaling)."""

    nu0 = jnp.asarray(getattr(params, "nu_par_i", 0.0), dtype=jnp.float64)
    if not bool(getattr(params, "braginskii_on", False)):
        return nu0
    if not bool(getattr(params, "braginskii_visc_i_on", True)):
        return nu0

    # In the hot-ion model, use Ti0 for the ion scaling; otherwise fall back to Te0.
    tau_i = float(getattr(params, "tau_i", 0.0))
    T0 = _Ti0_eff(params, eq) if tau_i > 0.0 else _Te0_eff(params, eq)
    Tref = jnp.asarray(getattr(params, "braginskii_Tref", 1.0), dtype=jnp.float64)
    return nu0 * (T0 / Tref) ** 2.5
