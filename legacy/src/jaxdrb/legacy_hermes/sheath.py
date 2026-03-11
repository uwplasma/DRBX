from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .primitives import limit_free


@dataclass(frozen=True)
class ParallelSheathState:
    """Hermes sheath guard-state reconstruction for open parallel ends.

    This mirrors the guard updates in
    `/Users/rogerio/local/hermes-3/src/sheath_boundary.cxx` for the variables
    needed by the strict parallel FV path:

    - density / pressure / temperature ghosts
    - electrostatic potential ghost
    - sheath midpoint velocities
    - velocity, momentum, and current ghosts

    The inputs and outputs use the active JAX `(nz, nx, ny)` layout.
    """

    n_ghost_low: jnp.ndarray
    n_ghost_high: jnp.ndarray
    Te_ghost_low: jnp.ndarray
    Te_ghost_high: jnp.ndarray
    Ti_ghost_low: jnp.ndarray
    Ti_ghost_high: jnp.ndarray
    pe_ghost_low: jnp.ndarray
    pe_ghost_high: jnp.ndarray
    pi_ghost_low: jnp.ndarray
    pi_ghost_high: jnp.ndarray
    phi_ghost_low: jnp.ndarray
    phi_ghost_high: jnp.ndarray
    ve_sheath_low: jnp.ndarray
    ve_sheath_high: jnp.ndarray
    vi_sheath_low: jnp.ndarray
    vi_sheath_high: jnp.ndarray
    ve_ghost_low: jnp.ndarray
    ve_ghost_high: jnp.ndarray
    vi_ghost_low: jnp.ndarray
    vi_ghost_high: jnp.ndarray
    nve_ghost_low: jnp.ndarray
    nve_ghost_high: jnp.ndarray
    nvi_ghost_low: jnp.ndarray
    nvi_ghost_high: jnp.ndarray
    j_ghost_low: jnp.ndarray
    j_ghost_high: jnp.ndarray
    mask: jnp.ndarray | None = None
    sign: jnp.ndarray | None = None


def _as_field(arr: jnp.ndarray | float, ref: jnp.ndarray) -> jnp.ndarray:
    out = jnp.asarray(arr, dtype=ref.dtype)
    return jnp.broadcast_to(out, ref.shape)


def _safe_temperature(temp: jnp.ndarray, floor: float) -> jnp.ndarray:
    return jnp.maximum(temp, jnp.asarray(floor, dtype=temp.dtype))


def _electron_side(
    *,
    sign: float,
    n_cell: jnp.ndarray,
    n_inner: jnp.ndarray,
    Te_cell: jnp.ndarray,
    Te_inner: jnp.ndarray,
    pe_cell: jnp.ndarray,
    pe_inner: jnp.ndarray,
    phi_cell: jnp.ndarray,
    phi_inner: jnp.ndarray,
    ve_cell: jnp.ndarray,
    nve_cell: jnp.ndarray,
    me_hat: float,
    secondary_electron_coef: float,
    wall_potential: jnp.ndarray,
    floor_potential: bool,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    n_ghost = limit_free(n_inner, n_cell, 0.0)
    Te_ghost = limit_free(Te_inner, Te_cell, 0.0)
    pe_ghost = limit_free(pe_inner, pe_cell, 0.0)
    phi_ghost = 2.0 * phi_cell - phi_inner

    n_sheath = 0.5 * (n_ghost + n_cell)
    Te_sheath = _safe_temperature(0.5 * (Te_ghost + Te_cell), 1.0e-10)
    phi_sheath = 0.5 * (phi_ghost + phi_cell)
    if floor_potential:
        phi_sheath = jnp.maximum(phi_sheath, wall_potential)

    ge = jnp.clip(float(secondary_electron_coef), 0.0, 1.0)
    prefactor = jnp.sqrt(Te_sheath / (2.0 * jnp.pi * float(me_hat)))
    exponent = jnp.exp(
        -(phi_sheath - wall_potential)
        / jnp.maximum(Te_sheath, jnp.asarray(1.0e-5, Te_sheath.dtype))
    )
    ve_sheath = float(sign) * (1.0 - ge) * prefactor * exponent
    ve_ghost = 2.0 * ve_sheath - ve_cell
    nve_ghost = 2.0 * float(me_hat) * n_sheath * ve_sheath - nve_cell
    return (
        n_ghost,
        Te_ghost,
        pe_ghost,
        phi_ghost,
        n_sheath,
        Te_sheath,
        ve_sheath,
        ve_ghost,
        nve_ghost,
    )


def _ion_side(
    *,
    ne_cell: jnp.ndarray,
    sign: float,
    ne_sheath: jnp.ndarray,
    Te_sheath: jnp.ndarray,
    n_cell: jnp.ndarray,
    n_inner: jnp.ndarray,
    Ti_cell: jnp.ndarray,
    Ti_inner: jnp.ndarray,
    pi_cell: jnp.ndarray,
    pi_inner: jnp.ndarray,
    vi_cell: jnp.ndarray,
    nvi_cell: jnp.ndarray,
    ion_mass: float,
    ion_charge: float,
    ion_adiabatic: float,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    n_ghost = limit_free(n_inner, n_cell, 0.0)
    Ti_ghost = limit_free(Ti_inner, Ti_cell, 0.0)
    pi_ghost = limit_free(pi_inner, pi_cell, 0.0)

    n_sheath = 0.5 * (n_ghost + n_cell)
    Ti_sheath = _safe_temperature(0.5 * (Ti_ghost + Ti_cell), 1.0e-5)
    s_i = jnp.clip(n_sheath / jnp.maximum(ne_sheath, 1.0e-10), 0.0, 1.0)

    grad_ne = ne_cell - ne_sheath
    grad_ni = n_cell - n_sheath
    small_grad = jnp.abs(grad_ni) < 1.0e-3
    grad_ne = jnp.where(small_grad, 1.0e-3, grad_ne)
    grad_ni = jnp.where(small_grad, 1.0e-3, grad_ni)

    ci_sq = jnp.clip(
        (float(ion_adiabatic) * Ti_sheath + float(ion_charge) * s_i * Te_sheath * grad_ne / grad_ni)
        / float(ion_mass),
        0.0,
        100.0,
    )
    vi_sheath = float(sign) * jnp.sqrt(ci_sq)
    vi_ghost = 2.0 * vi_sheath - vi_cell
    nvi_ghost = 2.0 * float(ion_mass) * n_sheath * vi_sheath - nvi_cell
    return n_ghost, Ti_ghost, pi_ghost, vi_sheath, vi_ghost, nvi_ghost, n_sheath


def build_parallel_sheath_state(
    *,
    n_e: jnp.ndarray,
    Te: jnp.ndarray,
    pe: jnp.ndarray,
    phi: jnp.ndarray,
    v_e: jnp.ndarray,
    n_i: jnp.ndarray,
    Ti: jnp.ndarray,
    pi: jnp.ndarray,
    v_i: jnp.ndarray,
    me_hat: float,
    ion_mass: float,
    ion_charge: float = 1.0,
    secondary_electron_coef: float = 0.0,
    wall_potential: jnp.ndarray | float = 0.0,
    floor_potential: bool = True,
    ion_adiabatic: float = 5.0 / 3.0,
    nve: jnp.ndarray | None = None,
    nvi: jnp.ndarray | None = None,
    mask: jnp.ndarray | None = None,
    sign: jnp.ndarray | None = None,
) -> ParallelSheathState:
    """Build literal Hermes sheath ghosts for the global open-field domain."""

    n_e = jnp.asarray(n_e)
    Te = jnp.asarray(Te)
    pe = jnp.asarray(pe)
    phi = jnp.asarray(phi)
    v_e = jnp.asarray(v_e)
    n_i = jnp.asarray(n_i)
    Ti = jnp.asarray(Ti)
    pi = jnp.asarray(pi)
    v_i = jnp.asarray(v_i)

    if n_e.shape[0] < 2:
        raise ValueError("build_parallel_sheath_state requires at least two parallel cells.")

    wall = _as_field(wall_potential, n_e[0])
    me = max(float(me_hat), 1.0e-12)
    mi = max(float(ion_mass), 1.0e-12)
    nve_cell = jnp.asarray(nve) if nve is not None else me * n_e * v_e
    nvi_cell = jnp.asarray(nvi) if nvi is not None else mi * n_i * v_i

    (
        n_ghost_low,
        Te_ghost_low,
        pe_ghost_low,
        phi_ghost_low,
        ne_sheath_low,
        Te_sheath_low,
        ve_sheath_low,
        ve_ghost_low,
        nve_ghost_low,
    ) = _electron_side(
        sign=-1.0,
        n_cell=n_e[0],
        n_inner=n_e[1],
        Te_cell=Te[0],
        Te_inner=Te[1],
        pe_cell=pe[0],
        pe_inner=pe[1],
        phi_cell=phi[0],
        phi_inner=phi[1],
        ve_cell=v_e[0],
        nve_cell=nve_cell[0],
        me_hat=me,
        secondary_electron_coef=secondary_electron_coef,
        wall_potential=wall,
        floor_potential=floor_potential,
    )
    (
        n_ghost_high,
        Te_ghost_high,
        pe_ghost_high,
        phi_ghost_high,
        ne_sheath_high,
        Te_sheath_high,
        ve_sheath_high,
        ve_ghost_high,
        nve_ghost_high,
    ) = _electron_side(
        sign=1.0,
        n_cell=n_e[-1],
        n_inner=n_e[-2],
        Te_cell=Te[-1],
        Te_inner=Te[-2],
        pe_cell=pe[-1],
        pe_inner=pe[-2],
        phi_cell=phi[-1],
        phi_inner=phi[-2],
        ve_cell=v_e[-1],
        nve_cell=nve_cell[-1],
        me_hat=me,
        secondary_electron_coef=secondary_electron_coef,
        wall_potential=wall,
        floor_potential=floor_potential,
    )

    (
        n_i_ghost_low,
        Ti_ghost_low,
        pi_ghost_low,
        vi_sheath_low,
        vi_ghost_low,
        nvi_ghost_low,
        _,
    ) = _ion_side(
        sign=-1.0,
        ne_cell=n_e[0],
        ne_sheath=ne_sheath_low,
        Te_sheath=Te_sheath_low,
        n_cell=n_i[0],
        n_inner=n_i[1],
        Ti_cell=Ti[0],
        Ti_inner=Ti[1],
        pi_cell=pi[0],
        pi_inner=pi[1],
        vi_cell=v_i[0],
        nvi_cell=nvi_cell[0],
        ion_mass=mi,
        ion_charge=ion_charge,
        ion_adiabatic=ion_adiabatic,
    )
    (
        n_i_ghost_high,
        Ti_ghost_high,
        pi_ghost_high,
        vi_sheath_high,
        vi_ghost_high,
        nvi_ghost_high,
        _,
    ) = _ion_side(
        sign=1.0,
        ne_cell=n_e[-1],
        ne_sheath=ne_sheath_high,
        Te_sheath=Te_sheath_high,
        n_cell=n_i[-1],
        n_inner=n_i[-2],
        Ti_cell=Ti[-1],
        Ti_inner=Ti[-2],
        pi_cell=pi[-1],
        pi_inner=pi[-2],
        vi_cell=v_i[-1],
        nvi_cell=nvi_cell[-1],
        ion_mass=mi,
        ion_charge=ion_charge,
        ion_adiabatic=ion_adiabatic,
    )

    return ParallelSheathState(
        n_ghost_low=n_ghost_low,
        n_ghost_high=n_ghost_high,
        Te_ghost_low=Te_ghost_low,
        Te_ghost_high=Te_ghost_high,
        Ti_ghost_low=Ti_ghost_low,
        Ti_ghost_high=Ti_ghost_high,
        pe_ghost_low=pe_ghost_low,
        pe_ghost_high=pe_ghost_high,
        pi_ghost_low=pi_ghost_low,
        pi_ghost_high=pi_ghost_high,
        phi_ghost_low=phi_ghost_low,
        phi_ghost_high=phi_ghost_high,
        ve_sheath_low=ve_sheath_low,
        ve_sheath_high=ve_sheath_high,
        vi_sheath_low=vi_sheath_low,
        vi_sheath_high=vi_sheath_high,
        ve_ghost_low=ve_ghost_low,
        ve_ghost_high=ve_ghost_high,
        vi_ghost_low=vi_ghost_low,
        vi_ghost_high=vi_ghost_high,
        nve_ghost_low=nve_ghost_low,
        nve_ghost_high=nve_ghost_high,
        nvi_ghost_low=nvi_ghost_low,
        nvi_ghost_high=nvi_ghost_high,
        j_ghost_low=nvi_ghost_low / mi - nve_ghost_low / me,
        j_ghost_high=nvi_ghost_high / mi - nve_ghost_high / me,
        mask=mask,
        sign=sign,
    )
