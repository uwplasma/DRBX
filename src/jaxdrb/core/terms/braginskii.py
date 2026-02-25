from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState
from jaxdrb.operators.fd2d import div_n_grad

from .context import TermContext


def _enabled(params, name: str) -> bool:
    return bool(getattr(params, name, False) or getattr(params, "braginskii_on", False))


def _collision_freq(params) -> jnp.ndarray:
    nu = jnp.asarray(getattr(params, "braginskii_nu_ei", 0.0), dtype=jnp.float64)
    nu_floor = float(getattr(params, "braginskii_nu_floor", 1e-12))
    return jnp.maximum(nu, nu_floor)


def braginskii_heat_exchange_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    if not _enabled(ctx.params, "braginskii_heat_exchange_on"):
        z = jnp.zeros_like(y.n)
        return DRBSystemState(
            n=z,
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    if not ctx.hot_on:
        z = jnp.zeros_like(y.n)
        return DRBSystemState(
            n=z,
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None,
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    n_eff = jnp.maximum(ctx.n_phys, float(ctx.params.n0_min))
    A_e = jnp.maximum(float(ctx.params.me_hat), 1e-6)
    A_i = 1.0
    nu = _collision_freq(ctx.params)
    Ti_eff = float(ctx.params.tau_i) + ctx.Ti

    Q_ei = 3.0 * (A_e / (A_e + A_i)) * nu * n_eff * (Ti_eff - ctx.Te_phys)
    dTe = (2.0 / 3.0) * Q_ei / n_eff
    dTi = -(2.0 / 3.0) * Q_ei / n_eff

    return DRBSystemState(
        n=jnp.zeros_like(y.n),
        omega=jnp.zeros_like(y.omega),
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=dTe,
        Ti=dTi if y.Ti is not None else None,
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def braginskii_friction_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    if not _enabled(ctx.params, "braginskii_friction_on"):
        z = jnp.zeros_like(y.n)
        return DRBSystemState(
            n=z,
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    n_eff = jnp.maximum(ctx.n_phys, float(ctx.params.n0_min))
    nu = _collision_freq(ctx.params)
    A_e = jnp.maximum(float(ctx.params.me_hat), 1e-6)
    A_i = 1.0
    coeff = float(getattr(ctx.params, "braginskii_friction_coeff", 0.51))

    dv = y.vpar_i - y.vpar_e
    F_ei = coeff * A_e * nu * n_eff * dv
    dvpar_e = F_ei / n_eff
    dvpar_i = -F_ei / n_eff

    dTe = jnp.zeros_like(y.Te)
    dTi = None if y.Ti is None else jnp.zeros_like(y.Ti)
    if bool(getattr(ctx.params, "braginskii_frictional_heating_on", True)) and ctx.hot_on:
        E_e = (A_i / (A_e + A_i)) * dv * F_ei
        E_i = (A_e / (A_e + A_i)) * dv * F_ei
        dTe = (2.0 / 3.0) * E_e / n_eff
        dTi = (2.0 / 3.0) * E_i / n_eff

    return DRBSystemState(
        n=jnp.zeros_like(y.n),
        omega=jnp.zeros_like(y.omega),
        vpar_e=dvpar_e,
        vpar_i=dvpar_i,
        Te=dTe,
        Ti=dTi if y.Ti is not None else None,
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def classical_diffusion_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    if not _enabled(ctx.params, "classical_diffusion_on"):
        z = jnp.zeros_like(y.n)
        return DRBSystemState(
            n=z,
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    B = getattr(ctx.geom, "B", None)
    if B is None:
        invB2 = 1.0
    else:
        invB2 = 1.0 / jnp.maximum(jnp.asarray(B), 1e-12) ** 2

    n_eff = jnp.maximum(ctx.n_phys, float(ctx.params.n0_min))
    nu_e = _collision_freq(ctx.params)
    nu_i = jnp.maximum(
        jnp.asarray(getattr(ctx.params, "braginskii_nu_ii", 0.0), dtype=jnp.float64),
        float(getattr(ctx.params, "braginskii_nu_floor", 1e-12)),
    )

    p_e = ctx.n_phys * ctx.Te_phys
    Ti_eff = float(ctx.params.tau_i) + (ctx.Ti if ctx.hot_on else jnp.zeros_like(ctx.Te_phys))
    p_i = ctx.n_phys * Ti_eff
    p_total = p_e + p_i

    custom_D = float(getattr(ctx.params, "classical_diffusion_custom_D", -1.0))
    if custom_D > 0.0:
        Dn = jnp.asarray(custom_D, dtype=jnp.float64)
    else:
        Dn = p_total * float(ctx.params.me_hat) * nu_e / jnp.maximum(n_eff, 1e-12)
        Dn = Dn * invB2

    grid = getattr(ctx.geom, "grid", None)
    if grid is None:
        dn = Dn * ctx.geom.laplacian(ctx.n_phys)
        dvpar_e = Dn * ctx.geom.laplacian(y.vpar_e)
        dvpar_i = Dn * ctx.geom.laplacian(y.vpar_i)
    else:
        if y.n.ndim == 2:
            dn = div_n_grad(ctx.n_phys, Dn, dx=grid.dx, dy=grid.dy, bc=ctx.bcs.n)
            dvpar_e = div_n_grad(y.vpar_e, Dn, dx=grid.dx, dy=grid.dy, bc=ctx.bcs.vpar_e)
            dvpar_i = div_n_grad(y.vpar_i, Dn, dx=grid.dx, dy=grid.dy, bc=ctx.bcs.vpar_i)
        else:
            dn = jax.vmap(
                lambda f, coeff: div_n_grad(f, coeff, dx=grid.dx, dy=grid.dy, bc=ctx.bcs.n)
            )(ctx.n_phys, Dn)
            dvpar_e = jax.vmap(
                lambda f, coeff: div_n_grad(f, coeff, dx=grid.dx, dy=grid.dy, bc=ctx.bcs.vpar_e)
            )(y.vpar_e, Dn)
            dvpar_i = jax.vmap(
                lambda f, coeff: div_n_grad(f, coeff, dx=grid.dx, dy=grid.dy, bc=ctx.bcs.vpar_i)
            )(y.vpar_i, Dn)

    custom_ke = float(getattr(ctx.params, "classical_diffusion_custom_kappa_e", -1.0))
    if custom_ke > 0.0:
        kappa_e = jnp.asarray(custom_ke, dtype=jnp.float64)
    else:
        kappa_e = 2.0 * p_e * nu_e * float(ctx.params.me_hat) * invB2
    custom_ki = float(getattr(ctx.params, "classical_diffusion_custom_kappa_i", -1.0))
    if custom_ki > 0.0:
        kappa_i = jnp.asarray(custom_ki, dtype=jnp.float64)
    else:
        kappa_i = 2.0 * p_i * nu_i * invB2

    if grid is None:
        dTe = kappa_e * ctx.geom.laplacian(ctx.Te_phys) / jnp.maximum(n_eff, 1e-12)
        dTi = (
            kappa_i * ctx.geom.laplacian(Ti_eff) / jnp.maximum(n_eff, 1e-12)
            if ctx.hot_on
            else jnp.zeros_like(ctx.Te_phys)
        )
    else:
        if y.n.ndim == 2:
            dTe = div_n_grad(
                ctx.Te_phys, kappa_e, dx=grid.dx, dy=grid.dy, bc=ctx.bcs.Te
            ) / jnp.maximum(n_eff, 1e-12)
            dTi = (
                div_n_grad(Ti_eff, kappa_i, dx=grid.dx, dy=grid.dy, bc=ctx.bcs.Ti)
                / jnp.maximum(n_eff, 1e-12)
                if ctx.hot_on
                else jnp.zeros_like(ctx.Te_phys)
            )
        else:
            dTe = jax.vmap(
                lambda f, coeff: div_n_grad(f, coeff, dx=grid.dx, dy=grid.dy, bc=ctx.bcs.Te)
            )(ctx.Te_phys, kappa_e) / jnp.maximum(n_eff, 1e-12)
            dTi = (
                jax.vmap(
                    lambda f, coeff: div_n_grad(f, coeff, dx=grid.dx, dy=grid.dy, bc=ctx.bcs.Ti)
                )(Ti_eff, kappa_i)
                / jnp.maximum(n_eff, 1e-12)
                if ctx.hot_on
                else jnp.zeros_like(ctx.Te_phys)
            )

    return DRBSystemState(
        n=dn,
        omega=jnp.zeros_like(y.omega),
        vpar_e=dvpar_e,
        vpar_i=dvpar_i,
        Te=dTe,
        Ti=dTi if y.Ti is not None else None,
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
