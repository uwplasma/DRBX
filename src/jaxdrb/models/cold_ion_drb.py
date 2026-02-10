from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from jaxdrb.models.params import DRBParams
from jaxdrb.models.bcs import bc_relaxation_1d
from jaxdrb.models.braginskii import chi_par_Te as chi_par_Te_eff
from jaxdrb.models.braginskii import eta_parallel as eta_parallel_eff
from jaxdrb.models.braginskii import nu_par_e as nu_par_e_eff
from jaxdrb.models.braginskii import nu_par_i as nu_par_i_eff
from jaxdrb.models.sheath import (
    apply_loizu_mpse_boundary_conditions,
    apply_loizu2012_mpse_full_linear_bc,
    sheath_energy_losses,
    sheath_bc_rate,
    sheath_loss_rate,
)


class Equilibrium(eqx.Module):
    """Background profiles along the field line used by the RHS.

    The evolving `State` is interpreted as a perturbation about this equilibrium.
    """

    n0: jnp.ndarray
    Te0: jnp.ndarray

    @classmethod
    def constant(
        cls,
        nl: int,
        *,
        n0: float = 1.0,
        Te0: float = 1.0,
        dtype=jnp.float64,
    ) -> "Equilibrium":
        return cls(
            n0=jnp.full((nl,), float(n0), dtype=dtype),
            Te0=jnp.full((nl,), float(Te0), dtype=dtype),
        )


class State(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
    vpar_e: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray

    @classmethod
    def zeros(cls, nl: int, dtype=jnp.complex128) -> "State":
        z = jnp.zeros((nl,), dtype=dtype)
        return cls(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z)

    @classmethod
    def random(
        cls,
        key: jax.Array,
        nl: int,
        *,
        amplitude: float = 1e-3,
        dtype=jnp.complex128,
    ) -> "State":
        keys = jr.split(key, 10)

        def cplx(kre, kim):
            re = jr.normal(kre, (nl,), dtype=jnp.float64)
            im = jr.normal(kim, (nl,), dtype=jnp.float64)
            z = re + 1j * im
            return (amplitude * z).astype(dtype)

        return cls(
            n=cplx(keys[0], keys[1]),
            omega=cplx(keys[2], keys[3]),
            vpar_e=cplx(keys[4], keys[5]),
            vpar_i=cplx(keys[6], keys[7]),
            Te=cplx(keys[8], keys[9]),
        )


class RHSDecomposition(eqx.Module):
    """Split of the cold-ion RHS into conservative/source/dissipative parts."""

    conservative: State
    source: State
    dissipative: State

    def total(self) -> State:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: State, b: State) -> State:
    return State(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        vpar_e=a.vpar_e + b.vpar_e,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
    )


def _state_scale(a: State, scale: float) -> State:
    return State(
        n=scale * a.n,
        omega=scale * a.omega,
        vpar_e=scale * a.vpar_e,
        vpar_i=scale * a.vpar_i,
        Te=scale * a.Te,
    )


def _state_zeros_like(y: State) -> State:
    z = jnp.zeros_like(y.n)
    return State(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z)


def phi_from_omega(
    omega: jnp.ndarray,
    kperp2: jnp.ndarray,
    *,
    kperp2_min: float,
    boussinesq: bool,
    n0: jnp.ndarray | None = None,
    n0_min: float = 1e-6,
    n: jnp.ndarray | None = None,
    non_boussinesq_perturbed_density_on: bool = False,
) -> jnp.ndarray:
    k2 = jnp.maximum(kperp2, kperp2_min)
    if boussinesq:
        return -omega / k2
    if n0 is None:
        raise ValueError("Non-Boussinesq polarization requires an equilibrium density n0.")
    if non_boussinesq_perturbed_density_on and n is not None:
        n_eff = jnp.maximum(jnp.asarray(n0) + jnp.real(jnp.asarray(n)), n0_min)
    else:
        n_eff = jnp.maximum(jnp.asarray(n0), n0_min)
    return -omega / (k2 * n_eff)


def rhs_nonlinear_decomposed(
    t: float,
    y: State,
    params: DRBParams,
    geom,
    *,
    kx: float,
    ky: float,
    eq: Equilibrium | None = None,
) -> RHSDecomposition:
    """Return split RHS terms for cold-ion DRB in flux-tube form.

    The split is:
    - conservative: ideal parallel couplings used by conservative-gate subsets,
    - source: drives/curvature free-energy injection terms,
    - dissipative: resistivity, closures, diffusion, sinks, and boundary losses/relaxations.
    """

    k2 = geom.kperp2(kx, ky)
    if eq is None:
        eq = Equilibrium.constant(int(y.n.size), n0=1.0, Te0=1.0)
    phi = phi_from_omega(
        y.omega,
        k2,
        kperp2_min=params.kperp2_min,
        boussinesq=params.boussinesq,
        n0=eq.n0,
        n0_min=params.n0_min,
        n=y.n,
        non_boussinesq_perturbed_density_on=bool(
            getattr(params, "non_boussinesq_perturbed_density_on", False)
        ),
    )

    dpar = geom.dpar
    C = geom.curvature

    def d2par(f: jnp.ndarray) -> jnp.ndarray:
        return dpar(dpar(f))

    # Electron inertia handling:
    # - For me_hat > 0: evolve vpar_e with an inertial Ohm's law.
    # - For me_hat = 0: treat Ohm's law as an algebraic constraint and relax to it.
    use_algebraic_ohm = params.me_hat == 0.0

    # Drives from background gradients: -[phi, n0] -> -i ky omega_n phi
    drive_n = -1j * ky * params.omega_n * phi
    drive_Te = -1j * ky * params.omega_Te * phi

    # Curvature operators
    if params.curvature_on:
        C_phi = C(kx, ky, phi)
        C_p = C(kx, ky, y.n + y.Te)
        C_T = (2.0 / 3.0) * C(kx, ky, (7.0 / 2.0) * y.Te + y.n - phi)
    else:
        C_phi = jnp.zeros_like(phi)
        C_p = jnp.zeros_like(phi)
        C_T = jnp.zeros_like(phi)

    # Perp diffusion in Fourier space: D * ∇_⊥^2 f -> -D k_⊥^2 f
    lap_n = -k2 * y.n
    lap_omega = -k2 * y.omega
    lap_Te = -k2 * y.Te

    # Continuity conventions: C(p) - C(phi), with vpar_e_eff in compressibility.
    grad_par_phi_pe = dpar(phi - y.n - float(params.alpha_Te_ohm) * y.Te)
    eta_eff = jnp.maximum(eta_parallel_eff(params, eq, Te_state=y.Te), 1e-12)
    vpar_e_eff = jnp.where(use_algebraic_ohm, y.vpar_i + grad_par_phi_pe / eta_eff, y.vpar_e)

    # Parallel current (n0 = 1 normalization)
    jpar = y.vpar_i - vpar_e_eff

    conservative = State(
        n=-dpar(vpar_e_eff),
        omega=dpar(jpar),
        vpar_e=jnp.where(
            use_algebraic_ohm,
            jnp.zeros_like(y.vpar_e),
            grad_par_phi_pe / jnp.maximum(float(params.me_hat), 1e-12),
        ),
        vpar_i=-dpar(phi),
        Te=-(2.0 / 3.0) * dpar(vpar_e_eff),
    )

    source = State(
        n=drive_n + (C_p - C_phi),
        omega=C_p,
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=drive_Te + C_T,
    )

    if use_algebraic_ohm:
        dvpar_e_eta = -eta_eff * (y.vpar_e - vpar_e_eff)
    else:
        dvpar_e_eta = -(eta_eff * (y.vpar_e - y.vpar_i)) / jnp.maximum(float(params.me_hat), 1e-12)

    dissipative = State(
        n=params.Dn * lap_n - float(getattr(params, "nu_sink_n", 0.0)) * y.n,
        omega=params.DOmega * lap_omega,
        vpar_e=dvpar_e_eta
        + nu_par_e_eff(params, eq, Te_state=y.Te) * d2par(y.vpar_e)
        - float(getattr(params, "nu_sink_vpar", 0.0)) * y.vpar_e,
        vpar_i=nu_par_i_eff(params, eq, Te_state=y.Te) * d2par(y.vpar_i)
        - float(getattr(params, "nu_sink_vpar", 0.0)) * y.vpar_i,
        Te=params.DTe * lap_Te
        + chi_par_Te_eff(params, eq, Te_state=y.Te) * d2par(y.Te)
        - float(getattr(params, "nu_sink_Te", 0.0)) * y.Te,
    )

    # Optional MPSE (sheath) boundary conditions for open field lines.
    # Model 0: velocity-only (legacy). Model 1: Loizu 2012 "full set" (linearized, model-aligned).
    if int(getattr(params, "sheath_bc_model", 0)) == 1:
        dn_bc, domega_bc, dvpar_e_bc, dvpar_i_bc, dTe_bc = apply_loizu2012_mpse_full_linear_bc(
            params=params,
            geom=geom,
            eq=eq,
            kperp2=k2,
            phi=phi,
            n=y.n,
            omega=y.omega,
            vpar_e=vpar_e_eff,
            vpar_i=y.vpar_i,
            Te=y.Te,
            dpar=dpar,
            d2par=d2par,
        )
        dissipative = _state_add(
            dissipative,
            State(n=dn_bc, omega=domega_bc, vpar_e=dvpar_e_bc, vpar_i=dvpar_i_bc, Te=dTe_bc),
        )
    else:
        dvpar_e_sh, dvpar_i_sh = apply_loizu_mpse_boundary_conditions(
            params=params,
            geom=geom,
            eq=eq,
            phi=phi,
            vpar_e=vpar_e_eff,
            vpar_i=y.vpar_i,
            Te=y.Te,
        )
        dissipative = _state_add(
            dissipative,
            State(
                n=jnp.zeros_like(y.n),
                omega=jnp.zeros_like(y.omega),
                vpar_e=dvpar_e_sh,
                vpar_i=dvpar_i_sh,
                Te=jnp.zeros_like(y.Te),
            ),
        )

    # Optional sheath heat transmission / energy losses.
    dTe_sh, _ = sheath_energy_losses(params=params, geom=geom, Te=y.Te)
    dissipative = _state_add(
        dissipative,
        State(
            n=jnp.zeros_like(y.n),
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=dTe_sh,
        ),
    )

    # Optional additional damping localized at sheath nodes.
    if bool(getattr(params, "sheath_end_damp_on", False)):
        bc = sheath_bc_rate(params, geom)
        if bc is not None:
            nu_bc, mask = bc
            dissipative = _state_add(
                dissipative,
                State(
                    n=-nu_bc * mask * y.n,
                    omega=-nu_bc * mask * y.omega,
                    vpar_e=jnp.zeros_like(y.vpar_e),
                    vpar_i=jnp.zeros_like(y.vpar_i),
                    Te=-nu_bc * mask * y.Te,
                ),
            )

    # Optional volumetric sheath-loss proxy.
    nu_loss = sheath_loss_rate(params, geom)
    dissipative = _state_add(
        dissipative,
        State(
            n=-nu_loss * y.n,
            omega=-nu_loss * y.omega,
            vpar_e=-nu_loss * y.vpar_e,
            vpar_i=-nu_loss * y.vpar_i,
            Te=-nu_loss * y.Te,
        ),
    )

    # Optional user-defined boundary conditions along l (weak relaxation).
    if getattr(params, "line_bcs", None) is not None and params.line_bcs.enabled:
        dl = float(geom.dl)
        dissipative = _state_add(
            dissipative,
            State(
                n=bc_relaxation_1d(y.n, bc=params.line_bcs.n, dl=dl),
                omega=bc_relaxation_1d(y.omega, bc=params.line_bcs.omega, dl=dl),
                vpar_e=bc_relaxation_1d(y.vpar_e, bc=params.line_bcs.vpar_e, dl=dl),
                vpar_i=bc_relaxation_1d(y.vpar_i, bc=params.line_bcs.vpar_i, dl=dl),
                Te=bc_relaxation_1d(y.Te, bc=params.line_bcs.Te, dl=dl),
            ),
        )

    return RHSDecomposition(conservative=conservative, source=source, dissipative=dissipative)


def rhs_nonlinear(
    t: float,
    y: State,
    params: DRBParams,
    geom,
    *,
    kx: float,
    ky: float,
    eq: Equilibrium | None = None,
) -> State:
    """Cold-ion drift-reduced Braginskii-like RHS in flux-tube (single-(kx,ky)) form.

    For a single Fourier mode, the nonlinear Poisson bracket self-interaction vanishes, so this
    implementation is linear in `y` but kept in this form for future extension.
    """
    split = rhs_nonlinear_decomposed(t, y, params, geom, kx=kx, ky=ky, eq=eq)
    if not bool(getattr(params, "operator_split_on", False)):
        return split.total()

    out = _state_zeros_like(y)
    if bool(getattr(params, "operator_conservative_on", True)):
        out = _state_add(out, split.conservative)
    if bool(getattr(params, "operator_source_on", True)):
        out = _state_add(out, split.source)
    if bool(getattr(params, "operator_dissipative_on", True)):
        out = _state_add(out, split.dissipative)
    return out


def equilibrium(nl: int, dtype=jnp.complex128) -> State:
    return State.zeros(nl, dtype=dtype)


def default_equilibrium(nl: int, *, n0: float = 1.0) -> Equilibrium:
    return Equilibrium.constant(nl, n0=n0, Te0=1.0)
