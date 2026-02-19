from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from jaxdrb.core import DRBSystem, DRBSystemState, LineEquilibrium, LineGeometryAdapter
from jaxdrb.core.compat import coerce_system_params
from jaxdrb.models.params import DRBParams


Equilibrium = LineEquilibrium


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
    """Return split RHS terms for cold-ion DRB in flux-tube form."""

    sys_params = coerce_system_params(params, hot_ion_on=False, em_on=False)
    geom_adapter = LineGeometryAdapter(geom=geom, params=sys_params, kx=float(kx), ky=float(ky))
    system = DRBSystem(params=sys_params, geom=geom_adapter)
    sys_state = DRBSystemState(
        n=y.n,
        omega=y.omega,
        vpar_e=y.vpar_e,
        vpar_i=y.vpar_i,
        Te=y.Te,
        Ti=None,
        psi=None,
        N=None,
    )
    split = system.rhs_split(t, sys_state)
    return RHSDecomposition(
        conservative=State(
            n=split.conservative.n,
            omega=split.conservative.omega,
            vpar_e=split.conservative.vpar_e,
            vpar_i=split.conservative.vpar_i,
            Te=split.conservative.Te,
        ),
        source=State(
            n=split.source.n,
            omega=split.source.omega,
            vpar_e=split.source.vpar_e,
            vpar_i=split.source.vpar_i,
            Te=split.source.Te,
        ),
        dissipative=State(
            n=split.dissipative.n,
            omega=split.dissipative.omega,
            vpar_e=split.dissipative.vpar_e,
            vpar_i=split.dissipative.vpar_i,
            Te=split.dissipative.Te,
        ),
    )


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
