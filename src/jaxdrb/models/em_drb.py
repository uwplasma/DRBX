from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from jaxdrb.core.line import CoreLineModel, LineEquilibrium
from jaxdrb.core.state import CoreState
from jaxdrb.models.params import DRBParams

Equilibrium = LineEquilibrium


class State(eqx.Module):
    """Electromagnetic extension state (reduced, Ampère-closed).

    This model eliminates `vpar_e` in favor of an inductive variable `psi ~ -A_parallel`
    together with an Ampère closure for the parallel current.
    """

    n: jnp.ndarray
    omega: jnp.ndarray
    psi: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray

    @classmethod
    def zeros(cls, nl: int, dtype=jnp.complex128) -> "State":
        z = jnp.zeros((nl,), dtype=dtype)
        return cls(n=z, omega=z, psi=z, vpar_i=z, Te=z)

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
            psi=cplx(keys[4], keys[5]),
            vpar_i=cplx(keys[6], keys[7]),
            Te=cplx(keys[8], keys[9]),
        )


class RHSDecomposition(eqx.Module):
    """Split of the EM RHS into conservative/source/dissipative parts."""

    conservative: State
    source: State
    dissipative: State

    def total(self) -> State:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: State, b: State) -> State:
    return State(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        psi=a.psi + b.psi,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
    )


def _state_zeros_like(y: State) -> State:
    z = jnp.zeros_like(y.n)
    return State(n=z, omega=z, psi=z, vpar_i=z, Te=z)


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
    """Return split RHS terms for EM DRB in flux-tube form."""

    core = CoreLineModel(params=params, hot_ion_on=False, em_on=True)
    core_state = CoreState.from_optional(
        n=y.n, omega=y.omega, vpar_e=jnp.zeros_like(y.n), vpar_i=y.vpar_i, Te=y.Te, psi=y.psi
    )
    split = core.rhs_decomposed(t, core_state, geom, kx=kx, ky=ky, eq=eq)
    return RHSDecomposition(
        conservative=State(
            n=split.conservative.n,
            omega=split.conservative.omega,
            psi=split.conservative.psi,
            vpar_i=split.conservative.vpar_i,
            Te=split.conservative.Te,
        ),
        source=State(
            n=split.source.n,
            omega=split.source.omega,
            psi=split.source.psi,
            vpar_i=split.source.vpar_i,
            Te=split.source.Te,
        ),
        dissipative=State(
            n=split.dissipative.n,
            omega=split.dissipative.omega,
            psi=split.dissipative.psi,
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
    """Electromagnetic drift-reduced Braginskii-like RHS (flux-tube, single-(kx,ky)).

    This is a compact inductive extension intended to expose finite-beta effects while preserving
    the matrix-free solver workflow. It uses:

    - Ampere closure: j_parallel = -∇_⊥^2 psi = +k_⊥^2 psi
    - Induction/Ohm-like equation: (beta/2 + m_hat_e k_⊥^2) ∂_t psi = -∇_||(phi - n - Te) - eta j_||

    The electron parallel velocity is eliminated using j_|| = v_||i - v_||e.
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
