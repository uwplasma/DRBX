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
    """Hot-ion electrostatic extension state (adds an ion-temperature field Ti)."""

    n: jnp.ndarray
    omega: jnp.ndarray
    vpar_e: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray
    Ti: jnp.ndarray

    @classmethod
    def zeros(cls, nl: int, dtype=jnp.complex128) -> "State":
        z = jnp.zeros((nl,), dtype=dtype)
        return cls(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z, Ti=z)

    @classmethod
    def random(
        cls,
        key: jax.Array,
        nl: int,
        *,
        amplitude: float = 1e-3,
        dtype=jnp.complex128,
    ) -> "State":
        keys = jr.split(key, 12)

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
            Ti=cplx(keys[10], keys[11]),
        )


class RHSDecomposition(eqx.Module):
    """Split of the hot-ion RHS into conservative/source/dissipative parts."""

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
        Ti=a.Ti + b.Ti,
    )


def _state_zeros_like(y: State) -> State:
    z = jnp.zeros_like(y.n)
    return State(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z, Ti=z)


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
    """Return split RHS terms for hot-ion DRB in flux-tube form."""

    core = CoreLineModel(params=params, hot_ion_on=True, em_on=False)
    core_state = CoreState.from_optional(
        n=y.n, omega=y.omega, vpar_e=y.vpar_e, vpar_i=y.vpar_i, Te=y.Te, Ti=y.Ti
    )
    split = core.rhs_decomposed(t, core_state, geom, kx=kx, ky=ky, eq=eq)
    return RHSDecomposition(
        conservative=State(
            n=split.conservative.n,
            omega=split.conservative.omega,
            vpar_e=split.conservative.vpar_e,
            vpar_i=split.conservative.vpar_i,
            Te=split.conservative.Te,
            Ti=split.conservative.Ti,
        ),
        source=State(
            n=split.source.n,
            omega=split.source.omega,
            vpar_e=split.source.vpar_e,
            vpar_i=split.source.vpar_i,
            Te=split.source.Te,
            Ti=split.source.Ti,
        ),
        dissipative=State(
            n=split.dissipative.n,
            omega=split.dissipative.omega,
            vpar_e=split.dissipative.vpar_e,
            vpar_i=split.dissipative.vpar_i,
            Te=split.dissipative.Te,
            Ti=split.dissipative.Ti,
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
    """Hot-ion drift-reduced Braginskii-like RHS (electrostatic, flux-tube, single-(kx,ky)).

    This extends the cold-ion model by:

    - adding an ion temperature field `Ti`,
    - including an ion pressure contribution in the ion parallel momentum,
    - including ion pressure in the curvature drive through the total pressure perturbation.

    The implementation is intentionally compact (to keep the core solver matrix-free and
    differentiable) while providing a hot-ion branch that supports regression tests and
    literature-aligned trend scans. For scope/roadmap items (e.g. additional closures in 3D),
    see `docs/model/limitations.md`.
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
