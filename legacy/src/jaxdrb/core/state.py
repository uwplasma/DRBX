from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class CoreState(eqx.Module):
    """Superset state used by unified DRB RHS implementations."""

    n: jnp.ndarray
    omega: jnp.ndarray
    vpar_e: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray
    Ti: jnp.ndarray
    psi: jnp.ndarray
    N: jnp.ndarray

    @classmethod
    def zeros(cls, shape: tuple[int, ...], dtype=jnp.float64) -> "CoreState":
        z = jnp.zeros(shape, dtype=dtype)
        return cls(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z, Ti=z, psi=z, N=z)

    @classmethod
    def zeros_like(cls, ref: "CoreState") -> "CoreState":
        z = jnp.zeros_like(ref.n)
        return cls(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z, Ti=z, psi=z, N=z)

    @classmethod
    def from_optional(
        cls,
        *,
        n: jnp.ndarray,
        omega: jnp.ndarray,
        vpar_e: jnp.ndarray,
        vpar_i: jnp.ndarray,
        Te: jnp.ndarray,
        Ti: jnp.ndarray | None = None,
        psi: jnp.ndarray | None = None,
        N: jnp.ndarray | None = None,
    ) -> "CoreState":
        """Build a CoreState, filling missing fields with zeros."""

        z = jnp.zeros_like(n)
        return cls(
            n=n,
            omega=omega,
            vpar_e=vpar_e,
            vpar_i=vpar_i,
            Te=Te,
            Ti=z if Ti is None else Ti,
            psi=z if psi is None else psi,
            N=z if N is None else N,
        )

    def add(self, other: "CoreState") -> "CoreState":
        return CoreState(
            n=self.n + other.n,
            omega=self.omega + other.omega,
            vpar_e=self.vpar_e + other.vpar_e,
            vpar_i=self.vpar_i + other.vpar_i,
            Te=self.Te + other.Te,
            Ti=self.Ti + other.Ti,
            psi=self.psi + other.psi,
            N=self.N + other.N,
        )


class CoreSplit(eqx.Module):
    """Split of a RHS into conservative/source/dissipative parts (compat)."""

    conservative: CoreState
    source: CoreState
    dissipative: CoreState

    def total(self) -> CoreState:
        return self.conservative.add(self.source).add(self.dissipative)


class DRBSystemState(eqx.Module):
    """Unified system state with optional fields for memory efficiency."""

    n: jnp.ndarray
    omega: jnp.ndarray
    vpar_e: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray
    Ti: jnp.ndarray | None = None
    psi: jnp.ndarray | None = None
    N: jnp.ndarray | None = None

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, ...],
        *,
        dtype=jnp.float64,
        hot_ion: bool = False,
        em: bool = False,
        neutrals: bool = False,
    ) -> "DRBSystemState":
        z = jnp.zeros(shape, dtype=dtype)
        return cls(
            n=z,
            omega=z,
            vpar_e=z,
            vpar_i=z,
            Te=z,
            Ti=z if hot_ion else None,
            psi=z if em else None,
            N=z if neutrals else None,
        )

    def to_core(self) -> CoreState:
        return CoreState.from_optional(
            n=self.n,
            omega=self.omega,
            vpar_e=self.vpar_e,
            vpar_i=self.vpar_i,
            Te=self.Te,
            Ti=self.Ti,
            psi=self.psi,
            N=self.N,
        )


class DRBSystemSplit(eqx.Module):
    conservative: DRBSystemState
    source: DRBSystemState
    dissipative: DRBSystemState

    def total(self) -> DRBSystemState:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: DRBSystemState, b: DRBSystemState) -> DRBSystemState:
    def _opt_add(x: jnp.ndarray | None, y: jnp.ndarray | None) -> jnp.ndarray | None:
        if x is None and y is None:
            return None
        if x is None:
            return y
        if y is None:
            return x
        return x + y

    return DRBSystemState(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        vpar_e=a.vpar_e + b.vpar_e,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
        Ti=_opt_add(a.Ti, b.Ti),
        psi=_opt_add(a.psi, b.psi),
        N=_opt_add(a.N, b.N),
    )


def _state_zeros_like(y: DRBSystemState) -> DRBSystemState:
    z = jnp.zeros_like(y.n)
    return DRBSystemState(
        n=z,
        omega=z,
        vpar_e=z,
        vpar_i=z,
        Te=z,
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
