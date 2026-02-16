from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class CoreState(eqx.Module):
    """Superset state used by unified DRB RHS implementations.

    This state intentionally includes all fields used across cold-ion, hot-ion,
    and EM variants. Models that do not evolve a given field should pass zeros
    (or use :meth:`from_optional` to fill missing fields with zeros).
    """

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
    """Split of a RHS into conservative/source/dissipative parts."""

    conservative: CoreState
    source: CoreState
    dissipative: CoreState

    def total(self) -> CoreState:
        return self.conservative.add(self.source).add(self.dissipative)
