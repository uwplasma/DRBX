from __future__ import annotations

from typing import ClassVar, Protocol

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.bc import BC1D, BC2D


class GeometryAdapter(Protocol):
    """Minimal geometry adapter interface for the unified DRB system.

    Concrete geometry adapters (2D, FCI 3D, field-line) should implement the
    subset of methods actually used by the unified RHS. The goal is to keep
    physics in the core system while geometry owns discretization details.
    """

    name: str
    ndim: int

    def shape(self) -> tuple[int, ...]: ...

    # Perpendicular operators (x/y on a plane).
    def ddx(self, f: jnp.ndarray) -> jnp.ndarray: ...

    def ddy(self, f: jnp.ndarray) -> jnp.ndarray: ...

    def laplacian(self, f: jnp.ndarray) -> jnp.ndarray: ...

    def biharmonic(self, f: jnp.ndarray) -> jnp.ndarray: ...

    def inv_laplacian(self, f: jnp.ndarray) -> jnp.ndarray: ...

    def inv_div_n_grad(self, n_eff: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray: ...

    def bracket(
        self,
        phi: jnp.ndarray,
        f: jnp.ndarray,
        *,
        bc_phi: BC2D | None = None,
        bc_f: BC2D | None = None,
    ) -> jnp.ndarray: ...

    # Parallel operator(s).
    def dpar(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray: ...

    def d2par(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray: ...

    # Curvature operator (if available).
    def curvature(self, f: jnp.ndarray) -> jnp.ndarray: ...

    def kappa_profile(self) -> jnp.ndarray | float: ...

    def sheath_mask_sign(self) -> tuple[jnp.ndarray, jnp.ndarray]: ...

    def enforce_bc_relaxation(self, f: jnp.ndarray, *, nu: float) -> jnp.ndarray: ...

    # Boundary conditions helper (optional).
    def apply_bc(self, f: jnp.ndarray, bc: BC1D | BC2D) -> jnp.ndarray: ...


class GeometryBase(eqx.Module):
    """Lightweight base class for concrete geometry adapters."""

    name: ClassVar[str] = ""
    ndim: ClassVar[int] = 0

    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def ddx(self, f: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def ddy(self, f: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def laplacian(self, f: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def biharmonic(self, f: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def inv_laplacian(self, f: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def inv_div_n_grad(self, n_eff: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def bracket(
        self,
        phi: jnp.ndarray,
        f: jnp.ndarray,
        *,
        bc_phi: BC2D | None = None,
        bc_f: BC2D | None = None,
    ) -> jnp.ndarray:
        raise NotImplementedError

    def dpar(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        raise NotImplementedError

    def d2par(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        raise NotImplementedError

    def curvature(self, f: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def kappa_profile(self) -> jnp.ndarray | float:
        raise NotImplementedError

    def sheath_mask_sign(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError

    def enforce_bc_relaxation(self, f: jnp.ndarray, *, nu: float) -> jnp.ndarray:
        raise NotImplementedError

    def apply_bc(self, f: jnp.ndarray, bc: BC1D | BC2D) -> jnp.ndarray:
        return f
