"""Magnetic-field providers: one protocol for MGRID, coil and equilibrium fields.

A provider evaluates the physical Cartesian field ``B(x)`` in tesla at points
``x`` of shape ``(..., 3)`` in metres. Every provider is a JAX pytree whose
leaves are its parameters (coil shapes and currents, equilibrium arrays), so
``jax.grad``/``jax.jvp`` through ``provider.B`` give derivatives with respect
to those parameters, and ``jax.jacfwd`` with respect to ``x`` gives grad B.

Implementations
---------------
``EssosCoilField``
    ESSOS ``Coils`` evaluated with ``essos.fields.BiotSavart`` (installed
    ``essos`` package). Coil-parameter derivatives flow through ESSOS.

The MGRID compatibility path stays in
:mod:`drbx.fci_braginskii.geometry_build.Bfield_evaluator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

@runtime_checkable
class MagneticFieldProvider(Protocol):
    """Physical Cartesian vacuum or equilibrium field."""

    nfp: int
    stellsym: bool
    provenance: dict[str, Any]

    def B(self, xyz: jnp.ndarray) -> jnp.ndarray:
        """Cartesian B [T] at ``xyz`` (``(..., 3)`` in m)."""

    def inside(self, xyz: jnp.ndarray) -> jnp.ndarray:
        """Boolean validity mask of the field at ``xyz`` (``(...,)``)."""


def grad_B(provider: MagneticFieldProvider, xyz: jnp.ndarray) -> jnp.ndarray:
    """``dB_i/dx_j`` at ``xyz`` by forward-mode AD; shape ``(..., 3, 3)``."""

    flat = jnp.reshape(jnp.asarray(xyz), (-1, 3))
    jac = jax.vmap(jax.jacfwd(lambda p: provider.B(p[None])[0]))(flat)
    return jnp.reshape(jac, (*jnp.shape(xyz)[:-1], 3, 3))


def B_cylindrical(provider: MagneticFieldProvider, R, phi, Z):
    """``(B_R, B_phi, B_Z)`` on broadcast cylindrical arrays."""

    R, phi, Z = jnp.broadcast_arrays(jnp.asarray(R), jnp.asarray(phi), jnp.asarray(Z))
    b = provider.B(jnp.stack((R * jnp.cos(phi), R * jnp.sin(phi), Z), axis=-1))
    c, s = jnp.cos(phi), jnp.sin(phi)
    return b[..., 0] * c + b[..., 1] * s, -b[..., 0] * s + b[..., 1] * c, b[..., 2]


def _hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(k), _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value


def _meta(**items: Any) -> tuple:
    """Hashable provenance (static pytree metadata) from keyword items."""

    return tuple(sorted((k, _hashable(v)) for k, v in items.items()))


def _chunked_points(fn, xyz: jnp.ndarray, chunk: int) -> jnp.ndarray:
    flat = jnp.reshape(jnp.asarray(xyz, dtype=jnp.float64), (-1, 3))
    n = flat.shape[0]
    pad = (-n) % chunk
    padded = jnp.concatenate([flat, jnp.ones((pad, 3), flat.dtype)], axis=0) if pad else flat
    out = jax.lax.map(fn, padded.reshape(-1, chunk, 3))
    return jnp.reshape(out.reshape(-1, 3)[:n], jnp.shape(xyz))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EssosCoilField:
    """ESSOS coils through ``essos.fields.BiotSavart`` (installed package)."""

    coils: Any
    nfp: int = field(metadata=dict(static=True), default=1)
    stellsym: bool = field(metadata=dict(static=True), default=False)
    chunk: int = field(metadata=dict(static=True), default=4096)
    meta: tuple = field(metadata=dict(static=True), default=())

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self.meta)

    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> "EssosCoilField":
        from essos.coils import Coils

        jax.config.update("jax_enable_x64", True)
        coils = Coils.from_json(str(path))
        return cls(
            coils=coils,
            nfp=int(coils.nfp),
            stellsym=bool(coils.stellsym),
            meta=_meta(kind="essos_coils", source=Path(path).name),
            **kwargs,
        )

    def B(self, xyz: jnp.ndarray) -> jnp.ndarray:
        from essos.fields import BiotSavart

        field_ = BiotSavart(self.coils)
        return _chunked_points(jax.vmap(field_.B), xyz, self.chunk)

    def inside(self, xyz: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones(jnp.shape(xyz)[:-1], dtype=bool)
