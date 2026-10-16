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
``FilamentCoilField``
    Closed polyline filaments (MAKEGRID ``coils.*`` files) with the exact
    straight-segment Biot--Savart law, the discretisation MAKEGRID uses to
    build mgrid files. Derivatives flow to the filament nodes and currents.

The MGRID compatibility path stays in
:mod:`drbx.fci_braginskii.geometry_build.Bfield_evaluator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

MU0_OVER_4PI = 1.0e-7


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


def _segment_biot_savart(nodes: jnp.ndarray, currents: jnp.ndarray, points: jnp.ndarray) -> jnp.ndarray:
    """Exact field of closed straight-segment filaments at ``points`` (P, 3).

    ``nodes`` is (C, N, 3), each filament closed from node N-1 back to node 0.
    Uses B = mu0 I / 4pi * (|a|+|b|) (a x b) / (|a||b| (|a||b| + a.b)).
    """

    a = points[:, None, None, :] - nodes[None]
    b = points[:, None, None, :] - jnp.roll(nodes, -1, axis=1)[None]
    na = jnp.linalg.norm(a, axis=-1)
    nb = jnp.linalg.norm(b, axis=-1)
    factor = (na + nb) / (na * nb * (na * nb + jnp.sum(a * b, axis=-1)))
    dB = jnp.cross(a, b) * factor[..., None]
    return MU0_OVER_4PI * jnp.einsum("c,pcnk->pk", currents, dB)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FilamentCoilField:
    """Closed polyline coil filaments with exact segment Biot--Savart."""

    nodes: jnp.ndarray
    currents: jnp.ndarray
    nfp: int = field(metadata=dict(static=True), default=1)
    stellsym: bool = field(metadata=dict(static=True), default=False)
    chunk: int = field(metadata=dict(static=True), default=256)
    meta: tuple = field(metadata=dict(static=True), default=())

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self.meta)

    def B(self, xyz: jnp.ndarray) -> jnp.ndarray:
        return _chunked_points(lambda p: _segment_biot_savart(self.nodes, self.currents, p), xyz, self.chunk)

    def inside(self, xyz: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones(jnp.shape(xyz)[:-1], dtype=bool)

    @classmethod
    def from_makegrid(
        cls,
        path: str | Path,
        group_currents: dict[int, float] | None = None,
        *,
        stellsym: bool = True,
        **kwargs,
    ) -> "FilamentCoilField":
        """Load a MAKEGRID ``coils.*`` file.

        Each filament ends at a line carrying its group number; its current is
        ``group_currents[group]`` times the per-node current written in the
        file (MAKEGRID convention). Filaments whose total current is zero are
        dropped. MAKEGRID closes a filament by repeating its first node; the
        repeat is removed here, and the closing segment is implicit.
        """

        jax.config.update("jax_enable_x64", True)
        nfp = 1
        filaments: list[np.ndarray] = []
        currents: list[float] = []
        groups: list[int] = []
        names: list[str] = []
        block: list[list[float]] = []
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                parts = raw.split()
                if not parts:
                    continue
                head = parts[0].lower()
                if head == "periods":
                    nfp = int(parts[1])
                    continue
                if head in {"begin", "mirror", "end"}:
                    continue
                block.append([float(v) for v in parts[:4]])
                if len(parts) >= 5:
                    group = int(parts[4])
                    xyz = np.asarray(block, dtype=np.float64)
                    scale = 1.0 if group_currents is None else float(group_currents.get(group, 0.0))
                    current = scale * xyz[0, 3]
                    pts = xyz[:-1, :3] if np.allclose(xyz[0, :3], xyz[-1, :3]) else xyz[:, :3]
                    if current != 0.0:
                        filaments.append(pts)
                        currents.append(current)
                        groups.append(group)
                        names.append(parts[5] if len(parts) > 5 else str(group))
                    block = []
        lengths = {f.shape[0] for f in filaments}
        if len(lengths) != 1:
            raise ValueError(f"filaments must share one node count, got {sorted(lengths)}")
        return cls(
            nodes=jnp.asarray(np.stack(filaments)),
            currents=jnp.asarray(currents),
            nfp=nfp,
            stellsym=stellsym,
            meta=_meta(
                kind="makegrid_filaments",
                source=Path(path).name,
                groups=groups,
                names=names,
                group_currents=dict(group_currents or {}),
            ),
            **kwargs,
        )
