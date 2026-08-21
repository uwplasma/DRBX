"""Matrix-free Galerkin transfers for source-owned FCI face fields.

This deliberately contains no FCI tracing or boundary policy.  A caller
supplies the *homogeneous*, fine source-edge gradient ``G_f``.  The helpers
then put its cell and edge arguments in a plane-local control-volume owner
space and form its weighted adjoint with ``jax.linear_transpose``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from ..geometry import LocalControlVolumeCellGeometry3D
from .fci_operators import (
    LocalOutgoingFciFaceTopology3D,
    expand_local_control_volume_owner_field,
    prolong_local_outgoing_fci_face_owner_field,
    restrict_local_outgoing_fci_face_field,
)


FineGradient = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class LocalFciFaceGalerkinTransfer:
    """Cell/face owner transfers and diagonal Galerkin measures.

    Cell and face ownership are intentionally separate.  ``cells`` supplies
    ``P_c/R_c``; ``face_topology`` supplies the explicit source-edge
    ``P_e/R_e`` and edge/aggregate measures.  Thus a face aggregate may
    differ from its source cell aggregate without losing Galerkin adjointness.
    """

    cells: LocalControlVolumeCellGeometry3D
    face_topology: LocalOutgoingFciFaceTopology3D

    def __post_init__(self) -> None:
        if not isinstance(self.cells, LocalControlVolumeCellGeometry3D):
            raise TypeError("cells must be LocalControlVolumeCellGeometry3D")
        if not isinstance(self.face_topology, LocalOutgoingFciFaceTopology3D):
            raise TypeError("face_topology must be LocalOutgoingFciFaceTopology3D")
        if self.face_topology.layout != self.cells.layout:
            raise ValueError("cells and face_topology must share a HaloLayout3D")

    @property
    def cell_mass(self) -> jnp.ndarray:
        """The coarse cell mass ``M_c`` (aggregate fluid volume)."""

        return jnp.asarray(self.cells.aggregate_volume, dtype=jnp.float64)

    @property
    def active_owner(self) -> jnp.ndarray:
        return jnp.asarray(self.cells.is_active_owner, dtype=bool)

    @property
    def active_face_owner(self) -> jnp.ndarray:
        return jnp.asarray(self.face_topology.is_active_owner, dtype=bool)

    def _check_owner_local(self) -> None:
        """Reject a transfer requiring a remote owner halo.

        The eta-only RLP lowerings have local owners by construction.  A
        future remote-owner variant needs a reverse owner scatter, which must
        not be silently approximated by indexed local adds here.
        """

        try:
            if bool(jnp.any(self.cells.owner_is_remote)):
                raise ValueError(
                    "LocalFciFaceGalerkinTransfer requires local owner maps; "
                    "remote owner maps need distributed owner restriction"
                )
        except jax.errors.TracerBoolConversionError:
            pass

    def cell_prolong(self, values_owner: jnp.ndarray) -> jnp.ndarray:
        """``P_c``: prolong an owner cell field to fine storage cells."""

        values = jnp.asarray(values_owner, dtype=jnp.float64)
        if values.shape != self.cells.shape:
            raise ValueError("values_owner must match cells.shape")
        return expand_local_control_volume_owner_field(
            values, self.cells,
        )

    def face_prolong(self, values_owner: jnp.ndarray) -> jnp.ndarray:
        """``P_e``: prolong a source-aggregate face field to fine edges."""

        return prolong_local_outgoing_fci_face_owner_field(
            values_owner, self.face_topology
        )

    def cell_restrict(self, fine_values: jnp.ndarray) -> jnp.ndarray:
        """``R_c=M_c^-1 P_c^T M_f`` for fine cell-average values."""

        return self._restrict_cell(fine_values)

    def face_restrict(self, fine_values: jnp.ndarray) -> jnp.ndarray:
        """``R_e=M_e^-1 P_e^T W_e`` for source-owned fine edge values."""

        return restrict_local_outgoing_fci_face_field(fine_values, self.face_topology)

    def _restrict_cell(self, fine_values: jnp.ndarray) -> jnp.ndarray:
        self._check_owner_local()
        values = jnp.asarray(fine_values, dtype=jnp.float64)
        if values.shape != self.cells.shape:
            raise ValueError("fine_values must match cells.shape")
        summed = jnp.zeros_like(values).at[
            self.cells.owner_i, self.cells.owner_j, self.cells.owner_k
        ].add(jnp.asarray(self.cells.raw_volume, dtype=jnp.float64) * values)
        result = summed / jnp.maximum(self.cell_mass, 1.0e-30)
        return jnp.where(self.active_owner, result, 0.0)

    def coarse_gradient(
        self,
        values_owner: jnp.ndarray,
        fine_gradient: FineGradient,
    ) -> jnp.ndarray:
        """Return ``G_c = R_e G_f P_c`` in source-aggregate face space."""

        fine = jnp.asarray(fine_gradient(self.cell_prolong(values_owner)), dtype=jnp.float64)
        if fine.shape != self.cells.shape:
            raise ValueError("fine_gradient must return an array with cells.shape")
        return self.face_restrict(fine)

    def coarse_divergence(
        self,
        face_values_owner: jnp.ndarray,
        fine_gradient: FineGradient,
    ) -> jnp.ndarray:
        """Return ``D_c=-M_c^-1 G_c^T M_e`` matrix-free.

        ``fine_gradient`` must be a pure homogeneous linear map.  Affine wall
        traces belong in a separate RHS and must not be captured here.
        """

        q = jnp.asarray(face_values_owner, dtype=jnp.float64)
        if q.shape != self.cells.shape:
            raise ValueError("face_values_owner must match cells.shape")
        zero_cell = jnp.zeros(self.cells.shape, dtype=q.dtype)

        def coarse_map(cell_values):
            return self.coarse_gradient(cell_values, fine_gradient)

        adjoint = jax.linear_transpose(coarse_map, zero_cell)(
            self.face_topology.aggregate_measure * q
        )[0]
        result = -adjoint / jnp.maximum(self.cell_mass, 1.0e-30)
        return jnp.where(self.active_owner, result, 0.0)

    def fine_divergence(self, fine_face_values: jnp.ndarray, fine_gradient: FineGradient) -> jnp.ndarray:
        """Reference ``D_f=-M_f^-1 G_f^T W_e`` used for validation."""

        q = jnp.asarray(fine_face_values, dtype=jnp.float64)
        if q.shape != self.cells.shape:
            raise ValueError("fine_face_values must match cells.shape")
        zero = jnp.zeros(self.cells.shape, dtype=q.dtype)
        adjoint = jax.linear_transpose(fine_gradient, zero)(
            self.face_topology.edge_measure * q
        )[0]
        return -adjoint / jnp.maximum(jnp.asarray(self.cells.raw_volume), 1.0e-30)


def build_local_fci_face_galerkin_transfer(
    cells: LocalControlVolumeCellGeometry3D,
    face_topology: LocalOutgoingFciFaceTopology3D,
) -> LocalFciFaceGalerkinTransfer:
    """Pair cell-owner transfers with an explicit outgoing-face topology.

    The topology owns ``W_e=edge_measure`` and
    ``M_e=aggregate_measure=P_e.T W_e``.  For a field-line edge of length
    ``ell`` and transverse area ``A_perp``, its edge measure is
    ``A_perp*ell``; the compatible first value is the fluid dual volume.
    Magnetic factors belong in ``F/B``, not in this geometric measure.
    """

    return LocalFciFaceGalerkinTransfer(cells, face_topology)


__all__ = [
    "FineGradient",
    "LocalFciFaceGalerkinTransfer",
    "build_local_fci_face_galerkin_transfer",
]
