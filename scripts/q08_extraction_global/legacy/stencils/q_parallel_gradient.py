"""Host contraction of frozen Q directional rows for direct parallel gradient.

The source is a checked schema-v2 :class:`PreparedQ`. This module does no
tracing, reconstruction, or fitting. No separate coefficient artifact is saved.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np

from .q_parallel import PreparedQ, sha256_array

GRADIENT_SCHEMA = 'drbx.q-direct-gradient.v1'
SLOT_ORDER = ('minus_cap', 'plus_cap', 'raw_midpoint')


@dataclass(frozen=True)
class PreparedParallelGradient:
    metadata: dict
    raw: np.ndarray
    donor: np.ndarray
    owner_raw: np.ndarray
    owner_weight: np.ndarray
    coefficient_D: np.ndarray     # (raw, slot, donor)
    coefficient_N: np.ndarray
    lift_D_node: np.ndarray       # (raw, slot, wall node)
    lift_D_tangent: np.ndarray    # (raw, slot, theta/eta)
    lift_N_normal: np.ndarray     # (raw, slot, wall node)

    def runtime_view(self, *, include_caps: bool = False):
        """Return only application arrays; three slots are diagnostic only."""
        slots = slice(None) if include_caps else slice(2, 3)
        return ParallelGradientRuntime(
            {**self.metadata, 'slot_order': SLOT_ORDER if include_caps else ('raw_midpoint',)},
            self.raw, self.donor, self.owner_raw, self.owner_weight,
            self.coefficient_D[:, slots], self.coefficient_N[:, slots],
            self.lift_D_node[:, slots], self.lift_D_tangent[:, slots],
            self.lift_N_normal[:, slots])


@dataclass(frozen=True)
class ParallelGradientRuntime:
    metadata: dict
    raw: object
    donor: object
    owner_raw: object
    owner_weight: object
    coefficient_D: object
    coefficient_N: object
    lift_D_node: object
    lift_D_tangent: object
    lift_N_normal: object

    @property
    def nbytes(self):
        return sum(getattr(self, name).nbytes for name in (
            'raw', 'donor', 'owner_raw', 'owner_weight', 'coefficient_D',
            'coefficient_N', 'lift_D_node', 'lift_D_tangent', 'lift_N_normal'))


def prepare_parallel_gradient(prepared_q: PreparedQ) -> PreparedParallelGradient:
    """Contract the stored logical gradient and b at both caps and midpoint."""
    if not isinstance(prepared_q, PreparedQ):
        raise TypeError('a checked PreparedQ source is required')
    q = prepared_q.validate()
    n = q.metadata['n']
    ijk = np.column_stack(np.unravel_index(q.raw, (n, n, n)))
    center = np.column_stack(((ijk[:, 0] + .5) / n,
                              (ijk[:, 1] + .5) * (2*np.pi/n),
                              (ijk[:, 2] + .5) * (2*np.pi/n)))
    if q.slot_points.shape != (len(q.raw), 3, 3) or q.magnetic_b.shape != q.slot_points.shape:
        raise ValueError('expected minus cap, plus cap, raw midpoint slots')
    if not np.allclose(q.slot_points[:, 2], center, rtol=0, atol=2e-14):
        raise ValueError('slot 2 is not the raw midpoint')
    separation = (q.slot_points[:, 1, 2] - q.slot_points[:, 0, 2])
    if not np.allclose(separation, q.metadata['span'] * (2*np.pi/n), rtol=0, atol=2e-14):
        raise ValueError('cap order or frozen span mismatch')
    wall = q.raw // (n*n) >= n-2
    d_tangent = np.where(wall[:, None, None], q.magnetic_b[:, :, 1:], 0.)
    if np.any(q.boundary_gradient_D_trace[~wall] != 0) or np.any(q.boundary_gradient_N_normal[~wall] != 0):
        raise ValueError('non-wall gradient boundary lift')
    # Both operands are logical-coordinate quantities. No metric conversion.
    contract = lambda rows: np.einsum('rsa,rsad->rsd', q.magnetic_b, rows)
    lift = lambda rows: np.einsum('rsa,rsaj->rsj', q.magnetic_b, rows)
    source_hashes = {name: sha256_array(getattr(q, name)) for name in (
        'raw', 'raw_to_owner', 'raw_weight', 'donor', 'mask', 'row_count',
        'choice', 'owner_raw', 'owner_weight', 'slot_points', 'magnetic_b',
        'row_gradient_D', 'row_gradient_N', 'boundary_gradient_D_trace',
        'boundary_gradient_N_normal')}
    identity = hashlib.sha256(json.dumps({
        'gradient_schema': GRADIENT_SCHEMA,
        'source_metadata': {k: q.metadata[k] for k in (
            'schema', 'campaign_identity', 'source_identity', 'geometry_identity',
            'topology_hash', 'trace_hash', 'span', 'n')},
        'source_arrays': source_hashes}, sort_keys=True).encode()).hexdigest()
    metadata = dict(schema=GRADIENT_SCHEMA, identity=identity,
                    source_schema=q.metadata['schema'],
                    campaign_identity=q.metadata['campaign_identity'],
                    source_identity=q.metadata['source_identity'],
                    geometry_identity=q.metadata['geometry_identity'],
                    topology_hash=q.metadata['topology_hash'],
                    trace_hash=q.metadata['trace_hash'],
                    span=q.metadata['span'], n=n, n_owner=q.metadata['n_owner'],
                    choice_counts={str(int(k)): int(np.count_nonzero(q.choice == k))
                                   for k in np.unique(q.choice)},
                    source_array_hashes=source_hashes)
    return PreparedParallelGradient(metadata, q.raw, q.donor, q.owner_raw,
                                    q.owner_weight, contract(q.row_gradient_D),
                                    contract(q.row_gradient_N),
                                    lift(q.boundary_gradient_D_trace), d_tangent,
                                    lift(q.boundary_gradient_N_normal))
