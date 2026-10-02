"""Prepare a symmetric-eta FCI gradient from frozen scalar cap-value rows.

Only scalar values enter the action. Field-gradient rows, B weights and divB
are not contracted. The shared PreparedQ is validated before extracting this
lean, geometry-only runtime; no tracing, fitting or donor selection occurs.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import hashlib
import json
import numpy as np
from .q_parallel import PreparedQ, sha256_array

SCHEMA = 'drbx.q-traced-gradient.v1'


@dataclass(frozen=True)
class TracedGradientRuntime:
    metadata: dict
    raw: object
    donor: object
    owner_raw: object
    owner_weight: object
    coefficient_D: object
    coefficient_N: object
    lift_D_node: object
    lift_D_query: object
    lift_N_normal: object

    @property
    def nbytes(self):
        return sum(getattr(self, f.name).nbytes for f in fields(self)
                   if f.name != 'metadata')


def eta_difference_weights(q: PreparedQ) -> np.ndarray:
    """Return minus/plus/center weights after checking symmetric eta geometry.

    q must already be validated. b^eta = d eta/d s is contravariant, not a
    component along a unit coordinate vector. Eta must parameterize the trace.
    """
    n = q.metadata['n']
    if q.slot_points.shape != (len(q.raw), 3, 3) or q.magnetic_b.shape != q.slot_points.shape:
        raise ValueError('expected minus, plus, midpoint geometry slots')
    ijk = np.column_stack(np.unravel_index(q.raw, (n, n, n)))
    center = np.column_stack(((ijk[:, 0]+.5)/n,
                             (ijk[:, 1]+.5)*(2*np.pi/n),
                             (ijk[:, 2]+.5)*(2*np.pi/n)))
    width = q.metadata['span']*(2*np.pi/n)
    if not np.allclose(q.slot_points[:, 2], center, rtol=0, atol=2e-14):
        raise ValueError('slot 2 is not the raw midpoint')
    offsets = q.slot_points[:, :2, 2] - center[:, None, 2]
    if not np.allclose(offsets, [-width/2, width/2], rtol=0, atol=2e-14):
        raise ValueError('caps must be symmetric in eta with the declared span')
    beta = q.magnetic_b[:, :, 2]
    if (not np.isfinite(beta).all() or np.any(beta == 0) or
            np.any(np.sign(beta) != np.sign(beta[:, 2:3]))):
        raise ValueError('eta must parameterize the field line without a turning point')
    scale = beta[:, 2]/width
    return np.column_stack((-scale, scale, np.zeros(len(scale))))


def prepare_traced_gradient(prepared_q: PreparedQ) -> TracedGradientRuntime:
    """Build b0^eta * (f_plus-f_minus)/(alpha*deta), including affine BCs."""
    if not isinstance(prepared_q, PreparedQ):
        raise TypeError('a checked PreparedQ source is required')
    q = prepared_q.validate()
    weights = eta_difference_weights(q)
    wall = q.raw//q.metadata['n']**2 >= q.metadata['n']-2
    for name in ('boundary_value_D_trace', 'boundary_value_N_normal'):
        if np.any(getattr(q, name)[~wall] != 0):
            raise ValueError('non-wall scalar boundary lift')
    names = ('raw', 'donor', 'owner_raw', 'owner_weight', 'slot_points',
             'magnetic_b', 'row_value_D', 'row_value_N',
             'boundary_value_D_trace', 'boundary_value_N_normal', 'choice')
    hashes = {name: sha256_array(getattr(q, name)) for name in names}
    source = {k:q.metadata[k] for k in ('schema', 'campaign_identity',
        'source_identity', 'geometry_identity', 'topology_hash', 'trace_hash',
        'span', 'n', 'n_owner')}
    identity = hashlib.sha256(json.dumps(dict(schema=SCHEMA, source=source,
        arrays=hashes), sort_keys=True).encode()).hexdigest()
    metadata = {**source, 'source_schema':source['schema'], 'schema':SCHEMA,
        'identity':identity, 'source_array_hashes':hashes,
        'formula':'b_eta_center * (f_plus - f_minus) / total_eta_span'}
    result = TracedGradientRuntime(metadata, q.raw, q.donor, q.owner_raw, q.owner_weight,
        np.einsum('rs,rsd->rd', weights, q.row_value_D),
        np.einsum('rs,rsd->rd', weights, q.row_value_N),
        np.einsum('rs,rsj->rj', weights, q.boundary_value_D_trace),
        weights*wall[:, None],
        np.einsum('rs,rsj->rj', weights, q.boundary_value_N_normal))
    if any(not np.isfinite(getattr(result, f.name)).all()
           for f in fields(result) if f.name != 'metadata'):
        raise ValueError('nonfinite traced-gradient coefficients')
    return result
