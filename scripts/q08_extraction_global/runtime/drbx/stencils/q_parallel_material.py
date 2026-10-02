"""Checked paired scalar views for the bounded Q07 five-field material block."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .q_parallel import sha256_array
from .q_parallel_divergence import prepare_scalar_slots
from .q_traced_gradient import eta_difference_weights


@dataclass(frozen=True)
class MaterialRuntime:
    metadata: dict
    inner: object
    outer: object
    magnetic_L: object
    b_eta: object
    eta_step: object

    @property
    def nbytes(self):
        return (self.inner.nbytes + self.outer.nbytes + self.magnetic_L.nbytes
                + self.b_eta.nbytes + self.eta_step.nbytes)


def prepare_material_transport(inner_q, outer_q):
    """Reuse h/32 centered caps and h/16 extra upwind points; never retrace.

    Both views must carry identical geometry, complete-owner projection,
    center and frozen repair choices. No field-dependent selection is allowed.
    """
    a, b = inner_q.validate(), outer_q.validate()
    if a.metadata['span'] != 1/32 or b.metadata['span'] != 1/16:
        raise ValueError('expected inner h/32 and outer h/16')
    for key in ('n', 'n_owner', 'geometry_identity', 'topology_hash',
                'source_identity', 'trace_hash', 'campaign_identity'):
        if a.metadata[key] != b.metadata[key]:
            raise ValueError(f'paired source mismatch: {key}')
    for key in ('raw', 'owners', 'owner_raw', 'owner_weight', 'choice'):
        if not np.array_equal(getattr(a, key), getattr(b, key)):
            raise ValueError(f'paired support/projection mismatch: {key}')
    for key in ('slot_points', 'magnetic_b'):
        if not np.array_equal(getattr(a, key)[:, 2], getattr(b, key)[:, 2]):
            raise ValueError(f'paired center mismatch: {key}')
    eta_difference_weights(a)
    eta_difference_weights(b)
    inner, outer = prepare_scalar_slots(a), prepare_scalar_slots(b)
    beta = a.magnetic_b[:, 2, 2].copy()
    step = np.asarray((2*np.pi/a.metadata['n'])/64)
    arrays = dict(magnetic_L=sha256_array(a.magnetic_L),
                  b_eta=sha256_array(beta), eta_step=sha256_array(step))
    contract = dict(schema='drbx.q-material.v1', inner=inner.metadata['identity'],
        outer=outer.metadata['identity'], arrays=arrays,
        products='primitive slots; coefficients at raw center; project last',
        potential='G(phi)+tau*G(Ti) from same inner slots',
        correction='five-point oriented characteristic upwind minus centered',
        production=False)
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return MaterialRuntime({**contract, 'identity': identity}, inner, outer,
                           a.magnetic_L, beta, step)
