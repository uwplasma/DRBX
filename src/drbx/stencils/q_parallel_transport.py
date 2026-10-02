"""Host preparation for the bounded Q07 centered density transport block.

Research-library consumer of accepted Q rows, not a production RHS selector.
"""
from dataclasses import dataclass
import hashlib
import json

from .q_parallel import sha256_array
from .q_parallel_divergence import prepare_scalar_slots


@dataclass(frozen=True)
class DensityTransportRuntime:
    metadata: dict
    scalar: object
    magnetic_L: object

    @property
    def nbytes(self):
        return self.scalar.nbytes + self.magnetic_L.nbytes


def prepare_density_transport(prepared_q):
    """Build -D(n*Ve) with factors reconstructed at the same three slots.

    Q07 currently fixes total cap separation h/32. Primitive BCs are applied
    before multiplication; no product-owner approximation or product-wall solve
    is introduced. The tube div(B)/B correction uses the center product.
    """
    q = prepared_q.validate()
    if q.metadata['span'] != 1/32:
        raise ValueError('Q07 density transport requires frozen h/32 span')
    scalar = prepare_scalar_slots(q)
    contract = dict(schema='drbx.q-density-transport.v1',
        scalar_identity=scalar.metadata['identity'],
        magnetic_L_sha256=sha256_array(q.magnetic_L),
        product='primitive-slot-product', rhs='-D(n*Ve)',
        characteristic_correction=False, limiter=False)
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return DensityTransportRuntime({**contract, 'identity': identity}, scalar,
                                   q.magnetic_L)
