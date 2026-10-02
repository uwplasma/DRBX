"""Host views for existing Q07 diffusion channels and coefficient-slot audit."""
from dataclasses import dataclass
import hashlib
import json
from .q_parallel import sha256_array
from .q_parallel_gradient import prepare_parallel_gradient

CHANNELS = ('density', 'Te', 'Ti', 'Vi', 'Ve', 'vorticity')


@dataclass(frozen=True)
class CoefficientDiffusionRuntime:
    metadata: dict
    gradient: object
    magnetic_L: object

    @property
    def nbytes(self):
        return self.gradient.nbytes + self.magnetic_L.nbytes


def prepare_coefficient_diffusion(prepared_q):
    """Audit D(chi*cap-gradient(f)) on either accepted diffusion span.

    chi is supplied at the actual three slots at runtime, not multiplied
    after projection. No physical chi(T) model is selected here. Unit chi
    must replay the frozen diffusion; there is no owner-field D(Gf).
    """
    q = prepared_q.validate()
    gradient = prepare_parallel_gradient(q).runtime_view(include_caps=True)
    contract = dict(schema='drbx.q-coefficient-diffusion.v1',
        gradient_identity=gradient.metadata['identity'],
        magnetic_L=sha256_array(q.magnetic_L),span=q.metadata['span'],
        product='coefficient at each slot times accepted cap gradient')
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return CoefficientDiffusionRuntime({**contract, 'identity':identity}, gradient, q.magnetic_L)
