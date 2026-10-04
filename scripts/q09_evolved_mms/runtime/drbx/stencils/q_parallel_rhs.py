"""Preparation for the prescribed-boundary six-field Q07 research assembly.

The material/current divergence uses the qualified geometry-consistent center
weight. The accepted cap-gradient diffusion keeps its original weights.
"""
from dataclasses import dataclass, fields, replace
import hashlib
import json
import numpy as np

from .q_parallel import sha256_array
from .q_parallel_divergence import prepare_kappa
from .q_parallel_material import prepare_material_transport


@dataclass(frozen=True)
class SixFieldRuntime:
    metadata: dict
    material: object
    diffusion: object
    bmag: object


def prepare_six_field_rhs(inner_q, outer_q, geom, *, diffusion_span,
                          center_b_atol=1e-12):
    """Reuse checked rows; explicitly select one of the two diffusion spans.

    Geometry must be the evaluator used to prepare the input rows. No donor
    selection, tracing, boundary-law choice or production registration occurs.
    The returned kappa audit remains host evidence, outside the JAX runtime.
    """
    material = prepare_material_transport(inner_q, outer_q)
    if diffusion_span not in (1/16, 1/32):
        raise ValueError('explicit accepted diffusion span required')
    audit = prepare_kappa(inner_q, geom, center_b_atol=center_b_atol)
    weights = np.array(material.magnetic_L, copy=True)
    weights[:, 2] = audit.kappa - weights[:, 0] - weights[:, 1]
    if not np.isfinite(weights).all():
        raise ValueError('nonfinite balanced material weights')
    contract = dict(base=material.metadata['identity'],
                    formula='L_center=div(b)-L_minus-L_plus',
                    weights=sha256_array(weights),
                    geometry_responses=audit.response_hashes)
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    material = replace(material, magnetic_L=weights, metadata={
        **material.metadata, 'identity': identity, 'balanced_tube': contract})
    diffusion_q = inner_q if diffusion_span == 1/32 else outer_q
    diffusion = diffusion_q.runtime_view()
    meta = dict(schema='drbx.q-six-field-prescribed.v1',
                material_identity=identity,
                diffusion_span=diffusion_span,
                diffusion_arrays={f.name: sha256_array(getattr(diffusion, f.name))
                                  for f in fields(diffusion) if f.name != 'metadata'},
                bmag=sha256_array(audit.magnetic_B),
                fields=('n', 'Te', 'Ti', 'Vi', 'Ve', 'omega'),
                boundary='prescribed primitive D/physical-normal N; no endpoint SAT',
                production=False)
    meta['identity'] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    return SixFieldRuntime(meta, material, diffusion,
                           audit.magnetic_B.copy()), audit
