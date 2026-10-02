"""Host preparation of bounded traced-Q direct and tube divergence candidates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
import numpy as np

from .q_parallel import PreparedQ, sha256_array
from .q_parallel_gradient import prepare_parallel_gradient

SCHEMA = 'drbx.q-bounded-divergence.v1'
KAPPA_STEPS = (1e-4, 5e-5, 2.5e-5)
_FD4 = np.array([1., -8., 8., -1.]) / 12.


@dataclass(frozen=True)
class KappaAudit:
    kappa: np.ndarray                 # baseline div(b), one per raw midpoint
    step_values: np.ndarray           # (3, n_raw), independent step lengths
    identity_value: np.ndarray        # frozen div(B)/B - b dot grad(log B)
    magnetic_B: np.ndarray            # midpoint magnitude
    response_hashes: tuple[str, ...]  # center, then each step response

    @property
    def identity_discrepancy(self):
        return self.kappa - self.identity_value

    @property
    def step_sensitivity(self):
        return np.max(np.abs(self.step_values - self.kappa[None, :]), axis=0)


def prepare_kappa(prepared_q: PreparedQ, geom, *, steps=KAPPA_STEPS,
                  center_b_atol=1e-12) -> KappaAudit:
    """Differentiate J*b at raw midpoints using frozen geometry and fourth-order FD.

    Independently evaluate the continuum product identity using saved div(B)/B
    and a separate fourth-order derivative of log(B). The difference is an
    uncertainty diagnostic, not a cancellation enforced in coefficients.
    ``center_b_atol`` controls only the geometry replay check, not arithmetic.
    """
    q = prepared_q.validate()
    if tuple(steps) != KAPPA_STEPS:
        raise ValueError('frozen kappa audit steps required')
    if not np.isfinite(center_b_atol) or center_b_atol < 0:
        raise ValueError('finite nonnegative center_b_atol required')
    points = q.slot_points[:, 2]
    J0, b0, B0 = (np.asarray(x) for x in geom(points))
    nr = len(q.raw)
    if J0.shape != (nr,) or b0.shape != (nr, 3) or B0.shape != (nr,):
        raise ValueError('geometry center response shape mismatch')
    if (not np.isfinite(J0).all() or not np.isfinite(b0).all() or
        not np.isfinite(B0).all() or np.any(J0 <= 0) or np.any(B0 <= 0)):
        raise ValueError('invalid center geometry response')
    if not np.allclose(b0, q.magnetic_b[:, 2], rtol=0, atol=center_b_atol):
        raise ValueError('center magnetic b differs from checked Q source')
    hashes = [sha256_array(J0)+':'+sha256_array(b0)+':'+sha256_array(B0)]
    values = []
    identity = None
    for step in steps:
        displaced = np.broadcast_to(points[:, None, None, :], (nr, 3, 4, 3)).copy()
        for a in range(3):
            displaced[:, a, :, a] += np.array([-2., -1., 1., 2.]) * step
        J, b, B = (np.asarray(x) for x in geom(displaced.reshape(-1, 3)))
        J = J.reshape(nr, 3, 4)
        b = b.reshape(nr, 3, 4, 3)
        B = B.reshape(nr, 3, 4)
        if (not np.isfinite(J).all() or not np.isfinite(b).all() or
            not np.isfinite(B).all() or np.any(J <= 0) or np.any(B <= 0)):
            raise ValueError('invalid displaced geometry response')
        weights = _FD4 / step
        numerator = sum(np.einsum('rq,q->r', J[:, a]*b[:, a, :, a], weights)
                        for a in range(3))
        values.append(numerator/J0)
        hashes.append(sha256_array(J)+':'+sha256_array(b)+':'+sha256_array(B))
        if identity is None:
            grad_log_B = np.stack([np.einsum('rq,q->r', np.log(B[:, a]), weights)
                                   for a in range(3)], axis=-1)
            identity = q.magnetic_L[:, 2] - np.einsum('ra,ra->r', b0, grad_log_B)
    step_values = np.stack(values)
    if not np.isfinite(step_values).all() or not np.isfinite(identity).all():
        raise ValueError('nonfinite kappa or magnetic identity')
    return KappaAudit(step_values[0], step_values, identity, B0, tuple(hashes))


@dataclass(frozen=True)
class ParallelDivergenceRuntime:
    metadata: dict
    raw: object
    donor: object
    owner_raw: object
    owner_weight: object
    coefficient_D: object
    coefficient_N: object
    lift_D_node: object
    lift_D_query: object
    lift_D_tangent: object
    lift_N_normal: object

    @property
    def nbytes(self):
        return sum(getattr(self, name).nbytes for name in (
            'raw','donor','owner_raw','owner_weight','coefficient_D','coefficient_N',
            'lift_D_node','lift_D_query','lift_D_tangent','lift_N_normal'))


@dataclass(frozen=True)
class PreparedParallelDivergence:
    metadata: dict
    kappa_audit: KappaAudit
    direct: ParallelDivergenceRuntime
    tube: ParallelDivergenceRuntime
    scalar_slots: ParallelDivergenceRuntime

    def runtime_view(self, candidate: str) -> ParallelDivergenceRuntime:
        if candidate == 'direct': return self.direct
        if candidate == 'tube': return self.tube
        raise ValueError('candidate must be direct or tube')

    def scalar_view(self) -> ParallelDivergenceRuntime:
        """Three-slot value diagnostic, not an owner-centered scalar action."""
        return self.scalar_slots


def prepare_tube_divergence(prepared_q: PreparedQ) -> ParallelDivergenceRuntime:
    """Contract only tube rows, without kappa audits or new geometry evaluation."""
    q=prepared_q.validate()
    wall=q.raw//(q.metadata['n']**2)>=q.metadata['n']-2
    if (np.any(q.boundary_value_D_trace[~wall]!=0) or
        np.any(q.boundary_value_N_normal[~wall]!=0)):
        raise ValueError('non-wall scalar boundary lift')
    hashes={k:sha256_array(getattr(q,k)) for k in (
        'raw','donor','owner_raw','owner_weight','row_value_D','row_value_N',
        'boundary_value_D_trace','boundary_value_N_normal','magnetic_L','choice')}
    source={k:q.metadata[k] for k in ('schema','campaign_identity','source_identity',
        'geometry_identity','topology_hash','trace_hash','span','n','n_owner')}
    identity=hashlib.sha256(json.dumps(dict(source=source,arrays=hashes,
        operator='tube-only-v1'),sort_keys=True).encode()).hexdigest()
    metadata={**source,'schema':SCHEMA,'identity':identity,'candidate':'tube',
              'slots':('raw_midpoint',),'source_array_hashes':hashes}
    view=ParallelDivergenceRuntime(metadata,q.raw,q.donor,q.owner_raw,q.owner_weight,
        np.einsum('rs,rsd->rd',q.magnetic_L,q.row_value_D),
        np.einsum('rs,rsd->rd',q.magnetic_L,q.row_value_N),
        np.einsum('rs,rsj->rj',q.magnetic_L,q.boundary_value_D_trace),
        q.magnetic_L*wall[:,None],np.zeros((len(q.raw),2)),
        np.einsum('rs,rsj->rj',q.magnetic_L,q.boundary_value_N_normal))
    for name in ('coefficient_D','coefficient_N','lift_D_node','lift_D_query',
                 'lift_D_tangent','lift_N_normal'):
        if not np.isfinite(getattr(view,name)).all():
            raise ValueError('nonfinite tube coefficient')
    return view


def prepare_scalar_slots(prepared_q: PreparedQ) -> ParallelDivergenceRuntime:
    """Reuse checked cap/center value rows without magnetic derivative audits.

    These rows can be shared by nonlinear transport consumers. No new fits,
    geometry calls, gradients, or changes to the selected Q operators occur.
    """
    q = prepared_q.validate()
    wall = q.raw // q.metadata['n']**2 >= q.metadata['n']-2
    if (np.any(q.boundary_value_D_trace[~wall] != 0) or
            np.any(q.boundary_value_N_normal[~wall] != 0)):
        raise ValueError('non-wall scalar boundary lift')
    names = ('raw', 'donor', 'owner_raw', 'owner_weight', 'row_value_D',
             'row_value_N', 'boundary_value_D_trace', 'boundary_value_N_normal')
    hashes = {name: sha256_array(getattr(q, name)) for name in names}
    source = {k: q.metadata[k] for k in ('schema', 'campaign_identity',
        'source_identity', 'geometry_identity', 'topology_hash', 'trace_hash',
        'span', 'n', 'n_owner')}
    identity = hashlib.sha256(json.dumps(dict(source=source, arrays=hashes,
        operator='scalar-slots-v1'), sort_keys=True).encode()).hexdigest()
    metadata = {**source, 'schema': SCHEMA, 'identity': identity,
        'candidate': 'scalar_slots', 'slots': ('minus_cap', 'plus_cap', 'raw_midpoint'),
        'source_array_hashes': hashes}
    return ParallelDivergenceRuntime(metadata, q.raw, q.donor, q.owner_raw,
        q.owner_weight, q.row_value_D, q.row_value_N, q.boundary_value_D_trace,
        np.broadcast_to(wall[:, None].astype(float), (len(q.raw), 3)).copy(),
        np.zeros((len(q.raw), 3, 2)), q.boundary_value_N_normal)


def prepare_parallel_divergence(prepared_q: PreparedQ, geom) -> PreparedParallelDivergence:
    """Build both fixed candidates from checked Q05 rows and same frozen geom."""
    q = prepared_q.validate()
    start=time.perf_counter()
    audit = prepare_kappa(q, geom)
    geometry_seconds=time.perf_counter()-start
    start=time.perf_counter()
    gradient = prepare_parallel_gradient(q)
    nr = len(q.raw)
    wall = q.raw // (q.metadata['n']**2) >= q.metadata['n']-2
    if (np.any(q.boundary_value_D_trace[~wall] != 0) or
        np.any(q.boundary_value_N_normal[~wall] != 0)):
        raise ValueError('non-wall scalar boundary lift')
    query = np.broadcast_to(wall[:, None].astype(float), (nr, 3)).copy()
    direct_query = np.zeros_like(query)
    direct_query[:, 2] = audit.kappa * query[:, 2]
    kappa = audit.kappa
    source_arrays = {name: sha256_array(getattr(q, name)) for name in (
        'row_value_D','row_value_N','boundary_value_D_trace',
        'boundary_value_N_normal','magnetic_L','slot_points','magnetic_b')}
    identity = hashlib.sha256(json.dumps(dict(
        schema=SCHEMA,gradient_identity=gradient.metadata['identity'],
        source_arrays=source_arrays,geometry_responses=audit.response_hashes,
        kappa_steps=KAPPA_STEPS),sort_keys=True).encode()).hexdigest()
    metadata = dict(schema=SCHEMA,identity=identity,
        source_schema=q.metadata['schema'],campaign_identity=q.metadata['campaign_identity'],
        source_identity=q.metadata['source_identity'],geometry_identity=q.metadata['geometry_identity'],
        topology_hash=q.metadata['topology_hash'],trace_hash=q.metadata['trace_hash'],
        span=q.metadata['span'],n=q.metadata['n'],n_owner=q.metadata['n_owner'],
        gradient_identity=gradient.metadata['identity'],kappa_steps=KAPPA_STEPS,
        source_array_hashes=source_arrays,geometry_response_hashes=audit.response_hashes)
    common=(q.raw,q.donor,q.owner_raw,q.owner_weight)
    direct=ParallelDivergenceRuntime({**metadata,'candidate':'direct','slots':('raw_midpoint',)},
        *common,
        gradient.coefficient_D[:,2]+kappa[:,None]*q.row_value_D[:,2],
        gradient.coefficient_N[:,2]+kappa[:,None]*q.row_value_N[:,2],
        gradient.lift_D_node[:,2]+kappa[:,None]*q.boundary_value_D_trace[:,2],
        direct_query,gradient.lift_D_tangent[:,2],
        gradient.lift_N_normal[:,2]+kappa[:,None]*q.boundary_value_N_normal[:,2])
    tube=ParallelDivergenceRuntime({**metadata,'candidate':'tube','slots':('raw_midpoint',)},
        *common,
        np.einsum('rs,rsd->rd',q.magnetic_L,q.row_value_D),
        np.einsum('rs,rsd->rd',q.magnetic_L,q.row_value_N),
        np.einsum('rs,rsj->rj',q.magnetic_L,q.boundary_value_D_trace),
        q.magnetic_L*query,
        np.zeros((nr,2)),
        np.einsum('rs,rsj->rj',q.magnetic_L,q.boundary_value_N_normal))
    scalar=ParallelDivergenceRuntime({**metadata,'candidate':'scalar_slots',
                                      'slots':('minus_cap','plus_cap','raw_midpoint')},
        *common,q.row_value_D,q.row_value_N,q.boundary_value_D_trace,
        query,np.zeros((nr,3,2)),q.boundary_value_N_normal)
    for view in (direct,tube,scalar):
        for name in ('coefficient_D','coefficient_N','lift_D_node','lift_D_query',
                     'lift_D_tangent','lift_N_normal'):
            if not np.isfinite(getattr(view,name)).all():
                raise ValueError('nonfinite divergence coefficient')
    metadata['geometry_seconds']=geometry_seconds
    metadata['row_contraction_seconds']=time.perf_counter()-start
    return PreparedParallelDivergence(metadata,audit,direct,tube,scalar)
