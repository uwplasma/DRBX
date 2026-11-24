"""Prepare the manufactured field with explicitly supplied perpendicular and parallel inputs."""
from ..options import WALL_LAW_NAMES
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .manufactured_field_data import flow_inputs, C_J, FIELD_PROFILES, _hit_parallel_jets
from ..evaluators.magnetic_field import Evaluator

def _parallel_geometry(evaluator,nodes,hits):
 from ..evaluators.magnetic_derivatives import FD_STEPS, magnetic_derivatives
 rows={}
 for prefix,points in [('',nodes),('hit_',hits)]:
  if len(points):
   mag=magnetic_derivatives(evaluator,points,FD_STEPS[-1])
   b=mag['b'];c=mag['along_db']-mag['along_logmag'][:,None]*b
   rows.update({prefix+'b':b,prefix+'c':c,prefix+'along_db':mag['along_db'],prefix+'along_logmag':mag['along_logmag']})
  else:
   rows.update({prefix+k:np.empty((0,3)) for k in ('b','c','along_db')})
   rows[prefix+'along_logmag']=np.empty(0)
 return rows


@dataclass(frozen=True)
class MmsPreparationOptions:
    field_profile:str='wall_flat_temperature'
    wall_laws:tuple[str,...]=('conducting_wall','floating_wall')
    current_coefficient:float=C_J

    def __post_init__(self):
        if (self.field_profile not in FIELD_PROFILES or not self.wall_laws
                or len(set(self.wall_laws))!=len(self.wall_laws)
                or any(wall_law not in WALL_LAW_NAMES for wall_law in self.wall_laws)):
            raise ValueError('Invalid owner MMS field profile or wall laws')


def prepare_owners(out,owner_p,read_q,field_manifest,*,flow,chi,D,
                   options=MmsPreparationOptions(),devices=None):
    """Create MMS data on the caller's CPU devices, one canonical owner at a time."""
    import jax
    from types import SimpleNamespace
    from ..operator_build.parallel.job_storage import atomic_json
    from .cache import _OWNER_SCHEMA, WALL_LAW_MEMBER_NAMES, _manufactured_field_members, _save_owner_members, digest
    from .manufactured_field_data import _manufactured_field_rows

    E,P,N=(owner_p.manifest[key] for key in ('n_eta','P','N'))
    parts=owner_p.parts
    edges=tuple(owner_p.edges)
    if len(edges)!=parts+1 or not 1<=parts<=E//4:
        raise ValueError('At least four planes are required per owner')
    if edges!=tuple((p*E+parts-1)//parts for p in range(parts+1)):
        raise ValueError('perpendicular and parallel owner cuts must use canonical ceiling edges')
    inputs=field_manifest['files']
    if ((field_manifest['N'],field_manifest['eta_planes'],field_manifest['symmetry'],
         field_manifest['field_periods'])!=(N,E,owner_p.manifest['symmetry'],
                                          owner_p.manifest['field_periods'])):
        raise ValueError('Owner P and continuous field manifests differ')
    devices=tuple(jax.devices('cpu')[:parts] if devices is None else devices)
    if (len(devices)!=parts or len(set(devices))!=parts
            or any(d.platform!='cpu' or d.process_index!=jax.process_index() for d in devices)):
        raise ValueError('Supply one distinct local CPU device per MMS owner')
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    flow=Path(flow)
    leaves=('points','wall_points','h','jac','K','B','A','divA','wall_A_row','wall_J','wall_ginv_u')
    owners=[]
    replicated=None
    for owner,(begin,end) in enumerate(zip(edges[:-1],edges[1:])):
        owner_p.verify(owner,leaves=leaves)
        ref=SimpleNamespace(**owner_p.read(owner,leaves=leaves))
        ids,hit=map(np.asarray,read_q(owner))
        n=(end-begin)*P
        if (ref.points.shape!=(end-begin,P,3) or ref.wall_points.shape!=(end-begin,N,3)
                or ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0)
                or np.any(ids[1:]<=ids[:-1]) or hit.shape!=(len(ids),3)):
            raise ValueError('Owner nodal, P-wall or Q-hit inputs have incompatible layouts')
        with jax.default_device(devices[owner]):
            ev=Evaluator(inputs['metric_cache'],inputs['field_table'])
            parallel=_parallel_geometry(ev,ref.points.reshape(n,3),hit)
            divb=-parallel['along_logmag']
            coeff,modes=flow_inputs(flow)
            chi_local=jax.device_put(chi,devices[owner]);D_local=jax.device_put(D,devices[owner])
            if replicated is None:
                # The exact-plane map replicates its in-plane series and its eta spline; the eta-filtered map
                # is one series.
                from ..evaluators.exact_plane_metric import JaxExactPlaneMetricEvaluator
                metric=ev.tables[0]
                base,spline=(metric.tree_flatten()[0] if isinstance(metric,JaxExactPlaneMetricEvaluator)
                             else (metric,()))
                def tables(label,pytree):
                    return {f'{label}_{i}':dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes)
                            for i,a in enumerate(jax.tree_util.tree_leaves(pytree))}
                replicated={**tables('metric_base',base),**tables('eta_spline',spline),
                            **tables('field_table',ev.tables[1]),**tables('flow_coeff',coeff),
                            **tables('chi',chi_local),**tables('D',D_local)}
            groups=dict(parallel_geometry=parallel,hit_layout=dict(ids=ids))
            hit_jets={}
            for wall_law in options.wall_laws:
                data,jets=_manufactured_field_rows(ref,hit,parallel,divb,coeff,modes,chi_local,D_local,
                            wall_law=wall_law,field_profile=options.field_profile,c_j=options.current_coefficient)
                name, first, second = WALL_LAW_MEMBER_NAMES[wall_law]
                groups[name]=_manufactured_field_members(data)
                hit_jets[first],hit_jets[second]=_hit_parallel_jets(jets,parallel,wall_law=wall_law,field_profile=options.field_profile)
            groups['wall_hit_parallel_derivatives']=hit_jets
            files=_save_owner_members(out,owner,groups)
        owners.append(dict(begin=begin,end=end,node_count=n,wall_count=(end-begin)*N,
                           hit_count=len(ids),files=files))
    manifest=dict(schema=_OWNER_SCHEMA,N=N,eta_planes=E,nodes_per_plane=P,wall_per_plane=N,
                  symmetry=field_manifest['symmetry'],field_periods=field_manifest['field_periods'],
                  edges=edges,owner_count=parts,field_profile=options.field_profile,wall_laws=options.wall_laws,
                  owners=owners,replicated=replicated,
                  source=dict(manufactured_ion_flow_fit_sha256=digest(flow),owner_perpendicular_manifest_sha256=digest(owner_p.root/'manifest.json'),
                              **{label:digest(inputs[key]) for key,label in (('metric_cache','metric_cache_sha256'),('field_table','field_table_sha256'))}))
    atomic_json(out/'manifest.json',manifest)
    return manifest
