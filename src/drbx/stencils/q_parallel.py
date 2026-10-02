"""Frozen traced-Q host preparation and versioned chunk persistence.

This is a research-library path, not a production solver selector. Preparation
receives complete owner traces and geometry callbacks; no campaign/MMS module is
imported. The accepted selective policy and wall algebra live in Q-specific
modules. Runtime arrays are precontracted, while directional rows remain in the
saved preparation for future gradient/divergence work.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import time
import numpy as np

from .q_parallel_support import Hybrid, BalancedHybrid, LocalHybrid, selective_indicators
from .q_parallel_wall import Wall

SCHEMA = 'drbx.q-traced-diffusion.v2'
FROZEN_IDENTITY = 'c0d64dba761087a3273d29cd4728689c4f4b5b67f838873a845d1cdb1a809f8b'
_ARRAYS = ('owners','raw','raw_to_owner','raw_weight','owner_raw','owner_weight','slot_points',
           'donor','mask','diffusion_D','diffusion_N','boundary_D_node','boundary_D_tangent',
           'boundary_N_normal','boundary_wall_nodes','boundary_wall_queries','boundary_wall_normal',
           'boundary_value_D_trace','boundary_gradient_D_trace','boundary_value_N_normal',
           'boundary_gradient_N_normal',
           'choice','indicator','noise','row_value_D','row_gradient_D','row_value_N','row_gradient_N',
           'magnetic_b','magnetic_L','row_count')


def sha256_array(a):
    x = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def topology_from_arrays(centers, active_owner, aggregate_id, raw_volume):
    """Build frozen compact owner ordering from canonical RLP topology arrays."""
    centers = tuple(np.asarray(x) for x in centers)
    n = len(centers[0])
    if any(len(x) != n for x in centers): raise ValueError('cubic grid required')
    active = np.flatnonzero(np.asarray(active_owner).ravel())
    lookup = np.full(n**3, -1, int); lookup[active] = np.arange(len(active))
    ro = lookup[np.asarray(aggregate_id).ravel()]
    rv = np.asarray(raw_volume).ravel().copy()
    if np.any(ro < 0) or np.any(rv <= 0): raise ValueError('invalid owner topology or volume')
    vol = np.bincount(ro, weights=rv)
    pts = np.stack(np.meshgrid(*centers, indexing='ij'), axis=-1).reshape(-1,3)
    xy = np.column_stack((pts[:,0]*np.cos(pts[:,1]), pts[:,0]*np.sin(pts[:,1])))
    centroid = np.column_stack([np.bincount(ro,weights=rv*xy[:,a])/vol for a in (0,1)])
    order = np.argsort(ro,kind='stable')
    starts = np.r_[0,np.cumsum(np.bincount(ro,minlength=len(vol)))]
    g = SimpleNamespace(dr=1/n,dtheta=2*np.pi/n,deta=2*np.pi/n,eta_period=2*np.pi,owner_centroid_xy=centroid)
    return SimpleNamespace(n=n,centers=centers,ro=ro,rv=rv,vol=vol,pts=pts,xy=xy,order=order,starts=starts,g=g)


def raw_members(t, owners):
    return np.concatenate([t.order[t.starts[o]:t.starts[o+1]] for o in owners])


def magnetic_coefficients(t, raw, ends, geom):
    """Frozen six-slot, two-span b/L construction with finite-difference divB."""
    points = t.pts[raw]
    slots = np.stack((ends[:,0],ends[:,1],points,ends[:,2],ends[:,3],points),axis=1)
    J,b,B = geom(slots.reshape(-1,3)); J=J.reshape(-1,6);b=b.reshape(-1,6,3);B=B.reshape(-1,6)
    step=1e-4
    pp=np.broadcast_to(points[:,None,None,:],(len(points),3,4,3)).copy()
    for a in range(3): pp[:,a,:,a] += np.array([-2,-1,1,2])*step
    jj,bb,BB=geom(pp.reshape(-1,3));jj=jj.reshape(-1,3,4);bb=bb.reshape(-1,3,4,3);BB=BB.reshape(-1,3,4)
    wt=np.array([1,-8,8,-1])/(12*step)
    db=sum(np.einsum('pq,q->p',jj[:,a]*bb[:,a,:,a]*BB[:,a],wt) for a in range(3))/J[:,2]
    L=np.zeros((len(raw),2,6));scale=B[:,2]*b[:,2,2]
    for ai,alpha in enumerate((1/16,1/32)):
        s=ai*3
        L[:,ai,s]=-scale/(alpha*t.g.deta*B[:,s])
        L[:,ai,s+1]=scale/(alpha*t.g.deta*B[:,s+1])
        L[:,ai,s+2]=db/B[:,2]
    return slots,b,L


@dataclass(frozen=True)
class PreparedQ:
    metadata: dict
    owners: np.ndarray
    raw: np.ndarray
    raw_to_owner: np.ndarray
    raw_weight: np.ndarray
    owner_raw: np.ndarray
    owner_weight: np.ndarray
    slot_points: np.ndarray
    donor: np.ndarray
    mask: np.ndarray
    diffusion_D: np.ndarray
    diffusion_N: np.ndarray
    boundary_D_node: np.ndarray
    boundary_D_tangent: np.ndarray
    boundary_N_normal: np.ndarray
    boundary_wall_nodes: np.ndarray
    boundary_wall_queries: np.ndarray
    boundary_wall_normal: np.ndarray
    boundary_value_D_trace: np.ndarray
    boundary_gradient_D_trace: np.ndarray
    boundary_value_N_normal: np.ndarray
    boundary_gradient_N_normal: np.ndarray
    choice: np.ndarray
    indicator: np.ndarray
    noise: np.ndarray
    row_value_D: np.ndarray
    row_gradient_D: np.ndarray
    row_value_N: np.ndarray
    row_gradient_N: np.ndarray
    magnetic_b: np.ndarray
    magnetic_L: np.ndarray
    row_count: np.ndarray

    @property
    def nbytes(self): return sum(getattr(self,k).nbytes for k in _ARRAYS)

    def runtime_view(self):
        """Drop directional diagnostics and wall geometry before device staging."""
        runtime_metadata={k:v for k,v in self.metadata.items() if k!='row_diagnostics'}
        return QRuntime(runtime_metadata,self.raw,self.donor,self.diffusion_D,self.diffusion_N,
                        self.boundary_D_node,self.boundary_D_tangent,self.boundary_N_normal,
                        self.owner_raw,self.owner_weight)

    def validate(self, expected_identity=FROZEN_IDENTITY):
        m=self.metadata
        if m.get('schema') != SCHEMA or m.get('campaign_identity') != expected_identity:
            raise ValueError('Q artifact identity/schema mismatch')
        if m.get('span') not in (1/16,1/32) or m.get('n') not in (32,48,64):
            raise ValueError('invalid Q span/grid')
        if (m.get('trace_steps')!=64 or m.get('trace_method')!='RK4' or
            any(not m.get(k) for k in ('source_identity','geometry_identity','topology_hash','trace_hash'))):
            raise ValueError('incomplete frozen Q identity')
        if len(self.raw)==0 or len(self.owners)==0 or np.any(np.diff(self.raw_to_owner)<0):
            raise ValueError('raw rows must be nonempty and grouped by owner')
        if len(np.unique(self.raw))!=len(self.raw) or len(np.unique(self.owners))!=len(self.owners):
            raise ValueError('duplicate owner/raw rows')
        nr=len(self.raw); no=len(self.owners); nd=self.donor.shape[1]
        shapes={'raw_to_owner':(nr,), 'raw_weight':(nr,), 'owner_raw':(no,self.owner_raw.shape[1]),
                'owner_weight':self.owner_raw.shape,'donor':(nr,nd),'mask':(nr,nd),
                'diffusion_D':(nr,nd),'diffusion_N':(nr,nd),'boundary_D_node':(nr,35),
                'boundary_D_tangent':(nr,3,2),'boundary_N_normal':(nr,35),
                'row_value_D':(nr,3,nd),'row_gradient_D':(nr,3,3,nd),
                'row_value_N':(nr,3,nd),'row_gradient_N':(nr,3,3,nd),
                'boundary_value_D_trace':(nr,3,35),
                'boundary_gradient_D_trace':(nr,3,3,35),
                'boundary_value_N_normal':(nr,3,35),
                'boundary_gradient_N_normal':(nr,3,3,35)}
        for k,shape in shapes.items():
            if getattr(self,k).shape!=shape: raise ValueError(f'{k} shape mismatch')
        if np.any(self.donor[~self.mask]!=0) or np.any(self.row_count!=self.mask.sum(1)):
            raise ValueError('masked donor padding mismatch')
        if np.any(self.donor[self.mask]<0) or np.any(self.donor[self.mask]>=m['n_owner']):
            raise ValueError('donor outside owner state')
        for ids in (self.donor[i,:self.row_count[i]] for i in range(nr)):
            if len(np.unique(ids))!=len(ids): raise ValueError('duplicate donors in a row')
        if not np.allclose(np.sum(self.owner_weight,axis=1),1,rtol=0,atol=1e-12):
            raise ValueError('owner projection incomplete')
        if np.any(self.owner_raw<0) or np.any(self.owner_raw>=nr): raise ValueError('owner raw index')
        used=self.owner_raw[self.owner_weight!=0]
        if not np.array_equal(np.sort(used),np.arange(nr)):
            raise ValueError('raw member lost or repeated in owner projection')
        for i in range(no):
            hit=self.owner_raw[i,self.owner_weight[i]!=0]
            if np.any(self.raw_to_owner[hit]!=i) or not np.allclose(self.owner_weight[i,self.owner_weight[i]!=0],
                                                                     self.raw_weight[hit],rtol=0,atol=0):
                raise ValueError('owner projection weights mismatch')
        if any(not np.isfinite(getattr(self,k)).all() for k in _ARRAYS if getattr(self,k).dtype.kind in 'fc'):
            raise ValueError('nonfinite Q payload')
        nonwall = self.raw // (m['n'] ** 2) < m['n'] - 2
        for name in ('boundary_D_node', 'boundary_D_tangent', 'boundary_N_normal'):
            if np.any(getattr(self, name)[nonwall] != 0):
                raise ValueError('boundary action on a non-wall row')
        return self


@dataclass(frozen=True)
class QRuntime:
    metadata: dict
    raw: np.ndarray
    donor: np.ndarray
    diffusion_D: np.ndarray
    diffusion_N: np.ndarray
    boundary_D_node: np.ndarray
    boundary_D_tangent: np.ndarray
    boundary_N_normal: np.ndarray
    owner_raw: np.ndarray
    owner_weight: np.ndarray

    @property
    def nbytes(self):
        return sum(getattr(self,k).nbytes for k in ('raw','donor','diffusion_D','diffusion_N',
                    'boundary_D_node','boundary_D_tangent','boundary_N_normal','owner_raw','owner_weight'))


def prepare_chunk(t, owners, ends, geom, jacobian, *, span, source_identity, geometry_identity,
                  campaign_identity=FROZEN_IDENTITY, raw=None,
                  frozen_choices=None, choice_provenance=None):
    """Prepare one legacy v2 span view of complete saved owner traces.

    See :func:`prepare_paired_chunks` for the callback and frozen-choice contract.
    Call that interface when both spans are needed to share preparation work.
    """
    if span not in (1/16,1/32): raise ValueError('unsupported frozen span')
    pair=prepare_paired_chunks(t,owners,ends,geom,jacobian,
                              source_identity=source_identity,geometry_identity=geometry_identity,
                              campaign_identity=campaign_identity,raw=raw,
                              frozen_choices=frozen_choices,choice_provenance=choice_provenance)
    return pair[0 if span==1/16 else 1]


def prepare_paired_chunks(t, owners, ends, geom, jacobian, *, source_identity, geometry_identity,
                          campaign_identity=FROZEN_IDENTITY, raw=None,
                          frozen_choices=None, choice_provenance=None):
    """Return ``(outer_h16, inner_h32)`` legacy v2 :class:`PreparedQ` views.

    ``geom(points)->(Jacobian determinant,b contravariant,B magnitude)`` and
    ``jacobian(points)->coordinate Jacobian`` must match the frozen evaluators.
    Saved RK4 endpoints must contain every raw member of the sorted owners.
    Geometry retains the original six-slot batch, repeated center, query order
    and physical magnetic coefficients. Support selection (or accepted integer
    ``frozen_choices`` with provenance), wall elimination, donors and complete
    physical-volume owner projection are computed once for both spans.

    Shared donor/projection/choice arrays must be treated as immutable. Each
    returned view retains independent span action rows and the unchanged v2
    save/load contract. Metadata timers describe setup work only; the common
    magnetic/support times and shared part of packing belong to the shared
    preparation, not twice its cost.
    No callback or support construction occurs inside warmed runtime application.
    """
    if not source_identity or not geometry_identity: raise ValueError('source/geometry identity required')
    owners=np.asarray(owners,int)
    if len(owners)==0 or not np.array_equal(owners,np.unique(owners)): raise ValueError('owners must be sorted unique')
    if np.any(owners<0) or np.any(owners>=len(t.vol)): raise ValueError('owner outside topology')
    expected=raw_members(t,owners)
    if raw is None: raw=expected
    raw=np.asarray(raw,int)
    if not np.array_equal(raw,expected): raise ValueError('trace must contain every raw member in owner order')
    if frozen_choices is not None:
        frozen_choices=np.asarray(frozen_choices)
        if not choice_provenance or frozen_choices.shape!=(len(raw),) or frozen_choices.dtype.kind not in 'iu':
            raise ValueError('frozen choices require integer raw choices and provenance')
    elif choice_provenance is not None:
        raise ValueError('choice provenance without frozen choices')
    ends=np.asarray(ends,float)
    if ends.shape!=(len(raw),4,3): raise ValueError('saved endpoint shape mismatch')
    tick=time.perf_counter();slot,b,L=magnetic_coefficients(t,raw,ends,geom)
    magnetic_seconds=time.perf_counter()-tick
    compact,balanced,local=Hybrid(t),BalancedHybrid(t),LocalHybrid(t)
    if frozen_choices is not None:
        inner=raw//(t.n*t.n)<=compact.last
        if (not np.isin(frozen_choices[inner],(0,1,2)).all() or
            np.any(frozen_choices[~inner]!=-1)):
            raise ValueError('invalid frozen inner/outer choice')
    cols=[]; metas=[]; choices=[]; indicators=[]; noises=[]
    row_tick=time.perf_counter()
    for pos,rr in enumerate(raw):
        ijk=np.array(np.unravel_index(rr,(t.n,)*3)); p=slot[pos]
        if ijk[0]>=t.n-2:
            wall=Wall(jacobian,t,*ijk)
            W,I,GW,GI=wall.basis(p); VN,BN,DN,EN=wall.maps(p)
            ids=wall.ids
            # D: frozen nonconstant lift.  Values at wall-projected queries
            # are separate data; they do not enter diffusion (gradient only).
            bcDnode=-GI.reshape(6,3,4,35).sum(axis=2)
            bcDvalue=-I.reshape(6,4,35).sum(axis=1)
            bcN=EN
            col=dict(ids=ids,V_D=I,G_D=GI,V_N=VN,G_N=DN,
                     Dnode=bcDnode,Dvalue=bcDvalue,Nnormal=bcN,Nvalue=BN,wall_nodes=wall.wall,
                     wall_queries=np.column_stack((np.ones(6),p[:,1:])),normal=wall.a,
                     meta={'family':'wall','condition':wall.condition,'reproduction':wall.solve_residual})
            choice=-1;score=np.zeros((3,2));noise=np.zeros(3)
        else:
            if np.any(p[:,0]>1): raise ValueError('non-wall cap crosses physical wall')
            if ijk[0]<=compact.last:
                if frozen_choices is not None:
                    choice=int(frozen_choices[pos])
                    ids,V,D,m=(compact,balanced,local)[choice].rows(ijk,p)
                    if any(len(pl['donors'])!=28 or pl['rank']!=15 for pl in m['planes']):
                        raise RuntimeError('frozen inner 28/rank invariant failed')
                    # Unused score placeholders are explicitly identified in metadata.
                    score=np.zeros((3,2));noise=np.zeros(3)
                else:
                    maps=[]; metadata=[]
                    for obj in (compact,balanced,local):
                        ids,V,D,m=obj.rows(ijk,p);maps.append((ids,V,D));metadata.append(m)
                        if any(len(pl['donors'])!=28 or pl['rank']!=15 for pl in m['planes']):
                            raise RuntimeError('frozen inner 28/rank invariant failed')
                    choice,score,noise=selective_indicators(t,rr,p,b[pos],maps)
                    ids,V,D=maps[choice];m=metadata[choice]
            else:
                ids,V,D,m=compact.rows(ijk,p)
                choice=-1;score=np.zeros((3,2));noise=np.zeros(3)
            col=dict(ids=ids,V_D=V,G_D=D,V_N=V,G_N=D,Dnode=np.zeros((6,3,35)),
                     Dvalue=np.zeros((6,35)),Nnormal=np.zeros((6,3,35)),Nvalue=np.zeros((6,35)),
                     wall_nodes=np.zeros((35,3)),
                     wall_queries=np.zeros((6,3)),normal=np.zeros((35,3)),meta=m)
        if ijk[0]<t.n-2 and abs(col['G_D'].sum(axis=-1)*np.array([t.g.dr,t.g.dtheta,t.g.deta])).max()>1e-8:
            raise RuntimeError('constant-gradient reproduction failure')
        cols.append(col);metas.append(col['meta']);choices.append(choice);indicators.append(score);noises.append(noise)
    rows_seconds=time.perf_counter()-row_tick
    shared_pack_tick=time.perf_counter()
    maxd=max(len(c['ids']) for c in cols);nr=len(raw)
    donor=np.zeros((nr,maxd),np.int32);mask=np.zeros((nr,maxd),bool)
    for pos,c in enumerate(cols):
        d=len(c['ids']);donor[pos,:d]=c['ids'];mask[pos,:d]=True
        c['parallel_D']=np.einsum('si,sid->sd',b[pos],c['G_D'])
        c['parallel_N']=np.einsum('si,sid->sd',b[pos],c['G_N'])
        c['parallel_D_node']=np.einsum('si,sij->sj',b[pos],c['Dnode'])
        c['parallel_N_normal']=np.einsum('si,sij->sj',b[pos],c['Nnormal'])
    owner_index={int(o):i for i,o in enumerate(owners)}
    raw_to_owner=np.array([owner_index[int(t.ro[r])] for r in raw],np.int32)
    raw_weight=t.rv[raw]/t.vol[t.ro[raw]]
    maxm=max(np.bincount(raw_to_owner))
    owner_raw=np.zeros((len(owners),maxm),np.int32);owner_weight=np.zeros((len(owners),maxm))
    for i in range(len(owners)):
        hit=np.flatnonzero(raw_to_owner==i);owner_raw[i,:len(hit)]=hit;owner_weight[i,:len(hit)]=raw_weight[hit]
    choice=np.array(choices,np.int8);indicator=np.array(indicators);noise=np.array(noises)
    row_count=mask.sum(1).astype(np.int32)
    shared=(owners,raw,raw_to_owner,raw_weight,owner_raw,owner_weight,donor,mask,
            choice,indicator,noise,row_count)
    shared_packing_seconds=time.perf_counter()-shared_pack_tick
    return tuple(_pack_span(t,ends,slot,b,L,cols,metas,shared,span=span,
                 source_identity=source_identity,geometry_identity=geometry_identity,
                 campaign_identity=campaign_identity,frozen_choices=frozen_choices,
                 choice_provenance=choice_provenance,magnetic_seconds=magnetic_seconds,
                 rows_seconds=rows_seconds,shared_packing_seconds=shared_packing_seconds) for span in (1/16,1/32))


def _pack_span(t,ends,slot,b,L,cols,metas,shared,*,span,source_identity,geometry_identity,
               campaign_identity,frozen_choices,choice_provenance,magnetic_seconds,rows_seconds,
               shared_packing_seconds):
    """Slice original six-slot rows without changing contraction arithmetic."""
    (owners,raw,raw_to_owner,raw_weight,owner_raw,owner_weight,donor,mask,
     choice,indicator,noise,row_count)=shared
    maxd=donor.shape[1];maxm=owner_raw.shape[1]
    pack_tick=time.perf_counter()
    ai=0 if span==1/16 else 1; ss=np.array([0,1,2]) if ai==0 else np.array([3,4,5])
    nr=len(raw);ns=len(ss)
    arrays={k:np.zeros((nr,ns,maxd)) for k in ('row_value_D','row_value_N')}
    arrays.update({k:np.zeros((nr,ns,3,maxd)) for k in ('row_gradient_D','row_gradient_N')})
    dD=np.zeros((nr,maxd));dN=np.zeros_like(dD);bd=np.zeros((nr,35));bt=np.zeros((nr,3,2));bn=np.zeros((nr,35))
    wn=np.zeros((nr,35,3));wq=np.zeros((nr,3,3));wa=np.zeros((nr,35,3))
    bvd=np.zeros((nr,ns,35));bgd=np.zeros((nr,ns,3,35))
    bvn=np.zeros((nr,ns,35));bgn=np.zeros((nr,ns,3,35))
    for pos,c in enumerate(cols):
        d=len(c['ids'])
        for name,source in (('row_value_D','V_D'),('row_value_N','V_N'),('row_gradient_D','G_D'),('row_gradient_N','G_N')):
            arrays[name][pos,...,:d]=c[source][ss]
        pgD=c['parallel_D'];pgN=c['parallel_N']
        dD[pos,:d]=L[pos,ai]@pgD;dN[pos,:d]=L[pos,ai]@pgN
        bd[pos]=L[pos,ai]@c['parallel_D_node']
        bn[pos]=L[pos,ai]@c['parallel_N_normal']
        if c['meta']['family'] == 'wall':
            bt[pos]=L[pos,ai,ss,None]*b[pos,ss,1:]
        wn[pos]=c['wall_nodes'];wq[pos]=c['wall_queries'][ss];wa[pos]=c['normal']
        bvd[pos]=c['Dvalue'][ss];bgd[pos]=c['Dnode'][ss]
        bvn[pos]=c['Nvalue'][ss];bgn[pos]=c['Nnormal'][ss]
    packing_seconds=shared_packing_seconds+time.perf_counter()-pack_tick
    metadata=dict(schema=SCHEMA,campaign_identity=campaign_identity,source_identity=str(source_identity),
                  geometry_identity=str(geometry_identity),
                  n=t.n,n_owner=len(t.vol),span=float(span),trace_steps=64,trace_method='RK4',
                  topology_hash=sha256_array(t.ro)+':'+sha256_array(t.rv),
                  trace_hash=sha256_array(raw)+':'+sha256_array(ends),
                  support='selective compact28/gradient guard; structured outer; quartic physical wall',
                  max_donors=maxd,max_raw_members=maxm,
                  families={name:sum(m['family']==name for m in metas) for name in {m['family'] for m in metas}},
                  max_condition=max(float(m['condition']) for m in metas),
                  max_reproduction=max(float(m['reproduction']) for m in metas),
                  magnetic_seconds=magnetic_seconds,rows_seconds=rows_seconds,packing_seconds=packing_seconds,
                  row_diagnostics=[dict(family=m['family'],condition=float(m['condition']),
                        reproduction=float(m['reproduction']),
                        plane_selections=[p.get('selection',{}) for p in m.get('planes',[])],
                        plane_donors=[p.get('donors',[]) for p in m.get('planes',[])],
                        plane_rank=[p.get('rank') for p in m.get('planes',[])],
                        plane_initial_rank=[p.get('initial_rank') for p in m.get('planes',[])],
                        plane_expansion=[p.get('expansion') for p in m.get('planes',[])],
                        plane_exchanges=[p.get('exchanges',[]) for p in m.get('planes',[])],
                        plane_reproduction=[p.get('reproduction') for p in m.get('planes',[])])
                        for m in metas])
    if frozen_choices is not None:
        metadata.update(choice_provenance=str(choice_provenance),
                        choice_hash=sha256_array(frozen_choices),
                        selection_diagnostics='not reevaluated; accepted choices reused')
    return PreparedQ(metadata,owners,raw,raw_to_owner,raw_weight,owner_raw,owner_weight,slot[:,ss],
                     donor,mask,dD,dN,bd,bt,bn,wn,wq,wa,bvd,bgd,bvn,bgn,choice,
                     indicator,noise,arrays['row_value_D'],arrays['row_gradient_D'],
                     arrays['row_value_N'],arrays['row_gradient_N'],b[:,ss],L[:,ai,ss],row_count).validate(campaign_identity)


def save_chunk(prepared, path):
    """Atomically save a self-checksummed, versioned Q chunk."""
    prepared.validate(prepared.metadata['campaign_identity'])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    member_hashes={k:sha256_array(getattr(prepared,k)) for k in _ARRAYS}
    manifest=dict(prepared.metadata,array_hashes=member_hashes)
    with tmp.open('wb') as f:
        np.savez_compressed(f,metadata=np.array(json.dumps(manifest,sort_keys=True,
                                                       default=lambda x:x.item() if isinstance(x,np.generic) else x)),
                            **{k:getattr(prepared,k) for k in _ARRAYS})
    os.replace(tmp,path)
    return path


def load_chunk(path, *, expected_identity=FROZEN_IDENTITY, source_identity=None, geometry_identity=None,
               topology_hash=None, trace_hash=None, span=None):
    with np.load(path,allow_pickle=False) as z:
        m=json.loads(str(z['metadata']))
        for key,expected in (('campaign_identity',expected_identity),('source_identity',source_identity),
                             ('geometry_identity',geometry_identity),
                             ('topology_hash',topology_hash),('trace_hash',trace_hash),('span',span)):
            if expected is not None and m.get(key)!=expected: raise ValueError(f'Q {key} mismatch')
        arrays={k:z[k].copy() for k in _ARRAYS}
    hashes = m.pop('array_hashes')
    if set(hashes) != set(_ARRAYS):
        raise ValueError('Q array checksum manifest incomplete')
    for k,h in hashes.items():
        if sha256_array(arrays[k])!=h: raise ValueError(f'Q {k} checksum mismatch')
    return PreparedQ(m,**arrays).validate(expected_identity)
