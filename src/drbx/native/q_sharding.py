"""Host Q localization by complete owners, with a proven periodic donor halo.

Local plans consume ``[lower halo | owned planes | upper halo | zero trash]``
owner values. They return only the bank's selected owned outputs, in original
bank order. Global grid raw IDs and exact query identities stay in host
receipts; they are never reinterpreted as local owner IDs. This foundation
provides unpadded local plans; it is not production full-domain solver wiring.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from numbers import Integral
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P
from .owner_plane_layout import PlaneLayout, plane_major_permutation, exchange_plane_halo
from .q_parallel import QBoundaryData
from drbx.stencils.q_plan import QPlan
from drbx.stencils.q_parallel import sha256_array

DEFAULT_HALO = 2


@dataclass(frozen=True)
class QShardLayout:
    n: int
    n_shards: int
    shard: int
    halo: int
    layout: PlaneLayout
    n_global_raw: int
    extended_owner_ids: np.ndarray   # global owners, followed by -1 trash
    owner_map: np.ndarray            # global owner -> canonical local input row
    owners: np.ndarray               # global output owner identities
    owner_selector: np.ndarray      # corresponding global bank owner rows
    raw_ids: np.ndarray              # GLOBAL raw grid cell IDs
    raw_selector: np.ndarray         # corresponding global bank source rows
    owner_raw: np.ndarray            # local source rows (zero padding safe)
    owner_weight: np.ndarray
    wall_selector: np.ndarray        # global compact wall row positions
    wall_index: np.ndarray           # local source rows of walls
    query_ids: np.ndarray            # global exact table entries
    query_table: np.ndarray          # local logical coordinates
    wall_node_query: np.ndarray      # local exact table maps
    wall_slot_query: np.ndarray
    donor: np.ndarray                # local extended owner input rows
    nonzero: np.ndarray
    reach: tuple[int, int]

    @property
    def n_owner(self):return len(self.extended_owner_ids)

    @property
    def trash(self):return self.n_owner-1

    @property
    def planes_per_shard(self):return self.n//self.n_shards

    @property
    def owned_input_slice(self):
        m=self.layout.m
        return slice(self.halo*m,(self.halo+self.planes_per_shard)*m)

    def validate(self):
        nr=len(self.raw_ids)
        if self.donor.shape != self.nonzero.shape or self.donor.shape[0]!=nr:
            raise ValueError('local Q source shape mismatch')
        if np.any(self.donor<0) or np.any(self.donor>=self.n_owner):
            raise ValueError('local Q donor bounds')
        if np.any(self.donor[~self.nonzero]!=self.trash):
            raise ValueError('local Q padding must gather trash')
        if np.any(self.donor[self.nonzero]==self.trash):
            raise ValueError('local Q weighted donor lost')
        if self.owner_raw.shape!=self.owner_weight.shape or len(self.owner_raw)!=len(self.owners):
            raise ValueError('local Q projection shape')
        active=self.owner_weight!=0
        if np.any(self.owner_raw<0) or (nr and np.any(self.owner_raw>=nr)):
            raise ValueError('local Q projection index')
        if not np.array_equal(np.sort(self.owner_raw[active]),np.arange(nr)):
            raise ValueError('local Q complete-owner coverage')
        if not np.allclose(self.owner_weight.sum(1),1,atol=1e-12,rtol=0):
            raise ValueError('local Q projection weights')
        if len(self.wall_index)!=len(self.wall_selector) or np.any(self.wall_index>=nr):
            raise ValueError('local Q wall selector')
        for ids in (self.wall_node_query,self.wall_slot_query):
            if np.any(ids<0) or np.any(ids>=len(self.query_table)):
                raise ValueError('local Q query index')
        return self


@dataclass(frozen=True)
class LocalQBank:
    """Host localized payload, deliberately distinct from global QBank schema."""
    shard: QShardLayout
    arrays: dict
    diagnostics: dict
    global_identity: str

    def __getattr__(self,name):
        arrays=object.__getattribute__(self,'arrays')
        if name in arrays:return arrays[name]
        raise AttributeError(name)


@dataclass(frozen=True)
class LocalQPlan:
    plan: QPlan
    shard: QShardLayout


def _donor_nonzero(bank):
    """Union of every scalar and BOTH frozen diffusion actions, D and N."""
    nz=np.any(bank.row_value_D!=0,axis=1)|np.any(bank.diffusion_D!=0,axis=0)
    wall=bank.wall_index
    nz[wall]|=np.any(bank.row_value_N_wall!=0,axis=1)|np.any(bank.diffusion_N_wall!=0,axis=0)
    return nz


def q_donor_reach(bank,raw_to_owner):
    """Signed periodic offsets for every nonzero scalar/diffusion donor.

    Returns ``(nonzero_mask, offset_table, (minimum, maximum))``. No inferred
    support width or P face halo is substituted for the actual Q rows.
    """
    bank.validate()
    n=int(bank.metadata['n'])
    canonical=np.asarray(raw_to_owner).reshape(-1).astype(np.int64)
    if sha256_array(canonical)!=bank.metadata['topology_hash'].split(':')[0]:
        raise ValueError('Q bank frozen topology identity mismatch')
    perm,_,m=plane_major_permutation(canonical,n)
    if len(perm)!=int(bank.metadata['n_owner']):
        raise ValueError('Q bank/topology owner count mismatch')
    raw_to_owner=np.asarray(raw_to_owner).reshape(-1)
    selected=np.zeros(len(perm),dtype=bool);selected[bank.owners]=True
    valid=raw_to_owner>=0
    expected=np.flatnonzero(valid & selected[np.maximum(raw_to_owner,0)])
    if (not np.array_equal(np.sort(bank.raw),expected) or
        not np.array_equal(raw_to_owner[bank.raw],bank.owners[bank.raw_to_owner])):
        raise ValueError('Q bank must contain every raw member of each output owner')
    nz=_donor_nonzero(bank)
    offsets=((perm[bank.donor]//m-bank.raw[:,None]%n+n//2)%n)-n//2
    offsets=np.where(nz,offsets,0)
    reach=(int(offsets[nz].min()),int(offsets[nz].max())) if np.any(nz) else (0,0)
    return nz,offsets,reach


def q_shard_layouts(bank,raw_to_owner,n_shards,*,halo=DEFAULT_HALO):
    """Validate complete source closure and return an unpadded receipt per shard.

    Empty shards are allowed for bounded banks. Their source/output arrays have
    zero length; full RHS callers should skip them. Input topology must satisfy
    the shared single-ring/single-plane and equal-owner-per-plane contracts.
    """
    if isinstance(n_shards,bool) or not isinstance(n_shards,Integral):
        raise ValueError('Q shard count must be an integer')
    if isinstance(halo,bool) or not isinstance(halo,Integral):
        raise ValueError('Q halo must be an integer')
    n=int(bank.metadata['n']);n_shards=int(n_shards);halo=int(halo)
    if n_shards<1 or n%n_shards:
        raise ValueError('Q shard count must divide plane count')
    if halo<0 or (n_shards>1 and n//n_shards<halo):
        raise ValueError('Q owned block must be at least the nonnegative halo')
    nz,offsets,reach=q_donor_reach(bank,raw_to_owner)
    if max(abs(reach[0]),abs(reach[1]))>halo:
        raise ValueError(f'Q nonzero donor reach {reach} exceeds declared halo {halo}')
    perm,inverse,m=plane_major_permutation(raw_to_owner,n)
    layout=PlaneLayout(perm,inverse,m);p=n//n_shards
    result=[]
    for s in range(n_shards):
        lo=s*p;hi=(s+1)*p
        planes=(lo-halo+np.arange(p+2*halo))%n
        ext=inverse[(planes[:,None]*m+np.arange(m)).ravel()]
        trash=len(ext)
        owner_map=np.full(len(perm),trash,dtype=np.int32)
        owner_map[ext]=np.arange(len(ext))
        owned_global=inverse[lo*m:hi*m]
        owner_map[owned_global]=np.arange(halo*m,(halo+p)*m)
        owner_selector=np.flatnonzero((perm[bank.owners]//m>=lo)&(perm[bank.owners]//m<hi))
        source_keep=np.isin(bank.raw_to_owner,owner_selector)
        raw_selector=np.flatnonzero(source_keep)
        raw_map=np.full(len(bank.raw),-1,np.int32);raw_map[raw_selector]=np.arange(len(raw_selector))
        weights=bank.owner_weight[owner_selector]
        projected=raw_map[bank.owner_raw[owner_selector]]
        projected=np.where(weights!=0,projected,0).astype(np.int32)
        wall_selector=np.flatnonzero(source_keep[bank.wall_index])
        wall_index=raw_map[bank.wall_index[wall_selector]]
        nodes=bank.wall_node_query[wall_selector];slots=bank.wall_slot_query[wall_selector]
        query_ids=np.unique(np.concatenate((nodes.ravel(),slots.ravel())))
        query_map=np.full(len(bank.query_table),-1,np.int32);query_map[query_ids]=np.arange(len(query_ids))
        local_nz=nz[raw_selector]
        donor=np.where(local_nz,owner_map[bank.donor[raw_selector]],trash).astype(np.int32)
        if np.any(local_nz&(donor==trash)):
            raise ValueError('Q nonzero donor outside owned+halo window')
        selected_offset=offsets[raw_selector][local_nz]
        local_reach=(int(selected_offset.min()),int(selected_offset.max())) if len(selected_offset) else (0,0)
        result.append(QShardLayout(n,n_shards,s,halo,layout,len(bank.raw),np.r_[ext,-1],owner_map,
            bank.owners[owner_selector],owner_selector,bank.raw[raw_selector],raw_selector,
            projected,weights,wall_selector,wall_index,query_ids,bank.query_table[query_ids],
            query_map[nodes],query_map[slots],donor,local_nz,local_reach).validate())
    return tuple(result)


_RAW = ('raw_weight','mask','row_count','row_value_D','magnetic_b')
_WALL = ('row_value_N_wall','boundary_value_D_trace','boundary_value_N_normal','boundary_wall_normal')
_SPAN_RAW = ('diffusion_D','magnetic_L')
_SPAN_WALL = ('diffusion_N_wall','boundary_D_node','boundary_D_tangent','boundary_N_normal')


def localize_q_bank(bank,raw_to_owner,n_shards,*,halo=DEFAULT_HALO):
    """Subset geometry/queries and remap bank donors without claiming v1 validity."""
    identity=bank.identity
    result=[]
    for sh in q_shard_layouts(bank,raw_to_owner,n_shards,halo=halo):
        r,w=sh.raw_selector,sh.wall_selector
        arrays={name:bank.arrays[name][r] for name in _RAW}
        arrays.update({name:bank.arrays[name][w] for name in _WALL})
        arrays.update({name:bank.arrays[name][:,r] for name in _SPAN_RAW})
        arrays.update({name:bank.arrays[name][:,w] for name in _SPAN_WALL})
        arrays.update(owners=sh.owners,raw=sh.raw_ids,raw_to_owner=np.repeat(
            np.arange(len(sh.owners),dtype=np.int32),np.sum(sh.owner_weight!=0,axis=1)),
            owner_raw=sh.owner_raw,owner_weight=sh.owner_weight,donor=sh.donor,
            wall_index=sh.wall_index,query_table=sh.query_table,
            wall_node_query=sh.wall_node_query,wall_slot_query=sh.wall_slot_query)
        diagnostics={k:v[r] for k,v in bank.diagnostics.items()}
        result.append(LocalQBank(sh,arrays,diagnostics,identity))
    return tuple(result)


def localize_q_plan(plan,bank,raw_to_owner,n_shards,*,halo=DEFAULT_HALO):
    """Select QPlan rows/projections; original global coefficients are unchanged."""
    if plan.n_owner!=bank.metadata['n_owner']:
        raise ValueError('Q plan/bank owner identity mismatch')
    for name in ('donor','owner_raw','owner_weight','wall_index','row_value_D','row_value_N_wall'):
        if not np.array_equal(np.asarray(getattr(plan,name)),getattr(bank,name)):
            raise ValueError(f'Q plan/bank identity mismatch: {name}')
    if plan.diffusion_span not in (1/16,1/32):
        raise ValueError('Q plan explicit accepted diffusion span required')
    ai=0 if plan.diffusion_span==1/16 else 1
    for name in ('diffusion_D','diffusion_N_wall','boundary_D_node',
                 'boundary_D_tangent','boundary_N_normal'):
        if not np.array_equal(np.asarray(getattr(plan,name)),getattr(bank,name)[ai]):
            raise ValueError(f'Q plan/bank identity mismatch: {name}')
    for name in ('boundary_value_D_trace','boundary_value_N_normal'):
        if not np.array_equal(np.asarray(getattr(plan,name)),getattr(bank,name)):
            raise ValueError(f'Q plan/bank identity mismatch: {name}')
    raw_names={'row_value_D','diffusion_D','magnetic_L','b_eta','bmag'}
    wall_names={'row_value_N_wall','boundary_value_D_trace','boundary_value_N_normal',
                'diffusion_N_wall','boundary_D_node','boundary_D_tangent','boundary_N_normal'}
    result=[]
    for sh in q_shard_layouts(bank,raw_to_owner,n_shards,halo=halo):
        arrays={}
        for f in fields(QPlan)[2:]:
            name=f.name;value=np.asarray(getattr(plan,name))
            if name in raw_names:arrays[name]=value[sh.raw_selector]
            elif name in wall_names:arrays[name]=value[sh.wall_selector]
            elif name=='eta_step':arrays[name]=value if value.ndim==0 else value[sh.raw_selector]
            elif name in ('donor','owner_raw','owner_weight','wall_index'):arrays[name]=getattr(sh,name)
            else:raise ValueError(f'unknown QPlan localization field {name}')
        result.append(LocalQPlan(QPlan(sh.n_owner,plan.diffusion_span,**arrays),sh))
    return tuple(result)


def local_owner_values(values,shard: QShardLayout):
    """Global ``(...,owners)`` values -> local window with finite zero trash."""
    x=jnp.asarray(values)
    if x.shape[-1]!=len(shard.layout.perm):raise ValueError('global owner values shape mismatch')
    ext=jnp.take(x,shard.extended_owner_ids[:-1],axis=-1)
    return jnp.concatenate((ext,jnp.zeros(x.shape[:-1]+(1,),dtype=x.dtype)),axis=-1)


def local_boundary_data(boundary: QBoundaryData,shard: QShardLayout):
    """Subset legacy live BC arrays by global bank raw row, retaining field axes."""
    for i,a in enumerate(boundary):
        axis=-3 if i==2 else -2
        if jnp.ndim(a)<-axis or jnp.shape(a)[axis]!=shard.n_global_raw:
            raise ValueError('global Q boundary raw shape mismatch')
    return QBoundaryData(*(jnp.take(jnp.asarray(a),shard.raw_selector,axis=-3 if i==2 else -2)
                           for i,a in enumerate(boundary)))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ShardedQPlan:
    """Array-stacked local plans plus host identity/trim receipts.

    Dummy identity rows gather a separate finite guard input; numerical donor
    padding retains zero trash. Neither row is a physical owner/output.
    """
    plan: QPlan
    shards: tuple
    raw_selectors: np.ndarray
    owner_gather: np.ndarray
    raw_gather: np.ndarray
    n_global_raw: int
    n: int
    m: int
    n_shards: int
    halo: int
    inverse: object

    def tree_flatten(self):
        children=(self.plan,self.raw_selectors,self.owner_gather,self.raw_gather,self.inverse)
        return children,(self.n_global_raw,self.n,self.m,self.n_shards,self.halo)

    @classmethod
    def tree_unflatten(cls,static,children):
        plan,selectors,owners,raw,inverse=children
        return cls(plan,(),selectors,owners,raw,*static,inverse)


def _pad_q_rows(a,count,fill=0):
    a=np.asarray(a)
    out=np.full((count,)+a.shape[1:],fill,dtype=a.dtype)
    out[:len(a)]=a
    return out


def shard_q_plan(plan,bank,raw_to_owner,n_shards,*,halo=DEFAULT_HALO):
    """Stack complete-owner local plans, with explicitly excluded dummy rows."""
    locals_=localize_q_plan(plan,bank,raw_to_owner,n_shards,halo=halo)
    nr=max(len(q.shard.raw_ids) for q in locals_)+1
    no=max(len(q.shard.owners) for q in locals_)
    nw=max(len(q.shard.wall_index) for q in locals_)
    dummy=nr-1
    plans=[];selectors=[]
    owner_gather=np.empty(len(bank.owners),np.int32);raw_gather=np.empty(len(bank.raw),np.int32)
    for i,local in enumerate(locals_):
        q,sh=local.plan,local.shard;r=len(sh.raw_ids);w=len(sh.wall_index)
        # One extra input after zero trash is a PRIVATE finite guard row.
        guard=sh.n_owner
        donor=_pad_q_rows(q.donor,nr,sh.trash)
        donor[r:,0]=guard
        value=_pad_q_rows(q.row_value_D,nr)
        value[r:,:,0]=1.
        nvalue=_pad_q_rows(q.row_value_N_wall,nw)
        nvalue[w:,:,0]=1.
        wall=_pad_q_rows(q.wall_index,nw,dummy)
        oraw=_pad_q_rows(q.owner_raw,no,dummy)
        oweight=_pad_q_rows(q.owner_weight,no)
        # Zero-weight member padding gathers benign dummy source rows.
        oraw=np.where(oweight!=0,oraw,dummy).astype(np.int32)
        arrays=dict(donor=donor,row_value_D=value,row_value_N_wall=nvalue,
                    wall_index=wall,owner_raw=oraw,owner_weight=oweight)
        for name in ('diffusion_D','magnetic_L','b_eta'):
            arrays[name]=_pad_q_rows(getattr(q,name),nr)
        arrays['bmag']=_pad_q_rows(q.bmag,nr,1.)
        step=np.asarray(q.eta_step)
        arrays['eta_step']=np.full(nr,float(step.flat[0]) if step.size else float(np.asarray(plan.eta_step).flat[0]),dtype=step.dtype)
        if step.ndim:arrays['eta_step'][:r]=step
        for name in ('diffusion_N_wall','boundary_value_D_trace','boundary_value_N_normal',
                     'boundary_D_node','boundary_D_tangent','boundary_N_normal'):
            arrays[name]=_pad_q_rows(getattr(q,name),nw)
        plans.append(QPlan(sh.n_owner+1,q.diffusion_span,**arrays))
        selectors.append(_pad_q_rows(sh.raw_selector,nr,len(bank.raw)))
        owner_gather[sh.owner_selector]=i*no+np.arange(len(sh.owners))
        raw_gather[sh.raw_selector]=i*nr+np.arange(r)
    stacked=jax.tree.map(lambda *a:np.stack(a),*plans)
    sh=locals_[0].shard
    return ShardedQPlan(stacked,tuple(q.shard for q in locals_),np.stack(selectors),
                        owner_gather,raw_gather,len(bank.raw),sh.n,sh.layout.m,sh.n_shards,sh.halo,sh.layout.inverse)


def _stack_q_boundary(boundary,sharded):
    arrays=[]
    for i,a in enumerate(boundary):
        a=jnp.asarray(a);axis=a.ndim-(3 if i==2 else 2)
        if axis<0 or a.shape[axis]!=sharded.n_global_raw:
            raise ValueError('global Q boundary raw shape mismatch')
        shape=list(a.shape);shape[axis]=1
        with_dummy=jnp.concatenate((a,jnp.zeros(shape,dtype=a.dtype)),axis=axis)
        gathered=jnp.take(with_dummy,sharded.raw_selectors,axis=axis)
        arrays.append(jnp.moveaxis(gathered,axis,0))
    return QBoundaryData(*arrays)


def sharded_q_rhs(sharded,state,inner_boundary,outer_boundary,phi,phi_boundary,
                  coefficients,*,kinds,phi_kind,tau,mu,mesh=None,characteristic_method="eig",
                  psi="phi_plus_tau_ti"):
    """Full prescribed SixFieldAction under shard_map; returns global bank order.

    Owner state/phi retain global old owner numbering at this interface. Halos
    are exchanged once for six fields and phi together. All dummy diagnostics
    are trimmed. Finite state validity is reduced across all owned input planes,
    including global owners outside a bounded source/donor closure.
    """
    from .q_plan import apply_q_plan
    from .q_parallel_rhs import SixFieldAction
    x=jnp.asarray(state);pvalue=jnp.asarray(phi)
    global_n_owner=sharded.n*sharded.m
    if x.ndim<2 or x.shape[-2:]!=(6,global_n_owner) or x.dtype.kind!='f':
        raise ValueError('real (...,6,global_n_owner) state required')
    if pvalue.shape!=x.shape[:-2]+(x.shape[-1],) or pvalue.dtype.kind!='f':
        raise ValueError('real phi with matching batch/global owner shape required')
    if mesh is None:
        devices=np.asarray(jax.devices()[:sharded.n_shards],object)
        if len(devices)!=sharded.n_shards:raise RuntimeError('insufficient devices for Q shards')
        mesh=Mesh(devices,('z',))
    if mesh.shape.get('z')!=sharded.n_shards:raise ValueError('Q mesh shard count mismatch')
    xp=jnp.take(x,sharded.inverse,axis=-1);pp=jnp.take(pvalue,sharded.inverse,axis=-1)
    bi,bo,pb=(_stack_q_boundary(b,sharded) for b in (inner_boundary,outer_boundary,phi_boundary))
    p=sharded.n//sharded.n_shards;m=sharded.m;h=sharded.halo
    def body(plan_s,xs,bis,bos,ps,pbs,coef):
        plan_s=jax.tree.map(lambda a:a[0],plan_s)
        bis,bos,pbs=(jax.tree.map(lambda a:a[0],b) for b in (bis,bos,pbs))
        joint=jnp.concatenate((xs,ps[...,None,:]),axis=-2)
        owned=jnp.moveaxis(joint,-1,0).reshape((p,m)+joint.shape[:-1])
        extended=exchange_plane_halo(owned,h,'z',sharded.n_shards)
        ext=jnp.moveaxis(extended.reshape((-1,)+joint.shape[:-1]),0,-1)
        zero=jnp.zeros(ext.shape[:-1]+(1,),ext.dtype)
        guard=jnp.broadcast_to(jnp.array([1.,1.,1.,0.,0.,0.,0.],ext.dtype),ext.shape[:-2]+(7,))[...,None]
        ext=jnp.concatenate((ext,zero,guard),axis=-1)
        got=apply_q_plan(plan_s,ext[...,:6,:],bis,bos,ext[...,6,:],pbs,coef,
                         kinds=kinds,phi_kind=phi_kind,tau=tau,mu=mu,
                         characteristic_method=characteristic_method,**({} if psi == 'phi_plus_tau_ti' else {'psi': psi}))
        finite_local=(jnp.all(jnp.isfinite(xs),axis=(-2,-1)) &
                      jnp.all(jnp.isfinite(got.combined),axis=(-2,-1)) &
                      jnp.all(jnp.isfinite(got.diffusion),axis=(-2,-1)))
        finite=lax.pmin(finite_local.astype(jnp.int32),'z').astype(bool)
        got=got._replace(inputs_valid=got.inputs_valid&finite[...,None])
        return jax.tree.map(lambda a:a[None],got)
    specs=(P('z'),P(*([None]*(xp.ndim-1)), 'z'),P('z'),P('z'),
           P(*([None]*(pp.ndim-1)), 'z'),P('z'),P())
    result=jax.shard_map(body,mesh=mesh,in_specs=specs,out_specs=P('z'),check_vma=False)(
        sharded.plan,xp,bi,bo,pp,pb,jnp.asarray(coefficients))
    def owner(a):
        a=jnp.moveaxis(a,0,-3)
        a=a.reshape(a.shape[:-3]+(-1,a.shape[-1]))
        return jnp.take(a,sharded.owner_gather,axis=-2)
    def raw(a):
        a=jnp.moveaxis(a,0,-2)
        a=a.reshape(a.shape[:-2]+(-1,))
        return jnp.take(a,sharded.raw_gather,axis=-1)
    return SixFieldAction(*(owner(getattr(result,k)) for k in ('centered','correction','diffusion','combined')),
        jax.tree.map(raw,result.raw_current),raw(result.raw_electron_material),raw(result.inputs_valid),
        raw(result.eigensystem_admissible))
