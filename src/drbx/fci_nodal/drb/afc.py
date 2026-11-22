"""Accepted parameter-free semi-discrete AFC on wrong-sign K_N pairs.

Parameter-free semidiscrete algebraic flux correction from the 2026-10-23 baseline.
The budgets do not contain dt, mass, affine loads or tuning parameters.
Only n/Te/Ti natural-Neumann diffusion is enabled by the operator switch.
"""
from ..options import CONDUCTING_WALL
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class OwnerLimiterGraph:
    K:object
    row:object
    i:object
    j:object
    v:object
    diag:object
    node_valid:object
    edge_valid:object
    j_nodes:object
    j_edges:object


jax.tree_util.register_dataclass(OwnerLimiterGraph,data_fields=list(OwnerLimiterGraph.__dataclass_fields__),meta_fields=[])


def build_owner_limiter_graph(mesh,nodes,K_rows,edges,diagonal):
    """Edges are (canonical edge ID, i ID, j ID, value), each stored at i."""
    from ..operators.parallel.owner_products import LocalLayout
    from ..operators.parallel.owner_routes import locate, owner_array, pack_owner_csr, pack_requests
    K_rows={p:A.copy() for p,A in K_rows.items()}
    for A in K_rows.values():
        A.eliminate_zeros()
        A.sort_indices()
    K=pack_owner_csr(mesh,nodes,nodes,K_rows)
    cap=max(1,max(len(edge[0]) for edge in edges.values()))
    edge_ids={p:np.pad(edge[0],(0,cap-len(edge[0])),constant_values=-1)
              for p,edge in edges.items()}
    edge_layout=LocalLayout(sum(len(edge[0]) for edge in edges.values()),cap,mesh.size,edge_ids)
    left={};weights={};valid={};receivers={p:[] for p in edges};other={}
    for p,(ordinal,i,j,v) in edges.items():
        src,local=locate(nodes,i);dst,target=locate(nodes,j)
        if np.any(src!=p) or np.any(i>=j) or np.any(ordinal[1:]<=ordinal[:-1]):
            raise ValueError('AFC edges must be canonical, ascending and stored at i')
        left[p]=np.pad(local,(0,cap-len(local)),constant_values=nodes.cap)
        weights[p]=np.pad(v,(0,cap-len(v)))
        valid[p]=np.arange(cap)<len(v)
        other[p]=j
        for peer in np.unique(dst):
            take=dst==peer
            receivers[int(peer)].append((ordinal[take],target[take]))
    routing={};right={}
    for p,items in receivers.items():
        ids=np.concatenate([a for a,_ in items]) if items else np.empty(0,np.int64)
        loc=np.concatenate([b for _,b in items]) if items else np.empty(0,np.int32)
        perm=np.argsort(ids,kind='stable');routing[p]=ids[perm];right[p]=loc[perm]
    end_cap=max(1,max(map(len,routing.values())))
    for p,j in right.items():
        right[p]=np.pad(j,(0,end_cap-len(j)),constant_values=nodes.cap)
    tasks=pack_requests(mesh,nodes,other,cap)
    reverse=pack_requests(mesh,edge_layout,routing,end_cap)
    row={}
    for p,A in K_rows.items():
        r=np.repeat(np.arange(A.shape[0],dtype=np.int32),np.diff(A.indptr))
        row[p]=np.pad(r,(0,K.arrays[0].shape[1]-len(r)),constant_values=nodes.cap)
    return OwnerLimiterGraph(K,owner_array(mesh,row,packed=True),owner_array(mesh,left,packed=True),
                      owner_array(mesh,right,packed=True),owner_array(mesh,weights,packed=True),
                      nodes.put(diagonal,mesh),owner_array(mesh,{p:x>=0 for p,x in nodes.ids_by_owner.items()}),
                      owner_array(mesh,valid,packed=True),tasks,reverse)


def owner_action(graph,u,chi,mass):
    """Same limiter budgets, with i and j scatter phases in canonical edge order."""
    from ..operators.parallel.owner_routes import take_requests
    from ..operators.parallel.owner_sparse import _expand
    from jax.sharding import PartitionSpec as P
    mesh=graph.K.mesh
    scalar=u.ndim==1
    if scalar:
        u=u[:,None]
    chi=jnp.asarray(chi).reshape(1,-1)
    high=-(graph.K@u)*chi

    def flux(row,col,send,perm,i,v,u,uj,chi):
        idx=_expand(u,send[0],graph.K.delta,mesh)[perm[0]]
        mn=u.at[row[0]].min(idx[col[0]],mode='drop')
        mx=u.at[row[0]].max(idx[col[0]],mode='drop')
        f=v[0,:,None]*chi*(u[i[0]]-uj)
        return mn,mx,f
    def mapped(fn,inputs,output):
        return jax.shard_map(fn,mesh=mesh,in_specs=inputs,out_specs=output,check_vma=False)
    spec=P('records')
    mn,mx,f=mapped(flux,(spec,)*8+(P(),),(spec,)*3)(
        graph.row,graph.K.arrays[1],graph.K.arrays[6],graph.K.arrays[8],
        graph.i,graph.v,u,take_requests(mesh,graph.j_nodes,u),chi)
    fj=take_requests(mesh,graph.j_edges,f)

    def budgets(i,j,u,high,diag,mn,mx,f,fj,chi):
        i,j=i[0],j[0]
        low=high.at[i].add(-f,mode='drop').at[j].add(fj,mode='drop')
        plus=jnp.zeros_like(u).at[i].add(jnp.maximum(f,0.),mode='drop').at[j].add(jnp.maximum(-fj,0.),mode='drop')
        minus=jnp.zeros_like(u).at[i].add(jnp.minimum(f,0.),mode='drop').at[j].add(jnp.minimum(-fj,0.),mode='drop')
        qp=chi*diag[:,None]*(mx-u)-low;qm=chi*diag[:,None]*(mn-u)-low
        rp=jnp.where(plus>0,jnp.clip(qp/jnp.where(plus>0,plus,1.),0.,1.),1.)
        rm=jnp.where(minus<0,jnp.clip(qm/jnp.where(minus<0,minus,1.),0.,1.),1.)
        return rp,rm
    rp,rm=mapped(budgets,(spec,)*9+(P(),),(spec,spec))(
        graph.i,graph.j,u,high,graph.diag,mn,mx,f,fj,chi)
    other=take_requests(mesh,graph.j_nodes,jnp.stack((rp,rm),axis=1))
    def limit(i,u,rp,rm,other,f):
        del u
        return jnp.where(f>0,jnp.minimum(rp[i[0]],other[:,1]),jnp.where(f<0,jnp.minimum(rm[i[0]],other[:,0]),1.))
    alpha=mapped(limit,(spec,)*6,spec)(graph.i,u,rp,rm,other,f)
    rejected=(1-alpha)*f

    def finish(i,j,u,rejected,received,alpha,valid):
        correction=jnp.zeros_like(u).at[i[0]].add(-rejected,mode='drop').at[j[0]].add(received,mode='drop')
        count=jnp.concatenate((jnp.sum((alpha<1.)&valid[0,:,None],axis=0),jnp.sum(valid[0])[None]))
        counts=jax.lax.all_gather(count,'records')
        sums=jax.lax.fori_loop(0,mesh.size,lambda p,s:s+counts[p],jnp.zeros_like(count))
        stats=dict(fraction_limited=sums[:-1]/sums[-1],alpha_min=jax.lax.pmin(jnp.min(alpha,axis=0),'records'),
                   wrong_sign_pairs=sums[-1])
        return correction,stats
    correction,stats=mapped(finish,(spec,)*7,(spec,P()))(
        graph.i,graph.j,u,rejected,take_requests(mesh,graph.j_edges,rejected),alpha,graph.edge_valid)
    result=(high+correction)/mass[:,None]
    return (result[:,0] if scalar else result),correction,stats


def owner_positivity_bound(graph,mass,chi):
    """Device-valued form of stepper.positivity_bound."""
    from jax.sharding import PartitionSpec as P
    def reduce(diag,mass,valid):
        return jax.lax.pmax(jnp.max(jnp.where(valid,diag/mass,-jnp.inf)),'records')
    rate=jax.shard_map(reduce,mesh=graph.K.mesh,in_specs=(P('records'),)*3,out_specs=P(),
                      check_vma=False)(graph.diag,mass,graph.node_valid)
    bounds=1/(rate*jnp.asarray(chi)[:3])
    return dict(low_diagonal_over_Hp_max=rate,bounds_n_Te_Ti=bounds,minimum=jnp.min(bounds))


def owner_correction(graph,U,chi,mass,wall_ops,*,wall_law):
    """Natural-Neumann AFC additions with the existing strong-row projection."""
    from .walls import project_mass
    _,integrated,stats=owner_action(graph,U[:,:3],chi[:3],mass)
    delta=jnp.zeros_like(U).at[:,:3].set(integrated/mass[:,None])
    hard=[(5,jnp.ones_like(wall_ops.Js))]
    if wall_law==CONDUCTING_WALL:
        hard.append((1,(wall_ops.curvature_flux!=0).astype(U.dtype)))
    return project_mass(wall_ops,delta,hard),stats
