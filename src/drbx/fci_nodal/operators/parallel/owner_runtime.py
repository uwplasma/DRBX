"""bounded indexed parallel assembly Q actions on resident owner rows and sparse halo exchanges."""
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from . import checked_characteristics as cf
from . import linear_ops as ops, record_runtime as rr
from .status import checked, initial_status, status_add


FLUX_BOX_ADVECTIVE_MASS_FIELDS = (True, True, True, True, True, False)


@lru_cache(None)
def leg_kernel(mesh,tau,mu):
 def rhs(t,st,g,sg):
  d=t-g;ab,valid,codes=cf.action(st,d,jnp,tau,mu)
  v=.5*(-sg[:,None]*jnp.einsum('pij,pj->pi',cf.material(st,jnp,tau,mu),d[:,:5])+ab)
  v=jnp.concatenate([v,(.5*(-sg*st[:,3]+jnp.abs(st[:,3]))*d[:,5])[:,None]],1)
  return v,valid,codes
 return jax.jit(jax.shard_map(rhs,mesh=mesh,in_specs=(P('records'),)*4,
                              out_specs=(P('records'),)*3,check_vma=False))


@lru_cache(None)
def status_kernel(mesh):
 def summarize(valid,codes,states,ids):return rr.reduce_status(status_add(initial_status(),valid,codes,states,ids))
 return jax.jit(jax.shard_map(summarize,mesh=mesh,in_specs=(P('records'),)*4,out_specs=P(),check_vma=False))


@jax.jit
def face_inputs(L,R,meta,U,bu):
 f=.5*(L@U+R@U);f=jnp.where(meta['valid'][:,None],f,jnp.array([1.,1.,1.,0.,0.,0.]))
 j=(meta['ell']**2/8)[:,None]*(R@bu-L@bu)
 return f,j


@jax.jit
def deposit(L,R,meta,v):
 y=(meta['ell']**2/8)[:,None]*(.5*meta['mu'][:,None]*v)
 return R.transpose_apply(y)-L.transpose_apply(y)


class OwnedRuntime:
 def configure(self):
  self.wall_kernel=leg_kernel(self.mesh,*self.params)
  self.summarize=status_kernel(self.mesh)
 def operators(self):
  return {**{k:self.p[k] for k in ['C','CT','G','D','R','RT']},'B':self.B,'BT':self.BT,
          'LN':self.LN,'LD':self.LD,'observation_L':self.L,'observation_R':self.R}
 def arrays(self):
  return self.p,self.B,self.BT,self.LN,self.LD,self.L,self.R,self.meta,self.wall_ids
 def _input(self,x,layout,neutral):
  sh=NamedSharding(self.mesh,P('records'))
  if hasattr(x,'sharding') and x.ndim>0 and x.sharding.is_equivalent_to(sh,x.ndim):
   if x.shape[0]!=layout.parts*layout.cap:raise ValueError('Owner input shape differs from the saved layout')
   return x
  return layout.put(x,self.mesh,neutral)
 def node_input(self,x,state=False):
  return self._input(x,self.nodes,[1,1,1,0,0,0] if state else 0)
 def wall_input(self,x,state=True):
  return self._input(x,self.walls,[1,1,1,0,0,0] if state else 0)
 def wall_flux(self,U,g):
  U=self.node_input(U,True);g=self.wall_input(g)
  flux,valid,codes=self.wall_kernel(ops.trace(self.p,U),g,g,self.p['sg'])
  checked(self.summarize(valid,codes,g,self.wall_ids),'wall')
  return flux
 def sat(self,U,g):
  return -ops.apply_lift(self.p,self.wall_flux(U,g))


@lru_cache(None)
def coupled_records(mesh,params,rhs_chunk,*,robust=True):
    from .absolute_flux import absolute_action
    tau,mu=params

    def apply(f,j,ids):
        size=len(f);pad=(-size)%rhs_chunk
        f=jnp.concatenate([f,jnp.broadcast_to(jnp.array([1.,1.,1.,0.,0.,0.]),(pad,6))])
        j=jnp.pad(j,((0,pad),(0,0)));ids=jnp.pad(ids,(0,pad),constant_values=-1)
        def body(carry,z):
            ff,jj,ii=z
            if robust:
                status=carry
                a=absolute_action(ff,jj,tau,mu)
                _,valid,codes,_=cf.spectral(ff,jnp,tau,mu)
                codes=codes|jnp.where(jnp.all(jnp.isfinite(a),axis=1),0,64)
            else:
                status,recovery=carry
                a,valid,codes,failed=cf.action(ff,jj,jnp,tau,mu,robust=False)
                recovery=recovery|jnp.any(failed & (ii>=0))
            v=jnp.concatenate([a,(jnp.abs(ff[:,3])*jj[:,5])[:,None]],1)
            status=status_add(status,(valid&(codes==0))|(ii<0),codes,ff,ii)
            return (status if robust else (status,recovery)),v
        initial=initial_status() if robust else (initial_status(),jnp.asarray(False))
        carry,v=jax.lax.scan(body,initial,(f.reshape(-1,rhs_chunk,6),
                             j.reshape(-1,rhs_chunk,6),ids.reshape(-1,rhs_chunk)))
        values=v.reshape(-1,6)[:size]
        if robust:
            return values,rr.reduce_status(carry)
        status,recovery=carry
        return values,rr.reduce_status(status),jax.lax.pmax(recovery,'records')
    return jax.jit(jax.shard_map(apply,mesh=mesh,in_specs=(P('records'),)*3,
                              out_specs=(P('records'),P()) if robust else (P('records'),P(),P()),
                              check_vma=False))


@lru_cache(None)
def coupled_action(mesh,params,rhs_chunk,*,robust=True):
    """JIT entry for sharded physical fields, wall hits and native Q loads."""
    from .absolute_flux import absolute_action
    tau,mu=params

    def apply(p,B,BT,LN,LD,L,R,meta,U,psi,g,load_N,load_D,chi,mass_ratio,dirichlet_mass_correction,
              *,hit_active=None):
        t_A=g
        j_wall=g[:,0]*(g[:,3]-g[:,4])
        center=ops.corrected_center_wall_band(p,U,t_A,jw=j_wall)
        f,j=face_inputs(L,R,meta,U,rr.bw_action(p,B,U))
        records=coupled_records(mesh,params,rhs_chunk,robust=robust)(f,j,meta['ids'])
        record,status=records[:2]
        acc=deposit(L,R,meta,record)
        jump=-rr.bw_transpose(p,BT,acc)/p['H'][:,None]
        t=g
        d=t-g
        if robust:
            ab=absolute_action(g,d,tau,mu)
        else:
            ab,_,_,wall_failed=cf.action(g,d,jnp,tau,mu,robust=False)
            wall_failed=jnp.any(wall_failed if hit_active is None else wall_failed & hit_active)
        flux=.5*(-p['sg'][:,None]*jnp.einsum(
            'pij,pj->pi',cf.material(g,jnp,tau,mu),d[:,:5])+ab)
        flux=jnp.concatenate([flux,(.5*(-p['sg']*g[:,3]+jnp.abs(g[:,3]))*d[:,5])[:,None]],1)
        sat=-ops.apply_lift(p,flux)
        potential=rr.phi_action(p,psi)
        pattern=jnp.array([True,True,True,False,False,True])
        diffusion=chi[None,:]*jnp.where(pattern[None,:],LN@U+load_N,
                   (LD@U+load_D)*dirichlet_mass_correction[:,None])
        weight=jnp.where(jnp.asarray(FLUX_BOX_ADVECTIVE_MASS_FIELDS)[None,:],1.,mass_ratio[:,None])
        result=((center+jump+sat+potential)*weight+diffusion*mass_ratio[:,None],status,
                diffusion*mass_ratio[:,None])
        return result if robust else (*result,records[2]|wall_failed)
    return jax.jit(apply)
