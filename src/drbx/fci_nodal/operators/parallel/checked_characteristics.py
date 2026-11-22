"""Robust closed-form characteristic action, explicit validity, no replacement flux/basis."""
import numpy as np
from . import closed_form_characteristics as general

MAX_CONDITION=1e10

PSI = 'phi_plus_tau_pi'

class MaterialStateError(ValueError):pass

def spectral(u,xp,tau,mu):
 system=general.eigensystem(u[:,:5],tau,mu,PSI,xp,robust=True);lam,rs,ls,root_valid=system;cond=general.production_condition(rs,ls,xp);speeds=xp.stack(lam,axis=1);distinct=xp.ones(len(u),dtype=bool)
 for i in range(5):
  for j in range(i):distinct=distinct&(speeds[:,i]!=speeds[:,j])
 finite=xp.all(xp.isfinite(u),axis=1);codes=xp.where(finite,0,1).astype(xp.int32);codes=codes|xp.where(u[:,0]>0,0,2)|xp.where(u[:,1]>0,0,4)|xp.where(u[:,2]>=0,0,8);codes=codes|xp.where(root_valid&distinct&xp.all(xp.isfinite(speeds),axis=1),0,16)|xp.where(xp.isfinite(cond)&(cond<=MAX_CONDITION),0,32)
 return system,codes==0,codes,cond

def check(valid,codes,u,location='state',ids=None):
 valid=np.asarray(valid,bool);codes=np.asarray(codes);bad=np.flatnonzero(~valid)
 if not len(bad):return
 k=int(bad[0]);ident=k if ids is None else int(np.asarray(ids)[k]);reasons=[name for bit,name in [(1,'non-finite state'),(2,'n <= 0'),(4,'Te <= 0'),(8,'Ti < 0'),(16,'non-real/coalescing/non-finite characteristic roots'),(32,'condition check failed (limit1e10)'),(64,'non-finite characteristic result')] if int(codes[k])&bit]
 raise MaterialStateError(f'{len(bad)} invalid states in evaluated {location} batch; first offending {location} {ident}; state={np.asarray(u)[k].tolist()}; reason={"; ".join(reasons)}')

def action(u,x,xp,tau,mu):
 system,valid,codes,cond=spectral(u,xp,tau,mu);value=general.absolute_apply(u[:,:5],x[:,:5],tau,mu,PSI,xp,sysm=system,robust=True);codes=codes|xp.where(xp.all(xp.isfinite(value),axis=1),0,64);return value,codes==0,codes

def absolute_data(u,z,xp,tau,mu):
 system,valid,codes,cond=spectral(u,xp,tau,mu)
 if z is None:value=general.absolute(u[:,:5],tau,mu,PSI,xp,sysm=system,robust=True);E=None
 else:value,E=general.abs_derivative_contraction(u[:,:5],z[:,:5],tau,mu,PSI,xp,sysm=system,robust=True)
 good=xp.all(xp.isfinite(value),axis=(1,2))
 if E is not None:good=good&xp.all(xp.isfinite(E),axis=(1,2))
 codes=codes|xp.where(good,0,64);return value,E,codes==0,codes

def material(u,xp,tau,mu):return general.material(u[:,:5],tau,mu,PSI,xp)
