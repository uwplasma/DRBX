"""Bounded-audited value-row adapter; frozen builders, explicit arbitrary queries.

No PreparedQ span metadata is changed. Donors are chosen once for all ten points.
"""
import numpy as np
import fields as m
from drbx.stencils.q_parallel_wall import Wall
def reconstruct(t,objects,wall_cache,rr,p,state,phi,choice,jac):
 ijk=np.array(np.unravel_index(rr,(t.n,)*3));wall=ijk[0]>=t.n-2
 if wall:
  if rr not in wall_cache:wall_cache[rr]=Wall(jac,t,*ijk)
  w=wall_cache[rr];_,VD,_,_=w.basis(p);VN,BN,_,_=w.maps(p);BD=-VD.reshape(len(p),4,35).sum(1);ids=w.ids
  nodes,ng=m.fields(w.wall);nv=np.einsum('sa,scfa->scf',w.a,ng);wp=p.copy();wp[:,0]=1;query=m.fields(wp)[0]
  pn,pg=m.phi_fields(w.wall);pnormal=np.einsum('sa,sca->sc',w.a,pg);pq=m.phi_fields(wp)[0]
  family='wall';condition=w.condition
 else:
  obj=objects[choice] if ijk[0]<=objects[0].last else objects[0]
  ids,VD,_,meta=obj.rows(ijk,p);VN=VD;BD=BN=np.zeros((len(p),35));family=meta['family'];condition=meta['condition']
  if ijk[0]<=objects[0].last:assert all(len(x['donors'])==28 and x['rank']==15 for x in meta['planes'])
 D=np.einsum('sd,dcf->csf',VD,state[ids]);N=np.einsum('sd,dcf->csf',VN,state[ids]);pD=np.einsum('sd,dc->cs',VD,phi[ids]);pN=np.einsum('sd,dc->cs',VN,phi[ids])
 if wall:
  D+=np.einsum('sj,jcf->csf',BD,nodes)+query.transpose(1,0,2);N+=np.einsum('sj,jcf->csf',BN,nv)
  pD+=np.einsum('sj,jc->cs',BD,pn)+pq.T;pN+=np.einsum('sj,jc->cs',BN,pnormal)
 return D,N,pD,pN,dict(ids=ids,VD=VD,VN=VN,BD=BD,BN=BN,wall=wall,family=family,condition=condition)
