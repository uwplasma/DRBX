"""Geometry-only support checks and q3 face-row assembly for the frozen candidate."""
import time
import numpy as np
import kernels as r
c=r
ap=r

import topology


# Frozen from support_worker.py
def ring_helper(t):
 cache={};members={};omega={}
 def mem(o):
  o=int(o)
  if o not in members:members[o]=r.members(t,o);omega[o]=t.rv[members[o]]/t.vol[o]
  return members[o]
 def ring(i,k):
  key=(int(i),int(k))
  if key not in cache:
   ids=np.unique(t.ro.reshape((t.n,t.n,t.n))[i,:,k]);th=[]
   for o in ids:
    mm=mem(o);a=t.pts[mm,1]
    th.append(float(a[0]) if len(mm)==1 else float(np.angle(np.mean(np.exp(1j*a)))%(2*np.pi)))
   cache[key]=(ids,np.asarray(th))
  return cache[key]
 return mem,omega,ring


# Frozen from support_worker.py
def ringwise(t,key,p,integ,mem,omega,ring,fitcache):
 n=t.n;ax,i=map(int,key[:2]);anchor=i if ax==0 else i+.5;layers=np.arange(i-2,i+2) if ax==0 else np.arange(i-1,i+3);rid=np.where(layers<0,-layers-1,layers)
 L,D=r.radial(layers+.5-anchor,p[:,0]*n-anchor);D*=n
 maxres=0.;minar=7;minur=7;maxdon=0;systems=0;row_ids=[];row_coef=[]
 for qindex,q in enumerate(p):
  ei=r.nearest(t.centers[2],q[2],4,t.g.eta_period);ev,ed=r.eta_rows(t.centers[2][ei],q[2],t.g.eta_period,t.g.deta)
  for l,radial in enumerate(rid):
   theta=float((q[1]+(np.pi if layers[l]<0 else 0))%(2*np.pi))
   for e,kk in enumerate(ei):
    ids,angles=ring(radial,kk);sel=r.nearest(angles,theta,7,2*np.pi);don=ids[sel];nodes=angles[sel]
    ident=(int(radial),int(kk),*map(int,don))
    if ident not in fitcache:
     A=np.array([omega[o]@c.cardinal(nodes,t.pts[mem(o),1])[0] for o in don])
     U=np.array([c.cardinal(nodes,t.pts[mem(o),1])[0].mean(axis=0) for o in don])
     C,rank,_=c.pinv(A,np.ones(7));CU,urank,_=c.pinv(U,np.ones(7));fitcache[ident]=(A,U,C,CU,rank,urank)
    A,U,C,CU,rank,urank=fitcache[ident];vv,dd=c.cardinal(nodes,[theta]);target=np.vstack((vv,dd))
    res=max(c.resid(target,C,A),c.resid(target,CU,U));maxres=max(maxres,res);minar=min(minar,rank);minur=min(minur,urank);maxdon=max(maxdon,len(don));systems+=1
    value=vv[0]@C;deriv=dd[0]@C
    weight=(integ[qindex,0]*D[qindex,l]*ev[e]*value
            +integ[qindex,1]*L[qindex,l]*ev[e]*deriv
            +integ[qindex,2]*L[qindex,l]*ed[e]*value)
    row_ids.extend(don);row_coef.extend(weight)
 donors,coefficients=compact(row_ids,row_coef)
 return dict(supported=maxres<=1e-9,min_actual_rank=minar,min_uniform_rank=minur,max_residual=maxres,cubic_expansion=0,quartic_expansion=0,max_plane_donors=maxdon,systems=systems,full_rank=minar==7 and minur==7,row_ids=donors,row_coefficients=coefficients)


# Frozen from support_worker.py
def coupled(t,key,p,integ,mem,omega,ring,fitcache):
 n=t.n;ax,i=map(int,key[:2]);layers=np.arange(i-2,i+2) if ax==0 else np.arange(i-1,i+3);rid=np.where(layers<0,-layers-1,layers)
 center=np.array((p[4,0]*np.cos(p[4,1]),p[4,0]*np.sin(p[4,1])));scale=max(t.g.dr,p[4,0]*t.g.dtheta)
 ei,ev,ed=ap.eta_plane_rows(t,p);planes=np.unique(ei);worst=0.;minar=15;minur=15;maxdon=0;cubicextra=0;quartextra=0;valid=True;allfull=True;row_ids=[];row_coef=[]
 def donors(lo,hi,kk):
  selected=[]
  for radial in range(lo,hi+1):
   ids,angles=ring(radial,kk);sel=r.nearest(angles,p[4,1],min(7,len(ids)),2*np.pi);selected.extend(map(int,ids[sel]))
  return np.asarray(selected,int)
 def systems(don,kk):
  # Reuse only exactly identical geometry/plane/support moments; targets vary by face.
  cachekey=(int(kk),float(center[0]),float(center[1]),float(scale),*map(int,don))
  if cachekey not in fitcache:
   A=[];U=[]
   for oid in don:
    mm=mem(oid);b=ap.basis(t.xy[mm],center,scale,ap.EXP4)
    A.append(omega[int(oid)]@b);U.append(b.mean(axis=0))
   A=np.asarray(A);U=np.asarray(U)
   root=1/(1+np.linalg.norm((t.g.owner_centroid_xy[don]-center)/scale,axis=1)**2)
   C4,ra4,_=ap.pinv(A,root);CU4,ru4,_=ap.pinv(U,root)
   C3,ra3,_=ap.pinv(A[:,:10],root);CU3,ru3,_=ap.pinv(U[:,:10],root)
   fitcache[cachekey]=(A,U,C4,CU4,ra4,ru4,C3,CU3,ra3,ru3)
  return fitcache[cachekey]
 for kk in planes:
  t3=ap.target_for_plane(p,integ,ei,ev,ed,kk,center,scale,r.EXP);t4=ap.target_for_plane(p,integ,ei,ev,ed,kk,center,scale,ap.EXP4)
  cubicok=False
  for level in range(4):
   lo=max(0,int(rid.min())-level);hi=min(n-1,int(rid.max())+level);don=donors(lo,hi,kk)
   A,U,C4,CU4,ra4,ru4,C3,CU3,ra3,ru3=systems(don,kk)
   res3=max(ap.residual(t3,C3,A[:,:10]),ap.residual(t3,CU3,U[:,:10]))
   if res3<=1e-9:cubicok=True;break
  if not cubicok:valid=False;worst=max(worst,res3);continue
  cubicextra=max(cubicextra,level);base_lo,base_hi=lo,hi
  quarticok=False
  for extra in range(3):
   lo=max(0,base_lo-extra);hi=min(n-1,base_hi+extra);don=donors(lo,hi,kk)
   A,U,C4,CU4,ra4,ru4,C3,CU3,ra3,ru3=systems(don,kk)
   res4=max(ap.residual(t4,C4,A),ap.residual(t4,CU4,U))
   if res4<=1e-9:quarticok=True;break
  if not quarticok:valid=False
  worst=max(worst,res4);minar=min(minar,ra4);minur=min(minur,ru4);maxdon=max(maxdon,len(don));quartextra=max(quartextra,extra);allfull&=(ra4==15 and ru4==15)
  if quarticok:row_ids.extend(don);row_coef.extend(t4@C4)
 donors,coefficients=compact(row_ids,row_coef)
 return dict(supported=valid,min_actual_rank=minar,min_uniform_rank=minur,max_residual=worst,cubic_expansion=cubicextra,quartic_expansion=quartextra,max_plane_donors=maxdon,systems=len(planes),full_rank=allfull,row_ids=donors,row_coefficients=coefficients)


# Frozen from support_worker.py
def support_execute(t,ref,faceids,family):
 begun=time.process_time();n=t.n;keys=topology.decode(n,faceids);points,weights=r.num.quadrature(t.faces,keys,3,face=True)
 needed=np.flatnonzero(np.isin(family,(6,7)));integrands=np.zeros((len(faceids),9,3))
 if len(needed):
  tensor=ref._perpendicular_flux_tensor(points[needed].reshape(-1,3)).reshape(len(needed),9,3,3)
  integrands[needed]=weights[needed,:,None]*tensor[np.arange(len(needed)),:,keys[needed,0],:]
 mem,omega,ring=ring_helper(t);fitcache={};couplecache={};out=[]
 for j,(key,fa) in enumerate(zip(keys,family)):
  value=ringwise(t,key,points[j],integrands[j],mem,omega,ring,fitcache) if fa==6 else coupled(t,key,points[j],integrands[j],mem,omega,ring,couplecache)
  out.append(value)
 return out,dict(cpu_seconds=time.process_time()-begun,peak_rss_gib=r.rss(),angular_factorizations=len(fitcache),coupled_factorizations=len(couplecache),faces=len(faceids))


# Frozen from assembly_worker.py
def compact(ids,coef):
 ids=np.asarray(ids,int);coef=np.asarray(coef,float);unique,inv=np.unique(ids,return_inverse=True);w=np.bincount(inv,weights=coef,minlength=len(unique))
 keep=np.abs(w)>0;return unique[keep],w[keep]


# Frozen from assembly_worker.py
def raw_singleton_row(t,key,p,integ):
 n=t.n;ax,i=map(int,key[:2]);anchor=i if ax==0 else i+.5;layers=np.arange(i-2,i+2) if ax==0 else np.arange(i-1,i+3)
 L,D=r.radial(layers+.5-anchor,p[:,0]*n-anchor);D*=n;radial=np.where(layers<0,-layers-1,layers)
 ids=[];coeff=[]
 for q,point in enumerate(p):
  ti=r.nearest(t.centers[1],point[1],7,2*np.pi);tv,td=r.theta_rows(t.centers[1][ti],point[1])
  ei=r.nearest(t.centers[2],point[2],4,t.g.eta_period);ev,ed=r.eta_rows(t.centers[2][ei],point[2],t.g.eta_period,t.g.deta)
  theta=(ti[None,:]+np.where(layers<0,n//2,0)[:,None])%n
  raw=((radial[:,None,None]*n+theta[:,None,:])*n+ei[None,:,None])
  oid=t.ro[raw]
  if not np.all(t.starts[oid+1]-t.starts[oid]==1):raise ValueError('ordinary donor is aggregated')
  w=(integ[q,0]*D[q,:,None,None]*ev[None,:,None]*tv[None,None,:]
     +integ[q,1]*L[q,:,None,None]*ev[None,:,None]*td[None,None,:]
     +integ[q,2]*L[q,:,None,None]*ed[None,:,None]*tv[None,None,:])
  ids.extend(oid.ravel());coeff.extend(w.ravel())
 return compact(ids,coeff)


# Frozen from assembly_worker.py
def ringwise_row(t,key,p,integ,mem,omega,ring,fitcache):
 n=t.n;ax,i=map(int,key[:2]);anchor=i if ax==0 else i+.5;layers=np.arange(i-2,i+2) if ax==0 else np.arange(i-1,i+3)
 L,D=r.radial(layers+.5-anchor,p[:,0]*n-anchor);D*=n;radial=np.where(layers<0,-layers-1,layers)
 ids=[];coeff=[]
 for q,point in enumerate(p):
  ei=r.nearest(t.centers[2],point[2],4,t.g.eta_period);ev,ed=r.eta_rows(t.centers[2][ei],point[2],t.g.eta_period,t.g.deta)
  for l,rr in enumerate(radial):
   theta=float((point[1]+(np.pi if layers[l]<0 else 0))%(2*np.pi))
   for e,kk in enumerate(ei):
    owners,angles=ring(rr,kk);sel=r.nearest(angles,theta,7,2*np.pi);don=owners[sel];nodes=angles[sel];tag=(int(rr),int(kk),*map(int,don))
    if tag not in fitcache:
     A=np.array([omega[o]@c.cardinal(nodes,t.pts[mem(o),1])[0] for o in don]);U=np.array([c.cardinal(nodes,t.pts[mem(o),1])[0].mean(axis=0) for o in don])
     C,ra,_=c.pinv(A,np.ones(7));CU,ru,_=c.pinv(U,np.ones(7));fitcache[tag]=(A,U,C,CU,ra,ru)
    A,U,C,CU,ra,ru=fitcache[tag];v,d=c.cardinal(nodes,[theta]);targets=np.vstack((v,d))
    assert max(c.resid(targets,C,A),c.resid(targets,CU,U))<=1e-9
    value=v[0]@C;deriv=d[0]@C
    w=integ[q,0]*D[q,l]*ev[e]*value+integ[q,1]*L[q,l]*ev[e]*deriv+integ[q,2]*L[q,l]*ed[e]*value
    ids.extend(don);coeff.extend(w)
 return compact(ids,coeff)


# Frozen from assembly_worker.py
def coupled_row(t,key,p,integ,mem,omega,ring):
 n=t.n;ax,i=map(int,key[:2]);layers=np.arange(i-2,i+2) if ax==0 else np.arange(i-1,i+3);rid=np.where(layers<0,-layers-1,layers)
 center=np.array((p[4,0]*np.cos(p[4,1]),p[4,0]*np.sin(p[4,1])));scale=max(t.g.dr,p[4,0]*t.g.dtheta)
 ei,ev,ed=ap.eta_plane_rows(t,p);ids=[];coeff=[]
 def donors(lo,hi,kk):
  out=[]
  for radial in range(lo,hi+1):
   owners,angles=ring(radial,kk);sel=r.nearest(angles,p[4,1],min(7,len(owners)),2*np.pi);out.extend(owners[sel])
  return np.asarray(out,int)
 for kk in np.unique(ei):
  t3=ap.target_for_plane(p,integ,ei,ev,ed,kk,center,scale,r.EXP);t4=ap.target_for_plane(p,integ,ei,ev,ed,kk,center,scale,ap.EXP4)
  for level in range(4):
   lo=max(0,int(rid.min())-level);hi=min(n-1,int(rid.max())+level);don=donors(lo,hi,kk)
   A,U,root,C4,ra4,ru4,_,_,_=ap.fit(t,don,center,scale,t4)
   C3,ra3,_=ap.pinv(A[:,:10],root);CU3,ru3,_=ap.pinv(U[:,:10],root)
   res3=max(ap.residual(t3,C3,A[:,:10]),ap.residual(t3,CU3,U[:,:10]))
   if res3<=1e-9:break
  else:raise RuntimeError(('unsupported cubic',key,kk))
  for extra in range(3):
   donor=donors(max(0,lo-extra),min(n-1,hi+extra),kk)
   A,U,root,C4,ra4,ru4,_,_,res4=ap.fit(t,donor,center,scale,t4)
   if res4<=1e-9:break
  else:raise RuntimeError(('unsupported quartic',key,kk))
  ids.extend(donor);coeff.extend(t4@C4)
 return compact(ids,coeff)


# Frozen from assembly_worker.py
def assembly_rows(t,ref,faceids,family,prepared=None):
 started=time.process_time();n=t.n;keys=topology.decode(n,faceids);p,w=r.num.quadrature(t.faces,keys,3,face=True)
 active=np.flatnonzero(family!=0);integ=np.zeros((len(faceids),9,3))
 if len(active):
  tensor=ref._perpendicular_flux_tensor(p[active].reshape(-1,3)).reshape(len(active),9,3,3)
  integ[active]=w[active,:,None]*tensor[np.arange(len(active)),:,keys[active,0],:]
 mem,omega,ring=ring_helper(t);ringcache={};top=None;out=[]
 for j,(key,fa) in enumerate(zip(keys,family)):
  if fa==0:ids=np.array([],int);coef=np.array([],float);bc=None
  elif fa in (1,2,3,4):
   if top is None:
    grid={'grid.y.centers':t.centers[1],'grid.z.centers':t.centers[2]};top=(t.ro,t.rv,t.vol,t.pts,t.order,t.starts,grid)
   z=r.boundary_map(n,'full',key,int(faceids[j]),p[j],w[j],integ[j],top,t.g.eta_period)
   ids=z['donor_ids'];coef=z['W'];bc=dict(value_loading=z['value_loading'],tangential_loading=z['tangential_loading'],trace_donor_points=z['trace_donor_points'],trace_target_points=z['trace_target_points'])
  elif fa==5:ids,coef=raw_singleton_row(t,key,p[j],integ[j]);bc=None
  elif fa in (6,7) and prepared is not None:
   ids,coef=prepared[int(faceids[j])];bc=None
  elif fa==6:ids,coef=ringwise_row(t,key,p[j],integ[j],mem,omega,ring,ringcache);bc=None
  elif fa==7:ids,coef=coupled_row(t,key,p[j],integ[j],mem,omega,ring);bc=None
  else:raise ValueError(fa)
  out.append((ids,coef,bc))
 return out,dict(cpu_seconds=time.process_time()-started,peak_rss_gib=r.rss(),faces=len(faceids),ringwise_factorizations=len(ringcache)),p,integ
