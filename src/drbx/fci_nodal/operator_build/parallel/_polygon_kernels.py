"""Numerical helpers bound once per polygon route."""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from scipy.special import roots_legendre

from ...geometry_build.dual_cells import build as build_cells
from ...geometry_build.trace_rays import sample
from ...geometry_build.compiled_fieldline_tracer import trace
from ._geometry import metric
from ._repair import seeds


def bind(options, trace_sides):
 N=options.resolution
 TAU=2*np.pi
 HOP=TAU/N
 SUB=options.face_subdivision

 @lru_cache(maxsize=1)
 def build():
  return build_cells(N)


 def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]

 def clip(poly,values):
     if not len(poly):return poly
     f=poly@values;out=[]
     for i in range(len(poly)):
         a=poly[i-1];b=poly[i];fa=f[i-1];fb=f[i]
         if (fa>=0)!=(fb>=0):out.append((fb*a-fa*b)/(fb-fa))
         if fb>=0:out.append(b)
     return np.array(out).reshape(-1,3)

 def inpoly(poly,land,quad):
     for a,b in zip(quad,np.roll(quad,-1,axis=0)):
         poly=clip(poly,cross(b-a,land-a))
         if len(poly)<3:return np.empty((0,3))
     return poly

 def area_centroid(poly,xy):
     if len(poly)<3:return 0,None
     points=poly@xy;w=cross(points[1:-1]-points[0],points[2:]-points[0])/2;a=w.sum()
     if a<=1e-18:return 0,None
     b=((poly[0]+poly[1:-1]+poly[2:])*w[:,None]).sum(0)/(3*a)
     return a,b

 @lru_cache(maxsize=2)
 def tiles(sub=4):
     cells=build()[1];out=[];owner=[]
     angles=np.unique(np.round(np.concatenate([np.linspace(t0,t1,sub+1)%TAU for r0,r1,t0,t1 in cells]),13))
     for j,(r0,r1,t0,t1) in enumerate(cells):
         local=np.r_[angles-TAU,angles,angles+TAU];ts=np.r_[t0,local[(local>t0+1e-12)&(local<t1-1e-12)],t1];ts.sort()
         for a,b in zip(ts[:-1],ts[1:]):
             q=np.array([[r0*np.cos(a),r0*np.sin(a)],[r1*np.cos(a),r1*np.sin(a)],[r1*np.cos(b),r1*np.sin(b)],[r0*np.cos(b),r0*np.sin(b)]])
             if r0==0:q=q[:3]
             out.append(q);owner.append(j)
     return out,np.array(owner),np.array([q.min(0) for q in out]),np.array([q.max(0) for q in out])


 gu=SimpleNamespace(P=build()[0].P,build=build,cross=cross,inpoly=inpoly,area_centroid=area_centroid,tiles=tiles,metric=metric)

 @lru_cache(maxsize=2)
 def sectors(kind):
  if kind=='half':return np.linspace(0,TAU,4*N+1)
  quads,ow,*_=gu.tiles(SUB);aa=[];cells=gu.build()[1]
  for q,j in zip(quads,ow):
   if cells[j,1]==1:aa.extend(np.arctan2(q[1:3,1],q[1:3,0])%TAU)
  a=np.unique(np.round(np.array(aa)%TAU,12));return np.r_[0,a[(a>1e-10)&(a<TAU-1e-10)],TAU]
 def transform(xy,kind):
  shape=xy.shape;v=np.asarray(xy).reshape(-1,2);r=np.linalg.norm(v,axis=1);th=np.arctan2(v[:,1],v[:,0])%TAU;aa=sectors(kind);idx=np.clip(np.searchsorted(aa,th,side='right')-1,0,len(aa)-2);a=aa[idx];b=aa[idx+1];c=np.cos((b-a)/2)/np.cos(th-(a+b)/2);A=N/c-(N-1);rp=A*r+(N-1)*(c-1);inside=r<=(N-1)/N*c+2e-12;at_edge=(abs(th-a)<2e-12)|(abs(th-b)<2e-12);rp[inside|at_edge]=r[inside|at_edge];jac=A*rp/r;jac[inside]=1
  return (v*(rp/r)[:,None]).reshape(shape),jac.reshape(shape[:-1])

 x,w=roots_legendre(16);x=(x+1)/2;w=w/2
 u=x[:,None];v=x[None,:];BQ=np.stack([np.broadcast_to(1-u,(16,16)),np.broadcast_to(u*(1-v),(16,16)),np.broadcast_to(u*v,(16,16))],-1).reshape(-1,3);WQ=(w[:,None]*w[None,:]*u).ravel()


 def moment(xx,poly,kind=None):
  if kind is None:
   a,b=gu.area_centroid(poly,xx);return a,b,b@xx
  aa=0.;bb=np.zeros(3);mm=np.zeros(2)
  for j in range(1,len(poly)-1):
   coeff=BQ@poly[[0,j,j+1]];ref=coeff@xx;actual,jac=transform(ref,kind);t=poly[[0,j,j+1]]@xx;det=gu.cross(t[1]-t[0],t[2]-t[0]);ww=WQ*jac*det;aa+=ww.sum();bb+=ww@coeff;mm+=ww@actual
  return aa,bb/aa,mm/aa

 def arc_tiles():
  quads,owners,*_=gu.tiles(SUB);new=[];cells=gu.build()[1]
  for q,owner in zip(quads,owners):
   z=q.copy()
   if cells[owner,1]==1:
    th=np.arctan2(z[1:3,1],z[1:3,0]);dt=(th[1]-th[0])%TAU;z[1:3]/=np.cos(dt/2)
   new.append(z)
  return new,owners,np.array([q.min(0) for q in new]),np.array([q.max(0) for q in new])

 def intersections(xx,lands,c,tiles):
  quads,owners,bb0,bb1=tiles;s=0 if c in [0,1] else 1;ls=lands[:,s];left=np.flatnonzero(np.all(bb1>=ls.min(0)-1e-13,axis=1)&np.all(bb0<=ls.max(0)+1e-13,axis=1))
  for l in left:
   lp=gu.inpoly(np.eye(3),ls,quads[l])
   if len(lp)<3:continue
   if c==0:
    pts=lp@lands[:,1];right=np.flatnonzero(np.all(bb1>=pts.min(0)-1e-13,axis=1)&np.all(bb0<=pts.max(0)+1e-13,axis=1))
    for r in right:
     pol=gu.inpoly(lp,lands[:,1],quads[r]);area,_=gu.area_centroid(pol,xx)
     if area>1e-16:yield (owners[l],owners[r]),pol
   else:
    area,_=gu.area_centroid(lp,xx)
    if area>1e-16:yield (owners[l],),lp

 def metric_chunks(ev,q):
  all=[]
  for lo in range(0,len(q),2048):all.append(gu.metric(ev,q[lo:lo+2048],False))
  return {k:np.concatenate([a[k] for a in all]) for k in ['J','B','M','b']}

 def classify(reach):return np.where(reach[:,0],np.where(reach[:,1],0,1),np.where(reach[:,1],2,3))
 def contains(x,t):
  mat=np.column_stack([t[1]-t[0],t[2]-t[0]])
  if abs(np.linalg.det(mat))<1e-18:return None
  v=np.linalg.solve(mat,x-t[0]);b=np.r_[1-v.sum(),v];return b if b.min()>=-1e-10 else None

 def inverse(q):
  r=q[:,0];th=q[:,1];aa=sectors('half');ii=np.clip(np.searchsorted(aa,th,side='right')-1,0,len(aa)-2);a=aa[ii];b=aa[ii+1];c=np.cos((b-a)/2)/np.cos(th-(a+b)/2);rp=(r-(N-1)*(c-1))/(N/c-(N-1));rp[r<=(N-1)/N*c+2e-12]=r[r<=(N-1)/N*c+2e-12];return rp[:,None]*np.c_[np.cos(th),np.sin(th)]


 def observe(ev,ref,k,label):
  xy,_=transform(ref,'half');q=seeds(xy,(k+1)*HOP);zz=trace_sides(ev,q,[-HOP,HOP],[label+f' side{s}' for s in [-1,1]])
  tau=np.stack([np.where(z['hit'],abs((z['wall'][:,2]-q[:,2]+np.pi)%TAU-np.pi)/HOP,1.) for z in zz],1);pp=[sample(z,.5) for z in zz]
  return dict(classes=classify(tau>=.5),maps=np.stack([p[:,:2] for p in pp],1),arcs=np.stack([p[:,2] for p in pp],1),wallhit=np.stack([z['wall'] for z in zz],1),hitell=np.stack([z['wall_length'] for z in zz],1))


 def cut(poly,line,positive):
  if not len(poly):return np.empty((0,2))
  a,b,orient=line;d=gu.cross(b-a,poly-a)*orient*(1 if positive else -1);out=[]
  for x,y,fx,fy in zip(poly,np.roll(poly,-1,axis=0),d,np.roll(d,-1)):
   if fx>=0:out.append(x)
   if (fx>0 and fy<0) or (fy>0 and fx<0):out.append(x+fx/(fx-fy)*(y-x))
  if len(out)<3:return np.empty((0,2))
  out=np.array(out);out=out[np.linalg.norm(out-np.roll(out,1,axis=0),axis=1)>1e-15]
  return out if len(out)>=3 else np.empty((0,2))

 def field_at(x,poly,vals):
  distances=np.linalg.norm(poly-x,axis=1);nearest=np.argmin(distances)
  if distances[nearest]<2e-14:return vals[nearest].copy()
  for j in range(1,len(poly)-1):
   ix=[0,j,j+1];bc=contains(x,poly[ix])
   if bc is not None:return np.einsum('i,i...->...',bc,vals[ix])
  # Independent half-plane intersection arithmetic can place a shared-edge
  # point a few ulps outside a thin fan triangle. Bound the correction in xy.
  for j in range(1,len(poly)-1):
   ix=[0,j,j+1];t=poly[ix];mat=np.column_stack([t[1]-t[0],t[2]-t[0]])
   if np.linalg.det(mat)==0:continue
   u=np.linalg.solve(mat,x-t[0]);bc=np.r_[1-u.sum(),u];clipped=np.maximum(bc,0);clipped/=clipped.sum()
   if np.linalg.norm(clipped@t-x)<2e-14:return np.einsum('i,i...->...',clipped,vals[ix])
  raise ValueError(('point outside direction polygon',x.tolist(),poly.tolist(),distances.tolist()))


 def overlaps(p,c,tiles,selected=None):
  rows=[]
  for i,xx in enumerate(p['xy']):
   if selected is not None and not selected[i]:continue
   for owners,poly in intersections(xx,p['maps'][i],c,tiles):
    area,bary,point=moment(xx,poly,'half');q=seeds(point[None],0)[0];rows.append((i,owners,area,bary,q,poly))
  return rows


 def aggregate(w,psi,side,P):
  out=np.zeros((N*P,2));ids=w[:,9]==0
  np.add.at(out,(w[ids,0].astype(int),((side[ids]+1)//2).astype(int)),psi[ids]);return out


 def make_port(jobs, v, e, local, local_id):
  BISECT = options.bisections

  def port(j):
   a,b,_=jobs[j];d=v['xy'][b]-v['xy'][a];mid=(e['left'][j]+e['right'][j])/2;t=float(np.dot(mid-v['xy'][a],d)/np.dot(d,d));scale=2**(BISECT+1);tt=round(t*scale)/scale;assert abs(t-tt)<1e-9;return (1-tt)*local[local_id[a]]+tt*local[local_id[b]]

  return port

 return SimpleNamespace(
     sectors=sectors, transform=transform, moment=moment, BQ=BQ, WQ=WQ,
     arc_tiles=arc_tiles, intersections=intersections, metric_chunks=metric_chunks,
     classify=classify, contains=contains, inverse=inverse, observe=observe,
     cut=cut, field_at=field_at, overlaps=overlaps, aggregate=aggregate, make_port=make_port,
 )
