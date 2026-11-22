"""Cubic dense ray histories from a caller-supplied exact-key index or local tracing."""
import hashlib
from pathlib import Path

import numpy as np

from .storage import arrays, connect_ro
from .compiled_fieldline_tracer import trace


class Rays:
 def __init__(self,ev,steps,indexes=(),kernel=None,relocations=None,plane=None,request_plan=False,key_namespace=''):
  self.ev=ev;self.steps=steps;self.kernel=kernel;self.relocations=relocations or {}
  self.key_namespace=key_namespace.encode();self.plane=plane;self.request_plan=request_plan;self.dbs=[]
  try:
   for path in indexes:self.dbs.append((connect_ro(path),Path(path).parent))
  except BaseException:
   self.close()
   raise
 def close(self):
  for db,_ in self.dbs:db.close()
  self.dbs.clear()
 def read(self,seeds,delta,label=''):
  seeds=np.array(seeds,float);seeds[:,1:]%=2*np.pi;delta=np.broadcast_to(delta,(len(seeds),)).copy()
  if not self.dbs:
   return dict(seed=seeds,delta=delta,**trace(self.ev,seeds,delta,self.steps))
  found={}
  for i,(q,d) in enumerate(zip(seeds,delta)):
   key=hashlib.sha256(self.key_namespace+self.kernel.encode()+np.asarray(q,dtype='<f8').tobytes()+np.asarray(d,dtype='<f8').tobytes()).hexdigest()
   row=None
   for db,base in self.dbs:
    row=db.execute('select file,idx,status from rays where key=?'+(' and plane=?' if self.request_plan else ''),(key,self.plane) if self.request_plan else (key,)).fetchone()
    if row and row[2]=='complete':break
   if row is None or row[2]!='complete':
    raise KeyError(dict(reason='Exact history missing',plane=self.plane,label=label,key=key,seed=q.tolist(),delta=float(d)))
   p=Path(row[0]);p=p if p.is_absolute() else base/p;p=Path(self.relocations.get(str(p),str(p)))
   found.setdefault(p,[]).append((i,row[1]))
  out={}
  for p,inds in found.items():
   z=arrays(p);ii,jj=np.asarray(inds).T
   if tuple(z['history'].shape[1:])!=(self.steps,4,3):raise RuntimeError('Unexpected dense history shape')
   if not out:out={k:np.empty((len(seeds),)+v.shape[1:],v.dtype) for k,v in z.items()}
   for k in out:out[k][ii]=z[k][jj]
  if not np.array_equal(out['seed'],seeds) or not np.array_equal(out['delta'],delta):raise RuntimeError('Ray bytes do not match exact request')
  return out
 def sample(self,z,fraction):
  t=np.broadcast_to(fraction,(len(z['seed']),));i=np.minimum((t*self.steps).astype(int),self.steps-1);s=t*self.steps-i;c=z['history'][np.arange(len(t)),i];return np.einsum('ni,nij->nj',np.column_stack([np.ones(len(t)),s,s*s,s*s*s]),c)
 def observe(self,xy,k,hop):
  if not len(xy):
   return np.empty(0,bool),np.empty((0,2,2)),np.empty(0)
  pol=np.column_stack([np.minimum(np.linalg.norm(xy,axis=1),1-1e-10),np.arctan2(xy[:,1],xy[:,0])%(2*np.pi),np.full(len(xy),(k+1)*hop)]);pol[:,0]=np.maximum(pol[:,0],1e-10);zs=[self.read(pol,s*hop) for s in [-1,1]];times=[np.where(z['hit'],abs((z['wall'][:,2]-pol[:,2]+np.pi)%(2*np.pi)-np.pi)/hop,1.) for z in zs];good=(times[0]>=.5)&(times[1]>=.5);pos=[self.sample(z,.5) for z in zs];return good,np.stack([p[:,:2] for p in pos],1),pos[0][:,2]+pos[1][:,2]
