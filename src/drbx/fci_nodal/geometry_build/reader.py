from pathlib import Path
import sqlite3
from collections import OrderedDict
import numpy as np
from .storage import load, verify, arrays
from .._hashing import sha256_file as sha

GEOMETRY_SCHEMA = 'drbx.geometry.partitioned.v2'


class GeometryReader:
 def __init__(self,root,verify_index=True):
  self.root=Path(root);self.manifest=load(self.root/'geometry.json');p=self.root/'index.sqlite'
  if verify_index and sha(p)!=self.manifest['index_sha256']:raise RuntimeError('Geometry index changed')
  self.db=sqlite3.connect(p.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
  self.has_leaf_hashes=bool(self.db.execute('select 1 from sqlite_master where name="leaf_hashes"').fetchone());self.validated=OrderedDict()
 def verify_leaf(self,path):
  if path in self.validated:self.validated.move_to_end(path);return
  if self.has_leaf_hashes:
   row=self.db.execute('select sha256 from leaf_hashes where path=?',(path,)).fetchone()
   if not row or sha(self.root/path/'complete.json')!=row[0]:raise RuntimeError('Leaf manifest changed')
   verify(self.root/path)
  self.validated[path]=True
  while len(self.validated)>128:self.validated.popitem(last=False)
 def close(self):
  self.db.close()
 def shape(self,name):return tuple(self.manifest['arrays'][name]['shape'])
 def read(self,name,lo,hi):
  spec=self.manifest['arrays'][name];n=spec['shape'][0]
  if not 0<=lo<=hi<=n:raise IndexError((name,lo,hi,n))
  if hi-lo>32000:raise MemoryError('Use bounded logical ranges of at most32000 rows')
  out=np.empty((hi-lo,*spec['shape'][1:]),np.dtype(spec['dtype']));coverage=np.zeros(hi-lo,np.uint8)
  for path,key,idskey,a,b in self.db.execute('select path,key,ids_key,lo,hi from chunks where name=? and lo<? and hi>? order by sequence',(name,hi,lo)):
   self.verify_leaf(path)
   z=arrays(self.root/path,[key,idskey]);ids=z[idskey];i=np.searchsorted(ids,lo);j=np.searchsorted(ids,hi);which=ids[i:j]-lo
   if np.any(coverage[which]):raise RuntimeError('Duplicate logical records')
   out[which]=z[key][i:j];coverage[which]=1
  if not np.all(coverage):raise RuntimeError(('Missing immutable range',name,lo,hi,np.flatnonzero(~coverage.astype(bool))[:10].tolist()))
  return out
 def ranges(self,name='faces',outer=32000,inner=8192):
  """Original source shards reset inner8192; geometry files do not."""
  n=self.shape(name)[0]
  for lo in range(0,n,outer):
   hi=min(lo+outer,n)
   for a in range(lo,hi,inner):yield (lo,hi,a,min(a+inner,hi))
 def route(self,physical_plane,owners):return min(physical_plane*owners//self.manifest['E'],owners-1)


 @property
 def faces(self):return RangeArray(self,'faces')

class RangeArray:
 """Bounded c['face'][lo:hi] interface with an explicit bounded slice cap."""
 def __init__(self,reader,name):self.reader=reader;self.name=name;self.shape=reader.shape(name);self.dtype=np.dtype(reader.manifest['arrays'][name]['dtype'])
 def __len__(self):return self.shape[0]
 def __getitem__(self,item):
  if not isinstance(item,slice) or item.step not in [None,1]:raise TypeError('Use explicit canonical slice; arbitrary gathers belong to a destination index')
  lo,hi,_=item.indices(len(self))
  if hi-lo>32000:raise MemoryError('Global geometry gather refused; use bounded indexed ranges')
  return self.reader.read(self.name,lo,hi)
 def __array__(self,*args,**kw):raise MemoryError('Global geometry materialization is not supported')
