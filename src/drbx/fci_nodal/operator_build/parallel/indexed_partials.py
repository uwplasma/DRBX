import numpy as np
from scipy import sparse

class Indexed:
 def __init__(self,rows,cols,value):self.rows=np.asarray(rows,np.int64);self.cols=None if cols is None else np.asarray(cols,np.int64);self.value=value
 def __add__(self,other):
  if other is None:return self
  rows=np.union1d(self.rows,other.rows)
  if self.cols is None:
   shape=(len(rows),)+self.value.shape[1:];a=np.zeros(shape,dtype=self.value.dtype);b=np.zeros(shape,dtype=other.value.dtype)
   a[np.searchsorted(rows,self.rows)]=self.value;b[np.searchsorted(rows,other.rows)]=other.value
   return Indexed(rows,None,a+b)
  cols=np.union1d(self.cols,other.cols)
  def remap(x):
   a=x.value.tocoo();return sparse.csr_matrix((a.data,(np.searchsorted(rows,x.rows)[a.row],np.searchsorted(cols,x.cols)[a.col])),shape=(len(rows),len(cols)))
  return Indexed(rows,cols,remap(self)+remap(other))
 def subset(self,lo,hi):
  ix=np.flatnonzero((self.rows>=lo)&(self.rows<hi));v=self.value[ix]
  if self.cols is None:return Indexed(self.rows[ix],None,v)
  used=np.unique(v.indices);return Indexed(self.rows[ix],self.cols[used],v[:,used].tocsr())
 def put(self,z,prefix):
  z[prefix+'rows']=self.rows
  if self.cols is None:z[prefix+'value']=self.value
  else:
   z[prefix+'cols']=self.cols;z[prefix+'data']=self.value.data;z[prefix+'indices']=self.value.indices;z[prefix+'indptr']=self.value.indptr
 @classmethod
 def get(cls,z,prefix):
  rows=z[prefix+'rows']
  if prefix+'cols' not in z:return cls(rows,None,z[prefix+'value'])
  cols=z[prefix+'cols'];return cls(rows,cols,sparse.csr_matrix((z[prefix+'data'],z[prefix+'indices'],z[prefix+'indptr']),shape=(len(rows),len(cols))))

class Tree:
 def __init__(self):self.levels=[]
 def push(self,m):
  level=0
  while self.levels and self.levels[-1][0]==level:
   _,q=self.levels.pop();m=q if m is None else m if q is None else q+m;level+=1
  self.levels.append((level,m))
 def total(self):
  m=None
  for _,q in reversed(self.levels):m=q if m is None else m if q is None else q+m
  return m
