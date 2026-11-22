"""Compact face leaf and fixed-order owner reduction from bounded indexed parallel assembly."""
import numpy as np
from scipy import sparse

from .indexed_partials import Indexed, Tree


def row(val,col,n):
 nr,width=val.shape;a=sparse.csr_matrix((val.copy().ravel(),col.ravel().astype(np.int32 if n<2**31 else np.int64),np.arange(nr+1,dtype=np.int64)*width),shape=(nr,n));a.eliminate_zeros();return a


def face_leaf(lo,hi,options,F,T):
 K=Tree();C=Tree();H=None
 for a in range(lo,hi,options.face_chunk):
  ff=F[a:min(hi,a+options.face_chunk)];ol=ff[:,0].astype(int);orr=ff[:,1].astype(int)
  kl,vl,cl,validl=T.evaluate(ol,ff[:,5:7]);kr,vr,cr,validr=T.evaluate(orr,ff[:,7:9]);ids=np.union1d(cl[vl!=0],cr[vr!=0])
  # Padding columns carry zero values; remap them to any valid local column.
  il=np.searchsorted(ids,cl);ir=np.searchsorted(ids,cr);il[vl==0]=0;ir[vr==0]=0
  L=row(vl,il,len(ids));R=row(vr,ir,len(ids));g=(sparse.diags(1/ff[:,9])@(R-L)).tocsr();A=((L+R)*.5).tocsr();w=ff[:,9]*ff[:,10];wg=g.multiply(w[:,None]).tocsr()
  K.push(Indexed(ids,ids,(g.T@wg).tocsr()));C.push(Indexed(ids,ids,(A.T@wg).tocsr()))
  h=Indexed(ids,None,np.asarray(A.T@w));H=h if H is None else H+h
 return dict(K=K.total(),C=C.total(),H0=H)


def csr_rows(part,lo,hi,ncols):
 rows=part.rows-lo;ptr=np.zeros(hi-lo+1,np.int64);ptr[rows+1]=np.diff(part.value.indptr);np.cumsum(ptr,out=ptr)
 return sparse.csr_matrix((part.value.data,part.cols[part.value.indices],ptr),shape=(hi-lo,ncols))


