"""Owner-row CSR products with explicit sparse halo exchange.

No all-gather or full-vector reduction in an application. Only the indices
needed by another owner are sent. Transposed observations reverse the same
halo schedule after local deposition, rather than sending record vectors.
"""
from functools import lru_cache
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.sparse import BCSR
from jax.scipy.linalg import lu_solve

def _expand(x,send,delta,mesh):
 chunks=[x]
 for k,d in enumerate(delta):
  v=x[send[k]]
  chunks.append(jax.lax.ppermute(v,'records',[(p,(p+d)%mesh.size) for p in range(mesh.size)]))
 return jnp.concatenate(chunks,axis=0)

@lru_cache(None)
def csr_kernel(mesh,rows,cols,delta,width,transpose):
 def product(data,index,ptr,send,valid,order,inverse,x):
  data,index,ptr,send,valid,order,inverse=data[0],index[0],ptr[0],send[0],valid[0],order[0],inverse[0]
  if not transpose:
   xx=_expand(x,send,delta,mesh)[order]
   return BCSR((data,index,ptr),shape=(rows,cols+len(delta)*width),indices_sorted=True,unique_indices=True)@xx
  ext=BCSR((data,index,ptr),shape=(cols+len(delta)*width,rows),indices_sorted=True,unique_indices=True)@x
  ext=ext[inverse]
  y=ext[:cols]
  for k,d in enumerate(delta):
   ghost=ext[cols+k*width:cols+(k+1)*width]
   recv=jax.lax.ppermute(ghost,'records',[(p,(p-d)%mesh.size) for p in range(mesh.size)])
   y=y.at[send[k]].add(recv*valid[k,:,None])
  return y
 return jax.jit(jax.shard_map(product,mesh=mesh,in_specs=(P('records'),)*8,out_specs=P('records'),check_vma=False))

@jax.tree_util.register_pytree_node_class
class HaloCSR:
 def __init__(self,arrays,mesh,rows,cols,delta,width):
  self.arrays=arrays;self.mesh=mesh;self.rows=rows;self.cols=cols;self.delta=tuple(delta);self.width=width
 def tree_flatten(self):return self.arrays,(self.mesh,self.rows,self.cols,self.delta,self.width)
 @classmethod
 def tree_unflatten(cls,aux,children):return cls(children,*aux)
 def __matmul__(self,x):return csr_kernel(self.mesh,self.rows,self.cols,self.delta,self.width,False)(*self.arrays[:3],*self.arrays[6:],x)
 def transpose_apply(self,x):return csr_kernel(self.mesh,self.rows,self.cols,self.delta,self.width,True)(*self.arrays[3:6],*self.arrays[6:],x)


@lru_cache(None)
def solve_kernel(mesh,n):
 def solve(ids,lu,piv,valid,support,H,rhs):
  ids,lu,piv,valid=ids[0],lu[0],piv[0],valid[0]
  b=rhs[ids]*valid[:,:,None]
  y=jax.vmap(lambda a,p,b:lu_solve((a,p),b))(lu,piv,b)
  local=jnp.zeros_like(rhs).at[ids.reshape(-1)].add((y*valid[:,:,None]).reshape(-1,rhs.shape[1]))
  return jnp.where(support[:,None],local,rhs/H[:,None])
 return jax.jit(jax.shard_map(solve,mesh=mesh,in_specs=(P('records'),)*7,out_specs=P('records'),check_vma=False))

@jax.tree_util.register_pytree_node_class
class OwnedSolve:
 def __init__(self,arrays,mesh,n):self.arrays=arrays;self.mesh=mesh;self.n=n
 def tree_flatten(self):return self.arrays,(self.mesh,self.n)
 @classmethod
 def tree_unflatten(cls,aux,children):return cls(children,*aux)
 def __call__(self,H,rhs):return solve_kernel(self.mesh,self.n)(*self.arrays,H,rhs)
