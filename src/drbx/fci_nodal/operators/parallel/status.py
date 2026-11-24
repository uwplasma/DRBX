import numpy as np
import jax,jax.numpy as jnp
from . import checked_characteristics as cf
def initial_status():return (jnp.int64(0),jnp.int64(-1),jnp.int32(0),jnp.zeros(6))
def status_add(status,valid,codes,u,ids):
 count,ident,code,state=status;bad=~valid;k=jnp.argmax(bad);n=jnp.sum(bad);take=(count==0)&(n>0)
 return count+n,jnp.where(take,ids[k],ident),jnp.where(take,codes[k],code),jnp.where(take,u[k],state)
def checked(status,location):
 # One device-to-host synchronization for all face/wall validity flags.
 count,ident,code,state=jax.device_get(status)
 if int(count):cf.check(np.array([False]),np.array([code]),np.asarray(state)[None,:],location,np.array([ident]))
