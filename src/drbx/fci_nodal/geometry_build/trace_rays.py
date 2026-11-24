import numpy as np
def sample(z,fraction):
 t=np.broadcast_to(fraction,(len(z['seed']),));i=np.minimum((t*64).astype(int),63);s=t*64-i;c=z['history'][np.arange(len(t)),i]
 return np.einsum('ni,nij->nj',np.column_stack([np.ones(len(t)),s,s*s,s*s*s]),c)
