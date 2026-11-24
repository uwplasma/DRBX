import numpy as np


from ._geometry import TAU

def seeds(xy,eta):return np.c_[np.clip(np.linalg.norm(xy,axis=1),1e-10,1-1e-10),np.arctan2(xy[:,1],xy[:,0])%TAU,np.full(len(xy),eta)]

def edges(t):return np.unique(np.sort(np.concatenate([t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]]),axis=1),axis=0)
