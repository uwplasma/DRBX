"""Positive existing-point degree-2 cubature."""
import numpy as np

def moments(x,degree):
 return np.array([x[:,0]**a*x[:,1]**b for total in range(degree+1) for a in range(total+1) for b in [total-a]])

def reduce_pair(x,weights,degree):
 cap=(degree+1)*(degree+2)//2
 if np.any(weights<=0):raise ArithmeticError('Nonpositive source cubature weight')
 if len(weights)<=cap:return np.arange(len(weights)),weights.copy(),0.
 center=(x.min(0)+x.max(0))/2;scale=np.maximum(np.ptp(x,axis=0),1e-15);V=moments((x-center)/scale,degree);total=weights.sum();w=weights/total;active=[];wa=[]
 for i,wi in enumerate(w):
  active.append(i);wa.append(wi)
  if len(active)<=cap:continue
  A=V[:,active];u,s,vh=np.linalg.svd(A,full_matrices=True);null=vh[-1];positive=null>0
  if not positive.any():null=-null;positive=null>0
  ratios=np.where(positive,np.asarray(wa)/np.where(positive,null,1),np.inf);j=int(np.argmin(ratios));new=np.asarray(wa)-ratios[j]*null;new[j]=0
  if new.min() < -1e-13:raise ArithmeticError(('Negative reduced weight',float(new.min())))
  keep=new>0;active=np.asarray(active)[keep].tolist();wa=new[keep].tolist()
 ids=np.array(active);ans=np.array(wa)*total;error=float(np.max(abs(V[:,ids]@ans-V@weights))/total)
 if len(ids)>cap or np.any(ans<=0) or error>1e-10:raise ArithmeticError(('Cubature moment/positivity failure',len(ids),error))
 return ids,ans,error
