"""bounded indexed parallel assembly mass windows and per-plane pair assembly."""
import numpy as np
from scipy import sparse


def wall_batch(ev,T,layout,wall,a,nodes):
 mu=wall[:,6];ell=wall[:,8]
 wp=wall[:,1:4];ft=layout.landing(wall[:,0].astype(int),wall[:,4:6]);d=ft-wp;d[:,1:]=(d[:,1:]+np.pi)%(2*np.pi)-np.pi
 _,Mw,_,_=ev(wp)
 _,Ms,_,_=ev(wp+.5*d)
 value=ev.volume_derivatives(wp+.25*d)
 Bh,_,div=value['B'],value['M'],value['divB']
 As=mu*Mw*np.exp(div/Bh[:,2]*.5*d[:,2])/Ms;rr=As/mu;Vwall=ell*(mu+As)/4
 _,value,col,valid=T.evaluate(wall[:,0].astype(int),wall[:,4:6]);valid&=value!=0;rows=np.broadcast_to(np.arange(len(wall))[:,None],value.shape);R=sparse.csr_matrix((value[valid],(rows[valid],col[valid]-a)),shape=(len(wall),nodes))
 return R,rr,Vwall


