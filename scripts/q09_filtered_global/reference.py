"""Smooth-only independent continuum source, same Q09 flux-difference formula."""
import numpy as np
from scripts.q09_evolved_mms.mms import smooth_fields, ContinuumReference, COEFFICIENTS

FIELDS=('values','gradients','phi_gradient','diffusion','kappa','bmag','owner_raw','owner_weight')


def prepare(bank,geometry,geom):
    p=bank.diagnostics['slot_points'][:,2]; J,b,B=map(np.asarray,geom(p))
    values,grad,_,pg=map(np.asarray,smooth_fields(p,0.))
    diff=[];kap=[]
    for h in (1e-4,5e-5,2.5e-5):
        pp=np.broadcast_to(p[:,None,None,:],(len(p),3,4,3)).copy()
        for d in range(3): pp[:,d,:,d]+=np.array([-2,-1,1,2])*h
        flat=pp.reshape(-1,3); jj,bb,_=map(np.asarray,geom(flat))
        gg=np.asarray(smooth_fields(flat,0.)[1]); gf=np.einsum('pa,pfa->pf',bb,gg)
        flux=(jj[:,None,None]*bb[:,:,None]*gf[:,None,:]).reshape(len(p),3,4,3,6)
        jb=(jj[:,None]*bb).reshape(len(p),3,4,3); w=np.array([1,-8,8,-1])/(12*h)
        df=sum(np.einsum('rqf,q->rf',flux[:,d,:,d],w) for d in range(3))/J[:,None]
        kk=sum(np.einsum('rq,q->r',jb[:,d,:,d],w) for d in range(3))/J
        diff.append(COEFFICIENTS*np.sum(df[bank.owner_raw]*bank.owner_weight[...,None],axis=1));kap.append(kk)
    np.testing.assert_allclose(b[:,2],geometry['b_eta'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(B,geometry['bmag'],rtol=0,atol=1e-10)
    arrays=dict(values=values,gradients=np.einsum('ra,rfa->rf',b,grad),phi_gradient=np.einsum('ra,ra->r',b,pg),diffusion=diff[0],kappa=kap[0],bmag=B,owner_raw=bank.owner_raw,owner_weight=bank.owner_weight)
    if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError('nonfinite smooth reference')
    diagnostics=dict(steps=[1e-4,5e-5,2.5e-5],diffusion_step_max=[float(abs(v-diff[0]).max()) for v in diff[1:]],kappa_step_max=[float(abs(v-kap[0]).max()) for v in kap[1:]],source='independent coordinate flux; filtered b; no Q row forcing')
    return ContinuumReference(**arrays,diagnostics=diagnostics)
