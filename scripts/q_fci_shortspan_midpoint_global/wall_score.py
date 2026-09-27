"""Frozen wall candidate action score, extracted from 2026-09-27 pilot."""
import numpy as np
from scripts.q_fci_return_campaign import numerics as qnum
def score(ctx,q,seeds,obj,kind):
    n=ctx['N'];H=2*np.pi/n;cached=ctx.get('_q_target_geometry')
    if cached is not None and np.array_equal(cached['seeds'],seeds[:12]) and np.array_equal(cached['midpoint'],q):
        J,b,Jmid=cached['J'],cached['b'],cached['Jmid_q']
    else:J,b,_=qnum.base(ctx,seeds[:12]);Jmid=qnum.base(ctx,q[None])[0][0]
    if cached is not None and np.array_equal(cached['midpoint'],q) and 'ell_n_m' in cached:ell=cached['ell_n_m']
    else:
        A=ctx['evaluator']._position_and_jacobian(np.array([[1.,q[1],q[2]]]))[1][0]
        sqrtguu=np.linalg.norm(np.linalg.inv(A)[0]);ell=(1/n)/sqrtguu
    EO=obj['Eowner'].reshape(4,12,-1);EB=obj['Ebc'].reshape(4,12,-1)
    ownerrows=[];bcrows=[];staged=[];const=[]
    for conf in range(2):
        for axis in range(3):
            lo=np.zeros(EO.shape[-1]);lb=np.zeros(EB.shape[-1]);sids=(6*conf+2*axis,6*conf+2*axis+1)
            d=abs(seeds[sids[1],axis]-seeds[sids[0],axis])/2
            for sid,sign in zip(sids,(-1,1)):
                z=sign*J[sid]*b[sid,axis]*b[sid,2]/(2*d*Jmid)
                lo+=z*(4*(EO[3,sid]-EO[2,sid])/(H/8)-(EO[1,sid]-EO[0,sid])/(H/4))/3
                lb+=z*(4*(EB[3,sid]-EB[2,sid])/(H/8)-(EB[1,sid]-EB[0,sid])/(H/4))/3
            ownerrows.append(lo);bcrows.append(lb)
            # Independent staged action on all 60 basis functions.
            lhs=lo@obj['M']+lb@obj['C']
            rhs=np.zeros(60)
            T=obj['target'].reshape(4,12,60)
            for sid,sign in zip(sids,(-1,1)):
                z=sign*J[sid]*b[sid,axis]*b[sid,2]/(2*d*Jmid)
                rhs+=z*(4*(T[3,sid]-T[2,sid])/(H/8)-(T[1,sid]-T[0,sid])/(H/4))/3
            staged.append(float(np.max(abs(lhs-rhs))))
            const.append(float(abs(np.sum(lo)+(np.sum(lb) if kind=='D' else 0))))
    owner_l1=np.array([np.sum(abs(x)) for x in ownerrows]);bc_l1=np.array([np.sum(abs(x/(ell if kind=='N' else 1))) for x in bcrows])
    A_score=max(float(np.sum(owner_l1[3*i:3*i+3])+np.sum(bc_l1[3*i:3*i+3])) for i in range(2))
    cells=np.concatenate([obj['index'].members(int(x)) for x in obj['donors']]) if 'index' in obj else None
    return dict(A=A_score,owner_A=max(float(np.sum(owner_l1[3*i:3*i+3])) for i in range(2)),bc_A=max(float(np.sum(bc_l1[3*i:3*i+3])) for i in range(2)),
                max_staged_defect=max(staged),max_constant_action=max(const),ell_n_m=float(ell),
                endpoint_combined_l1=float(np.max(np.sum(abs(obj['Eowner']),axis=1)+np.sum(abs(obj['Ebc']/(ell if kind=='N' else 1)),axis=1))))
