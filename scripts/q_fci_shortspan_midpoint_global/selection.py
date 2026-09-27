"""Local-column Q amplification rows and frozen strict family selection."""
import numpy as np
from scripts.q_fci_return_campaign import numerics as qnum

FAMILIES=('K5','K6','K7','K8','rect_2_3','rect_3_3','rect_4_4','nearest_count_4_4')

def prepare_target_geometry(ctx,seeds,midpoint,base=qnum.base,wall=False):
    """One bounded geometry record per raw target, shared by all candidates."""
    seeds=np.asarray(seeds);midpoint=np.asarray(midpoint)
    cached=ctx.get('_q_target_geometry')
    if cached is not None and np.array_equal(cached['seeds'],seeds) and np.array_equal(cached['midpoint'],midpoint):
        if wall and 'ell_n_m' not in cached:
            A=ctx['evaluator']._position_and_jacobian(np.array([[1.,midpoint[1],midpoint[2]]]))[1][0]
            cached['ell_n_m']=(1/ctx['N'])/np.linalg.norm(np.linalg.inv(A)[0])
        return
    J,b,_=base(ctx,seeds)
    pair_midpoints=np.asarray([(seeds[2*i]+seeds[2*i+1])/2 for i in range(len(seeds)//2)])
    Jmid_pair=base(ctx,pair_midpoints)[0]
    Jmid_q=base(ctx,midpoint[None])[0][0]
    ctx['_q_target_geometry']=dict(seeds=seeds.copy(),midpoint=midpoint.copy(),J=J,b=b,Jmid_pair=Jmid_pair,Jmid_q=Jmid_q)
    if wall:
        A=ctx['evaluator']._position_and_jacobian(np.array([[1.,midpoint[1],midpoint[2]]]))[1][0]
        ctx['_q_target_geometry']['ell_n_m']=(1/ctx['N'])/np.linalg.norm(np.linalg.inv(A)[0])

def local_effective_rows(ctx,seeds,M,owner,base=qnum.base):
    """Score the exact donor union plus target correction, no domain-width row."""
    ids=np.union1d(M.indices,[owner]);local=M[:,ids].tocsr();target=int(np.searchsorted(ids,owner));ns=len(seeds)
    cached=ctx.get('_q_target_geometry')
    if cached is not None and np.array_equal(cached['seeds'],seeds):
        J,b,Jmid=cached['J'],cached['b'],cached['Jmid_pair']
    else:
        J,b,_=base(ctx,seeds);mid=np.asarray([(seeds[2*i]+seeds[2*i+1])/2 for i in range(ns//2)]);Jmid=base(ctx,mid)[0]
    H=2*np.pi/ctx['N'];rows=[]
    # Existing interior map order h/4 then h/8.
    for conf,alpha in enumerate((.25,.125)):
        for axis in range(3):
            neg,pos=6*conf+2*axis,6*conf+2*axis+1;d=abs(float(seeds[pos,axis]-seeds[neg,axis]))/2
            coeff=np.zeros(len(ids))
            for sid,sign in ((neg,-1),(pos,1)):
                z=sign*J[sid]*b[sid,axis]*b[sid,2]/(2*d*Jmid[3*conf+axis])
                inner=(4*(local.getrow(3*ns+sid)-local.getrow(2*ns+sid))/(H/8)-(local.getrow(ns+sid)-local.getrow(sid))/(H/4))/3
                coeff+=z*inner.toarray()[0]
            correction=float(np.sum(coeff));coeff[target]-=correction
            rows.append(dict(config=conf,alpha=alpha,axis=axis,l1=float(np.sum(abs(coeff))),l2=float(np.linalg.norm(coeff)),uncentered_sum=correction,centered_sum=float(np.sum(coeff)),correction_owner=int(owner),correction_value=-correction,local_columns=len(ids)))
    return rows

def select_interior(scores):
    """Frozen strict min-A lexicographic selector; no invented tie band."""
    if not scores:return None
    return min(scores,key=lambda f:(scores[f]['A'],scores[f]['max_endpoint_row_l1'],scores[f]['max_member_radius_scaled'],scores[f]['summed_donor_count'],FAMILIES.index(f)))

def select_wall(scores):
    if not scores:return None
    return min(scores,key=lambda f:(scores[f]['A'],scores[f]['endpoint_combined_l1'],scores[f]['radius'],scores[f]['donor_count'],FAMILIES.index(f)))
