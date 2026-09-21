"""Unmodified reference selector body for bounded preflight checks."""
import numpy as np
from collections import Counter
from numerics import ex
run=ex.run;an=ex.an;directional=ex.directional

def selector(rows,original,saved,pools,center,scale,target,whole,max_exchanges=32):
    union=np.unique(np.concatenate([saved,*[p['rows'] for p in pools.values()]]))
    A,distance=run.compact.row_moments(rows,union,center,scale,5)
    weight=run.compact.distance_weight(distance);vectors=A[:,run.prior.F_M3]*weight[:,None];t=target[run.prior.F_M3]
    source,target_points=an.points(rows,union,center,scale)
    sectors=directional.sector(0.5*(source+target_points));directions=rows['direction'][union].astype(str);planes=rows['source_plane'][union]
    lookup={int(row):i for i,row in enumerate(union)};selected=np.array([lookup[int(row)] for row in saved],int);protected=set(map(int,original)) if not whole else set()
    required_sector={(int(planes[i]),int(sectors[i])) for i in selected};required_direction={(int(planes[i]),str(directions[i])) for i in selected}
    def factor(indices):
        U,s,Vt=np.linalg.svd(vectors[indices].T,full_matrices=False)
        tol=np.finfo(float).eps*max(vectors[indices].T.shape)*s[0]
        rank=int(np.count_nonzero(s>tol))
        if rank!=len(t):return None
        beta=(U.T@t)/s
        return U,s,beta,float(beta@beta)
    fac=factor(selected)
    if fac is None:raise RuntimeError('initial cubic rank deficient')
    initial=fac[3];log=[];rejected=0;unstable_removals=0;stop='cap';last_gain=None
    for iteration in range(max_exchanges):
        U,s,beta,obj=fac;Z=(vectors@U)/s[None,:];alpha=Z@beta;leverage=np.sum(Z*Z,axis=1)
        available=np.setdiff1d(np.arange(len(union)),selected,assume_unique=False)
        sector_count=Counter((int(planes[i]),int(sectors[i])) for i in selected)
        dir_count=Counter((int(planes[i]),str(directions[i])) for i in selected)
        offers=[]
        for position,out in enumerate(selected):
            if int(union[out]) in protected:continue
            eligible=available[planes[available]==planes[out]]
            if (int(planes[out]),int(sectors[out])) in required_sector and sector_count[(int(planes[out]),int(sectors[out]))]==1:
                eligible=eligible[sectors[eligible]==sectors[out]]
            if (int(planes[out]),str(directions[out])) in required_direction and dir_count[(int(planes[out]),str(directions[out]))]==1:
                eligible=eligible[directions[eligible]==directions[out]]
            if not len(eligible):continue
            denominator=1.0-leverage[out]
            if denominator<=1e-10:
                unstable_removals+=1;continue
            cross=Z[eligible]@Z[out]
            removal_cost=alpha[out]**2/denominator
            gains=(alpha[eligible]+cross*alpha[out]/denominator)**2/(1+leverage[eligible]+cross*cross/denominator)-removal_cost
            order=np.lexsort((union[eligible],-gains));best=int(order[0]);inside=int(eligible[best]);gain=float(gains[best])
            offers.append((-gain,int(union[out]),int(union[inside]),position,inside))
        offers.sort();last_gain=(-offers[0][0]/obj) if offers and obj else 0.
        if not offers or -offers[0][0]<=1e-8*obj:
            stop='relative_gain';break
        accepted=False
        for negative_gain,out_id,in_id,position,inside in offers[:8]:
            if -negative_gain<=1e-8*obj:break
            trial=selected.copy();trial[position]=inside;newfac=factor(trial)
            if newfac is None or newfac[3]>=obj*(1-1e-8):rejected+=1;continue
            log.append({'out':out_id,'in':in_id,'removed_original':out_id in set(map(int,original)),
                        'objective_before':obj,'objective_after':newfac[3],
                        'predicted_relative_gain':-negative_gain/obj,'verified_relative_gain':(obj-newfac[3])/obj})
            selected=trial;fac=newfac;accepted=True;break
        if not accepted:stop='no_verified_top_offer';break
    support=union[selected]
    assert len(support)==len(saved) and len(np.unique(support))==len(saved)
    assert Counter(map(int,rows['source_plane'][support]))==Counter(map(int,rows['source_plane'][saved]))
    assert required_sector <= {(int(planes[i]),int(sectors[i])) for i in selected}
    assert required_direction <= {(int(planes[i]),str(directions[i])) for i in selected}
    if not whole:assert set(map(int,original))<=set(map(int,support))
    return support,{'initial_objective':initial,'final_objective':fac[3],'ratio':fac[3]/initial,'accepted_exchanges':len(log),'stop':stop,'last_predicted_relative_gain':last_gain,'rejected_verifications':rejected,'unstable_removals_skipped':unstable_removals,'swaps':log,'original_rows_absent_at_end':len(set(map(int,original))-set(map(int,support))),'candidate_count':len(union)}
