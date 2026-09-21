"""Plane-batched exchange scoring with unchanged SVD verification.

Near ranking ties or the stopping threshold, replay the oracle scalar scoring.
No change to candidate pools, objective, coverage, budget or accepted-swap cap.
"""
from numerics import ex
import numpy as np
from collections import Counter

def selector(rows,original,saved,pools,center,scale,target,whole,max_exchanges=32,*,record_swaps=True):
 union=np.unique(np.concatenate([saved,*[p['rows'] for p in pools.values()]]))
 A,distance=ex.run.compact.row_moments(rows,union,center,scale,5)
 weight=ex.run.compact.distance_weight(distance);vectors=A[:,ex.run.prior.F_M3]*weight[:,None];t=target[ex.run.prior.F_M3]
 source,endpoint=ex.an.points(rows,union,center,scale);sectors=ex.directional.sector(.5*(source+endpoint))
 directions=rows['direction'][union].astype(str);planes=rows['source_plane'][union]
 selected=np.searchsorted(union,saved);original_set=set(map(int,original));is_original=np.isin(union,original)
 plane_values,plane_code=np.unique(planes,return_inverse=True);_,direction_code=np.unique(directions,return_inverse=True)
 nsector=int(np.max(sectors))+1;ndir=int(np.max(direction_code))+1
 sector_label=plane_code*nsector+sectors;direction_label=plane_code*ndir+direction_code
 required_sector=np.bincount(sector_label[selected],minlength=len(plane_values)*nsector)>0
 required_direction=np.bincount(direction_label[selected],minlength=len(plane_values)*ndir)>0
 available_mask=np.ones(len(union),bool);available_mask[selected]=False
 plane_members=[np.flatnonzero(plane_code==p) for p in range(len(plane_values))]
 def factor(indices):
  U,s,Vt=np.linalg.svd(vectors[indices].T,full_matrices=False)
  tol=np.finfo(float).eps*max(vectors[indices].T.shape)*s[0]
  if np.count_nonzero(s>tol)!=len(t):return None
  beta=(U.T@t)/s;return U,s,beta,float(beta@beta)
 def scalar_offers(Z,alpha,leverage,available,sc,dc):
  offers=[];unstable=0
  for position,out in enumerate(selected):
   if not whole and is_original[out]:continue
   eligible=available[planes[available]==planes[out]]
   if required_sector[sector_label[out]] and sc[sector_label[out]]==1:eligible=eligible[sectors[eligible]==sectors[out]]
   if required_direction[direction_label[out]] and dc[direction_label[out]]==1:eligible=eligible[directions[eligible]==directions[out]]
   if not len(eligible):continue
   denominator=1.-leverage[out]
   if denominator<=1e-10:unstable+=1;continue
   cross=Z[eligible]@Z[out];removal_cost=alpha[out]**2/denominator
   gains=(alpha[eligible]+cross*alpha[out]/denominator)**2/(1+leverage[eligible]+cross*cross/denominator)-removal_cost
   best=int(np.lexsort((union[eligible],-gains))[0]);inside=int(eligible[best])
   offers.append((-float(gains[best]),int(union[out]),int(union[inside]),position,inside))
  offers.sort();return offers,unstable
 fac=factor(selected)
 if fac is None:raise RuntimeError('initial cubic rank deficient')
 initial=fac[3];log=[];count=0;rejected=0;unstable_total=0;fallbacks=0;stop='cap';last_gain=None
 for iteration in range(max_exchanges):
  U,s,beta,obj=fac;Z=(vectors@U)/s[None,:];alpha=Z@beta;leverage=np.sum(Z*Z,axis=1)
  sc=np.bincount(sector_label[selected],minlength=len(required_sector));dc=np.bincount(direction_label[selected],minlength=len(required_direction))
  offers=[];ambiguous_positions=set();unstable=0;guard=1e-10*abs(obj)
  for p,members in enumerate(plane_members):
   positions=np.flatnonzero(plane_code[selected]==p)
   if not whole:positions=positions[~is_original[selected[positions]]]
   inside=members[available_mask[members]]
   if not len(positions) or not len(inside):continue
   outside=selected[positions]
   only_s=required_sector[sector_label[outside]] & (sc[sector_label[outside]]==1)
   only_d=required_direction[direction_label[outside]] & (dc[direction_label[outside]]==1)
   eligible=(~only_s[:,None] | (sectors[outside,None]==sectors[inside][None,:])) & (~only_d[:,None] | (direction_code[outside,None]==direction_code[inside][None,:]))
   denominator=1.-leverage[outside];valid=np.any(eligible,axis=1)&(denominator>1e-10)
   unstable+=int(np.sum(np.any(eligible,axis=1)&(denominator<=1e-10)))
   if not np.any(valid):continue
   outside=outside[valid];positions=positions[valid];eligible=eligible[valid];denominator=denominator[valid,None]
   cross=Z[outside]@Z[inside].T;removal=alpha[outside,None]**2/denominator
   gains=(alpha[inside][None,:]+cross*alpha[outside,None]/denominator)**2/(1+leverage[inside][None,:]+cross*cross/denominator)-removal
   gains[~eligible]=-np.inf
   winners=np.argmax(gains,axis=1);best=gains[np.arange(len(outside)),winners]
   if len(inside)>1:
    second=np.partition(gains,-2,axis=1)[:,-2]
    ambiguous_positions.update(map(int,positions[(best-second)<=guard]))
   for position,out,in_idx,gain in zip(positions,outside,inside[winners],best):
    offers.append((-float(gain),int(union[out]),int(union[in_idx]),int(position),int(in_idx)))
  offers.sort()
  # SVD state is unchanged from the oracle. Recompute scalar candidate scores
  # if batched dot-product rounding could matter near any attempted top offer.
  positive=[x for x in offers[:8] if -x[0]>1e-8*obj]
  needs_fallback=any(x[3] in ambiguous_positions for x in positive)
  needs_fallback |= any(abs(offers[j][0]-offers[j+1][0])<=guard for j in range(min(8,len(offers)-1)) if -offers[j][0]>1e-8*obj)
  needs_fallback |= bool(offers and abs(-offers[0][0]-1e-8*obj)<=guard)
  if needs_fallback:
   offers,unstable=scalar_offers(Z,alpha,leverage,np.flatnonzero(available_mask),sc,dc);fallbacks+=1
  unstable_total+=unstable;last_gain=(-offers[0][0]/obj) if offers and obj else 0.
  if not offers or -offers[0][0]<=1e-8*obj:stop='relative_gain';break
  accepted=False
  for negative_gain,out_id,in_id,position,inside in offers[:8]:
   if -negative_gain<=1e-8*obj:break
   trial=selected.copy();trial[position]=inside;newfac=factor(trial)
   if newfac is None or newfac[3]>=obj*(1-1e-8):rejected+=1;continue
   if record_swaps:log.append({'out':out_id,'in':in_id,'removed_original':out_id in original_set,'objective_before':obj,'objective_after':newfac[3],'predicted_relative_gain':-negative_gain/obj,'verified_relative_gain':(obj-newfac[3])/obj})
   available_mask[selected[position]]=True;available_mask[inside]=False;selected=trial;fac=newfac;count+=1;accepted=True;break
  if not accepted:stop='no_verified_top_offer';break
 support=union[selected]
 assert len(support)==len(saved) and len(np.unique(support))==len(saved)
 assert Counter(map(int,rows['source_plane'][support]))==Counter(map(int,rows['source_plane'][saved]))
 assert np.all(np.bincount(sector_label[selected],minlength=len(required_sector))[required_sector]>0)
 assert np.all(np.bincount(direction_label[selected],minlength=len(required_direction))[required_direction]>0)
 if not whole:assert original_set<=set(map(int,support))
 return support,{'initial_objective':initial,'final_objective':fac[3],'ratio':fac[3]/initial,'accepted_exchanges':count,'stop':stop,'last_predicted_relative_gain':last_gain,'rejected_verifications':rejected,'unstable_removals_skipped':unstable_total,'swaps':log,'original_rows_absent_at_end':len(original_set-set(map(int,support))),'candidate_count':len(union),'scalar_ranking_fallbacks':fallbacks}
