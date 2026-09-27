"""Streaming complete-owner signed Q reduction; scientific gate is reported, never tuned."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
from .reuse import EXPECTED,validate_reuse
from . import numerical_runner as nr

BANDS=((0.,.25),(.25,.5),(.5,.75),(.75,.875),(.875,1.00000001))
CHANNELS=('N','E','P','R','R_half','N-E','E-P','P-R','N-R','R_step')
def topology(input_root,n):
    cfg=json.loads(nr.CONFIG.read_text());p=Path(input_root)/cfg['geometry']/f'{n}x{n}x{n}'
    with np.load(p/'rlp_topology.npz') as z:
        ids=np.flatnonzero(z['is_active_owner'].ravel());lookup=np.full(n**3,-1,int);lookup[ids]=np.arange(len(ids));labels=lookup[z['aggregate_id'].ravel()];rawvol=z['raw_volume'].ravel().copy()
    if len(ids)!=EXPECTED[n][0] or np.any(labels<0):raise RuntimeError('canonical topology owner mismatch')
    count=np.bincount(labels,minlength=len(ids));vol=np.bincount(labels,weights=rawvol,minlength=len(ids));radial=np.arange(n**3)//(n*n)
    rmin=np.full(len(ids),n,int);rmax=np.full(len(ids),-1,int);np.minimum.at(rmin,labels,radial);np.maximum.at(rmax,labels,radial)
    with np.load(p/'base_geometry.npz') as z:uc=z['grid.x.centers'].copy()
    region=np.full(len(ids),'ordinary',dtype='<U24')
    region[count>1]='aggregate_transition';region[(count==1)&(rmax>=n-7)]='wall';region[rmin==1]='first_ring';region[rmin==0]='axis_core'
    band=np.full(len(ids),'mixed_band',dtype='<U24')
    for lo,hi in BANDS:band[(uc[rmin]>=lo)&(uc[rmax]<hi)]=f'[{lo},{hi})'
    groups={'global':np.ones(len(ids),bool)}
    for name in sorted(set(region)):groups['region:'+name]=region==name
    for name in sorted(set(band)):groups['radial:'+name]=band==name
    return count,vol,groups
def update(acc,v,d,reference):
    # Signed components are added first, then squared with frozen physical volume.
    s=np.sum(d,axis=-1);r=np.sum(reference,axis=-1)
    acc['volume']+=float(np.sum(v));acc['count']+=len(v)
    acc['squared']+=float(np.dot(v,abs(s)**2));acc['reference_squared']+=float(np.dot(v,abs(r)**2))
    acc['max']=max(acc['max'],float(np.max(abs(s))))
    acc['direction_squared']+=np.einsum('n,ni->i',v,abs(d)**2).real
    acc['signed_sum']+=complex(np.dot(v,s))
def empty():return dict(volume=0.,count=0,squared=0.,reference_squared=0.,max=0.,direction_squared=np.zeros(3),signed_sum=0j)
def result(a):
    if a['volume']==0:return None
    err=math.sqrt(a['squared']/a['volume']);ref=math.sqrt(a['reference_squared']/a['volume'])
    return dict(owner_count=a['count'],physical_volume=a['volume'],absolute_rms=err,relative_rms=err/ref if ref>1e-14 else None,reference_rms=ref,maximum=a['max'],direction_rms=np.sqrt(a['direction_squared']/a['volume']).tolist(),signed_volume_mean=[(a['signed_sum']/a['volume']).real,(a['signed_sum']/a['volume']).imag])
def reduce_resolution(campaign,input_root,screen,n):
    from .campaign import validate_outputs,validate_dispatch
    root=Path(campaign);dispatch=validate_dispatch(root);validation=validate_outputs(root,'global',n,require_complete=True)
    out=root/'global';design=json.loads((out/'design.json').read_text());exact_fields=json.loads((Path(screen)/'design.json').read_text())['method']['fields']
    if tuple(exact_fields)!=nr.exact.FIELDS:raise RuntimeError('primary exact field order differs')
    count,volume,groups=topology(input_root,n);cases=None;acc={};expected_owner=0
    for owners in dispatch['stages']['global'][str(n)]:
        file=nr.chunk_path(out,n,owners);receipt=json.loads(file.with_suffix('.json').read_text())
        if cases is None:cases=[tuple(x) for x in receipt['cases']]
        if [tuple(x) for x in receipt['cases']]!=cases:raise RuntimeError('field/BC case order differs')
        with np.load(file) as z:
            ids=z['owners'];v=z['volume'];N=z['N'];E=z['E'];P=z['P'];R=z['R'];Rh=z['R_half'];raw=z['raw_count']
            if not np.array_equal(ids,owners) or ids[0]!=expected_owner or not np.array_equal(raw,count[ids]) or not np.allclose(v,volume[ids],rtol=3e-13,atol=1e-15):raise RuntimeError('owner/raw/volume identity differs')
            expected_owner=int(ids[-1])+1
            if N.shape!=(len(ids),len(cases),2,3) or E.shape!=N.shape or P.shape!=(len(ids),len(exact_fields),2,3) or R.shape!=(len(ids),len(exact_fields),3) or Rh.shape!=R.shape:raise RuntimeError('signed channel shape differs')
            if any(not np.all(np.isfinite(x)) for x in (N,E,P,R,Rh)):raise RuntimeError('nonfinite signed channel')
            if np.max(abs(P[:,exact_fields.index('constant')]))>1e-12 or np.max(abs(R[:,exact_fields.index('constant')]))>1e-12:raise RuntimeError('constant exact action differs')
            if not np.array_equal(P[:,exact_fields.index('homogeneous_D')],P[:,exact_fields.index('simple_zero_N')]) or not np.array_equal(R[:,exact_fields.index('homogeneous_D')],R[:,exact_fields.index('simple_zero_N')]):raise RuntimeError('exact affine identity differs')
            for ci,(field,bc) in enumerate(cases):
                primary=field in exact_fields;fi=exact_fields.index(field) if primary else None
                for span in range(2):
                    nv=N[:,ci,span];ev=E[:,ci,span];pv=P[:,fi,span] if primary else None;rv=R[:,fi] if primary else None;rh=Rh[:,fi] if primary else None
                    channels={'N':nv,'E':ev,'N-E':nv-ev}
                    if primary:channels.update(P=pv,R=rv,R_half=rh,**{'E-P':ev-pv,'P-R':pv-rv,'N-R':nv-rv,'R_step':rv-rh})
                    reference=rv if primary else ev
                    for group,mask in groups.items():
                        local=mask[ids]
                        if not np.any(local):continue
                        for channel,d in channels.items():
                            key=(group,field,bc,span,channel)
                            if key not in acc:acc[key]=empty()
                            update(acc[key],v[local],d[local],reference[local])
    if expected_owner!=EXPECTED[n][0]:raise RuntimeError('missing final owner')
    structured={}
    for (group,field,bc,span,channel),a in acc.items():
        structured.setdefault(group,{}).setdefault(f'{field}:{bc}',{}).setdefault('h/8' if span==0 else 'h/4',{})[channel]=result(a)
    summary=dict(schema='q-numerical-reduction-v1',N=n,execution_complete=True,owners=validation['owners'],raw_targets=validation['raw_targets'],physical_volume=float(np.sum(volume)),field_case_order=[list(x) for x in cases],primary_exact_fields=exact_fields,stress_reference_status='P/R unavailable for lambda0.5 and lambda0.25 stress fields; no N-R or primary gate reported for them',groups=structured)
    path=root/f'summary_N{n}.json';nr.write(path,summary);return summary
def order(a,b,ratio):return math.log(a/b)/math.log(ratio) if a and b and a>0 and b>0 else None
def reduce_campaign(campaign,input_root,screen):
    root=Path(campaign);reuse=validate_reuse(screen,root/'global/exact_reuse_manifest.json')
    data={str(n):reduce_resolution(root,input_root,screen,n) for n in EXPECTED}
    cases=data['32']['field_case_order'];orders={}
    for field,bc in cases:
        if field not in data['32']['primary_exact_fields']:continue
        key=f'{field}:{bc}'
        for group in data['32']['groups']:
            if any(group not in data[str(n)]['groups'] for n in EXPECTED):continue
            for span in ('h/8','h/4'):
                channel='N-R';values=[data[str(n)]['groups'][group][key][span][channel]['absolute_rms'] for n in EXPECTED]
                orders.setdefault(group,{}).setdefault(key,{})[span]=dict(absolute_rms=values,orders=[order(values[0],values[1],48/32),order(values[1],values[2],64/48)])
    gates={key:dict(orders=orders['global'][key]['h/8']['orders'],passes=all(x is not None and x>=1.8 for x in orders['global'][key]['h/8']['orders'])) for key in orders.get('global',{}) if not key.startswith('constant:')}
    final=dict(schema='q-numerical-global-summary-v1',execution_complete=True,science_gate='N-R h/8 order >=1.8 on both intervals for nonconstant primary cases',gate_results=gates,constant_control='zero reference; convergence order undefined and excluded from order gate',all_primary_gates_pass=bool(gates) and all(x['passes'] for x in gates.values()),orders=orders,exact_reuse=reuse,stress_reference_status=data['32']['stress_reference_status'])
    nr.write(root/'global_summary.json',final);return dict(execution_complete=True,global_summary=str(root/'global_summary.json'),all_primary_gates_pass=final['all_primary_gates_pass'])
def selftest():
    a=empty();update(a,np.array([1.,3.]),np.array([[2.,-2.,0.],[1.,2.,-3.]]),np.ones((2,3)))
    r=result(a)
    assert r['absolute_rms']==0. and r['direction_rms'][0]>0
    assert order(4.,2.,2.)==1.
    return True
if __name__=='__main__':print(selftest())
