"""Replay the frozen balanced candidate against local research evidence.

Evidence files are read only by this optional local check, never by run_all.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_ENABLE_X64']='true'
import argparse,time
from pathlib import Path
import numpy as np
from . import model,storage as io,primitives as r
from .support import Hybrid,BalancedHybrid
from .fields import FIELDS,BASELINE_FIELDS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();start=time.monotonic();evidence=a.workspace/'work/q_fci_balanced_inner_20260929'
    old=a.workspace/'work/q-fci-layered-ad54c332-20260929T144546Z'
    plan=io.read(old/'plan.json');receipt=io.read(evidence/'receipt.json')
    assert tuple(receipt['fields'])==FIELDS
    results=[]
    for n in (32,48,64):
        ctx,t=model.context(n,a.workspace);state=model.states(t)
        with np.load(evidence/f'actions_N{n}.npz') as z:
            owners=z['owners'];raw=z['raw'];expected=z['error']
        where={int(rr):j for j,rr in enumerate(raw)};ends=np.full((len(raw),4,3),np.nan)
        for bi,oo in enumerate(plan[str(n)]['global_']):
            if not set(oo).intersection(owners):continue
            with np.load(old/'global'/f'N{n}'/f'trace_{bi:06d}.npz') as z:
                for pos,rr in enumerate(z['raw']):
                    if int(rr) in where:ends[where[int(rr)]]=z['ends'][pos]
        assert np.isfinite(ends).all()
        slots,b,L,O,R,Rh=model.slots_and_action(ctx,t,raw,ends)
        errors=np.empty((2,len(raw),2,len(FIELDS)),complex);counts=set();max_repro=0.
        for mi,h in enumerate((Hybrid(t),BalancedHybrid(t))):
            for j,rr in enumerate(raw):
                ijk=np.array(np.unravel_index(rr,(n,)*3))
                ids,V,D,meta=h.rows(ijk,slots[j])
                errors[mi,j]=L[j]@np.einsum('si,sid->sd',b[j],D)@state[ids]-O[j]
                max_repro=max(max_repro,meta['reproduction'])
                if mi==1 and ijk[0]<=h.last:
                    counts.update(len(p['donors']) for p in meta['planes'])
        projected=np.array([[np.einsum('r,raf->af',t.rv[r.members(t,o)]/t.vol[o],errors[mi,[where[int(rr)] for rr in r.members(t,o)]]) for o in owners] for mi in (0,1)])
        diff=np.max(abs(projected-expected),axis=(1,2,3))
        assert np.all(diff<2e-9) and counts=={28}
        z=dict(n=n,owners=len(owners),raw=len(raw),fields=len(FIELDS),compact_error_max=float(diff[0]),balanced_error_max=float(diff[1]),plane_donor_counts=sorted(counts),max_reproduction=max_repro)
        results.append(z);print(z,flush=True)
    io.write(a.output,dict(pass_=True,comparison='N-O signed complete-owner replay of both rules',runs=results,wall_seconds=time.monotonic()-start,gpu_execution_tested=False))

if __name__=='__main__':main()
