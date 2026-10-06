"""Actual HSX core/transition/bulk/wall traces, row/reference and CPU/GPU replay."""
from pathlib import Path
import time
import numpy as np
from . import common as c
from .geometry import setup
from .prepare import trace_points,build


def owners_for(t,last):
    n=t.n
    raw=np.array([0,n*n+2*n, max(1,last-1)*n*n+3*n+1,
        (last+1)*n*n+4*n+2,(n//2)*n*n+5*n+3,(n-2)*n*n+6*n+4,(n-1)*n*n+7*n+5])
    return np.unique(t.ro[raw])


def state(t):
    from scripts.q09_evolved_mms.mms import smooth_fields
    x=np.zeros((len(t.vol),6));p=np.zeros(len(t.vol))
    for start in range(0,len(t.pts),2048):
        sl=slice(start,start+2048);v,_,phi,_=map(np.asarray,smooth_fields(t.pts[sl],0.))
        np.add.at(x,t.ro[sl],v*t.rv[sl,None]);np.add.at(p,t.ro[sl],phi*t.rv[sl])
    return (x/t.vol[:,None]).T,p/t.vol


def action(bank,geom,x,phi):
    import jax
    from drbx.stencils.q_plan import lower_q_plan
    from drbx.native.q_plan import stage_q_plan,apply_q_plan
    from scripts.q08_extraction_global import common as cat
    from scripts.q09_evolved_mms.campaign import KINDS
    plan=stage_q_plan(lower_q_plan(bank,diffusion_span=1/32,**geom));bc=tuple(jax.tree.map(jax.numpy.asarray,b) for b in cat.boundaries(bank,1))
    ans=[]
    for kinds,pk in KINDS:
        out=apply_q_plan(plan,x,bc[0],bc[1],phi,bc[2],cat.COEFF,kinds=tuple(kinds),phi_kind=pk,tau=cat.TAU,mu=cat.MU,characteristic_method='polynomial')
        if not np.asarray(out.inputs_valid).all() or not np.asarray(out.eigensystem_admissible).all():raise ValueError('invalid bounded RHS')
        ans.append(np.stack([np.asarray(z) for z in out[:4]]))
    result=np.stack(ans)
    if not np.isfinite(result).all(): raise ValueError('nonfinite bounded action')
    return result


def cpu(run,ident):
    from drbx.stencils.q_parallel import raw_members
    from drbx.stencils.q_artifact import save_q_bank,load_q_bank
    from scripts.q08_extraction_global.common import atomic_npz
    from scripts.q09_evolved_mms.mms import ContinuumReference
    from scripts.q09_evolved_mms.campaign import resource_measurement
    import jax
    if jax.default_backend()!='cpu':raise ValueError('CPU preflight backend')
    tick=time.perf_counter();records=[];files={}
    for n in c.GRIDS:
        e=setup(run,n);t=e['t'];last=c.read(Path(run)/'inputs/plan.json')[str(n)]['last_aggregate']
        owners=owners_for(t,last);raw=raw_members(t,owners)
        ends,trace=trace_points(e,t.pts[raw],capacity=256)
        twice,_=trace_points(e,t.pts[raw],capacity=256,steps=128)
        rk=float(abs(ends-twice).max())
        if rk>1e-9: raise ValueError(('RK64/128 reach disagreement',rk))
        # Exercise the lighter worker initialization as well: it does not load
        # raw MAKEGRID splines, only the identical map and filtered table.
        host=setup(run,n,tracing=False)
        for a,b in zip(e['geom'](t.pts[raw]),host['geom'](t.pts[raw])):
            np.testing.assert_array_equal(a,b)
        e=host
        bank,geom,ref=build(e,owners,ends,ident,c.digest(dict(identity=ident,n=n,role='bounded filtered traces')))
        independent=ContinuumReference.prepare(bank,geom,e['geom'])
        replay=float(abs(np.asarray(ref.rhs(0.,'complete'))-np.asarray(independent.rhs(0.,'complete'))).max())
        np.testing.assert_allclose(ref.rhs(0.,'complete'),independent.rhs(0.,'complete'),rtol=1e-11,atol=1e-8)
        x,phi=state(t);a=action(bank,geom,x,phi)
        folder=Path(run)/'bounded'/f'N{n}';folder.mkdir(parents=True,exist_ok=True)
        save_q_bank(bank,folder/'bank.npz');load_q_bank(folder/'bank.npz',expected_identity=bank.identity)
        atomic_npz(folder/'data.npz',points=t.pts[raw],ends=ends,action=a,state=x,phi=phi,**geom)
        for path in (folder/'bank.npz',folder/'data.npz'):files[str(path.relative_to(run))]=c.sha(path)
        records.append(dict(n=n,owners=owners.tolist(),raw=len(raw),trace=trace,rk64_128_max=rk,reference_replay_max=replay,reference_steps=ref.diagnostics))
        print(f'CPU preflight N{n}: owners={len(owners)} raw={len(raw)} reference={replay:.3g}',flush=True)
        del e,bank,x,phi,a; jax.clear_caches()
    import concurrent.futures,multiprocessing
    from .prepare import bounded_worker_probe
    with concurrent.futures.ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        probe=pool.submit(bounded_worker_probe,str(run),ident).result()
    folder=Path(run)/'worker_preflight/cpu/N32/chunk_000000'
    for name in ('stats.json',*probe['files']):files[str((folder/name).relative_to(run))]=c.sha(folder/name)
    r=dict(passed=True,identity=ident,records=records,worker_probe=probe,seconds=time.perf_counter()-tick,resources=resource_measurement(),files=files)
    c.write(Path(run)/'preflight_cpu.json',r);return r


def gpu(run,ident):
    import jax
    from drbx.stencils.q_artifact import load_q_bank
    from scripts.q09_evolved_mms.campaign import hardware
    c.require(run,'preflight_cpu',ident);inventory=hardware();records=[]
    for n in c.GRIDS:
        e=setup(run,n);folder=Path(run)/'bounded'/f'N{n}'
        bank=load_q_bank(folder/'bank.npz')
        with np.load(folder/'data.npz') as z: d={k:z[k].copy() for k in z.files}
        # Replay the production trace batch shape, including its padded tail.
        ends,_=trace_points(e,d['points'],capacity=16384)
        np.testing.assert_allclose(ends,d['ends'],rtol=0,atol=2e-10)
        geom={k:d[k] for k in ('magnetic_L','b_eta','eta_step','bmag')}
        actual=action(bank,geom,d['state'],d['phi'])
        np.testing.assert_allclose(actual,d['action'],rtol=1e-11,atol=1e-8)
        records.append(dict(n=n,trace_max=float(abs(ends-d['ends']).max()),action_max=float(abs(actual-d['action']).max())))
        del e,bank,d,actual; jax.clear_caches()
    c.write(Path(run)/'preflight_gpu.json',dict(passed=True,identity=ident,hardware=inventory,records=records,files={'preflight_cpu.json':c.sha(Path(run)/'preflight_cpu.json')}))
