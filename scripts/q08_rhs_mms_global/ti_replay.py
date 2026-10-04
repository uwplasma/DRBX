"""Focused Ti diagnostic replay. Old references/banks are read-only.

Reference audit is portable. Global N execution requires a GPU and existing
full-domain Q08 banks. CPU bounded checks are explicitly test-only. No tracing,
row preparation, eigensolve, diffusion or complete RHS rerun is performed.
"""
from pathlib import Path
import argparse
import csv
import fcntl
import io
import json
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from scripts.q08_rhs_mms_global.campaign import read, write, sha, check_source, load, BASE_ID

SCI_ID = 'e8303a1a3b93580ebfabd2e0c68b50dc4e8f9f9d90057840e9f38fb7cf389652'
COUNTS = {32:25376, 48:86016, 64:202304}
LAST = {32:10, 48:15, 64:21}


def checked_path(root, name):
    p = root/name
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
        raise ValueError('input path escapes root')
    return p


def references(science, n):
    """Recover exact per-owner signed O/R, never infer N from reduced norms."""
    receipt = read(science/f'references_N{n}.json')
    if receipt['identity'] != SCI_ID or not receipt['passed'] or receipt['owners'] != COUNTS[n]:
        raise ValueError('wrong original scientific reference receipt')
    for rel, expected in receipt['files'].items():
        if sha(checked_path(science, rel)) != expected:
            raise ValueError('reference content changed: '+rel)
    owners = []; volume = []; radial = []; oo = []; rr = []
    algebra = 0.
    for index in range(receipt['chunks']):
        path = science/'references'/f'N{n}'/f'chunk_{index:06d}.npz'
        with np.load(path, allow_pickle=False) as z:
            # NpzFile does not cache decoded members; read large arrays once.
            O,R,ids,vol,rad=(z[k] for k in ('O','R','owners','volume','radial'))
            if O.shape != (2,22,len(ids),31) or R.shape != (22,len(ids),31):
                raise ValueError('reference shape')
            for a in (O,R):
                if not np.isfinite(a).all():
                    raise ValueError('nonfinite reference')
                error = a[...,29]-(a[...,30]-a[...,27])
                algebra = max(algebra, float(abs(error).max()))
                np.testing.assert_allclose(a[...,29], a[...,30]-a[...,27], atol=1e-10, rtol=1e-13)
            # This term does not depend on diffusion span. Establish exact reuse.
            np.testing.assert_array_equal(O[0,...,29],O[1,...,29])
            oo.append(-O[1,...,29]); rr.append(-R[...,29])
            owners.extend(ids); volume.extend(vol); radial.extend(rad)
    if not np.array_equal(owners, np.arange(COUNTS[n])):
        raise ValueError('incomplete or duplicate owners')
    result = dict(O=np.concatenate(oo,axis=1), R=np.concatenate(rr,axis=1),
                  owners=np.asarray(owners), volume=np.asarray(volume), radial=np.asarray(radial))
    if np.any(result['volume'] <= 0) or not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('invalid owner measure/reference')
    return result, dict(n=n, owners=len(owners), chunks=receipt['chunks'],
                       receipt_sha256=sha(science/f'references_N{n}.json'),
                       old_positive_sign_identity_max=algebra, diffusion_span_independent=True)


def scalar_action(plan, ti, boundary, *, kind):
    """Same raw scalar reconstruction/difference/multiplication as full RHS."""
    import jax.numpy as jnp
    from drbx.native.q_plan import _scalar
    from scripts.q08_extraction_global import common as c
    values = _scalar(plan, ti, boundary, boundary, (kind,),
                     jnp.take(ti, plan.donor, axis=-1), slots=(1,3)).value
    scale = plan.b_eta/(2*plan.eta_step)
    raw = -(c.MU*c.TAU*(scale*(values[...,0,:,1]-values[...,0,:,0])))
    return jnp.sum(jnp.take(raw,plan.owner_raw,axis=-1)*plan.owner_weight,axis=-1)


def scalar_boundary(bank, case):
    """Slice Ti before restoring dense arrays; reuse qualified BC producer."""
    from scripts.q08_extraction_global import common as c
    from boundary_values import compact_boundaries
    from boundary_cache import restore
    cls, arrays = compact_boundaries(bank,case,c)[0]
    sliced = []
    for shape, axis, template, rows in arrays:
        if axis != 1:
            raise ValueError('unexpected primitive boundary layout')
        sliced.append(((1,*shape[1:]),axis,template[2:3],rows[:,2:3]))
    return restore(bank.wall_index, [(cls,sliced)])[0]


def preflight(output, inputs, *, gpu=False):
    """Saved 21-owner CPU check against actual full six-field implementation."""
    import jax
    if gpu and (jax.default_backend() != 'gpu' or not jax.config.jax_enable_x64):
        raise RuntimeError('GPU preflight requires actual GPU float64')
    from scripts.q08_extraction_global import common as c
    from drbx.stencils.q_parallel import load_chunk
    from drbx.stencils.q_bank import build_q_bank
    from drbx.stencils.q_plan import lower_q_plan
    from drbx.native.q_plan import apply_q_plan
    from drbx.native.q_parallel import QBoundaryData
    from scripts.q08_rhs_mms_global.science import numerical, oracle
    inp = Path(inputs)/'bounded'
    records = []
    for n in COUNTS:
        bank = build_q_bank(*(load_chunk(inp/f'N{n}_h{d}.npz') for d in (16,32)))
        with np.load(inp/f'N{n}_inputs.npz') as z:
            data = {k:z[k].copy() for k in z.files}
        geom = {k:data[k] for k in ('magnetic_L','b_eta','eta_step','bmag')}
        plan = lower_q_plan(bank,diffusion_span=1/32,**geom)
        O = oracle(bank,geom)[1,...,29]
        for kind in ('D','N'):
            scalar = jax.jit(lambda p,x,b:scalar_action(p,x,b,kind=kind))
            full = jax.jit(lambda p,x,bi,bo,phi,pb:apply_q_plan(p,x,bi,bo,phi,pb,c.COEFF,
                kinds=(kind,)*6,phi_kind=kind,tau=c.TAU,mu=c.MU,characteristic_method='polynomial'))
            maximum = oracle_max = 0.
            for case in range(22):
                state = np.broadcast_to(np.r_[c.BASE,.2][:,None],(6,plan.n_owner)).copy()
                state[:,data['donor_ids']] = data['state'][case]
                phi = np.zeros(plan.n_owner); phi[data['donor_ids']] = data['phi'][case]
                bc = tuple(QBoundaryData(*(data[f'{name}_bc_{i}'][case] for i in range(4)))
                           for name in ('inner','outer','phi'))
                sb = scalar_boundary(bank,case)
                for a,b in zip(sb,bc[0]):
                    np.testing.assert_array_equal(a,b[2:3])
                narrow = np.asarray(scalar(plan,state[2:3],sb))
                expected = numerical(bank,full(plan,state,*bc[:2],phi,bc[2]))[:,29]
                np.testing.assert_allclose(narrow,expected,atol=1e-8,rtol=1e-11)
                maximum = max(maximum,float(abs(narrow-expected).max()))
                oracle_max = max(oracle_max,float(abs(narrow-O[case]).max()))
                if case == 0 and abs(narrow-O[case]).max() > 1e-7:
                    raise ValueError('constant gate')
            records.append(dict(n=n,kind=kind,cases=22,owners=len(bank.owners),
                                scalar_full_max_abs=maximum,NO_max=oracle_max))
        jax.clear_caches()
    result = dict(passed=True,test_only=not gpu,gpu=gpu,identity=check_source()[0],records=records,
                  device=str(jax.devices()[0]),
                  scope='bounded scalar diagnostic/full RHS equivalence; not global qualification')
    write(output/('bounded_gpu_preflight.json' if gpu else 'bounded_preflight.json'),result)
    return result


def reduce_scalar(N, ref, n):
    from scripts.q08_rhs_mms_global.science import masks
    # N: two BCs, case, owner. Retain owner actions for independent rescoring.
    O,R = ref['O'],ref['R']; w = ref['volume']; ids = ref['owners']
    error = np.stack((N-O,O[None]-R[None]+np.zeros_like(N),N-R),axis=-1)
    if not np.isfinite(error).all():
        raise ValueError('nonfinite scalar action')
    stats = dict(sum2=[],signed=[],maximum=[],max_owner=[],volume=[],count=[],reference_sum2=[])
    for mask in masks(ref['radial'],n,LAST[n]):
        e=error[:,:,mask,:]; weight=w[mask]
        if not mask.any():
            raise ValueError('missing global region')
        stats['sum2'].append(np.einsum('o,bcom->bcm',weight,e*e))
        stats['signed'].append(np.einsum('o,bcom->bcm',weight,e))
        stats['maximum'].append(abs(e).max(axis=2))
        stats['max_owner'].append(ids[mask][abs(e).argmax(axis=2)])
        stats['reference_sum2'].append(np.einsum('o,co->c',weight,R[:,mask]**2))
        stats['volume'].append(float(weight.sum()));stats['count'].append(int(mask.sum()))
    return {k:np.asarray(v) for k,v in stats.items()}


def run_grid(output, old, science, n, host_gib, identity):
    import jax
    from scripts.q08_extraction_global import gpu as oldgpu, common as c
    from drbx.stencils.q_plan import lower_q_plan
    from drbx.native.q_plan import stage_q_plan
    from scripts.q08_rhs_mms_global.resources import host_guard
    from scripts.q08_extraction_global.common import atomic_npz
    if jax.default_backend() != 'gpu' or not jax.config.jax_enable_x64:
        raise RuntimeError('full-grid Ti replay requires actual GPU float64; no CPU fallback')
    pre = read(output/'bounded_gpu_preflight.json')
    if pre.get('identity') != identity or pre.get('passed') is not True or pre.get('gpu') is not True:
        raise ValueError('same-source bounded GPU preflight required')
    ref, rr = references(science,n)
    paths = (old/'data'/f'N{n}'/'state.npy',old/'cpu'/f'N{n}',old/'inputs'/'plan.json')
    if not all(p.exists() for p in paths):
        raise FileNotFoundError('full immutable bank/state inputs missing; do not regenerate')
    provenance = read(science/'verification.json')['records']
    cpu = science/'provenance'/f'cpu_N{n}.json'
    if sha(cpu) != provenance[f'provenance/cpu_N{n}.json']:
        raise ValueError('original CPU receipt changed')
    pinned = read(cpu)
    if pinned.get('campaign_identity') != BASE_ID or not pinned.get('passed'):
        raise ValueError('original CPU receipt identity')
    for index, expected_hash in pinned['chunk_receipts'].items():
        if sha(paths[1]/f'chunk_{int(index):06d}'/'stats.json') != expected_hash:
            raise ValueError('bank receipt differs from original campaign')
    _, estimate = oldgpu.estimate_merge(old,n,BASE_ID,host_gib)
    guard = host_guard(estimate,host_gib)
    devices, inventory = oldgpu.device_inventory(False)
    oldgpu._device_guard(estimate,inventory)
    bank, geom, views, _ = oldgpu.merge_chunks(old,n,BASE_ID,host_gib)
    del views
    np.testing.assert_array_equal(bank.owners,ref['owners'])
    # State hash is pinned by the original data receipt, not by path alone.
    data_receipt = read(old/f'data_N{n}.json')
    if data_receipt.get('campaign_identity') != BASE_ID or not data_receipt.get('passed'):
        raise ValueError('state receipt identity')
    original = science/'provenance'/f'data_N{n}.json'
    if (sha(original) != provenance[f'provenance/data_N{n}.json'] or
            sha(old/f'data_N{n}.json') != sha(original)):
        raise ValueError('data receipt differs from original scientific provenance')
    expected = data_receipt['files']['state.npy']
    if sha(paths[0]) != expected:
        raise ValueError('state content changed')
    state = np.load(paths[0],mmap_mode='r')
    if state.shape != (22,6,COUNTS[n]):
        raise ValueError('state shape')
    plan = stage_q_plan(lower_q_plan(bank,diffusion_span=1/32,**geom),device=devices[0])
    calls = {kind:jax.jit(lambda p,x,b,kind=kind:scalar_action(p,x,b,kind=kind)) for kind in ('D','N')}
    record = output/f'N{n}.json'; payload = output/f'N{n}.npz'
    if record.exists():
        saved=read(record)
        if (saved.get('identity') != identity or saved.get('passed') is not True or
                saved.get('sha256') != sha(payload) or saved.get('reference') != rr or
                saved.get('state_sha256') != expected):
            raise ValueError('incompatible/corrupt completed checkpoint')
        return saved
    result=np.empty((2,22,COUNTS[n])); tick=time.perf_counter()
    for case in range(22):
        boundary=scalar_boundary(bank,case)
        for ki,kind in enumerate(('D','N')):
            dest=output/f'N{n}_{kind}_{case:02d}.npz'; receipt=dest.with_suffix('.json')
            if receipt.exists():
                r=read(receipt)
                if (r.get('identity')!=identity or r.get('sha256')!=sha(dest) or
                        (r.get('n'),r.get('kind'),r.get('case'))!=(n,kind,case)):
                    raise ValueError('scalar checkpoint changed')
                with np.load(dest) as z: result[ki,case]=z['N']
            else:
                N=np.asarray(calls[kind](plan,np.asarray(state[case,2:3]),boundary))
                if N.shape!=(COUNTS[n],) or not np.isfinite(N).all():
                    raise ValueError('invalid scalar result')
                if case==0 and abs(N-ref['O'][case]).max()>1e-7:
                    raise ValueError('constant gate')
                atomic_npz(dest,N=N)
                write(receipt,dict(identity=identity,n=n,kind=kind,case=case,sha256=sha(dest)))
                result[ki,case]=N
        print(f'Ti N{n}: {case+1}/22 states, D/N',flush=True)
    stats=reduce_scalar(result,ref,n)
    atomic_npz(payload,N=result,**ref,**{f'stat_{k}':v for k,v in stats.items()})
    r=dict(passed=True,identity=identity,n=n,owners=COUNTS[n],cases=22,scalar_BCs=['D','N'],
           original_kind_to_Ti=[kinds[2] for kinds,_ in c.KINDS],diffusion_span_independent=True,
           reference=rr,state_sha256=expected,cpu_receipt_sha256=sha(cpu),
           bank_identity=bank.identity,sha256=sha(payload),host_guard=guard,
           backend=jax.default_backend(),device=str(devices[0]),x64=bool(jax.config.jax_enable_x64),
           seconds=time.perf_counter()-tick,peak_rss_gib=oldgpu.peak_rss_gib())
    write(record,r)
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('reference-audit','bounded','bounded-gpu','gpu','analyze'))
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--science-run',type=Path)
    p.add_argument('--baseline-run',type=Path)
    p.add_argument('--baseline-source',type=Path,default=HERE.with_name('q08_extraction_global'))
    p.add_argument('--n',type=int,choices=tuple(COUNTS))
    p.add_argument('--host-gib',type=float)
    a=p.parse_args();out=a.run.resolve();out.mkdir(parents=True,exist_ok=True)
    for inp in (a.science_run,a.baseline_run):
        if inp and (out==inp.resolve() or out.is_relative_to(inp.resolve()) or inp.resolve().is_relative_to(out)):
            raise ValueError('new output directory required; inputs immutable')
    with (out/'replay.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity,_=check_source()
        binding=dict(identity=identity,original_science_identity=SCI_ID,
                     science_run=str(a.science_run.resolve()) if a.science_run else None,
                     baseline_run=str(a.baseline_run.resolve()) if a.baseline_run else None,
                     baseline_source=str(a.baseline_source.resolve()))
        binding_path=out/f'binding_{a.stage}.json'
        if binding_path.exists() and read(binding_path)!=binding:raise ValueError('binding changed')
        write(binding_path,binding)
        if a.stage=='reference-audit':
            if a.science_run is None:p.error('--science-run required')
            from scripts.q08_extraction_global.common import atomic_npz
            records=[]
            for n in COUNTS:
                ref,r=references(a.science_run,n);records.append(r)
                atomic_npz(out/f'references_N{n}.npz',**ref)
                print(f'verified corrected references N{n}: {r["owners"]} owners',flush=True)
            write(out/'reference_audit.json',dict(passed=True,identity=identity,records=records,
                numerical_replay_complete=False,scope='sign-corrected references only; no N action or corrected N-O/N-R scores'))
        elif a.stage in ('bounded','bounded-gpu'):
            gpu=a.stage=='bounded-gpu'
            inputs=(a.baseline_run/'inputs' if a.baseline_run else a.baseline_source/'inputs')
            load(a.baseline_source.resolve(),out,gpu=gpu)
            print(json.dumps(preflight(out,inputs,gpu=gpu)))
        elif a.stage=='gpu':
            if not a.baseline_run or not a.science_run or a.n is None or a.host_gib is None:
                p.error('gpu requires --baseline-run --science-run --n --host-gib')
            load(a.baseline_source.resolve(),out,gpu=True)
            print(json.dumps(run_grid(out,a.baseline_run.resolve(),a.science_run.resolve(),a.n,a.host_gib,identity)))
        else:
            analyze(out,identity)


def analyze(out,identity):
    """Independently recompute all reductions from retained scalar owner actions."""
    from scripts.q08_rhs_mms_global.science import REGIONS,METRICS
    from scripts.q08_extraction_global import common as c
    all_stats=[];files={}
    for n in COUNTS:
        r=read(out/f'N{n}.json');path=out/f'N{n}.npz'
        if (r.get('identity')!=identity or not r.get('passed') or r.get('backend')!='gpu'
                or r.get('sha256')!=sha(path)):
            raise ValueError('missing/corrupt GPU replay')
        with np.load(path) as z:
            ref={k:z[k] for k in ('O','R','owners','volume','radial')}
            if not np.array_equal(ref['owners'],np.arange(COUNTS[n])) or z['N'].shape!=(2,22,COUNTS[n]):
                raise ValueError('owner/action coverage')
            s=reduce_scalar(z['N'],ref,n)
            for k,v in s.items():np.testing.assert_array_equal(v,z['stat_'+k])
            if abs(z['N'][:,0]-z['O'][0]).max()>1e-7:raise ValueError('constant gate')
        all_stats.append(s)
        for f in (path,out/f'N{n}.json'):files[f.name]=sha(f)
    # resolution, region, scalar BC, case, metric
    rms=np.stack([np.sqrt(s['sum2']/s['volume'][:,None,None,None]) for s in all_stats])
    good=(rms[:-1]>1e-30)&(rms[1:]>1e-30)
    order=np.full_like(rms[:-1],np.nan)
    np.log(np.divide(rms[:-1],rms[1:],out=np.ones_like(rms[:-1]),where=good),out=order,where=good)
    order/=np.log(np.array([1.5,4/3]))[:,None,None,None,None]
    stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(['BC','case','region','metric','rms32','rms48','rms64','p32_48','p48_64'])
    for ri,region in enumerate(REGIONS):
        for bi,bc in enumerate(('D','N')):
            for ci,case in enumerate(c.DESIGNS):
                for mi,metric in enumerate(METRICS):
                    writer.writerow([bc,case['name'],region,metric,*rms[:,ri,bi,ci,mi],
                                     *[x if np.isfinite(x) else '' for x in order[:,ri,bi,ci,mi]]])
    (out/'orders.csv').write_bytes(stream.getvalue().encode())
    files['orders.csv']=sha(out/'orders.csv')
    write(out/'analysis.json',dict(identity=identity,term='electron_ti_compensation',
        h_denominator=32,eta_planes=5,scope='focused scalar global replay; complete RHS unchanged',
        scalar_BCs=['D','N'],original_kind_to_Ti=['D','N','D','N'],
        diffusion_span_independent=True,rms=rms.tolist(),orders=np.where(np.isfinite(order),order,None).tolist(),
        order_axes=['interval','region','scalar BC','case','metric'],rms_axes=['resolution','region','scalar BC','case','metric']))
    files['analysis.json']=sha(out/'analysis.json')
    write(out/'completion.json',dict(passed=True,identity=identity,files=files,
        scientific_order_accepted=False,production_promoted=False))


if __name__=='__main__':main()
