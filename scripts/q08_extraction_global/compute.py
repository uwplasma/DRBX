"""CPU-only frozen legacy versus extracted Q08 complete-owner replay worker."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.update(JAX_ENABLE_X64='true',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
from pathlib import Path
import fcntl
import importlib
import json
import resource
import sys
import tempfile
import time
from dataclasses import replace,fields
import hashlib
import numpy as np
from . import common as c

ENV=None


def _span_runtime(base,diffusion_q,span):
    """Reuse span-independent audits with a fully refreshed diffusion identity."""
    from drbx.stencils.q_parallel import sha256_array
    diffusion=diffusion_q.runtime_view()
    metadata={**base.metadata,'diffusion_span':span,
              'diffusion_arrays':{f.name:sha256_array(getattr(diffusion,f.name))
                  for f in fields(diffusion) if f.name!='metadata'}}
    metadata.pop('identity',None)
    metadata['identity']=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    return replace(base,metadata=metadata,diffusion=diffusion)


def _rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=='darwin' else 1024**2)


def _atomic_npy(path,array):
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.save(stream,array,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def _data(n,t,rundir,identity):
    root=Path(rundir)/'data'/f'N{n}';root.mkdir(parents=True,exist_ok=True)
    receipt=root/'data.json'
    paths={key:root/f'{key}.npy' for key in ('state','phi','raw_to_owner')}
    with (root/'.writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():
            record=json.loads(receipt.read_text())
            if record.get('identity')!=identity or any(c.sha(paths[k])!=h for k,h in record['files'].items()):
                raise ValueError('Q08 owner data identity/content mismatch')
        else:
            five=np.zeros((c.NF,5,len(t.vol)));phi=np.zeros((c.NF,len(t.vol)))
            for start in range(0,len(t.pts),4096):
                sl=slice(start,start+4096);val,_=c.fields(t.pts[sl]);p,_=c.phi_fields(t.pts[sl])
                for ci in range(c.NF):
                    for fi in range(5):np.add.at(five[ci,fi],t.ro[sl],t.rv[sl]*val[:,ci,fi])
                    np.add.at(phi[ci],t.ro[sl],t.rv[sl]*p[:,ci])
            five/=t.vol;phi/=t.vol
            state=np.concatenate((five,(c.WOFF+np.einsum('cfo,f->co',five,c.WC))[:,None]),axis=1)
            if not np.isfinite(state).all() or not np.isfinite(phi).all() or not np.all(state[:,:3]>0):
                raise ValueError('Q08 invalid owner catalogue state')
            for k,a in dict(state=state,phi=phi,raw_to_owner=t.ro).items():_atomic_npy(paths[k],a)
            c.atomic_json(receipt,dict(identity=identity,n=n,files={k:c.sha(p) for k,p in paths.items()},
                shapes=dict(state=list(state.shape),phi=list(phi.shape),raw_to_owner=list(t.ro.shape))))
    arrays={k:np.load(p,mmap_mode='r',allow_pickle=False) for k,p in paths.items()}
    if (arrays['state'].shape!=(c.NF,6,len(t.vol)) or arrays['phi'].shape!=(c.NF,len(t.vol)) or
        not np.array_equal(arrays['raw_to_owner'],t.ro)):
        raise ValueError('Q08 owner data coverage/shape mismatch')
    return arrays


def initialize(n,canonical_root,inputdir,rundir,with_data=True):
    """Initializer for bounded spawn CPU workers; geometry never retraces inputs."""
    global ENV
    import jax
    if jax.default_backend()!='cpu' or not jax.config.jax_enable_x64:
        raise ValueError('Q08 compute worker requires CPU/x64')
    canonical_root,inputdir,rundir=map(lambda p:Path(p).resolve(),(canonical_root,inputdir,rundir))
    model=c.load_geometry_model(inputdir);ctx,t=model.context(n,canonical_root)
    geom=lambda p:model.geom(ctx,np.asarray(p))
    jac=lambda p:ctx['evaluator']._position_and_jacobian(p)[1]
    config=json.loads((inputdir/'geometry_model/inputs.json').read_text())
    geometry_files=[canonical_root/config['metric_cache'],canonical_root/config['makegrid']]
    geometry_files += [canonical_root/config['geometry']/f'{n}x{n}x{n}'/key for key in ('base_geometry.npz','rlp_topology.npz')]
    geometry_content={str(p.relative_to(canonical_root)):c.sha(p) for p in geometry_files}
    model_content={str(p.relative_to(inputdir)):c.sha(p) for p in sorted((inputdir/'geometry_model').rglob('*.py'))}
    geometry_identity='compact_c3:'+c.digest(dict(canonical=geometry_content,model=model_content,config=config))
    tracepath=inputdir/f'N{n}_traces.npz';choicepath=inputdir/f'choices_N{n}.npz'
    with np.load(tracepath,allow_pickle=False) as z:
        ends=z['ends'].copy()
        if ends.shape!=(n**3,4,3) or not np.isfinite(ends).all():
            raise ValueError('Q08 merged trace shape/finite mismatch')
        if 'raw' in z.files and not np.array_equal(z['raw'],np.arange(n**3)):
            raise ValueError('Q08 merged trace raw-grid ordering mismatch')
        if 'points' in z.files and not np.array_equal(z['points'],t.pts):
            raise ValueError('Q08 merged trace point ordering mismatch')
    with np.load(choicepath,allow_pickle=False) as z:choices=z['choice'].copy()
    if choices.shape!=(n**3,) or choices.dtype.kind not in 'iu':
        raise ValueError('Q08 frozen choice shape/dtype mismatch')
    lineage=dict(trace=c.sha(tracepath),choice=c.sha(choicepath),geometry=geometry_identity,
                 catalogue=c.DESIGNS,omega_weights=c.WC.tolist(),omega_offset=float(c.WOFF),
                 field_producer_sha256=c.sha(c.__file__),observation_producer_sha256=c.sha(__file__))
    data_identity=c.digest(lineage)
    data=_data(n,t,rundir,data_identity) if with_data else {}
    ENV=dict(n=n,t=t,ctx=ctx,geom=geom,jac=jac,ends=ends,choices=choices,
             source_identity=lineage['trace'],choice_provenance=lineage['choice'],
             geometry_identity=geometry_identity,lineage=lineage,data_identity=data_identity,
             inputdir=inputdir,rundir=rundir,**data)
    return ENV


def build_data(n,canonical_root,inputdir,rundir):
    initialize(n,canonical_root,inputdir,rundir,with_data=True)
    return json.loads((Path(rundir)/'data'/f'N{n}'/'data.json').read_text())


def chunk(index,owners):
    """Replay 31 stored arrays and all 22×4×2 six-field action components."""
    if ENV is None or 'state' not in ENV:
        raise RuntimeError('Q08 initialized worker with owner data required')
    import jax
    from drbx.stencils.q_parallel import prepare_paired_chunks,raw_members,_ARRAYS
    from drbx.stencils.q_parallel_rhs import prepare_six_field_rhs
    from drbx.stencils.q_bank import build_q_bank
    from drbx.stencils.q_artifact import save_q_bank,load_q_bank
    from drbx.stencils.q_plan import lower_q_plan
    from drbx.native.q_plan import apply_q_plan
    legacy_stencils=importlib.import_module(__package__+'.legacy.stencils.q_parallel')
    legacy_rhs=importlib.import_module(__package__+'.legacy.stencils.q_parallel_rhs')
    legacy_apply=importlib.import_module(__package__+'.legacy.native.q_parallel_rhs').apply_six_field_rhs
    e=ENV;n=e['n'];owners=np.asarray(owners)
    if (owners.ndim!=1 or owners.dtype.kind not in 'iu' or len(owners)==0 or
        np.any(owners<0) or np.any(owners>=len(e['t'].vol)) or len(np.unique(owners))!=len(owners)):
        raise ValueError('Q08 chunk complete owner ids required')
    owners=owners.astype(np.int64);raw=raw_members(e['t'],owners);ends=e['ends'][raw]
    kw=dict(source_identity=e['source_identity'],geometry_identity=e['geometry_identity'],raw=raw,
            frozen_choices=e['choices'][raw],choice_provenance=e['choice_provenance'])
    tick=time.perf_counter()
    old=tuple(legacy_stencils.prepare_chunk(e['t'],owners,ends,e['geom'],e['jac'],span=s,**kw) for s in (1/16,1/32))
    old_prepare=time.perf_counter()-tick;tick=time.perf_counter()
    pair=prepare_paired_chunks(e['t'],owners,ends,e['geom'],e['jac'],**kw)
    paired_prepare=time.perf_counter()-tick
    for before,after in zip(old,pair):
        for name in _ARRAYS:
            a,b=getattr(before,name),getattr(after,name)
            if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():
                raise ValueError(f'Q08 legacy/paired array mismatch {name}')
    bank=build_q_bank(*pair,identity=e['lineage'])
    bi,bo,pb=c.boundaries(bank)
    metrics=[];plan_bytes=[];times=[];geometry=None
    # Both coefficient audits are independent legacy/new computations. Their
    # geometry-consistent material/B rows do not depend on diffusion span.
    legacy_base,_=legacy_rhs.prepare_six_field_rhs(old[1],old[0],e['geom'],diffusion_span=1/32,center_b_atol=1e-10)
    base,_=prepare_six_field_rhs(pair[1],pair[0],e['geom'],diffusion_span=1/32,center_b_atol=1e-10)
    for span in (1/16,1/32):
        tick=time.perf_counter()
        legacy_rt=legacy_base if span==1/32 else _span_runtime(legacy_base,old[0],span)
        runtime=base if span==1/32 else _span_runtime(base,pair[0],span)
        plan=lower_q_plan(bank,diffusion_span=span,material_runtime=runtime)
        plan_bytes.append(plan.nbytes)
        geometry=dict(magnetic_L=runtime.material.magnetic_L,b_eta=runtime.material.b_eta,
                      eta_step=runtime.material.eta_step,bmag=runtime.bmag)
        for ki,(kinds,phi_kind) in enumerate(c.KINDS):
            kwargs=dict(kinds=kinds,phi_kind=phi_kind,tau=c.TAU,mu=c.MU)
            expected=legacy_apply(legacy_rt,e['state'],bi,bo,e['phi'],pb,c.COEFF,**kwargs)
            actual=apply_q_plan(plan,e['state'],bi,bo,e['phi'],pb,c.COEFF,**kwargs)
            metrics.append(dict(span=span,kind=ki,**c.check_outputs(actual,expected)))
        times.append(time.perf_counter()-tick)
        del legacy_rt,runtime,plan,expected,actual
    path=e['rundir']/'cpu'/f'N{n}'/f'chunk_{int(index):06d}'
    path.mkdir(parents=True,exist_ok=True)
    bankpath=save_q_bank(bank,path/'bank.npz')
    # Checked round trip is part of CPU evidence before a receipt is emitted.
    load_q_bank(bankpath,expected_identity=bank.identity)
    c.atomic_npz(path/'geometry.npz',**geometry)
    stats=dict(passed=True,n=n,index=int(index),owners=owners.tolist(),raw=raw.tolist(),
        files={'bank.npz':c.sha(bankpath),'geometry.npz':c.sha(path/'geometry.npz')},
        all_arrays_bitwise=True,max_scaled=max(m['max_scaled_error'] for m in metrics),
        max_abs=max(m['max_abs_error'] for m in metrics),components=metrics,
        timings=dict(legacy_prepare_seconds=old_prepare,paired_prepare_seconds=paired_prepare,
                     span_replay_seconds=times),peak_rss_gib=_rss(),plan_bytes=plan_bytes,
        bank_identity=bank.identity,footprint=bank.footprint(),input_identity=e['data_identity'])
    c.atomic_json(path/'stats.json',stats)
    return stats


def bounded_replay(directory,n):
    """Independent persisted seven-owner actions against current C3 preparation.

    Floating saved-array comparisons are reported across platforms; integer
    identities and ordering remain exact. The inherited action budget gates
    every persisted component N, with no continuum/reference computation.
    """
    if ENV is None or ENV['n']!=n or 'state' not in ENV:
        raise ValueError('Q08 bounded replay needs initialized matching grid')
    from drbx.stencils.q_parallel import load_chunk,prepare_paired_chunks,raw_members,_ARRAYS
    from drbx.stencils.q_bank import build_q_bank
    from drbx.stencils.q_parallel_rhs import prepare_six_field_rhs
    from drbx.stencils.q_plan import lower_q_plan
    from drbx.native.q_plan import apply_q_plan
    directory=Path(directory);e=ENV
    saved=tuple(load_chunk(directory/f'N{n}_h{denom}.npz') for denom in (16,32))
    owners=saved[0].owners;raw=raw_members(e['t'],owners);ends=e['ends'][raw]
    pair=prepare_paired_chunks(e['t'],owners,ends,e['geom'],e['jac'],raw=raw,
        source_identity=e['source_identity'],geometry_identity=e['geometry_identity'],
        frozen_choices=e['choices'][raw],choice_provenance=e['choice_provenance'])
    row_metrics={}
    for ai,(before,after) in enumerate(zip(saved,pair)):
        for key in _ARRAYS:
            a,b=getattr(after,key),getattr(before,key)
            if a.shape!=b.shape or a.dtype!=b.dtype:
                raise ValueError('Q08 bounded saved row shape/dtype mismatch '+key)
            exact=a.tobytes()==b.tobytes()
            if a.dtype.kind in 'biu':
                if not np.array_equal(a,b):raise ValueError('Q08 bounded saved row ordering/identity mismatch '+key)
                error=0.
            else:
                if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Q08 nonfinite bounded rows')
                error=float(np.max(abs(a-b),initial=0))
            row_metrics[f'h{16 if ai==0 else 32}_{key}']=dict(bitwise_equal=exact,max_abs_error=error)
    bank=build_q_bank(*pair,identity=e['lineage']);bi,bo,pb=c.boundaries(bank)
    base,_=prepare_six_field_rhs(pair[1],pair[0],e['geom'],diffusion_span=1/32,center_b_atol=1e-10)
    def project(a):
        return np.sum(a[:,bank.owner_raw,:]*bank.owner_weight[None,:,:,None],axis=-2)
    replay={}
    with np.load(directory/f'actions_N{n}.npz',allow_pickle=False) as z:
        if not np.array_equal(z['owners'],owners) or not np.array_equal(z['raw'],raw):
            raise ValueError('Q08 bounded persisted source identity mismatch')
        for span in (1/16,1/32):
            rt=base if span==1/32 else _span_runtime(base,pair[0],span)
            plan=lower_q_plan(bank,diffusion_span=span,material_runtime=rt)
            for ki,(kinds,phi_kind) in enumerate(c.KINDS):
                result=apply_q_plan(plan,e['state'],bi,bo,e['phi'],pb,c.COEFF,
                    kinds=kinds,phi_kind=phi_kind,tau=c.TAU,mu=c.MU)
                current=project(np.asarray(result.raw_current.divergence_physical)[...,None])
                omega_current=project(np.asarray(result.raw_current.vorticity_current)[...,None])
                phi_force=project(np.asarray(result.raw_current.electron_phi)[...,None])
                components={key:np.asarray(getattr(result,key)) for key in ('centered','correction','diffusion','combined')}
                components.update(current=current,omega_current=omega_current,phi_force=phi_force,
                    omega_advection=components['centered'][...,5:6]-omega_current)
                for key,a in components.items():
                    tag=f's{int(1/span)}_k{ki}_{key}_N'
                    replay[tag]=c.check_outputs(a,z[tag])
    return dict(passed=True,n=n,owners=owners.tolist(),saved_row_arrays=row_metrics,
                all_saved_arrays_bitwise=all(v['bitwise_equal'] for v in row_metrics.values()),
                max_scaled=max(v['max_scaled_error'] for v in replay.values()),components=replay,
                reference_sha256=c.sha(directory/f'actions_N{n}.npz'))
