"""Bounded GPU-only replay/benchmark of the opt-in Q polynomial splitter.

Reuses immutable N32 Q08 banks. Host work is input/setup/validation only;
both reference and candidate numerical actions execute on actual GPUs.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import shutil
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
BASE_ID='ff1bff86e4af5045b84856d16550f0f45edcbe03b83fc83cc5584557af4c8722'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def check_source():
    manifest=json.loads((HERE/'manifest.json').read_text())
    for rel,h in manifest['files'].items():
        if sha(HERE/rel)!=h:raise ValueError('candidate source mismatch '+rel)
    return manifest


def validate(output):
    receipt=json.loads((output/'completion.json').read_text())
    if not receipt['passed']:raise ValueError('diagnostic incomplete or failed')
    for rel,h in receipt['files'].items():
        if sha(output/rel)!=h:raise ValueError('output checksum '+rel)
    result=json.loads((output/'results.json').read_text())
    if len(result['replays'])!=12 or not all(x['passed'] for x in result['replays']):
        raise ValueError('incomplete case/BC matrix')
    if {(x['case'],x['kinds']) for x in result['replays']}!={(c,k) for c in (0,1,21) for k in range(4)}:
        raise ValueError('wrong case/BC coverage')
    if set(result['timings'])!={'eig_full','polynomial_full','eig_split','polynomial_split',
                               'polynomial_sharded_1','polynomial_sharded_4'}:
        raise ValueError('missing timing component')
    if not result['split_replay']['passed'] or not all(x['passed'] for x in result['shard_replay']):
        raise ValueError('component/shard replay')
    if {x['devices'] for x in result['shard_replay']}!={1,4}:raise ValueError('shard coverage')
    import math
    for timing in result['timings'].values():
        if len(timing['seconds'])!=3 or not all(math.isfinite(v) and v>0 for v in timing['seconds']):
            raise ValueError('timing samples')
    return dict(passed=True,output=str(output),candidate_identity=receipt['candidate_identity'])


def run(args):
    output=args.output.resolve();baseline=args.baseline_source.resolve();oldrun=args.run.resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('use a new empty output directory')
    if output==oldrun or oldrun in output.parents:raise ValueError('keep diagnostic separate from original RUN')
    output.mkdir(parents=True,exist_ok=True)
    for name in ('compiler','provenance','source'):(output/name).mkdir()
    manifest=check_source();identity=sha(HERE/'manifest.json')
    for rel in (*manifest['files'],'manifest.json'):
        dest=output/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HERE/rel,dest)
    start=time.monotonic()
    def interrupted(signum,frame):
        raise TimeoutError('bounded diagnostic interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    result=dict(candidate_identity=identity,baseline_identity=BASE_ID,timings={},replays=[],shard_replay=[])
    def save():
        result['elapsed_seconds']=time.monotonic()-start
        write(output/'results.json',result)
    try:
        # Use frozen campaign imports, with only the reviewed five-file overlay.
        sys.path.insert(0,str(baseline.parents[1]))
        from scripts.q08_extraction_global import bootstrap,campaign
        bootstrap.configure(gpu=True,output=output)
        if campaign.HERE.resolve()!=baseline:raise ValueError('wrong baseline import')
        if campaign.digest(campaign.design())!=BASE_ID:raise ValueError('baseline identity')
        import drbx.native
        drbx.native.__path__.insert(0,str(HERE/'overlay'))
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
        from scripts.q08_extraction_global import gpu,common
        from drbx.stencils.q_plan import lower_q_plan
        from drbx.native import q_plan,q_sharding
        from drbx.native.q_characteristic_polynomial import polynomial_characteristic_split
        from drbx.native.fci_parallel_production_flux import parallel_characteristic_split,parallel_matrix_from_state
        for name in ('q_plan','q_sharding','q_parallel_material','q_parallel_characteristic','q_characteristic_polynomial'):
            module=importlib.import_module('drbx.native.'+name)
            if Path(module.__file__).resolve()!=HERE/'overlay'/f'{name}.py':raise ValueError('overlay import '+name)
        devices,inventory=gpu.device_inventory()
        if len(devices)!=4 or any('A100' not in d.device_kind for d in devices):raise ValueError('four actual A100s required')
        if not jax.config.jax_enable_x64 or jax.default_backend()!='gpu':raise ValueError('GPU x64 required')
        # Standard receipts, unlike the earlier external profiler's copied aliases.
        cpu=campaign.require(oldrun,'cpu_N32',BASE_ID)
        data=campaign.require(oldrun,'data_N32',BASE_ID)
        for name,h in data['files'].items():
            if sha(oldrun/'data/N32'/name)!=h:raise ValueError('data changed '+name)
        chunks,estimate=gpu.estimate_merge(oldrun,32,BASE_ID,args.host_gib)
        for index,(path,_) in enumerate(chunks):
            key=str(index)
            if cpu['chunk_receipts'].get(key)!=sha(path/'stats.json'):raise ValueError('CPU chunk receipt '+key)
        gpu._device_guard(estimate,inventory)
        if shutil.disk_usage(output).free<3*(1<<30):raise RuntimeError('free disk below 3 GiB')
        prov=dict(identity=identity,manifest=manifest,baseline=str(baseline),run=str(oldrun),
                  inventory=inventory,jax=jax.__version__,python=sys.version,
                  affinity=sorted(os.sched_getaffinity(0)),host_gib=args.host_gib,
                  cpu_receipt=cpu,data_receipt=data,estimate=estimate,
                  baseline_design_sha256=sha(baseline/'design.json'))
        write(output/'provenance/inputs.json',prov)
        setup=time.monotonic();bank,geometry,views,_=gpu.merge_chunks(oldrun,32,BASE_ID,args.host_gib)
        del views
        plan=lower_q_plan(bank,diffusion_span=1/16,**geometry)
        result['setup_seconds']=time.monotonic()-setup
        state=np.load(oldrun/'data/N32/state.npy',mmap_mode='r')
        phi=np.load(oldrun/'data/N32/phi.npy',mmap_mode='r')
        topology=np.load(oldrun/'data/N32/raw_to_owner.npy',mmap_mode='r')
        if state.shape[0]!=22 or len(bank.raw)!=32768:raise ValueError('N32 catalogue/coverage')

        def sync(tree):
            for v in jax.tree.leaves(tree):
                if hasattr(v,'block_until_ready'):v.block_until_ready()
            return tree

        def place(tree,device):return sync(jax.tree.map(lambda x:jax.device_put(x,device),tree))

        def check(actual,expected):
            metrics=common.check_outputs(jax.device_get(actual),jax.device_get(expected),atol=1e-8,rtol=1e-11)
            return dict(passed=True,metrics=metrics)

        def compiled(fn,arguments,name,candidate):
            t=time.monotonic();exe=jax.jit(fn).lower(*arguments).compile()
            text=exe.as_text();(output/'compiler'/f'{name}.txt').write_text(text)
            targets=re.findall(r'custom_call_target\s*=\s*"([^"]+)"',text.lower())
            if candidate and any(any(term in target for term in ('geev','eig_real','callback')) for target in targets):
                raise ValueError('candidate contains eig/host callback '+str(targets))
            return exe,time.monotonic()-t,targets

        def bench(exe,arguments,name,compile_s,targets):
            if time.monotonic()-start>1500:raise TimeoutError('bounded diagnostic time limit')
            out=sync(exe(*arguments));times=[]
            for _ in range(3):
                t=time.monotonic();out=sync(exe(*arguments));times.append(time.monotonic()-t)
            result['timings'][name]=dict(compile_seconds=compile_s,seconds=times,
                median_seconds=float(np.median(times)),custom_calls=targets)
            save();return out

        plan_gpu=place(plan,devices[0]);coeff=place(common.COEFF,devices[0])
        # Three fixed fields: smooth first, constant and held-out short wave;
        # all four uniform/mixed D/N patterns. Only first case is timed.
        cached={}
        for case in (1,0,21):
            if time.monotonic()-start>1500:raise TimeoutError('bounded diagnostic time limit')
            t=time.monotonic();bi,bo,pb=common.boundaries(bank,case)
            result.setdefault('boundary_setup_seconds',{})[str(case)]=time.monotonic()-t
            values=place((state[case],bi,bo,phi[case],pb),devices[0])
            arguments=(plan_gpu,*values,coeff)
            for ki,(kinds,pk) in enumerate(common.KINDS):
                if ki not in cached:
                    def fn(*a,method,kinds=kinds,pk=pk):
                        return q_plan.apply_q_plan(*a,kinds=kinds,phi_kind=pk,tau=common.TAU,mu=common.MU,
                                                   characteristic_method=method)
                    pair=[]
                    for method in ('eig','polynomial'):
                        pair.append(compiled(lambda *a,m=method:fn(*a,method=m),arguments,f'{method}_k{ki}',method=='polynomial'))
                    cached[ki]=pair
                outputs=[]
                for method,(exe,ct,targets) in zip(('eig','polynomial'),cached[ki]):
                    if case==1 and ki==0:out=bench(exe,arguments,method+'_full',ct,targets)
                    else:out=sync(exe(*arguments))
                    outputs.append(out)
                if not bool(jnp.all(outputs[0].eigensystem_admissible)&jnp.all(outputs[0].inputs_valid)):
                    raise ValueError('baseline invalid state/inputs')
                result['replays'].append(dict(case=case,kinds=ki,**check(outputs[1],outputs[0])))
                save()
                if case==1 and ki==0:
                    reference=jax.device_get(outputs[0]);saved_args=arguments
                    saved_boundary=(bi,bo,pb)
            del outputs,values,arguments

        # Actual reconstructed principal matrices, not artificial timing data.
        xx,bi,bo,pp,pb=saved_args[1:-1]
        slots=q_plan.reconstruct_q_state(plan_gpu,xx,bi,bo,kinds=common.KINDS[0][0])
        centers=jnp.moveaxis(slots.value[:5,:,2],0,-1)
        matrices=parallel_matrix_from_state(centers,common.TAU,common.MU)
        split_outputs=[]
        for name,fn,arguments in (
            ('eig_split',lambda m,n:parallel_characteristic_split(m,n),(matrices,plan_gpu.b_eta)),
            ('polynomial_split',lambda s,n:polynomial_characteristic_split(s,common.TAU,common.MU,n),(centers,plan_gpu.b_eta))):
            exe,ct,targets=compiled(fn,arguments,name,name.startswith('polynomial'))
            split_outputs.append(bench(exe,arguments,name,ct,targets))
        # Generic tuple comparison (the original profiler's named-field helper
        # cannot compare tuples); retain exact boolean admission masks.
        metrics=[]
        for got,ref in zip(split_outputs[1],split_outputs[0],strict=True):
            a,b=np.asarray(got),np.asarray(ref)
            if a.dtype.kind=='b':
                if not np.array_equal(a,b):raise ValueError('split validity differs')
                metrics.append(dict(boolean_equal=True))
            else:
                delta=np.abs(a-b);gate=1e-8+1e-11*np.abs(b)
                if not np.all(np.isfinite(a)) or np.any(delta>gate):raise ValueError('split replay failed')
                metrics.append(dict(max_abs=float(delta.max()),max_scaled=float((delta/gate).max())))
        result['split_replay']=dict(passed=True,metrics=metrics);save()
        # Candidate one/four-GPU execution only; baseline remains monolithic.
        bi,bo,pb=saved_boundary
        for count in (1,4):
            mesh=Mesh(np.asarray(devices[:count],object),('z',))
            t=time.monotonic();sh=q_sharding.shard_q_plan(plan,bank,topology,count)
            sh=gpu._place_plan(sh,mesh);sync(sh)
            args_sh=(sh,*place((state[1],bi,bo,phi[1],pb,common.COEFF),NamedSharding(mesh,P())))
            result.setdefault('shard_setup_seconds',{})[str(count)]=time.monotonic()-t
            fn=lambda *a:q_sharding.sharded_q_rhs(*a,kinds=common.KINDS[0][0],phi_kind='D',
                tau=common.TAU,mu=common.MU,mesh=mesh,characteristic_method='polynomial')
            name=f'polynomial_sharded_{count}'
            exe,ct,targets=compiled(fn,args_sh,name,True)
            got=bench(exe,args_sh,name,ct,targets)
            result['shard_replay'].append(dict(devices=count,**check(got,reference)));save()
            del got,exe,args_sh,sh
        check_source()
        if campaign.digest(campaign.design())!=BASE_ID:raise ValueError('baseline changed during run')
        result['host_peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1<<20)
        result['device_memory']=[{k:int(v) for k,v in (d.memory_stats() or {}).items()
                                 if isinstance(v,(int,np.integer))} for d in devices]
        save()
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()
               and p.parts[len(output.parts)] in ('compiler','provenance','source')}
        files['results.json']=sha(output/'results.json')
        write(output/'completion.json',dict(passed=True,candidate_identity=identity,files=files))
        validate(output)
    except Exception as exc:
        result['error']=repr(exc);result['traceback']=traceback.format_exc();save()
        write(output/'completion.json',dict(passed=False,candidate_identity=identity,error=repr(exc)))
        raise


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--baseline-source',type=Path,required=True)
    r.add_argument('--run',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    r.add_argument('--host-gib',type=float,required=True)
    v=sub.add_parser('validate');v.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='run':run(args)
    else:print(json.dumps(validate(args.output)))


if __name__=='__main__':main()
