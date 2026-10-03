"""Revised GPU stage; reuses immutable legacy banks and changes only GPU splitter.

The core orchestration/validator below derives from the frozen Q08 gpu.py.
CPU reference uses its eig method; candidate always uses polynomial on device.
"""
from pathlib import Path
import gc,json,os,time,re,hashlib,uuid
import numpy as np
import jax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from scripts.q08_extraction_global import gpu as old
from scripts.q08_extraction_global.gpu import (
    _common,_sync,_place_plan,_unit_path,_valid_record,
    _valid_metrics,_device_guard,device_inventory,checked_chunks,
    sha256_file,digest,write_json,peak_rss_gib,lock,merge_chunks,
    lower_q_plan,apply_q_plan,shard_q_plan,sharded_q_rhs,sha256_array)
from boundary_cache import CachedAPI
from boundary_values import catalogue_replay
from timing import PhaseLedger,record_totals
from literal_reference import LiteralReference


def estimate_merge(run,n,identity,host_memory_gib):
    chunks,estimate=old.estimate_merge(run,n,identity,host_memory_gib)
    # Cache keeps only wall rows plus exact shared nonwall templates, not 22
    # dense field copies. Bound includes all22 full-grid states and literal
    # chunk smooth-state caches, plus temporary wall-index metadata.
    extra=2*estimate['one_case_boundary_bytes']
    estimate['boundary_cache_budget_bytes']=extra
    estimate['estimated_host_peak_upper_bytes']+=extra
    available=min(estimate['host_budget_bytes'],estimate.get('observed_host_available_bytes',2**63))
    if estimate['estimated_host_peak_upper_bytes']>available:
        raise MemoryError('compressed BC cache exceeds host preflight')
    return chunks,estimate


def compiler_guard(text,*,test_cpu=False):
    targets=sorted(set(re.findall(r'custom_call_target\s*=\s*"([^"]+)"',text)))
    if any(any(word in target.lower() for word in ('geev','eig_real','callback')) for target in targets):
        raise ValueError('candidate contains eigensolver/host callback')
    if not test_cpu and any('lapack' in target.lower() for target in targets):
        raise ValueError('candidate contains CPU LAPACK')
    return targets


class CompilerReceipts:
    """Read/hash/parse each immutable proof once per validation invocation."""
    def __init__(self, run, *, test_cpu=False):
        self.run = Path(run).resolve()
        self.test_cpu = test_cpu
        self.entries = {}

    def get(self, name):
        path = self.run/name
        if path.is_symlink() or not path.resolve().is_relative_to(self.run):
            raise ValueError('compiler path escapes run')
        if name not in self.entries:
            content = path.read_bytes()
            self.entries[name] = (hashlib.sha256(content).hexdigest(),
                compiler_guard(content.decode(), test_cpu=self.test_cpu))
        return self.entries[name]


def memory_snapshot(devices):
    # Hardware/driver inventory is established once at preflight. Runtime
    # allocator statistics do not require spawning nvidia-smi 352 times.
    return [dict(id=d.id,local_hardware_id=d.local_hardware_id,platform=d.platform,
        kind=d.device_kind,process_index=d.process_index,
        memory_stats={k:int(v) for k,v in (d.memory_stats() or {}).items()
                      if isinstance(v,(int,np.integer))}) for d in devices]


def validate_timing_record(run,r,*,repeats=7,test_cpu=False,compilers=None):
    if r.get('method')!='polynomial':raise ValueError('wrong characteristic method')
    samples=r.get('warm_seconds',[])
    if len(samples)!=repeats or not all(np.isfinite(v) and v>0 for v in samples):raise ValueError('warm timing coverage')
    if r.get('warm_median_seconds')!=float(np.median(samples)):raise ValueError('warm median mismatch')
    compilers=compilers or CompilerReceipts(run,test_cpu=test_cpu)
    receipt,targets=compilers.get(r['compiler'])
    if receipt!=r['compiler_sha256']:raise ValueError('compiler content receipt')
    if targets!=r['custom_calls']:raise ValueError('compiler targets differ')


def candidate_record(path,identity,no,nr,run,*,test_cpu=False,repeats=7,compilers=None):
    if not path.exists():return False
    if not _valid_record(path,identity,no,nr):raise ValueError('existing candidate record failed identity/replay gate')
    record=json.loads(path.read_text())
    if record.get('test_only') is not test_cpu:raise ValueError('candidate checkpoint backend mismatch')
    validate_timing_record(run,record,test_cpu=test_cpu,repeats=repeats,compilers=compilers)
    return True


def run_resolution(run,n,identity,host_memory_gib,*,baseline_run,baseline_identity,test_cpu=False,cases=None,
                   common_api=None,warm_repeats=7):
    """Validate all22×4BC×2span on actual1/4GPU, with per-case checked resume.

    ``cases`` may restrict only explicit forced-CPU test runs. No full-grid
    arrays are merged before host/device resource preflight. Timings separate
    host preparation, transfer, first synchronized JIT and warm full-operator
    calls; none is a measured full integrator timestep or kernel-only claim.
    """
    ledger=PhaseLedger();attempt=uuid.uuid4().hex
    api=CachedAPI(_common(common_api),optimize=common_api is None);run=Path(run);baseline_run=Path(baseline_run);root=run/'gpu'/f'N{n}';root.mkdir(parents=True,exist_ok=True)
    compilers=CompilerReceipts(run,test_cpu=test_cpu)
    if not jax.config.jax_enable_x64:raise RuntimeError('Q08 replay requires JAX_ENABLE_X64=true')
    if isinstance(warm_repeats,bool) or not isinstance(warm_repeats,int) or warm_repeats<1:raise ValueError('positive warm sample count required')
    if cases is not None and not test_cpu:raise ValueError('actual GPU replay cannot restrict the22-state matrix')
    if not test_cpu and (run/f'gpu_N{n}.json').exists():
        return validate_resolution(run,n,identity,baseline_run=baseline_run,baseline_identity=baseline_identity)
    # Smooth nonconstant case first makes the mandatory merge audit exercise
    # donor/projection ordering; constants are still replayed in the matrix.
    cases=(1,0,*range(2,22)) if cases is None else tuple(cases)
    if not cases or len(set(cases))!=len(cases) or any(isinstance(c,bool) or not isinstance(c,int) or c<0 or c>=22 for c in cases):raise ValueError('invalid test case subset')
    if len(api.KINDS)!=4:raise ValueError('four inherited boundary combinations required')
    devices,inventory=device_inventory(test_cpu)
    chunks,estimate=estimate_merge(baseline_run,n,baseline_identity,host_memory_gib)
    _device_guard(estimate,inventory)
    api.cache.max_bytes=estimate['boundary_cache_budget_bytes']
    data=baseline_run/'data'/f'N{n}'
    for name in ('state.npy','phi.npy','raw_to_owner.npy'):
        if not (data/name).is_file():raise ValueError(f'missing GPU input {name}')
    inputs={name:sha256_file(data/name) for name in ('state.npy','phi.npy','raw_to_owner.npy')}
    sources={str(Path(__file__).name):sha256_file(__file__)}
    for name in ('boundary_cache.py','boundary_values.py','timing.py','literal_reference.py'):
        sources[name]=sha256_file(Path(__file__).with_name(name))
    from scripts.q08_extraction_global import common as common_module
    sources['common']=sha256_file(common_module.__file__)
    # Parent campaign identity carries broader source/input provenance; these
    # direct source receipts additionally protect the wrapper and runtime.
    for module in ('drbx.native.q_plan','drbx.native.q_sharding','drbx.native.q_parallel_material','drbx.native.q_parallel_characteristic','drbx.native.q_characteristic_polynomial','drbx.stencils.q_plan'):
        obj=__import__(module,fromlist=['__file__']);sources[module]=sha256_file(obj.__file__)
    grid_identity=digest(dict(campaign=identity,n=n,inputs=inputs,sources=sources,
        cpu=[dict(index=r['index'],files=r['files']) for _,r in chunks],
        hardware=dict(jax_version=inventory['jax_version'],device_kinds=[d['kind'] for d in inventory['devices']]),
        backend='forced_cpu_test' if test_cpu else 'A100_gpu',method='polynomial',baseline_identity=baseline_identity,warm_repeats=warm_repeats,matrix=dict(cases=cases,kinds=api.KINDS,spans=(1/16,1/32))))
    summary=dict(identity=grid_identity,campaign_identity=identity,baseline_identity=baseline_identity,baseline_run=str(baseline_run.resolve()),method="polynomial",warm_repeats=warm_repeats,n=n,passed=False,inventory=inventory,
        preflight=estimate,inputs=inputs,sources=sources,cpu_file_receipts=[dict(index=r['index'],files=r['files']) for _,r in chunks],test_only=bool(test_cpu),matrix_units=2*4*len(cases)*2,
        timing_scope='synchronized full Q operator setup/transfer/first/warm calls; not an integrator timestep',records=[])
    with lock(root):
        write_json(root/'preflight.json',summary)
        ledger.lap('preflight_and_provenance')
        tick=time.perf_counter();bank,geom,views,_=merge_chunks(baseline_run,n,baseline_identity,host_memory_gib)
        summary['merge_seconds']=time.perf_counter()-tick;summary['merged_identity']=bank.identity
        ledger.lap('merge_checked_banks')
        state=np.load(data/'state.npy',mmap_mode='r');phi=np.load(data/'phi.npy',mmap_mode='r')
        topology=np.load(data/'raw_to_owner.npy',mmap_mode='r')
        if state.shape!=(22,6,bank.metadata['n_owner']) or phi.shape!=(22,bank.metadata['n_owner']):raise ValueError('full22-state input shape mismatch')
        if topology.shape!=(n**3,) or sha256_array(np.asarray(topology,dtype=np.int64))!=bank.metadata['topology_hash'].split(':')[0]:raise ValueError('GPU input topology identity mismatch')
        ledger.lap('input_load_and_topology_check')
        summary['boundary_catalogue_replay']=(catalogue_replay(bank,api.api) if common_api is None
            else dict(test_only=True,reason='explicit test API'))
        ledger.lap('boundary_catalogue_replay')
        for span in (1/16,1/32):
            literal_reference=LiteralReference(views,span,api)
            tick=time.perf_counter();plan=lower_q_plan(bank,diffusion_span=span,**geom)
            with jax.default_device(jax.devices('cpu')[0]):
                cpu_plan=jax.device_put(plan,jax.devices('cpu')[0])
            variants={}
            for count in (1,4):
                mesh=Mesh(np.asarray(devices[:count],object),('z',))
                start=time.perf_counter();stacked=shard_q_plan(plan,bank,topology,count)
                host_seconds=time.perf_counter()-start
                start=time.perf_counter();placed=_sync(_place_plan(stacked,mesh));transfer=time.perf_counter()-start
                variants[count]=(placed,mesh,dict(host_plan_seconds=host_seconds,plan_transfer_seconds=transfer,
                    logical_stacked_plan_bytes=sum(a.nbytes for a in jax.tree.leaves(stacked))))
                del stacked
            summary.setdefault('plan_setup',[]).append(dict(span=span,seconds=time.perf_counter()-tick,
                variants={str(k):v[2] for k,v in variants.items()}))
            ledger.lap('plan_setup_and_placement')
            for ki,(kinds,pk) in enumerate(api.KINDS):
                cpu_call=jax.jit(lambda qp,x,ib,ob,p,pb,c:apply_q_plan(qp,x,ib,ob,p,pb,c,
                    kinds=tuple(kinds),phi_kind=pk,tau=api.TAU,mu=api.MU),backend='cpu')
                audit_path=root/'merge_audits'/f's{int(1/span)}_k{ki}.json'
                merge_audited=_valid_record(audit_path,grid_identity,len(bank.owners),len(bank.raw))
                if merge_audited:summary.setdefault('cpu_merge_audits',[]).append(str(audit_path.relative_to(run)))
                calls={count:jax.jit(lambda qp,x,ib,ob,p,pb,c,mesh=mesh:sharded_q_rhs(qp,x,ib,ob,p,pb,c,
                    kinds=tuple(kinds),phi_kind=pk,tau=api.TAU,mu=api.MU,mesh=mesh,characteristic_method="polynomial")) for count,(_,mesh,_) in variants.items()}
                for case in cases:
                    paths={count:_unit_path(root,span,ki,case,count) for count in (1,4)}
                    if merge_audited and all(candidate_record(path,grid_identity,len(bank.owners),len(bank.raw),run,test_cpu=test_cpu,repeats=warm_repeats,compilers=compilers) for path in paths.values()):
                        summary['records'].extend(str(p.relative_to(run)) for p in paths.values());ledger.lap('checkpoint_and_control');continue
                    x=np.asarray(state[case]);p=np.asarray(phi[case])
                    shared_work_id=f'{attempt}/s{int(1/span)}_k{ki}_c{case}'
                    ledger.lap('checkpoint_and_control')
                    with jax.default_device(jax.devices('cpu')[0]):
                        start=time.perf_counter();bi,bo,pb=api.boundaries(bank,case);bc_seconds=time.perf_counter()-start
                        ledger.lap('boundary_prepare')
                        cpu_args=jax.tree.map(lambda a:jax.device_put(a,jax.devices('cpu')[0]),(x,bi,bo,p,pb,np.asarray(api.COEFF)))
                        _sync(cpu_args);ledger.lap('cpu_argument_placement')
                        cpu_before=cpu_call._cache_size();start=time.perf_counter()
                        expected=jax.tree.map(np.asarray,_sync(cpu_call(cpu_plan,*cpu_args)))
                        cpu_seconds=time.perf_counter()-start
                        ledger.lap('cpu_merged_reference')
                    if not merge_audited:
                        start=time.perf_counter();literal=literal_reference(x,p,case,kinds,pk)
                        audit=api.check_outputs(expected,literal)
                        write_json(audit_path,dict(identity=grid_identity,campaign_identity=identity,merged_identity=bank.identity,passed=True,span=span,kind=ki,case=case,metrics=audit,
                            literal_chunk_eager_seconds=time.perf_counter()-start,chunk_actions=len(views)))
                        summary.setdefault('cpu_merge_audits',[]).append(str(audit_path.relative_to(run)))
                        merge_audited=True;del literal
                        ledger.lap('literal_chunk_merge_audit')
                    del cpu_args
                    for count in (1,4):
                        path=paths[count]
                        if candidate_record(path,grid_identity,len(bank.owners),len(bank.raw),run,test_cpu=test_cpu,repeats=warm_repeats,compilers=compilers):summary['records'].append(str(path.relative_to(run)));ledger.lap('checkpoint_and_control');continue
                        placed,mesh,_=variants[count]
                        ledger.lap('checkpoint_and_control')
                        start=time.perf_counter();args=jax.tree.map(lambda a:jax.device_put(a,NamedSharding(mesh,P())),(x,bi,bo,p,pb,np.asarray(api.COEFF)))
                        _sync(args);transfer_seconds=time.perf_counter()-start
                        ledger.lap('gpu_argument_placement')
                        before=calls[count]._cache_size();start=time.perf_counter();actual=_sync(calls[count](placed,*args));first=time.perf_counter()-start
                        ledger.lap('gpu_first_synchronized')
                        metrics=api.check_outputs(actual,expected)
                        ledger.lap('gpu_output_transfer_and_compare')
                        compiler_path=root/'compiler'/f's{int(1/span)}_k{ki}_d{count}.txt'
                        if not compiler_path.exists():
                            compiler_path.parent.mkdir(exist_ok=True)
                            compiler_path.write_text(calls[count].lower(placed,*args).compile().as_text())
                        compiler_name=str(compiler_path.relative_to(run))
                        compiler_hash,targets=compilers.get(compiler_name)
                        ledger.lap('compiler_proof')
                        samples=[]
                        for _ in range(warm_repeats):
                            start=time.perf_counter();_sync(calls[count](placed,*args));samples.append(time.perf_counter()-start)
                        ledger.lap('gpu_warm_repeats')
                        current_memory=memory_snapshot(devices)
                        ledger.lap('allocator_snapshot')
                        record=dict(identity=grid_identity,campaign_identity=identity,n=n,span=span,kind=ki,case=case,devices=count,
                            passed=True,test_only=bool(test_cpu),method='polynomial',compiler=compiler_name,compiler_sha256=compiler_hash,custom_calls=targets,metrics=metrics,cpu_reference_seconds=cpu_seconds,shared_work_id=shared_work_id,
                            cpu_reference_compiled_new_signature=cpu_call._cache_size()>cpu_before,
                            cpu_reference_path='full merged compact CPU JIT; first uncompleted case per span/kind audited against literal chunk compact eager',
                            boundary_host_seconds=bc_seconds,case_transfer_seconds=transfer_seconds,
                            first_synchronized_seconds=first,compiled_new_signature=calls[count]._cache_size()>before,
                            warm_seconds=samples,warm_median_seconds=float(np.median(samples)),
                            actual_device_memory=current_memory,host_peak_rss_gib=peak_rss_gib())
                        write_json(path,record);summary['records'].append(str(path.relative_to(run)));print(json.dumps(dict(n=n,completed=len(summary['records']),total=summary['matrix_units'],case=case,kind=ki,span=span,devices=count)),flush=True)
                        del actual,args
                        ledger.lap('record_write_and_cleanup')
                    del expected,bi,bo,pb,x,p
            del variants,calls,plan,cpu_plan,cpu_call
            summary.setdefault('literal_reference_work',[]).append(dict(span=span,**literal_reference.stats))
            del literal_reference
            gc.collect()
            ledger.lap('span_cleanup')
        if len(summary['records'])!=summary['matrix_units'] or any(not _valid_record(run/name,grid_identity,len(bank.owners),len(bank.raw)) for name in summary['records']):raise ValueError('GPU matrix completion mismatch')
        summary['passed']=True;summary['host_peak_rss_gib']=peak_rss_gib();summary['boundary_cache']=api.cache.stats()
        summary['record_hashes']={name:sha256_file(run/name) for name in summary['records']}
        summary['audit_hashes']={name:sha256_file(run/name) for name in summary['cpu_merge_audits']}
        summary['record_timing_totals']=record_totals([json.loads((run/name).read_text()) for name in summary['records']])
        summary['invocation_timing']=ledger.finish()
        write_json(root/'stats.json',summary)
        write_json(run/f'gpu_N{n}.json',dict(summary,identity=identity,resolution_identity=grid_identity))
    return summary


def validate_resolution(run,n,identity,*,baseline_run,baseline_identity):
    """Fail closed unless the complete real-A100 matrix has fresh receipts."""
    run=Path(run);baseline_run=Path(baseline_run);summary=json.loads((run/f'gpu_N{n}.json').read_text())
    if summary.get('method')!='polynomial' or summary.get('baseline_identity')!=baseline_identity or summary.get('baseline_run')!=str(baseline_run.resolve()) or summary.get('warm_repeats')!=7:raise ValueError('candidate/baseline configuration mismatch')
    if summary.get('identity')!=identity or summary.get('campaign_identity')!=identity or summary.get('n')!=n or summary.get('passed') is not True or summary.get('test_only') is not False:
        raise ValueError('GPU resolution campaign/pass/backend mismatch')
    if summary.get('matrix_units')!=352 or len(summary.get('records',[]))!=352:
        raise ValueError('GPU full352-unit matrix incomplete')
    inventory=summary['inventory']
    if inventory.get('test_only_cpu_emulation') or len(inventory['devices'])!=4 or any(d['platform']!='gpu' or 'A100' not in d['kind'].upper() for d in inventory['devices']):
        raise ValueError('GPU real four-A100 receipt missing')
    expected={(span,kind,case,count) for span in (1/16,1/32) for kind in range(4) for case in range(22) for count in (1,4)}
    compilers=CompilerReceipts(run)
    actual=set();no=summary['preflight']['owner_count'];nr=n**3
    if set(summary['inputs'])!={'state.npy','phi.npy','raw_to_owner.npy'}:raise ValueError('GPU input receipt keys mismatch')
    for name in summary['records']:
        if summary.get('record_hashes',{}).get(name)!=sha256_file(run/name):raise ValueError('GPU record content receipt mismatch')
        r=json.loads((run/name).read_text())
        if not _valid_record(run/name,summary['resolution_identity'],no,nr) or r.get('test_only') is not False or r.get('campaign_identity')!=identity or r.get('n')!=n:raise ValueError('GPU unit identity/backend mismatch')
        if not np.isfinite(r['metrics']['max_scaled_error']) or r['metrics']['max_scaled_error']>1:raise ValueError('GPU replay budget receipt failed')
        validate_timing_record(run,r,compilers=compilers)
        actual.add((r['span'],r['kind'],r['case'],r['devices']))
    if actual!=expected:raise ValueError('GPU full matrix units mismatch')
    replay=summary.get('boundary_catalogue_replay',{})
    if replay.get('passed') is not True or replay.get('cases')!=22 or replay.get('all_arrays_bitwise') is not True or replay.get('nonwall_padding_checked') is not True:
        raise ValueError('optimized boundary catalogue replay missing')
    if summary.get('record_timing_totals')!=record_totals([json.loads((run/name).read_text()) for name in summary['records']]):
        raise ValueError('shared timing reduction mismatch')
    audits=summary.get('cpu_merge_audits',[])
    if len(audits)!=8 or len(set(audits))!=8:raise ValueError('CPU merged replay audits incomplete')
    audit_units=set()
    for name in audits:
        if summary.get('audit_hashes',{}).get(name)!=sha256_file(run/name) or not _valid_record(run/name,summary['resolution_identity'],no,nr):raise ValueError('CPU merge audit content mismatch')
        a=json.loads((run/name).read_text())
        if a.get('merged_identity')!=summary['merged_identity'] or a.get('case')!=1:raise ValueError('CPU merge audit bank/nonconstant case mismatch')
        audit_units.add((a['span'],a['kind']))
    if audit_units!={(s,k) for s in (1/16,1/32) for k in range(4)}:raise ValueError('CPU merge audit units mismatch')
    for name,receipt in summary['inputs'].items():
        if sha256_file(baseline_run/'data'/f'N{n}'/name)!=receipt:raise ValueError('GPU input content changed')
    current=checked_chunks(baseline_run,n,baseline_identity)
    if summary.get('cpu_file_receipts')!=[dict(index=r['index'],files=r['files']) for _,r in current]:raise ValueError('GPU CPU source files changed')
    from scripts.q08_extraction_global import common as common_module
    source_paths={'gpu_stage.py':__file__,'common':common_module.__file__}
    for name in ('boundary_cache.py','boundary_values.py','timing.py','literal_reference.py'):
        source_paths[name]=Path(__file__).with_name(name)
    for module in ('drbx.native.q_plan','drbx.native.q_sharding','drbx.native.q_parallel_material','drbx.native.q_parallel_characteristic','drbx.native.q_characteristic_polynomial','drbx.stencils.q_plan'):
        source_paths[module]=__import__(module,fromlist=['__file__']).__file__
    if summary['sources']!={k:sha256_file(p) for k,p in source_paths.items()}:raise ValueError('GPU runtime source content changed')
    return summary
