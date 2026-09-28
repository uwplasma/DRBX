"""Bounded complete-owner numerical Q preparation; no automatic global dispatch.

This research runner composes immutable eight-family fits into compact N/E rows.
P/R is reused from the separately verified exact screen. Every owner batch is
atomic and resumable. A caller must name owner IDs explicitly.
"""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,io,json,multiprocessing,os,resource,shutil,sys,time,zipfile
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_ENABLE_X64']='true'
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
# JAX and DRBX read cache configuration at import time, before worker init.
for flag in ('--campaign','--output'):
    if flag in sys.argv:
        _cache_root=Path(sys.argv[sys.argv.index(flag)+1])/'cache'
        _cache_root.mkdir(parents=True,exist_ok=True)
        os.environ['DRBX_CACHE_DIR']=str(_cache_root)
        os.environ['JAX_COMPILATION_CACHE_DIR']=str(_cache_root)
        break
import numpy as np
from scipy import sparse
from . import qcommon,candidate,wall_fit,wall_score,exact_screen as exact
from .selection import select_interior,select_wall,prepare_target_geometry,FAMILIES
from .compact import compose,apply,normal_contravariant,raw_owner_states,unpack_interior
from .fields import callable_field
from .span_contract import make_seeds,LEG_FACTORS
from .trace import trace_padded
from scripts.q_fci_return_campaign import numerics as qnum
from drbx.geometry.hsx_jax_field import JaxHsxMagneticField

HERE=Path(__file__).resolve().parent;CONFIG=HERE.parent/'q_fci_projected_campaign/configuration.json';MANIFEST=HERE.parent/'q_fci_return_campaign/input_manifest.json'
FIELDS=tuple(json.loads((HERE/'catalogue.json').read_text())['fields'])
PAIRS=json.loads((HERE/'catalogue.json').read_text())['boundary_pairs']
CTX=INDEX=STATE=TRACER=None;OUT=DESIGN=CACHE=MAPZIP=SCREEN=CHECK_REUSE=None;N=None;TRACE_FIRST=True
TRACE_CACHE=TRACE_STORE_SHA=None
def sha(p):return exact.sha(p)
def write(path,obj):exact.write_json(path,obj)
def peak_rss_gib():
    """getrusage uses bytes on macOS and KiB on Linux (including Perlmutter)."""
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak/(1024**3 if sys.platform=='darwin' else 1024**2)
def source_paths():
    root=HERE.parents[1]
    return [HERE/x for x in ('numerical_runner.py','exact_screen.py','qcommon.py','candidate.py','wall_fit.py','wall_score.py','selection.py','compact.py','trace.py','trace_store.py','gpu_trace.py','gpu_preflight.py','fields.py','span_contract.py','catalogue.json','reuse.py','campaign.py','reduce_numerical.py')]+[CONFIG,HERE.parent/'q_fci_return_campaign/numerics.py',HERE.parent/'q_fci_return_campaign/endpoints.py',HERE.parent/'q03_direct_campaign/frozen_mms.py']+[root/'src/drbx/geometry'/x for x in ('MetricEvaluator.py','Bfield_evaluator.py','hsx_jax_field.py','jax_metric_evaluator.py','jax_bfield_evaluator.py','fci_boundary_functional_reconstruction.py','solve_MMPDE.py')]+[root/'src/drbx/_host_guards.py']
TRACE_REUSE_SOURCES=('trace.py','span_contract.py','hsx_jax_field.py','jax_metric_evaluator.py','jax_bfield_evaluator.py','MetricEvaluator.py','Bfield_evaluator.py','numerics.py','endpoints.py','configuration.json')
def freeze_check_reuse(old_output,output,manifest,checks):
    old_output=Path(old_output);old=json.loads((old_output/'design.json').read_text())
    if old['input_manifest']!=manifest or old['rk4_512_raw_ids']!=checks:raise RuntimeError('RK4 check reuse input/sample differs')
    now={str(p.relative_to(HERE.parents[1])):sha(p) for p in source_paths()}
    relevant={key.removeprefix('DRBX/'):digest for key,digest in old['sources'].items() if Path(key).name in TRACE_REUSE_SOURCES}
    if not relevant or any(now.get(key)!=digest for key,digest in relevant.items()):raise RuntimeError('RK4 check reuse trace dependency differs')
    records={str(n):{} for n in (32,48,64)};receipts=[]
    for n in (32,48,64):
        for path in sorted((old_output/f'N{n}').glob('owners_*.json')):
            r=json.loads(path.read_text())
            if r['design_sha256']!=sha(old_output/'design.json'):raise RuntimeError('RK4 source receipt identity differs')
            for check in r['rk4_checks']:
                raw=str(check['raw'])
                if raw in records[str(n)] or not check['valid'] or check['reentry'] or check['max_normal_cell_reach']>2:raise RuntimeError('invalid or duplicate reused RK4 check')
                records[str(n)][raw]=check
            receipts.append(dict(path=str(path.relative_to(old_output)),sha256=sha(path)))
        if sorted(map(int,records[str(n)]))!=checks[str(n)]:raise RuntimeError('RK4 check reuse coverage differs')
    path=Path(output)/'rk4_512_reuse.json'
    write(path,dict(schema='q-rk4-512-reuse-v1',source_design_sha256=sha(old_output/'design.json'),trace_source_sha256=relevant,input_manifest=manifest,checks=records,source_receipts=receipts))
    return path.name
def freeze(input_root,output,screen,preflight=None,force_fresh_raw=(),rk4_check_cache=None,trace_mode='cpu_inline',rk4_steps=256):
    if trace_mode not in ('cpu_inline','gpu_cache') or rk4_steps not in (64,256):raise ValueError('unsupported trace mode or RK4 steps')
    if rk4_check_cache and (trace_mode!='cpu_inline' or rk4_steps!=256):raise ValueError('legacy RK4 checks require CPU-inline RK4-256')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(MANIFEST.read_text())
    for x in manifest['files']:
        p=Path(input_root)/x['path']
        if not p.is_file() or p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise RuntimeError(f'wrong immutable input {p}')
    screen=Path(screen);screen_design=screen/'design.json';assert screen_design.is_file()
    cache=None;sample={'roles':[]};representative_owners={str(n):[] for n in (32,48,64)}
    if preflight:
        preflight=Path(preflight);names=('sample.json','pilot_prepare.json','full_prepare.json','pilot_traces.npz','full_traces.npz')
        sample=json.loads((preflight/'sample.json').read_text())
        for role in sample['roles']:
            key=str(role['N']);owner=int(role['owner'])
            if owner not in representative_owners[key]:representative_owners[key].append(owner)
        for key in representative_owners:representative_owners[key].sort()
        cache_dir=output/'cache_inputs';cache_dir.mkdir(exist_ok=True)
        for name in names:shutil.copy2(preflight/name,cache_dir/name)
        archives={};source_maps={}
        for n in (32,48,64):
            archive=output/f'map_cache_N{n}.zip';maps={}
            for stage in ('pilot','full'):
                prep=json.loads((preflight/f'{stage}_prepare.json').read_text())
                for rec in prep['records']:
                    if rec['N']!=n:continue
                    member=f"raw{rec['raw_id']}_{rec['kind']}.npz";p=Path(rec['map_ref']['path']);digest=sha(p)
                    if digest!=rec['map_ref']['sha256']:raise RuntimeError(f'cached map identity differs: {p}')
                    maps[member]=p;source_maps[f'N{n}/{member}']=digest
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
                for member,p in sorted(maps.items()):z.write(p,arcname=member)
            archives[str(n)]=dict(name=archive.name,sha256=sha(archive),members=len(maps))
        cache=dict(source_root=str(preflight.resolve()),inputs={p:sha(cache_dir/p) for p in names},archives=archives,source_map_sha256=source_maps)
    from .reuse import freeze_reuse
    reuse=freeze_reuse(screen,output/'exact_reuse_manifest.json')
    checks={str(n):sorted({int(role['members'][0]['raw_id']) for role in sample['roles'] if role['N']==n}) if preflight else [] for n in (32,48,64)}
    check_reuse_name=freeze_check_reuse(rk4_check_cache,output,manifest,checks) if rk4_check_cache else None
    design=dict(schema='q-shortspan-numerical-v2',input_manifest=manifest,exact_design_sha256=sha(screen_design),exact_reuse_sha256=sha(output/'exact_reuse_manifest.json'),preflight_cache=cache,force_fresh_raw_ids=sorted(set(map(int,force_fresh_raw))),representative_owners=representative_owners,rk4_512_raw_ids=checks,rk4_512_reuse_sha256=sha(output/check_reuse_name) if check_reuse_name else None,
                trace_mode=trace_mode,method=dict(outer_spans=('h/8','h/4'),inner_eta_factors=tuple(map(float,LEG_FACTORS)),rk4_steps=rk4_steps,float64=True,wall_band='i>=N-7',families=FAMILIES,selection='strict lexicographic minimum A',owner_batch_max=64,fields=FIELDS),
                sources={str(p.relative_to(HERE.parents[1])):sha(p) for p in source_paths()})
    path=output/'design.json'
    if path.exists() and json.loads(path.read_text())!=design:raise RuntimeError('incompatible numerical output identity')
    write(path,design)
def validate(input_root,output,screen):
    output=Path(output);d=json.loads((output/'design.json').read_text());assert d['schema']=='q-shortspan-numerical-v2'
    if d.get('trace_mode','cpu_inline') not in ('cpu_inline','gpu_cache') or d['method']['rk4_steps'] not in (64,256):raise RuntimeError('unsupported trace mode or RK4 steps')
    if d['rk4_512_reuse_sha256'] and (d.get('trace_mode','cpu_inline')!='cpu_inline' or d['method']['rk4_steps']!=256):raise RuntimeError('incompatible legacy RK4 check reuse')
    expected=dict(outer_spans=['h/8','h/4'],inner_eta_factors=list(map(float,LEG_FACTORS)),rk4_steps=d['method']['rk4_steps'],float64=True,wall_band='i>=N-7',families=list(FAMILIES),selection='strict lexicographic minimum A',owner_batch_max=64,fields=list(FIELDS))
    if d['method']!=expected:raise RuntimeError('numerical method identity differs')
    if set(d['representative_owners'])!={'32','48','64'} or any(not isinstance(v,list) or v!=sorted(set(v)) for v in d['representative_owners'].values()):raise RuntimeError('representative subset differs')
    if d['sources']!={str(p.relative_to(HERE.parents[1])):sha(p) for p in source_paths()}:raise RuntimeError('numerical source differs')
    if sha(Path(screen)/'design.json')!=d['exact_design_sha256']:raise RuntimeError('exact screen identity differs')
    if sha(output/'exact_reuse_manifest.json')!=d['exact_reuse_sha256']:raise RuntimeError('exact reuse manifest differs')
    if d['rk4_512_reuse_sha256']:
        p=output/'rk4_512_reuse.json'
        if sha(p)!=d['rk4_512_reuse_sha256']:raise RuntimeError('RK4 check reuse manifest differs')
        m=json.loads(p.read_text());now={str(p.relative_to(HERE.parents[1])):sha(p) for p in source_paths()}
        if m['schema']!='q-rk4-512-reuse-v1' or m['input_manifest']!=d['input_manifest'] or any(now.get(k)!=v for k,v in m['trace_source_sha256'].items()) or any(sorted(map(int,m['checks'][str(n)]))!=d['rk4_512_raw_ids'][str(n)] for n in (32,48,64)):raise RuntimeError('RK4 check reuse identity differs')
    from .reuse import validate_reuse
    validate_reuse(screen,output/'exact_reuse_manifest.json')
    for x in d['input_manifest']['files']:
        p=Path(input_root)/x['path']
        if not p.is_file() or p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise RuntimeError(f'input hash differs {p}')
    if d['preflight_cache']:
        c=d['preflight_cache'];root=output/'cache_inputs'
        for name,digest in c['inputs'].items():
            if sha(root/name)!=digest:raise RuntimeError(f'preflight cache differs {name}')
        for archive in c['archives'].values():
            if sha(output/archive['name'])!=archive['sha256']:raise RuntimeError('map cache archive differs')
    return d
def init_worker(n,input_root,output,screen):
    global CTX,INDEX,STATE,TRACER,OUT,DESIGN,CACHE,MAPZIP,SCREEN,CHECK_REUSE,N,TRACE_FIRST
    global TRACE_CACHE,TRACE_STORE_SHA
    N=n;OUT=Path(output);DESIGN=json.loads((OUT/'design.json').read_text());TRACE_FIRST=True
    SCREEN=Path(screen)
    CHECK_REUSE=json.loads((OUT/'rk4_512_reuse.json').read_text())['checks'][str(n)] if DESIGN['rk4_512_reuse_sha256'] else {}
    sys.dont_write_bytecode=True
    cache=Path(os.environ.get('DRBX_CACHE_DIR',OUT/'cache'));cache.mkdir(parents=True,exist_ok=True)
    os.environ['DRBX_CACHE_DIR']=str(cache);os.environ['JAX_COMPILATION_CACHE_DIR']=str(cache)
    import jax
    if jax.default_backend()!='cpu':raise RuntimeError('Q numerical backend is not CPU')
    CTX=qnum.context(n,input_root,json.loads(CONFIG.read_text()));INDEX=qcommon.LocalOwnerMoments(CTX)
    TRACE_CACHE=TRACE_STORE_SHA=TRACER=None;MAPZIP=None
    if DESIGN.get('trace_mode','cpu_inline')=='gpu_cache':
        from .trace_store import TraceStore
        TRACE_CACHE=TraceStore(OUT,n);TRACE_STORE_SHA=sha(OUT/'traces'/f'complete_N{n}.json')
    else:TRACER=JaxHsxMagneticField.from_evaluators(CTX['evaluator'],CTX['bfield'])
    STATE=raw_owner_states(CTX,FIELDS,callable_field);CACHE={}
    if DESIGN['preflight_cache'] and TRACE_CACHE is None and DESIGN['method']['rk4_steps']==256:
        root=OUT/'cache_inputs';MAPZIP=zipfile.ZipFile(OUT/DESIGN['preflight_cache']['archives'][str(n)]['name'],'r')
        for stage in ('pilot','full'):
            prep=json.loads((root/f'{stage}_prepare.json').read_text())
            with np.load(root/f'{stage}_traces.npz') as z:seeds=z['seeds'].copy();ends=z['endpoints'].copy()
            for rec in prep['records']:
                if rec['N']==n:CACHE[rec['raw_id'],rec['kind']]=(rec,seeds[rec['trace_index']],ends[rec['trace_index']])
def raw_geometry(raw):
    n=CTX['N'];i,j,k=np.unravel_index(raw,(n,n,n));grid=CTX['artifact'].geometry.grid;axes=[getattr(grid,x) for x in 'xyz']
    q=np.array([a.centers[v] for a,v in zip(axes,(i,j,k))]);w=np.array([a.faces[v+1]-a.faces[v] for a,v in zip(axes,(i,j,k))]);return (i,j,k),q,w
def seeds_for(q,w,iswall):
    seeds,_=make_seeds(q,w)
    return seeds if iswall else seeds[np.r_[6:12,0:6]]
def exact_owner(owner):
    p=exact.chunk_path(SCREEN,N,owner//exact.OWNER_BATCH)
    r=json.loads(p.with_suffix('.json').read_text());assert r['sha256']==sha(p) and r['design_sha256']==DESIGN['exact_design_sha256']
    with np.load(p) as z:
        idx=int(owner-z['owner'][0]);assert z['owner'][idx]==owner
        return z['P'][idx].copy(),z['R'][idx].copy(),z['R_half'][idx].copy()
def fit(q,ijk,seeds,ends,owner,kind):
    start=time.process_time();prepare_target_geometry(CTX,seeds,q,wall=(kind!='interior'));geometry_cpu=time.process_time()-start
    admitted={};failed={};scores={}
    start=time.process_time()
    for family in FAMILIES:
        if kind=='interior':
            if family.startswith('K'):M,planes,fail,score=candidate.full_candidate(CTX,INDEX,q,seeds,ends,owner,family)
            else:
                M,planes,fail,score=candidate.candidate(CTX,INDEX,q,seeds,ends,owner,ijk,family)
                fail=list(fail)+[dict(plane=x['plane'],reason='direct_reproduction_'+str(x['direct_monomial_reproduction'])) for x in planes if x.get('direct_reproduction_exceeds_frozen_gate')]
            if fail or M is None:failed[family]=fail;continue
            admitted[family]=M;scores[family]=score
        else:
            obj,meta=wall_fit.prepare(CTX,INDEX,q,ijk,ends.reshape(48,3),family,kind,normalize_neumann=True)
            if obj is None or meta['failure']:failed[family]=meta['failure'];continue
            score=wall_score.score(CTX,q,seeds,obj,kind)
            if score['max_staged_defect']>1e-8 or score['max_constant_action']>1e-9:failed[family]=['action_or_constant_gate'];continue
            cells=np.concatenate([INDEX.members(int(x)) for x in obj['donors']]);xy=np.column_stack((INDEX.x[cells],INDEX.y[cells]));radius=float(np.max(np.linalg.norm((xy-obj['anchor'][:2])/obj['scale'],axis=1)))
            score.update(radius=radius,donor_count=meta['donor_count'],max_condition=meta['max_condition'],basis_reproduction=meta['basis_reproduction'],plane_counts=meta['plane_counts'])
            admitted[family]=obj;scores[family]=score
    factor_cpu=time.process_time()-start;start=time.process_time()
    choice=select_interior(scores) if kind=='interior' else select_wall(scores)
    select_cpu=time.process_time()-start
    if choice is None:raise RuntimeError(f'no admitted map for {ijk} {kind}: {failed}')
    chosen=admitted[choice]
    if kind!='interior':chosen=dict(chosen,U=chosen['prepared'].owner_map,V=chosen['prepared'].boundary_map)
    return chosen,choice,scores,failed,dict(geometry=geometry_cpu,moments_factorization=factor_cpu,selection=select_cpu)
def trace_target(seeds):
    global TRACE_FIRST
    H=2*np.pi/N;delta=np.repeat(LEG_FACTORS*H,12);t=time.process_time();out=trace_padded(TRACER,np.tile(seeds,(4,1)),delta,DESIGN['method']['rk4_steps']);elapsed=time.process_time()-t;first=TRACE_FIRST;TRACE_FIRST=False
    if not np.all(out[1]) or np.any(out[3]) or np.max((out[4]-1).clip(min=0)*N)>2:raise RuntimeError('frozen trace validity/reentry/reach gate')
    return np.asarray(out[0]).reshape(4,12,3),out,elapsed,first
def boundary_rows(points,data,kind):
    val,dx,dy,dz=wall_fit.basis(points,data['anchor'],float(data['scale']),float(data['H']))
    if kind=='D':return val
    normals=normal_contravariant(CTX,points);u,t=points[:,0],points[:,1]
    du=dx*np.cos(t)[:,None]+dy*np.sin(t)[:,None]
    dt=dx*(-u*np.sin(t))[:,None]+dy*(u*np.cos(t))[:,None]
    return normals[:,0,None]*du+normals[:,1,None]*dt+normals[:,2,None]*dz
def independent_wall_points(q,end):
    dt=(end[:,1]-q[1]+np.pi)%(2*np.pi)-np.pi;de=end[:,2]-q[2]
    a,b=float(dt.min()),float(dt.max());c,d=float(de.min()),float(de.max())
    return np.array([[1.,q[1]+a+f*(b-a),q[2]+c+g*(d-c)] for f in (.1,.5,.9) for g in (.1,.5,.9)])
def chunk_path(output,n,owners):return Path(output)/f'N{n}'/f'owners_{owners[0]:06d}_{owners[-1]:06d}.npz'
def completed(path,designhash,owners):
    receipt=path.with_suffix('.json')
    if not path.exists() and not receipt.exists():return False
    if not path.exists() or not receipt.exists():raise RuntimeError(f'interrupted numerical chunk: {path}')
    try:
        r=json.loads(receipt.read_text())
        if r['design_sha256']!=designhash or r['owners']!=owners or r['sha256']!=sha(path):raise RuntimeError(f'incompatible or corrupt numerical chunk: {path}')
        if r.get('trace_inventory_sha256') and r['trace_inventory_sha256']!=sha(path.parent.parent/'traces'/f"complete_N{r['N']}.json"):raise RuntimeError('numerical chunk trace inventory changed')
        return True
    except (OSError,ValueError,KeyError) as exc:raise RuntimeError(f'corrupt numerical chunk: {path}') from exc
def _run_chunk_unlocked(job):
    owners,designhash=job;path=chunk_path(OUT,N,owners);path.parent.mkdir(parents=True,exist_ok=True)
    if completed(path,designhash,owners):return dict(skipped=True,cpu_s=0)
    t0=time.process_time();times={k:0. for k in ('trace_compile_execute','trace_warm','trace_cache_io','rk4_512_check','moments_factorization','selection','geometry','field','application','diagnostics','io')}
    cases=[]
    for name in FIELDS:
        for bc in PAIRS.get(name,PAIRS['waves']):cases.append((name,bc))
    # Every owner is complete; one value/unknown per owner.
    Nout=np.zeros((len(owners),len(cases),2,3),complex);Eout=np.zeros_like(Nout);Pout=np.zeros((len(owners),len(exact.FIELDS),2,3),complex);Rout=np.zeros((len(owners),len(exact.FIELDS),3),complex);Rhout=np.zeros_like(Rout)
    raw_counts=[];volumes=[];mapmeta=[];compactarrays={};representatives={};max_endpoint=max_bc=max_independent_bc=max_crossing_bc=max_512=0.;trace_checks=[]
    frozen_checks=set(DESIGN['rk4_512_raw_ids'][str(N)])
    for oi,owner in enumerate(owners):
        members=INDEX.members(owner);weights=INDEX.volume[members]/CTX['volume'][owner]
        assert abs(float(sum(weights))-1)<2e-12 and len(members)>0
        raw_counts.append(len(members));volumes.append(float(CTX['volume'][owner]));Pout[oi],Rout[oi],Rhout[oi]=exact_owner(owner)
        for raw,weight in zip(members,weights):
            INDEX.begin_target();ijk,q,w=raw_geometry(int(raw));iswall=ijk[0]>=N-7;seeds=seeds_for(q,w,iswall)
            kinds=('D','N') if iswall else ('interior',)
            cache=CACHE.get((int(raw),kinds[0])) if int(raw) not in DESIGN['force_fresh_raw_ids'] else None;use_cache=False
            gpu_check=None
            if TRACE_CACHE is not None:
                start=time.process_time();traceinfo,gpu_check=TRACE_CACHE.get(int(raw),seeds);times['trace_cache_io']+=time.process_time()-start
                ends=traceinfo[0].reshape(4,12,3)
            elif cache is not None and np.array_equal(cache[1],seeds):
                ends=cache[2];traceinfo=None;use_cache=True
                if int(raw) in frozen_checks:
                    replay,traceinfo,spent,first=trace_target(seeds);times['trace_compile_execute' if first else 'trace_warm']+=spent
                    if np.max(abs(replay-ends))>1e-11:raise RuntimeError('frozen cached 256 endpoint differs')
            else:
                ends,traceinfo,spent,first=trace_target(seeds);times['trace_compile_execute' if first else 'trace_warm']+=spent
            if int(raw) in frozen_checks:
                if CHECK_REUSE:
                    saved=CHECK_REUSE[str(int(raw))]
                    if saved['owner']!=int(owner):raise RuntimeError('reused RK4 owner differs')
                    trace_checks.append(dict(saved,reused=True));max_512=max(max_512,saved['endpoint_max'])
                else:
                    if TRACE_CACHE is not None:
                        if gpu_check is None:raise RuntimeError('missing frozen GPU RK4-512 check')
                        check=gpu_check
                    else:
                        start=time.process_time();check=trace_padded(TRACER,np.tile(seeds,(4,1)),np.repeat(LEG_FACTORS*2*np.pi/N,12),512);times['rk4_512_check']+=time.process_time()-start
                    delta=np.asarray(check[0]).reshape(4,12,3)-ends
                    sensitivity=float(np.max(abs(delta)))
                    max_512=max(max_512,sensitivity);trace_checks.append(dict(owner=int(owner),raw=int(raw),endpoint_max=sensitivity,endpoint_signed_min=float(np.min(delta)),endpoint_signed_max=float(np.max(delta)),valid=bool(np.all(check[1])),reentry=bool(np.any(check[3])),max_normal_cell_reach=float(np.max((check[4]-1).clip(min=0)*N)),first_crossings=int(np.sum(check[2])),reused=False))
                    if not np.all(check[1]) or np.any(check[3]) or np.max((check[4]-1).clip(min=0)*N)>2:raise RuntimeError('RK4-512 geometry check failed')
            for kind in kinds:
                entry=CACHE.get((int(raw),kind)) if use_cache else None
                if entry is not None and np.array_equal(entry[1],seeds):
                    rec=entry[0]
                    start=time.process_time();member=f"raw{int(raw)}_{kind}.npz"
                    payload=MAPZIP.read(member)
                    if hashlib.sha256(payload).hexdigest()!=rec['map_ref']['sha256']:raise RuntimeError('cached map content differs')
                    with np.load(io.BytesIO(payload)) as z:data={k:z[k].copy() for k in z.files}
                    times['io']+=time.process_time()-start;choice=rec['selected'];score=rec['score'];failed=rec.get('failed',{})
                else:
                    data,choice,scores,failed,parts=fit(q,ijk,seeds,ends,owner,kind)
                    for key,value in parts.items():times[key]+=value
                    score=scores[choice]
                start=time.process_time();comp=compose(CTX,seeds,q,data,kind,owner);times['geometry']+=time.process_time()-start
                mapid=len(mapmeta);mapmeta.append(dict(owner=int(owner),raw=int(raw),kind=kind,selected=choice,score=score,failed=failed,source='cache' if entry is not None else 'fresh',crossings=int(np.sum(traceinfo[2])) if traceinfo is not None else None,first_crossing_points=np.asarray(traceinfo[8])[np.asarray(traceinfo[2])].tolist() if traceinfo is not None else None,max_normal_cell_reach=float(np.max((traceinfo[4]-1).clip(min=0)*N)) if traceinfo is not None else None,max_endpoint_error=0.,max_imposed_bc_residual=0.,max_independent_bc_residual=0.,max_first_crossing_bc_residual=0.))
                compactarrays[f'm{mapid}_donors']=comp['donors'];compactarrays[f'm{mapid}_A']=comp['A'];compactarrays[f'm{mapid}_B']=comp['B'];compactarrays[f'm{mapid}_wall_nodes']=comp['wall_nodes']
                if kind!='interior':
                    independent=independent_wall_points(q,ends.reshape(-1,3));indrows=boundary_rows(independent,data,kind)
                    crosspoints=np.asarray(traceinfo[8])[np.asarray(traceinfo[2])] if traceinfo is not None else np.empty((0,3));crossrows=boundary_rows(crosspoints,data,kind) if len(crosspoints) else None
                    wall_normals=normal_contravariant(CTX,comp['wall_nodes'])
                    independent_normals=normal_contravariant(CTX,independent)
                    crossing_normals=normal_contravariant(CTX,crosspoints) if len(crosspoints) else None
                if int(owner) in DESIGN['representative_owners'][str(N)] and int(raw)==int(members[0]):
                    if kind=='interior':representatives[f'm{mapid}_endpoint']=(data if sparse.issparse(data) else unpack_interior(data)).toarray()[:,comp['donors']]
                    else:
                        for key in ('M','C','Eowner','Ebc','target'):representatives[f'm{mapid}_{key}']=data[key]
                for ci,(name,bc) in enumerate(cases):
                    if kind!='interior' and bc!=kind:continue
                    field=callable_field(name);f0=STATE[name][owner];donor=STATE[name][comp['donors']]
                    start=time.process_time()
                    if kind=='interior':boundary=np.empty(0)
                    else:
                        v,g=field(comp['wall_nodes']);boundary=v if kind=='D' else np.einsum('ni,ni->n',wall_normals,g)
                    times['field']+=time.process_time()-start
                    start=time.process_time();Nval=apply(comp,donor,f0,boundary,kind);Eval=(comp['K']@field(ends.reshape(-1,3))[0]).reshape(2,3);times['application']+=time.process_time()-start
                    Nout[oi,ci]+=weight*Nval;Eout[oi,ci]+=weight*Eval
                    start=time.process_time()
                    exact_end=field(ends.reshape(-1,3))[0]
                    if kind=='interior':pred=(data if sparse.issparse(data) else unpack_interior(data))[:,comp['donors']]@donor
                    else:
                        rhs=boundary-f0 if kind=='D' else boundary;pred=data['target']@(data['U']@(donor-f0)+data['V']@rhs)+f0
                        coeff=data['U']@(donor-f0)+data['V']@rhs
                        imposed=float(np.max(abs(data['C']@coeff+(f0 if kind=='D' else 0)-boundary)))
                        max_bc=max(max_bc,imposed);mapmeta[-1]['max_imposed_bc_residual']=max(mapmeta[-1]['max_imposed_bc_residual'],imposed)
                        iv,ig=field(independent);itruth=iv if kind=='D' else np.einsum('ni,ni->n',independent_normals,ig)
                        idiff=float(np.max(abs(indrows@coeff+(f0 if kind=='D' else 0)-itruth)))
                        max_independent_bc=max(max_independent_bc,idiff);mapmeta[-1]['max_independent_bc_residual']=max(mapmeta[-1]['max_independent_bc_residual'],idiff)
                        if len(crosspoints):
                            cv,cg=field(crosspoints);ctruth=cv if kind=='D' else np.einsum('ni,ni->n',crossing_normals,cg)
                            cdiff=float(np.max(abs(crossrows@coeff+(f0 if kind=='D' else 0)-ctruth)))
                            max_crossing_bc=max(max_crossing_bc,cdiff);mapmeta[-1]['max_first_crossing_bc_residual']=max(mapmeta[-1]['max_first_crossing_bc_residual'],cdiff)
                    endpoint_error=float(np.max(abs(pred-exact_end)));max_endpoint=max(max_endpoint,endpoint_error);mapmeta[-1]['max_endpoint_error']=max(mapmeta[-1]['max_endpoint_error'],endpoint_error)
                    times['diagnostics']+=time.process_time()-start
    tmp=path.with_name(path.stem+'.tmp.npz');start=time.process_time();np.savez_compressed(tmp,owners=np.asarray(owners),raw_count=np.asarray(raw_counts),volume=np.asarray(volumes),N=Nout,E=Eout,P=Pout,R=Rout,R_half=Rhout,**compactarrays,**representatives);os.replace(tmp,path);times['io']+=time.process_time()-start
    import jax
    receipt=dict(schema='q-numerical-owner-chunk-v2',N=N,owners=owners,raw_count=int(sum(raw_counts)),map_count=len(mapmeta),cases=cases,source_design_sha256=designhash,design_sha256=designhash,sha256=sha(path),stage_cpu_s=times,cpu_s=time.process_time()-t0,maxrss_gib=peak_rss_gib(),cpu_backend=jax.default_backend(),affinity_cpus=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,thread_limits={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','JAX_PLATFORMS')},max_endpoint_error=max_endpoint,max_imposed_bc_residual=max_bc,max_independent_bc_residual=max_independent_bc,max_first_crossing_bc_residual=max_crossing_bc,max_RK4_256_to_512_endpoint=max_512,rk4_checks=trace_checks,map_meta=mapmeta)
    receipt.update(trace_mode=DESIGN.get('trace_mode','cpu_inline'),rk4_steps=DESIGN['method']['rk4_steps'],trace_inventory_sha256=TRACE_STORE_SHA,max_RK4_primary_to_512_endpoint=max_512)
    if DESIGN['method']['rk4_steps']!=256:receipt.pop('max_RK4_256_to_512_endpoint')
    write(path.with_suffix('.json'),receipt)
    return dict(skipped=False,cpu_s=receipt['cpu_s'],path=str(path))
def run_chunk(job):
    owners,designhash=job;path=chunk_path(OUT,N,owners);path.parent.mkdir(parents=True,exist_ok=True)
    if completed(path,designhash,owners):return dict(skipped=True,cpu_s=0)
    lock=path.with_suffix('.lock')
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc:raise RuntimeError(f'active or interrupted chunk writer: {lock}') from exc
    try:
        os.write(fd,f'{os.getpid()}\n'.encode());os.close(fd)
        if completed(path,designhash,owners):return dict(skipped=True,cpu_s=0)
        return _run_chunk_unlocked(job)
    finally:
        lock.unlink(missing_ok=True)
def worker_policy(workers,host_memory_gib,worker_memory_gib):
    if workers<1 or worker_memory_gib<=0 or host_memory_gib<=0:raise ValueError('positive workers and memory budgets required')
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    maximum=min(affinity,int(host_memory_gib//worker_memory_gib))
    if workers>maximum:raise ValueError(f'{workers} workers exceed affinity/memory maximum {maximum} ({affinity} CPUs, {host_memory_gib} GiB host, {worker_memory_gib} GiB/worker)')
    return dict(requested_workers=workers,effective_workers=workers,affinity_cpu_count=affinity,host_memory_gib=host_memory_gib,worker_memory_gib=worker_memory_gib)
def run_jobs(n,owner_groups,input_root,output,screen,workers,host_memory_gib,worker_memory_gib):
    policy=worker_policy(workers,host_memory_gib,worker_memory_gib)
    design=validate(input_root,output,screen);dh=sha(Path(output)/'design.json')
    flat=[int(owner) for group in owner_groups for owner in group]
    expected={32:25376,48:86016,64:202304}[n]
    if not flat or len(set(flat))!=len(flat) or min(flat)<0 or max(flat)>=expected:raise ValueError('unique valid complete owner IDs required')
    if any(not group or len(group)>design['method']['owner_batch_max'] or list(group)!=sorted(group) for group in owner_groups):raise ValueError('invalid stable complete-owner chunk')
    if design.get('trace_mode')=='gpu_cache':
        from .trace_store import TraceStore
        from .gpu_trace import load_grid
        _,_,labels=load_grid(input_root,n,json.loads(CONFIG.read_text()))
        TraceStore(output,n).require_raw_ids(np.flatnonzero(np.isin(labels,flat)))
    jobs=[(list(group),dh) for group in owner_groups]
    if workers==1:
        init_worker(n,input_root,output,screen);done=[run_chunk(x) for x in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(n,input_root,output,screen)) as pool:done=list(pool.map(run_chunk,jobs))
    receipt=dict(N=n,owners=len(flat),first_owner=min(flat),last_owner=max(flat),chunks=len(jobs),computed=sum(not x['skipped'] for x in done),chunk_cpu_s=sum(x['cpu_s'] for x in done),design_sha256=dh,worker_policy=policy)
    write(Path(output)/f'N{n}/run_{min(flat):06d}_{max(flat):06d}_{len(flat)}.json',receipt)
    print(json.dumps(receipt));return receipt
def run(n,owners,input_root,output,screen,workers,host_memory_gib=6.,worker_memory_gib=3.):
    if not owners or len(set(owners))!=len(owners):raise ValueError('explicit unique owner IDs required')
    batch=64;sorted_owners=sorted(owners)
    return run_jobs(n,[sorted_owners[j:j+batch] for j in range(0,len(sorted_owners),batch)],input_root,output,screen,workers,host_memory_gib,worker_memory_gib)
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','run'));p.add_argument('--input-root',required=True);p.add_argument('--output',required=True);p.add_argument('--exact-screen',required=True);p.add_argument('--preflight-cache');p.add_argument('--force-fresh-raw',type=int,nargs='*',default=[]);p.add_argument('--rk4-check-cache');p.add_argument('--N',type=int,choices=(32,48,64));p.add_argument('--owners',type=int,nargs='*');p.add_argument('--workers',type=int,default=1);p.add_argument('--host-memory-gib',type=float,default=6.);p.add_argument('--worker-memory-gib',type=float,default=3.);a=p.parse_args()
    if a.command=='freeze':freeze(a.input_root,a.output,a.exact_screen,a.preflight_cache,a.force_fresh_raw,a.rk4_check_cache)
    else:
        if a.N is None:raise ValueError('--N required')
        run(a.N,a.owners,a.input_root,a.output,a.exact_screen,a.workers,a.host_memory_gib,a.worker_memory_gib)
if __name__=='__main__':main()
