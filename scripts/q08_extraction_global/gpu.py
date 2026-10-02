"""Content-checked Q08 CPU-to-A100 implementation replay, one state at a time.

No tracing, references, support tuning or production selection occurs here.
The campaign-only merged view preserves literal checked CPU chunk identities.
Full dense merge/staging is refused when its declared conservative resource
estimate exceeds the supplied host budget or observed GPU free memory.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import gc,json,os,time,zipfile,csv,subprocess,fcntl,resource
from contextlib import contextmanager
import numpy as np
import jax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from .common import sha as sha256_file,digest,atomic_json as write_json
from drbx.stencils.q_artifact import load_q_bank
from drbx.stencils.q_bank import QBank,content_identity,SCHEMA
from drbx.stencils.q_parallel import sha256_array
from drbx.stencils.q_plan import lower_q_plan
from drbx.stencils.query_tables import ExactQueryTable
from drbx.native.q_plan import apply_q_plan
from drbx.native.q_sharding import shard_q_plan,sharded_q_rhs

GIB=1<<30
MERGED_SCHEMA='drbx.q08-campaign-merged.v1'
GEOMETRY=('magnetic_L','b_eta','eta_step','bmag')
RAW=('raw','raw_to_owner','raw_weight','donor','mask','row_count','row_value_D','magnetic_b')
WALL=('row_value_N_wall','boundary_value_D_trace','boundary_value_N_normal','boundary_wall_normal')
SPAN_RAW=('diffusion_D','magnetic_L')
SPAN_WALL=('diffusion_N_wall','boundary_D_node','boundary_D_tangent','boundary_N_normal')
PAD_DONOR=('donor','mask','row_value_D','row_value_N_wall','diffusion_D','diffusion_N_wall')
MATCH_KEYS=('campaign_identity','geometry_identity','topology_hash','support','n','n_owner','trace_steps','trace_method')


@contextmanager
def lock(root):
    with (Path(root)/"stage.lock").open("a") as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GPU resolution already running")
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def peak_rss_gib():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(GIB if os.uname().sysname=="Darwin" else GIB/1024)


def _common(api=None):
    if api is not None:return api
    from . import common
    return common


def _headers(path):
    """NPY headers only, including decoded manifest bytes, before dense loading."""
    out={}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):raise ValueError('unexpected NPZ member')
            with archive.open(name) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(stream)
                else:raise ValueError('unsupported NPZ header version')
            out[name[:-4]]=dict(shape=tuple(shape),dtype=np.dtype(dtype),bytes=int(np.prod(shape,dtype=np.int64))*dtype.itemsize)
    return out


def checked_chunks(run,n,identity):
    root=Path(run)/'cpu'/f'N{n}'
    paths=sorted(root.glob('chunk_[0-9][0-9][0-9][0-9][0-9][0-9]'))
    if not paths:raise ValueError(f'no completed CPU chunks for N{n}')
    result=[]
    for index,path in enumerate(paths):
        if path.name!=f'chunk_{index:06d}':raise ValueError('CPU chunk sequence incomplete')
        receipt=json.loads((path/'stats.json').read_text())
        if not receipt.get('passed') or receipt.get('n')!=n or receipt.get('index')!=index or receipt.get('campaign_identity')!=identity:
            raise ValueError('CPU chunk campaign/pass identity mismatch')
        files=receipt.get('files',{})
        for name in ('bank.npz','geometry.npz'):
            if files.get(name)!=sha256_file(path/name):raise ValueError(f'CPU {name} content receipt mismatch')
        result.append((path,receipt))
    return result


def estimate_merge(run,n,identity,host_memory_gib):
    """Conservative declared bound, not an RSS/device measurement or forecast."""
    if not np.isfinite(host_memory_gib) or host_memory_gib<=0:raise ValueError('positive host memory budget required')
    chunks=checked_chunks(run,n,identity)
    headers=[_headers(path/'bank.npz') for path,_ in chunks]
    nr=sum(h['arrays__raw']['shape'][0] for h in headers)
    no=sum(h['arrays__owners']['shape'][0] for h in headers)
    nw=sum(h['arrays__wall_index']['shape'][0] for h in headers)
    nd=max(h['arrays__donor']['shape'][1] for h in headers)
    nm=max(h['arrays__owner_raw']['shape'][1] for h in headers)
    logical=sum(v['bytes'] for h in headers for k,v in h.items() if k!='manifest')
    manifests=sum(h['manifest']['bytes'] for h in headers)
    # Worst donor width and owner-member width; exact query upper bound retains
    # every input query. This bound includes optional diagnostic gradients.
    padded=0
    for name,entry in headers[0].items():
        if not name.startswith(('arrays__','diagnostics__')):continue
        key=name.split('__',1)[1];shape=list(entry['shape'])
        if key in ('owners','owner_raw','owner_weight'):shape[0]=no
        elif key=='query_table':shape[0]=sum(h[name]['shape'][0] for h in headers)
        elif key in SPAN_RAW:shape[1]=nr
        elif key in SPAN_WALL:shape[1]=nw
        elif key in WALL or key in ('wall_index','wall_node_query','wall_slot_query'):shape[0]=nw
        else:shape[0]=nr
        if key in PAD_DONOR or key in ('row_gradient_D','row_gradient_N'):shape[-1]=nd
        if key in ('owner_raw','owner_weight'):shape[-1]=nm
        padded+=int(np.prod(shape,dtype=np.int64))*entry['dtype'].itemsize
    # Live dense legacy BCs: two 6-field span tables plus scalar phi tables.
    boundary=nr*(2*6+1)*(35+3+6+35)*8
    state=no*7*8
    reference=(no*4*6+nr*16)*8
    # Host: merged arrays + both stacked variants + CPU views/query receipts +
    # one chunk decoder + live one-case BC/ref + library/metadata safety margin.
    host=5*padded+2*logical+6*manifests+3*boundary+2*reference+3*state
    # GPU0 may hold monolithic and a quarter sharded plan concurrently, plus
    # replicated global BC inputs, outputs and compilation/work temporaries.
    device=3*(2*padded+2*boundary+reference+state)
    report=dict(n=n,chunks=len(chunks),raw_count=nr,owner_count=no,wall_count=nw,
        max_donors=nd,max_members=nm,input_decoded_array_bytes=logical,
        decoded_manifest_bytes=manifests,merged_padded_upper_bytes=padded,
        one_case_boundary_bytes=boundary,one_case_state_bytes=state,
        one_case_reference_bytes=reference,estimated_host_peak_upper_bytes=host,
        estimated_gpu0_peak_upper_bytes=device,host_budget_bytes=int(host_memory_gib*GIB),
        estimate_scope='conservative allocation bound; excludes unbounded external allocator/library overhead; not measured peak')
    if nr!=n**3:raise ValueError('CPU chunks do not contain the full raw grid')
    if host>report['host_budget_bytes']:raise MemoryError(f'Q08 full merge host resource preflight refused: {report}')
    try:
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        report['observed_host_available_bytes']=available
        if host>available:raise MemoryError('Q08 full merge exceeds observed available host memory')
    except FileNotFoundError:pass
    return chunks,report


def _pad_last(a,width):
    if a.shape[-1]==width:return a
    out=np.zeros(a.shape[:-1]+(width,),dtype=a.dtype);out[...,:a.shape[-1]]=a
    return out


@dataclass(frozen=True)
class ChunkView:
    path: Path
    bank: QBank
    geometry: dict
    receipt: dict
    raw_offset: int
    owner_offset: int


@dataclass(frozen=True)
class MergedQBank:
    """Trusted campaign-only bank view, distinct from persisted package bank-v1."""
    metadata: dict
    arrays: dict
    diagnostics: dict
    receipts: tuple
    member_hashes: dict

    def __getattr__(self,name):
        a=object.__getattribute__(self,'arrays')
        if name in a:return a[name]
        raise AttributeError(name)

    @property
    def identity(self):
        return digest(dict(schema=MERGED_SCHEMA,provenance=self.metadata,
                           members=self.member_hashes,receipts=self.receipts))

    def validate(self):
        if self.metadata.get('schema')!=MERGED_SCHEMA:raise ValueError('campaign merged schema mismatch')
        for group,values in (('arrays',self.arrays),('diagnostics',self.diagnostics)):
            for name,a in values.items():
                if self.member_hashes[f'{group}:{name}']!=sha256_array(a):raise ValueError('merged campaign content changed')
        nr=len(self.raw);no=len(self.owners)
        if nr!=self.metadata['n']**3 or not np.array_equal(self.owners,np.arange(self.metadata['n_owner'])):
            raise ValueError('merged campaign full-owner/raw coverage incomplete')
        if len(np.unique(self.raw))!=nr or not np.array_equal(np.sort(self.raw),np.arange(nr)):
            raise ValueError('merged campaign raw identity incomplete')
        if np.any(self.owner_raw<0) or np.any(self.owner_raw>=nr) or self.owner_raw.shape!=self.owner_weight.shape:
            raise ValueError('merged projection index/shape')
        active=self.owner_weight!=0
        if not np.array_equal(np.sort(self.owner_raw[active]),np.arange(nr)) or not np.allclose(self.owner_weight.sum(1),1,rtol=0,atol=1e-12):
            raise ValueError('merged complete-owner projection')
        for i in range(no):
            rows=self.owner_raw[i,active[i]]
            if np.any(self.raw_to_owner[rows]!=i) or not np.array_equal(self.owner_weight[i,active[i]],self.raw_weight[rows]):
                raise ValueError('merged owner/member projection mismatch')
        return self


def merge_chunks(run,n,identity,host_memory_gib):
    """Exact two-pass dense merge with recoverable literal CPU chunk views."""
    chunks,estimate=estimate_merge(run,n,identity,host_memory_gib)
    parts=[];diag=[];geometry=[];records=[];raw_offset=0;owner_offset=0
    table=ExactQueryTable();nd=estimate['max_donors'];nm=estimate['max_members']
    first=None
    for path,receipt in chunks:
        bank=load_q_bank(path/'bank.npz');bank.validate()
        if bank.metadata['n']!=n:raise ValueError('CPU bank grid mismatch')
        if first is None:first=bank.metadata
        elif any(bank.metadata.get(k)!=first.get(k) for k in MATCH_KEYS):raise ValueError('CPU bank geometry/topology/policy provenance mismatch')
        if bank.metadata.get('has_gradients'):raise ValueError('GPU campaign expects scalar compact banks without host gradients')
        if receipt.get('owners')!=bank.owners.tolist():raise ValueError('CPU owner receipt mismatch')
        if 'raw' in receipt and receipt['raw']!=bank.raw.tolist():raise ValueError('CPU raw receipt mismatch')
        if receipt.get('raw_hash',sha256_array(bank.raw))!=sha256_array(bank.raw):raise ValueError('CPU raw hash receipt mismatch')
        with np.load(path/'geometry.npz',allow_pickle=False) as z:
            if set(z.files)!=set(GEOMETRY):raise ValueError('CPU material geometry manifest mismatch')
            geom={k:z[k].copy() for k in GEOMETRY}
        # lower_q_plan validates finite positive geometry and field shapes.
        lower_q_plan(bank,diffusion_span=1/32,**geom)
        p={k:np.asarray(v) for k,v in bank.arrays.items()}
        old_queries={k:p[k].copy() for k in ('query_table','wall_node_query','wall_slot_query')}
        qmap=table.add_many(p['query_table'])
        p['wall_node_query']=qmap[p['wall_node_query']];p['wall_slot_query']=qmap[p['wall_slot_query']]
        p['raw_to_owner']=p['raw_to_owner']+owner_offset
        p['owner_raw']=_pad_last(p['owner_raw']+raw_offset,nm)
        p['owner_weight']=_pad_last(p['owner_weight'],nm)
        p['wall_index']=p['wall_index']+raw_offset
        for k in PAD_DONOR:p[k]=_pad_last(p[k],nd)
        records.append(dict(path=path,receipt=receipt,metadata=bank.metadata,identity=bank.identity,
            raw_offset=raw_offset,owner_offset=owner_offset,nraw=len(bank.raw),nowner=len(bank.owners),
            nwall=len(bank.wall_index),wall_offset=sum(r['nwall'] for r in records),
            width=bank.donor.shape[1],members=bank.owner_raw.shape[1],queries=old_queries,
            eta_step=geom['eta_step']))
        parts.append(p);diag.append(bank.diagnostics);geometry.append(geom)
        raw_offset+=len(bank.raw);owner_offset+=len(bank.owners)
    arrays={}
    for k in parts[0]:
        if k=='query_table':arrays[k]=table.array()
        else:arrays[k]=np.concatenate([p[k] for p in parts],axis=1 if k in SPAN_RAW+SPAN_WALL else 0)
    diagnostics={k:np.concatenate([d[k] for d in diag]) for k in diag[0]}
    geom={k:np.concatenate([g[k] for g in geometry]) for k in GEOMETRY if k!='eta_step'}
    steps=[g['eta_step'] for g in geometry]
    if all(a.ndim==0 and a.dtype==steps[0].dtype and a.tobytes()==steps[0].tobytes() for a in steps):geom['eta_step']=steps[0]
    else:geom['eta_step']=np.concatenate([np.full(r['nraw'],a.item(),dtype=a.dtype) if a.ndim==0 else a for r,a in zip(records,steps)])
    lineage=tuple(dict(index=r['receipt']['index'],bank_identity=r['identity'],files=r['receipt']['files'],
                       source_identity=r['metadata']['source_identity'],trace_hash=r['metadata']['trace_hash']) for r in records)
    metadata={k:first[k] for k in MATCH_KEYS}
    metadata.update(schema=MERGED_SCHEMA,source_identity='chunk-content:'+digest(lineage),
        trace_hash=sha256_array(arrays['raw'])+':'+digest([r['trace_hash'] for r in lineage]),
        trace_hash_role='aggregate saved chunk endpoint receipts, no new endpoint data',has_gradients=False)
    hashes={f'{group}:{k}':sha256_array(a) for group,values in (('arrays',arrays),('diagnostics',diagnostics)) for k,a in values.items()}
    merged=MergedQBank(metadata,arrays,diagnostics,lineage,hashes).validate()
    views=[]
    for r in records:
        sl=slice(r['raw_offset'],r['raw_offset']+r['nraw']);ol=slice(r['owner_offset'],r['owner_offset']+r['nowner'])
        wl=slice(r['wall_offset'],r['wall_offset']+r['nwall'])
        a={}
        for k,v in arrays.items():
            if k in ('owners','owner_raw','owner_weight'):a[k]=v[ol]
            elif k in SPAN_RAW:a[k]=v[:,sl]
            elif k in SPAN_WALL:a[k]=v[:,wl]
            elif k in WALL or k in ('wall_index','wall_node_query','wall_slot_query'):a[k]=v[wl]
            elif k=='query_table':continue
            else:a[k]=v[sl]
            if k in PAD_DONOR:a[k]=a[k][...,:r['width']]
        a.update(r['queries'])
        a['owner_raw']=a['owner_raw'][...,:r['members']]-r['raw_offset']
        a['owner_weight']=a['owner_weight'][...,:r['members']]
        a['wall_index']=a['wall_index']-r['raw_offset'];a['raw_to_owner']=a['raw_to_owner']-r['owner_offset']
        bank=QBank(r['metadata'],a,{k:v[sl] for k,v in diagnostics.items()}).validate()
        if bank.identity!=r['identity']:raise ValueError('merged view does not replay literal chunk content identity')
        g={k:(r['eta_step'] if k=='eta_step' else geom[k][sl]) for k in GEOMETRY}
        views.append(ChunkView(r['path'],bank,g,r['receipt'],r['raw_offset'],r['owner_offset']))
    del parts,diag,geometry,records
    gc.collect()
    return merged,geom,tuple(views),estimate


def device_inventory(test_cpu=False):
    """Actual hardware only; CPU emulation is explicitly a test-mode receipt."""
    devices=jax.devices('cpu' if test_cpu else 'gpu')
    if len(devices)<4:raise RuntimeError('Q08 GPU replay requires four actual devices (or explicit forced4CPU test mode)')
    devices=devices[:4]
    if not test_cpu and any(d.platform!='gpu' or 'A100' not in d.device_kind.upper() for d in devices):
        raise RuntimeError('Q08 remote replay requires four actual NVIDIA A100 GPUs')
    smi=[]
    if not test_cpu:
        output=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        for row in csv.reader(output.stdout.splitlines()):
            i,uuid,name,total,free=(v.strip() for v in row)
            smi.append(dict(index=int(i),uuid=uuid,name=name,total_bytes=int(total)*(1<<20),free_bytes=int(free)*(1<<20)))
        if len(smi)<4:raise RuntimeError('cannot establish four-GPU memory inventory')
    records=[]
    for d in devices:
        stats=d.memory_stats() or {}
        records.append(dict(id=d.id,local_hardware_id=d.local_hardware_id,platform=d.platform,
            kind=d.device_kind,process_index=d.process_index,memory_stats={k:int(v) for k,v in stats.items() if isinstance(v,(int,np.integer))}))
    return devices,dict(test_only_cpu_emulation=bool(test_cpu),jax_version=jax.__version__,devices=records,
        nvidia_smi=smi,preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE','default'),
        host_platform=os.uname().sysname)


def _device_guard(estimate,inventory):
    if inventory['test_only_cpu_emulation']:return
    # Under a preallocated JAX pool, observed free driver memory plus the
    # process's observed unused reservation remain available to this run.
    for d in inventory['devices']:
        row=next((r for r in inventory['nvidia_smi'] if r['index']==d['local_hardware_id']),None)
        if row is None:raise RuntimeError('GPU identity/memory ordinal cannot be matched safely')
        stats=d['memory_stats'];used=stats.get('bytes_in_use',0)
        pool_free=max(0,stats.get('bytes_reserved',0)-used)
        available=row['free_bytes']+pool_free
        if 'bytes_limit' in stats:available=min(available,max(0,stats['bytes_limit']-used))
        if estimate['estimated_gpu0_peak_upper_bytes']>available:
            raise MemoryError(f'Q08 GPU resource preflight refused: estimated {estimate["estimated_gpu0_peak_upper_bytes"]} bytes, device {d["id"]} available {available}; no resource/tolerance relaxation')


def _sync(tree):
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf,'block_until_ready'):leaf.block_until_ready()
    return tree


def _cpu_reference(chunks,x,phi,case,span,kinds,pk,api):
    """One assembled reference; each literal CPU chunk action is released."""
    outputs=[];definition=None
    with jax.default_device(jax.devices('cpu')[0]):
        for chunk in chunks:
            plan=lower_q_plan(chunk.bank,diffusion_span=span,**chunk.geometry)
            bi,bo,pb=api.boundaries(chunk.bank,case)
            result=_sync(apply_q_plan(plan,x,bi,bo,phi,pb,api.COEFF,
                kinds=tuple(kinds),phi_kind=pk,tau=api.TAU,mu=api.MU))
            leaves,treedef=jax.tree.flatten(result)
            if definition is None:definition=treedef
            elif definition!=treedef:raise ValueError('CPU output structure changed across chunks')
            outputs.append(tuple(np.asarray(a) for a in leaves))
    # All source/owner groups are concatenated in exact chunk order; no owner
    # reduction crosses a chunk because every chunk owns complete members.
    return jax.tree.unflatten(definition,[np.concatenate([part[i] for part in outputs],axis=0) for i in range(len(outputs[0]))])


def _place_plan(sharded,mesh):
    return type(sharded)(jax.device_put(sharded.plan,NamedSharding(mesh,P('z'))),(),
        jax.device_put(sharded.raw_selectors,NamedSharding(mesh,P('z'))),
        jax.device_put(sharded.owner_gather,NamedSharding(mesh,P())),
        jax.device_put(sharded.raw_gather,NamedSharding(mesh,P())),sharded.n_global_raw,
        sharded.n,sharded.m,sharded.n_shards,sharded.halo,
        jax.device_put(sharded.inverse,NamedSharding(mesh,P())))


def _unit_path(root,span,kind,case,count):
    return root/'records'/f's{int(1/span)}_k{kind}_c{case:02d}_d{count}.json'


def _valid_metrics(metrics,no,nr):
    try:
        floats=('divergence_homogeneous','divergence_lift','divergence_physical','vorticity_homogeneous','vorticity_lift','vorticity_current','electron_phi','electron_ti_compensation','electron_generalized_force')
        spec={k:([no,6],'float64') for k in ('centered','correction','diffusion','combined')}
        spec.update({'raw_current.'+k:([nr],'float64') for k in floats})
        spec.update({k:([nr],'bool') for k in ('raw_current.inputs_valid','inputs_valid','eigensystem_admissible')})
        spec['raw_electron_material']=([nr],'float64')
        leaves=metrics['leaves']
        if set(leaves)!=set(spec):return False
        for key,(shape,dtype) in spec.items():
            v=leaves[key]
            if v['shape']!=shape or v['dtype']!=dtype:return False
            for metric in ('max_scaled_error','max_abs_error'):
                if not np.isfinite(v[metric]) or v[metric]<0:return False
            if v['max_scaled_error']>1 or (dtype=='bool' and (v['max_scaled_error']!=0 or v['max_abs_error']!=0)):return False
        return all(metrics[k]==max(v[k] for v in leaves.values()) for k in ('max_scaled_error','max_abs_error'))
    except (KeyError,TypeError,ValueError):return False


def _valid_record(path,identity,no=None,nr=None):
    if not path.exists():return False
    try:r=json.loads(path.read_text())
    except (json.JSONDecodeError,OSError):return False
    return r.get('identity')==identity and r.get('passed') is True and (no is None or _valid_metrics(r.get('metrics'),no,nr))


def run_resolution(run,n,identity,host_memory_gib,*,test_cpu=False,cases=None,
                   common_api=None,warm_repeats=7):
    """Validate all22×4BC×2span on actual1/4GPU, with per-case checked resume.

    ``cases`` may restrict only explicit forced-CPU test runs. No full-grid
    arrays are merged before host/device resource preflight. Timings separate
    host preparation, transfer, first synchronized JIT and warm full-operator
    calls; none is a measured full integrator timestep or kernel-only claim.
    """
    api=_common(common_api);run=Path(run);root=run/'gpu'/f'N{n}';root.mkdir(parents=True,exist_ok=True)
    if not jax.config.jax_enable_x64:raise RuntimeError('Q08 replay requires JAX_ENABLE_X64=true')
    if isinstance(warm_repeats,bool) or not isinstance(warm_repeats,int) or warm_repeats<1:raise ValueError('positive warm sample count required')
    if cases is not None and not test_cpu:raise ValueError('actual GPU replay cannot restrict the22-state matrix')
    cases=tuple(range(22)) if cases is None else tuple(cases)
    if not cases or len(set(cases))!=len(cases) or any(isinstance(c,bool) or not isinstance(c,int) or c<0 or c>=22 for c in cases):raise ValueError('invalid test case subset')
    if len(api.KINDS)!=4:raise ValueError('four inherited boundary combinations required')
    devices,inventory=device_inventory(test_cpu)
    chunks,estimate=estimate_merge(run,n,identity,host_memory_gib)
    _device_guard(estimate,inventory)
    data=run/'data'/f'N{n}'
    for name in ('state.npy','phi.npy','raw_to_owner.npy'):
        if not (data/name).is_file():raise ValueError(f'missing GPU input {name}')
    inputs={name:sha256_file(data/name) for name in ('state.npy','phi.npy','raw_to_owner.npy')}
    sources={str(Path(__file__).name):sha256_file(__file__)}
    from . import common as common_module
    sources['common']=sha256_file(common_module.__file__)
    # Parent campaign identity carries broader source/input provenance; these
    # direct source receipts additionally protect the wrapper and runtime.
    for module in ('drbx.native.q_plan','drbx.native.q_sharding','drbx.stencils.q_plan'):
        obj=__import__(module,fromlist=['__file__']);sources[module]=sha256_file(obj.__file__)
    grid_identity=digest(dict(campaign=identity,n=n,inputs=inputs,sources=sources,
        cpu=[dict(index=r['index'],files=r['files']) for _,r in chunks],
        hardware=dict(jax_version=inventory['jax_version'],device_kinds=[d['kind'] for d in inventory['devices']]),
        backend='forced_cpu_test' if test_cpu else 'A100_gpu',matrix=dict(cases=cases,kinds=api.KINDS,spans=(1/16,1/32))))
    summary=dict(identity=grid_identity,campaign_identity=identity,n=n,passed=False,inventory=inventory,
        preflight=estimate,inputs=inputs,sources=sources,cpu_file_receipts=[dict(index=r['index'],files=r['files']) for _,r in chunks],test_only=bool(test_cpu),matrix_units=2*4*len(cases)*2,
        timing_scope='synchronized full Q operator setup/transfer/first/warm calls; not an integrator timestep',records=[])
    with lock(root):
        write_json(root/'preflight.json',summary)
        tick=time.perf_counter();bank,geom,views,_=merge_chunks(run,n,identity,host_memory_gib)
        summary['merge_seconds']=time.perf_counter()-tick;summary['merged_identity']=bank.identity
        state=np.load(data/'state.npy',mmap_mode='r');phi=np.load(data/'phi.npy',mmap_mode='r')
        topology=np.load(data/'raw_to_owner.npy',mmap_mode='r')
        if state.shape!=(22,6,bank.metadata['n_owner']) or phi.shape!=(22,bank.metadata['n_owner']):raise ValueError('full22-state input shape mismatch')
        if topology.shape!=(n**3,) or sha256_array(np.asarray(topology,dtype=np.int64))!=bank.metadata['topology_hash'].split(':')[0]:raise ValueError('GPU input topology identity mismatch')
        for span in (1/16,1/32):
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
            for ki,(kinds,pk) in enumerate(api.KINDS):
                cpu_call=jax.jit(lambda qp,x,ib,ob,p,pb,c:apply_q_plan(qp,x,ib,ob,p,pb,c,
                    kinds=tuple(kinds),phi_kind=pk,tau=api.TAU,mu=api.MU),backend='cpu')
                audit_path=root/'merge_audits'/f's{int(1/span)}_k{ki}.json'
                merge_audited=_valid_record(audit_path,grid_identity,len(bank.owners),len(bank.raw))
                if merge_audited:summary.setdefault('cpu_merge_audits',[]).append(str(audit_path.relative_to(run)))
                calls={count:jax.jit(lambda qp,x,ib,ob,p,pb,c,mesh=mesh:sharded_q_rhs(qp,x,ib,ob,p,pb,c,
                    kinds=tuple(kinds),phi_kind=pk,tau=api.TAU,mu=api.MU,mesh=mesh)) for count,(_,mesh,_) in variants.items()}
                for case in cases:
                    paths={count:_unit_path(root,span,ki,case,count) for count in (1,4)}
                    if merge_audited and all(_valid_record(path,grid_identity,len(bank.owners),len(bank.raw)) for path in paths.values()):
                        summary['records'].extend(str(p.relative_to(run)) for p in paths.values());continue
                    x=np.asarray(state[case]);p=np.asarray(phi[case])
                    with jax.default_device(jax.devices('cpu')[0]):
                        start=time.perf_counter();bi,bo,pb=api.boundaries(bank,case);bc_seconds=time.perf_counter()-start
                        cpu_args=jax.tree.map(lambda a:jax.device_put(a,jax.devices('cpu')[0]),(x,bi,bo,p,pb,np.asarray(api.COEFF)))
                        cpu_before=cpu_call._cache_size();start=time.perf_counter()
                        expected=jax.tree.map(np.asarray,_sync(cpu_call(cpu_plan,*cpu_args)))
                        cpu_seconds=time.perf_counter()-start
                    if not merge_audited:
                        start=time.perf_counter();literal=_cpu_reference(views,x,p,case,span,kinds,pk,api)
                        audit=api.check_outputs(expected,literal)
                        write_json(audit_path,dict(identity=grid_identity,campaign_identity=identity,merged_identity=bank.identity,passed=True,span=span,kind=ki,case=case,metrics=audit,
                            literal_chunk_eager_seconds=time.perf_counter()-start,chunk_actions=len(views)))
                        summary.setdefault('cpu_merge_audits',[]).append(str(audit_path.relative_to(run)))
                        merge_audited=True;del literal
                    del cpu_args
                    for count in (1,4):
                        path=paths[count]
                        if _valid_record(path,grid_identity,len(bank.owners),len(bank.raw)):summary['records'].append(str(path.relative_to(run)));continue
                        placed,mesh,_=variants[count]
                        start=time.perf_counter();args=jax.tree.map(lambda a:jax.device_put(a,NamedSharding(mesh,P())),(x,bi,bo,p,pb,np.asarray(api.COEFF)))
                        _sync(args);transfer_seconds=time.perf_counter()-start
                        before=calls[count]._cache_size();start=time.perf_counter();actual=_sync(calls[count](placed,*args));first=time.perf_counter()-start
                        metrics=api.check_outputs(actual,expected)
                        samples=[]
                        for _ in range(warm_repeats):
                            start=time.perf_counter();_sync(calls[count](placed,*args));samples.append(time.perf_counter()-start)
                        _,current_inventory=device_inventory(test_cpu)
                        record=dict(identity=grid_identity,campaign_identity=identity,n=n,span=span,kind=ki,case=case,devices=count,
                            passed=True,test_only=bool(test_cpu),metrics=metrics,cpu_reference_seconds=cpu_seconds,
                            cpu_reference_compiled_new_signature=cpu_call._cache_size()>cpu_before,
                            cpu_reference_path='full merged compact CPU JIT; first uncompleted case per span/kind audited against literal chunk compact eager',
                            boundary_host_seconds=bc_seconds,case_transfer_seconds=transfer_seconds,
                            first_synchronized_seconds=first,compiled_new_signature=calls[count]._cache_size()>before,
                            warm_seconds=samples,warm_median_seconds=float(np.median(samples)),
                            actual_device_memory=current_inventory['devices'],host_peak_rss_gib=peak_rss_gib())
                        write_json(path,record);summary['records'].append(str(path.relative_to(run)))
                        del actual,args
                    del expected,bi,bo,pb,x,p
            del variants,calls,plan,cpu_plan,cpu_call
            gc.collect()
        if len(summary['records'])!=summary['matrix_units'] or any(not _valid_record(run/name,grid_identity,len(bank.owners),len(bank.raw)) for name in summary['records']):raise ValueError('GPU matrix completion mismatch')
        summary['passed']=True;summary['host_peak_rss_gib']=peak_rss_gib()
        summary['record_hashes']={name:sha256_file(run/name) for name in summary['records']}
        summary['audit_hashes']={name:sha256_file(run/name) for name in summary['cpu_merge_audits']}
        write_json(root/'stats.json',summary)
        write_json(run/f'gpu_N{n}.json',dict(summary,identity=identity,resolution_identity=grid_identity))
    return summary


def validate_resolution(run,n,identity):
    """Fail closed unless the complete real-A100 matrix has fresh receipts."""
    run=Path(run);summary=json.loads((run/f'gpu_N{n}.json').read_text())
    if summary.get('identity')!=identity or summary.get('campaign_identity')!=identity or summary.get('n')!=n or summary.get('passed') is not True or summary.get('test_only') is not False:
        raise ValueError('GPU resolution campaign/pass/backend mismatch')
    if summary.get('matrix_units')!=352 or len(summary.get('records',[]))!=352:
        raise ValueError('GPU full352-unit matrix incomplete')
    inventory=summary['inventory']
    if inventory.get('test_only_cpu_emulation') or len(inventory['devices'])!=4 or any(d['platform']!='gpu' or 'A100' not in d['kind'].upper() for d in inventory['devices']):
        raise ValueError('GPU real four-A100 receipt missing')
    expected={(span,kind,case,count) for span in (1/16,1/32) for kind in range(4) for case in range(22) for count in (1,4)}
    actual=set();no=summary['preflight']['owner_count'];nr=n**3
    if set(summary['inputs'])!={'state.npy','phi.npy','raw_to_owner.npy'}:raise ValueError('GPU input receipt keys mismatch')
    for name in summary['records']:
        if summary.get('record_hashes',{}).get(name)!=sha256_file(run/name):raise ValueError('GPU record content receipt mismatch')
        r=json.loads((run/name).read_text())
        if not _valid_record(run/name,summary['resolution_identity'],no,nr) or r.get('test_only') is not False or r.get('campaign_identity')!=identity or r.get('n')!=n:raise ValueError('GPU unit identity/backend mismatch')
        if not np.isfinite(r['metrics']['max_scaled_error']) or r['metrics']['max_scaled_error']>1:raise ValueError('GPU replay budget receipt failed')
        actual.add((r['span'],r['kind'],r['case'],r['devices']))
    if actual!=expected:raise ValueError('GPU full matrix units mismatch')
    audits=summary.get('cpu_merge_audits',[])
    if len(audits)!=8 or len(set(audits))!=8:raise ValueError('CPU merged replay audits incomplete')
    audit_units=set()
    for name in audits:
        if summary.get('audit_hashes',{}).get(name)!=sha256_file(run/name) or not _valid_record(run/name,summary['resolution_identity'],no,nr):raise ValueError('CPU merge audit content mismatch')
        a=json.loads((run/name).read_text())
        if a.get('merged_identity')!=summary['merged_identity']:raise ValueError('CPU merge audit bank mismatch')
        audit_units.add((a['span'],a['kind']))
    if audit_units!={(s,k) for s in (1/16,1/32) for k in range(4)}:raise ValueError('CPU merge audit units mismatch')
    for name,receipt in summary['inputs'].items():
        if sha256_file(run/'data'/f'N{n}'/name)!=receipt:raise ValueError('GPU input content changed')
    current=checked_chunks(run,n,identity)
    if summary.get('cpu_file_receipts')!=[dict(index=r['index'],files=r['files']) for _,r in current]:raise ValueError('GPU CPU source files changed')
    from . import common as common_module
    source_paths={'gpu.py':__file__,'common':common_module.__file__}
    for module in ('drbx.native.q_plan','drbx.native.q_sharding','drbx.stencils.q_plan'):
        source_paths[module]=__import__(module,fromlist=['__file__']).__file__
    if summary['sources']!={k:sha256_file(p) for k,p in source_paths.items()}:raise ValueError('GPU runtime source content changed')
    return summary
