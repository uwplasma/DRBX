"""Read-only independent reduction of original CPU receipts."""
import json,hashlib
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def require(run,stage,identity):
    r=read(run/f"{stage}.json")
    if not r.get("passed") or r.get("campaign_identity")!=identity:raise ValueError(stage+" identity/gate")
    return r

def chunk_valid(folder,identity,owners=None):
    p=folder/'stats.json'
    if not p.exists():return False
    r=read(p)
    if not r.get('passed') or r.get('campaign_identity')!=identity:return False
    if owners is not None and r.get('owners')!=list(owners):raise ValueError('checkpoint owner mismatch')
    if set(r.get('files',{}))!={'bank.npz','geometry.npz'}:raise ValueError('checkpoint manifest')
    for rel,h in r['files'].items():
        if sha(folder/rel)!=h:raise ValueError('corrupt checkpoint '+str(folder/rel))
    return True

def validate_cpu_readonly(run,n,identity):
    import numpy as np
    plan=read(run/'inputs/plan.json')[str(n)];owners=[];raw_ids=[];max_scaled=0.;files={};total_bytes=0
    data=require(run,f'data_N{n}',identity)
    for rel,h in data['files'].items():
        if sha(run/'data'/f'N{n}'/rel)!=h:raise ValueError('owner data changed')
    ro=np.load(run/'data'/f'N{n}'/'raw_to_owner.npy',allow_pickle=False)
    order=np.argsort(ro,kind='stable');starts=np.r_[0,np.cumsum(np.bincount(ro))]
    components={'centered','correction','diffusion','combined'}
    leaves=components|{'raw_current.'+name for name in ('divergence_homogeneous','divergence_lift','divergence_physical','vorticity_homogeneous','vorticity_lift','vorticity_current','electron_phi','electron_ti_compensation','electron_generalized_force','inputs_valid')}|{'raw_electron_material','inputs_valid','eigensystem_admissible'}
    for i,oo in enumerate(plan['chunks']):
        folder=run/'cpu'/f'N{n}'/f'chunk_{i:06d}'
        if not chunk_valid(folder,identity,oo):raise ValueError('missing or invalid chunk '+str(folder))
        r=read(folder/'stats.json');owners+=oo
        raw=np.concatenate([order[starts[o]:starts[o+1]] for o in oo]);raw_ids.extend(r['raw'])
        if not np.array_equal(r['raw'],raw):raise ValueError('raw owner membership/order')
        records=r['components'];keys=[(m['span'],m['kind']) for m in records]
        if len(keys)!=8 or set(keys)!={(span,k) for span in (1/16,1/32) for k in range(4)}:raise ValueError('CPU case coverage')
        checked_max=0.
        for record in records:
            if set(record['leaves'])!=leaves:raise ValueError('CPU leaf coverage')
            for name,metric in record['leaves'].items():
                expected=[22,len(oo),6] if name in components else [22,len(raw)]
                if metric['shape']!=expected:raise ValueError('CPU state/leaf shape coverage')
                boolean=name.endswith('inputs_valid') or name=='eigensystem_admissible'
                if metric['dtype']!=('bool' if boolean else 'float64'):raise ValueError('CPU leaf precision')
                scaled=metric['max_scaled_error'];absolute=metric['max_abs_error']
                if not np.isfinite([scaled,absolute]).all() or not 0<=scaled<=1 or absolute<0:raise ValueError('CPU nonfinite/failed replay')
                checked_max=max(checked_max,scaled)
        if not np.isfinite(r['max_scaled']) or r['max_scaled']!=checked_max or r['all_arrays_bitwise'] is not True:raise ValueError('replay summary gate')
        max_scaled=max(max_scaled,checked_max)
        files[str(i)]=sha(folder/'stats.json');total_bytes+=sum((folder/f).stat().st_size for f in r['files'])
    if not np.array_equal(np.sort(owners),np.arange(plan['n_owner'])):raise ValueError('complete owner coverage')
    if not np.array_equal(np.sort(raw_ids),np.arange(plan['n_raw'])):raise ValueError('complete raw coverage')
    result=dict(passed=True,campaign_identity=identity,n=n,owners=len(owners),raw=plan['n_raw'],chunks=len(files),max_scaled=max_scaled,bytes=total_bytes,chunk_receipts=files)
    if read(run/f'cpu_N{n}.json') != result:raise ValueError('saved CPU completion differs from independent reduction')
    return result
