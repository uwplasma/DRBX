"""Checkpointed, complete-owner exact-gradient P/R screen.

The field and seed definitions are copied verbatim from the frozen 2026-09-27 Q
catalogue and corrected span contract. This stage does no tracing or fitting.
"""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,os,resource,sys,time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true'
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import numpy as np
from scripts.q_fci_return_campaign import numerics as qnum
from .fields import callable_field

HERE=Path(__file__).resolve().parent
FIELDS=('common','homogeneous_D','simple_zero_N','x_lambda2_m1','y_lambda2_m1','x_lambda4_m1','y_lambda4_m1','constant')
INPUT_MANIFEST=HERE.parent/'q_fci_return_campaign/input_manifest.json'
CONFIG=HERE.parent/'q_fci_projected_campaign/configuration.json'
REF=1e-4;HALF=5e-5;CHUNK_POINTS=512;OWNER_BATCH=256
CTX=None;ORDER=None;STARTS=None;LABELS=None;RAWVOL=None;OUT=None;DESIGN=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def write_json(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def source_files():return [HERE/n for n in ('exact_screen.py','fields.py','span_contract.py','catalogue.json')]+[CONFIG,HERE.parent/'q_fci_return_campaign/numerics.py',HERE.parent/'q_fci_return_campaign/endpoints.py']
def freeze(input_root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);manifest=json.loads(INPUT_MANIFEST.read_text())
    for entry in manifest['files']:
        p=Path(input_root)/entry['path']
        if not p.is_file() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise RuntimeError(f'input identity mismatch: {p}')
    sources={str(p.relative_to(HERE.parents[2])):sha(p) for p in source_files()}
    design=dict(schema='q-shortspan-pr-v1',method=dict(fields=FIELDS,outer_spans=('h/8','h/4'),seed_order='span,axis,negative-positive',reference_step=REF,reference_half=HALF,chunk_points=CHUNK_POINTS,owner_batch=OWNER_BATCH,projection='complete owner frozen raw-volume'),inputs=manifest,sources=sources,input_root=str(Path(input_root).resolve()))
    path=out/'design.json'
    if path.exists() and json.loads(path.read_text())!=design:raise RuntimeError('incompatible design; use fresh output')
    write_json(path,design);return design
def check_design(out,input_root):
    design=json.loads((Path(out)/'design.json').read_text())
    if design['input_root']!=str(Path(input_root).resolve()):raise RuntimeError('input root differs from frozen design')
    if design['sources']!={str(p.relative_to(HERE.parents[2])):sha(p) for p in source_files()}:raise RuntimeError('source identity differs')
    for e in design['inputs']['files']:
        p=Path(input_root)/e['path']
        if not p.is_file() or p.stat().st_size!=e['bytes']:raise RuntimeError(f'input missing or resized: {p}')
    return design
def init_worker(n,input_root,out):
    global CTX,ORDER,STARTS,LABELS,RAWVOL,OUT,DESIGN
    out=Path(out);DESIGN=check_design(out,input_root);OUT=out
    sys.dont_write_bytecode=True
    cache=out/'cache';cache.mkdir(exist_ok=True);os.environ['DRBX_CACHE_DIR']=str(cache);os.environ['JAX_COMPILATION_CACHE_DIR']=str(cache)
    CTX=qnum.context(n,input_root,json.loads(CONFIG.read_text()))
    LABELS=np.asarray(CTX['topology']['compact_raw_owner'],int).ravel();RAWVOL=np.asarray(CTX['artifact'].polar_angular_geometry.raw_volume,float).ravel()
    ORDER=np.argsort(LABELS,kind='stable');STARTS=np.r_[0,np.cumsum(np.bincount(LABELS,minlength=len(CTX['volume'])))]
    assert len(LABELS)==n**3 and len(CTX['volume'])==len(STARTS)-1
def points_for(raw,n):
    grid=CTX['artifact'].geometry.grid;ijk=np.array(np.unravel_index(raw,(n,n,n))).T
    return np.column_stack([getattr(grid,a).centers[ijk[:,k]] for k,a in enumerate('xyz')])
def points(q,n):
    width=np.array((1/n,2*np.pi/n,2*np.pi/n));seeds=np.broadcast_to(q[:,None,:],(len(q),12,3)).copy()
    for span,alpha in enumerate((.125,.25)):
        for axis in range(3):
            seeds[:,6*span+2*axis,axis]-=alpha*width[axis]/2
            seeds[:,6*span+2*axis+1,axis]+=alpha*width[axis]/2
    ref=np.broadcast_to(q[:,None,:],(len(q),12,3)).copy();half=ref.copy()
    for axis in range(3):
        for j,m in enumerate((-2,-1,1,2)):
            ref[:,4*axis+j,axis]+=m*REF;half[:,4*axis+j,axis]+=m*HALF
    return seeds,ref,half
def geometry(coords):
    jj=[];bb=[]
    for lo in range(0,len(coords),CHUNK_POINTS):
        j,b,_=qnum.base(CTX,coords[lo:lo+CHUNK_POINTS]);jj.append(j);bb.append(b)
    return np.concatenate(jj),np.concatenate(bb)
def actions(q,seeds,ref,half):
    k=len(q);allpts=np.vstack((q,seeds.reshape(-1,3),ref.reshape(-1,3),half.reshape(-1,3)))
    unique,inv=np.unique(allpts,axis=0,return_inverse=True)
    qi=inv[:k];si=inv[k:k+12*k].reshape(k,12);ri=inv[k+12*k:k+24*k].reshape(k,12);hi=inv[k+24*k:].reshape(k,12)
    st=time.process_time();J,b=geometry(unique);tc=time.process_time()-st
    P=np.zeros((k,len(FIELDS),2,3),complex);R=np.zeros((k,len(FIELDS),3),complex);Rh=np.zeros_like(R)
    st=time.process_time()
    for fi,name in enumerate(FIELDS):
        if name=='simple_zero_N':P[:,fi]=P[:,1];R[:,fi]=R[:,1];Rh[:,fi]=Rh[:,1];continue
        grad=callable_field(name)(unique)[1];dot=np.einsum('pi,pi->p',b,grad);flux=J[:,None]*b*dot[:,None]
        for span in range(2):
            for axis in range(3):
                neg,pos=6*span+2*axis,6*span+2*axis+1
                sep=seeds[:,pos,axis]-seeds[:,neg,axis]
                P[:,fi,span,axis]=(flux[si[:,pos],axis]-flux[si[:,neg],axis])/(sep*J[qi])
        for idx,step,target in ((ri,REF,R),(hi,HALF,Rh)):
            for axis in range(3):
                a=4*axis;v=flux[idx[:,a:a+4],axis]
                target[:,fi,axis]=(v[:,0]-8*v[:,1]+8*v[:,2]-v[:,3])/(12*step*J[qi])
    return P,R,Rh,tc,time.process_time()-st,len(unique)
def chunk_path(out,n,chunk):return Path(out)/f'N{n}'/f'chunk_{chunk:05d}.npz'
def completed(path,digest,lo,hi):
    receipt=path.with_suffix('.json')
    if not path.exists() or not receipt.exists():return False
    try:
        r=json.loads(receipt.read_text())
        return r['design_sha256']==digest and r['lo']==lo and r['hi']==hi and r['sha256']==sha(path)
    except (ValueError,KeyError,OSError):return False
def run_chunk(job):
    n,lo,hi,digest=job;chunk=lo//OWNER_BATCH;path=chunk_path(OUT,n,chunk);path.parent.mkdir(exist_ok=True)
    if completed(path,digest,lo,hi):return dict(chunk=chunk,skipped=True,cpu_s=0)
    start=time.process_time();raw=ORDER[STARTS[lo]:STARTS[hi]];labels=LABELS[raw];q=points_for(raw,n);seeds,ref,half=points(q,n)
    P,R,Rh,geomcpu,fieldcpu,unique=actions(q,seeds,ref,half)
    owner=np.arange(lo,hi);denom=np.asarray(CTX['volume'])[owner]
    if not np.array_equal(np.unique(labels),owner):raise RuntimeError('incomplete owner batch')
    got=np.bincount(labels-lo,weights=RAWVOL[raw],minlength=len(owner));assert np.allclose(got,denom,rtol=2e-13,atol=1e-15)
    def project(a):
        weighted=(a.T*(RAWVOL[raw]/denom[labels-lo])).T
        out=np.zeros((len(owner),)+a.shape[1:],complex)
        np.add.at(out,labels-lo,weighted)
        return out
    pp,rr,hh=project(P),project(R),project(Rh)
    tmp=path.with_name(path.stem+'.tmp.npz');np.savez_compressed(tmp,owner=owner,volume=denom,raw_count=np.bincount(labels-lo,minlength=len(owner)),P=pp,R=rr,R_half=hh)
    os.replace(tmp,path)
    receipt=dict(schema='q-pr-owner-chunk-v1',N=n,lo=lo,hi=hi,raw_count=len(raw),unique_geometry_points=unique,geometry_cpu_s=geomcpu,field_cpu_s=fieldcpu,cpu_s=time.process_time()-start,maxrss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,design_sha256=digest,sha256=sha(path))
    write_json(path.with_suffix('.json'),receipt)
    return receipt
def run(n,input_root,out,workers):
    if workers not in (1,2):raise ValueError('workers must be 1 or 2')
    design=check_design(out,input_root);digest=sha(Path(out)/'design.json');out=Path(out);out.mkdir(exist_ok=True)
    config=json.loads(CONFIG.read_text())
    topology=Path(input_root)/config['geometry']/f'{n}x{n}x{n}/rlp_topology.npz'
    with np.load(topology) as z:total=int(np.count_nonzero(z['is_active_owner']))
    expected={32:25376,48:86016,64:202304}[n]
    if total!=expected:raise RuntimeError(f'canonical owner count mismatch: {total} != {expected}')
    jobs=[(n,lo,min(lo+OWNER_BATCH,total),digest) for lo in range(0,total,OWNER_BATCH)]
    # Fixed chunk order makes serial/parallel reduction deterministic.
    if workers==1:
        init_worker(n,input_root,out);done=[run_chunk(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(n,input_root,out)) as pool:done=list(pool.map(run_chunk,jobs))
    write_json(out/f'N{n}/stage_receipt.json',dict(N=n,owners=total,chunks=len(jobs),workers=workers,measured_chunk_cpu_s=sum(x['cpu_s'] for x in done),completed=sum(not x.get('skipped',False) for x in done),design_sha256=digest))
    print(json.dumps(dict(N=n,chunks=len(jobs),computed=sum(not x.get('skipped',False) for x in done),cpu_s=sum(x['cpu_s'] for x in done))))
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','run'));p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--N',type=int,choices=(32,48,64));p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    if a.command=='freeze':freeze(a.input_root,a.output)
    else:
        if a.N is None:raise ValueError('--N required for run')
        run(a.N,a.input_root,a.output,a.workers)
if __name__=='__main__':main()
