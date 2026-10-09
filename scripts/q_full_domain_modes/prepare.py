"""Spawn CPU workers for immutable traces; commit a chunk only after reload."""
from pathlib import Path
import concurrent.futures, multiprocessing, os, time
from types import SimpleNamespace
from . import common as c
ENV=None

def init_worker(values):
    global ENV
    a=SimpleNamespace(**values);a.run=Path(a.run);a.repo_root=Path(a.repo_root);a.canonical_root=Path(a.canonical_root)
    c.guard(a.stage_seconds,a.worker_gib)
    backend=c.bootstrap(a.run/'caches'/'prepare'/str(os.getpid()),a.repo_root)
    import numpy as np
    from .geometry_model import model
    ctx,t=model.context(a.n,a.canonical_root)
    geom=lambda p:model.geom(ctx,p)
    jac=lambda p:ctx['evaluator']._position_and_jacobian(p)[1]
    with np.load(c.HERE/'inputs'/f'N{a.n}_traces.npz') as z:ends=z['ends'].copy()
    with np.load(c.HERE/'inputs'/f'choices_N{a.n}.npz') as z:choices=z['choice'].copy()
    if ends.shape!=(a.n**3,4,3) or choices.shape!=(a.n**3,):raise ValueError('incomplete retained endpoints/support')
    ENV=(a,np,t,geom,jac,ends,choices,backend)

def worker(task):
    a,np,t,geom,jac,ends,choices,backend=ENV;index,owners=task
    folder=a.run/'chunks'/f'{index:06d}';p=folder/'stats.json'
    old=c.checked(a.run,p,a.identity)
    if old is not None:
        if old['owners']!=owners:raise ValueError('changed chunk owners')
        return old
    from drbx.stencils.q_parallel import raw_members,prepare_paired_chunks
    from drbx.stencils.q_bank import build_q_bank
    from drbx.stencils.q_parallel_rhs import prepare_six_field_rhs
    from drbx.stencils.q_artifact import save_q_bank,load_q_bank
    tick=time.perf_counter();owners=np.asarray(owners,dtype=np.int64);raw=raw_members(t,owners)
    if len(raw)>128:raise ValueError('chunk exceeds frozen preparation capacity')
    outer,inner=prepare_paired_chunks(t,owners,ends[raw],geom,jac,
        raw=raw,source_identity=c.sha(c.HERE/'inputs'/f'N{a.n}_traces.npz'),
        geometry_identity='compact_c3:'+c.sha(c.HERE/'input_manifest.json'),
        frozen_choices=choices[raw],choice_provenance=c.sha(c.HERE/'inputs'/f'choices_N{a.n}.npz'))
    bank=build_q_bank(outer,inner,identity={'full_modes_campaign':a.identity},include_gradients=False)
    rt,_=prepare_six_field_rhs(inner,outer,geom,diffusion_span=1/32,center_b_atol=1e-10)
    geometry=dict(magnetic_L=rt.material.magnetic_L,b_eta=rt.material.b_eta,eta_step=rt.material.eta_step,bmag=rt.bmag,owners=bank.owners,raw=bank.raw)
    folder.mkdir(parents=True,exist_ok=True);save_q_bank(bank,folder/'bank.npz')
    load_q_bank(folder/'bank.npz',expected_identity=bank.identity)
    with (folder/'geometry.npz.tmp').open('wb') as f:np.savez_compressed(f,**geometry)
    (folder/'geometry.npz.tmp').replace(folder/'geometry.npz')
    # Deliberately uncommitted payload is never trusted on resume.
    marker=a.run/'recovery'/f'interrupt_{index:06d}.json'
    if a.interrupt_after_payload==index and not marker.exists():
        c.write(marker,dict(scope='bounded recovery control',uncommitted_payload=True));raise RuntimeError('intentional bounded interruption after payload')
    overlap=compare_fixture(bank,geometry,np)
    result=dict(passed=True,index=index,owners=owners.tolist(),raw=raw.tolist(),seconds=time.perf_counter()-tick,
        peak_rss_gib=c.peak_gib(),backend=backend,bank_identity=bank.identity,footprint=bank.footprint(),overlap=overlap)
    return c.receipt(a,p,result,(folder/'bank.npz',folder/'geometry.npz'))

def compare_fixture(bank,geometry,np):
    from drbx.stencils.q_artifact import load_q_bank
    ref=load_q_bank(c.HERE/'inputs/bounded/bank.npz')
    with np.load(c.HERE/'inputs/bounded/geometry.npz') as z:gref={k:z[k].copy() for k in z.files}
    overlap=np.intersect1d(bank.raw,ref.raw);max_error=0.;checks=0
    for raw in overlap:
        i=int(np.flatnonzero(bank.raw==raw)[0]);j=int(np.flatnonzero(ref.raw==raw)[0])
        if not np.array_equal(bank.donor[i,bank.mask[i]],ref.donor[j,ref.mask[j]]):raise ValueError('fixture donors differ')
        count=int(bank.row_count[i]);wc=np.flatnonzero(bank.wall_index==i);wr=np.flatnonzero(ref.wall_index==j)
        pairs=[(bank.row_value_D[i,:,:count],ref.row_value_D[j,:,:count]),(bank.diffusion_D[:,i,:count],ref.diffusion_D[:,j,:count]),
            (bank.raw_weight[i],ref.raw_weight[j]),(bank.magnetic_b[i],ref.magnetic_b[j]),(bank.magnetic_L[:,i],ref.magnetic_L[:,j])]
        for k in ('magnetic_L','b_eta','bmag'):pairs.append((geometry[k][i],gref[k][j]))
        pairs.append((geometry['eta_step'],gref['eta_step']))
        if len(wc)!=len(wr):raise ValueError('wall partition differs')
        if len(wc):
            wi,wj=int(wc[0]),int(wr[0])
            for k in ('row_value_N_wall','boundary_value_D_trace','boundary_value_N_normal','boundary_wall_normal'):
                x,y=getattr(bank,k)[wi],getattr(ref,k)[wj]
                if k=='row_value_N_wall':x,y=x[:,:count],y[:,:count]
                pairs.append((x,y))
            for k in ('diffusion_N_wall','boundary_D_node','boundary_D_tangent','boundary_N_normal'):
                x,y=getattr(bank,k)[:,wi],getattr(ref,k)[:,wj]
                if k=='diffusion_N_wall':x,y=x[:,:count],y[:,:count]
                pairs.append((x,y))
            for k in ('wall_node_query','wall_slot_query'):pairs.append((bank.query_table[getattr(bank,k)[wi]],ref.query_table[getattr(ref,k)[wj]]))
        for x,y in pairs:
            np.testing.assert_allclose(x,y,atol=1e-8,rtol=1e-11)
            if np.size(x):max_error=max(max_error,float(np.max(np.abs(np.asarray(x)-y))))
            checks+=1
    return dict(raw_count=len(overlap),checks=checks,max_absolute_error=max_error,passed=True,tolerance={'atol':1e-8,'rtol':1e-11})

def selection(args):
    p=c.read(c.HERE/'inputs/plan.json')[str(args.n)];chunks=p['chunks']
    ids=sorted(set(args.chunks if args.chunks is not None else range(len(chunks))))
    if args.max_chunks is not None:
        if args.max_chunks<1:raise ValueError('max chunks positive')
        ids=ids[:args.max_chunks]
    if not ids or min(ids)<0 or max(ids)>=len(chunks):raise ValueError('bad chunk selection')
    bounded=len(ids)<len(chunks)
    if not bounded and not args.remote_ok:raise ValueError('full prepare requires --remote-ok')
    if args.interrupt_after_payload is not None and not bounded:raise ValueError('interruption control bounded-only')
    return p,ids,bounded

def run(args):
    plan,ids,bounded=selection(args)
    admitted=c.admission(args);reserve=admitted['reserve_gib'];effective=admitted['effective_concurrency']
    if effective<1:raise MemoryError('host cannot admit a worker plus controller')
    c.guard(args.stage_seconds,reserve)
    pending=[];resumed=[]
    for i in ids:
        r=c.checked(args.run,f'chunks/{i:06d}/stats.json',args.identity)
        if r is None:pending.append((i,plan['chunks'][i]))
        elif r['owners']!=plan['chunks'][i]:raise ValueError('chunk selection provenance')
        else:resumed.append(i)
    committed_path='receipts/prepare-bounded.json' if bounded else 'receipts/prepare.json'
    old=c.checked(args.run,committed_path,args.identity)
    if old is not None and not pending and old['selected_chunks']==ids:
        print('prepare: checked complete checkpoint; no recomputation',flush=True)
        return old
    tick=time.perf_counter();values=vars(args).copy()
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(values,)) as pool:
            it=iter(pending);live={}
            def submit():
                try:item=next(it)
                except StopIteration:return
                live[pool.submit(worker,item)]=item[0]
            for _ in range(min(effective,len(pending))):submit()
            while live:
                ready,_=concurrent.futures.wait(live,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in ready:
                    r=f.result();del live[f];print(f"prepared chunk {r['index']} RSS={r['peak_rss_gib']:.3f} GiB",flush=True);submit()
    files=[args.run/f'chunks/{i:06d}/stats.json' for i in ids]
    if bounded:
        from .campaign import child
        child(args,'merge-bounded')
        files.extend((args.run/'prepared-bounded/bank.npz',args.run/'prepared-bounded/geometry.npz',args.run/'prepared-bounded/merge.json'))
    if not bounded:
        # Separate process: free worker geometry and cap the merger independently.
        from .campaign import child
        child(args,'merge')
        files.extend((args.run/'prepared/bank.npz',args.run/'prepared/geometry.npz',args.run/'prepared/merge.json'))
    return c.receipt(args,'receipts/prepare-bounded.json' if bounded else 'receipts/prepare.json',
        dict(passed=True,full_domain=not bounded,selected_chunks=ids,resumed_chunks=resumed,new_chunks=[i for i,_ in pending],
            requested_workers=args.workers,effective_workers=effective,host_gib=args.host_gib,worker_gib=args.worker_gib,
            seconds=time.perf_counter()-tick,scope='tooling only' if bounded else 'complete owner/raw/donor closure'),files)

def merge(args,bounded=False):
    """Two-pass merge avoids retaining all individual banks alongside the result."""
    import numpy as np
    from drbx.stencils.q_bank import QBank,IDENTITY_KEYS
    from drbx.stencils.q_artifact import load_q_bank,save_q_bank
    from drbx.stencils.q_parallel import sha256_array
    from .operators import topology
    plan,ids,_=selection(args) if bounded else (c.read(c.HERE/'inputs/plan.json')[str(args.n)],None,None)
    if ids is None:ids=list(range(len(plan['chunks'])))
    paths=[args.run/f'chunks/{i:06d}/bank.npz' for i in ids]
    t=topology(args.canonical_root/f'geometry_artifacts/rlp_convergence_32_48_64_20260917/{args.n}x{args.n}x{args.n}')
    nr=no=nw=nq=nd=nm=0;first=None
    for i,path in zip(ids,paths):
        c.checked(args.run,f'chunks/{i:06d}/stats.json',args.identity) or (_ for _ in ()).throw(ValueError('missing chunk'))
        b=load_q_bank(path)
        if first is None:first=b
        nr+=len(b.raw);no+=len(b.owners);nw+=len(b.wall_index);nq+=len(b.query_table);nd=max(nd,b.donor.shape[-1]);nm=max(nm,b.owner_raw.shape[-1])
    expected_owners=np.array([v for i in ids for v in plan['chunks'][i]])
    from drbx.stencils.q_parallel import raw_members
    expected_raw=raw_members(t,expected_owners)
    if (nr,no)!=(len(expected_raw),len(expected_owners)):raise ValueError('owner/raw totals')
    span={'diffusion_D','magnetic_L','diffusion_N_wall','boundary_D_node','boundary_D_tangent','boundary_N_normal'}
    wall={'wall_index','row_value_N_wall','boundary_value_D_trace','boundary_value_N_normal','boundary_wall_normal','wall_node_query','wall_slot_query','diffusion_N_wall','boundary_D_node','boundary_D_tangent','boundary_N_normal'}
    owner={'owners','owner_raw','owner_weight'};padding={'donor','mask','row_value_D','row_value_N_wall','diffusion_D','diffusion_N_wall'}
    arrays={}
    for k,x in first.arrays.items():
        shape=list(x.shape);axis=1 if k in span else 0
        shape[axis]=nq if k=='query_table' else no if k in owner else nw if k in wall else nr
        if k in padding:shape[-1]=nd
        if k in {'owner_raw','owner_weight'}:shape[-1]=nm
        arrays[k]=np.zeros(shape,dtype=x.dtype)
    diagnostics={k:np.empty((nr,*x.shape[1:]),dtype=x.dtype) for k,x in first.diagnostics.items()}
    geometry={};r=o=w=q=0;chunks=[]
    for i,path in zip(ids,paths):
        b=load_q_bank(path);rr,oo,ww,qq=map(len,(b.raw,b.owners,b.wall_index,b.query_table));chunks.append(b.identity)
        for k,x in b.arrays.items():
            axis=1 if k in span else 0;start,length=(q,qq) if k=='query_table' else (o,oo) if k in owner else (w,ww) if k in wall else (r,rr)
            index=[slice(None)]*x.ndim;index[axis]=slice(start,start+length)
            if k in padding or k in {'owner_raw','owner_weight'}:index[-1]=slice(0,x.shape[-1])
            if k=='raw_to_owner':x=x+o
            elif k in {'wall_index','owner_raw'}:x=x+r
            elif k in {'wall_node_query','wall_slot_query'}:x=x+q
            arrays[k][tuple(index)]=x
        for k,x in b.diagnostics.items():diagnostics[k][r:r+rr]=x
        with np.load(path.parent/'geometry.npz') as z:
            for k in ('magnetic_L','b_eta','eta_step','bmag'):
                x=z[k]
                if x.ndim==0:
                    if k in geometry:np.testing.assert_array_equal(geometry[k],x)
                    else:geometry[k]=x.copy()
                else:
                    if k not in geometry:geometry[k]=np.empty((nr,*x.shape[1:]),x.dtype)
                    geometry[k][r:r+rr]=x
        r+=rr;o+=oo;w+=ww;q+=qq
    np.testing.assert_array_equal(arrays['owners'],expected_owners)
    np.testing.assert_array_equal(arrays['raw'],expected_raw)
    np.testing.assert_array_equal(t.ro[arrays['raw']],arrays['owners'][arrays['raw_to_owner']])
    np.testing.assert_allclose(arrays['raw_weight'],t.rv[arrays['raw']]/t.vol[t.ro[arrays['raw']]],rtol=2e-13)
    np.testing.assert_allclose(diagnostics['slot_points'][:,2],t.pts[arrays['raw']],rtol=0,atol=2e-14)
    donor_closed=bool(np.isin(arrays['donor'][arrays['mask']],arrays['owners']).all())
    if not bounded and not donor_closed:raise ValueError('donor closure failed')
    m=dict(first.metadata);m['merged_chunks']=chunks
    with np.load(c.HERE/'inputs'/f'N{args.n}_traces.npz') as z:trace_hash=sha256_array(arrays['raw'])+':'+sha256_array(z['ends'][arrays['raw']])
    m['trace_hash']=trace_hash;m['span_metadata']=[dict(v,**{k:m[k] for k in IDENTITY_KEYS}) for v in m['span_metadata']]
    bank=QBank(m,arrays,diagnostics).validate();folder=args.run/('prepared-bounded' if bounded else 'prepared');folder.mkdir(parents=True,exist_ok=True)
    save_q_bank(bank,folder/'bank.npz');load_q_bank(folder/'bank.npz',expected_identity=bank.identity)
    geometry.update(owners=bank.owners,raw=bank.raw)
    with (folder/'geometry.npz.tmp').open('wb') as f:np.savez_compressed(f,**geometry)
    (folder/'geometry.npz.tmp').replace(folder/'geometry.npz')
    return dict(passed=True,full_domain=not bounded,owners=no,raw=nr,donor_closure=donor_closed,pinned_halos=bounded,bank_identity=bank.identity,
        footprint=bank.footprint(),peak_rss_gib=c.peak_gib(),overlap=compare_fixture(bank,geometry,np))
