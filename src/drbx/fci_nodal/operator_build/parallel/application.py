"""bounded indexed parallel assembly direct owner application export from indexed Q construction."""
from contextlib import ExitStack, closing
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor

from ..perp.owner_bundle import OwnerPerpendicularBundle
from .application_routes import (MAX_NNZ, ROW_WINDOW, SOURCE_WINDOW, Sink, TransposeRows,
                                 column_requests, build_parallel_routes, mapbin, rows)
from .indexed_input import IndexedInput
from ..._hashing import sha256_file as sha
from .job_storage import atomic_json, mapped, read
from ...operators.nodal.nodal_layout_specs import build_zernike_core_ring_layout
from ...operators.nodal.nodal_layout import eta_planes, node_points
from ...operators.parallel import MOMENT_REDUCED_FACE_QUADRATURE_DEGREE
from ...operators.parallel.weights import basis


def _wall_rows(ids,wall,plane_paths,P,n):
    owners=wall[:,0].astype(np.int64)//P
    parts=[];order=[]
    for plane in np.unique(owners):
        ix=np.flatnonzero(owners==plane);folder=plane_paths[plane]
        original=read(folder/'wall_ids.npy')
        local=np.searchsorted(original,ids[ix])
        if np.any(local>=len(original)) or not np.array_equal(original[local],ids[ix]):
            raise ValueError('Physical wall IDs differ from the construction row map')
        ptr=read(folder/'Rw/indptr.npy')
        counts=ptr[local+1]-ptr[local]
        new=np.r_[0,np.cumsum(counts)]
        if new[-1]>MAX_NNZ:
            raise MemoryError('Wall row window exceeds its entry bound')
        where=np.repeat(ptr[local]-new[:-1],counts)+np.arange(new[-1])
        cols=read(folder/'Rw/indices.npy')[where]
        vals=read(folder/'Rw/data.npy')[where]
        parts.append(sparse.csr_matrix((vals,cols,new),shape=(len(ix),n)))
        order.append(ix)
    return sparse.vstack(parts,format='csr')[np.argsort(np.concatenate(order))]


def _read_owner_magnetic_field(bundle, owner, q, tolerance, size):
    leaves = ('B', 'points')
    bundle.verify(owner, leaves=leaves)
    p = bundle.read(owner, leaves=leaves)
    b = p['B'].reshape(-1)
    if (p['points'].shape != q.shape or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(p['points'])) or np.max(np.abs(p['points']-q)) > tolerance):
        raise ValueError('P B nodes differ from Q nodes; stop, no interpolation')
    if b.dtype != np.float64 or b.shape != (size,) or not np.all(np.isfinite(b)&(b>0)):
        raise ValueError('Expected positive float64 P-normalized B in canonical node order')
    return b


def _raw_owner(folder,owner,edges,route,plane_paths,geometry,fit,recipe,mu,B,operator_sizes):
    N,P,n=operator_sizes
    a,b=map(int,edges[owner:owner+2]);count=b-a
    wallids=mapbin(route/'wall_ids.bin')
    routed=mapbin(route/'record_ordinals.bin')
    nw,nr=len(wallids),len(routed)
    direct_recipe = recipe is None
    if direct_recipe:
        recipe = mapbin(route/'record_ids.bin')
        mu = mapbin(route/'record_mu.bin', np.float64)
    np.save(folder/'nodes_ids.npy',np.arange(a,b,dtype=np.int32))
    np.save(folder/'walls_ids.npy',wallids.astype(np.int32))
    np.save(folder/'records_ids.npy',routed.astype(np.int32))
    def array(key,size,dtype=np.float64):
        return mapped(folder/(key+'.npy'),(size,),dtype)

    H,B2=array('H',count),array('B2',count)
    for lo in range(0,count,ROW_WINDOW):
        hi=min(lo+ROW_WINDOW,count)
        B2[lo:hi]=B[lo:hi]**2
    wall_data={key:array(key,nw) for key in ('mu_wall','sg','leg')}
    record_data={key:array(key,nr,recipe.dtype if key=='ids' else np.float64)
                 for key in ('ids','ell','mu')}
    support=np.zeros(count,bool)
    names=('C','CT','G','D','Rwall','B','BT','LN','LD','L','R')
    with ExitStack() as stack:
        sinks={key:stack.enter_context(Sink(folder/key,n)) for key in names}
        CT=stack.enter_context(closing(TransposeRows(route,'C')))
        BT=stack.enter_context(closing(TransposeRows(route,'K_N')))
        for plane in range(a//P,b//P):
            plane_path=plane_paths[plane]
            h0=read(plane_path/'wall_credited_parallel_mass.npy')
            mb=read(plane_path/'MB_V.npy')
            for lo in range(0,P,ROW_WINDOW):
                hi=min(lo+ROW_WINDOW,P);first,last=plane*P+lo,plane*P+hi
                h=h0[lo:hi];hd=h-mb[lo:hi]
                H[first-a:last-a]=h
                K=rows(plane_path/'K',lo,hi,n)
                KB=rows(plane_path/'MB_Gram',lo,hi,n)
                C=rows(plane_path/'C',lo,hi,n)
                ct=CT.read(first,last,n)
                if K.nnz+KB.nnz>MAX_NNZ:
                    raise MemoryError('K+MB row window exceeds the application entry bound')
                for key,matrix in [('C',C),('CT',ct),('G',(sparse.diags(1/h)@C).tocsr()),
                    ('D',(-sparse.diags(1/h)@ct).tocsr()),('B',(sparse.diags(1/h)@K).tocsr()),
                    ('BT',BT.read(first,last,n)),('LN',(-sparse.diags(1/h)@K).tocsr()),
                    ('LD',(-sparse.diags(1/hd)@(K+KB)).tocsr())]:
                    sinks[key].append(matrix)
        for lo in range(0,nw,ROW_WINDOW):
            hi=min(lo+ROW_WINDOW,nw);ids=wallids[lo:hi];wall=geometry.take('wall',ids)
            R=_wall_rows(ids,wall,plane_paths,P,n)
            columns=R.indices[R.data!=0]
            if np.any(columns<a) or np.any(columns>=b):
                raise ValueError('Wall observation crosses its owner block')
            support[columns-a]=True
            sinks['Rwall'].append(R)
            wall_data['mu_wall'][lo:hi]=wall[:,6]
            wall_data['sg'][lo:hi]=wall[:,7]
            wall_data['leg'][lo:hi]=wall[:,7]*wall[:,8]
        width=fit['ids'].shape[1]
        if SOURCE_WINDOW*width*128>512*2**20:
            raise MemoryError('moment-reduced face quadrature fit batch exceeds the source working-byte budget')
        for lo in range(0,nr,SOURCE_WINDOW):
            hi=min(lo+SOURCE_WINDOW,nr)
            index = slice(lo,hi) if direct_recipe else routed[lo:hi]
            ids = recipe[index]
            F=geometry.take('faces',ids)
            if direct_recipe and (np.any(F[:,0]<a) or np.any(F[:,0]>=b)
                                  or np.any(F[:,1]<0) or np.any(F[:,1]>=n)):
                raise ValueError('moment-reduced face quadrature face differs from its construction owner or is nonphysical')
            record_data['ids'][lo:hi]=ids;record_data['mu'][lo:hi]=mu[index]
            record_data['ell'][lo:hi]=F[:,9]
            for side,foot,key in [(0,slice(5,7),'L'),(1,slice(7,9),'R')]:
                own=F[:,side].astype(np.int64);j=fit['index'][own%len(fit['index'])]
                if np.any(j<0):
                    raise ValueError('Missing moment-reduced face quadrature fit')
                d=basis((F[:,foot]-fit['center'][j])*N);v=np.matmul(d[:,None,:],fit['C'][j])[:,0,:]
                values=np.c_[1-v.sum(1),v];counts=fit['ncol'][j]
                valid=np.arange(width)[None,:]<counts[:,None]
                col=fit['ids'][j]+(own//P*P)[:,None]
                if int(np.sum(counts))>MAX_NNZ:
                    raise MemoryError('moment-reduced face quadrature batch exceeds the application entry bound')
                A=sparse.csr_matrix((values[valid],col[valid].astype(np.int32),
                                     np.r_[0,np.cumsum(counts)]),shape=(len(ids),n))
                sinks[key].append(A)
    blocks=[]
    for plane in range(a//P,b//P):
        start=plane*P-a
        ids=np.flatnonzero(support[start:start+P])
        if not len(ids):
            continue
        if 96*len(ids)**2>2**30:
            raise MemoryError('An indivisible plane factor exceeds the application workspace')
        M=rows(plane_paths[plane]/'M',0,P,n)[ids][:,ids+plane*P].tocsc()
        lu,piv=lu_factor(M.toarray())
        k=len(blocks);blocks.append(len(ids))
        for label,value in [('ids',ids+start),('lu',lu),('piv',piv)]:
            np.save(folder/f'block{k}_{label}.npy',value,allow_pickle=False)
    operators={}
    for key in names:
        requests=column_requests(folder,folder/key,edges,owner)
        operators[key]=dict(nnz=sinks[key].nnz,requests=requests)
    return dict(nodes=count,walls=nw,records=nr,blocks=blocks,operators=operators)


def _reduce(summaries,cache_chunk):
    parts=len(summaries)
    counts={key:[item[key] for item in summaries] for key in ('nodes','walls','records')}
    caps={key:max(1,max(values)) for key,values in counts.items()}
    caps['records']=((caps['records']+cache_chunk-1)//cache_chunk)*cache_chunk
    operators={}
    for key in summaries[0]['operators']:
        work=[row['operators'][key] for row in summaries]
        deltas={(p-int(peer))%parts for p,item in enumerate(work) for peer in item['requests']}
        operators[key]=dict(nnz=max(1,max(row['nnz'] for row in work)),delta=sorted(deltas),
                            width=max([0]+[v for row in work for v in row['requests'].values()]))
    blocks=max(1,max(len(s['blocks']) for s in summaries))
    width=max([1]+[v for s in summaries for v in s['blocks']])
    return caps,counts,operators,blocks,width


def _write_owner(product,raw,all_raw,owner,edges,reduction,summary,n,params):
    """Padded v3 owner descriptor, using bounded indexed parallel assembly's canonical halo permutations."""
    caps,_,ops,nb,w=reduction
    parts=len(all_raw);start,stop=map(int,edges[owner:owner+2])
    arrays={}
    disk_arrays={}
    def allocate(name,shape,dtype,packed=False,fill=0):
        diskshape=(1,*shape) if packed else shape
        data=mapped(product/(name+'.npy'),diskshape,dtype)
        if fill:
            data[...]=fill
        disk_arrays[name]=data
        return data[0] if packed else data

    def descriptor(name,value,packed=False):
        shape=[parts,*value.shape] if packed else [parts*len(value),*value.shape[1:]]
        path=product/(name+'.npy')
        disk_arrays.pop(name).flush()
        arrays[name]=dict(shape=shape,dtype=str(value.dtype),files={str(owner):_file(path)})
        return dict(kind='array',name=name)

    def _file(path):
        return dict(path=path.relative_to(product.parent).as_posix(),sha256=sha(path),
                    bytes=path.stat().st_size)

    def vector(name,cap,neutral=0,source=None):
        old=read(raw/((source or name)+'.npy'))
        dest=allocate(name,(cap,*old.shape[1:]),old.dtype,fill=neutral)
        for lo in range(0,len(old),SOURCE_WINDOW):
            hi=min(lo+SOURCE_WINDOW,len(old));dest[lo:hi]=old[lo:hi]
        return descriptor(name,dest)

    def csr(name,kind,transpose=False):
        spec=ops[name];rows=caps[kind];cols=caps['nodes']
        delta,width=spec['delta'],spec['width']
        ext=cols+len(delta)*width
        global_ids=n+np.arange(ext,dtype=np.int64)
        global_ids[:stop-start]=np.arange(start,stop)
        send=allocate(name+'_3',(len(delta),width),np.int32,True)
        valid=allocate(name+'_4',(len(delta),width),bool,True)
        for k,d in enumerate(delta):
            sender=(owner-d)%parts;src=raw/f'{name}_request_{sender}.npy'
            if src.is_file():
                rr=read(src);global_ids[cols+k*width:cols+k*width+len(rr)]=rr
            receiver=(owner+d)%parts;dest=all_raw[receiver]/f'{name}_request_{owner}.npy'
            if dest.is_file():
                rr=read(dest);send[k,:len(rr)]=rr-start;valid[k,:len(rr)]=True
        order=allocate(name+'_5',(ext,),np.int32,True)
        inverse=allocate(name+'_6',(ext,),np.int32,True)
        order[:]=np.argsort(global_ids,kind='stable')
        inverse[order]=np.arange(ext,dtype=np.int32)
        sorted_global=global_ids[order]
        old_data=mapbin(raw/name/'data.bin',np.float64)
        old_cols=mapbin(raw/name/'indices.bin',np.int32)
        old_ptr=mapbin(raw/name/'indptr.bin')
        nnz=len(old_data)
        if max(ext,rows,spec['nnz'])>=2**31:
            raise ValueError('Owner CSR exceeds signed int32 capacity')
        data=allocate(name+'_0',(spec['nnz'],),np.float64,True)
        idx=allocate(name+'_1',(spec['nnz'],),np.int32,True)
        ptr=allocate(name+'_2',(rows+1,),np.int32,True,fill=nnz)
        for lo in range(0,nnz,SOURCE_WINDOW):
            hi=min(lo+SOURCE_WINDOW,nnz)
            columns=old_cols[lo:hi]
            ci=np.searchsorted(sorted_global,columns).astype(np.int32)
            if np.any(ci>=ext) or not np.array_equal(sorted_global[ci],columns):
                raise ValueError('Missing owner halo column')
            data[lo:hi]=old_data[lo:hi];idx[lo:hi]=ci
        ptr[:len(old_ptr)]=old_ptr
        values=[data,idx,ptr,send,valid,order,inverse]
        return dict(kind='csr',rows=rows,cols=cols,delta=delta,width=width,
                    transpose=transpose,arrays=[descriptor(name+'_'+str(k),v,True)
                                                for k,v in enumerate(values)])

    ids=allocate('block_ids',(nb,w),np.int32,True)
    valid=allocate('block_valid',(nb,w),bool,True)
    lu=allocate('block_lu',(nb,w,w),np.float64,True)
    piv=allocate('block_piv',(nb,w),np.int32,True)
    support=allocate('block_support',(caps['nodes'],),bool)
    d=np.arange(w,dtype=np.int32)
    for k in range(nb):
        lu[k,d,d]=1.
        piv[k]=d
    for k,size in enumerate(summary['blocks']):
        ii=read(raw/f'block{k}_ids.npy');ids[k,:size]=ii
        valid[k,:size]=True;support[ii]=True
        lu[k,:size,:size]=read(raw/f'block{k}_lu.npy')
        piv[k,:size]=read(raw/f'block{k}_piv.npy')
    blocks=dict(kind='solve',n=caps['nodes'],arrays=[
        descriptor('block_'+key,value,key!='support') for key,value in
        [('ids',ids),('lu',lu),('piv',piv),('valid',valid),('support',support)]])
    scalar=lambda value:dict(kind='scalar',value=value,dtype='float64')
    p={key:vector(key,caps['nodes'],1) for key in ('H','B2')}
    p.update({key:vector('wall_'+key,caps['walls'],source='mu_wall' if key=='mu' else key)
              for key in ('mu','sg','leg')})
    p.update(blocks=blocks,tau=scalar(params[0]),mu_model=scalar(params[1]))
    for key in ('C','CT','G','D','R'):
        p[key]=csr('Rwall' if key=='R' else key,'walls' if key=='R' else 'nodes')
    if p['R']['delta']:
        raise ValueError('Wall observations cross owners')
    p['RT']=dict(kind='wall_transpose',source='R')
    tree=dict(p=dict(kind='dict',items=p))
    tree.update({key:csr(key,'nodes') for key in ('B','BT','LN','LD')})
    tree.update({key:csr(key,'records',True) for key in ('L','R')})
    meta={key:vector(key,caps['records'],1 if key=='ell' else -1 if key=='ids' else 0)
          for key in ('ell','mu','ids')}
    mask=allocate('valid',(caps['records'],),bool);mask[:summary['records']]=True
    meta['valid']=descriptor('valid',mask)
    tree['meta']=dict(kind='dict',items=meta)
    tree['wall_ids']=vector('wall_ids',caps['walls'],-1,'walls_ids')
    layouts={}
    for key,name in (('nodes','nodes_ids'),('walls','walls_ids'),('records','records_ids')):
        old=read(raw/(name+'.npy'))
        data=allocate(name,(caps[key],),np.int32,fill=-1);data[:len(old)]=old;data.flush()
        layouts[key]=_file(product/(name+'.npy'))
    return dict(tree=dict(kind='dict',items=tree),arrays=arrays,layouts=layouts)


def export_parallel_application(q_runtime,indexed_geometry,out,*,options,owner_p,
                          rhs_chunk=16384,cache_chunk=4096):
    """Export local Q actions using owner P leaves."""
    owner_p = OwnerPerpendicularBundle(owner_p)
    q_runtime=Path(q_runtime).resolve()
    manifest=q_runtime/'owners/manifest.json'
    build=json.loads(manifest.read_text())
    if build['format']!='drbx.parallel.owner_construction.v1':
        raise ValueError('Expected the indexed Q construction owner manifest')
    out=Path(out).resolve()
    if out.is_relative_to(q_runtime):
        raise ValueError('Use a separate attempt directory for the application export')
    if min(rhs_chunk,cache_chunk)<1:
        raise ValueError('Owner chunk sizes must be positive')
    owners=build['owners']
    parts=len(owners)
    if options.owner_count!=parts:
        raise ValueError('Q construction owner count differs from runtime owner_count')
    params=[options.tau,options.mu]
    from .construction_provenance import owner_plane_paths, verify_parallel_construction

    plane_paths = owner_plane_paths(q_runtime, build)
    fit_dir=(q_runtime/build['fits']).resolve()
    if not fit_dir.is_relative_to(q_runtime):
        raise ValueError('Q fit store escaped its root')
    fit={key:read(fit_dir/(key+'.npy')) for key in ('ids','ncol','index','C','center')}
    out.mkdir(parents=True,exist_ok=False)
    products=out/'products';products.mkdir()
    with closing(IndexedInput(indexed_geometry)) as geometry:
        identity=geometry.identity()
        if build['index_binding']!=identity:
            raise ValueError('Construction and application indexed geometry differ')
        meta=json.loads((geometry.handoff.parent/'geometry.json').read_text())
        N,E,P,n=(int(geometry.info[key]) for key in ('N','E','P','NN'))
        opts=(options.resolution,options.symmetry,options.field_periods)
        expected=(N,meta['symmetry'],meta['field_periods'])
        if any(a not in (None,b) for a,b in zip(opts,expected)) or E!=len(plane_paths):
            raise ValueError('Application options differ from the Q construction')
        edges=np.array([((p*E+parts-1)//parts)*P for p in range(parts+1)],dtype=np.int64)
        if ([item['begin']*P for item in owners]!=edges[:-1].tolist()
                or n!=E*P or n>=2**31):
            raise ValueError('Invalid application owner plane layout')
        B_hash = hashlib.sha256()
        moment_reduced_face_quadrature = verify_parallel_construction(build, geometry, plane_paths)
        degree, recipe_sha = moment_reduced_face_quadrature['degree'], moment_reduced_face_quadrature['sha256']
        recipe = mu = None
        if MOMENT_REDUCED_FACE_QUADRATURE_DEGREE != degree:
            raise ValueError('Invalid moment-reduced face quadrature recipe degree')
        plane_edges = tuple(int(edge)//P for edge in edges)
        if (owner_p.edges != plane_edges
                or tuple(owner_p.manifest[key] for key in ('N','symmetry','field_periods','P'))
                != (*expected, P)):
            raise ValueError('perpendicular and parallel owner layouts differ')
        layout = build_zernike_core_ring_layout(N, n_eta=E)
        scale = max(1., *(float(np.max(np.abs(values))) for values in
                         (layout.node_u, layout.node_theta, eta_planes(layout))))
        tolerance = 64*np.finfo(float).eps*scale
        routes=build_parallel_routes(out,geometry,plane_paths,parts,P,n)
        folders=[];summaries=[]
        for p,route in enumerate(routes):
            folder=out/f'raw_{p:05d}';folder.mkdir();folders.append(folder)
            a,b = map(int,edges[p:p+2])
            field = _read_owner_magnetic_field(owner_p,p,node_points(layout,planes=slice(*plane_edges[p:p+2])),tolerance,b-a)
            B_hash.update(np.ascontiguousarray(field))
            summaries.append(_raw_owner(folder,p,edges,route,plane_paths,geometry,fit,recipe,mu,field,(N,P,n)))
    reduction=_reduce(summaries,cache_chunk)
    caps,counts,_,_,_=reduction
    desc=[]
    for p,raw in enumerate(folders):
        product=products/f'owner{p:04d}';product.mkdir()
        desc.append(_write_owner(product,raw,folders,p,edges,reduction,summaries[p],n,params))
    arrays=desc[0]['arrays']
    tree=desc[0]['tree']
    for name,array in arrays.items():
        if any(item['arrays'][name]['shape']!=array['shape'] for item in desc):
            raise ValueError('Application owner shapes differ')
        array['files']={str(p):item['arrays'][name]['files'][str(p)] for p,item in enumerate(desc)}
    if any(item['tree']!=tree for item in desc):
        raise ValueError('Application owner product trees differ')
    layouts={key:dict(n=sum(counts[key]),cap=caps[key],
                      files={str(p):item['layouts'][key] for p,item in enumerate(desc)}) for key in caps}
    structure=dict(owned_node_counts=counts['nodes'],owned_wall_counts=counts['walls'],
                   owned_record_counts=counts['records'],owner_node_edges=edges.tolist(),
                   record_cap=caps['records'],record_order='canonical moment-reduced face quadrature ordinal within owner',
                   sparse_exchange='cyclic ppermute of requested columns')
    atomic_json(products/'metadata.json',dict(version=3,N=N,eta_planes=E,symmetry=meta['symmetry'],
                field_periods=meta['field_periods'],owner_count=parts,params=params,face_quadrature_degree=degree,
                rhs_chunk=rhs_chunk,cache_chunk=cache_chunk,tree=tree,arrays=arrays,layouts=layouts,
                structure=structure,source_manifest_sha256=sha(manifest),indexed_geometry=identity,
                recipe_sha256=recipe_sha,B_sha256=B_hash.hexdigest(),complete=True))
    return products
