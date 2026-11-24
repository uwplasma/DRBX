"""bounded indexed parallel assembly canonical source-window routing and disk CSR rows for application."""
from contextlib import closing
from heapq import merge
from itertools import islice
import sqlite3

import numpy as np
from scipy import sparse

from .job_storage import mapped, read


ROW_WINDOW=256
SOURCE_WINDOW=32000
MAX_NNZ=1_000_000


def append(path, data, dtype):
    with path.open('ab') as stream:
        np.asarray(data,dtype).tofile(stream)


def mapbin(path,dtype=np.int64):
    return np.memmap(path,dtype=dtype,mode='r') if path.stat().st_size else np.empty(0,dtype)


class Sink:
    """Append a bounded CSR window and expose disk-mapped owner rows."""
    def __init__(self,path,columns):
        self.path=path
        self.columns=columns
        path.mkdir()
        for name in ('data','indices','indptr'):
            (path/(name+'.bin')).touch()
        self.rows=self.nnz=0

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        if exc[0] is None:
            append(self.path/'indptr.bin',[self.nnz],np.int64)

    def append(self,A):
        A=A.tocsr()
        A.sum_duplicates();A.sort_indices()
        if A.nnz>MAX_NNZ:
            raise MemoryError('Application CSR window exceeds its entry bound')
        append(self.path/'data.bin',A.data,np.float64)
        append(self.path/'indices.bin',A.indices,np.int32)
        append(self.path/'indptr.bin',A.indptr[:-1]+self.nnz,np.int64)
        self.nnz+=A.nnz
        self.rows+=A.shape[0]


def rows(folder,begin,end,n):
    ptr=read(folder/'indptr.npy')[begin:end+1]
    lo,hi=int(ptr[0]),int(ptr[-1])
    if hi-lo>MAX_NNZ:
        raise MemoryError('Construction CSR row window exceeds application budget')
    data=read(folder/'data.npy')[lo:hi].copy()
    ids=read(folder/'indices.npy')[lo:hi].copy()
    return sparse.csr_matrix((data,ids,ptr-lo),shape=(end-begin,n))


def _route_face_pair_quadrature(paths, plane_paths, parts):
    """Merge per-plane IDs in the construction export's canonical order."""
    E = len(plane_paths)
    dtype = [('ids', np.int64), ('mu', np.float64), ('owner', np.int64)]

    def plane_stream(folder, owner):
        ids = read(folder/'moment_reduced_face_quadrature.npy')
        mu = read(folder/'mu.npy')
        if ids.ndim != 1 or ids.dtype != np.int64 or mu.shape != ids.shape or mu.dtype != np.float64:
            raise ValueError('Invalid owner moment-reduced face quadrature recipe leaves')
        for face, weight in zip(ids, mu, strict=True):
            yield int(face), float(weight), owner

    streams = [plane_stream(folder, plane*parts//E) for plane, folder in enumerate(plane_paths)]
    last = -1
    merged = merge(*streams, key=lambda row: row[0])
    ordinal = 0
    while batch := list(islice(merged, SOURCE_WINDOW)):
        data = np.asarray(batch, dtype=dtype)
        ids = data['ids']
        if ids[0] <= last or np.any(ids[1:] <= ids[:-1]):
            raise ValueError('Invalid canonical moment-reduced face quadrature recipe order')
        last = int(ids[-1])
        for owner in np.unique(data['owner']):
            take = np.flatnonzero(data['owner'] == owner)
            append(paths[owner]/'record_ordinals.bin', ordinal + take, np.int64)
            append(paths[owner]/'record_ids.bin', ids[take], np.int64)
            append(paths[owner]/'record_mu.bin', data['mu'][take], np.float64)
        ordinal += len(batch)
        if ordinal >= 2**31:
            raise ValueError('moment-reduced face quadrature recipe exceeds signed int32 capacity')


def build_parallel_routes(root,geometry,plane_paths,parts,P,n):
    """Source windows ordered as bounded indexed parallel assembly, with direct append to owner files."""
    E=len(plane_paths)
    names=['wall_ids','record_ordinals']
    names.extend(('record_ids', 'record_mu'))
    names.extend(key+'_'+suffix for key in ('C','K_N') for suffix in ('row','col','value'))
    paths=[]
    for owner in range(parts):
        folder=root/f'route_{owner:05d}'
        folder.mkdir()
        for key in names:
            (folder/(key+'.bin')).touch()
        paths.append(folder)

    def partition(nodes):
        return np.minimum((nodes//P)*parts//E,parts-1)

    for lo,hi in geometry.ranges('wall',SOURCE_WINDOW):
        nodes=geometry.read('wall',lo,hi)[:,0].astype(np.int64)
        owners=partition(nodes)
        for p in np.unique(owners):
            ids=lo+np.flatnonzero(owners==p)
            append(paths[p]/'wall_ids.bin',ids,np.int64)
    _route_face_pair_quadrature(paths, plane_paths, parts)
    for plane,folder in enumerate(plane_paths):
        H=read(folder/'wall_credited_parallel_mass.npy')
        for key,source in [('C','C'),('K_N','K')]:
            for lo in range(0,P,ROW_WINDOW):
                hi=min(lo+ROW_WINDOW,P)
                A=rows(folder/source,lo,hi,n)
                if key=='K_N':
                    A=(sparse.diags(1/H[lo:hi])@A).tocsr()
                co=A.tocoo()
                owners=partition(co.col)
                for p in np.unique(owners):
                    take=owners==p
                    append(paths[p]/(key+'_row.bin'),co.col[take],np.int64)
                    append(paths[p]/(key+'_col.bin'),plane*P+lo+co.row[take],np.int64)
                    append(paths[p]/(key+'_value.bin'),co.data[take],np.float64)
    return paths


class TransposeRows:
    """bounded indexed parallel assembly external routing, retaining the original COO order and value bytes."""
    def __init__(self,folder,key):
        self.row=mapbin(folder/(key+'_row.bin'))
        self.col=mapbin(folder/(key+'_col.bin'))
        self.value=mapbin(folder/(key+'_value.bin'),np.float64)
        self.db=sqlite3.connect(folder/(key+'_rowindex.sqlite'))
        self.db.execute('pragma temp_store=FILE')
        self.db.execute('pragma cache_size=-8192')
        self.db.execute('create table offsets(row integer,ordinal integer primary key)')
        for lo in range(0,len(self.row),SOURCE_WINDOW):
            self.db.executemany('insert into offsets values (?,?)',
                  ((int(v),lo+i) for i,v in enumerate(self.row[lo:lo+SOURCE_WINDOW])))
        self.db.execute('create index row_idx on offsets(row,ordinal)')
        self.db.commit()

    def read(self,a,b,n):
        count=self.db.execute('select count(*) from offsets where row>=? and row<?',(a,b)).fetchone()[0]
        if count>MAX_NNZ:
            raise MemoryError('Application transpose window exceeds its entry bound')
        ordinal=np.fromiter((row[0] for row in self.db.execute(
             'select ordinal from offsets where row>=? and row<? order by ordinal',(a,b))),
             np.int64,count=count)
        return sparse.csr_matrix((self.value[ordinal],(self.row[ordinal]-a,self.col[ordinal])),
                                 shape=(b-a,n))

    def close(self):
        self.db.close()


def column_requests(folder,path,edges,owner):
    """Remote columns sorted by canonical ID, with a file-backed unique index."""
    requests={}
    columns=mapbin(path/'indices.bin',np.int32)
    with closing(sqlite3.connect(folder/(path.name+'_columns.sqlite'))) as db:
        db.execute('pragma temp_store=FILE')
        db.execute('pragma cache_size=-8192')
        db.execute('create table ids(id integer primary key)')
        for lo in range(0,len(columns),SOURCE_WINDOW):
            db.executemany('insert or ignore into ids values (?)',
                           ((int(c),) for c in np.unique(columns[lo:lo+SOURCE_WINDOW])))
        for peer,(start,stop) in enumerate(zip(edges[:-1],edges[1:])):
            if peer==owner:
                continue
            query=' from ids where id>=? and id<?'
            bounds=(int(start),int(stop))
            size=db.execute('select count(*)'+query,bounds).fetchone()[0]
            if size:
                dest=mapped(folder/f'{path.name}_request_{peer}.npy',(size,),np.int32)
                cur=db.execute('select id'+query+' order by id',bounds)
                for lo in range(0,size,SOURCE_WINDOW):
                    batch=np.asarray(cur.fetchmany(SOURCE_WINDOW),np.int32).ravel()
                    dest[lo:lo+len(batch)]=batch
                requests[str(peer)]=size
    return requests
