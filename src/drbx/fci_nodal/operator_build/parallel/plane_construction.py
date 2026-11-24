"""bounded indexed parallel assembly owner windows, wall batches, and stable left-owner selection."""
import sqlite3

import numpy as np
from scipy import sparse

from .face_operators import csr_rows
from .indexed_partials import Indexed, Tree
from .jump_records import group_keys, select
from .mass import wall_batch
from .indexed_input import ROWS
from .job_storage import csr_leaf, mapped, read
from .pair_cubature import reduce_pair


FACE_FIELDS = ('K', 'C', 'H0')


def face_windows(folder, layout, a, b, leaves, plane):
    """Each plane starts its own 256-row windows, as in the gathered build."""
    n = layout.nodes
    blocks = dict(K=[], C=[])
    h0 = np.zeros(b-a)
    for lo in range(a, b, 256):
        hi = min(lo+256, b)
        trees = {key: Tree() for key in FACE_FIELDS}
        for leaf in leaves:
            path = leaf / f'plane_{plane:05d}' / f'{lo:012d}.npz'
            if path.is_file():
                with np.load(path, allow_pickle=False) as data:
                    for key, tree in trees.items():
                        tree.push(Indexed.get(data, key+'_'))
            else:
                for tree in trees.values():
                    tree.push(None)
        for key in ('K', 'C'):
            part = trees[key].total()
            blocks[key].append(csr_rows(part, lo, hi, n) if part is not None
                               else sparse.csr_matrix((hi-lo, n)))
        h = trees['H0'].total()
        if h is not None:
            h0[h.rows-a] = h.value
    for key in ('K', 'C'):
        csr_leaf(folder / key, sparse.vstack(blocks[key], format='csr'))
    np.save(folder / 'H0.npy', h0, allow_pickle=False)


def _take_all(reader, name, ids):
    shape = reader.shape(name)
    data = np.empty((len(ids), *shape[1:]), np.dtype(reader.entries[name]['dtype']))
    for lo in range(0, len(ids), ROWS):
        data[lo:lo+ROWS] = reader.take(name, ids[lo:lo+ROWS])
    return data


def mass_plane(folder, reader, layout, table, ev, a, b, ids):
    """Placed mass/mass-diffusion arithmetic in bounded indexed parallel assembly's owner-local layout."""
    n = layout.nodes
    wall = _take_all(reader, 'wall', ids)
    nodes=b-a
    rr=np.empty(len(wall));Vwall=np.empty(len(wall));parts=[]
    for lo in range(0,len(wall),1024):
        batch=slice(lo,lo+1024)
        R,rr[batch],Vwall[batch]=wall_batch(ev,table,layout,wall[batch],a,nodes)
        parts.append(R)
    R=sparse.vstack(parts,format='csr') if parts else sparse.csr_matrix((0,nodes))
    CSC=R.tocsc();H0=read(folder/'H0.npy');gram=[]
    fields={key: mapped(folder/(key+'.npy'),(nodes,),np.float64) for key in ('H','wall_credited_parallel_mass','MB_V')}
    for lo in range(a,b,256):
        hi=min(b,lo+256);local_lo=lo-a;local_hi=hi-a
        ii=np.unique(CSC.indices[CSC.indptr[local_lo]:CSC.indptr[local_hi]])
        selected=R[ii];cols=np.unique(selected.indices);right=selected[:,cols];left=selected[:,local_lo:local_hi];w=wall[ii];mu=w[:,6];ell=w[:,8];coef=mu*rr[ii]/ell
        KB=(left.T@right.multiply(coef[:,None])).tocsr();KB.indices=(cols[KB.indices]+a).astype(KB.indices.dtype,copy=False);KB._shape=(hi-lo,n);gram.append(KB)
        V=left.T@Vwall[ii];credit=left.T@(mu*ell);H=reader.read('H',lo,hi);Hw=H0[local_lo:local_hi]+credit
        if Hw.min()<=0 or (Hw-V).min()<=0:raise ArithmeticError('Mass positivity changed')
        for key,value in dict(H=H,wall_credited_parallel_mass=Hw,MB_V=V).items():
            fields[key][local_lo:local_hi]=value
    # Same plane pair block and Cholesky certificate as the placed gathered path.
    H=fields['wall_credited_parallel_mass']
    M=(sparse.diags(H)-R.T@sparse.diags(wall[:,6]*wall[:,8])@R).tocsr()
    support=np.flatnonzero(np.asarray(abs(R).sum(0)).ravel()>0)
    if len(support):
        A=M[support][:,support].toarray()/np.sqrt(H[support,None]*H[None,support])
        np.linalg.cholesky(A)
    M.indices+=a;M._shape=(nodes,n)
    R.indices+=a;R._shape=(len(ids),n)
    csr_leaf(folder/'M',M);csr_leaf(folder/'Rw',R);csr_leaf(folder/'MB_Gram',sparse.vstack(gram,format='csr'))
    for name,value in (('wall_ids.npy',ids),('wall_foot_to_hit_area_ratio.npy',rr),('V.npy',Vwall)):
        np.save(folder/name,value,allow_pickle=False)


def _vector_groups(reader, ids, layout):
    F = _take_all(reader, 'faces', ids)
    prov = _take_all(reader, 'face_splice_provenance', ids)
    keys = group_keys(F, prov, layout.N)
    return select(F, keys, ids, 2)


def _external_groups(folder, reader, ids, layout):
    """bounded indexed parallel assembly external stable grouping, with one intact numerical group per call."""
    db_path = folder / 'pair_groups.sqlite'
    with sqlite3.connect(db_path) as db:
        db.execute('pragma temp_store=FILE')
        db.execute('pragma cache_size=-8192')
        columns = ','.join('k'+str(i) for i in range(4))
        schema = ','.join('k'+str(i)+' integer' for i in range(4))
        db.execute(f'create table records(id integer primary key,{schema},xy blob,weight blob)')
        placeholders = ','.join('?' for _ in range(7))
        for lo in range(0, len(ids), ROWS):
            index = ids[lo:lo+ROWS]
            F = reader.take('faces', index)
            keys = group_keys(F, reader.take('face_splice_provenance', index), layout.N)
            xy = np.c_[F[:,2]*np.cos(F[:,3]),F[:,2]*np.sin(F[:,3])]
            db.executemany(f'insert into records values ({placeholders})',
                           ((int(idx), *map(int, key), xy[j].tobytes(), np.asarray(F[j,10]).tobytes())
                            for j,(idx,key) in enumerate(zip(index,keys))))
        db.execute(f'create index ordered on records({columns},id)')
        db.commit()
        pairs = 0
        where = ' and '.join('k'+str(i)+'=?' for i in range(4))
        schedule = 'count(*) desc,' + columns
        groups = db.execute(f'select {columns},count(*) from records where 1=1 group by {columns} order by {schedule}')
        selections = folder/'moment_reduced_face_quadrature_runs.bin'
        with selections.open('wb') as stream:
            for *key,count in groups:
                if count * 768 > 512*2**20:
                    raise MemoryError('One indivisible Q pair group exceeds the bounded indexed parallel assembly workspace bound')
                rows = db.execute(f'select id,xy,weight from records where {where} order by id',key).fetchall()
                xy = np.frombuffer(b''.join(row[1] for row in rows),np.float64).reshape(-1,2)
                weight = np.frombuffer(b''.join(row[2] for row in rows),np.float64)
                selected, mu, _ = reduce_pair(xy,weight,2)
                values = np.empty(len(selected),dtype=[('id','i8'),('mu','f8')])
                values['id'] = [rows[j][0] for j in selected]
                values['mu'] = mu
                values.tofile(stream)
                pairs += 1
        if not selections.stat().st_size:
            return np.empty(0,np.int64), np.empty(0,np.float64), pairs
        values = np.memmap(selections,dtype=[('id','i8'),('mu','f8')],mode='r')
        # Source IDs are unique across groups, so this sort is only an ordinal permutation.
        order = np.argsort(values['id'])
        return values['id'][order],values['mu'][order], pairs


def recipes(folder, reader, ids, layout):
    if len(ids)*768 <= 512*2**20:
        moment_reduced_face_quadrature, mu, pairs = _vector_groups(reader, ids, layout)
    else:
        moment_reduced_face_quadrature, mu, pairs = _external_groups(folder, reader, ids, layout)
    for name,value in (('moment_reduced_face_quadrature.npy',moment_reduced_face_quadrature),('mu.npy',mu)):
        np.save(folder/name,value,allow_pickle=False)
    return pairs
