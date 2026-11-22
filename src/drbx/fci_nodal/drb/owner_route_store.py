"""Publish owner route leaves with bounded row reads and load addressable pieces."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
from scipy import sparse

from ..operators.parallel.owner_products import LocalLayout
from .._hashing import sha256_file as digest
from ..operators.parallel.owner_sparse import HaloCSR
from ..operators.parallel.owner_routes import Requests, csr_piece, locate, request_layout
from .afc import OwnerLimiterGraph
from .wall_plan import OwnerWallGrid
from .walls import WallOps


SCHEMA = 'drbx.parallel.owner_routes.v1'
MATRICES = ('T','TT','Tc','TcT','F','K')
NODE_FIELDS = ('H',)
WALL_FIELDS = ('Js','curvature_flux','hw','bn','trace_gram','mass_trace_gram','A0plus')
HIT_FIELDS = ('Y','Khit','wall_foot_to_hit_area_ratio')


def addressable(mesh):
    return [(i,d) for i,d in enumerate(mesh.devices.flat) if d.process_index==jax.process_index()]


def replicate(mesh, value):
    return jax.make_array_from_process_local_data(NamedSharding(mesh,P()),np.asarray(value))


def read_checked(root, entry):
    root = Path(root).resolve()
    path = (root/entry['path']).resolve()
    if not path.is_relative_to(root) or digest(path)!=entry['sha256']:
        raise ValueError(f'Invalid rank-local route input {path}')
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def _entry(path, root):
    return dict(path=path.relative_to(root).as_posix(),sha256=digest(path))


def _owner_layout(meta, root, key):
    desc = meta['layouts'][key]
    parts = meta['owner_count']
    ids = {}
    for i in range(parts):
        entry = desc['files'][str(i)]
        path = (root/entry['path']).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path)!=entry['sha256']:
            raise ValueError('Owner layout checksum changed')
        ids[i] = np.load(path,allow_pickle=False)
    return LocalLayout(desc['n'],desc['cap'],parts,ids)


def publish_routes(out, wall, q_products, q, *, wall_per_plane):
    """Read one raw CSR/edge slab at a time; retain only global request metadata."""
    out,wall,q_products = map(Path,(out,wall,q_products))
    out.mkdir()
    meta = q.meta
    parts = meta['owner_count']
    metric = SimpleNamespace(size=parts)  # request_layout needs only the owner count.
    nodes = _owner_layout(meta,q_products,'nodes')
    cuts = tuple(q.edges)
    counts = tuple(b-a for a,b in zip(cuts[:-1],cuts[1:]))
    nw = wall_per_plane
    cap = max(counts)*nw
    p_walls = LocalLayout(cuts[-1]*nw,cap,parts,{
        p:np.pad(np.arange(a*nw,b*nw,dtype=np.int32),(0,cap-(b-a)*nw),constant_values=-1)
        for p,(a,b) in enumerate(zip(cuts[:-1],cuts[1:]))})
    files = {p:{} for p in range(parts)}
    raw = {p:_entry(wall/f'owner{p:04d}/wall_plan.npz',wall) for p in range(parts)}
    limiter = {p:_entry(wall/f'owner{p:04d}/limiter_graph.npz',wall) for p in range(parts)}
    csr = {}
    for p in range(parts):
        (out/f'owner{p:04d}').mkdir()

    def write(p,name,**leaves):
        path = out/f'owner{p:04d}'/f'{name}.npz'
        np.savez(path,**leaves)
        files[p][name] = _entry(path,out)

    def rows(p,name):
        if name=='afc_K':
            A = q.K(p,node_ids=nodes.ids_by_owner[p][nodes.ids_by_owner[p]>=0]).copy()
            A.eliminate_zeros()
        else:
            path = wall/raw[p]['path']
            with np.load(path,allow_pickle=False) as z:
                A = sparse.csr_matrix((z[name+'_data'],z[name+'_indices'],z[name+'_indptr']),
                                      shape=tuple(z[name+'_shape']))
        A.sum_duplicates()
        A.sort_indices()
        return A

    for name,source,target in (
        ('T',nodes,p_walls),('TT',p_walls,nodes),('Tc',nodes,p_walls),('TcT',p_walls,nodes),
        ('F',nodes,p_walls),('K',nodes,nodes),('afc_K',nodes,nodes),
    ):
        requests, nnz = {}, 1
        for p in range(parts):
            A = rows(p,name)
            if name!='afc_K':
                halo = {'F':2,'K':4}.get(name,0)
                width_source = nw if source is p_walls else q.width
                width_target = nw if target is p_walls else q.width
                ids = target.ids_by_owner[p]
                rr = np.repeat(ids[ids>=0],np.diff(A.indptr))//width_target
                cc = A.indices//width_source
                E = cuts[-1]
                if np.any(np.minimum((rr-cc)%E,(cc-rr)%E)>halo):
                    raise ValueError(f'{name} exceeds its periodic eta halo')
            requests[p] = np.unique(A.indices)
            nnz = max(nnz,A.nnz)
        slots,send,valid,delta,width = request_layout(metric,source,requests)
        csr[name] = dict(rows=target.cap,cols=source.cap,delta=delta,width=width)
        for p in range(parts):
            A = rows(p,name)
            piece = csr_piece(p,A,source,target,requests,slots,send,valid,delta,width,nnz)
            leaves = {str(i):a for i,a in enumerate(piece)}
            if name=='afc_K':
                r = np.repeat(np.arange(A.shape[0],dtype=np.int32),np.diff(A.indptr))
                leaves['row'] = np.pad(r,(0,nnz-len(r)),constant_values=nodes.cap)
            write(p,name,**leaves)

    # Canonical edge ordinals and their endpoints are metadata; weights and
    # diagonals stay in one owner's file at a time.
    edge_ids, other, left = {}, {}, {}
    receivers = {p:[] for p in range(parts)}
    for p in range(parts):
        with np.load(wall/limiter[p]['path'],allow_pickle=False) as z:
            ordinal,i,j = (z[k] for k in ('id','i','j'))
        src,local = locate(nodes,i)
        dst,remote = locate(nodes,j)
        if np.any(src!=p) or np.any(i>=j) or np.any(ordinal[1:]<=ordinal[:-1]):
            raise ValueError('Noncanonical limiter edge')
        edge_ids[p],other[p],left[p] = ordinal,j,local
        for peer in np.unique(dst):
            take = dst==peer
            receivers[int(peer)].append((ordinal[take],remote[take]))
    edge_cap = max(1,max(map(len,edge_ids.values())))
    edge_layout = LocalLayout(sum(map(len,edge_ids.values())),edge_cap,parts,{
        p:np.pad(v,(0,edge_cap-len(v)),constant_values=-1) for p,v in edge_ids.items()})
    reverse, right = {}, {}
    for p, items in receivers.items():
        ids = np.concatenate([a for a,_ in items]) if items else np.empty(0,np.int64)
        loc = np.concatenate([b for _,b in items]) if items else np.empty(0,np.int32)
        order = np.argsort(ids,kind='stable')
        reverse[p],right[p] = ids[order],loc[order]
    reverse_cap = max(1,max(map(len,reverse.values())))
    fslot,fsend,_,fd,fw = request_layout(metric,nodes,other)
    rslot,rsend,_,rd,rw = request_layout(metric,edge_layout,reverse)
    for p in range(parts):
        with np.load(wall/limiter[p]['path'],allow_pickle=False) as z:
            weights,diag = z['v'],z['diag']
        pad = edge_cap-len(weights)
        write(p,'afc',i=np.pad(left[p],(0,pad),constant_values=nodes.cap),
              j=np.pad(right[p],(0,reverse_cap-len(right[p])),constant_values=nodes.cap),
              v=np.pad(weights,(0,pad)),diag=diag,edge_valid=np.arange(edge_cap)<len(weights),
              fslot=np.pad(fslot[p],(0,edge_cap-len(fslot[p]))),fsend=fsend[p],
              rslot=np.pad(rslot[p],(0,reverse_cap-len(rslot[p]))),rsend=rsend[p])
    manifest = dict(schema=SCHEMA,owner_count=parts,edges=cuts,wall_per_plane=nw,csr=csr,files=files,
                    wall=raw,limiter=limiter,forward=(fd,fw),reverse=(rd,rw),
                    parallel_sha256=digest(q_products/'metadata.json'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out


@dataclass(frozen=True)
class RouteStore:
    root: Path
    wall: Path
    manifest: dict

    @classmethod
    def open(cls, root, wall):
        root = Path(root)
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest['schema']!=SCHEMA:
            raise ValueError('Unknown owner route schema')
        return cls(root,Path(wall),manifest)

    def read(self, owner, name):
        entry = self.manifest['files'][str(owner)][name]
        if entry['path']!=f'owner{owner:04d}/{name}.npz':
            raise ValueError('Route leaf is not in the selected owner directory')
        return read_checked(self.root,entry)

    def wall_rows(self, owner):
        entry = self.manifest['wall'][str(owner)]
        if entry['path']!=f'owner{owner:04d}/wall_plan.npz':
            raise ValueError('Wall cache is not in the selected owner directory')
        return read_checked(self.wall,entry)


def route_plans(store, mesh, nodes, p_walls, hits, *, edges, deta, q_products, constants):
    md = store.manifest
    counts = tuple(b-a for a,b in zip(edges[:-1],edges[1:]))
    if (tuple(md['edges'])!=edges or md['owner_count']!=mesh.size
            or md['wall_per_plane']!=p_walls.n//edges[-1]
            or md['parallel_sha256']!=digest(Path(q_products)/'metadata.json')):
        raise ValueError('Owner route lineage differs from the run')
    local = addressable(mesh)
    if not local:
        raise ValueError('No addressable owner route')
    owners = [i for i,_ in local]
    rows = {i:store.wall_rows(i) for i in owners}
    sharding = NamedSharding(mesh,P('records'))

    def put(values):
        first = values[owners[0]]
        pieces = [jax.device_put(values[i][None],d) for i,d in local]
        return jax.make_array_from_single_device_arrays((mesh.size,*first.shape),sharding,pieces)

    def routed(name):
        plan = {i:store.read(i,name) for i in owners}
        maps = tuple(put({i:plan[i][str(k)] for i in owners}) for k in range(10))
        return HaloCSR(maps,mesh,**md['csr'][name]), plan

    maps = {name:routed(name)[0] for name in MATRICES}
    for layout,fields in ((nodes,NODE_FIELDS),(p_walls,WALL_FIELDS),(hits,HIT_FIELDS)):
        for name in fields:
            pad = 1 if name in ('H','Js','trace_gram','mass_trace_gram','wall_foot_to_hit_area_ratio') else 0
            maps[name] = layout.put({i:rows[i][name] for i in owners},mesh,pad)
    maps.update({name:replicate(mesh,value) for name,value in constants.items()})
    k, k_plan = routed('afc_K')
    afc = {i:store.read(i,'afc') for i in owners}

    def leaf(name):
        return put({i:afc[i][name] for i in owners})

    graph = OwnerLimiterGraph(k,put({i:k_plan[i]['row'] for i in owners}),
        leaf('i'),leaf('j'),leaf('v'),nodes.put({i:afc[i]['diag'] for i in owners},mesh),
        nodes.put({i:nodes.ids_by_owner[i]>=0 for i in owners},mesh),leaf('edge_valid'),
        Requests(leaf('fslot'),leaf('fsend'),tuple(md['forward'][0]),md['forward'][1]),
        Requests(leaf('rslot'),leaf('rsend'),tuple(md['reverse'][0]),md['reverse'][1]))
    return WallOps(**maps),OwnerWallGrid(counts,md['wall_per_plane'],float(deta)),graph
