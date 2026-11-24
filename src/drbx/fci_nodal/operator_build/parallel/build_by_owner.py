"""Local bounded indexed parallel assembly construction graph for indexed Q geometry."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing, nullcontext
from dataclasses import asdict, replace
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import scipy

from .face_operators import face_leaf
from ...evaluators.magnetic_field import Evaluator as _Evaluator
from ...operators.parallel.geometry_io import Layout
from .plane_fits import MappedFits, fit_batch, fit_export
from .indexed_input import IndexedInput, ROWS
from ..._hashing import sha256_file as sha
from .plane_construction import face_windows, mass_plane, recipes
from .construction_provenance import face_quadrature_provenance, indexed_layout, owner_plane_paths
from .job_storage import atomic_json, bounded_map, fingerprint, guard_rss, read, recover, work


def Evaluator(options):
    check_periods = options.symmetry == 'nfp'
    return _Evaluator(options.metric_cache, options.field_table, filter_spec=options.filter_spec,
                      field_periods=options.field_periods if check_periods else None,
                      require_periods=check_periods, cache_derivative_batches=True)


_CTX = {}


class Faces:
    def __init__(self, reader):
        self.reader = reader

    def __len__(self):
        return self.reader.shape('faces')[0]

    def __getitem__(self, item):
        lo, hi, step = item.indices(len(self))
        if step != 1:
            raise ValueError('Face kernels use contiguous source chunks')
        return self.reader.read('faces', lo, hi)


def initialize(options, layout, edges, limit):
    guard_rss(limit)
    _CTX.clear()
    _CTX.update(options=options, layout=layout, edges=edges,
                geometry=IndexedInput(options.geometry))


def route_chunk(folder, arg):
    kind, lo, hi = arg
    reader, layout = _CTX['geometry'], _CTX['layout']
    values = reader.read(kind, lo, hi)
    ids = values[:, :2].astype(np.int64) if kind == 'faces' else values[:,0].astype(np.int64)
    keys = np.unique(ids % layout.period_nodes)
    planes = (ids[:,0] if kind == 'faces' else ids)//layout.P
    route = dict(keys=keys, planes=planes, lo=np.asarray(lo))
    np.savez(folder/'route.npz', **route)


def _route_export(folder, routes, layout):
    keys = np.zeros(layout.period_nodes,bool)
    trace = np.zeros_like(keys)
    for kind, paths in routes.items():
        for plane in range(layout.eta_planes):
            (folder/f'{kind}_{plane:05d}.bin').touch()
        for path in paths:
            with np.load(path/'route.npz',allow_pickle=False) as data:
                selected = data['keys']
                keys[selected] = True
                if kind == 'wall':
                    trace[selected] = True
                planes = data['planes']
                lo = data['lo']
                for plane in np.unique(planes):
                    with (folder/f'{kind}_{int(plane):05d}.bin').open('ab') as stream:
                        (np.flatnonzero(planes==plane).astype(np.int64) + lo).tofile(stream)
    np.save(folder/'keys.npy',np.flatnonzero(keys),allow_pickle=False)
    np.save(folder/'trace.npy',trace,allow_pickle=False)


def _fit(folder, arg):
    keys, route = arg
    return fit_batch(folder, _CTX['layout'], _CTX['options'], keys, read(route/'trace.npy'))


def _leaf(folder, arg):
    lo, hi, fit = arg
    geometry, options, layout = (_CTX[k] for k in ('geometry','options','layout'))
    table = MappedFits(layout, fit)
    leaves = face_leaf(lo,hi,options,Faces(geometry),table)
    for plane,(a,b) in enumerate(zip(_CTX['edges'][:-1],_CTX['edges'][1:])):
        owned = {key:part.subset(a,b) for key,part in leaves.items()}
        if not any(len(part.rows) for part in owned.values()):
            continue
        plane_path = folder/f'plane_{plane:05d}'
        plane_path.mkdir()
        for start in range(a,b,256):
            arrays = {}
            for key,part in owned.items():
                part.subset(start,min(start+256,b)).put(arrays,key+'_')
            np.savez(plane_path/f'{start:012d}.npz',**arrays)
    return dict(lo=lo,hi=hi,poly_error=table.poly_error)


def row_ids(folder, kind, plane):
    path = folder/f'{kind}_{plane:05d}.bin'
    return np.memmap(path,dtype=np.int64,mode='r') if path.stat().st_size else np.empty(0,np.int64)


def owner_job(folder, arg):
    begin,end,route,fit,leaves = arg
    layout,edges,geometry,options = (_CTX[key] for key in ('layout','edges','geometry','options'))
    table = MappedFits(layout, fit)
    if 'evaluator' not in _CTX:
        _CTX['evaluator'] = Evaluator(options)
    pairs = 0
    for plane in range(begin,end):
        output = folder/f'plane_{plane:05d}'
        output.mkdir()
        a,b = edges[plane:plane+2]
        face_windows(output,layout,a,b,leaves,plane)
        mass_plane(output,geometry,layout,table,_CTX['evaluator'],a,b,row_ids(route,'wall',plane))
        pairs += recipes(output,geometry,row_ids(route,'faces',plane),layout)
    record = dict(begin=begin,end=end,pairs=pairs)
    return record


def _publish_leaves(source, destination):
    for path in source.rglob('*.npy'):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.samefile(path):
            continue
        temp = target.with_name(target.name + '.' + uuid4().hex)
        os.link(path,temp)
        temp.replace(target)


def _publish_owner(out, index, source):
    """Expose the completed plane leaves by owner, with their global row IDs."""
    paths,result = source
    destination = out/'owners'/f'owner_{index:05d}'
    for path in paths:
        _publish_leaves(path,destination)
    record = dict(result, owner=index, directory=destination.relative_to(out).as_posix())
    atomic_json(out/'owners'/f'owner_{index:05d}.json',record)
    return record


def _export_parallel_construction(out, layout, options, geometry, fit, owners, edges, identity):
    if sum(record['pairs'] for _,record in owners)/layout.nodes > 16:
        raise ArithmeticError('moment-reduced face quadrature owner graph density guard exceeded')
    _publish_leaves(fit,out/'owners/fits')
    records = [_publish_owner(out,i,source) for i,source in enumerate(owners)]
    metadata = dict(format='drbx.parallel.owner_construction.v1', owners=records,
                    fits='owners/fits', plane_edges=edges.tolist(), index_binding=identity,
                    layout=indexed_layout(geometry),
                    metric_cache_sha256=sha(options.metric_cache),
                    field_table_sha256=sha(options.field_table),
                    rows='canonical state IDs and explicit canonical wall, fit and face IDs')
    metadata['moment_reduced_face_quadrature'] = face_quadrature_provenance(owner_plane_paths(out,metadata),geometry)
    atomic_json(out/'owners/manifest.json',metadata)


def _one(root, fn, *args):
    current = recover(root)
    if current is None:
        def action(folder, _):
            return fn(folder,*args)
        _,path,result = work(action,0,root,None)
        return path,result
    return current


def _build(options, geometry, layout, parts, root, identity):
    out = Path(options.out)
    edges = layout.plane_offsets
    stages = {}
    def collect(pool, name, fn, args):
        ordered = sorted(bounded_map(pool,fn,enumerate(args),root/name,options.workers),key=lambda x:x[0])
        stages[name] = [record for _,_,record in ordered]
        return [(path,result) for _,path,result in ordered]
    if options.workers == 1:
        initialize(options,layout,edges,None)
        pool_context = nullcontext(None)
    else:
        pool_context = ProcessPoolExecutor(options.workers,mp_context=mp.get_context('spawn'),
                                           initializer=initialize,
                                           initargs=(options,layout,edges,options.rss_limit_bytes))
    try:
        with pool_context as pool:
            routes = {}
            for kind in ('faces','wall'):
                routes[kind] = [path for path,_ in collect(pool,'route_'+kind,route_chunk,
                    ((kind,lo,hi) for lo,hi in geometry.ranges(kind)))]
            route,_ = _one(root/'routing',_route_export,routes,layout)
            keys = read(route/'keys.npy')
            batches = collect(pool,'fit_batches',_fit,((keys[i:i+64],route) for i in range(0,len(keys),64)))
            fit,_ = _one(root/'fit_store',fit_export,batches,layout)
            leaves = collect(pool,'face_leaves',_leaf,((lo,hi,fit)
                            for lo,hi in geometry.ranges('faces',options.shard_rows)))
            bounds = [(owner*layout.eta_planes+parts-1)//parts for owner in range(parts+1)]
            owner_ranges = list(zip(bounds,bounds[1:]))
            leaf_paths = [path for path,_ in leaves]
            if pool is None:
                owner_paths = [((path,), record) for path, record in collect(
                    pool,'plane_windows',owner_job,
                    ((a,b,route,fit,leaf_paths) for a,b in owner_ranges))]
            else:
                # Each plane owns its face windows, wall mass and complete jump
                # groups. Keep their reductions intact and publish in plane order.
                planes = collect(pool,'plane_windows',owner_job,
                    ((plane,plane+1,route,fit,leaf_paths) for plane in range(layout.eta_planes)))
                owner_paths = []
                for begin,end in owner_ranges:
                    owned = planes[begin:end]
                    paths = [path for path,_ in owned]
                    record = dict(begin=begin,end=end,pairs=sum(row['pairs'] for _,row in owned),
                                  peak_rss_bytes=max(row['peak_rss_bytes'] for _,row in owned))
                    owner_paths.append((paths, record))
        _export_parallel_construction(out,layout,options,geometry,fit,owner_paths,edges,identity)
    finally:
        if options.workers == 1:
            _CTX.pop('geometry').close()
            _CTX.clear()
    return stages


def build_indexed_parallel_operators(options):
    """Build or resume a binding to the placed immutable indexed input."""
    geometry_path = Path(options.geometry).resolve()
    out = Path(options.out).resolve()
    options = replace(options, geometry=geometry_path, out=out,
                      metric_cache=Path(options.metric_cache).resolve(),
                      field_table=Path(options.field_table).resolve())
    if options.face_chunk < 1 or not options.face_chunk <= options.shard_rows <= ROWS:
        raise ValueError('Indexed Q requires 1 <= face_chunk <= shard_rows <= 32000')
    if options.rss_limit_bytes is not None and options.rss_limit_bytes <= 0:
        raise ValueError('rss_limit_bytes must be positive')
    owners = min(options.workers,options.eta_planes) if options.owners is None else options.owners
    if not 1 <= owners <= options.eta_planes:
        raise ValueError('owners must be between 1 and eta_planes')
    if options.eta_planes // owners < 4:
        raise ValueError('Owner-only Q construction requires at least four planes per owner')
    source_root = geometry_path if geometry_path.is_dir() else geometry_path.parent
    if out.is_relative_to(source_root) or source_root.is_relative_to(out):
        raise ValueError('Indexed Q output must be separate from its geometry input')
    with closing(IndexedInput(geometry_path)) as geometry:
        identity = geometry.identity()
        h = range(geometry.shape('H')[0])
        layout = Layout.from_options(options,{'H':h})
        actual = (layout.N,layout.eta_planes,layout.P,layout.nodes)
        if actual != tuple(geometry.info[key] for key in ('N','E','P','NN')):
            raise ValueError('Q options differ from indexed geometry layout')
        producer = json.loads((geometry.handoff.parent/'geometry.json').read_text())
        if (producer['symmetry'],producer['field_periods']) != (options.symmetry,options.field_periods):
            raise ValueError('Q options differ from indexed geometry symmetry')
        package = Path(__file__).parent
        code = {p.name:sha(p) for p in sorted(package.glob('*.py'))}
        code['geometry_build/parallel_geometry_reader.py'] = sha(package.parent.parent/'geometry_build/parallel_geometry_reader.py')
        shared = ('_hashing.py', 'evaluators/magnetic_field.py', 'geometry_build/clipping_primitives.py',
                  'geometry_build/dual_cells.py', 'geometry_build/context.py', 'evaluators/metric_derivatives.py',
                  'geometry_build/trace_evaluator.py', 'geometry_build/trace_rays.py', 'geometry_build/compiled_fieldline_tracer.py',
                  'operators/parallel/geometry_io.py', 'operators/parallel/weights.py')
        code.update({name: sha(package.parent.parent/name) for name in shared})
        spec = asdict(options)
        for key in ('geometry','out','metric_cache','field_table'):
            spec[key] = str(spec[key])
        binding = dict(options=spec,geometry=identity,code=code,
                       versions=[sys.version,np.__version__,scipy.__version__],
                       inputs={key:sha(getattr(options,key)) for key in ('metric_cache','field_table')},
                       threads={k:os.environ.get(k) for k in ('NPROC','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                            'VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        binding = json.loads(json.dumps(binding))
        root = out.parent/(out.name+'_construction_work')
        root.mkdir(parents=True,exist_ok=True)
        with (root/'build.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            pin = root/'binding.json'
            if pin.exists():
                if json.loads(pin.read_text()) != binding:
                    raise ValueError('Indexed Q resume binding differs; use a new output path')
            else:
                if out.exists():
                    raise FileExistsError(out)
                atomic_json(pin,binding)
            final = root/'complete.json'
            if final.is_file() and json.loads(final.read_text())['products'] == fingerprint(out) and out.is_dir():
                return out
            # The on-disk export job can be replayed when a public file was lost.
            stats = _build(options,geometry,layout,owners,root,identity)
            atomic_json(final,dict(products=fingerprint(out),stages=stats))
    return out
