"""sharded geometry scheduling and canonical collection cubic rounds, owner prefixes and canonical collection on one host."""
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
from shutil import copytree
from tempfile import mkdtemp

import numpy as np

from ._shard_schedule import Scheduler, atomic_record
from .build import assembly_plan
from .indexed_output import Writer, publish_fits
from .storage import arrays, dump, fingerprint, load
from .._hashing import sha256_file as sha


def source_rounds(scheduler,root,planes):
    bases = [Path('source')/f'p{k:03d}' for k in range(planes)]

    def plans(order):
        for k,base in enumerate(bases):
            for sign in (1,-1):
                yield dict(kind='source_plan',output=base/f'q{order}_{sign}'/'plan',
                           plane=k,base=str(base),order=order,sign=sign)

    def integrations(k,base,order):
        for sign in (1,-1):
            p = base/f'q{order}_{sign}'/'plan'
            n = load(root/p/'complete.json')['arrays']['owner']['shape'][0]
            for lo in range(0,n,64):
                yield dict(kind='integrate',output=p.parent/f'leaf_{lo:09d}',plane=k,
                           plan=str(p),order=order,sign=sign,bounds=[lo,min(lo+64,n)])

    def base_jobs():
        for k,base in enumerate(bases):
            yield from integrations(k,base,3)
            yield dict(kind='observe',output=base/'observation',plane=k)

    def clips():
        for k,base in enumerate(bases):
            obs = base/'observation'
            n = load(root/obs/'complete.json')['arrays']['tris']['shape'][0]
            for lo in range(0,n,8):
                yield dict(kind='clip',output=base/'clip'/f'leaf_{lo:09d}',plane=k,
                           observation=str(obs),bounds=[lo,min(lo+8,n)])

    def fine_jobs():
        for k,base in enumerate(bases):
            yield from integrations(k,base,8)
        yield from clips()

    def metrics():
        for t in clips():
            clip = t['output']
            yield dict(kind='metric',output=clip.parent.parent/'metric'/clip.name,
                       plane=t['plane'],clip=str(clip),bounds=t['bounds'])

    scheduler.run('source_q3_plans',plans(3))
    scheduler.run('source_base',base_jobs())
    scheduler.run('source_q8_plans',plans(8))
    scheduler.run('source_fine',fine_jobs())
    scheduler.run('source_metrics',metrics())
    return [root/base for base in bases]


def owner_rounds(scheduler,root,options,m,ns,production,sources):
    wall = production.wall['wall']
    chunks = [[lo,min(lo+8192,len(wall))] for lo in range(0,len(wall),8192)]
    census = [dict(kind='census',output=f'census/p{k:03d}',plane=k,source=str(source))
              for k,source in enumerate(sources)]
    counts = [dict(kind='physical_count',output=f'census/wall_counts/chunk_{lo:012d}',bounds=[lo,hi])
              for lo,hi in chunks]
    scheduler.run('census',census+counts)
    meta = [load(root/t['output']/'complete.json')['meta'] for t in census]
    physical = [(arrays(root/t['output'],['count']),{}) for t in counts]
    polygons = production.faces
    plan = assembly_plan(m,ns,meta,physical,len(polygons),options.storage_bins)
    plan['original_face_prefix'] = np.cumsum([0, *(plan['p'] * row['source_records'] for row in meta)])
    atomic_record(root/'assembly_plan.json',plan)

    # Canonical commit order from sharded geometry scheduling and canonical collection owner_build.part_paths.
    parts = []
    parts += [dict(kind='plane',output=f'parts/plane{k:03d}',plane=k,source=str(source))
              for k,source in enumerate(sources)]
    parts += [dict(kind='polygon',output=f'parts/polygon_{lo:012d}',bounds=[lo,min(lo+8192,len(polygons))])
              for lo in range(0,len(polygons),8192)]
    parts += [dict(kind='physical_wall',output=f'parts/wall_{lo:012d}',bounds=[lo,hi],prefix=offset)
              for (lo,hi),offset in zip(chunks,plan['physical_prefixes'])]
    scheduler.run('owners',parts)
    return plan,parts


def content_pin(path):
    path = Path(path).resolve()
    if path.is_file():
        return (str(path),sha(path))
    if not path.is_dir():
        raise FileNotFoundError(path)
    return (str(path),fingerprint([(str(p.relative_to(path)),sha(p)) for p in sorted(path.rglob('*'))
                                 if p.is_file()]))


def build_identity(options,m):
    paths = list(m['files'].values())
    for path in (options.geometry_reference_key_table,m.get('polygon_receipt')):
        if path is not None:
            paths.append(path)
    code = Path(__file__).resolve().parent
    paths.extend(sorted(code.glob('*.py')))
    for name in ('_hashing.py', 'evaluators/magnetic_field.py', 'evaluators/metric_derivatives.py',
                 'operators/parallel/geometry_io.py', 'operators/parallel/weights.py'):
        paths.append(code.parent / name)
    fields = asdict(options)
    fields.pop('workers')  # Identical task identity after changing the number of local lanes.
    return fingerprint([fields,[content_pin(path) for path in dict.fromkeys(paths)]])


def collect(root,output,scheduler,m,plan,parts,identity):
    staging = Path(mkdtemp(dir=root/'attempts'))
    dump(staging/'assembly_plan.json',plan)
    writer = Writer(staging,m,plan)
    try:
        for part in parts:
            path = root/part['output']
            writer.collect_part(path,path.name,output)
        geometry = writer.finish(published_root=output)
        keys = writer.keys()
        tasks = [dict(kind='fit',output=f'fit_tasks/leaf_{lo:09d}',keys=keys[lo:lo+64])
                 for lo in range(0,len(keys),64)]
        scheduler.run('fits',tasks)
        batches = iter(tasks)
        publish_fits(staging,m,geometry,keys,lambda *_: arrays(root/next(batches)['output']))
        writer.publish_reader()
    finally:
        writer.close()
    for part in ('source','census'):
        if (root/part).is_dir():
            copytree(root/part,staging/part,copy_function=os.link)
    published = ('geometry.json','index.sqlite','parallel_geometry_reader.json','parallel_geometry_spans.sqlite','fit_manifest.json','plane_fit_keys.sqlite')
    source = root/'source'
    new_rays = sum(load(marker)['new_rays'] for marker in source.rglob('job_complete.json')) if source.is_dir() else 0
    dump(staging/'sharded_geometry_complete.json',dict(identity=identity,new_rays=new_rays,
                                             products={name:sha(staging/name) for name in published}))
    os.rename(staging,output)


def build_sharded(options,m,ns,production):
    output = Path(options.output).resolve()
    root = output.with_name(output.name+'.shard-work')
    root.mkdir(parents=True,exist_ok=True)
    for name in ('attempts','plans','workers'):
        (root/name).mkdir(exist_ok=True)
    with (root/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity = build_identity(options,m)
        marker = root/'build.json'
        if marker.exists() and load(marker)['identity']!=identity:
            raise RuntimeError('Resume build inputs or code changed')
        atomic_record(marker,dict(identity=identity))
        if output.exists():
            record = load(output/'sharded_geometry_complete.json')
            if record['identity']!=identity or any(sha(output/name)!=value for name,value in record['products'].items()):
                raise RuntimeError('Completed geometry publication changed')
            return output
        scheduler = Scheduler(root,options,m,identity)
        try:
            planes = m['N']//m['replication_periods']
            sources = source_rounds(scheduler,root,planes)
            plan,parts = owner_rounds(scheduler,root,options,m,ns,production,sources)
            collect(root,output,scheduler,m,plan,parts,identity)
        finally:
            scheduler.close()
    return output
