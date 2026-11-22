"""Local lanes from sharded geometry scheduling and canonical collection static_schedule, placement and static_store."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from hashlib import sha256
import multiprocessing
import os
from pathlib import Path
import resource
import sys
from tempfile import mkdtemp
import time

from .storage import dump, fingerprint, load
from .._hashing import sha256_file as sha


SHARDS = 16384
BLOCK = 8
COARSE = {'plane', 'census', 'physical_count', 'physical_wall', 'polygon', 'masks', 'wall', 'observe'}


def slot_for(task, planes, total):
    kind = task['kind']
    if kind not in COARSE:
        key = kind+'\0'+task['output']
        shard = int.from_bytes(sha256(key.encode()).digest()[:8],'little') % SHARDS
        return shard % total
    index = task['plane'] if 'plane' in task else task['bounds'][0]//8192
    mode = 1 if kind in {'plane','census','wall'} else 2
    index += {'polygon':1,'masks':2}.get(kind,0)
    if mode==2 and total>planes:
        return planes+index%(total-planes)
    return index%total


def atomic_record(path, value):
    temp = path.with_suffix('.pending')
    dump(temp,value)
    os.replace(temp,path)


def completed(root, task):
    target = root/task['output']
    marker = target/'job_complete.json'
    if not marker.exists():
        if target.exists():
            raise RuntimeError(f'Uncommitted canonical task: {target}')
        return False
    record = load(marker)
    if record['identity']!=task['identity']:
        raise RuntimeError(f'Resume task identity changed: {target}')
    for name,expected in record['products'].items():
        if sha(target/name)!=expected:
            raise RuntimeError(f'Resume task product changed: {target/name}')
    return True


def commit(root, task, execute):
    target = root/task['output']
    target.parent.mkdir(parents=True,exist_ok=True)
    private = Path(mkdtemp(dir=root/'attempts'))
    result, products, rays = execute(task,private,target)
    marker = dict(identity=task['identity'],result=result,new_rays=rays,
                  products={name:sha(private/name) for name in products})
    dump(private/'job_complete.json',marker)
    os.rename(private,target)


def _initialize(root, lane, options, config):
    import jax

    from ._shard_jobs import Jobs

    global _worker, _stats, _started, _receipt
    _started = time.monotonic()
    _receipt = root/'workers'/f'lane-{lane}.json'
    _stats = dict(lane=lane,pid=os.getpid(),tasks=0,task_wall_seconds=0.,new_rays=0)
    atomic_record(_receipt,_stats)
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_platforms','cpu')
    _worker = Jobs(root,options,config)


def _execute_block(root, block):
    started = time.monotonic()
    try:
        for task in block:
            commit(root,task,_worker.execute)
            _stats['tasks'] += 1
    finally:
        _stats['task_wall_seconds'] += time.monotonic()-started
        _stats['wall_seconds'] = time.monotonic()-_started
        _stats['peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        _stats['new_rays'] = sum(rays.traced for rays in _worker.rays.values())
        atomic_record(_receipt,_stats)


class Scheduler:
    def __init__(self,root,options,config,identity):
        self.root = root
        self.total = options.workers
        self.planes = config['N']//config['replication_periods']
        self.identity = identity
        mp = multiprocessing.get_context('spawn')
        self.pools = [ProcessPoolExecutor(max_workers=1,mp_context=mp,initializer=_initialize,
                      initargs=(root,lane,options,config)) for lane in range(self.total)]

    def tasks(self, label, descriptors):
        plan = []
        for descriptor in descriptors:
            task = dict(descriptor,output=str(descriptor['output']))
            task['identity'] = fingerprint([self.identity,task])
            plan.append(task)
        if len({task['output'] for task in plan})!=len(plan):
            raise RuntimeError(f'Duplicate canonical task in {label}')
        receipt = self.root/'plans'/f'{label}.json'
        if receipt.exists():
            if load(receipt)!=plan:
                raise RuntimeError(f'Resume request plan changed: {label}')
        else:
            atomic_record(receipt,plan)
        return plan

    def run(self,label,descriptors):
        plan = self.tasks(label,descriptors)
        lanes = [[] for _ in self.pools]
        for task in sorted(plan,key=lambda desc:desc['output']):
            if not completed(self.root,task):
                lanes[slot_for(task,self.planes,self.total)].append(task)

        def blocks(jobs):
            batch = []
            for task in jobs:
                if task['kind'] in COARSE and batch:
                    yield batch
                    batch = []
                batch.append(task)
                if task['kind'] in COARSE or len(batch)==BLOCK:
                    yield batch
                    batch = []
            if batch:
                yield batch

        ready = [iter(blocks(lane)) for lane in lanes]
        pending = {}

        def submit(lane):
            block = next(ready[lane],None)
            if block is not None:
                pending[self.pools[lane].submit(_execute_block,self.root,block)] = lane

        for lane in range(self.total):
            submit(lane)
        while pending:
            done,_ = wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                lane = pending.pop(future)
                future.result()
                submit(lane)
        count = len(plan)-sum(len(lane) for lane in lanes)
        atomic_record(self.root/'plans'/f'{label}.done',dict(tasks=len(plan),reused=count))
        return plan

    def close(self):
        for pool in self.pools:
            pool.shutdown(wait=True,cancel_futures=True)
