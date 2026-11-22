"""One-CPU-device-per-rank owner MMS launch and publication."""
import json
import math
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
from threading import Thread
import time

from .._hashing import sha256_file as digest
from ._receipts import file_receipts, peak_rss, worker_code_files, write_json


RANK_RSS_ENV = 'DRBX_FCI_NODAL_WORKER_RSS_BYTES'
SCHEDULER_INDEX_ENV = 'SLURM_PROCID'
SCHEDULER_OPTIONS = ('rank_command', 'rank_index_env', 'collectives_interface')


def scheduler_options(options, *, owners):
    """Validate scheduler-only options and return their effective values."""
    command,index_env,interface = (options.get(key) for key in SCHEDULER_OPTIONS)
    scheduler = options.get('launch_mode') == 'scheduler'
    supplied = any(value is not None for value in (command,index_env,interface))
    if (scheduler or supplied) and not owners:
        raise ValueError('Scheduler owner options require [fci_nodal] owners')
    if not scheduler:
        if supplied:
            raise ValueError("Scheduler owner options require launch_mode = 'scheduler'")
        return
    if (not isinstance(command,(list,tuple)) or not command
            or any(not isinstance(item,str) or not item.strip() or '\0' in item for item in command)):
        raise ValueError('rank_command must be a nonempty list of nonempty strings')
    if index_env is None:
        index_env = SCHEDULER_INDEX_ENV
    if (not isinstance(index_env,str) or not index_env.isascii()
            or not index_env.isidentifier()):
        raise ValueError('rank_index_env must be an environment variable name')
    if interface is not None and (not isinstance(interface,str) or not interface
                                  or any(c.isspace() or c in ',\0' for c in interface)):
        raise ValueError('collectives_interface must be a single network interface name')
    return command,index_env,interface


def _limits(options, env):
    """Combine the rank option with the ordinary case worker guard."""
    requested = options.get('rank_rss_limit_bytes')
    if requested is not None and (type(requested) is not int or requested <= 0):
        raise ValueError('rank_rss_limit_bytes must be a positive integer or None')
    inherited = env.get(RANK_RSS_ENV)
    if inherited:
        inherited = int(inherited)
        if inherited <= 0:
            raise ValueError(f'{RANK_RSS_ENV} must be positive')
    else:
        inherited = None
    limit = min((x for x in (requested,inherited) if x is not None),default=None)
    timeout = options.get('run_timeout_seconds')
    if timeout is not None and (type(timeout) not in (int,float)
                                or timeout <= 0 or not math.isfinite(timeout)):
        raise ValueError('run_timeout_seconds must be positive and finite or None')
    return limit,timeout


def _coordinator():
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as reservation:
        reservation.bind(('127.0.0.1',0))
        return f'127.0.0.1:{reservation.getsockname()[1]}'


def _rank_environment(settings):
    env = {key:value for key,value in {**os.environ,**settings}.items()
           if not key.lower().endswith('_proxy')}
    for key in ('DRBX_HOST_DEVICE_COUNT','JAX_NUM_CPU_DEVICES','JAX_CPU_COLLECTIVES_IMPLEMENTATION'):
        env.pop(key,None)
    env.update(XLA_FLAGS='--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false',
               GLOO_SOCKET_IFNAME='lo0' if sys.platform=='darwin' else 'lo',
               JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
    return env


def _stop(children):
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def _team(directory, payloads, env, timeout):
    address = _coordinator()
    workers = []
    launched = time.monotonic()
    try:
        for rank,payload in enumerate(payloads):
            data = directory/f'rank{rank:04d}.pkl'
            with data.open('xb') as stream:
                pickle.dump(payload,stream)
            with (directory/f'rank{rank:04d}.log').open('xb') as log:
                workers.append(subprocess.Popen(
                    [sys.executable,'-B','-u','-m',__name__,address,str(rank),str(len(payloads)),str(data)],
                    env=env,stdout=log,stderr=subprocess.STDOUT))
        while True:
            codes = [child.poll() for child in workers]
            if any(code not in (None,0) for code in codes):
                raise RuntimeError(f'Owner rank failed: {codes}; see {directory}')
            if all(code==0 for code in codes):
                return time.monotonic()-launched
            if timeout is not None and time.monotonic()-launched>=timeout:
                raise TimeoutError(f'Owner rank timeout; see {directory}')
            time.sleep(.05)
    finally:
        _stop(workers)


def _scheduler_team(directory, payloads, env, timeout, command, index_env):
    for rank,payload in enumerate(payloads):
        with (directory/f'rank{rank:04d}.pkl').open('xb') as stream:
            pickle.dump(payload,stream)
    host = socket.gethostname()
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as reservation:
        reservation.bind((host,0))
        address = f'{host}:{reservation.getsockname()[1]}'
    # Release the reservation so the rank worker can bind this port.
    parts = len(payloads)
    args = [*command, sys.executable, '-B', '-u', '-m', __name__, '--scheduler-rank',
            address, str(parts), str(directory), index_env]
    launched = time.monotonic()
    with (directory/'scheduler.log').open('xb') as log:
        worker = subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            if worker.wait(timeout=timeout):
                raise RuntimeError(f'Owner rank command failed: {worker.returncode}; see {directory}')
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f'Owner rank timeout; see {directory}') from error
        finally:
            _stop([worker])
    return time.monotonic()-launched


def _scheduler_rank_index(env, key, parts):
    value = env.get(key,'')
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f'{key} must contain a decimal owner index')
    rank = int(value)
    if not 0 <= rank < parts:
        raise ValueError(f'{key} owner index {rank} is outside [0, {parts})')
    return rank


def _check_coordinator_host(address, rank, hostname):
    coordinator = address.rsplit(':',1)[0]
    if rank == 0 and hostname != coordinator:
        raise ValueError(f'Owner rank 0 must run on coordinator host {coordinator}; got {hostname}')


def _scheduler_rank(address, parts, directory, index_env):
    parts = int(parts)
    rank = _scheduler_rank_index(os.environ,index_env,parts)
    host = socket.gethostname()
    _check_coordinator_host(address,rank,host)
    directory = Path(directory)
    name = f'rank{rank:04d}'
    try:
        log = (directory/f'{name}.log').open('xb')
    except FileExistsError as error:
        raise ValueError(f'Duplicate scheduler task for owner {rank}') from error
    with log:
        if (directory.parent/'ranks'/f'{name}.json').exists():
            raise ValueError(f'Duplicate rank reply for owner {rank}')
        os.dup2(log.fileno(),1)
        os.dup2(log.fileno(),2)
        print(json.dumps(dict(rank=rank, hostname=host, pid=os.getpid())),flush=True)
        rank_worker(address,rank,parts,directory/f'{name}.pkl')


def _verify_replies(out, parts, edges, width, capture, rank_rss_limit):
    import numpy as np

    outputs = out/'owners'
    ranks = out/'ranks'
    expected_dirs = {f'owner{i:04d}' for i in range(parts)}
    if {p.name for p in outputs.iterdir()}!=expected_dirs:
        raise ValueError('Owner output union does not cover the mesh')
    summaries, sources, peaks = [], [], []
    names = {'nodal_state.npy','node_ids.npy'} | ({'static_rhs.npz'} if capture else set())
    for owner, (a,b) in enumerate(zip(edges[:-1],edges[1:],strict=True)):
        reply = json.loads((ranks/f'rank{owner:04d}.json').read_text())
        folder = outputs/f'owner{owner:04d}'
        peak = reply['peak_rss_bytes']
        if reply['rank']!=owner or reply['edges']!=[a,b] or type(peak) is not int or peak<=0:
            raise ValueError(f'Invalid rank reply for {owner}')
        if rank_rss_limit is not None and peak>=rank_rss_limit:
            raise MemoryError(f'Owner {owner} reached its RSS limit: {peak} bytes')
        if {p.name for p in folder.iterdir()}!=names or reply['files']!=file_receipts([folder]):
            raise ValueError(f'Incomplete owner files or changed checksum for {owner}')
        ids = np.load(folder/'node_ids.npy',mmap_mode='r',allow_pickle=False)
        if ids.dtype!=np.int32 or not np.array_equal(ids,np.arange(a*width,b*width)):
            raise ValueError(f'Noncanonical output rows for {owner}')
        shape = (len(ids),6)
        state = np.load(folder/'nodal_state.npy',mmap_mode='r',allow_pickle=False)
        if state.shape!=shape or state.dtype!=np.float64:
            raise ValueError(f'Invalid owner state shape or dtype for {owner}')
        if capture:
            with np.load(folder/'static_rhs.npz',allow_pickle=False) as z:
                if set(z.files)!={'rhs','rhs_manufactured'} or any(
                        z[key].shape!=shape or z[key].dtype!=np.float64 for key in z.files):
                    raise ValueError(f'Invalid static owner RHS for {owner}')
        summaries.append(reply['summary'])
        sources.append(reply['sources'])
        peaks.append(peak)
    common = json.dumps(summaries[0],sort_keys=True,allow_nan=False)
    if any(json.dumps(value,sort_keys=True,allow_nan=False)!=common for value in summaries[1:]):
        raise ValueError('Owner rank scalar summaries differ')
    if any(source!=sources[0] for source in sources[1:]):
        raise ValueError('Owner rank source fingerprints differ')
    if summaries[0]['edges']!=list(edges):
        raise ValueError('Owner scalar summary edges differ')
    return summaries[0], max(peaks), sources[0]


def launch_rank_run(options, out, inputs, settings, *, checksums):
    """Run only the prepared owner MMS stage and return its verified products."""
    from ..operator_build.perp.sbp_setup import _layout
    from ..drb.owner_route_store import publish_routes
    from ._owner_inputs import OwnerParallelInputs, validate_inputs

    scheduler = scheduler_options(options,owners=True)
    env = _rank_environment(settings)
    if scheduler is not None:
        command,index_env,interface = scheduler
        if interface is None:
            env.pop('GLOO_SOCKET_IFNAME',None)
        else:
            env['GLOO_SOCKET_IFNAME'] = interface
    rank_rss_limit, timeout = _limits(options,env)
    if rank_rss_limit is not None:
        env[RANK_RSS_ENV] = str(rank_rss_limit)

    parts = options['runtime'].owner_count
    p,meta,construction,indexed = validate_inputs(inputs,parts=parts)
    wall = inputs['wall_data']
    q = OwnerParallelInputs(inputs['owner_parallel_construction'],meta,construction,p)
    routes = wall/'routes'
    supplemental = not routes.is_dir()
    if supplemental:
        # Historical prepared products are read-only. Add their derived routes
        # to this attempt before spawning any rank.
        layout = _layout(dict(p.manifest,n=p.manifest['N']))
        boundary = layout.walls[0]
        publish_routes(out/'routes',wall,inputs['owner_parallel'],q,
                       wall_per_plane=layout.blocks[boundary.block_idx].sides[boundary.side].N)
        routes = out/'routes'
    catalog_paths = [p.root/'manifest.json',p.root/'identity.json',
        inputs['owner_parallel']/'metadata.json',inputs['owner_parallel_construction']/'manifest.json',
        wall/'manifest.json',wall/'constants.npz',inputs['prepared']/'manifest.json',
        routes/'manifest.json']
    catalogs = {}
    for file in catalog_paths:
        path = str(file.resolve())
        expected = checksums.get(path)
        # The legacy P catalog has no internal identity, and the supplemental
        # route catalog was produced in this attempt.
        if file==p.root/'identity.json' and 'source_identity' not in p.manifest:
            continue
        if file.is_relative_to(out):
            expected = digest(file)
        if expected is None:
            raise ValueError(f'Missing verified catalog: {file}')
        catalogs[path] = expected

    directory = out/'launch'
    directory.mkdir()
    (out/'ranks').mkdir()
    payloads = []
    for owner in range(parts):
        directory_q = q.root.parent/construction['owners'][owner]['directory']
        mass = {}
        for plane in q.planes(owner):
            for name in ('wall_credited_parallel_mass.npy','MB_V.npy'):
                path = str((directory_q/f'plane_{plane:05d}'/name).resolve())
                if path not in checksums:
                    raise ValueError(f'Missing verified mass leaf: {path}')
                mass[path] = checksums[path]
        context = dict(owner=owner,owner_count=parts,edges=p.edges[owner:owner+2], indexed=indexed,
                       catalogs=catalogs,mass=mass,routes=routes)
        payloads.append((options,out,inputs,context))
    if scheduler is None:
        seconds = _team(directory,payloads,env,timeout)
    else:
        seconds = _scheduler_team(directory,payloads,env,timeout,command,index_env)
    summary,rss,sources = _verify_replies(
        out,parts,p.edges,q.width,options.get('capture_static_rhs'),rank_rss_limit)
    write_json(out/'result.json',summary)
    write_json(out/'launch.json',dict(seconds=seconds,peak_rss_bytes=rss,owner_count=parts,
               rank_rss_limit_bytes=rank_rss_limit,run_timeout_seconds=timeout))
    products = dict(owner_state=out/'owners',summary=out/'result.json',
                    owner_rank_receipts=out/'ranks',launch_mode=out/'launch.json')
    if supplemental:
        products['owner_route_supplement'] = routes
    sources['case/_rank_launch.py'] = digest(Path(__file__))
    return products,0,rss,sources


def _guard():
    if not (value := os.environ.get(RANK_RSS_ENV)):
        return
    limit = int(value)

    def watch():
        while True:
            if peak_rss()>=limit:
                os.write(2,b'Owner rank exceeded RSS limit\n')
                os._exit(97)
            time.sleep(.05)
    Thread(target=watch,daemon=True).start()


def rank_worker(address, owner, parts, payload):
    _guard()
    import jax

    owner,parts = int(owner),int(parts)
    jax.distributed.initialize(coordinator_address=address,coordinator_bind_address=address,
        num_processes=parts,process_id=owner,local_device_ids=[0],
        initialization_timeout=45,heartbeat_timeout_seconds=30,shutdown_timeout_seconds=5)
    with Path(payload).open('rb') as stream:
        options,out,inputs,rank = pickle.load(stream)
    if (owner,parts)!=(rank['owner'],rank['owner_count']):
        raise ValueError('Launch arguments and rank payload disagree')
    from ._mms_stages import run

    summary = run(options,out,inputs,rank=rank)
    folder = out/'owners'/f'owner{owner:04d}'
    receipt = dict(rank=owner,pid=os.getpid(),edges=rank['edges'],summary=summary,
                   files=file_receipts([folder]),sources=worker_code_files('mms_run'),
                   peak_rss_bytes=peak_rss())
    jax.distributed.shutdown()
    with (out/'ranks'/f'rank{owner:04d}.json').open('x') as stream:
        json.dump(receipt,stream,sort_keys=True,allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--scheduler-rank']:
        _scheduler_rank(*sys.argv[2:])
    else:
        rank_worker(*sys.argv[1:])
