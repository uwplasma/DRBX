"""Four-GPU, large-batch tracing stage; reconstruction remains on CPUs.

Run as its own process, before ``campaign run``. This module intentionally
does not import numerical_runner/exact_screen, which force the CPU backend.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import threading
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np

from . import trace_store as store

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def verify_stage(campaign, stage, n, input_root):
    """Verify frozen sources/inputs without initializing a CPU JAX runtime."""
    root = Path(campaign)
    dispatch = store.load(root/'dispatch.json')
    stage_root = root/stage
    design = store.load(stage_root/'design.json')
    if dispatch['source_design_sha256'][stage] != store.sha(stage_root/'design.json'):
        raise RuntimeError('stage design differs from dispatch')
    if design.get('trace_mode') != 'gpu_cache' or design['method']['rk4_steps'] not in (64, 256):
        raise RuntimeError('a frozen gpu_cache campaign with 64 or 256 RK4 steps is required')
    for relative, digest in design['sources'].items():
        if store.sha(REPO/relative) != digest:
            raise RuntimeError(f'frozen source differs: {relative}')
    for entry in design['input_manifest']['files']:
        path = Path(input_root)/entry['path']
        if (not path.is_file() or path.stat().st_size != entry['bytes']
                or store.sha(path) != entry['sha256']):
            raise RuntimeError(f'frozen input differs: {path}')
    groups = dispatch['stages'][stage][str(n)]
    owners = [int(owner) for group in groups for owner in group]
    if not owners or owners != sorted(set(owners)):
        raise RuntimeError('invalid owner coverage')
    return stage_root, design, owners


def load_grid(input_root, n, config):
    directory = Path(input_root)/config['geometry']/f'{n}x{n}x{n}'
    with np.load(directory/'base_geometry.npz') as z:
        centers = [z[f'grid.{a}.centers'].copy() for a in 'xyz']
        faces = [z[f'grid.{a}.faces'].copy() for a in 'xyz']
    with np.load(directory/'rlp_topology.npz') as z:
        owner_ids = np.flatnonzero(z['is_active_owner'].ravel())
        lookup = np.full(n**3, -1, dtype=np.int64)
        lookup[owner_ids] = np.arange(len(owner_ids))
        labels = lookup[z['aggregate_id'].ravel()]
    if np.any(labels < 0) or len(labels) != n**3:
        raise RuntimeError('canonical raw-owner topology differs')
    return centers, faces, labels


def load_host_field(input_root, config, cpu_device):
    """Load geometry/B coefficients once; no MMS states or fits are built."""
    import jax
    from drbx.geometry.MetricEvaluator import MetricEvaluator
    from drbx.geometry.Bfield_evaluator import bfield_evaluator_from_makegrid
    from drbx.geometry.hsx_jax_field import JaxHsxMagneticField

    with np.load(Path(input_root)/config['metric_cache']) as z:
        metric = MetricEvaluator.from_cache_payload(z, prefix='metric_evaluator_')
    bfield = bfield_evaluator_from_makegrid(
        Path(input_root)/config['makegrid'], currents=np.array(config['currents']), method='cubic')
    with jax.default_device(cpu_device):
        return JaxHsxMagneticField.from_evaluators(metric, bfield)


def select_devices(available, indices):
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('distinct local GPU indices required')
    if any(i < 0 or i >= len(available) for i in indices):
        raise RuntimeError(f'requested GPU indices {indices}, but only {len(available)} GPUs are visible')
    devices = [available[i] for i in indices]
    if any(device.platform != 'gpu' for device in devices):
        raise RuntimeError('GPU tracing never falls back to CPU')
    return devices


def trace_queue(root, n, centers, faces, groups, resident_fields, trace_fn):
    """One host thread per GPU, disjoint dynamic chunks, bounded live arrays.

    ``resident_fields`` is a sequence of (device, immutable field) pairs. The
    injected trace_fn is also used by CPU correctness tests; production passes
    trace_fixed_batch. No process pool competes for GPU memory.
    """
    metadata = store.load(Path(root)/'design.json')
    sample = set(store.load(Path(root)/f'plan_N{n}.json')['check_raw_ids'])
    pending = queue.Queue()
    skipped = 0
    for block, raw in groups:
        if store.completed(root, n, block, raw):
            skipped += 1
        else:
            pending.put((block, raw))
    stop = threading.Event()

    def worker(device, field):
        count = 0
        try:
            while not stop.is_set():
                try:
                    block, raw = pending.get_nowait()
                except queue.Empty:
                    break
                path = store.chunk_path(root, n, block)
                with store.exclusive(path.with_suffix('.lock')):
                    if store.completed(root, n, block, raw):
                        continue
                    seeds = store.seed_batch(centers, faces, n, raw)
                    legs, deltas = store.pack_legs(seeds, n)
                    start = time.monotonic()
                    outputs = trace_fn(field, legs, deltas, metadata['rk4_steps'],
                                       metadata['batch_raw_cells']*48, device)
                    elapsed = time.monotonic()-start
                    checks = {}
                    start = time.monotonic()
                    for idx, raw_id in enumerate(raw):
                        if raw_id in sample:
                            q, d = store.pack_legs(seeds[idx:idx+1], n)
                            checks[raw_id] = trace_fn(field, q, d, 512, 64, device)
                    check_elapsed = time.monotonic()-start
                    execution = dict(device=str(device), backend=device.platform, device_kind=getattr(device, 'device_kind', None),
                                     trace_wall_s=elapsed, check_wall_s=check_elapsed,
                                     timing='synchronized; includes transfers and any first-call compilation',
                                     primary_real_legs=len(legs),
                                     primary_capacity=metadata['batch_raw_cells']*48,
                                     rk4_steps=metadata['rk4_steps'])
                    store.save_chunk(root, n, block, raw, seeds, outputs, checks, execution)
                    count += 1
        except BaseException:
            stop.set()
            raise
        return count

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(resident_fields)) as pool:
        futures = [pool.submit(worker, device, field) for device, field in resident_fields]
        computed = sum(future.result() for future in futures)
    result = store.finish_store(root, n)
    return dict(N=n, computed=computed, skipped=skipped, raw_count=result['raw_count'],
                devices=len(resident_fields))


def run(campaign, stage, n, input_root, device_indices=(0, 1, 2, 3), batch_raw_cells=128):
    # This entry point owns its process. Environment selection must precede
    # JAX imports; importing the CPU campaign runner here would defeat it.
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        raise RuntimeError('CUDA_VISIBLE_DEVICES is empty; retain the GPU visibility supplied by Slurm')
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    os.environ['JAX_ENABLE_X64'] = 'true'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    cache = Path(campaign)/'gpu_jax_cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(cache.resolve())
    os.environ['DRBX_CACHE_DIR'] = str(cache.resolve())
    stage_root, design, owners = verify_stage(campaign, stage, n, input_root)
    import jax
    from .trace import trace_fixed_batch
    jax.config.update('jax_enable_x64', True)
    devices = select_devices(jax.local_devices(backend='gpu'), list(device_indices))
    config = store.load(HERE.parent/'q_fci_projected_campaign/configuration.json')
    centers, faces, labels = load_grid(input_root, n, config)
    if owners[-1] >= int(labels.max())+1:
        raise RuntimeError('requested owner outside topology')
    raw = np.flatnonzero(np.isin(labels, owners))
    # A stage-wide lock also protects creation of the immutable plan/receipt.
    # Other resolutions may be prepared sequentially; no duplicated full-node launcher.
    with store.exclusive(stage_root/'gpu_trace.lock'):
        root, groups = store.freeze_store(stage_root, n, raw, batch_raw_cells)
        if all(store.completed(root, n, block, ids) for block, ids in groups):
            result = store.finish_store(root, n)
            return dict(N=n, computed=0, skipped=len(groups), raw_count=result['raw_count'])
        host_field = load_host_field(input_root, config, jax.local_devices(backend='cpu')[0])
        resident = []
        for device in devices:
            field = jax.device_put(host_field, device)
            jax.block_until_ready(field)
            resident.append((device, field))
        result = trace_queue(root, n, centers, faces, groups, resident, trace_fixed_batch)
        store.write(root/f'run_N{n}.json', dict(
            **result, jax_version=jax.__version__, backend='gpu',
            devices_detail=[dict(id=d.id, kind=d.device_kind) for d in devices],
            rk4_steps=design['method']['rk4_steps'], batch_raw_cells=batch_raw_cells))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--stage', choices=('preflight', 'pilot', 'global'), required=True)
    parser.add_argument('--N', type=int, choices=(32, 48, 64), required=True)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--batch-raw-cells', type=int, default=128,
                        help='128 raw cells = 6144 simultaneous trajectories per GPU')
    args = parser.parse_args()
    if args.batch_raw_cells < 1:
        parser.error('--batch-raw-cells must be positive')
    print(json.dumps(run(args.campaign, args.stage, args.N, args.input_root,
                         args.devices, args.batch_raw_cells)))


if __name__ == '__main__':
    main()
