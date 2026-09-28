"""Host-only, checksummed GPU trace interchange for the CPU reconstruction.

No JAX imports or backend environment changes are allowed here. A completed
trace stage is immutable and tied to one numerical design and raw-cell plan.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import json
import os
from pathlib import Path

import numpy as np

from .reuse import sha, write
from .span_contract import LEG_FACTORS, make_seeds

OUTPUT_NAMES = ('endpoint', 'valid', 'crossed', 'reentered', 'max_u',
                'min_j', 'first_cross_step', 'max_point', 'first_cross_point')
VECTOR_OUTPUTS = {0, 7, 8}
BOOL_OUTPUTS = {1, 2, 3}


def load(path):
    return json.loads(Path(path).read_text())


@contextmanager
def exclusive(path):
    """Fail closed for another writer or an unreviewed stale lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(f'{os.getpid()}\n')
        yield
    finally:
        path.unlink()


def seed_batch(centers, faces, n, raw_ids):
    """Keep the established interior h4/h8 and wall h8/h4 seed orders."""
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    ijk = np.array(np.unravel_index(raw_ids, (n, n, n))).T
    result = []
    for cell in ijk:
        q = np.array([centers[a][cell[a]] for a in range(3)])
        w = np.array([faces[a][cell[a]+1]-faces[a][cell[a]] for a in range(3)])
        seeds, _ = make_seeds(q, w)
        result.append(seeds if cell[0] >= n-7 else seeds[np.r_[6:12, 0:6]])
    return np.asarray(result, dtype=np.float64).reshape(-1, 12, 3)


def pack_legs(seeds, n):
    """Flatten raw, leg factor, seed in exactly the CPU tracer's order."""
    seeds = np.asarray(seeds, dtype=np.float64)
    if seeds.ndim != 3 or seeds.shape[1:] != (12, 3):
        raise ValueError('expected (raw,12,3) seeds')
    return (np.tile(seeds, (1, 4, 1)).reshape(-1, 3),
            np.tile(np.repeat(LEG_FACTORS * (2*np.pi/n), 12), len(seeds)))


def validate_trace(outputs, n):
    """The same valid/no-reentry/two-cell-reach gates as the inline path."""
    if len(outputs) != 9:
        raise RuntimeError('nine trace diagnostics required')
    count = len(outputs[0])
    for i, value in enumerate(outputs):
        shape = (count, 3) if i in VECTOR_OUTPUTS else (count,)
        if value.shape != shape:
            raise RuntimeError(f'trace diagnostic shape differs: {OUTPUT_NAMES[i]}')
        if i in BOOL_OUTPUTS and value.dtype != np.dtype(bool):
            raise RuntimeError('trace flag dtype differs')
        if i in (0, 4, 5, 7, 8) and value.dtype != np.dtype(np.float64):
            raise RuntimeError('trace requires float64')
    if (not np.all(outputs[1]) or np.any(outputs[3])
            or np.max(np.maximum(outputs[4]-1, 0), initial=0)*n > 2):
        raise RuntimeError('trace validity/reentry/reach gate failed')
    for i in (0, 4, 5, 7):
        if not np.all(np.isfinite(outputs[i])):
            raise RuntimeError('nonfinite trace diagnostic')
    if not np.all(np.isfinite(outputs[8][outputs[2]])):
        raise RuntimeError('missing first-crossing coordinates')
    if np.any(outputs[6][outputs[2]] < 0) or np.any(outputs[6][~outputs[2]] != -1):
        raise RuntimeError('first-crossing flags/step index differ')


def raw_groups(raw_ids, batch_raw_cells):
    raw = np.asarray(raw_ids, dtype=np.int64)
    if batch_raw_cells < 1 or not len(raw) or np.any(np.diff(raw) <= 0):
        raise ValueError('positive batch size and sorted unique nonempty raw IDs required')
    keys = raw // batch_raw_cells
    splits = np.flatnonzero(np.diff(keys)) + 1
    return [(int(group[0] // batch_raw_cells), group.tolist())
            for group in np.split(raw, splits)]


def chunk_path(root, n, block):
    return Path(root) / f'N{n}' / f'trace_{block:06d}.npz'


def freeze_store(stage_root, n, raw_ids, batch_raw_cells):
    root = Path(stage_root) / 'traces'
    root.mkdir(parents=True, exist_ok=True)
    design = load(Path(stage_root) / 'design.json')
    if design.get('trace_mode') != 'gpu_cache':
        raise RuntimeError('freeze a new campaign with --trace-mode gpu_cache first')
    groups = raw_groups(raw_ids, batch_raw_cells)
    metadata = dict(schema='q-gpu-trace-design-v1',
                    numerical_design_sha256=sha(Path(stage_root)/'design.json'),
                    rk4_steps=design['method']['rk4_steps'], check_steps=512,
                    batch_raw_cells=batch_raw_cells, legs_per_raw=48,
                    backend='gpu', dtype='float64')
    path = root/'design.json'
    if path.exists() and load(path) != metadata:
        raise RuntimeError('trace design differs; use a new campaign folder')
    write(path, metadata)
    plan = dict(N=n, groups=groups, raw_count=len(raw_ids),
                check_raw_ids=sorted(set(design['rk4_512_raw_ids'][str(n)])
                                     & set(map(int, raw_ids))))
    path = root/f'plan_N{n}.json'
    if path.exists() and load(path) != json.loads(json.dumps(plan)):
        raise RuntimeError('trace coverage plan differs')
    write(path, plan)
    return root, groups


def completed(root, n, block, raw_ids):
    path = chunk_path(root, n, block)
    receipt = path.with_suffix('.json')
    if not path.exists() and not receipt.exists():
        return False
    if not path.exists() or not receipt.exists():
        raise RuntimeError(f'interrupted trace chunk: {path}')
    r = load(receipt)
    expected = dict(schema='q-gpu-trace-chunk-v1', N=n, raw_ids=list(raw_ids),
                    trace_design_sha256=sha(Path(root)/'design.json'),
                    plan_sha256=sha(Path(root)/f'plan_N{n}.json'))
    if any(r.get(key) != value for key, value in expected.items()) or sha(path) != r['sha256']:
        raise RuntimeError(f'incompatible or corrupt trace chunk: {path}')
    return True


def save_chunk(root, n, block, raw_ids, seeds, outputs, checks, execution):
    """Called under a writer lock after the device has synchronized."""
    validate_trace(outputs, n)
    count = len(raw_ids)
    if outputs[0].shape != (48*count, 3) or seeds.shape != (count, 12, 3):
        raise RuntimeError('raw/seed/trace coverage differs')
    expected = sorted(set(load(Path(root)/f'plan_N{n}.json')['check_raw_ids']) & set(raw_ids))
    if sorted(checks) != expected:
        raise RuntimeError('RK4-512 sample coverage differs')
    arrays = dict(raw_ids=np.asarray(raw_ids, dtype=np.int64), seeds=seeds)
    for name, value in zip(OUTPUT_NAMES, outputs):
        arrays[name] = value.reshape((count, 48) + value.shape[1:])
    for raw, check in checks.items():
        validate_trace(check, n)
        if check[0].shape != (48, 3):
            raise RuntimeError('RK4-512 trace coverage differs')
        for name, value in zip(OUTPUT_NAMES, check):
            arrays[f'check_{raw}_{name}'] = value
    path = chunk_path(root, n, block)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    write(path.with_suffix('.json'), dict(
        schema='q-gpu-trace-chunk-v1', N=n, raw_ids=list(raw_ids),
        check_raw_ids=expected, trace_design_sha256=sha(Path(root)/'design.json'),
        plan_sha256=sha(Path(root)/f'plan_N{n}.json'), sha256=sha(path),
        bytes=path.stat().st_size, execution=execution))


def finish_store(root, n):
    """Publish a complete inventory only after all chunks pass hashes."""
    root = Path(root)
    plan = load(root/f'plan_N{n}.json')
    chunks = []
    for block, raw in plan['groups']:
        if not completed(root, n, block, raw):
            raise RuntimeError(f'missing trace chunk {block}')
        path = chunk_path(root, n, block)
        r = load(path.with_suffix('.json'))
        chunks.append(dict(block=block, sha256=r['sha256'],
                           receipt_sha256=sha(path.with_suffix('.json'))))
    result = dict(schema='q-gpu-trace-complete-v1', N=n,
                  trace_design_sha256=sha(root/'design.json'),
                  plan_sha256=sha(root/f'plan_N{n}.json'), chunks=chunks,
                  raw_count=plan['raw_count'])
    write(root/f'complete_N{n}.json', result)
    return result


class TraceStore:
    """Bounded LRU reader; never silently fall back to CPU tracing."""

    def __init__(self, stage_root, n, max_chunks=4):
        self.root = Path(stage_root)/'traces'
        self.n = n
        self.design = load(self.root/'design.json')
        if (self.design['numerical_design_sha256'] != sha(Path(stage_root)/'design.json')
                or self.design['schema'] != 'q-gpu-trace-design-v1'):
            raise RuntimeError('trace/numerical design identity differs')
        self.plan = load(self.root/f'plan_N{n}.json')
        inventory = load(self.root/f'complete_N{n}.json')
        if (inventory['trace_design_sha256'] != sha(self.root/'design.json')
                or inventory['plan_sha256'] != sha(self.root/f'plan_N{n}.json')
                or inventory['N'] != n or inventory['raw_count'] != self.plan['raw_count']):
            raise RuntimeError('trace completion identity differs')
        self.groups = dict(self.plan['groups'])
        self.inventory = {r['block']: r for r in inventory['chunks']}
        if sorted(self.inventory) != sorted(self.groups):
            raise RuntimeError('trace completion coverage differs')
        self.max_chunks = max_chunks
        self.cache = OrderedDict()

    def require_raw_ids(self, raw_ids):
        available = {raw for group in self.groups.values() for raw in group}
        if not set(map(int, raw_ids)).issubset(available):
            raise RuntimeError('trace store does not cover the requested owners')

    def get(self, raw, seeds):
        block = int(raw)//self.design['batch_raw_cells']
        if block not in self.cache:
            if block not in self.groups or not completed(self.root, self.n, block, self.groups[block]):
                raise RuntimeError(f'missing trace block for raw {raw}')
            path = chunk_path(self.root, self.n, block)
            frozen = self.inventory[block]
            if (sha(path.with_suffix('.json')) != frozen['receipt_sha256']
                    or load(path.with_suffix('.json'))['sha256'] != frozen['sha256']):
                raise RuntimeError('trace chunk changed after completion')
            with np.load(path, allow_pickle=False) as z:
                self.cache[block] = {key: z[key].copy() for key in z.files}
            while len(self.cache) > self.max_chunks:
                self.cache.popitem(last=False)
        self.cache.move_to_end(block)
        data = self.cache[block]
        idx = int(np.searchsorted(data['raw_ids'], raw))
        if idx == len(data['raw_ids']) or data['raw_ids'][idx] != raw:
            raise RuntimeError(f'missing trace raw {raw}')
        if not np.array_equal(data['seeds'][idx], seeds):
            raise RuntimeError('GPU/CPU seed ordering or coordinates differ')
        result = tuple(data[name][idx] for name in OUTPUT_NAMES)
        validate_trace(result, self.n)
        check = None
        if raw in self.plan['check_raw_ids']:
            check = tuple(data[f'check_{raw}_{name}'] for name in OUTPUT_NAMES)
            validate_trace(check, self.n)
        return result, check
