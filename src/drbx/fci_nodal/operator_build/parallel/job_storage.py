"""Local durable jobs and bounded compatibility exports from bounded indexed parallel assembly."""
from concurrent.futures import FIRST_COMPLETED, wait
import json
import os
import resource
import sys
import threading
import time
from uuid import uuid4

import numpy as np

from ..._hashing import sha256_file as sha


ROWS = 8192


def mapped(path, shape, dtype):
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=tuple(map(int,shape)))


def read(path):
    return np.load(path, mmap_mode='r', allow_pickle=False)


def atomic_json(path, value):
    temp = path.with_name(path.name + '.' + uuid4().hex)
    with temp.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def peak_rss():
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return size if sys.platform == 'darwin' else size * 1024


def guard_rss(limit):
    if limit is None:
        return

    def check():
        while True:
            current = peak_rss()
            if current >= limit:
                os.write(2, f'Q build PID {os.getpid()} exceeded its peak RSS cap: {current} bytes\n'.encode())
                os._exit(86)
            time.sleep(0.05)

    threading.Thread(target=check, daemon=True).start()


def fingerprint(directory):
    return {p.relative_to(directory).as_posix(): sha(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def recover(root):
    receipt = root / 'complete.json'
    if receipt.is_file():
        saved = json.loads(receipt.read_text())
        attempt = root / saved['attempt']
        if saved['files'] == fingerprint(attempt) and attempt.is_dir():
            return attempt, saved['result']
    return None


def work(fn, ordinal, root, arg):
    root.mkdir(parents=True, exist_ok=True)
    attempt = root / ('attempt-' + uuid4().hex)
    attempt.mkdir()
    result = fn(attempt, arg)
    result = dict(result or {}, pid=os.getpid(), peak_rss_bytes=peak_rss())
    atomic_json(root / 'complete.json', dict(attempt=attempt.name, files=fingerprint(attempt), result=result))
    return ordinal, attempt, result


def bounded_map(pool, fn, jobs, root, workers):
    """bounded indexed parallel assembly completion dispatch with at most two descriptor tasks per worker."""
    pending = set()
    for ordinal, arg in jobs:
        job = root / f'{ordinal:08d}'
        current = recover(job)
        if current is not None:
            yield ordinal, *current
            continue
        if pool is None:
            yield work(fn, ordinal, job, arg)
            continue
        pending.add(pool.submit(work, fn, ordinal, job, arg))
        if len(pending) >= workers * 2:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()


def csr_leaf(folder, matrix):
    folder.mkdir()
    for key in ('data', 'indices', 'indptr'):
        np.save(folder / (key + '.npy'), getattr(matrix, key), allow_pickle=False)


