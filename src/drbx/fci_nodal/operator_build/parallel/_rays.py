"""Local exact-key lookup and ordered dispatch of dense field-line tracing."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from hashlib import sha256
from itertools import islice
import json
import math
import multiprocessing
import re

import numpy as np

from ._io import dump, load, save
from ._geometry import TAU
from ...geometry_build.trace_rays import sample

SIZES = (8, 16, 128)
STEPS = 64


def key(q, d):
    return sha256(np.asarray(q, dtype='<f8').tobytes()
                  + np.asarray(d, dtype='<f8').tobytes()).hexdigest()


def batch_size(n, workers):
    return next((s for s in SIZES if s >= math.ceil(n / workers)), 128)


def _initialize_trace_worker(tracer, evaluator_factory):
    global _trace, _evaluator
    _trace, _evaluator = tracer, evaluator_factory()


def trace_pool(workers, tracer, evaluator_factory):
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=_initialize_trace_worker, initargs=(tracer, evaluator_factory))


def _trace_batch(seeds, delta, size, extrema):
    _evaluator.trace_batch_size = size
    _evaluator.beta_min, _evaluator.beta_max, _evaluator.max_u = extrema
    result = _trace(_evaluator, seeds, delta, steps=STEPS)
    return result, (_evaluator.beta_min, _evaluator.beta_max, _evaluator.max_u)


def merge_extrema(ev, extrema):
    beta_min, beta_max, max_u = extrema
    ev.beta_min = min(ev.beta_min, beta_min)
    ev.beta_max = max(ev.beta_max, beta_max)
    ev.max_u = max(ev.max_u, max_u)
    if ev.beta_min * ev.beta_max <= 0:
        raise ValueError('B^eta zero or sign change')


class RayCache:
    """Use a caller-owned work directory and optional matching-archive lookup.

    The on-disk files belong to the map/table/kernel/options supplied for this
    route. Use a fresh root for different inputs. The archive callback, when
    supplied, returns a full dense record for the normalized seed and delta.
    """

    def __init__(self, root, options, tracer, archive=None, prior_ledgers=()):
        self.root = root / 'rays'
        self.options = options
        self.tracer = tracer
        self.archive = archive
        self._pool = None
        self.entries = {}
        inherited = max((ledger['new_rays'] for ledger in prior_ledgers), default=0)
        self.ledger = dict(new_rays=inherited, inherited_count=inherited,
                           cap=options.max_rays, calls=[])
        journal = self.root / 'ledger.json'
        if journal.is_file():
            self.ledger = json.loads(journal.read_text())
            if (self.ledger['cap'] != options.max_rays
                    or self.ledger.get('inherited_count', 0) != inherited):
                raise ValueError('The saved route and its ray-budget inputs differ')
            for call in self.ledger['calls']:
                self._index_call(call)

    def _index_call(self, call):
        for batch in call['batches']:
            for index, cache_key in enumerate(batch['keys']):
                self.entries[cache_key] = (batch['file'], index)

    def _write_ledger(self, ledger):
        dump(self.root / 'ledger.json', ledger)

    def append_call(self, call):
        ledger = self.ledger
        replacement = dict(ledger, new_rays=ledger['new_rays'] + call['rays'],
                           calls=[*ledger['calls'], call])
        self._write_ledger(replacement)
        self.ledger = replacement
        self._index_call(call)

    @contextmanager
    def tracing(self, evaluator_factory):
        """Keep each spawned worker's evaluator and compiled kernels for the route."""
        if self.options.workers == 1:
            yield
            return
        pool = trace_pool(self.options.workers, self.tracer, evaluator_factory)
        self._pool = pool
        try:
            yield
        finally:
            self._pool = None
            pool.shutdown(cancel_futures=True)

    def _trace_missing(self, ev, seeds, delta, missing, size, width):
        batches = ((lo, missing[lo:lo + width]) for lo in range(0, len(missing), width))
        if self._pool is None:
            for lo, ids in batches:
                yield lo, ids, self.tracer(ev, seeds[ids], delta[ids], steps=STEPS)
            return

        # These accumulators affect validation only. Merge worker extrema in
        # serial batch order to retain the sign check across worker boundaries.
        extrema = ev.beta_min, ev.beta_max, ev.max_u

        def submit(lo, ids):
            return lo, ids, self._pool.submit(_trace_batch, seeds[ids], delta[ids], size, extrema)

        pending = deque(submit(lo, ids) for lo, ids in islice(batches, self.options.workers))
        for lo, ids in batches:
            pending.append(submit(lo, ids))
            yield self._collect_batch(ev, pending.popleft())
        while pending:
            yield self._collect_batch(ev, pending.popleft())

    @staticmethod
    def _collect_batch(ev, batch):
        lo, ids, future = batch
        result, extrema = future.result()
        merge_extrema(ev, extrema)
        return lo, ids, result

    def book(self):
        return self.ledger

    def check_call_budget(self, count):
        budget = self.book()
        assert budget['new_rays'] + count <= budget['cap'], (
            'ray cap', budget['new_rays'], count, budget['cap'])

    def check_projection_budget(self, count):
        budget = self.book()
        if budget['new_rays'] + count > budget['cap']:
            raise RuntimeError('plane_seeded_wall_rows unit projection exceeds cap before bisection')

    def plane_charged(self, k, prefix=''):
        return sum(c['rays'] for c in self.book()['calls'] if c['label'].startswith(prefix)
                   and re.search(r'(?:half|plane|strip)' + str(k) + r'(?!\d)', c['label']))

    @lru_cache(maxsize=8)
    def _file(self, filename):
        return load(self._file_path(filename))

    def _file_path(self, filename):
        return self.root / filename

    def lookup(self, seed, delta):
        q = np.array(seed, float)
        q[1:] %= TAU
        location = self.entries.get(key(q, delta))
        if location is None:
            return self.archive(q, delta) if self.archive is not None else None
        filename, index = location
        return {name: array[index] for name, array in self._file(filename).items()}

    def trace_cached(self, ev, seeds, delta, label):
        seeds = np.array(seeds, float)
        seeds[:, 1:] %= TAU
        delta = np.broadcast_to(delta, (len(seeds),)).copy()
        keys = [key(q, d) for q, d in zip(seeds, delta)]
        if not len(seeds):
            return dict(seed=seeds, delta=delta, history=np.empty((0, STEPS, 4, 3)),
                        hit=np.empty(0, bool), wall=np.empty((0, 3)), wall_length=np.empty(0))
        found, missing = {}, []
        for i, k in enumerate(keys):
            if k not in found:
                found[k] = self.lookup(seeds[i], delta[i])
                if found[k] is None:
                    missing.append(i)
        budget = self.book()
        if missing:
            self.check_call_budget(len(missing))
            workers = self.options.trace_sizing_workers
            size = batch_size(len(missing), workers)
            width = size * max(1, min(4, math.ceil(math.ceil(len(missing) / size) / workers)))
            ev.trace_batch_size = size
            batches = []
            for lo, ids, actual in self._trace_missing(ev, seeds, delta, missing, size, width):
                actual.update(seed=seeds[ids], delta=delta[ids])
                filename = f"{len(budget['calls']):08d}_{lo}.npz"
                save(self.root / filename, **actual)
                batch_keys = [keys[i] for i in ids]
                batches.append(dict(file=filename, keys=batch_keys))
                for index, cache_key in enumerate(batch_keys):
                    found[cache_key] = {name: data[index] for name, data in actual.items()}
            call = dict(label=label, rays=len(missing), batch_size=size, batches=batches,
                        identity_segment=None)
            self.append_call(call)
        return {name: np.array([found[k][name] for k in keys]) for name in found[keys[0]]}

    def trace_sides(self, ev, seeds, deltas, labels):
        return [self.trace_cached(ev, seeds, delta, label) for delta, label in zip(deltas, labels)]
