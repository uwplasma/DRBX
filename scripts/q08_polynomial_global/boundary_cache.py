"""Exact, bounded wall-row cache for repeated manufactured BC queries."""
import time
import numpy as np


class BoundaryCache:
    def __init__(self, producer, max_bytes=0):
        self.producer = producer
        self.max_bytes = max_bytes
        self.entries = {}
        self.bytes = 0
        self.hits = self.misses = 0
        self.producer_seconds = self.restore_seconds = 0.0

    def __call__(self, bank, case):
        key = (bank.identity, int(case))
        if key in self.entries:
            start = time.perf_counter()
            nr, wall, groups = self.entries[key]
            result = []
            for cls, arrays in groups:
                values = []
                for shape, axis, template, rows in arrays:
                    a = np.empty(shape, dtype=rows.dtype)
                    view = np.moveaxis(a, axis, 0)
                    view[:] = template
                    view[wall] = rows
                    values.append(a)
                result.append(cls(*values))
            self.hits += 1
            self.restore_seconds += time.perf_counter() - start
            return tuple(result)

        start = time.perf_counter()
        result = self.producer(bank, case)
        self.producer_seconds += time.perf_counter() - start
        nr = len(bank.raw)
        wall = np.asarray(bank.wall_index, dtype=np.int64).copy()
        if len(np.unique(wall)) != len(wall) or np.any((wall < 0) | (wall >= nr)):
            raise ValueError('invalid wall rows in BC cache')
        nonwall = np.setdiff1d(np.arange(nr), wall)
        groups = []
        size = wall.nbytes
        for gi, group in enumerate(result):
            arrays = []
            for original in group:
                a = np.asarray(original)
                axis = 1 if gi < 2 else 0
                view = np.moveaxis(a, axis, 0)
                if len(view) != nr or a.dtype != np.float64:
                    raise ValueError('BC cache expects float64 raw-cell arrays')
                template = (view[nonwall[0]].copy() if len(nonwall)
                            else np.zeros(view.shape[1:], a.dtype))
                # Exact bits, including signed zero. Only small batches are
                # gathered while proving all nonwall rows share the template.
                for start in range(0, len(nonwall), 128):
                    block = np.ascontiguousarray(view[nonwall[start:start + 128]])
                    target = np.broadcast_to(template.view(np.uint64), block.shape)
                    if not np.array_equal(block.view(np.uint64), target):
                        raise ValueError('nonuniform nonwall BC values cannot be compressed')
                rows = view[wall].copy()
                size += rows.nbytes + template.nbytes
                arrays.append((a.shape, axis, template, rows))
            groups.append((type(group), arrays))
        if self.max_bytes and self.bytes + size > self.max_bytes:
            raise MemoryError('exact boundary cache exceeds declared memory budget')
        self.entries[key] = (nr, wall, groups)
        self.bytes += size
        self.misses += 1
        return result

    def stats(self):
        return dict(entries=len(self.entries), hits=self.hits, producer_calls=self.misses,
                    retained_bytes=self.bytes, budget_bytes=self.max_bytes,
                    producer_seconds=self.producer_seconds, restore_seconds=self.restore_seconds,
                    scope='Exact wall rows and bitwise-uniform nonwall templates; no dense all-case cache')


class CachedAPI:
    def __init__(self, api):
        self.api = api
        self.cache = BoundaryCache(api.boundaries)

    def boundaries(self, bank, case):
        return self.cache(bank, case)

    def __getattr__(self, name):
        return getattr(self.api, name)
