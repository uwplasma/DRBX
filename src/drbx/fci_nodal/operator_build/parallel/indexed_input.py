"""bounded indexed parallel assembly input_reader.IndexedGeometryArray.take against the placed indexed immutable geometry frames frame reader."""
from collections import OrderedDict
import json
from pathlib import Path
import sqlite3

import numpy as np

from ..._hashing import sha256_file as sha
from ...geometry_build import parallel_geometry_reader


ROWS = 32000
BYTES = 64 * 2**20
INPUTS = ('H', 'faces', 'wall', 'face_splice_provenance')


class IndexedInput:
    def __init__(self, handoff):
        self.handoff = Path(handoff)
        if self.handoff.is_dir():
            self.handoff /= 'parallel_geometry_reader.json'
        self.info = json.loads(self.handoff.read_text())
        if self.info['schema'] != parallel_geometry_reader.PARALLEL_GEOMETRY_READER_SCHEMA:
            raise ValueError('Unsupported Q geometry reader schema')
        self.entries = {name: self.info['geometry'][name] for name in INPUTS}
        if any(entry['format'] != 'drbx.parallel.owner_geometry_containers.v1' or entry['array'] != name
               for name, entry in self.entries.items()):
            raise ValueError('Invalid Q owner entry')
        span_paths = {Path(entry['index']).resolve() for entry in self.entries.values()}
        if span_paths != {(self.handoff.parent / 'parallel_geometry_spans.sqlite').resolve()}:
            raise ValueError('Q geometry entries must share the placed span index')
        self.index = span_paths.pop()
        self.db = None
        self.cache = OrderedDict()
        self.verified = set()
        self.cache_bytes = 0

    def identity(self):
        public = self.handoff.parent / 'index.sqlite'
        meta = json.loads((self.handoff.parent / 'geometry.json').read_text())
        if sha(self.index) != self.info['index_sha256']:
            raise ValueError('Geometry Q span index differs')
        if sha(public) != self.info['geometry_index_sha256'] or meta['index_sha256'] != self.info['geometry_index_sha256']:
            raise ValueError('Geometry public index differs')
        for name, entry in self.entries.items():
            if any(entry[key] != meta['arrays'][name][key] for key in ('shape', 'dtype')):
                raise ValueError(f'Geometry Q input schema differs for {name}')
        return dict(handoff_sha256=sha(self.handoff), index_sha256=self.info['index_sha256'],
                    geometry_index_sha256=self.info['geometry_index_sha256'])

    def shape(self, name):
        return tuple(self.entries[name]['shape'])

    def _frame(self, ref, key):
        locator = (ref, key)
        if locator in self.cache:
            self.cache.move_to_end(locator)
            return self.cache[locator]
        if not ref.startswith(parallel_geometry_reader.FRAME_PREFIX):
            raise ValueError('Unsupported geometry frame reference')
        array = parallel_geometry_reader.get(ref, [key], verify=ref not in self.verified)[key]
        self.verified.add(ref)
        while self.cache and (self.cache_bytes + array.nbytes > BYTES or len(self.cache) >= 128):
            _, prior = self.cache.popitem(last=False)
            self.cache_bytes -= prior.nbytes
        if array.nbytes <= BYTES:
            self.cache[locator] = array
            self.cache_bytes += array.nbytes
        return array

    def ranges(self, name, rows=ROWS):
        n = self.shape(name)[0]
        for lo in range(0, n, rows):
            yield lo, min(lo + rows, n)

    def read(self, name, lo, hi):
        if not 0 <= lo <= hi <= self.shape(name)[0]:
            raise IndexError((name, lo, hi))
        if hi - lo > ROWS:
            raise MemoryError('Q geometry slices are limited to 32000 rows')
        return self.take(name, np.arange(lo, hi, dtype=np.int64))

    def take(self, name, ids):
        ids = np.asarray(ids, np.int64)
        shape = self.shape(name)
        dtype = np.dtype(self.entries[name]['dtype'])
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= shape[0]):
            raise IndexError(f'Invalid canonical row IDs for {name}')
        result_shape = (len(ids), *shape[1:])
        if len(ids) > ROWS or int(np.prod(result_shape)) * dtype.itemsize > BYTES:
            raise MemoryError('Q geometry selection exceeds the bounded reader contract')
        result = np.empty(result_shape, dtype)
        if not len(ids):
            return result
        if self.db is None:
            self.db = sqlite3.connect(self.index.as_uri() + '?mode=ro&immutable=1', uri=True)
            self.db.execute('pragma cache_size=-4096')
        order = np.argsort(ids, kind='stable')
        sorted_ids = ids[order]
        at = 0
        while at < len(ids):
            current = int(sorted_ids[at])
            row = self.db.execute(
                'select lo,hi,ref,key,local_lo from spans where array=? and lo<=? order by lo desc limit 1',
                (name, current)).fetchone()
            if row is None or current >= row[1]:
                raise ValueError(f'Uncovered Q row {name}[{current}]')
            lo, hi, ref, key, local = row
            end = np.searchsorted(sorted_ids, hi)
            values = self._frame(ref, key)
            if (values.dtype != dtype or values.shape[1:] != shape[1:]
                    or local < 0 or local + hi - lo > len(values)):
                raise ValueError(f'Invalid geometry payload for {name}')
            result[order[at:end]] = values[sorted_ids[at:end] - lo + local]
            at = end
        return result

    def close(self):
        if self.db is not None:
            self.db.close()
        self.cache.clear()
