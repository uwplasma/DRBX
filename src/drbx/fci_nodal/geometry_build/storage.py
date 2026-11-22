"""Local immutable leaves and published indexed immutable geometry frames frames."""
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from .._hashing import sha256_file as sha
from .parallel_geometry_reader import DONE, HEADER, MAGIC, encode


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_value,
                                     separators=(',', ':')).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=json_value, indent=2) + '\n')


def connect_ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)


def arrays(root, keys=None):
    root = Path(root)
    if root.suffix == '.npz':
        with np.load(root, allow_pickle=False) as source:
            return {key: source[key] for key in (source.files if keys is None else keys)}
    record = load(root / 'complete.json')
    result = {}
    for key in record['arrays'] if keys is None else keys:
        spec = record['arrays'][key]
        container = record.get('containers', {}).get(spec['path'])
        path = root / (container['path'] if container else spec['path'])
        if 'offset' not in spec:
            result[key] = np.load(path, mmap_mode='r', allow_pickle=False)
        elif np.prod(spec['shape'], dtype=np.int64):
            offset = (container['offset'] if container else 0) + spec['offset']
            result[key] = np.memmap(path, dtype=spec['dtype'], mode='r', offset=offset,
                                    shape=tuple(spec['shape']), order='C')
        else:
            result[key] = np.empty(tuple(spec['shape']), dtype=spec['dtype'])
    return result


def verify(root):
    root = Path(root)
    record = load(root / 'complete.json')
    if 'frame' in record:
        from .parallel_geometry_reader import frame_info
        container = record['containers']['data.bin']
        frame = dict(record['frame'], path=str((root / container['path']).resolve()))
        frame_info(encode(frame, root), verify=True)
    else:
        for name, digest in {x['path']: x['sha256'] for x in record['arrays'].values()}.items():
            container = record.get('containers', {}).get(name)
            if container:
                with (root / container['path']).open('rb') as source:
                    source.seek(container['offset'])
                    actual = hashlib.sha256(source.read(container['bytes'])).hexdigest()
            else:
                actual = sha(root / name)
            if actual != digest:
                raise RuntimeError(f'Geometry leaf changed: {root}')
    return record


class FrameWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'geometry.gds'

    def write(self, relative, values, meta):
        leaf = self.root / relative
        identity = fingerprint(relative)
        entries = {}
        payload = bytearray()
        for name, value in values.items():
            value = np.asarray(value)
            if value.dtype.hasobject:
                raise TypeError('Object arrays are unsupported')
            raw = value.tobytes(order='C')
            payload.extend(bytes((-len(payload)) % 64))
            entries[name] = dict(path='data.bin', offset=len(payload), shape=list(value.shape),
                                 dtype=value.dtype.str, bytes=len(raw))
            payload.extend(raw)
        digest = hashlib.sha256(payload).hexdigest()
        for entry in entries.values():
            entry['sha256'] = digest
        record = dict(identity=identity, meta=meta, arrays=entries,
                      data_files={'data.bin': dict(bytes=len(payload), sha256=digest)},
                      storage='geometry-indexed-stage-v1')
        body = json.dumps(dict(logical=relative, record=record), default=json_value,
                          separators=(',', ':')).encode()
        leaf.mkdir(parents=True, exist_ok=True)
        with self.path.open('ab') as container:
            start = container.tell()
            header = HEADER.pack(MAGIC, len(body), len(payload))
            padding = bytes((-(start + len(header) + len(body))) % 64)
            data = start + len(header) + len(body) + len(padding)
            checksum = hashlib.sha256(header + body + padding)
            checksum.update(payload)
            container.write(header + body + padding)
            container.write(payload)
            container.write(checksum.digest() + DONE)
        frame = dict(path=self.path.name, offset=start, bytes=data + len(payload) + 40 - start,
                     sha256=checksum.hexdigest())
        record.update(frame=frame, containers={'data.bin': dict(
            path=str(Path(*(['..'] * len(Path(relative).parts))) / self.path.name),
            offset=data, bytes=len(payload))})
        dump(leaf / 'complete.json', record)
        return record
