"""Atomic, content-checked storage for compact paired Q banks.

Legacy Q v2 and P v2/v3 readers are unchanged. A bank is one complete-owner
work unit; the research runner owns campaign orchestration and writer locking.
"""
from pathlib import Path
import json
import os
import tempfile
import numpy as np

from .q_bank import QBank, SCHEMA, _json, content_identity
from .q_parallel import sha256_array


def save_q_bank(bank, path):
    bank.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=SCHEMA, metadata=bank.metadata, identity=bank.identity,
                    arrays={k: sha256_array(a) for k, a in bank.arrays.items()},
                    diagnostics={k: sha256_array(a) for k, a in bank.diagnostics.items()})
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, manifest=np.array(_json(manifest)),
                **{f'arrays__{k}': a for k, a in bank.arrays.items()},
                **{f'diagnostics__{k}': a for k, a in bank.diagnostics.items()})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return path


def load_q_bank(path, *, expected_identity=None, source_identity=None,
                geometry_identity=None, topology_hash=None, trace_hash=None,
                spans=None, additional_identity=None):
    """Read and rehash all content on every load, including unchanged-stat files."""
    try:
        with np.load(path, allow_pickle=False) as z:
            manifest = json.loads(str(z['manifest']))
            if manifest.get('schema') != SCHEMA:
                raise ValueError('Q bank artifact schema mismatch')
            groups = {}
            expected_members = {'manifest'}
            for group in ('arrays', 'diagnostics'):
                hashes = manifest[group]
                if not isinstance(hashes, dict):
                    raise ValueError('Q bank checksum manifest incomplete')
                values = {}
                for k, checksum in hashes.items():
                    member = f'{group}__{k}'
                    expected_members.add(member)
                    a = z[member].copy()
                    if sha256_array(a) != checksum:
                        raise ValueError(f'Q bank {member} checksum mismatch')
                    values[k] = a
                groups[group] = values
            if set(z.files) != expected_members:
                raise ValueError('Q bank checksum manifest incomplete/unexpected members')
        m = manifest['metadata']
        digest = content_identity(m, groups['arrays'], groups['diagnostics'])
        if digest != manifest['identity']:
            raise ValueError('Q bank content identity mismatch')
        if expected_identity is not None and digest != expected_identity:
            raise ValueError('Q bank expected identity mismatch')
        for k, expected in (('source_identity', source_identity), ('geometry_identity', geometry_identity),
                            ('topology_hash', topology_hash), ('trace_hash', trace_hash),
                            ('additional_identity', additional_identity)):
            if expected is not None and m.get(k) != expected:
                raise ValueError(f'Q bank {k} mismatch')
        if spans is not None and tuple(m.get('spans', ())) != tuple(spans):
            raise ValueError('Q bank spans mismatch')
        return QBank(m, **groups).validate()
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Q bank artifact incomplete or invalid manifest') from exc


def artifact_footprint(bank, path):
    """Disk compression and a conservative decoded-buffer load bound.

    No device allocation is inferred. The bound includes a decoder copy of
    the largest member plus all retained buffers, but excludes zip/Python
    library overhead and is explicitly not a measured peak RSS.
    """
    result = bank.footprint()
    result['compressed_bytes'] = Path(path).stat().st_size
    largest = max((a.nbytes for a in (*bank.arrays.values(), *bank.diagnostics.values())), default=0)
    result['load_array_temporary_bound_bytes'] = result['host_logical_bytes'] + largest
    result['load_bound_excludes_library_overhead'] = True
    return result
