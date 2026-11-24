"""Byte comparisons and difference receipts for current-format owner cases."""

from contextlib import closing
import json

import numpy as np

from .backend import backend_module


def array_difference(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    comparable = actual.shape == reference.shape and actual.dtype == reference.dtype
    equal = comparable and actual.tobytes() == reference.tobytes()
    row = dict(bytes_equal=equal, max_abs=None, max_abs_reference=None)
    if not comparable:
        row.update(actual_shape=actual.shape, reference_shape=reference.shape,
                   actual_dtype=actual.dtype.str, reference_dtype=reference.dtype.str)
    if reference.dtype.kind in 'biu':
        row['max_abs_reference'] = max(abs(int(reference.min())), abs(int(reference.max()))) if reference.size else 0
        if comparable:
            if equal:
                row['max_abs'] = 0
            elif reference.dtype.itemsize < 8:
                row['max_abs'] = int(np.max(np.abs(np.subtract(actual, reference, dtype=np.int64)), initial=0))
            else:
                row['max_abs'] = max((abs(int(a)-int(b)) for a,b in zip(actual.flat,reference.flat)), default=0)
    elif reference.dtype.kind in 'fc':
        finite = np.isfinite(reference)
        row['max_abs_reference'] = float(np.max(np.abs(reference[finite]), initial=0))
        if comparable:
            if equal:
                row['max_abs'] = 0.0
            elif np.array_equal(actual[~finite], reference[~finite], equal_nan=True) and np.isfinite(actual[finite]).all():
                wide = np.result_type(actual.dtype, np.float64)
                delta = np.subtract(actual[finite], reference[finite], dtype=wide)
                row['max_abs'] = float(np.max(np.abs(delta), initial=0))
    return row


def require_bytes(differences):
    mismatches = [name for name, row in differences.items() if not row['bytes_equal']]
    assert not mismatches, f'{len(mismatches)} differences (see JSON receipt): {mismatches[:12]}'


def compare_array_files(actual, reference, record, *, prefix='', path_key=None):
    """Compare every NPY/NPZ member at its current relative product path."""
    if actual.is_dir() and reference.is_dir():
        def members(root):
            paths = {}
            for path in root.rglob('*'):
                if path.suffix in {'.npy', '.npz'}:
                    relative = path.relative_to(root)
                    key = relative if path_key is None else path_key(relative)
                    paths.setdefault(key, []).append(relative)
            return paths

        left, right = members(actual), members(reference)
        if left.keys() != right.keys():
            record(prefix+'array paths', sorted(map(str, left)), sorted(map(str, right)))
        for side, paths in (('actual', left), ('reference', right)):
            for key, files in sorted(paths.items()):
                if len(files) > 1:
                    record(f'{prefix}{key}/{side} path collision', sorted(map(str, files)), [])
        for key in sorted(left.keys() & right.keys()):
            if len(left[key]) == len(right[key]) == 1:
                compare_array_files(actual/left[key][0], reference/right[key][0],
                                    record, prefix=f'{prefix}{key}/')
    elif actual.suffix == reference.suffix == '.npy':
        record(prefix.rstrip('/'), np.load(actual, mmap_mode='r', allow_pickle=False),
               np.load(reference, mmap_mode='r', allow_pickle=False))
    elif actual.suffix == reference.suffix == '.npz':
        with np.load(actual, allow_pickle=False) as left, np.load(reference, allow_pickle=False) as right:
            if set(left.files) != set(right.files):
                record(prefix+'members', sorted(left.files), sorted(right.files))
            for name in sorted(set(left.files) & set(right.files)):
                record(prefix+name, left[name], right[name])
    else:
        raise ValueError(f'Expected matching array products: {actual}, {reference}')


def compare_indexed(actual, reference, record):
    reader = backend_module('geometry_build').GeometryReader
    with closing(reader(actual)) as left, closing(reader(reference)) as right:
        a, b = left.manifest['arrays'], right.manifest['arrays']
        record('geometry/array schema', json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        for name in sorted(a.keys() & b.keys()):
            rows = min(left.shape(name)[0], right.shape(name)[0])
            for lo in range(0, rows, 8192):
                hi = min(lo+8192, rows)
                record(f'geometry/{name}/{lo}:{hi}', left.read(name, lo, hi), right.read(name, lo, hi))

    arrays = backend_module('geometry_build.storage').arrays
    left = {p.name for p in (actual/'fits').glob('leaf_*')}
    right = {p.name for p in (reference/'fits').glob('leaf_*')}
    if left != right:
        record('geometry/fit paths', sorted(left), sorted(right))
    for leaf in sorted(left & right):
        values, expected = (arrays(root/'fits'/leaf) for root in (actual, reference))
        if values.keys() != expected.keys():
            record(f'geometry/fits/{leaf}/members', sorted(values), sorted(expected))
        for name in sorted(values.keys() & expected.keys()):
            record(f'geometry/fits/{leaf}/{name}', values[name], expected[name])


def owner_rows(directory):
    shards = sorted(directory.glob('owner[0-9][0-9][0-9][0-9]'))
    ids, state, rhs, forced = [], [], [], []
    for shard in shards:
        ids.append(np.load(shard/'node_ids.npy', allow_pickle=False))
        state.append(np.load(shard/'nodal_state.npy', allow_pickle=False))
        with np.load(shard/'static_rhs.npz', allow_pickle=False) as saved:
            rhs.append(saved['rhs'])
            forced.append(saved['rhs_manufactured'])
    ids = np.concatenate(ids)
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(len(ids))):
        raise ValueError('Owner state must cover each canonical node exactly once')
    return {key: np.concatenate(parts)[order]
            for key, parts in (('step_3', state), ('rhs', rhs), ('rhs_manufactured', forced))}
