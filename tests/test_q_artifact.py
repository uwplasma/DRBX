"""Q bank artifact integrity, stale identities and unchanged-stat mutations."""
import json
import os
import numpy as np
import pytest

from tests.test_q_bank import patch, bank
from drbx.stencils.q_artifact import save_q_bank, load_q_bank, artifact_footprint


def _rewrite(path, mutate, *, compressed=True):
    with np.load(path, allow_pickle=False) as z:
        payload = {k: z[k].copy() for k in z.files}
    mutate(payload)
    with path.open('wb') as f:
        (np.savez_compressed if compressed else np.savez)(f, **payload)


def test_exact_artifact_roundtrip_and_footprint(bank, tmp_path):
    path = save_q_bank(bank, tmp_path / 'q.npz')
    loaded = load_q_bank(path, expected_identity=bank.identity,
                         geometry_identity=bank.metadata['geometry_identity'])
    assert loaded.identity == bank.identity
    for group in ('arrays', 'diagnostics'):
        for k, a in getattr(bank, group).items():
            b = getattr(loaded, group)[k]
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    footprint = artifact_footprint(bank, path)
    assert footprint['compressed_bytes'] == path.stat().st_size
    assert footprint['host_logical_bytes'] == footprint['runtime_logical_bytes'] + footprint['host_diagnostic_bytes']
    assert footprint['load_array_temporary_bound_bytes'] >= footprint['host_logical_bytes']
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('key,value', [('geometry_identity', 'wrong'), ('source_identity', 'wrong'),
    ('topology_hash', 'wrong'), ('trace_hash', 'wrong'), ('expected_identity', 'wrong'),
    ('spans', (1/32,)), ('additional_identity', {'evaluator': 'different'})])
def test_wrong_or_stale_identity(bank, tmp_path, key, value):
    path = save_q_bank(bank, tmp_path / 'q.npz')
    with pytest.raises(ValueError, match='mismatch'):
        load_q_bank(path, **{key: value})


def test_tampered_array_with_unchanged_timestamp(bank, tmp_path):
    path = save_q_bank(bank, tmp_path / 'q.npz')
    _rewrite(path, lambda p: None, compressed=False)
    stamp = path.stat()
    load_q_bank(path)
    def change(p):
        p['arrays__row_value_D'][0, 0, 0] += 1
    _rewrite(path, change, compressed=False)
    assert path.stat().st_size == stamp.st_size
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match='checksum'):
        load_q_bank(path)


def test_metadata_tamper_and_incomplete_manifest(bank, tmp_path):
    path = save_q_bank(bank, tmp_path / 'q.npz')
    def change(p):
        m = json.loads(str(p['manifest'])); m['metadata']['geometry_identity'] = 'tampered'
        p['manifest'] = np.array(json.dumps(m))
    _rewrite(path, change)
    with pytest.raises(ValueError, match='content identity'):
        load_q_bank(path)
    save_q_bank(bank, path)
    def omit(p):
        del p['arrays__raw']
    _rewrite(path, omit)
    with pytest.raises(ValueError, match='incomplete'):
        load_q_bank(path)
