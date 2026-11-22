"""Data-free receipt-chain and adoption tests."""

from dataclasses import replace
from hashlib import sha256
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_receipt_chain_invalidates_dependents(tmp_path, monkeypatch, runtime_module):
    receipts = runtime_module('case._receipts')
    backend = tmp_path/'backend'
    driver = backend/'case'
    driver.mkdir(parents=True)
    (driver/'_worker.py').write_text('# supplied by the driver')
    direct = backend/'evaluators'
    direct.mkdir(parents=True)
    loaded = direct/'_host_guards.py'
    loaded.write_text('# imported by field table')
    unused = direct/'unused.py'
    unused.write_text('# never imported')
    monkeypatch.setattr(receipts, 'BACKEND_ROOT', backend)
    module = ModuleType('test_backend_source_inventory')
    module.__file__ = str(loaded)
    name = receipts.__package__.rsplit('.',1)[0] + '.evaluators.test_backend_source_inventory'
    names = ('metric', 'field_table', 'polygon')
    current = {}
    calls = []

    def build():
        for stage_name in names:
            def work(out):
                calls.append(stage_name)
                path = out/'product.json'
                receipts.write_json(path, {'stage':stage_name})
                with monkeypatch.context() as worker:
                    if stage_name == 'field_table':
                        worker.setitem(sys.modules, name, module)
                    return {'product':path}, 0, 0, receipts.worker_code_files(stage_name)

            current[stage_name] = receipts.stage(
                tmp_path, stage_name, settings={}, options={}, inputs=(), expected=('product',),
                action=work, dependencies=tuple(current[key] for key in receipts.PREDECESSORS[stage_name]),
            )

    build()
    metric = tmp_path/'metric/receipt.json'
    original = metric.read_bytes()
    receipts.write_json(metric, dict(current['metric'][0], reviewer='changed current receipt'))
    with pytest.raises(ValueError, match='Stale stage field_table: predecessor receipts differ'):
        receipts.verify_chain(tmp_path, 'field_table')
    metric.write_bytes(original)
    assert set(current['field_table'][0]['identity']['code_files']) == {'case/_worker.py', 'evaluators/_host_guards.py'}
    unused.write_text('# changed unimported source')
    receipts.verify_chain(tmp_path, 'polygon')
    loaded.write_text('# changed imported source')
    with pytest.raises(ValueError, match='Stale stage field_table: code identity changed'):
        receipts.verify_chain(tmp_path, 'field_table')
    calls.clear()
    build()
    assert calls == ['field_table', 'polygon']
    receipts.verify_chain(tmp_path, 'polygon')


@pytest.mark.parametrize('size', (0, 23, 2 * 1024**2 + 31))
def test_stream_sha256_matches_whole_file(size, tmp_path, runtime_module):
    hashing = runtime_module('_hashing')
    contents = (b'\x00sample\xff' * ((size + 7) // 8))[:size]
    path = tmp_path/'data'
    path.write_bytes(contents)
    with path.open('rb') as stream:
        assert hashing.sha256_stream(stream) == sha256(contents).hexdigest()


GEOMETRY_SOURCES = (
    'geometry_build/dual_cells.py', 'geometry_build/_volume_base.py',
    'geometry_build/_shard_jobs.py', 'geometry_build/_geometry_worker.py',
)


@pytest.mark.parametrize('changed_name', GEOMETRY_SOURCES)
def test_geometry_fingerprint_includes_executed_source(changed_name, tmp_path, monkeypatch, runtime_module):
    receipts = runtime_module('case._receipts')
    for name in GEOMETRY_SOURCES:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# geometry source')
    monkeypatch.setattr(receipts, 'BACKEND_ROOT', tmp_path)

    files = receipts.worker_code_files('geometry')
    assert set(files) == set(GEOMETRY_SOURCES)
    assert receipts.worker_code_files('metric') == {}
    identity = dict(code_files=files, code_sha256=receipts.code_identity(files))
    receipts.verify_code('geometry', identity)

    omitted = {name: digest for name, digest in files.items() if name != changed_name}
    with pytest.raises(ValueError, match='code identity changed'):
        receipts.verify_code('geometry', dict(code_files=omitted, code_sha256=receipts.code_identity(omitted)))
    (tmp_path/changed_name).write_text('# changed geometry source')
    with pytest.raises(ValueError, match='code identity changed'):
        receipts.verify_code('geometry', identity)


def test_adoption_refuses_wrong_table_hash(tmp_path, runtime_module):
    receipts = runtime_module('case._receipts')
    adopted = runtime_module('case._adoption')
    indexed = tmp_path/'geometry'
    indexed.mkdir()
    table = tmp_path/'table'
    table.write_text('table bytes')
    schema = runtime_module('geometry_build.reader').GEOMETRY_SCHEMA
    receipts.write_json(indexed/'geometry.json', dict(schema=schema, N=16, E=16, symmetry='nfp', field_periods=4,
                                                      field_table_sha256='wrong-table'))
    origin = dict(products={'indexed_geometry':str(indexed)})
    with pytest.raises(ValueError, match='field-table hash'):
        adopted.validate_adoption('geometry', origin, (16, 16, 'nfp', 4), {'field_table':table},
                                  SimpleNamespace(field_filter=None))


def test_runtime_resolution_limits_and_overrides(runtime_module):
    runtime = runtime_module().RuntimeOptions()
    driver = runtime_module('case.case')
    assert driver._settings('mms_prepare')['XLA_FLAGS'].endswith('=1')
    runtime.check_mms_owner_count(16, 2)
    with pytest.raises(ValueError, match='one owner'):
        runtime.check_mms_owner_count(32, 2)
    runtime.check_first_milestone(16)
    with pytest.raises(ValueError, match='resolution 16 only'):
        runtime.check_first_milestone(32)
    varied = replace(runtime, owner_count=2, cpu_device_count=2, single_owner_from_resolution=None,
                     first_milestone_resolution=32)
    assert driver._settings('mms_prepare', host_cpu_devices=varied.cpu_device_count)['XLA_FLAGS'].endswith('=2')
    varied.check_mms_owner_count(32, 2)
    varied.check_first_milestone(32)
    varied.check_cpu_device_capacity(2)
    with pytest.raises(ValueError, match='needs that many CPU devices'):
        varied.check_cpu_device_capacity(1)
