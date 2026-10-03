"""Exact BC values, bounded cache, reference reuse and timing provenance."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/q08_polynomial_global'), str(ROOT)]
from scripts.q08_extraction_global import common
import boundary_values as bv
from boundary_cache import CachedAPI, restore
from gpu_stage import CompilerReceipts
from timing import PhaseLedger, record_totals


def bank(wall=(4, 0, 2)):
    rng = np.random.default_rng(2191)
    points = rng.normal(size=(45, 3))
    points[0] = [1., -0., 0.]
    nw = len(wall)
    return SimpleNamespace(identity='bounded-bc-fixture', raw=np.arange(6),
        wall_index=np.asarray(wall, dtype=int), query_table=points,
        wall_node_query=np.tile(np.arange(35), (nw, 1)),
        wall_slot_query=np.tile([36, 37, 38, 40, 41], (nw, 1)),
        boundary_wall_normal=rng.normal(size=(nw, 35, 3)))


@pytest.mark.parametrize('wall', [(), (4, 0, 2), tuple(range(6))])
def test_all_catalogue_cases_bitwise_including_nonwall_padding(wall):
    b = bank(wall)
    expected = common.boundaries(b)
    for case in range(common.NF):
        groups = bv.compact_boundaries(b, case, common, batch_rows=2)
        actual = restore(b.wall_index, groups)
        for a, e in zip(actual, expected):
            for x, y in zip(a, e):
                assert x.shape == y[case].shape
                assert x.dtype == y.dtype
                assert x.tobytes() == y[case].tobytes()


def test_compact_cache_does_not_evaluate_unused_cases_and_protects_storage(monkeypatch):
    b = bank()
    expected = common.boundaries(b, 21)
    api = CachedAPI(common, optimize=True)
    monkeypatch.setattr(common, 'fields', lambda *a: pytest.fail('all-state producer called'))
    first, second = api.boundaries(b, 21), api.boundaries(b, 21)
    first[0][0][:] = -100
    for a, e in zip(second, expected):
        for x, y in zip(a, e):
            assert x.tobytes() == y.tobytes()
    assert api.cache.misses == 1 and api.cache.hits == 1
    limited = CachedAPI(common, optimize=True)
    limited.cache.max_bytes = 1
    with pytest.raises(MemoryError, match='budget'):
        limited.boundaries(b, 0)
    assert not limited.cache.entries


def test_live_boundary_gate_detects_formula_drift(monkeypatch):
    assert bv.catalogue_replay(bank(), common)['all_arrays_bitwise']
    original = bv.fields
    def wrong(*args):
        v, g = original(*args)
        return v+.001, g
    monkeypatch.setattr(bv, 'fields', wrong)
    with pytest.raises(ValueError, match='bitwise replay'):
        bv.catalogue_replay(bank(), common)


def test_compiler_read_once_per_invocation_and_fresh_check_detects_changes(tmp_path, monkeypatch):
    path = tmp_path/'compiler.txt'
    path.write_text('custom_call_target="cusolver_getrf_ffi"')
    original, calls = Path.read_bytes, []
    def counted(self):
        calls.append(self)
        return original(self)
    monkeypatch.setattr(Path, 'read_bytes', counted)
    cache = CompilerReceipts(tmp_path)
    a = cache.get('compiler.txt')
    assert cache.get('compiler.txt') == a
    assert calls == [path]
    path.write_text('custom_call_target="cusolver_geev_ffi"')
    with pytest.raises(ValueError, match='eigensolver'):
        CompilerReceipts(tmp_path).get('compiler.txt')
    outside = tmp_path.parent/'outside-proof.txt'
    outside.write_text('')
    with pytest.raises(ValueError, match='escapes'):
        cache.get('../outside-proof.txt')


def test_timing_is_exclusive_and_shared_work_counted_once():
    ticks = iter([0., 2., 5., 9.])
    ledger = PhaseLedger(clock=lambda: next(ticks))
    ledger.lap('BC'); ledger.lap('reference')
    result = ledger.finish()
    assert result['elapsed_seconds'] == 9
    assert result['phases_seconds'] == dict(BC=2., reference=3., finalize=4.)
    assert result['unaccounted_seconds'] == 0
    r = dict(shared_work_id='invocationA/case1', boundary_host_seconds=2.,
        cpu_reference_seconds=3., first_synchronized_seconds=4.,
        warm_seconds=[1., 1.], case_transfer_seconds=.5)
    total = record_totals([r, r.copy()])
    assert total['shared_evaluations'] == 1 and total['cpu_reference_seconds'] == 3
    assert total['warm_seconds'] == 4
    # A resumed half-pair genuinely repeats shared work: do not undercount it.
    again = dict(r, shared_work_id='invocationB/case1')
    assert record_totals([r, again])['shared_evaluations'] == 2
    with pytest.raises(ValueError, match='inconsistent'):
        record_totals([r, dict(r, cpu_reference_seconds=8.)])


def test_literal_reference_reuses_only_lowering(monkeypatch):
    import literal_reference as lr
    chunks = [SimpleNamespace(bank=i, geometry={}) for i in range(2)]
    calls = dict(lower=0, action=0)
    def lower(b, **kw):
        calls['lower'] += 1
        return b
    def action(plan, x, *args, **kw):
        calls['action'] += 1
        return (np.asarray([x[plan]]),)
    monkeypatch.setattr(lr, 'lower_q_plan', lower)
    monkeypatch.setattr(lr, 'apply_q_plan', action)
    api = SimpleNamespace(boundaries=lambda *a: (None, None, None), COEFF=None, TAU=1., MU=1836.)
    ref = lr.LiteralReference(chunks, 1/32, api)
    for k in range(4):
        result = ref(np.array([k, k+1.]), None, 1, ('D',)*6, 'D')
        np.testing.assert_array_equal(result[0], [k, k+1.])
    assert calls == dict(lower=2, action=8)
    assert ref.stats['lowerings'] == 2 and ref.stats['actions'] == 8
