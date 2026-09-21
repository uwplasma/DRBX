from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import json
import pytest

HERE = Path(__file__).resolve().parents[1] / 'scripts/hsx_remote_qualification'


def load_runner():
    spec=importlib.util.spec_from_file_location('clean_remote_runner_test',HERE/'parallel_runner.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def test_plan_rejects_missing_and_overlapping_global_ranges():
    runner=load_runner();numeric=SimpleNamespace(_face_count=lambda n:3)
    plan={'schema':runner.PLAN_SCHEMA,'resolution':1,'coverage':'global','units':[
        {'id':'a','kind':'face','first':0,'last':3}, {'id':'b','kind':'cell','first':0,'last':1}]}
    runner._validate_plan(plan,numeric)
    plan['units'].append({'id':'c','kind':'face','first':1,'last':2})
    with pytest.raises(ValueError,match='duplicate face'):runner._validate_plan(plan,numeric)
    plan['units']=plan['units'][:1]
    with pytest.raises(ValueError,match='incomplete global cell'):runner._validate_plan(plan,numeric)


def test_clean_inputs_exclude_reconstruction_dependent_history():
    config=json.loads((HERE/'configuration.json').read_text())
    for forbidden in ('derivative_root','bounded_root','factor_root','cross_root'):
        assert forbidden not in config['inputs']
    assert config['candidate']['name'].endswith('selection_v3')
    paths=[x['path'] for x in json.loads((HERE/'input_manifest.json').read_text())['files']]
    assert not any('derivatives/' in p or 'cross_cache' in p or '.actions.npz' in p for p in paths)


def test_memory_budget_caps_workers_and_rejects_impossible_budget():
    runner = load_runner()
    assert runner._effective_worker_count(12, memory_budget_gib=10,
        worker_memory_gib=2, memory_reserve_gib=2) == 4
    assert runner._effective_worker_count(2, memory_budget_gib=10,
        worker_memory_gib=2, memory_reserve_gib=2) == 2
    with pytest.raises(ValueError, match='supplied together'):
        runner._effective_worker_count(12, memory_budget_gib=10,
            worker_memory_gib=None, memory_reserve_gib=2)
    with pytest.raises(ValueError, match='cannot accommodate'):
        runner._effective_worker_count(12, memory_budget_gib=2,
            worker_memory_gib=2, memory_reserve_gib=1)


def test_resume_validates_existing_chunks_before_dispatch(tmp_path, monkeypatch):
    runner = load_runner()
    units = [{'id': 'a', 'kind': 'cell', 'indices': [0]},
             {'id': 'b', 'kind': 'cell', 'indices': [1]}]
    settings = {'output_root': str(tmp_path), 'runtime_path': str(tmp_path/'config.json')}
    (tmp_path/'config.json').write_text('{}')
    existing = runner._chunk_path(tmp_path, 32, units[0])
    existing.parent.mkdir(parents=True)
    existing.touch()
    validated = []
    monkeypatch.setattr(runner, '_validated_checkpoint',
        lambda state, unit, path: validated.append(unit['id']))
    reused, pending = runner._partition_units_for_resume(
        settings, {'resolution': 32, 'units': units}, object())
    assert validated == ['a']
    assert [item['id'] for item in reused] == ['a']
    assert pending == [units[1]]


def test_new_schema_rejects_historical_plan():
    runner=load_runner()
    with pytest.raises(ValueError,match='unsupported work plan'):
        runner._validate_plan({'schema':'drbx.hsx-matched-cubic-work-plan-v2'},None)


def test_complete_and_failed_scientific_verdict_are_distinct(tmp_path,monkeypatch):
    # This tests only gate bookkeeping; it is not synthetic-geometry evidence.
    runner=load_runner();repo=HERE.parents[1]
    numeric=runner._bootstrap(repo,repo,tmp_path)
    for n,e in ((32,.2),(48,.2*(32/48)**2),(64,.05)):
        case={'statistics':{field:{'candidate':{f'matched_{a}':{'absolute_l2':e} for a in 'ABC'},
                                  'baseline':{'C':{'absolute_l2':e}}} for field in numeric.FIELDS},
              'verification':{'finite_complete_owner_coverage':True,'constant_field_action_max_abs':1e-3,
                              'centered_decomposition_max_abs':0}}
        (tmp_path/f'N{n}.json').write_text(json.dumps(case))
        (tmp_path/f'N{n}.preflight.json').write_text('{}')
        fields={'actual_vorticity':{'direct_q3_minus_ibp_q3_representative_rms':1e-5,
                 'quadrature_q4_minus_q3_rms':1e-5,'finite_difference_step_sensitivity_rms':1e-5}}
        fields.update({f:{'selected_order':3,'rules':{'3':{'reference_difference_rms':1e-5,
                         'high_reference_q7_minus_q5_rms':1e-5}}} for f in numeric.FIELDS[1:]})
        (tmp_path/f'N{n}.qualification.json').write_text(json.dumps({'fields':fields}))
    monkeypatch.setattr(numeric,'_config',lambda p:{'paths':{'output':str(tmp_path),'reference_root':str(tmp_path)},'candidate':{},'scope':{}})
    monkeypatch.setattr(numeric,'_source_identity',lambda c:{})
    result=numeric._merge(SimpleNamespace(config=tmp_path/'unused'))
    assert result['convergence_passed'] and not result['invariants_checked']
    d=json.loads((tmp_path/'N64.qualification.json').read_text())
    d['fields'][numeric.FIELDS[1]]['rules']['3']['reference_difference_rms']=.1
    (tmp_path/'N64.qualification.json').write_text(json.dumps(d))
    result=numeric._merge(SimpleNamespace(config=tmp_path/'unused'))
    assert result['status']=='computation completed' and not result['convergence_passed']
