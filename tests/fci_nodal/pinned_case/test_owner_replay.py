"""Pinned one- and two-owner replay against the current case."""

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from tests.fci_nodal.support.cold_case import (
    cold_options, current_case_directory, mms_profile, stage_products, verified_adoptions,
)
from tests.fci_nodal.support.identity import array_difference, compare_array_files, owner_rows, require_bytes
from tests.fci_nodal.support.owner_size_bounds import assert_recorded_bounds

pytestmark = pytest.mark.pinned_case


@pytest.fixture(scope='module')
def owner_replays(tmp_path_factory):
    from drbx.fci_nodal.case import case as driver
    from drbx.fci_nodal.case._receipts import receipt_products, verify_chain

    pinned = current_case_directory()
    scratch = tmp_path_factory.mktemp('owner_replays')
    execute = driver._execute
    adoptions = verified_adoptions(pinned)

    def observe_worker(name, options, out, inputs, settings):
        if name != 'mms_run':
            return execute(name, options, out, inputs, settings)
        reply = out/'worker_reply.pkl'
        env = dict(os.environ, **settings)
        env.pop('DRBX_HOST_DEVICE_COUNT', None)
        subprocess.run(
            [sys.executable, '-B', '-c',
             'from tests.fci_nodal.support.owner_replay import observe_case_worker; observe_case_worker()',
             name, str(out), str(reply)],
            input=pickle.dumps((options, inputs)), env=env, check=True, timeout=120)
        with reply.open('rb') as stream:
            return pickle.load(stream)

    runs = {}
    for parts in (1, 2):
        root = scratch/f'case_{parts}'
        opts = cold_options(pinned, owners=parts)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(driver, '_execute', observe_worker)
            driver.build_case(opts, root, adopt=adoptions)
            profile = mms_profile(opts,pinned)
            result = driver.run_mms(root,profile)
            receipts = {root/name/'receipt.json': (root/name/'receipt.json').read_bytes()
                        for name in ('owner_inputs','wall_data','mms_prepare','mms_run')}
            cached_run = driver.run_mms(root,profile) == result
        preserved_receipts = all(p.read_bytes() == prior for p, prior in receipts.items())
        chain = verify_chain(root, 'mms_run')
        output = receipt_products(chain['mms_run'][0])
        arrays = owner_rows(output['owner_state'])
        with np.load(output['summary'].parent/'observations.npz', allow_pickle=False) as observed:
            endpoint_observation = array_difference(observed['step_3'],arrays['step_3'])
            arrays.update({key: observed[key] for key in ('psi', 'A', 'j', 'step_1', 'step_2')})
        bound = result['positivity_bound']
        arrays.update(dt=result['dt'], positivity=np.r_[bound['low_diagonal_over_Hp_max'],
                                                        bound['bounds_n_Te_Ti'], bound['minimum']])
        runs[parts] = dict(arrays=arrays,result=result,chain=chain,root=root,
                           endpoint_observation=endpoint_observation,
                           cached_run=cached_run,preserved_receipts=preserved_receipts)
    return pinned, scratch, runs


def compare_runtime(actual, reference):
    return {key: array_difference(actual[key], expected) for key, expected in reference.items()}


def test_coupled_rhs_matches_owners_and_reference(owner_replays):
    pinned, scratch, runs = owner_replays
    reference = owner_rows(stage_products(pinned,'mms_run')['owner_state'])
    one, two = (runs[parts]['arrays'] for parts in (1,2))
    differences = {'one_vs_reference': compare_runtime(one,reference),
                   'two_vs_one': compare_runtime(two,one)}
    (scratch/'coupled_differences.json').write_text(json.dumps(differences,indent=2)+'\n')
    require_bytes(differences['one_vs_reference'])
    assert_recorded_bounds(two,one)


def test_owner_case_endpoint_matches_reference(owner_replays):
    pinned, scratch, runs = owner_replays
    reference = stage_products(pinned,'mms_run')
    summary = json.loads(reference['summary'].read_text())
    one = runs[1]
    differences = {'P':{}, 'MMS':{}}
    for group, stage, role in (
        ('P','perpendicular_operators','owner_perpendicular_bundle'),
        ('MMS','mms_run','owner_state'),
    ):
        actual, expected = (stage_products(root,stage)[role] for root in (one['root'],pinned))
        def record(name, left, right):
            differences[group][name] = array_difference(left,right)
        compare_array_files(actual,expected,record)
    for parts, run in runs.items():
        differences['MMS'][f'observed endpoint/{parts}'] = run['endpoint_observation']
    actual_summary = stage_products(one['root'],'mms_run')['summary']
    differences['MMS']['summary'] = array_difference(actual_summary.read_bytes(),reference['summary'].read_bytes())
    (scratch/'owner_replay_differences.json').write_text(json.dumps(differences,indent=2)+'\n')
    for difference in differences.values():
        require_bytes(difference)

    for parts, run in runs.items():
        result, stages = run['result'], run['chain']
        assert run['cached_run'] and run['preserved_receipts']
        assert set(stages) == {'owner_inputs', 'wall_data', 'mms_prepare', 'mms_run'}
        assert sum(receipt['rays_traced'] for receipt, _ in stages.values()) == 0
        assert tuple(result['edges']) == tuple(np.arange(parts+1)*(16//parts))
        for key in ('steps', 'time', 'dt', 'positivity_bound'):
            assert result[key] == summary[key], (parts,key)
