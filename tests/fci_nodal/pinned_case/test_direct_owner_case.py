"""Pinned owner builds and ranked replay through the public case driver."""

from dataclasses import replace
import json
from pathlib import Path
import os
import pickle
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.fci_nodal.support.cold_case import (
    current_case_directory, cold_options, mms_profile, read_stage, stage_products, verified_adoptions,
)
from tests.fci_nodal.support.identity import array_difference, compare_array_files, require_bytes
from tests.fci_nodal.support.owner_io_audit import LOG_ENV, STAGE_ENV, file_observations, install_open_audit

pytestmark = pytest.mark.pinned_case


def recorded_outputs(case, observations=None):
    doc = read_stage(case,'mms_run')
    folders = sorted(Path(doc['products']['owner_state']).glob('owner*'))
    ids = np.concatenate([np.load(p/'node_ids.npy',allow_pickle=False) for p in folders])
    order = np.argsort(ids)
    np.testing.assert_array_equal(ids[order],np.arange(len(ids)))
    rows = {'step_3':[],'rhs':[],'rhs_manufactured':[]}
    if observations is not None:
        rows.update(step_1=[],step_2=[])
    for folder in folders:
        state = np.load(folder/'nodal_state.npy',allow_pickle=False)
        rows['step_3'].append(state)
        with np.load(folder/'static_rhs.npz',allow_pickle=False) as archive:
            for name in ('rhs','rhs_manufactured'):
                rows[name].append(archive[name])
        if observations is not None:
            local = observations/folder.name
            for name in ('step_1','step_2'):
                rows[name].append(np.load(local/f'{name}.npy',allow_pickle=False))
            rows.setdefault('observed_step_3',[]).append(np.load(local/'step_3.npy',allow_pickle=False))
    return {name:np.concatenate(parts)[order] for name,parts in rows.items()}


def build_audited_replay(tmp_path_factory, owners):
    from drbx.fci_nodal.case import case as driver

    pinned = current_case_directory()
    tmp_path = tmp_path_factory.mktemp("direct_owner_case")
    options = cold_options(pinned, owners=owners)
    options = replace(options, parallel=replace(options.parallel, workers=1))
    profile = mms_profile(options,pinned)
    case = tmp_path/'case'
    log = tmp_path/'file_events'
    log.mkdir()
    observations = tmp_path/'observations'
    observations.mkdir()
    bootstrap = tmp_path/'audit_bootstrap'
    bootstrap.mkdir()
    (bootstrap/'sitecustomize.py').write_text(
        'from tests.fci_nodal.support.owner_io_audit import install_open_audit\ninstall_open_audit()\n')
    adoptions = verified_adoptions(pinned)
    execute = driver._execute
    stage = driver.stage

    def with_stage(case_dir,name,**kwargs):
        with pytest.MonkeyPatch.context() as active:
            active.setenv(STAGE_ENV,name)
            return stage(case_dir,name,**kwargs)

    def execute_owner(name,opts,out,inputs,settings):
        assert name not in adoptions
        if name!='mms_run':
            return execute(name,opts,out,inputs,settings)
        env = dict(os.environ,**settings)
        env.pop('DRBX_HOST_DEVICE_COUNT',None)
        reply = out/'worker_reply.pkl'
        subprocess.run([sys.executable,'-B','-c',
                        'from tests.fci_nodal.support.owner_io_audit import observe_owner_worker; observe_owner_worker()',
                        name,str(out),str(reply)],input=pickle.dumps((opts,inputs)),env=env,check=True,timeout=120)
        with reply.open('rb') as stream:
            return pickle.load(stream)

    with pytest.MonkeyPatch.context() as active:
        active.setenv(LOG_ENV,str(log))
        active.setenv('DRBX_TEST_OWNER_SNAPSHOTS',str(observations))
        active.setenv('PYTHONPATH',os.pathsep.join((str(bootstrap),os.environ.get('PYTHONPATH',''))))
        install_open_audit()
        active.setattr(driver,'stage',with_stage)
        active.setattr(driver,'_execute',execute_owner)
        driver.build_case(options,case,adopt=adoptions)
        active.setenv(STAGE_ENV,'owner_receipt_validation')
        result = driver.run_mms(case,profile)
    return SimpleNamespace(pinned=pinned,options=options,case=case,log=log,observations=observations,
                           result=result,directory=tmp_path)


@pytest.fixture(scope='module')
def direct_owner_replay(tmp_path_factory):
    return build_audited_replay(tmp_path_factory, owners=2)


@pytest.fixture(scope='module')
def direct_reference_replay(tmp_path_factory):
    return build_audited_replay(tmp_path_factory, owners=1)


def test_direct_owner_case_matches_reference(direct_reference_replay):
    from drbx.fci_nodal.operator_build.perp.owner_bundle import OwnerPerpendicularBundle

    replay = direct_reference_replay
    pinned,case,result = replay.pinned,replay.case,replay.result
    log,observations = replay.log,replay.observations
    tmp_path = replay.directory
    audit = file_observations(log,case,pinned)
    (tmp_path/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')

    differences = {'P':{}, 'MMS':{}}
    for group, stage, role in (
        ('P','perpendicular_operators','owner_perpendicular_bundle'),
        ('MMS','mms_run','owner_state'),
    ):
        actual, expected = (stage_products(root,stage)[role] for root in (case,pinned))
        def record(name, left, right):
            differences[group][name] = array_difference(left,right)
        compare_array_files(actual,expected,record)

    arrays = recorded_outputs(case,observations)
    np.savez(tmp_path/'compared.npz',**arrays)
    differences['MMS']['observed endpoint'] = array_difference(arrays['observed_step_3'],arrays['step_3'])
    summaries = [stage_products(root,'mms_run')['summary'].read_bytes() for root in (case,pinned)]
    differences['MMS']['summary'] = array_difference(*summaries)
    (tmp_path/'owner_array_differences.json').write_text(json.dumps(differences,indent=2)+'\n')
    for difference in differences.values():
        require_bytes(difference)

    assert not audit['forbidden'], audit['forbidden']
    for name in ('parallel_operators','perpendicular_operators','owner_inputs','wall_data','mms_prepare','mms_run'):
        local = str(case/name)
        assert any(path.startswith(local+'/') for path in audit['activity'][name]['write'])
        assert read_stage(case,name)['rays_traced']==0

    p = stage_products(case,'perpendicular_operators')['owner_perpendicular_bundle']
    reader = OwnerPerpendicularBundle(p)
    assert reader.edges == (0,16)
    assert reader.manifest['source_identity'] == 'identity.json'
    assert reader.manifest['sbp_leaves']['aee_h']['halo'] == 0
    reader.read_owner_sbp(0, edges=reader.edges)
    q = stage_products(case,'parallel_operators')['parallel_construction']
    manifest = json.loads((q/'owners/manifest.json').read_text())
    export = stage_products(case,'owner_inputs')['owner_parallel']
    metadata = json.loads((export/'metadata.json').read_text())
    assert manifest['moment_reduced_face_quadrature']['sha256']==metadata['recipe_sha256']
    assert metadata['structure']['owned_node_counts']==[4064]

    assert result['steps']==3 and tuple(result['edges'])==(0,16)
    for stage in ('step_1','step_2'):
        assert arrays[stage].shape==arrays['step_3'].shape
        assert np.isfinite(arrays[stage]).all()


@pytest.mark.parametrize('launch', ('per_owner','scheduler'))
def test_two_rank_run_matches_local(request, launch):
    current_case_directory()
    try:
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
            listener.bind(('127.0.0.1',0))
            listener.listen(1)
            with socket.create_connection(listener.getsockname(),timeout=1):
                pass
    except OSError as error:
        pytest.skip(f'Loopback TCP unavailable for the two-process owner test: {error}')

    from drbx.fci_nodal.case import run_mms

    replay = request.getfixturevalue('direct_owner_replay')
    baseline = Path(replay.result['owner_state'])
    options = dict(mms_profile(replay.options,replay.pinned),launch_mode=launch,run_timeout_seconds=120)
    if launch == 'scheduler':
        command = Path(__file__).resolve().parents[1]/'support/local_rank_command.py'
        index = 'DRBX_TEST_SCHEDULER_INDEX'
        options.update(rank_command=[sys.executable,'-B',str(command),'2',index],
                       rank_index_env=index,
                       collectives_interface='lo0' if sys.platform=='darwin' else 'lo')
    result = run_mms(replay.case,options)
    distributed = Path(result['owner_state'])
    assert result['steps']==replay.result['steps']==3
    assert result['edges']==replay.result['edges']==[0,8,16]
    for key in ('time','dt','positivity_bound'):
        assert result[key]==replay.result[key]

    def counts(infos):
        def iterations(info):
            return {key:value['iterations'] for key,value in info.items()}
        return dict(static=iterations(infos['static']),static_forced=iterations(infos['static_forced']),
                    stages=[[iterations(info) for info in stage] for stage in infos['stages']])

    assert counts(result['solves'])==counts(replay.result['solves'])
    for owner in ('owner0000','owner0001'):
        for name in ('node_ids.npy','nodal_state.npy'):
            a,b = (np.load(root/owner/name,allow_pickle=False) for root in (distributed,baseline))
            assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(), (owner,name)
        with (np.load(distributed/owner/'static_rhs.npz',allow_pickle=False) as ranked,
              np.load(baseline/owner/'static_rhs.npz',allow_pickle=False) as local):
            assert set(ranked.files)==set(local.files)=={'rhs','rhs_manufactured'}
            for name in ranked.files:
                a,b = ranked[name],local[name]
                assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(), (owner,name)
