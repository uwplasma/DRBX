"""Implementation/bookkeeping regression on a frozen actual-HSX excerpt.

This fixture is not a global accuracy or structural-certification gate.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from q03_direct_campaign import campaign as q
from q03_direct_campaign.common import SCHEMA, FIELDS, atomic_json, sha
from q03_direct_campaign.numerics import POLICY, OwnerIndex, owner_matrix, fit_batch


@pytest.fixture(scope='module')
def hsx():
    path = Path(__file__).parent/'data/q03_direct_hsx_excerpt.npz'
    assert sha(path) == json.loads(path.with_suffix('.json').read_text())['sha256']
    with np.load(path) as z:
        return {k:z[k].copy() for k in z.files}


@pytest.fixture
def prepared(hsx, tmp_path):
    path = tmp_path/'inputs'; path.mkdir()
    for k,v in hsx.items():
        if not k.startswith('expected.'):
            np.save(path/(k+'.npy'),v)
    atomic_json(path/'manifest.json', {
        'schema':SCHEMA,'resolution':32,'scope':'sample','fields':FIELDS,
        'identity':{'policy':POLICY}, 'reference':{'qualification_available':True},
        'files':{p.name:sha(p) for p in path.glob('*.npy')},
    })
    return path


def test_index_moments_and_action_match_raw_owner_prototype(hsx):
    index=OwnerIndex(hsx['owner.centers'],hsx['owner.planes'],32)
    faces=np.flatnonzero(~hsx['face.boundary'])
    selected=[index.select(hsx['face.center'][f],hsx['face.plane'][f],hsx['face.minus'][f],hsx['face.plus'][f]) for f in faces]
    donors=np.stack([v[0] for v in selected]);distance=np.stack([v[1] for v in selected])
    np.testing.assert_array_equal(donors,hsx['expected.donors'][faces])
    P=owner_matrix(donors,hsx['owner.centers'],hsx['owner.moments'],hsx['face.center'][faces],hsx['face.scale'][faces],32)
    co,diag=fit_batch(P,distance,hsx['target'][faces])
    np.testing.assert_allclose(co,hsx['expected.coefficients'][faces],rtol=2e-9,atol=2e-14)
    assert diag['cubic_defect'].max() < 1e-10
    # Independently test the weighted primal solve on spaced geometry cases.
    for j in np.linspace(0,len(faces)-1,8,dtype=int):
        w=(1+distance[j])**-1.5
        beta=np.linalg.lstsq(P[j]*w[:,None],hsx['owner.states'][donors[j]]*w[:,None],rcond=None)[0]
        np.testing.assert_allclose(co[j]@hsx['owner.states'][donors[j]],hsx['target'][faces[j]]@beta,rtol=1e-8,atol=2e-14)


def test_serial_parallel_complete_cell_replay_and_resume(prepared, hsx, tmp_path):
    serial=tmp_path/'serial'; parallel=tmp_path/'parallel'
    result=q.run(prepared,serial,chunk_size=128,save_coefficients=True)
    q.run(prepared,parallel,workers=2,chunk_size=128,save_coefficients=True)
    with np.load(serial/'actions.npz') as a, np.load(parallel/'actions.npz') as b:
        np.testing.assert_array_equal(a['direct'],b['direct'])
        np.testing.assert_allclose(a['direct'],hsx['expected.direct'],rtol=1e-8,atol=1e-10)
    before={p.name:p.stat().st_mtime_ns for p in (serial/'chunks').glob('*.npz')}
    q.run(prepared,serial,chunk_size=128,save_coefficients=True)
    assert before=={p.name:p.stat().st_mtime_ns for p in (serial/'chunks').glob('*.npz')}
    assert result['completed'] and not result['convergence_passed']
    assert not result['structural_properties_certified']
    with pytest.raises(RuntimeError,match='incompatible run'):
        q.run(prepared,serial,chunk_size=64,save_coefficients=True)
    path=next((serial/'chunks').glob('*.npz'))
    with path.open('ab') as stream: stream.write(b'corrupt')
    with pytest.raises(RuntimeError,match='corrupt chunk'):
        q.run(prepared,serial,chunk_size=128,save_coefficients=True)


def test_input_corruption_and_missing_chunk_rejected(prepared,tmp_path):
    output=tmp_path/'output'
    q.run(prepared,output,chunk_size=128)
    from q03_direct_campaign.common import load_inputs
    d,meta=load_inputs(prepared)
    ident=json.loads((output/'run.json').read_text())['identity']
    path=next((output/'chunks').glob('*.npz'))
    path.with_suffix('.json').unlink();path.unlink()
    with pytest.raises(RuntimeError,match='missing chunk'):
        q.assemble(d,meta,output,ident,128)
    with (prepared/'target.npy').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(RuntimeError,match='corrupt input'):
        q.run(prepared,tmp_path/'new')


def test_sample_orders_and_duplicate_run_rejected(prepared,tmp_path):
    output=tmp_path/'output'
    with q.exclusive(output/'campaign.lock'):
        with pytest.raises(RuntimeError,match='already running'):
            q.run(prepared,output)
    q.run(prepared,output)
    summary=json.loads((output/'summary.json').read_text())
    outputs=[]
    for N in (32,48,64):
        p=tmp_path/f'N{N}';p.mkdir();summary['resolution']=N
        atomic_json(p/'summary.json',summary);outputs.append(p)
    with pytest.raises(ValueError,match='only complete global'):
        q.orders(outputs,tmp_path/'orders.json')


def test_order_gate_requires_both_intervals_and_reference_budget(tmp_path):
    outputs=[]
    for N in (32,48,64):
        p=tmp_path/f'N{N}';p.mkdir();outputs.append(p)
        atomic_json(p/'summary.json',{'resolution':N,'completed':True,'scope':'global',
            'identity':{'source':{'test':'bookkeeping-only'},'policy':POLICY},
            'field_contract':{'test':'bookkeeping only'},'diffusivity':1.0,
            'metrics':{f:{'L2':N**-2,'reference_qualified':True} for f in FIELDS[:-1]}})
    assert q.orders(outputs,tmp_path/'orders.json')['global_operator_convergence_passed']
    p=outputs[1]/'summary.json';s=json.loads(p.read_text());s['metrics'][FIELDS[0]]['reference_qualified']=False;atomic_json(p,s)
    assert not q.orders(outputs,tmp_path/'orders.json')['global_operator_convergence_passed']
    s['metrics'][FIELDS[0]]['reference_qualified']=True;s['metrics'][FIELDS[0]]['L2']*=2;atomic_json(p,s)
    assert not q.orders(outputs,tmp_path/'orders.json')['global_operator_convergence_passed']


def test_remote_identity_rejects_changed_seeds_and_environment(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from q03_direct_campaign import remote
    seeds=tmp_path/'seeds';seeds.mkdir()
    here=tmp_path/'code';here.mkdir()
    metric=tmp_path/'metric';metric.write_bytes(b'bookkeeping metric')
    magnetic=tmp_path/'magnetic';magnetic.write_bytes(b'bookkeeping magnetic')
    data=seeds/'numeric';data.write_bytes(b'bookkeeping array')
    atomic_json(seeds/'manifest.json',{'schema':'q03-direct-seeds-v1','policy':POLICY,
        'external':{'metric_cache':{'sha256':sha(metric)},'makegrid':{'sha256':sha(magnetic)}},
        'files':{'numeric':sha(data)}})
    atomic_json(here/'seed_manifest.json',{'manifest_sha256':sha(seeds/'manifest.json')})
    atomic_json(here/'geometry_source_manifest.json',{})
    monkeypatch.setattr(remote,'HERE',here)
    a=SimpleNamespace(seeds=seeds,metric_cache=metric,makegrid=magnetic,output=tmp_path/'campaign')
    remote.verify(a);remote.verify(a)
    data.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='changed/missing seed'):remote.verify(a)
    data.write_bytes(b'bookkeeping array')
    monkeypatch.setattr(remote,'source_identity',lambda:{'changed':'implementation'})
    with pytest.raises(RuntimeError,match='incompatible campaign'):remote.verify(a)


def test_remote_validator_reassembles_but_cannot_certify_samples(prepared,tmp_path):
    from types import SimpleNamespace
    from q03_direct_campaign import remote
    import shutil
    output=tmp_path/'campaign';identity={'bookkeeping':'actual-HSX sample'}
    for N in (32,48,64):
        dest=output/'inputs'/f'N{N}'
        shutil.copytree(prepared,dest)
        meta=json.loads((dest/'manifest.json').read_text())
        # All three are the same N32 fixture, deliberately NOT global data.
        meta['identity']['campaign']=identity
        atomic_json(dest/'manifest.json',meta)
        q.run(dest,output/'results'/f'N{N}')
    args=SimpleNamespace(output=output)
    with pytest.raises(ValueError):remote.validate(args,identity)
    assert not (output/'validation.json').exists()
    with (output/'results/N32/actions.npz').open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(RuntimeError,match='changed action'):remote.validate(args,identity)
