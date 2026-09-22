"""P05 global-runner bookkeeping and actual-HSX bounded incidence replay."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'scripts/p05_material_campaign'
sys.path.insert(0,str(HERE))


def load_campaign():
    spec=importlib.util.spec_from_file_location('p05_campaign_test',HERE/'campaign.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def load_numeric(tmp_path):
    c=load_campaign()
    return c.runner._bootstrap(ROOT,ROOT,tmp_path)


@pytest.mark.parametrize('n',[48,64])
def test_real_hsx_complete_owner_incidence(n,tmp_path):
    c=load_campaign();numeric=load_numeric(tmp_path)
    f=c.arrays(HERE/f'preflight_N{n}.npz')
    out=np.zeros((3,int(f['raw_owner'].max())+1))
    c.kernel.scatter(numeric,n,f['raw_owner'],f['indices'],f['delta_flux'],out)
    replay=-out[:,f['owners']]/(float(f['rho_star'])*f['volume'][None,:])
    np.testing.assert_allclose(replay,f['action_delta'],atol=1e-13,rtol=1e-10)


def test_receipt_rejects_corruption_and_changed_identity(tmp_path):
    c=load_campaign();p=tmp_path/'one.npz';ids=np.array([1,4],dtype=np.int64)
    assert not c.validate_chunk(p,'id',ids)
    np.savez(p,indices=ids)
    import hashlib
    c.write(p.with_suffix('.json'),{'identity':'id','indices_sha256':hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),'sha256':c.sha(p)})
    assert c.validate_chunk(p,'id',ids)
    with pytest.raises(RuntimeError,match='incompatible'):c.validate_chunk(p,'new',ids)
    with p.open('ab') as f:f.write(b'corrupt')
    with pytest.raises(RuntimeError,match='corrupt'):c.validate_chunk(p,'id',ids)


def test_exact_global_face_partition(tmp_path):
    c=load_campaign()
    from types import SimpleNamespace
    a=SimpleNamespace(output=tmp_path)
    for n in (32,48,64):
        ids=np.concatenate([v for v,p in c.jobs(a,n)])
        np.testing.assert_array_equal(ids,np.arange(3*n**3+3*n**2))


def test_merge_requires_both_intervals_and_keeps_vorticity_diagnostic(tmp_path):
    # Scalar gate arithmetic only; these artificial errors are not geometry tests.
    c=load_campaign()
    from types import SimpleNamespace
    a=SimpleNamespace(output=tmp_path,resolutions=[32,48,64])
    for n in a.resolutions:
        np.savez(tmp_path/f'N{n}.npz',dummy=np.ones(1))
        c.write(tmp_path/f'N{n}.json',{'identity':'id','completed':True,
            'output_sha256':c.sha(tmp_path/f'N{n}.npz'),
            'statistics':{f:{'L2':1. if f=='actual_vorticity' else n**-2,'reference_qualified':True} for f in c.kernel.FIELDS}})
    result=c.merge(a,'id')
    assert result['material_operator_convergence_passed']
    assert not result['results']['actual_vorticity']['both_orders_ge_1_8']
    assert not result['positivity_dissipation_certified']
    p=tmp_path/'N48.json';case=json.loads(p.read_text());case['statistics'][c.kernel.FIELDS[1]]['L2']*=3;c.write(p,case)
    assert not c.merge(a,'id')['material_operator_convergence_passed']
    case['statistics'][c.kernel.FIELDS[1]]['L2']/=3
    case['statistics'][c.kernel.FIELDS[2]]['reference_qualified']=False;c.write(p,case)
    assert not c.merge(a,'id')['material_operator_convergence_passed']
