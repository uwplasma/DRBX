"""Portable catalogue/BC replay and every-leaf component gates."""
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
import os
import subprocess
import sys
import numpy as np
import pytest
from scripts.q08_extraction_global import common as c


def test_catalogue_uses_accepted_sixth_field():
    points=np.array([[.2,.3,.7],[1.,.9,1.2]])
    five,grad=c.fields(points);six,gsix=c.six_fields(points)
    assert c.NF==22 and c.KINDS[2][0]==('D','N','D','N','D','N')
    np.testing.assert_array_equal(six[...,:5],five)
    np.testing.assert_array_equal(six[...,5],c.WOFF+five@c.WC)
    np.testing.assert_array_equal(gsix[...,5,:],np.einsum('...fa,f->...a',grad,c.WC))
    np.testing.assert_allclose(six[:,0,5],.2,rtol=0,atol=1e-15)
    assert not np.array_equal(six[:,1:,5],.2+five[:,1:,1]-1.1)


def test_common_import_does_not_force_gpu_to_cpu():
    env=dict(os.environ,JAX_PLATFORMS='gpu',CUDA_VISIBLE_DEVICES='7')
    code="from scripts.q08_extraction_global import common; import os; assert os.environ['JAX_PLATFORMS']=='gpu'; assert os.environ['CUDA_VISIBLE_DEVICES']=='7'"
    subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=env,check=True)


@pytest.mark.parametrize('n',[32,48,64])
def test_bank_query_boundaries_match_accepted_c3_saved_arrays(n):
    from drbx.stencils.q_artifact import load_q_bank
    directory=Path(__file__).resolve().parents[2]/'work/q08_implementation_20261002'
    if not (directory/f'bank_N{n}.npz').exists():pytest.skip('local C3 bank fixture unavailable')
    bank=load_q_bank(directory/f'bank_N{n}.npz')
    # Merged GPU descriptors require only the array attributes, no schema.
    trusted=SimpleNamespace(**bank.arrays)
    result=c.boundaries(trusted)
    with np.load(directory/f'c3_fixtures/N{n}_inputs.npz') as z:
        for label,bc in zip(('inner','outer','phi'),result):
            for i,a in enumerate(bc):
                expected=z[f'{label}_bc_{i}']
                assert a.shape==expected.shape and a.dtype==expected.dtype
                assert a.tobytes()==expected.tobytes(),(label,i)
    for case in range(c.NF):
        for full,selected in zip(result,c.boundaries(trusted,case)):
            for a,b in zip(full,selected):np.testing.assert_array_equal(b,a[case])


class Inner(NamedTuple):
    current: object
    inputs_valid: object
class Outputs(NamedTuple):
    centered: object
    combined: object
    raw_current: object


def test_all_leaf_gate_detects_constituents_even_if_sum_unchanged():
    expected=Outputs(np.array([1.]),np.array([3.]),Inner(np.array([2.]),np.array([True])))
    actual=Outputs(np.array([1.+1e-9]),np.array([3.]),Inner(np.array([2.-1e-9]),np.array([True])))
    metrics=c.check_outputs(actual,expected)
    assert metrics['max_scaled_error']<1 and len(metrics['leaves'])==4
    wrong=actual._replace(raw_current=Inner(np.array([2.1]),np.array([True])))
    with pytest.raises(ValueError,match='raw_current.current'):c.check_outputs(wrong,expected)
    with pytest.raises(ValueError,match='boolean'):
        c.check_outputs(expected._replace(raw_current=Inner(np.array([2.]),np.array([False]))),expected)
    with pytest.raises(ValueError,match='nonfinite'):
        c.check_outputs(expected._replace(centered=np.array([np.nan])),expected)


def test_singlecase_boundary_allocation_is_bounded_before_selection(monkeypatch):
    original=np.zeros;shapes=[]
    bank=SimpleNamespace(raw=np.arange(100),wall_index=np.zeros(0,dtype=int))
    def checked(shape,*args,**kwargs):
        shapes.append(shape)
        if len(shape)>=3:assert shape[0]==1
        return original(shape,*args,**kwargs)
    monkeypatch.setattr(c.np,'zeros',checked)
    result=c.boundaries(bank,17)
    assert result[0][0].shape==(6,100,35) and shapes
    with pytest.raises(ValueError,match='catalogue case'):c.boundaries(bank,22)


def test_atomic_writes_and_same_stat_content_hash(tmp_path):
    path=tmp_path/'data.json';c.atomic_json(path,dict(a=1));stamp=path.stat();first=c.sha(path)
    c.atomic_json(path,dict(a=2));os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    assert c.sha(path)!=first and path.stat().st_size==stamp.st_size
    c.atomic_npz(tmp_path/'data.npz',value=np.arange(5))
    with np.load(tmp_path/'data.npz') as z:np.testing.assert_array_equal(z['value'],np.arange(5))
    assert not list(tmp_path.glob('*.tmp'))
