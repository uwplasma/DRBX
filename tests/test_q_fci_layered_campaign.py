"""Bookkeeping safeguards for the research-only layered Q campaign."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from scripts.q_fci_layered_global import storage, campaign
from scripts.q_fci_layered_global.fields import FIELDS,field


def test_complete_owner_chunking():
    t=SimpleNamespace(starts=np.array([0,64,65,67,127,128]))
    chunks=campaign.groups(t,range(5),64)
    assert chunks==[[0],[1,2,3,4]]
    assert [o for z in chunks for o in z]==list(range(5))
    with pytest.raises(ValueError):campaign.groups(t,range(5),32)


def test_checkpoints_detect_corruption_and_identity(tmp_path):
    path=tmp_path/'chunk.npz'
    storage.arrays(path,owners=np.array([3,7]),N=np.array([1+2j]))
    assert not storage.complete(path,'identity')
    storage.record(path,'identity',backend='cpu')
    assert storage.complete(path,'identity')
    with pytest.raises(RuntimeError):storage.complete(path,'other')
    with path.open('ab') as out:out.write(b'corruption')
    with pytest.raises(RuntimeError):storage.complete(path,'identity')


def test_memory_budget_limits_actual_workers(monkeypatch):
    monkeypatch.setattr(campaign.os,'sched_getaffinity',lambda _:set(range(60)),raising=False)
    a=SimpleNamespace(workers=100,memory_gib=11,per_worker_gib=2.5)
    assert campaign.workers(a)==4
    a.memory_gib=1
    with pytest.raises(ValueError):campaign.workers(a)


def test_manufactured_gradients_and_zero_neumann_control():
    points=np.array([[.3,1.13,2.47],[.82,4.2,6.21]])
    for name in FIELDS:
        value,grad=field(name,points)
        for axis in range(3):
            plus=points.copy();minus=points.copy();plus[:,axis]+=1e-6;minus[:,axis]-=1e-6
            fd=(field(name,plus)[0]-field(name,minus)[0])/2e-6
            np.testing.assert_allclose(fd,grad[:,axis],atol=3e-9,rtol=3e-8)
    wall=points.copy();wall[:,0]=1
    np.testing.assert_array_equal(field('simple_zero_N',wall)[1],0)


def test_reduction_preserves_bc_span_and_nor_axes(tmp_path):
    from scripts.q_fci_layered_global.fields import FIELDS
    plans={}
    for n in campaign.NS:
        plans[str(n)]={'global_':[list(range(6))],'owner_count':6,'last_aggregate':1}
    storage.write(tmp_path/'plan.json',plans)
    storage.write(tmp_path/'design.json',{'sources':storage.sources(),'plan_sha256':storage.sha(tmp_path/'plan.json')})
    ident=storage.sha(tmp_path/'design.json')
    for n in campaign.NS:
        o=np.zeros((6,2,len(FIELDS)),complex);r=np.zeros((6,len(FIELDS)),complex)
        num=np.zeros((6,2,2,len(FIELDS)),complex)
        for bc in range(2):
            for span in range(2):num[:,bc,span,:]=(bc+1)*(span+1)/n**2
        path=tmp_path/'global'/f'N{n}'/'score_000000.npz'
        storage.arrays(path,owners=np.arange(6),N=num,O=o,R=r,R_half=r,radial=np.array([0,1,2,n-3,n-2,n-1]),volume=np.arange(1,7))
        storage.record(path,ident)
        storage.write(tmp_path/'global'/f'validate_N{n}.json',{'identity':ident,'operational_pass':True})
    campaign.reduce(SimpleNamespace(campaign=tmp_path))
    summary=storage.read(tmp_path/'summary.json')
    row=next(x for x in summary['norms'] if x['Ngrid']==64 and x['BC']=='N' and x['alpha']==1/32 and x['region']=='global' and x['field']=='common')
    assert row['NO']==pytest.approx(4/64**2)
    assert row['NR']==row['NO'] and row['OR']==0
    order=next(x for x in summary['orders'] if x['BC']=='N' and x['alpha']==1/32 and x['region']=='global' and x['field']=='common' and x['channel']=='NO')
    assert order['orders']==pytest.approx([2,2])
    assert storage.read(tmp_path/'completion.json')['complete']
