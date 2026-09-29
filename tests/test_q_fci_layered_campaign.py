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
    maximum_order=next(x for x in summary['orders'] if x['BC']=='N' and x['alpha']==1/32 and x['region']=='inner_join' and x['field']=='common' and x['channel']=='NO_max')
    assert maximum_order['orders']==pytest.approx([2,2])
    with np.load(tmp_path/'results_N64.npz') as result:
        assert result['fields'].tolist()==list(FIELDS)
    assert row['NO_max_owner'] in range(6)
    assert storage.read(tmp_path/'completion.json')['complete']


def test_extended_catalogue_keeps_baseline_axes():
    from scripts.q_fci_layered_global.fields import BASELINE_FIELDS,FRESH_FIELDS
    assert len(BASELINE_FIELDS)==18 and len(FRESH_FIELDS)==8
    assert FIELDS[:18]==BASELINE_FIELDS
    assert len(set(FIELDS))==26


def test_balanced_exhaustion_never_enlarges_runtime_stencil(monkeypatch):
    """Failed 28-owner candidates must stop, not inherit full-pool fallback."""
    from scripts.q_fci_layered_global import support
    # Use canonical HSX when available; missing research inputs are not CI failures.
    from pathlib import Path
    from scripts.q_fci_layered_global import model
    root=Path(__file__).resolve().parents[2]
    config=json.loads((Path(model.__file__).parent/'inputs.json').read_text())
    if not (root/config['geometry']/'32x32x32/base_geometry.npz').exists():
        pytest.skip('canonical HSX research inputs unavailable')
    _,t=model.context(32,root,physical=False,magnetic=False)
    h=support.BalancedHybrid(t)
    monkeypatch.setattr(support,'good',lambda _:False)
    ijk=np.array([8,24,23]);p=t.pts[(8*32+24)*32+23][None]
    with pytest.raises(RuntimeError,match='no passing balanced28 support'):
        h.rows(ijk,p)


def test_balanced_rows_reproduce_moments_and_keep_outer_identical():
    from pathlib import Path
    from scripts.q_fci_layered_global import model,primitives as r
    from scripts.q_fci_layered_global.support import Hybrid,BalancedHybrid
    root=Path(__file__).resolve().parents[2]
    config=json.loads((Path(model.__file__).parent/'inputs.json').read_text())
    if not (root/config['geometry']/'48x48x48/base_geometry.npz').exists():
        pytest.skip('canonical HSX research inputs unavailable')
    _,t=model.context(48,root,physical=False,magnetic=False)
    old=Hybrid(t);new=BalancedHybrid(t)
    ijk=np.array([12,36,35]);center=t.pts[(12*48+36)*48+35]
    points=center+np.array([[.12/48,.15*t.g.dtheta,t.g.deta/32],[-.08/48,-.2*t.g.dtheta,-t.g.deta/32]])
    ids,V,D,meta=new.rows(ijk,points)
    assert all(len(p['donors'])==28 and p['rank']==15 for p in meta['planes'])
    # Independent owner observations for all transverse quartic monomials.
    xy=center[0]*np.array([np.cos(center[1]),np.sin(center[1])]);scale=max(t.g.dr,center[0]*t.g.dtheta)
    obs=np.array([(t.rv[r.members(t,o)]/t.vol[o])@r.basis(t.xy[r.members(t,o)],xy,scale,r.EXP4) for o in ids])
    B,du,dt=r.planar(points,xy,scale,r.EXP4)
    np.testing.assert_allclose(V@obs,B,atol=2e-10,rtol=2e-10)
    np.testing.assert_allclose(np.einsum('sid,df->sif',D,obs),np.stack((du,dt,np.zeros_like(du)),axis=1),atol=2e-9,rtol=2e-9)
    ijk[0]=new.last+1
    center=t.pts[(ijk[0]*48+ijk[1])*48+ijk[2]]
    for a,b in zip(old.rows(ijk,center[None])[:3],new.rows(ijk,center[None])[:3]):
        np.testing.assert_array_equal(a,b)
