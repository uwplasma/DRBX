"""Full evolved orchestration contracts, using explicit algebra controls."""
from types import SimpleNamespace
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.q09_evolved_global import campaign as c
from scripts.q09_evolved_global.reduce import orders_from_records
from scripts.q09_evolved_mms.cli import _run
from scripts.q09_evolved_mms.evolution import stage_stepper, step_count
from scripts.q09_evolved_mms.provider import load_checkpoint


@pytest.mark.parametrize('n', c.GRIDS)
def test_frozen_time_grid_and_counts(n):
    dt = c.END/c.COARSE_STEPS[n]
    assert [step_count(0, c.END, dt/2**j) for j in range(c.LEVELS)] == [c.COARSE_STEPS[n]*2**j for j in range(c.LEVELS)]
    assert all(step_count(0,c.END,dt/2**j)%5 == 0 for j in range(c.LEVELS))
    assert c.configuration()['total_steps'] == 5800


def setup_case(tmp_path):
    initial = np.ones((6,2)); vol = np.array([.2,.8]); target = lambda t: initial*np.exp(-t)
    provider = SimpleNamespace(manufactured=SimpleNamespace(state=target), volume=vol,
                               regions={'first': np.array([True,False])})
    args = SimpleNamespace(output=tmp_path,dt=c.END/100,end=c.END,mode='diffusion',
        kinds='DDDDDD',phi_kind='D',resume=True,checkpoint_every=25,snapshots=True,levels=c.LEVELS)
    stepper = stage_stepper(lambda x,t:(-x,True), lambda t:jnp.zeros_like(initial),vol)
    return args,provider,stepper


def test_sparse_checkpoint_interruption_snapshots_and_resume(tmp_path,monkeypatch):
    from scripts.q09_evolved_mms import cli
    args, provider, stepper = setup_case(tmp_path)
    original = cli.save_checkpoint; writes=[]
    def interrupted(path, signature, payload):
        original(path,signature,payload); writes.append((path.name,payload['accepted_steps']))
        if path.name=='level0.npz' and payload['accepted_steps']==40:
            raise RuntimeError('injected interruption after committed milestone')
    monkeypatch.setattr(cli,'save_checkpoint',interrupted)
    with pytest.raises(RuntimeError,match='injected'):
        _run(args,provider,None,'control',stepper)
    assert (tmp_path/'snapshots/level0_part2.npz').is_file()
    monkeypatch.setattr(cli,'save_checkpoint',original)
    _run(args,provider,None,'control',stepper)
    obs = dict(initial=provider.manufactured.state(0),target=provider.manufactured.state(c.END),
               volume=provider.volume,**{'region:first':provider.regions['first']})
    c.pilot.validate_case(tmp_path,'control',obs,'diffusion','DDDDDD','D',dt0=args.dt,end=args.end,levels=c.LEVELS)
    sig = c.pilot.signature('control','diffusion','DDDDDD','D',args.dt,end=args.end)
    assert load_checkpoint(tmp_path/'level0.npz',sig)['accepted_steps']==100
    assert len(list((tmp_path/'snapshots').glob('*.npz')))==5
    assert {step for name,step in writes if name=='level0.npz'}=={20,25,40}
    fresh=tmp_path/'fresh';fresh.mkdir();newargs=SimpleNamespace(**vars(args));newargs.output=fresh
    _run(newargs,provider,None,'control',stepper)
    for level in range(c.LEVELS):
        sig=c.pilot.signature('control','diffusion','DDDDDD','D',args.dt/2**level,end=args.end)
        np.testing.assert_array_equal(load_checkpoint(tmp_path/f'level{level}.npz',sig)['state'],
                                      load_checkpoint(fresh/f'level{level}.npz',sig)['state'])
    assert json.loads((tmp_path/'report.json').read_text())==json.loads((fresh/'report.json').read_text())


def test_spatial_reduction_nonuniform_ratios_and_coverage():
    records=[dict(level=c.LEVELS-1,case='control',field='n',region='global',part=5,n=n,rms=3/n**2) for n in c.GRIDS]
    orders=orders_from_records(records)
    np.testing.assert_allclose([r['order'] for r in orders],2.,atol=1e-14)
    with pytest.raises(ValueError,match='coverage'): orders_from_records(records[:-1])
    with pytest.raises(ValueError,match='duplicate'): orders_from_records(records+[records[0]])


def test_snapshot_validation_rejects_changed_time(tmp_path,monkeypatch):
    from scripts.q09_evolved_global import reduce as r
    args, provider, stepper=setup_case(tmp_path/'cases/control');args.output.mkdir(parents=True)
    _run(args,provider,None,'control',stepper)
    monkeypatch.setattr(r,'MODES',('diffusion',));monkeypatch.setattr(r.pilot,'KINDS',(('DDDDDD','D'),))
    monkeypatch.setattr(r.pilot,'case_key',lambda *a:'control')
    c.pilot.write(tmp_path/'observations.json',dict(provider='control'))
    np.savez(tmp_path/'observations.npz',initial=provider.manufactured.state(0),volume=provider.volume,
        **{f'target:part{i}':provider.manufactured.state(c.END*i/5) for i in range(1,6)})
    valid=r.snapshot_records(tmp_path,32)
    assert len(valid)==c.LEVELS*5*6
    cp=tmp_path/'cases/control/snapshots/level0_part1.npz'
    sig=c.pilot.signature('control','diffusion','DDDDDD','D',args.dt,end=args.end)
    payload=load_checkpoint(cp,sig);payload['time']*=2
    from scripts.q09_evolved_mms.provider import save_checkpoint
    save_checkpoint(cp,sig,payload)
    with pytest.raises(ValueError,match='snapshot time'):r.snapshot_records(tmp_path,32)


def test_runtime_is_accepted_pilot_and_live_sources_are_excluded():
    c.runtime_matches_pilot()
    closure=c.files()
    assert 'src/drbx/native/q_parallel_rhs.py' not in closure
    assert 'scripts/q09_evolved_mms/runtime/drbx/native/q_parallel_rhs.py' in closure
    assert 'scripts/q09_evolved_global/run_all.py' in closure


def test_complete_reducer_roundoff_and_full_tables(tmp_path):
    """End-to-end tables from labeled synthetic reductions, never HSX evidence."""
    from scripts.q09_evolved_global.reduce import reduce_all, FIELDS
    for n in c.GRIDS:
        grid=tmp_path/f'N{n}';grid.mkdir()
        records=[];cases={}
        for mode in c.MODES:
            for kinds,phi in c.pilot.KINDS:
                key=c.pilot.case_key(mode,kinds,phi)
                errors=[1/n**2]*6
                cases[key]=dict(runs=[dict(error={'global':dict(rms=errors)})]*c.LEVELS,
                    temporal_self_convergence=dict(self_difference_rms=[[1e-17]*6],observed_orders=[-1.]*6))
                for field in FIELDS:
                    for part in range(1,6):
                        records.append(dict(n=n,case=key,field=field,part=part,level=c.LEVELS-1,region='global',rms=1/n**2))
        c.pilot.write(grid/'summary.json',dict(cases=cases))
        c.pilot.write(grid/'time_history.json',dict(records=records))
        np.savez(grid/'observations.npz',volume=np.ones(2),target=np.ones((6,2)))
    reduce_all(tmp_path,'algebra_control')
    result=c.pilot.read(tmp_path/'analysis.json')
    assert len(result['spatial_orders'])==8*6*5*3
    assert result['temporal']['measured'] is False
    assert len((tmp_path/'orders.csv').read_text().splitlines())==721
    assert 'complete_NDNDND_phiD' in (tmp_path/'report.md').read_text()
