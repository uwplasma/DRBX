"""An execution upgrade cannot silently bless different numerical inputs/sources."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from perpendicular_structured import optimization_resume as u


@pytest.fixture
def release(tmp_path,monkeypatch):
    config={'quadrature':3};inputs={'immutable':'abc'}
    baseline={'a.py':'old'};active={'a.py':'new'}
    rule=dict(baseline_sources=baseline,execution_sources=active,configuration=config,inputs=inputs)
    policy=tmp_path/'policy.json'
    policy.write_text(json.dumps(dict(baseline_commit='before',campaigns={'p05':rule,'p06':rule})))
    monkeypatch.setattr(u,'POLICY',policy)
    return tmp_path,rule


@pytest.mark.parametrize('kind',['p05','p06'])
def test_explicit_upgrade_preserves_manifest_and_rejects_changed_execution(release,kind):
    output,rule=release
    payload=dict(sources=rule['baseline_sources'],inputs=rule['inputs'])
    payload['config' if kind=='p05' else 'configuration']=rule['configuration']
    if kind=='p05':payload['commit']='before'
    manifest=({'content':payload,'identity':u.canonical(payload)} if kind=='p05' else
              {'content_identity':payload,'sha256':u.canonical(payload)})
    frozen=json.dumps(manifest,sort_keys=True)
    with pytest.raises(ValueError,match='explicit'):u.check(output,kind,manifest,rule['execution_sources'])
    cert=u.adopt(output,kind,manifest,rule['execution_sources'],'after')
    assert json.dumps(manifest,sort_keys=True)==frozen
    assert u.check(output,kind,manifest,rule['execution_sources'])==cert
    assert u.adopt(output,kind,manifest,rule['execution_sources'],'after')==cert
    with pytest.raises(ValueError,match='mismatch'):u.check(output,kind,manifest,{'a.py':'other'})
    with pytest.raises(ValueError,match='mismatch'):u.check(output,kind,manifest,rule['execution_sources'],'different')
    payload['inputs']={'immutable':'changed'}
    with pytest.raises(ValueError):u.adopt(output,kind,manifest,rule['execution_sources'],'after')


def test_upgrade_rejects_unknown_baseline(release):
    output,rule=release
    payload=dict(config=rule['configuration'],inputs=rule['inputs'],sources={'a.py':'unknown'},commit='before')
    with pytest.raises(ValueError,match='tested optimization release'):
        u.adopt(output,'p05',{'content':payload,'identity':u.canonical(payload)},rule['execution_sources'],'after')
