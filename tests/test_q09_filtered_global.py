"""Filtered campaign plumbing tests; actual HSX replay is a separate preflight."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from scripts.q09_filtered_global import common as c
from scripts.q09_filtered_global.provider import FilteredProvider
from scripts.q09_evolved_mms.provider import PreparedProvider,FROZEN_IDENTITY
from scripts.q09_filtered_global.prepare import chunk_valid,SelectedEndpoints
from scripts.q09_evolved_mms.evolution import step_count


def test_time_budget_and_filtered_identity():
    cfg=c.configuration()
    assert cfg['cases']==24 and cfg['time_steps']==5800
    for n in c.GRIDS:assert step_count(0,c.END,c.END/c.STEPS[n])==c.STEPS[n]
    assert cfg['trace_order']==['outer_minus','outer_plus','inner_minus','inner_plus']
    assert c.GEOMETRY_ID.endswith(c.ARM)


def test_lineages_cannot_be_conflated():
    bank=SimpleNamespace(identity='a'*64,metadata={'geometry_identity':c.GEOMETRY_ID})
    args=(bank,{},None,None,None,{},dict(filtered_arm=c.ARM,prepared_bank_identity=bank.identity,filtered_campaign='b'*64))
    filtered=FilteredProvider(*args);filtered._validate_lineage()
    with pytest.raises(ValueError):PreparedProvider(*args)._validate_lineage()
    filtered.input_hashes['q08_receipt_identity']=FROZEN_IDENTITY
    with pytest.raises(ValueError):filtered._validate_lineage()


def test_chunk_resume_rejects_same_size_mutation(tmp_path):
    files={}
    for name in ('bank.npz','geometry.npz','reference.npz','reference.json'):
        (tmp_path/name).write_bytes(b'abc');files[name]=c.sha(tmp_path/name)
    c.write(tmp_path/'stats.json',dict(passed=True,campaign_identity='a',owners=[3],files=files))
    assert chunk_valid(tmp_path,'a',[3])
    (tmp_path/'reference.npz').write_bytes(b'abd')
    with pytest.raises(ValueError,match='content'):chunk_valid(tmp_path,'a',[3])


def test_receipt_coverage_and_authority(tmp_path):
    with pytest.raises(FileNotFoundError):c.require(tmp_path,'stage','a')
    c.write(tmp_path/'stage.json',dict(passed=True,identity='b'))
    with pytest.raises(ValueError):c.require(tmp_path,'stage','a')


def test_bounded_trace_index_cannot_supply_unknown_owner():
    ends=np.arange(24).reshape(2,4,3)
    points=SelectedEndpoints([7,19],ends)
    np.testing.assert_array_equal(points[[19,7]],ends[[1,0]])
    with pytest.raises(KeyError):points[[8]]
