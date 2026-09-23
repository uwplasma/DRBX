"""Geometry-independent checkpoint/locking checks; numerical audit uses actual HSX."""
import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/q_fci_return_campaign/campaign.py'
spec=importlib.util.spec_from_file_location('q_fci_campaign_test',PATH)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def test_incomplete_and_corrupt_checkpoint(tmp_path):
    p=tmp_path/'unit.npz'
    assert not c.completed(p,'source',[1,2])
    p.write_bytes(b'payload');c.receipt(p,'source',[1,2])
    assert c.completed(p,'source',[1,2])
    p.write_bytes(b'damaged')
    with pytest.raises(RuntimeError,match='corrupt/incompatible'):c.completed(p,'source',[1,2])

@pytest.mark.parametrize('identity,unit',[('changed',[1,2]),('source',[2,1]),('source',[1,2,3])])
def test_checkpoint_rejects_wrong_identity_and_coverage(tmp_path,identity,unit):
    p=tmp_path/'unit.npz';p.write_bytes(b'payload');c.receipt(p,'source',[1,2])
    with pytest.raises(RuntimeError):c.completed(p,identity,unit)

def test_writer_lock_rejects_second_writer(tmp_path):
    with c.lock(tmp_path/'lock'):
        with pytest.raises(RuntimeError,match='already has a writer'):
            with c.lock(tmp_path/'lock'):pass
    with c.lock(tmp_path/'lock'):pass
