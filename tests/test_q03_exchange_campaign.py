"""Bookkeeping and numerical replay on a bounded actual HSX excerpt."""
import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve();CODE=HERE.parents[1]/'scripts/q03_exchange_campaign'
sys.path.insert(0,str(CODE))
import campaign as q
import common
from numerics import fit
from oracle import selector as oracle
from optimized_exchange import selector

@pytest.fixture
def data():
 with np.load(HERE.parent/'data/q03_exchange_hsx_excerpt.npz') as a:return {k:a[k].copy() for k in a.files}

def test_frozen_hsx_optimized_and_oracle(data):
 for face in range(2):
  args=common.args_for(data,face,32);a,am=selector(*args);b,bm=oracle(*args)
  np.testing.assert_array_equal(a,b)
  assert [(s['out'],s['in']) for s in am['swaps']]==[(s['out'],s['in']) for s in bm['swaps']]
  coefficient,diag=fit(args[0],a,args[4],args[5],args[6]);assert diag['rank']==19 and np.all(np.isfinite(coefficient))

def test_chunk_resume_corruption_and_identity(data,tmp_path):
 ident={'test':'actual_hsx'};q.STATE=(data,32,ident);path=tmp_path/'one.npz';ids=np.array([0])
 assert not q.compute((ids,path))['reused'];assert q.compute((ids,path))['reused']
 with pytest.raises(RuntimeError,match='incompatible'):q.receipt_valid(path,{'test':'changed'},ids)
 with path.open('ab') as f:f.write(b'corrupt')
 with pytest.raises(RuntimeError,match='corrupt'):q.receipt_valid(path,ident,ids)

def test_stream_assembly_counts_faces_once(data,tmp_path,monkeypatch):
 ident={'test':'actual_hsx'};q.STATE=(data,32,ident);monkeypatch.setattr(q,'load',lambda *args:data)
 jobs=q.jobs_for(32,data,tmp_path,1)
 for job in jobs:q.compute(job)
 result=q.assemble('unused',32,tmp_path,ident,1);assert result['completed'] and result['faces']==2
 expected=np.zeros((len(data['volume']),4,2))
 for ids,path in jobs:
  with np.load(path) as a:
   for i,face in enumerate(ids):expected[data['face.minus'][face]]+=a['flux'][i];expected[data['face.plus'][face]]-=a['flux'][i]
 expected/=data['volume'][:,None,None]
 with np.load(tmp_path/'N32/actions.npz') as a:
  for j,f in enumerate(q.FIELDS):
   for k,o in enumerate(q.OBS):np.testing.assert_array_equal(a[f'action:{f}:{o}'],expected[:,j,k])
 jobs[-1][1].unlink()
 with pytest.raises(RuntimeError,match='missing chunk'):q.assemble('unused',32,tmp_path,ident,1)
