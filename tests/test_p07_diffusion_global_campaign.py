"""Bookkeeping safeguards for the research runner; HSX replay is documented separately."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
c=module('p07_campaign_test',ROOT/'scripts/p07_diffusion_global/campaign.py')
n=module('p07_numerics_test',ROOT/'scripts/p07_diffusion_global/numerics.py')

def test_all_face_numbering_and_incidence():
 # Enumeration and incidence identities only, not an idealized convergence gate.
 size=4;ids=np.arange(n.face_count(size));keys=n.face_keys(size,ids)
 np.testing.assert_array_equal(n.face_indices(size,keys),ids)
 lo,hi=n.incidence_for_keys(size,keys)
 for axis in range(3):
  mask=keys[:,0]==axis
  assert sum(mask & (lo<0))==size**2
  assert sum(mask & (hi<0))==size**2
  interior=mask&(lo>=0)&(hi>=0)
  assert np.all(hi[interior]-lo[interior]==size**(2-axis))
 counts=np.bincount(np.r_[lo[lo>=0],hi[hi>=0]],minlength=size**3)
 np.testing.assert_array_equal(counts,np.full(size**3,6))

def test_memory_budget_bounds_workers():
 a=SimpleNamespace(workers=8,memory_budget_gib=10,worker_memory_gib=3,memory_reserve_gib=1)
 assert c.effective_workers(a)==3
 a.memory_budget_gib=2
 with pytest.raises(ValueError,match='does not fit'):c.effective_workers(a)

def test_checkpoint_rejects_corruption_identity_and_incomplete_data(tmp_path):
 u={'scope':'global','id':'cell_q3_0','kind':'cell','order':3,'indices':[0,1]};ident='frozen'
 p=c.path_for(tmp_path,32,u)
 assert not c.valid_unit(tmp_path,32,u,ident)
 c.save(p,{'indices':np.array([0,1]),'numerator':np.ones((2,4)),'continuous_volume':np.ones(2)})
 assert not c.valid_unit(tmp_path,32,u,ident)
 c.write(p.with_suffix('.json'),{'identity':ident,'unit':u,'sha256':c.sha(p)})
 assert c.valid_unit(tmp_path,32,u,ident)
 with pytest.raises(ValueError,match='identity'):c.valid_unit(tmp_path,32,u,'different')
 with p.open('ab') as f:f.write(b'broken')
 with pytest.raises(ValueError,match='checksum'):c.valid_unit(tmp_path,32,u,ident)

def test_one_writer_lock(tmp_path):
 with c.locked(tmp_path):
  with pytest.raises(BlockingIOError):
   with c.locked(tmp_path):pass

def test_chunk_plans_cover_each_index_once():
 cfg={'face_chunk':3,'cell_chunk':2};units=c.units_for('preflight',np.arange(10),np.arange(7),cfg,(3,5,7))
 for kind,order,total in [('face',3,10),('cell',3,7),('cell',5,7),('cell',7,7)]:
  ids=[i for u in units if u['kind']==kind and u['order']==order for i in u['indices']]
  assert ids==list(range(total))

def test_assembly_sign_normalization_and_missing_coverage(tmp_path):
 size=2;nr=size**3;nf=n.face_count(size);raw=np.arange(nr);ids=np.arange(nf);keys=n.face_keys(size,ids);lo,hi=n.incidence_for_keys(size,keys)
 geom=SimpleNamespace(resolution=size,owner_flat_ids=raw,raw_owner=raw)
 ctx={'numerics':n,'geometry':geom,'volume':np.full(nr,2.),'raw_owner':raw}
 units=c.units_for('global',ids,raw,{'face_chunk':nf,'cell_chunk':nr});ident='frozen'
 flux=np.arange(nf*4).reshape(nf,4)/100
 for u in units:
  p=c.path_for(tmp_path,size,u)
  arrays={'indices':np.array(u['indices'])}
  if u['kind']=='face':arrays.update(flux=flux,oracle_flux=flux,constant_flux=np.zeros((nf,2)),lower_raw=lo,upper_raw=hi,donors=np.zeros((nf,1)),donor_count=np.ones(nf))
  else:arrays.update(numerator=np.full((nr,4),6.),continuous_volume=np.full(nr,3.))
  c.save(p,arrays);c.write(p.with_suffix('.json'),{'identity':ident,'unit':u,'sha256':c.sha(p)})
 out=c.aggregate(tmp_path,size,units,ident,ctx,raw,raw)
 expected=np.zeros((nr,4))
 for j in range(nf):
  if lo[j]>=0:expected[lo[j]]-=flux[j]
  if hi[j]>=0:expected[hi[j]]+=flux[j]
 np.testing.assert_allclose(out['action'],expected/2,atol=1e-15)
 np.testing.assert_array_equal(out['reference_q3'],np.full((nr,4),2.))
 np.testing.assert_array_equal(out['stored_volume_reference_q3'],np.full((nr,4),3.))
 np.testing.assert_allclose(out['balance_residual'],0,atol=1e-13)
 with pytest.raises(ValueError,match='overlapping'):c.aggregate(tmp_path,size,units+units,ident,ctx,raw,raw)

def test_validator_reassembles_and_rejects_changed_summary(tmp_path,monkeypatch):
 size=2;ids=np.arange(n.face_count(size));raw=np.arange(size**3);keys=n.face_keys(size,ids);lo,hi=n.incidence_for_keys(size,keys)
 geom=SimpleNamespace(resolution=size,owner_flat_ids=raw,raw_owner=raw)
 num=SimpleNamespace(face_count=n.face_count,masks=n.masks,select_owners=lambda *a:(raw,{}),closure=lambda *a:(raw,ids))
 ctx={'numerics':num,'geometry':geom,'centers':None,'volume':np.ones(len(raw)),'raw_owner':raw}
 cfg={'face_chunk':len(ids),'cell_chunk':len(raw)};ident='test';flux=np.arange(len(ids)*4).reshape(-1,4)/100
 for scope,file in [('preflight','preflight'),('global','result')]:
  units=c.units_for(scope,ids,raw,cfg)
  for u in units:
   p=c.path_for(tmp_path,size,u);a={'indices':np.array(u['indices'])}
   if u['kind']=='face':a.update(flux=flux,oracle_flux=flux,constant_flux=np.zeros((len(ids),2)),lower_raw=lo,upper_raw=hi,donors=np.zeros((len(ids),1)),donor_count=np.ones(len(ids)))
   else:a.update(numerator=np.ones((len(raw),4)),continuous_volume=np.ones(len(raw)))
   c.save(p,a);c.write(p.with_suffix('.json'),{'identity':ident,'unit':u,'sha256':c.sha(p)})
  a=c.aggregate(tmp_path,size,units,ident,ctx,raw,raw);p=tmp_path/f'N{size}/{file}.npz';c.save(p,a)
  c.write(tmp_path/f'N{size}/{scope}_plan.json',{'identity':ident,'units':units})
  c.write(p.with_suffix('.json'),{'identity':ident,'arrays_sha256':c.sha(p),'stats':c.summarize(a)})
 monkeypatch.setattr(c,'initialize',lambda *a:None);monkeypatch.setattr(c,'STATE',ctx)
 args=SimpleNamespace(input_root=tmp_path,output=tmp_path,resolutions=[size]);c.validate(args,cfg,ident)
 import json
 p=tmp_path/f'N{size}/result.json';d=json.loads(p.read_text());d['stats']['phi_mms']['l2']+=1;c.write(p,d)
 with pytest.raises(ValueError,match='statistics'):c.validate(args,cfg,ident)
