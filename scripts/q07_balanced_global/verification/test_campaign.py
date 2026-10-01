"""Algebra/reduction checks; randomized slots are bookkeeping, not MMS evidence."""
import os,sys,unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
os.environ['JAX_COMPILATION_CACHE_DIR']=str(P/'local/jax_cache')
import campaign as c
import analyze
import numpy as np
from drbx.native.q_parallel_material import material_from_slots
class Checks(unittest.TestCase):
 def test_paired_independent_direct_weight_change(self):
  rng=np.random.default_rng(873);q=np.empty((3,7,5,5));q[...,:3]=rng.uniform(.8,1.2,size=(3,7,5,3));q[...,3:]=rng.uniform(-.2,.2,size=(3,7,5,2));phi=rng.normal(size=(3,7,3));L=rng.normal(size=(7,3));beta=rng.uniform(.2,1,size=7);k=rng.normal(size=7)
  actual,_=c.kernel.paired(q,phi,L,beta,.01,k);newL=L.copy();newL[:,2]=k-L[:,:2].sum(axis=1)
  direct=material_from_slots(q,phi,newL,beta,.01,tau=c.m.TAU,mu=c.m.MU)
  np.testing.assert_allclose(actual[...,9:12],np.asarray(direct.centered)[...,:3],atol=3e-14,rtol=1e-13)
  np.testing.assert_allclose(actual[...,15:18],np.asarray(direct.combined)[...,:3],atol=3e-14,rtol=1e-13)
  np.testing.assert_array_equal(actual[...,3:6],actual[...,12:15])
 def test_constant_geometry_response(self):
  q=np.broadcast_to(c.m.BASE,(1,3,5,5)).copy();L=np.arange(9).reshape(3,3)*.3;k=np.array([.2,-.7,.4]);a,_=c.kernel.paired(q,np.zeros((1,3,3)),L,np.ones(3),.01,k)
  n,te,ti,vi,ve=c.m.BASE
  expect=k[:,None]*np.array([-n*ve,2*te/3*(.71*(vi-ve)-ve),-2*ti/3*ve])
  np.testing.assert_allclose(a[0,:,9:12],expect,atol=1e-15)
 def test_hash_ignores_atime_rejects_content(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x';p.write_bytes(b'abc');first=c.sha(p);p.read_bytes();self.assertEqual(first,c.sha(p));p.write_bytes(b'abd');self.assertNotEqual(first,c.sha(p))
 def test_hash_rewrite_with_identical_reported_metadata(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'same_size';p.write_bytes(b'abc');reported=p.stat()
   # Deterministically reproduce the Perlmutter report, even on fine-clock FS.
   with patch.object(Path,'stat',return_value=reported):
    first=c.sha(p);p.write_bytes(b'abd');second=c.sha(p)
   self.assertEqual(first,c.hashlib.sha256(b'abc').hexdigest())
   self.assertEqual(second,c.hashlib.sha256(b'abd').hexdigest())
   self.assertNotEqual(first,second)
 def test_cached_then_corrupted_checkpoint_same_metadata(self):
  with tempfile.TemporaryDirectory() as td,patch.object(c,'OUT',Path(td)),patch.object(c,'design',return_value={'test':1}):
   st=c.empty_stats();st['owners']=np.array([0]);st['count'][0]=1;st['volume'][0]=1;p=Path(td)/'chunks/N32/block_000000.npz';c.save_arrays(p,st)
   r=dict(identity=c.digest({'test':1}),sha256=c.sha(p),owners=[0],finite=True,complete_owner_coverage=True,constant_error=0.,minimum_thermodynamic_slot=1.);c.write_json(p.with_suffix('.json'),r)
   reported=p.stat();original_stat=Path.stat
   def frozen_stat(path,*args,**kwargs):
    return reported if path==p else original_stat(path,*args,**kwargs)
   with patch.object(Path,'stat',frozen_stat):
    self.assertIsNotNone(c.checked_saved(32,0,[0]));data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
    with self.assertRaisesRegex(ValueError,'chunk identity/coverage'):c.checked_saved(32,0,[0])
 def test_complete_owner_batching(self):
  from types import SimpleNamespace
  t=SimpleNamespace(starts=np.r_[0,np.cumsum([3,200,7,100,21])]);groups=list(c.groups(t,np.arange(5)))
  self.assertEqual([g.tolist() for g in groups],[[0],[1],[2,3,4]])
 def test_volume_weighted_statistics(self):
  rng=np.random.default_rng(6);N=rng.normal(size=(2,4,18,18));O=rng.normal(size=(2,18,18));R=rng.normal(size=O.shape);s=c.empty_stats();w=np.array([2.,3.])
  with patch.object(c,'plans',return_value={'last_aggregate':3}):c.accumulate(s,np.array([4,5]),w,np.array([2,10]),32,N,O,R)
  for ei,E in enumerate((N-O[:,None],np.broadcast_to(O[:,None]-R[:,None],N.shape),N-R[:,None])):
   np.testing.assert_allclose(s['sum2'][0,ei],(w[:,None,None,None]*E**2).sum(axis=0));np.testing.assert_allclose(s['signed_sum'][0,ei],(w[:,None,None,None]*E).sum(axis=0))
   np.testing.assert_allclose(s['maximum'][0,ei],abs(E).max(axis=0));np.testing.assert_array_equal(s['max_owner'][0,ei],np.array([4,5])[abs(E).argmax(axis=0)])
 def test_checkpoint_corruption_rejected(self):
  with tempfile.TemporaryDirectory() as td,patch.object(c,'OUT',Path(td)),patch.object(c,'design',return_value={'test':1}):
   s=c.empty_stats();s['owners']=np.array([0,1]);s['count'][0]=2;s['volume'][0]=1;p=Path(td)/'chunks/N32/block_000000.npz';c.save_arrays(p,s)
   r=dict(identity=c.digest({'test':1}),sha256=c.sha(p),owners=[0,1],finite=True,complete_owner_coverage=True,constant_error=0.,minimum_thermodynamic_slot=1.);c.write_json(p.with_suffix('.json'),r)
   self.assertIsNotNone(c.checked_saved(32,0,[0,1]));p.write_bytes(b'corrupt')
   with self.assertRaises(ValueError):c.checked_saved(32,0,[0,1])
 def test_full_reduction_and_completion(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);out=root/'results';out.mkdir();parts=[]
   for n in (32,48,64):
    st=c.empty_stats();st['owners']=np.array([0]);st['count'][:]=1;st['volume'][:]=1.;st['sum2'][:]=1/n**4;st['maximum'][:]=1/n**2;st['max_owner'][:]=0;st['signed_sum'][:]=1/n**2;st['reference_sum2'][:]=1.
    c.save_arrays(out/f'chunks/N{n}/block_000000.npz',st);parts.append(st)
   old={key:np.stack([x[key] for x in parts]) for key in ('sum2','maximum','signed_sum','volume','count')}
   cols=[0,1,2,5,6,7,10,11,12]
   for key in ('sum2','maximum','signed_sum'):
    a=np.zeros(old[key].shape[:-1]+(44,));a[...,cols]=old[key][...,:9];old[key]=a
   c.save_arrays(root/'inputs/evidence/q07_material_global_20260930/totals.npz',old)
   d=dict(owner_counts={str(n):1 for n in (32,48,64)},kinds=c.m.KINDS,cases=c.m.DESIGNS,terms=c.kernel.TERMS,scientific_gate='test')
   with patch.object(c,'OUT',out),patch.object(c,'HERE',root),patch.object(c,'design',return_value=d),patch.object(c,'plans',return_value={'global_':[[0]]}),patch.object(c,'validate'):
    analyze.main();analyze.main(validate_only=True)
    self.assertTrue(json.loads((out/'completion.json').read_text())['passed'])
    data=json.loads((out/'analysis.json').read_text());self.assertAlmostEqual(data['orders'][0][0][0][0][0][0],2.)
    with np.load(out/'totals.npz') as z:bad=dict(z)
    bad['sum2'][0,0,0,0,0,0]+=1.;c.save_arrays(out/'totals.npz',bad)
    with self.assertRaises(ValueError):analyze.main(validate_only=True)
if __name__=='__main__':unittest.main()
