"""Algebra/reduction checks; randomized slots are bookkeeping, not MMS evidence."""
import os,sys,unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',str(Path(os.environ.get('Q07_OUTPUT',str(P/'local')))/'test_jax_cache'))
import campaign as c
import analyze
import numpy as np
from drbx.native.q_parallel_material import material_from_slots
class Checks(unittest.TestCase):
 def test_bounded_replay_sum_budget(self):
  a=np.zeros((4,c.m.NF,15));b=np.zeros_like(a)
  for start in (0,):
   a[...,start:start+5]=.75e-8;a[...,start+5:start+10]=.75e-8;a[...,start+10:start+15]=1.5e-8
  r=c.bounded_action_replay(a,b)
  self.assertAlmostEqual(r['combined'],1.5e-8,delta=1e-20)
 def test_bounded_replay_constituent_cannot_hide_by_cancellation(self):
  a=np.zeros((c.m.NF,15));a[...,0]=1.25e-8;a[...,5]=-1.25e-8
  with self.assertRaisesRegex(ValueError,'bounded replay centered'):c.bounded_action_replay(a,np.zeros_like(a))
 def test_bounded_replay_sum_identity(self):
  a=np.zeros((c.m.NF,15));a[...,14]=5e-9
  with self.assertRaisesRegex(ValueError,'sum identity actual'):c.bounded_action_replay(a,np.zeros_like(a))
  with self.assertRaisesRegex(ValueError,'sum identity expected'):c.bounded_action_replay(np.zeros_like(a),a)
 def test_bounded_replay_rejects_nonfinite_and_shape(self):
  a=np.zeros((c.m.NF,15));a[0,0]=np.nan
  with self.assertRaisesRegex(ValueError,'nonfinite'):c.bounded_action_replay(a,np.zeros_like(a))
  with self.assertRaisesRegex(ValueError,'shape'):c.bounded_action_replay(np.zeros(30),np.zeros(15))
 def test_constant_geometry_response(self):
  q=np.broadcast_to(c.m.BASE,(1,3,5,5)).copy();k=np.array([.2,-.7,.4]);L=np.array([[-2.,2.,x] for x in k]);beta=np.ones(3)
  a=c.kernel.action(q,np.zeros((1,3,3)),L,beta,.01)
  n,te,ti,vi,ve=c.m.BASE;expect=k[:,None]*np.array([-n*ve,2*te/3*(.71*(vi-ve)-ve),-2*ti/3*ve,0.,0.])
  np.testing.assert_allclose(a[0,0],expect,atol=1e-14);np.testing.assert_array_equal(a[1],0.)
 def test_short_trace_offsets_and_corruption(self):
  from types import SimpleNamespace
  import frozen.trace
  def fake(tracer,p,dt,steps):
   self.assertEqual(steps,64);p=np.asarray(p).copy();p[:,2]+=np.asarray(dt);return p,np.ones(len(p),bool),np.zeros(len(p),bool),np.zeros(len(p),bool)
  with tempfile.TemporaryDirectory() as td,patch.object(c,'OUT',Path(td)),patch.object(c,'ENV',{'ctx':{'tracer':None}}),patch.object(c,'design',return_value={'test':1}),patch.object(frozen.trace,'trace',fake):
   pts=np.array([[.5,.3,1.]]);e,r=c.short_traces(48,0,np.array([2]),pts);h=2*np.pi/48
   np.testing.assert_allclose(e[0,:,2]-1,np.array([-2,2,-1,1])*h/64,rtol=0,atol=1e-15)
   again,_=c.short_traces(48,0,np.array([2]),pts);np.testing.assert_array_equal(e,again)
   (Path(td)/r['relative_path']).write_bytes(b'corrupt')
   with self.assertRaisesRegex(ValueError,'trace cache identity/hash'):c.short_traces(48,0,np.array([2]),pts)
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
 def test_c3_is_pinned_and_global_trace_gates_required(self):
  source=(P/'source/frozen_reference/frozen/model.py').read_text()
  self.assertIn("toroidal_method='compact_c3'",source)
  self.assertEqual(len(c.kernel.TERMS),15)
  self.assertIn('trace_validation_N',Path(c.__file__).read_text())
 def test_complete_owner_batching(self):
  from types import SimpleNamespace
  t=SimpleNamespace(starts=np.r_[0,np.cumsum([3,200,7,100,21])]);groups=list(c.groups(t,np.arange(5)))
  self.assertEqual([g.tolist() for g in groups],[[0],[1],[2,3,4]])
 def test_volume_weighted_statistics(self):
  rng=np.random.default_rng(6);N=rng.normal(size=(2,4,c.m.NF,len(c.kernel.TERMS)));O=rng.normal(size=(2,c.m.NF,len(c.kernel.TERMS)));R=rng.normal(size=O.shape);s=c.empty_stats();w=np.array([2.,3.])
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
 def test_span_specific_constant_roundoff_gate(self):
  with tempfile.TemporaryDirectory() as td,patch.object(c,'OUT',Path(td)),patch.object(c,'design',return_value={'test':1}):
   s=c.empty_stats();s['owners']=np.array([0]);s['count'][0]=1;s['volume'][0]=1;p=Path(td)/'chunks/N32/block_000000.npz';c.save_arrays(p,s)
   r=dict(identity=c.digest({'test':1}),sha256=c.sha(p),owners=[0],finite=True,complete_owner_coverage=True,constant_error=5e-8,constant_error_by_span=[5e-8],minimum_thermodynamic_slot=1.);c.write_json(p.with_suffix('.json'),r)
   self.assertIsNotNone(c.checked_saved(32,0,[0]))
   for values in ([2e-7],[5e-7]):
    r['constant_error_by_span']=values;c.write_json(p.with_suffix('.json'),r)
    with self.assertRaisesRegex(ValueError,'chunk gate'):c.checked_saved(32,0,[0])
 def test_full_reduction_and_completion(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);out=root/'results';out.mkdir();parts=[]
   for n in (32,48,64):
    st=c.empty_stats();st['owners']=np.array([0]);st['count'][:]=1;st['volume'][:]=1.;st['sum2'][:]=1/n**4;st['maximum'][:]=1/n**2;st['max_owner'][:]=0;st['signed_sum'][:]=1/n**2;st['reference_sum2'][:]=1.
    c.save_arrays(out/f'chunks/N{n}/block_000000.npz',st);parts.append(st)
   d=dict(owner_counts={str(n):1 for n in (32,48,64)},kinds=c.m.KINDS,cases=c.m.DESIGNS,terms=c.kernel.TERMS,scientific_gate='test')
   with patch.object(c,'OUT',out),patch.object(c,'HERE',root),patch.object(c,'design',return_value=d),patch.object(c,'plans',return_value={'global_':[[0]]}),patch.object(c,'validate'):
    analyze.main();analyze.main(validate_only=True)
    self.assertTrue(json.loads((out/'completion.json').read_text())['passed'])
    data=json.loads((out/'analysis.json').read_text());self.assertAlmostEqual(data['orders'][0][0][0][0][0][0],2.)
    with np.load(out/'totals.npz') as z:bad=dict(z)
    bad['sum2'][0,0,0,0,0,0]+=1.;c.save_arrays(out/'totals.npz',bad)
    with self.assertRaises(ValueError):analyze.main(validate_only=True)
if __name__=='__main__':unittest.main()
