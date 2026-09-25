"""Conservation/sign and checkpoint contracts for the new bracket campaign."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from p05_structured_global import campaign as c, numerics as k


def test_flux_sign_and_swapped_antisymmetry():
 h=np.array([[0.,0.,1.]]);v=np.arange(1.,9.)[None];g=np.zeros((1,3,8));g[0,:,0]=[1,2,0];g[0,:,1]=[3,4,0]
 gen,product,U=k.flux_arrays(h,np.ones(1),v,g,0)
 assert U[0,0]==2 and U[0,1]==4
 assert product[0,0]==4 and product[0,1]==4
 # -cross(h, grad(phi)).grad(f) = 2*3 - 1*4 = 2.
 assert np.dot(-np.cross(h[0],g[0,:,0]),g[0,:,1])==2
 assert np.dot(-np.cross(h[0],g[0,:,1]),g[0,:,0])==-2


def test_shared_flux_cancels_for_two_owners():
 t=SimpleNamespace(vol=np.array([2.,3.]));v=np.zeros((2,8));ep=np.array([[0,1]])
 f=dict(product=np.ones((1,8,2)),generator=np.zeros((1,8)),oracle_product=np.ones((1,8,2)),oracle_generator=np.zeros((1,8)),upwind=np.ones((1,8))*-.7)
 cell=dict(correction=np.zeros((1,8,2)),oracle_correction=np.zeros((1,8,2)),volume=np.ones(1),direct=np.zeros((1,8)))
 a,_=c.assemble_subset(t,v,0,ep,f,cell);b,_=c.assemble_subset(t,v,1,ep,f,cell)
 np.testing.assert_allclose(2*a+3*b,0,atol=1e-15)
 np.testing.assert_allclose(a[:,2],(a[:,0]+a[:,1])/2)


def test_checkpoint_rejects_corruption_and_changed_identity(tmp_path):
 unit=dict(n=32,stage='faces',start=0,stop=2);p=c.chunkpath(tmp_path,unit)
 c.save_npz(p,ids=np.array([1,2]));c.save_json(p.with_suffix('.json'),dict(identity='abc',unit=unit,sha256=c.sha(p)))
 assert c.valid(tmp_path,unit,'abc')
 assert not c.valid(tmp_path,unit,'changed')
 c.save_npz(p,ids=np.array([3,4]))
 assert not c.valid(tmp_path,unit,'abc')


def test_memory_cap_is_enforced():
 args=SimpleNamespace(workers=20,memory_budget_gib=10,worker_memory_gib=2,memory_reserve_gib=2)
 assert c.workers(args)==4
 args.memory_budget_gib=3
 with pytest.raises(ValueError,match='insufficient'):c.workers(args)


def test_high_order_control_curl_stays_inside_radial_domain():
 class Reference:
  def _metric(self,p):
   assert np.all((p[:,0]>=0)&(p[:,0]<=1))
   return dict(B=np.ones(len(p)),bcov=np.column_stack((0*p[:,0],p[:,0]**3,0*p[:,0])))
 p=np.array([[1e-5,.3,.2],[.5,.3,.2],[1-1e-5,.3,.2]])
 actual=k.curl_h(Reference(),p)
 expected=np.zeros_like(actual);expected[:,2]=3*p[:,0]**2
 np.testing.assert_allclose(actual,expected,atol=2e-10)


def test_dirichlet_catalogue_has_nonzero_wall_normal_derivatives(monkeypatch):
 class Ref:
  eta_period=1.
  def _fields_raw(self,p,t):
   z=np.zeros(len(p));return {'phi':(z,z,z,z)}
 monkeypatch.setattr(k,'omega',lambda ref,p:np.zeros(len(p)))
 points=np.array([[1.,.2,.13],[1.,.4,.41]])
 v,g=k.fields(Ref(),points)
 np.testing.assert_array_equal(v[:,4],0)
 assert np.min(abs(g[:,0,4]))>.05
 assert np.max(abs(v[:,5]))>.02 and np.max(abs(g[:,1:,5]))>.02


def test_complete_cartesian_velocity_owner_matches_direct_integral(monkeypatch):
 # Use the shared synthetic owner geometry fixture without any HSX reference data.
 import runpy
 helper=runpy.run_path(str(Path(__file__).with_name("test_perpendicular_structured_campaign.py")))
 context,project=helper["context"],helper["project"]
 t=context.__wrapped__();S=k.StructuredReconstruction(t);n=t.n;i,j,e=16,10,7
 def analytic(ref,p,**kwargs):
  v=np.tile(np.sin(p[:,1,None]),(1,8));g=np.zeros((len(p),3,8));g[:,1]=np.cos(p[:,1,None])
  v[:,0]=p[:,0]**2;g[:,:,0]=0;g[:,0,0]=2*p[:,0]
  v[:,6]=1;g[:,:,6]=0
  v[:,7]=v[:,0];g[:,:,7]=g[:,:,0]
  return v,g
 class Ref:
  finite_difference_step=2e-4
  def _metric(self,p):return dict(bcov=np.tile([0.,0.,1.],(len(p),1)),B=np.ones(len(p)),J=np.ones(len(p)))
 ref=Ref();monkeypatch.setattr(k,'fields',analytic)
 raw=np.ravel_multi_index((i,j,e),(n,)*3);oid=int(t.ro[raw]);values=np.column_stack([project(t,x) for x in analytic(ref,t.pts)[0].T])
 ids=[];ep=[]
 for axis in range(3):
  for side in (0,1):
   idx=[i,j,e];idx[axis]+=side
   fid=idx[0]*n*n+idx[1]*n+idx[2]+(0 if axis==0 else (n+1)*n*n+(axis-1)*n**3)
   ids.append(fid);ep.append([oid,-1] if side else [-1,oid])
 ids=np.array(ids);ep=np.array(ep);f=k.face_chunk(t,ref,S,values,ids);cells=k.cell_chunk(t,ref,S,values,np.array([raw]))
 got,oracle=c.assemble_subset(t,values,oid,ep,f,cells)
 # The candidate uses stored owner volume; direct oracle uses integrated J volume.
 expected=oracle[:,1]*cells['volume'].sum()/t.vol[oid]
 np.testing.assert_allclose(got[:,:3],np.repeat(expected[:,None],3,axis=1),atol=2e-11)
 np.testing.assert_allclose(got[:,3],got[:,0],atol=2e-11)
 assert f['constant_error'].max()<1e-10


def test_global_reduce_validates_complete_coverage_and_zero_control(tmp_path,monkeypatch):
 import json
 args=SimpleNamespace(output=tmp_path,input_root=tmp_path)
 monkeypatch.setattr(c,'current',lambda args:'test')
 def context(n,root):
  ro=np.array([0,1,1,2,3,4,5]);starts=np.array([0,1,3,4,5,6,7]);rad=np.array([.5/n,1.5/n,1.5/n,.25,.6,1-2/n,1-.5/n])
  return SimpleNamespace(vol=np.bincount(ro).astype(float),ro=ro,starts=starts,order=np.arange(7),pts=np.column_stack((rad,np.zeros((7,2)))))
 monkeypatch.setattr(k,'load_context',context)
 def unitlist(args,stage,n):return [dict(n=n,stage=stage,start=0,stop=6 if stage.startswith('face') else 1 if stage=='controls' else 7)]
 monkeypatch.setattr(c,'units',unitlist)
 for n in (32,48,64):
  t=context(n,None);ep=np.column_stack((np.arange(6),np.roll(np.arange(6),-1)))
  c.save_npz(tmp_path/f'N{n}.topology.npz',face_ids=np.arange(6),endpoints=ep)
  c.save_npz(tmp_path/f'N{n}.observations.npz',values=np.zeros((6,8)))
  c.save_npz(tmp_path/f'N{n}.selection.npz',owners=np.array([2]))
  for stage in ('faces','cells','face_reference','cell_reference','controls'):
   u=unitlist(args,stage,n)[0]
   if stage=='controls':z=dict(owner=np.array(2),q5=np.zeros((8,2)),q7=np.zeros((8,2)),q9=np.zeros((8,2)),q7_halfstep=np.zeros((8,2)))
   elif stage.startswith('face'):
    product=np.ones((6,8,2))*np.arange(6)[:,None,None]/n**3;product[:,5]=0
    z=dict(ids=np.arange(6),product=product,generator=np.zeros((6,8)),oracle_product=np.zeros((6,8,2)),oracle_generator=np.zeros((6,8)),upwind=np.zeros((6,8)),constant_error=np.zeros(6),support_residual=np.zeros(6))
   else:z=dict(ids=np.arange(7),owners=t.ro,correction=np.zeros((7,8,2)),oracle_correction=np.zeros((7,8,2)),direct=np.zeros((7,8)),volume=np.ones(7),constant_error=np.zeros(7),support_residual=np.zeros(7))
   path=c.chunkpath(tmp_path,u);c.save_npz(path,**z);c.save_json(path.with_suffix('.json'),dict(identity='test',unit=u,sha256=c.sha(path)))
 c.reduce(args)
 summary=json.loads((tmp_path/'summary.json').read_text())
 assert summary['status']=='computation complete'
 assert summary['results']['64']['independent_assembly_max']==0
 # A/B cancel in C here; zero diagnostic errors must serialize as finite orders.
 assert np.isfinite(summary['orders']).all()
 # Corruption is detected on prescribed validation, not silently accepted.
 path=c.chunkpath(tmp_path,unitlist(args,'faces',32)[0]);path.write_bytes(b'broken')
 with pytest.raises(ValueError,match='missing/incompatible'):c.reduce(args)
