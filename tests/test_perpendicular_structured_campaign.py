"""Point-functional contracts for the shared research reconstruction (no HSX files)."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from perpendicular_structured.reconstruction import StructuredReconstruction
from p07_combined_global.kernels import num

@pytest.fixture(scope='module')
def context():
 n=32;faces=(np.linspace(0,1,n+1),np.linspace(0,2*np.pi,n+1),np.linspace(0,1,n+1));centers=tuple((f[:-1]+f[1:])/2 for f in faces)
 ijk=np.array(np.unravel_index(np.arange(n**3),(n,)*3)).T;pts=np.column_stack([centers[a][ijk[:,a]] for a in range(3)])
 ro=np.empty((n,n,n),int);count=0
 for i in range(n):
  block=[32,8,4,2][i] if i<4 else 1
  for j in range(0,n,block):
   for k in range(n):ro[i,j:j+block,k]=count;count+=1
 ro=ro.ravel();rv=1+.1*np.cos(pts[:,1]);vol=np.bincount(ro,weights=rv);order=np.argsort(ro,kind='stable');starts=np.r_[0,np.cumsum(np.bincount(ro))]
 xy=np.column_stack((pts[:,0]*np.cos(pts[:,1]),pts[:,0]*np.sin(pts[:,1])));centroid=np.column_stack([np.bincount(ro,weights=rv*xy[:,a])/vol for a in range(2)])
 g=SimpleNamespace(dr=1/n,dtheta=2*np.pi/n,deta=1/n,eta_period=1.,owner_centroid_xy=centroid)
 return SimpleNamespace(n=n,faces=faces,centers=centers,pts=pts,xy=xy,ro=ro,rv=rv,vol=vol,order=order,starts=starts,g=g)

def project(t,f):return np.bincount(t.ro,weights=t.rv*f)/t.vol

def constant(p):return np.ones((len(p),1)),np.zeros((len(p),3,1))

@pytest.mark.parametrize('key',[(0,0,2,3),(0,1,2,3),(1,2,2,3),(0,8,2,3),(0,20,2,3),(0,30,2,3),(0,32,2,3),(1,31,2,3)])
def test_constant_and_side_rows(context,key):
 t=context;S=StructuredReconstruction(t);p,_=num.quadrature(t.faces,np.array([key]),3,face=True);r=S.rows(key,p[0]);v,g=r.apply(np.ones((len(t.vol),1)),constant)
 if key[0]==0 and key[1]==0:assert not len(r.donor_ids);return
 np.testing.assert_allclose(v,1,atol=1e-11);np.testing.assert_allclose(g,0,atol=1e-10)
 for row in S.side_rows(key,p[0]):
  if row is not None:
   v,g=row.apply(np.ones((len(t.vol),1)),constant);np.testing.assert_allclose(v,1,atol=1e-11);np.testing.assert_allclose(g,0,atol=1e-10)

def test_axis_quartic_owner_moments(context):
 t=context;S=StructuredReconstruction(t);key=(0,1,2,3);p,_=num.quadrature(t.faces,np.array([key]),3,face=True);p=p[0]
 f=t.xy[:,0]**4+t.xy[:,1]**3
 v,g=S.rows(key,p).apply(project(t,f)[:,None]);x=p[:,0]*np.cos(p[:,1]);y=p[:,0]*np.sin(p[:,1])
 np.testing.assert_allclose(v[:,0],x**4+y**3,atol=1e-12)
 np.testing.assert_allclose(g[:,0,0],4*x**3*np.cos(p[:,1])+3*y**2*np.sin(p[:,1]),atol=1e-11)
 np.testing.assert_allclose(g[:,1,0],p[:,0]*(-4*x**3*np.sin(p[:,1])+3*y**2*np.cos(p[:,1])),atol=1e-11)

def test_quartic_wall_lift_and_nonzero_normal(context):
 t=context;S=StructuredReconstruction(t);key=(0,32,4,5);p,_=num.quadrature(t.faces,np.array([key]),5,face=True);p=p[0]
 def analytic(p):
  r,th,eta=p.T;f=r**4-1+.3*np.sin(th)+.2*np.cos(2*np.pi*eta)
  g=np.stack((4*r**3,.3*np.cos(th),-.4*np.pi*np.sin(2*np.pi*eta)),axis=1)
  return f[:,None],g[:,:,None]
 row=S.rows(key,p);v,g=row.apply(project(t,analytic(t.pts)[0][:,0])[:,None],analytic);ev,eg=analytic(p)
 np.testing.assert_allclose(v,ev,atol=1e-12);np.testing.assert_allclose(g,eg,atol=2e-10)
 with pytest.raises(ValueError,match='trace'):row.apply(np.ones((len(t.vol),1)))

def test_adapted_jump_is_nontrivial_but_cubic_exact(context):
 t=context;S=StructuredReconstruction(t);key=(0,16,10,7);p,_=num.quadrature(t.faces,np.array([key]),3,face=True);p=p[0];left,right=S.side_rows(key,p)
 cubic=project(t,t.pts[:,0]**3)[:,None]
 np.testing.assert_allclose(left.apply(cubic)[0],right.apply(cubic)[0],atol=1e-12)
 nonpoly=project(t,np.exp(3*t.pts[:,0]))[:,None]
 assert np.max(abs(right.apply(nonpoly)[0]-left.apply(nonpoly)[0]))>1e-7


@pytest.mark.parametrize('key',[(0,1,2,3),(0,8,2,3),(0,20,2,3),(0,32,2,3),(1,31,2,3)])
def test_value_only_is_exact_full_action_and_cache_is_exact(context,key):
 t=context;S=StructuredReconstruction(t);p,_=num.quadrature(t.faces,np.array([key]),3,face=True)
 data=np.random.default_rng(129).normal(size=(len(t.vol),3))
 def trace(q):
  v=np.repeat(np.sin(q[:,1,None]),3,axis=1)
  g=np.zeros((len(q),3,3));g[:,1]=np.cos(q[:,1,None])
  return v,g
 first=S.rows(key,p[0]);second=S.rows(key,p[0])
 np.testing.assert_array_equal(first.value,second.value)
 np.testing.assert_array_equal(first.gradient,second.gradient)
 for row in (first,*S.side_rows(key,p[0])):
  if row is not None:
   np.testing.assert_array_equal(row.apply_value(data,trace),row.apply(data,trace)[0])
 assert S._basis.cache_info().currsize<=8192
