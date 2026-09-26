"""Structured physical-normal wall trace on an oblique affine metric."""
import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext
from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows
from drbx.native.fci_perpendicular_neumann_rows import lower_neumann_point_rows, apply_neumann_point_rows


def test_oblique_normal_trace_and_runtime_jvp():
    n=12
    faces=(np.linspace(0,1,n+1),np.linspace(0,2*np.pi,n+1),np.linspace(0,2*np.pi,n+1))
    centers=tuple((axis[:-1]+axis[1:])/2 for axis in faces)
    raw=np.arange(n**3)
    ijk=np.array(np.unravel_index(raw,(n,n,n))).T
    xy=np.column_stack((centers[0][ijk[:,0]]*np.cos(centers[1][ijk[:,1]]),
                        centers[0][ijk[:,0]]*np.sin(centers[1][ijk[:,1]])))
    context=PointRowContext.from_arrays(faces=faces,centers=centers,
        raw_to_owner=raw,raw_volume=np.ones(n**3),owner_volume=np.ones(n**3),
        owner_centroid_xy=xy,eta_period=2*np.pi,dr=1/n,dtheta=2*np.pi/n,deta=2*np.pi/n)
    # x=(u+0.2 theta,theta,eta): n dot grad = sqrt(1.04) f_u
    #                                      - 0.2/sqrt(1.04) f_theta.
    coeff=np.array((np.sqrt(1.04),-.2/np.sqrt(1.04),0.))
    point=np.array([[.98,centers[1][5],centers[2][5]]])
    row=prepare_neumann_point_rows(context,point,
        normal_coefficients=lambda q:np.broadcast_to(coeff,(len(q),3)))[0]
    donor_points=context.pts[row.donor_ids]
    field=1+.3*donor_points[:,0]+.2*np.sin(donor_points[:,1])+.1*donor_points[:,2]
    g=coeff[0]*.3+coeff[1]*.2*np.cos(row.boundary_points[:,1])
    value=row.value@field+row.boundary_value@g
    gradient=row.gradient@field+row.boundary_gradient@g
    expected=1+.3*point[0,0]+.2*np.sin(point[0,1])+.1*point[0,2]
    np.testing.assert_allclose(value,expected,atol=2e-12)
    np.testing.assert_allclose(gradient,[.3,.2*np.cos(point[0,1]),.1],atol=2e-11)
    anchored=prepare_neumann_point_rows(context,point,
        normal_coefficients=lambda q:np.broadcast_to(coeff,(len(q),3)),
        support_anchors=np.array([[centers[1][4],centers[2][4]]]))[0]
    anchored_points=context.pts[anchored.donor_ids]
    anchored_field=1+.3*anchored_points[:,0]+.2*np.sin(anchored_points[:,1])+.1*anchored_points[:,2]
    anchored_g=coeff[0]*.3+coeff[1]*.2*np.cos(anchored.boundary_points[:,1])
    np.testing.assert_allclose(anchored.value@anchored_field+anchored.boundary_value@anchored_g,
                               expected,atol=2e-12)
    assert row.condition<1e3 and row.constraint_residual<1e-10
    cache={}
    same_patch=np.array([point[0],point[0]+[0.,.01,.01]])
    anchors=np.repeat(point[:,1:],2,axis=0)
    uncached=prepare_neumann_point_rows(context,same_patch,
        normal_coefficients=lambda q:np.broadcast_to(coeff,(len(q),3)),support_anchors=anchors)
    cached=prepare_neumann_point_rows(context,same_patch,
        normal_coefficients=lambda q:np.broadcast_to(coeff,(len(q),3)),support_anchors=anchors,patch_cache=cache)
    assert len(cache)==1
    for left,right in zip(uncached,cached):
        np.testing.assert_array_equal(left.donor_ids,right.donor_ids)
        np.testing.assert_allclose(left.gradient,right.gradient,rtol=0,atol=0)
        np.testing.assert_allclose(left.boundary_gradient,right.boundary_gradient,rtol=0,atol=0)
    payload,queries=lower_neumann_point_rows((row,))
    owner=np.zeros((n**3,1));owner[row.donor_ids,0]=field
    boundary=(coeff[0]*.3+coeff[1]*.2*np.cos(queries[:,1]))[:,None]
    compiled=jax.jit(lambda f,b:apply_neumann_point_rows(payload,f,b))
    actual=compiled(jnp.asarray(owner),jnp.asarray(boundary))
    np.testing.assert_allclose(np.asarray(actual[0])[0,0],expected,atol=2e-12)
    tangent=jnp.ones_like(boundary)
    _,linear=jax.jvp(lambda b:compiled(jnp.asarray(owner),b)[0],
                     (jnp.asarray(boundary),),(tangent,))
    np.testing.assert_allclose(np.asarray(linear)[0,0],row.boundary_value.sum(),atol=2e-12)
