"""Compose frozen endpoint maps into separate compact owner and BC actions.

Research extraction of the Q optimization audit's map composition. Both outer
spans and all directions remain. A row acts on (donor value - target value);
the boundary row acts on imposed D value minus target value, or physical N data.
"""
import numpy as np
from scipy import sparse
from scripts.q_fci_return_campaign import numerics as qnum

def endpoint_action(J,b,Jmid,seeds,values,H):
    v=np.asarray(values).reshape(4,12);out=np.zeros((2,3),dtype=v.dtype)
    for conf in range(2):
        for axis in range(3):
            neg,pos=6*conf+2*axis,6*conf+2*axis+1
            d=abs(seeds[pos,axis]-seeds[neg,axis])/2
            for sid,sign in ((neg,-1),(pos,1)):
                z=sign*J[sid]*b[sid,axis]*b[sid,2]/(2*d*Jmid)
                out[conf,axis]+=z*(4*(v[3,sid]-v[2,sid])/(H/8)-(v[1,sid]-v[0,sid])/(H/4))/3
    return out

def endpoint_operator(ctx,seeds,midpoint,interior):
    J,b,_=qnum.base(ctx,np.asarray(seeds));Jmid=qnum.base(ctx,np.asarray(midpoint)[None])[0][0];H=2*np.pi/ctx['N']
    eye=np.eye(48)
    K=np.column_stack([endpoint_action(J,b,Jmid,seeds,eye[:,j],H).ravel() for j in range(48)])
    # Frozen inherited interior seed ordering is h/4 followed by h/8.
    if interior:K=K[[3,4,5,0,1,2]]
    return K,J,b,Jmid

def unpack_interior(z):
    return sparse.csr_matrix((z['endpoint_data'],z['endpoint_indices'],z['endpoint_indptr']),shape=tuple(z['endpoint_shape']))

def compose(ctx,seeds,midpoint,map_data,kind,owner):
    interior=kind=='interior';K,J,b,Jmid=endpoint_operator(ctx,seeds,midpoint,interior)
    if interior:
        M=map_data if sparse.issparse(map_data) else unpack_interior(map_data)
        ids=np.unique(M.indices);A=K@M[:,ids].toarray();B=np.zeros((6,0));nodes=np.empty((0,3))
    else:
        ids=np.asarray(map_data['donors'],int);A=K@map_data['Eowner'];B=K@map_data['Ebc'];nodes=map_data['wall_nodes']
    return dict(donors=ids.astype(np.int32),owner=int(owner),A=A,B=B,wall_nodes=nodes,K=K,J=J,b=b,Jmid=Jmid)

def apply(compact,donor_values,target_value,boundary_values,kind):
    rhs=boundary_values-target_value if kind=='D' else boundary_values
    return (compact['A']@(donor_values-target_value)+compact['B']@rhs).reshape(2,3)

def normal_contravariant(ctx,p):
    A=ctx['evaluator']._position_and_jacobian(np.asarray(p))[1]
    inv=np.linalg.inv(A);g=np.einsum('nik,njk->nij',inv,inv)
    return g[:,0,:]/np.sqrt(g[:,0,0])[:,None]

def raw_owner_states(ctx,names,field_callable):
    n=ctx['N'];grid=ctx['artifact'].geometry.grid;labels=np.asarray(ctx['topology']['compact_raw_owner']).ravel();vol=np.asarray(ctx['artifact'].polar_angular_geometry.raw_volume).ravel();den=np.asarray(ctx['volume'])
    result={name:np.zeros(len(den),complex) for name in names}
    for lo in range(0,n**3,4096):
        raw=np.arange(lo,min(lo+4096,n**3));ijk=np.array(np.unravel_index(raw,(n,n,n))).T
        q=np.column_stack([getattr(grid,a).centers[ijk[:,i]] for i,a in enumerate('xyz')])
        for name in names:
            v=field_callable(name)(q)[0]
            np.add.at(result[name],labels[raw],vol[raw]*v)
    for name in names:result[name]/=den
    return result
