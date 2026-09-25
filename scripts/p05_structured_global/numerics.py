"""Shared-structured P05 bracket kernels; all observations/BCs are external RHS data."""
import numpy as np
from perpendicular_structured.reconstruction import StructuredReconstruction, load_context
from perpendicular_structured.reference_geometry import reuse_metrics
from p07_combined_global.kernels import num
from p07_combined_global import topology

FIELDS=('phi_mms','actual_vorticity','smooth_regular','smooth_eta','zero_dirichlet','varying_dirichlet','constant','varying_generator')
PAIRS=((0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(7,4),(7,5))
CASES=('mms_omega','mms_regular','mms_eta','mms_zero_dirichlet','mms_varying_dirichlet','constant_control','varying_generator_zero','varying_generator_varying')
TIME=1e-6


def h_vector(ref,p):
    m=ref._metric(p)
    return m['bcov']/m['B'][:,None]


def curl_h(ref,p,step=2e-4):
    ds=[]
    for axis in range(3):
        h=step
        if axis==0:
            distance=np.minimum(p[:,0],1-p[:,0])
            if np.any(distance<=0):raise ValueError('curl reference requires interior radial points')
            # Only repair stencils that would leave the geometry domain. q3/q5
            # on the canonical N32/48/64 grids retain the original fixed step.
            h=np.where(distance<2*step,.2*distance,step)
        values=[]
        for factor in (-2,-1,1,2):
            q=p.copy();q[:,axis]+=factor*h;values.append(h_vector(ref,q))
        denominator=12*h if np.ndim(h)==0 else 12*h[:,None]
        ds.append((values[0]-8*values[1]+8*values[2]-values[3])/denominator)
    return np.stack((ds[1][:,2]-ds[2][:,1],ds[2][:,0]-ds[0][:,2],ds[0][:,1]-ds[1][:,0]),axis=1)


def omega(ref,p):
    q=p.copy();wall=q[:,0]>=1-1e-14
    q[wall,0]=1-3*ref.finite_difference_step
    a=ref._psi_raw(q)
    return ref._perpendicular_operator(q,np.stack(a[1:4],axis=-1),a[5])


def fields(ref,p,*,omega_gradient=False):
    """Frozen analytic catalogue; omega gradient is needed only by BC trace lift."""
    p=np.asarray(p);rr,th,et=p.T;k=2*np.pi/ref.eta_period;a=1-rr**2
    phi=ref._fields_raw(p,TIME)['phi']
    values=[phi[0],omega(ref,p)];grads=[np.stack(phi[1:4],axis=-1),np.zeros((len(p),3))]
    if omega_gradient:
        # Only tangential BC derivatives are consumed; normal omega is not differentiated.
        for ax in (1,2):
            z=[]
            for mul in (-2,-1,1,2):
                q=p.copy();q[:,ax]+=mul*ref.finite_difference_step;z.append(omega(ref,q))
            grads[1][:,ax]=(z[0]-8*z[1]+8*z[2]-z[3])/(12*ref.finite_difference_step)
    phase=2*th-k*et+.23;env=rr**2*a**4
    values.append(.08*env*np.cos(phase))
    grads.append(np.stack((.16*rr*a**3*(1-5*rr**2)*np.cos(phase),-.16*env*np.sin(phase),.08*k*env*np.sin(phase)),axis=1))
    phase=k*et+.31
    values.append(.08*a**4*np.sin(phase))
    grads.append(np.stack((-.64*rr*a**3*np.sin(phase),np.zeros(len(p)),.08*k*a**4*np.cos(phase)),axis=1))
    # Regular-chart fields with nonzero wall normal slope; both zero and varying trace.
    for phase0,zero in ((.27,True),(.61,False)):
        phase=th-k*et+phase0;c=np.cos(phase);s=np.sin(phase)
        radial=rr*a if zero else rr*(1+.2*rr**2)
        dr=1-3*rr**2 if zero else 1+.6*rr**2
        values.append(.1*radial*c)
        grads.append(.1*np.stack((dr*c,-radial*s,k*radial*s),axis=1))
    values.append(np.ones(len(p)));grads.append(np.zeros((len(p),3)))
    phase=2*th+k*et+.17;radial=rr**2*(1+.15*rr**2)
    values.append(.08*radial*np.cos(phase))
    grads.append(.08*np.stack(((2*rr+.6*rr**3)*np.cos(phase),-2*radial*np.sin(phase),-k*radial*np.sin(phase)),axis=1))
    return np.stack(values,axis=1),np.stack(grads,axis=2)


def boundary_trace(ref,p):
    """Bounded point cache shares prescribed traces across common/side/cell rows."""
    if not hasattr(ref,'_p05_trace_cache'):ref._p05_trace_cache={}
    cache=ref._p05_trace_cache
    if len(cache)>32768:cache.clear()
    keys=[(float(ref.finite_difference_step),row.tobytes()) for row in np.asarray(p)]
    missing={key:row for key,row in zip(keys,p) if key not in cache}
    if missing:
        v,g=fields(ref,np.asarray(list(missing.values())),omega_gradient=True)
        cache.update((key,(v[j],g[j])) for j,key in enumerate(missing))
    return np.stack([cache[key][0] for key in keys]),np.stack([cache[key][1] for key in keys])


def observations(t,ref,ids):
    v,_=fields(ref,t.pts[ids])
    return dict(ids=ids,owners=t.ro[ids],numerator=t.rv[ids,None]*v)


def flux_arrays(h,w,v,g,axis):
    # U is the actual bracket advection velocity density (rho*=1).
    U=-np.cross(h[:,None,:],g.transpose(0,2,1))[:,:,axis]
    generators=np.sum(w[:,None]*U,axis=0)
    products=np.array([[np.dot(w,U[:,a]*v[:,b]),np.dot(w,U[:,b]*v[:,a])] for a,b in PAIRS])
    return generators,products,U


@reuse_metrics
def face_chunk(t,ref,S,owner_values,ids,*,order=3,candidate=True):
    keys=topology.decode(t.n,ids);p,w=num.quadrature(t.faces,keys,order,face=True)
    F=len(FIELDS);P=len(PAIRS);nf=len(ids)
    out=dict(ids=ids,generator=np.zeros((nf,F)),product=np.zeros((nf,P,2)),oracle_generator=np.zeros((nf,F)),oracle_product=np.zeros((nf,P,2)),upwind=np.zeros((nf,P)),side_jump_max=np.zeros(nf),constant_error=np.zeros(nf),support_residual=np.zeros(nf),donor_count=np.zeros(nf,int))
    active=np.flatnonzero(~((keys[:,0]==0)&(keys[:,1]==0)))
    if not len(active):return out
    flat=p[active].reshape(-1,3);hv=h_vector(ref,flat).reshape(len(active),order**2,3)
    av,ag=fields(ref,flat)
    # Oracle only needs omega values: the omega primitive A is the IBP reference.
    av=av.reshape(len(active),order**2,F);ag=ag.reshape(len(active),order**2,3,F)
    def trace(q):return boundary_trace(ref,q)
    for ii,j in enumerate(active):
        axis=int(keys[j,0]);gg,pp,_=flux_arrays(hv[ii],w[j],av[ii],ag[ii],axis)
        out['oracle_generator'][j]=gg;out['oracle_product'][j]=pp
        if not candidate:continue
        row=S.rows(keys[j],p[j]);v,g=row.apply(owner_values,trace)
        gg,pp,U=flux_arrays(hv[ii],w[j],v,g,axis)
        out['generator'][j]=gg;out['product'][j]=pp
        left,right=S.side_rows(keys[j],p[j]);lv=trace(p[j])[0] if left is None else left.apply_value(owner_values,trace);rv=trace(p[j])[0] if right is None else right.apply_value(owner_values,trace)
        # Preserve the common face value: re-center left/right around it; only jump enters.
        jump=rv-lv
        for k,(a,b) in enumerate(PAIRS):out['upwind'][j,k]=-.5*np.dot(w[j],abs(U[:,a])*jump[:,b])
        out['side_jump_max'][j]=abs(jump).max()
        out['constant_error'][j]=max(abs(v[:,6]-1).max(),abs(g[:,:,6]).max(),abs(jump[:,6]).max())
        out['support_residual'][j]=max(r.diagnostics['max_residual'] for r in (row,left,right) if r is not None)
        out['donor_count'][j]=len(row.donor_ids)
    return out


@reuse_metrics
def cell_chunk(t,ref,S,owner_values,ids,*,order=3,candidate=True,step=2e-4):
    keys=np.array(np.unravel_index(ids,(t.n,)*3)).T;p,w=num.quadrature(t.faces,keys,order,face=False)
    flat=p.reshape(-1,3);F=len(FIELDS);P=len(PAIRS);nq=order**3
    ch=curl_h(ref,flat,step).reshape(len(ids),nq,3)
    av,ag=fields(ref,flat);av=av.reshape(len(ids),nq,F);ag=ag.reshape(len(ids),nq,3,F)
    metric=ref._metric(flat)
    J=abs(metric['J']).reshape(len(ids),nq)
    out=dict(ids=ids,owners=t.ro[ids],correction=np.zeros((len(ids),P,2)),oracle_correction=np.zeros((len(ids),P,2)),direct=np.zeros((len(ids),P)),volume=np.sum(w*J,axis=1),constant_error=np.zeros(len(ids)),support_residual=np.zeros(len(ids)))
    h=(metric['bcov']/metric['B'][:,None]).reshape(len(ids),nq,3)
    for j,key in enumerate(keys):
        anchor=owner_values[t.ro[ids[j]]] if owner_values is not None else np.zeros(F)
        div=-np.einsum('qa,qaf->qf',ch[j],ag[j])
        for k,(a,b) in enumerate(PAIRS):
            out['oracle_correction'][j,k]=[np.dot(w[j],(av[j,:,b]-anchor[b])*div[:,a]),np.dot(w[j],(av[j,:,a]-anchor[a])*div[:,b])]
            out['direct'][j,k]=np.dot(w[j],np.einsum('qa,qa->q',-np.cross(h[j],ag[j,:,:,a]),ag[j,:,:,b]))
        if candidate:
            rows=S.rows(key,p[j],location='cell');v,g=rows.apply(owner_values,lambda q:boundary_trace(ref,q))
            div=-np.einsum('qa,qaf->qf',ch[j],g)
            for k,(a,b) in enumerate(PAIRS):out['correction'][j,k]=[np.dot(w[j],(v[:,b]-anchor[b])*div[:,a]),np.dot(w[j],(v[:,a]-anchor[a])*div[:,b])]
            out['constant_error'][j]=max(abs(v[:,6]-1).max(),abs(g[:,:,6]).max())
            out['support_residual'][j]=rows.diagnostics['max_residual']
    return out
