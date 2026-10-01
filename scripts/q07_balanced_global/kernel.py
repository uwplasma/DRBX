"""Paired balanced h/32 versus h/128, all five fields, unchanged frozen rows."""
from types import SimpleNamespace
import numpy as np
import jax
import fields as m
from span_rows import reconstruct
from drbx.stencils.q_parallel_support import Hybrid,BalancedHybrid,LocalHybrid
from drbx.native.q_parallel_material import material_from_slots

FIELDS=('n','Te','Ti','Vi','Ve')
TERMS=[f'h{div}_{action}_{f}' for div in (32,128) for action in ('centered','correction','combined') for f in FIELDS]
MAX_RAW=128

@jax.jit
def material(v,p,L,b,delta):
    return material_from_slots(v,p,L,b,delta,tau=m.TAU,mu=m.MU)

def action(slots,phi,L,beta,delta):
    nr=slots.shape[1]
    if not 0<nr<=MAX_RAW:raise ValueError('raw batch bound')
    ix=np.r_[np.arange(nr),np.full(MAX_RAW-nr,nr-1)]
    a=material(slots[:,ix],phi[:,ix],L[ix][None],beta[ix][None],delta)
    if not np.asarray(a.inputs_finite).all() or not np.asarray(a.thermodynamic_states_positive).all():raise ValueError('invalid material state')
    if not np.asarray(a.eigensystem_admissible).all():raise ValueError('inadmissible characteristic split')
    return np.stack([np.asarray(a[i])[:,:nr] for i in (2,3,4)])

def center_reference(points,geom):
    J0,b0,B0=geom(points);values=[]
    for step in (1e-4,5e-5,2.5e-5):
        p=np.broadcast_to(points[:,None,None,:],(len(points),3,4,3)).copy()
        for d in range(3):p[:,d,:,d]+=np.array([-2,-1,1,2])*step
        J,b,_=geom(p.reshape(-1,3));J=J.reshape(-1,3,4);b=b.reshape(-1,3,4,3);w=np.array([1,-8,8,-1])/(12*step)
        values.append(sum(np.einsum('rq,q->r',J[:,d]*b[:,d,:,d],w) for d in range(3))/J0)
    q=SimpleNamespace(slot_points=np.stack([points]*3,axis=1),magnetic_b=np.stack([b0]*3,axis=1))
    R=m.continuum(q,values[0])[0]
    sensitivity=max(float(abs(m.continuum(q,k)[0]-R).max()) for k in values[1:])
    return b0,B0,values[0],R,sensitivity

def evaluate(t,owners,raw,ends32,ends128,choices,state,phi,geom,jac):
    points=t.pts[raw];nr=len(raw)
    if not np.array_equal(raw,np.concatenate([t.order[t.starts[o]:t.starts[o+1]] for o in owners])):raise ValueError('complete owner order')
    b,B,kappa,ref,sensitivity=center_reference(points,geom);beta=b[:,2]
    pp=[np.stack((e[:,0],e[:,2],points,e[:,3],e[:,1]),axis=1) for e in (ends32,ends128)]
    if any(np.any(p[:,:,0]>1) or not np.isfinite(p).all() for p in pp):raise ValueError('unqualified endpoint')
    objects=[Hybrid(t),BalancedHybrid(t),LocalHybrid(t)];wall_cache={};recs=[];amp=np.zeros((2,2));minimum=np.inf
    for ri,rr in enumerate(raw):
        p=np.concatenate([x[ri] for x in pp]);D,N,pD,pN,row=reconstruct(t,objects,wall_cache,int(rr),p,state.transpose(2,0,1),phi.T,int(choices[ri]),jac)
        recs.append((D,N,pD,pN));minimum=min(minimum,float(D[...,:3].min()),float(N[...,:3].min()))
        for ai,div in enumerate((32,128)):
            delta=2*np.pi/t.n/(2*div)
            for ki,key in enumerate(('VD','VN')):
                v=row[key][ai*5:(ai+1)*5]
                amp[ai,ki]=max(amp[ai,ki],float(abs((v[3]-v[1])*beta[ri]/(2*delta)).sum()))
    rec=[np.stack([r[i] for r in recs],axis=1) for i in range(4)]
    numerical=[];oracles=[];references=[]
    for ai,div in enumerate((32,128)):
        p=pp[ai];delta=2*np.pi/t.n/(2*div);gb=geom(p[:,[1,3,2]].reshape(-1,3));mag=gb[2].reshape(nr,3)
        if np.max(abs(gb[1].reshape(nr,3,3)[:,2]-b))>1e-10:raise ValueError("center-b replay")
        lm=-B*beta/(2*delta*mag[:,0]);lp=B*beta/(2*delta*mag[:,1]);L=np.stack((lm,lp,kappa-lm-lp),axis=1)
        exact=m.fields(p)[0].transpose(2,0,1,3);ep=m.phi_fields(p[:,[1,3,2]])[0].transpose(2,0,1)
        oracle=action(exact,ep,L,beta,delta);oracles.append(oracle.transpose(2,1,0,3).reshape(nr,m.NF,15))
        references.append(np.stack((ref,ref*0,ref)).transpose(2,1,0,3).reshape(nr,m.NF,15))
        sl=slice(ai*5,(ai+1)*5);acts=[]
        for kinds,pk in m.KINDS:
            use=np.array(kinds)=='D';slots=np.where(use[None,None,None,:],rec[0][:,:,sl],rec[1][:,:,sl]);ph=(rec[2] if pk=='D' else rec[3])[:,:,sl][:,:,[1,3,2]]
            a=action(slots,ph,L,beta,delta);acts.append(a.transpose(2,1,0,3).reshape(nr,m.NF,15))
        numerical.append(np.stack(acts,axis=1))
    Nraw=np.concatenate(numerical,axis=-1);Oraw=np.concatenate(oracles,axis=-1);Rraw=np.concatenate(references,axis=-1)
    owner_index={int(o):i for i,o in enumerate(owners)};oi=np.array([owner_index[int(t.ro[r])] for r in raw]);weight=t.rv[raw]/t.vol[t.ro[raw]]
    def project(x):
        a=np.zeros((len(owners),)+x.shape[1:]);np.add.at(a,oi,weight.reshape((-1,)+(1,)*(x.ndim-1))*x);return a
    N,O,R=map(project,(Nraw,Oraw,Rraw));constant_by_span=[float(abs(N[:,:,0,sl]-O[:,None,0,sl]).max()) for sl in (slice(0,15),slice(15,30))];constant=max(constant_by_span)
    if np.any(np.asarray(constant_by_span)>np.array([1e-7,4e-7])):raise ValueError('constant N-O gate '+str(constant_by_span))
    if not all(np.isfinite(x).all() for x in (N,O,R)):raise ValueError('nonfinite action')
    return N,O,R,dict(constant_error=constant,constant_error_by_span=constant_by_span,minimum_thermodynamic_slot=minimum,fallback_rows=0,oracle_fallback_rows=0,reference_sensitivity=sensitivity,owner_gradient_l1=amp.tolist())
