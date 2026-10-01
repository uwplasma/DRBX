"""Paired old/balanced material actions. No change to the characteristic correction."""
import numpy as np
import fields as m
from drbx.native.q_parallel import QBoundaryData
from drbx.native.q_parallel_material import material_from_slots,reconstruct_material_slots,stage_material_transport
from drbx.native.q_parallel_divergence import apply_raw_scalar_slots
from drbx.stencils.q_parallel_material import prepare_material_transport
from drbx.stencils.q_parallel_divergence import prepare_kappa
TERMS=[f'{method}_{action}_{field}' for method in ('original','balanced') for action in ('centered','correction','combined') for field in ('n','Te','Ti')]

def slice_bc(bc,sl):return QBoundaryData(*(x[sl] for x in bc))

def paired(stencil,phi,L,beta,delta,kappa):
    a=material_from_slots(stencil,phi,L,beta,delta,tau=m.TAU,mu=m.MU)
    if not np.all(a.inputs_finite) or not np.all(a.thermodynamic_states_positive):raise ValueError('invalid material state')
    center=np.asarray(a.centered)[...,:3];corr=np.asarray(a.correction)[...,:3]
    n,te,ti,vi,ve=np.moveaxis(stencil[:,:,2],-1,0)
    defect=kappa-L.sum(axis=-1)
    change=defect[None,:,None]*np.stack((-n*ve,2*te/3*(.71*(vi-ve)-ve),2*ti/3*((vi-ve)-vi)),axis=-1)
    balanced=center+change
    return np.concatenate((center,corr,center+corr,balanced,corr,balanced+corr),axis=-1),int(np.count_nonzero(~np.asarray(a.eigensystem_admissible)))

def evaluate(outer,inner,state,phi,geom):
    rt=stage_material_transport(prepare_material_transport(inner,outer));k=prepare_kappa(inner,geom,center_b_atol=1e-10)
    L=inner.magnetic_L;beta=inner.magnetic_b[:,2,2];delta=float(rt.eta_step)
    exact=m.stack(*(m.fields(q.slot_points)[0].transpose(2,0,1,3) for q in (outer,inner)))
    ep=m.phi_fields(inner.slot_points)[0].transpose(2,0,1)
    Oraw,of=paired(exact,ep,L,beta,delta,k.kappa)
    r=m.continuum(inner,k.kappa)[0][...,:3]
    Rraw=np.concatenate((r,r*0,r,r,r*0,r),axis=-1)
    sensitivity=max(float(abs(m.continuum(inner,kk)[0][...,:3]-r).max()) for kk in k.step_values[1:])
    project=lambda x: m.project(inner,x).transpose(1,0,2)
    O,R=project(Oraw),project(Rraw)
    bo,bi=m.boundary(outer),m.boundary(inner);pb=m.boundary(inner,m.phi_fields,False)
    N=np.zeros((len(inner.owners),4,m.NF,len(TERMS)));fallback=0;minimum=np.inf
    for start in range(0,m.NF,3):
        sl=slice(start,start+3)
        for ki,(kinds,pk) in enumerate(m.KINDS):
            zi=np.asarray(reconstruct_material_slots(rt.inner,state[sl],slice_bc(bi,sl),kinds=kinds))
            zo=np.asarray(reconstruct_material_slots(rt.outer,state[sl],slice_bc(bo,sl),kinds=kinds))
            slots=m.stack(zo,zi);ps=np.asarray(apply_raw_scalar_slots(rt.inner,phi[sl],slice_bc(pb,sl),kind=pk))
            raw,f=paired(slots,ps,L,beta,delta,k.kappa);fallback+=f;minimum=min(minimum,float(slots[...,:3].min()))
            N[:,ki,sl]=project(raw)
    if not all(np.isfinite(x).all() for x in (N,O,R)):raise ValueError('nonfinite action')
    constant=float(abs(N[:,:,0]-O[:,None,0]).max())
    if constant>1e-7:raise ValueError('constant N-O gate '+str(constant))
    if not np.array_equal(N[...,3:6],N[...,12:15]):raise ValueError('correction changed')
    return N,O,R,dict(constant_error=constant,minimum_thermodynamic_slot=minimum,fallback_rows=fallback,oracle_fallback_rows=of,reference_sensitivity=sensitivity,kappa_sensitivity=float(k.step_sensitivity.max()))
