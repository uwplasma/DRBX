"""Frozen actual-HSX projected fourth-order line derivative, preparation only."""
import math
import numpy as np
from scipy import sparse
from scripts.q_fci_return_campaign import numerics as q
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
E2=np.array([(a,d-a) for d in range(4) for a in range(d+1)])
@jax.jit
def advance(field,seeds,delta,steps):
 eta0=seeds[:,2]; state=jnp.stack((seeds[:,0]*jnp.cos(seeds[:,1]),seeds[:,0]*jnp.sin(seeds[:,1]),jnp.zeros(len(seeds))),axis=1)
 h=delta/steps; alive=jnp.ones(len(seeds),bool); bad=jnp.zeros(len(seeds),bool)
 def rhs(a,eta,active):
  r=jnp.linalg.norm(a[:,:2],axis=1); inside=jnp.isfinite(a).all(axis=1)&(r>0)&(r<1); use=active&inside
  p=jnp.stack((jnp.where(use,r,.5),jnp.where(use,jnp.mod(jnp.arctan2(a[:,1],a[:,0]),2*jnp.pi),0.),jnp.mod(eta,2*jnp.pi)),axis=1)
  B,Bmag=field(p); beta=B[:,2]/Bmag; badf=use&((beta<=1e-10)|(~jnp.isfinite(B).all(axis=1))|(~jnp.isfinite(Bmag)))
  safe=jnp.where(badf|(~use),1.,beta); b=B/Bmag[:,None]; c=jnp.cos(p[:,1]);s=jnp.sin(p[:,1])
  out=jnp.stack(((c*b[:,0]-r*s*b[:,1])/safe,(s*b[:,0]+r*c*b[:,1])/safe,1/safe),axis=1)
  return jnp.where((use&(~badf))[:,None],out,0.),use&(~badf),badf,jnp.where(use,beta,jnp.inf)
 def step(i,carry):
  a,live,error,bmin,bmax=carry;t=eta0+i*h
  k1,v1,e1,b1=rhs(a,t,live);k2,v2,e2,b2=rhs(a+.5*h[:,None]*k1,t+.5*h,v1)
  k3,v3,e3,b3=rhs(a+.5*h[:,None]*k2,t+.5*h,v2);k4,v4,e4,b4=rhs(a+h[:,None]*k3,t+h,v3)
  candidate=a+h[:,None]*(k1+2*k2+2*k3+k4)/6; rad2=jnp.sum(candidate[:,:2]**2,axis=1)
  valid=v4&jnp.isfinite(candidate).all(axis=1)&(rad2>0)&(rad2<1)
  return jnp.where(valid[:,None],candidate,a),valid,error|e1|e2|e3|e4,jnp.minimum(bmin,jnp.minimum(jnp.minimum(b1,b2),jnp.minimum(b3,b4))),jnp.maximum(bmax,jnp.maximum(jnp.maximum(b1,b2),jnp.maximum(b3,b4)))
 state,alive,bad,bmin,bmax=jax.lax.fori_loop(0,steps,step,(state,alive,bad,jnp.full(len(seeds),jnp.inf),jnp.zeros(len(seeds))))
 end=jnp.stack((jnp.linalg.norm(state[:,:2],axis=1),jnp.mod(jnp.arctan2(state[:,1],state[:,0]),2*jnp.pi),eta0+delta),axis=1)
 return end,jnp.abs(state[:,2]),alive,bad,bmin,bmax

def host_trace(ctx,seeds,delta,steps):
 s=np.column_stack([seeds[:,0]*np.cos(seeds[:,1]),seeds[:,0]*np.sin(seeds[:,1]),np.zeros(len(seeds))]);h=np.asarray(delta)/steps
 def rhs(a,eta):
  r=np.linalg.norm(a[:,:2],axis=1);assert np.all((r>0)&(r<1))
  p=np.column_stack([r,np.arctan2(a[:,1],a[:,0])%(2*np.pi),eta%(2*np.pi)]);_,b,_=q.base(ctx,p); assert np.all(b[:,2]>1e-10)
  cc=np.cos(p[:,1]);ss=np.sin(p[:,1]);return np.column_stack([(cc*b[:,0]-r*ss*b[:,1])/b[:,2],(ss*b[:,0]+r*cc*b[:,1])/b[:,2],1/b[:,2]])
 for i in range(steps):
  t=seeds[:,2]+i*h;k1=rhs(s,t);k2=rhs(s+.5*h[:,None]*k1,t+.5*h);k3=rhs(s+.5*h[:,None]*k2,t+.5*h);k4=rhs(s+h[:,None]*k3,t+h);s+=h[:,None]*(k1+2*k2+2*k3+k4)/6
 return np.column_stack([np.linalg.norm(s[:,:2],axis=1),np.arctan2(s[:,1],s[:,0])%(2*np.pi),seeds[:,2]+delta]),abs(s[:,2])

def project(ctx,keys,order,spacing=1.):
 pp,ww=q.quadrature(ctx,keys,order);points=pp.reshape(-1,3);position,A=ctx['evaluator']._position_and_jacobian(points);Bc=ctx['bfield'].evaluate_cartesian(position);B=np.linalg.norm(Bc,axis=1);J=np.abs(np.linalg.det(A));b=np.linalg.solve(A,Bc[...,None])[...,0]/B[:,None]
 J=J.reshape(pp.shape[:2]);b=b.reshape(*pp.shape[:2],3);A=A.reshape(*pp.shape[:2],3,3);weights=[];masses=[];proj=[];metrics=[];psis=[]
 for fi,key in enumerate(keys):
  a=int(key[0]);free=[j for j in range(3) if j!=a];grid=ctx['artifact'].geometry.grid;faces=[getattr(grid,k).faces for k in 'xyz'];center=np.array([.5*sum(faces[j][int(key[j+1]):int(key[j+1])+2]) for j in free]);scale=np.array([faces[j][int(key[j+1])+1]-faces[j][int(key[j+1])] for j in free]);uv=(pp[fi,:,free].T-center)/scale
  psi=np.prod(uv[:,None,:]**E2,axis=2);S=np.linalg.norm(np.cross(A[fi,:,:,free[0]],A[fi,:,:,free[1]]),axis=1);area=ww[fi]*S
  M=psi.T@(area[:,None]*psi);R=psi.T@(ww[fi]*J[fi]*b[fi,:,a]*b[fi,:,2]);sv=np.linalg.svd(M,compute_uv=False);assert sv[-1]>sv[0]*1e-13
  dual=np.linalg.solve(M,R);coef=area*(psi@dual)/(2*np.pi/ctx['N']*spacing)
  weights.append(coef);masses.append(M);proj.append(R);psis.append(psi);metrics.append({'rank':len(sv),'condition':float(sv[0]/sv[-1]),'projection_identity':float(np.max(abs(np.linalg.solve(M,psi.T@(area[:,None]*psi))-np.eye(10)))),'physical_area':float(area.sum()),'J_min':float(J[fi].min()),'b_eta_min':float(b[fi,:,2].min()),'endpoint_l1':float(2*sum(abs(coef))),'endpoint_l2':float(np.sqrt(2)*np.linalg.norm(coef))})
 return {'points':pp,'logical_weights':ww,'J':J,'b':b,'A':A,'weights':np.array(weights),'mass':np.array(masses),'R':np.array(proj),'psi':np.array(psis),'metadata':metrics}

def endpointmap(ctx,points):
 model=ctx['model'];n=ctx['N'];centers=np.asarray(model.grid.z.centers);delta=2*np.pi/n;rows=[];cols=[];vals=[];records=[]
 for pi,pt in enumerate(points):
  rr=(pt[2]-centers[0])/delta;near=round(rr)
  if abs(rr-near)*delta<2e-10:planes=np.array([near]);lw=np.array([1.])
  else:
   planes=math.floor(rr)+np.arange(-1,3);lw=np.array([np.prod([(rr-b)/(a-b) for b in planes if b!=a]) for a in planes])
  ids=[];cc=[];dmax=0.;defect=0.;condition=0.;sizes=[]
  for pl,ww in zip(planes,lw):
   test=pt.copy();test[2]=centers[int(pl)%n];donor,c,_,m=model.endpoint_pair(test,include_control=False);ids.extend(donor);cc.extend(ww*c);dmax=max(dmax,m['chosen']['max_scaled_distance']);defect=max(defect,m['chosen']['residual']);condition=max(condition,m['chosen']['condition']);sizes.append(len(donor))
  ids=np.array(ids);cc=np.array(cc);rows.extend([pi]*len(ids));cols.extend(ids);vals.extend(cc)
  records.append({'l1':float(sum(abs(cc))),'l2':float(np.linalg.norm(cc)),'plane_count':len(planes),'plane_indices':(planes%n).tolist(),'unwrapped_plane_indices':planes.tolist(),'eta_weights':lw.tolist(),'transverse_scaled_extent':dmax,'fit_residual':defect,'condition':condition,'donor_count':len(ids),'plane_donor_counts':sizes})
 return sparse.csr_matrix((vals,(rows,cols)),shape=(len(points),len(ctx['volume']))),records


def field_values_gradients(points):
    """Frozen four fields, sharing intermediates and omitting unused Hessians."""
    r,t,e=np.asarray(points).T;x=r*np.cos(t);y=r*np.sin(t);a=r*r;s=(1-a)**4
    v=np.column_stack((1+.2*a*s*np.sin(e),1+.2*x*s,1+.2*y*s*np.sin(e),np.full(len(r),1.2)))
    gx=np.zeros_like(v);gy=np.zeros_like(v);ge=np.zeros_like(v)
    gx[:,0]=2*x*(s-4*a*(1-a)**3)*np.sin(e);gy[:,0]=2*y*(s-4*a*(1-a)**3)*np.sin(e);ge[:,0]=a*s*np.cos(e)
    gx[:,1]=s-8*x*x*(1-a)**3;gy[:,1]=-8*x*y*(1-a)**3
    gx[:,2]=-8*x*y*(1-a)**3*np.sin(e);gy[:,2]=(s-8*y*y*(1-a)**3)*np.sin(e);ge[:,2]=y*s*np.cos(e)
    grad=.2*np.stack((gx*np.cos(t[:,None])+gy*np.sin(t[:,None]),r[:,None]*(-gx*np.sin(t[:,None])+gy*np.cos(t[:,None])),ge),axis=-1)
    return v,grad


def exact_flux(ctx,keys,order):
    p,w=q.quadrature(ctx,keys,order);shape=w.shape
    J,b,_=q.base(ctx,p.reshape(-1,3));g=field_values_gradients(p.reshape(-1,3))[1]
    b=b.reshape(*shape,3);grad=np.einsum('nqi,nqfi->nqf',b,g.reshape(*shape,4,3))
    normal=b[np.arange(len(keys))[:,None],np.arange(shape[1])[None,:],np.asarray(keys)[:,0,None]]
    return np.sum((w*J.reshape(shape)*normal)[:,:,None]*grad,axis=1)


def trace_batch(ctx,seeds,delta,steps=64):
    """Independent eta starts and steps; fixed padded batch, one compilation shape."""
    out=[[] for _ in range(6)];capacity=ctx['config']['policy']['trace_batch'];executed=0
    for lo in range(0,len(seeds),capacity):
        ss=seeds[lo:lo+capacity];dd=np.asarray(delta)[lo:lo+capacity];count=len(ss)
        if count<capacity:
            ss=np.vstack((ss,np.repeat(ss[-1:],capacity-count,axis=0)))
            dd=np.r_[dd,np.repeat(dd[-1],capacity-count)]
        rr=advance(ctx['rk_field'],jnp.asarray(ss),jnp.asarray(dd),jnp.asarray(steps));executed+=capacity
        for j,x in enumerate(rr):out[j].append(np.asarray(x)[:count])
    return [np.concatenate(a) for a in out],executed


def pack_csr(name,m):
    m=m.tocsr();m.sum_duplicates();m.sort_indices()
    return {name+'_data':m.data,name+'_indices':m.indices,name+'_indptr':m.indptr,name+'_shape':np.array(m.shape)}


def unpack_csr(name,z):
    return sparse.csr_matrix((z[name+'_data'],z[name+'_indices'],z[name+'_indptr']),shape=tuple(z[name+'_shape']))


def resolve_domain_legs(ctx,seeds,delta,trace_function=None,max_halvings=0):
    """Geometry-only interval reduction; reuse half legs, never query outside."""
    run=trace_function or (lambda s,d,steps:trace_batch(ctx,s,d,steps))
    steps=ctx['config']['policy']['trace_substeps']
    traced,executed=run(seeds,delta,steps);count=len(seeds)//4
    if len(seeds)!=4*count:raise ValueError('four leg blocks required')
    arrays=[np.asarray(x).copy() for x in traced];levels=np.zeros(count,dtype=int)
    initial_bad=int(np.sum(~arrays[2]|arrays[3]));real=len(seeds)
    for level in range(max_halvings+1):
        if np.any(arrays[3]):raise ValueError('invalid field/b_eta during tracing; shortening forbidden')
        invalid=np.flatnonzero(~arrays[2].reshape(4,count).all(axis=0))
        if not len(invalid):break
        if level==max_halvings:break
        levels[invalid]+=1
        # Prior half endpoints become full endpoints at the new interval.
        for a in arrays:
            a[invalid]=a[2*count+invalid];a[count+invalid]=a[3*count+invalid]
        ss=np.vstack((seeds[invalid],seeds[count+invalid]))
        dd=np.r_[delta[invalid],delta[count+invalid]]*np.tile(2.**(-levels[invalid]-1),2)
        short,work=run(ss,dd,steps);executed+=work;real+=len(ss)
        for a,b in zip(arrays,short):
            a[2*count+invalid]=b[:len(invalid)];a[3*count+invalid]=b[len(invalid):]
    return arrays,executed,real,levels,initial_bad


def prepare_faces(ctx,ids,order=5,trace_function=None,max_halvings=0):
    """Field-blind maps first; wall flux supplied separately, never traced."""
    import hashlib
    keys=ctx['keys'][ids];interior=np.flatnonzero(~((keys[:,0]==0)&(keys[:,1]==ctx['N'])))
    data={'ids':np.asarray(ids),'keys':keys,'interior_slots':interior,'construction_order':order}
    if not len(interior):
        data.update(donors=np.empty(0,int),trace_real_legs=0,trace_executed_legs=0,trace_valid=np.empty(0,bool),span_levels=np.empty(0,int),span_scale=np.empty(0),initial_invalid_legs=0)
        for m in ('full','half'):data.update(pack_csr(m,sparse.csr_matrix((len(ids),0))))
        data['map_digest']=np.array(hashlib.sha256(b'boundary').hexdigest());return data
    k=keys[interior];pr=project(ctx,k,order);pp=pr['points'];nf=len(k);nq=order**2;H=2*np.pi/ctx['N']
    seeds=np.tile(pp.reshape(-1,3),(4,1));delta=np.repeat(H*np.array([-.5,.5,-.25,.25]),nf*nq)
    traced,executed,real,levels,initial_bad=resolve_domain_legs(ctx,seeds,delta,trace_function,max_halvings)
    end,length,valid,bad,bmin,bmax=traced
    if not np.all(valid&~bad):
        wrong=np.flatnonzero(~valid|bad);examples=[{'face_key':k[(int(i)%(nf*nq))//nq].tolist(),'seed':seeds[i].tolist(),'delta':float(delta[i]),'bad_field':bool(bad[i])} for i in wrong[:12]]
        raise ValueError(f'unsupported interior traced leg(s): {len(wrong)}/{len(end)}; no fallback; examples={examples}')
    maps={};pointfaces=np.tile(np.repeat(interior,nq),2);scale=2.**(-levels);pw=np.r_[-pr['weights'].ravel()/scale,pr['weights'].ravel()/scale]
    E=sparse.csr_matrix((pw,(pointfaces,np.arange(2*nf*nq))),shape=(len(ids),2*nf*nq))
    metas=[]
    for m,sl,mul in (('full',slice(0,2*nf*nq),1),('half',slice(2*nf*nq,None),2)):
        G,meta=endpointmap(ctx,end[sl]);maps[m]=mul*(E@G);metas.extend(meta)
    donors=np.union1d(maps['full'].indices,maps['half'].indices)
    for m,F in maps.items():data.update(pack_csr(m,F[:,donors]))
    # Hash geometry-selected coefficients before applying any field observations.
    hh=hashlib.sha256()
    for name in ('full','half'):
        for suff in ('data','indices','indptr','shape'):hh.update(np.ascontiguousarray(data[name+'_'+suff]).tobytes())
    hh.update(donors.tobytes());data['map_digest']=np.array(hh.hexdigest())
    data.update(donors=donors,endpoints=end,span_levels=levels,span_scale=scale,initial_invalid_legs=initial_bad,trace_lengths=length,trace_valid=valid,trace_bad_field=bad,trace_beta_min=bmin,trace_beta_max=bmax,trace_real_legs=real,trace_executed_legs=executed,point_weights=pw,point_faces=pointfaces,points=pp,logical_weights=pr['logical_weights'],J=pr['J'],b=pr['b'],projection_weights=pr['weights'],projection_condition=np.array([x['condition'] for x in pr['metadata']]),endpoint_condition_max=max(x['condition'] for x in metas),endpoint_residual_max=max(x['fit_residual'] for x in metas),endpoint_l1_max=max(x['l1'] for x in metas),endpoint_max_scaled_extent=max(x['transverse_scaled_extent'] for x in metas),endpoint_plane_counts=np.bincount([x['plane_count'] for x in metas],minlength=5))
    return data


def evaluate_faces(ctx,data):
    """Only after field-blind map preparation. All channels share the same maps."""
    nf=len(data['ids']);out={};idx=data['interior_slots'];F={m:unpack_csr(m,data) for m in ('full','half')}
    for m in F:out[m+'_numerical']=F[m]@ctx['state'][data['donors']]
    for m,mul in (('full',1),('half',2)):
        flux=np.zeros((nf,4))
        if len(idx):
            count=len(data['endpoints'])//2;ep=data['endpoints'][:count] if m=='full' else data['endpoints'][count:]
            np.add.at(flux,data['point_faces'],mul*data['point_weights'][:,None]*field_values_gradients(ep)[0])
        out[m+'_exact']=flux
    for ch in ('numerical','exact'):out['fourth_'+ch]=(4*out['half_'+ch]-out['full_'+ch])/3
    inj=np.zeros((nf,4));oracle5=np.zeros_like(inj)
    if len(idx):
        shape=data['points'].shape[:2];grad=field_values_gradients(data['points'].reshape(-1,3))[1].reshape(*shape,4,3)
        bg=np.einsum('fqa,fqka->fqk',data['b'],grad);deta=bg/data['b'][:,:,2,None]
        inj[idx]=np.einsum('fq,fqk->fk',data['projection_weights']*(2*np.pi/ctx['N']),deta)
        axis=data['keys'][idx,0];normal=data['b'][np.arange(len(idx))[:,None],np.arange(shape[1])[None,:],axis[:,None]]
        oracle5[idx]=np.einsum('fq,fqk->fk',data['logical_weights']*data['J']*normal,bg)
    out['projection']=inj;out['construction_reference']=oracle5
    out['q5']=oracle5 if int(data['construction_order'])==5 else np.zeros_like(inj)
    if int(data['construction_order'])!=5 and len(idx):out['q5'][idx]=exact_flux(ctx,data['keys'][idx],5)
    for order in (9,11):
        a=np.zeros((nf,4))
        if len(idx):a[idx]=exact_flux(ctx,data['keys'][idx],order)
        out[f'q{order}']=a
    fourth=(4*F['half']-F['full'])/3
    for m,mat in (('full',F['full']),('half',F['half']),('fourth',fourth)):
        out[m+'_face_l1']=np.asarray(abs(mat).sum(axis=1)).ravel()
        out[m+'_face_l2']=np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    return out
