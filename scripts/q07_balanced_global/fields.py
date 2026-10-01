"""Frozen Q07 MMS catalogue and independent continuum first derivatives."""
import numpy as np
import q07_density_bounded as c
DESIGNS=[dict(name='constant'),dict(name='smooth')]
DESIGNS += [dict(name=f'wave_l{lam}_a{angle:g}_p{phase}',wavelength=lam,angle=angle,phase=phase*np.pi/2) for lam in (1.4,.7,.5,.35) for angle in (30.,75.) for phase in (0,1)]
DESIGNS += [dict(name=f'heldout_l{lam}_a{angle}',wavelength=lam,angle=angle,phase=np.pi/4) for lam in (.7,.35) for angle in (10.,110.)]
NF=len(DESIGNS);BASE=np.array([1.,1.1,.9,.13,.08]);AMP=np.array([.10,.05,.05,.03,.04]);TAU=1.;MU=1836.
KINDS=((('D',)*5,'D'),(('N',)*5,'N'),(('D','N','D','N','D'),'N'),(('N','D','N','D','N'),'D'))
COMPONENTS=('self_advection','pressure','thermal','minus_mu_tau_GTi','matched_force','phi_force','plus_mu_tau_GTi','correction')

def fields(p):
    p=np.asarray(p);u,th,e=np.moveaxis(p,-1,0);x=u*np.cos(th);y=u*np.sin(th)
    dx=np.stack((np.cos(th),-y,np.zeros_like(u)),axis=-1);dy=np.stack((np.sin(th),x,np.zeros_like(u)),axis=-1)
    vals=[];grads=[]
    for d in DESIGNS:
        vv=[];gg=[]
        for j in range(5):
            if d['name']=='constant':f=np.zeros_like(u);g=np.zeros_like(p)
            elif d['name']=='smooth':f,g=c.primitive_modes(p,'sx' if j%2==0 else 'sy')
            else:
                a=np.deg2rad(d['angle']+13*j);k=2*np.pi/d['wavelength'];m=1+j%2
                ph=k*(np.cos(a)*x+np.sin(a)*y)+m*e+d['phase']+.3*j
                dp=k*(np.cos(a)*dx+np.sin(a)*dy);dp[...,2]+=m
                f=np.cos(ph);g=-np.sin(ph)[...,None]*dp
            vv.append(BASE[j]+AMP[j]*f);gg.append(AMP[j]*g)
        vals.append(np.stack(vv,axis=-1));grads.append(np.stack(gg,axis=-2))
    return np.stack(vals,axis=-2),np.stack(grads,axis=-3)

def phi_fields(p):
    a,da=c.primitive_modes(p,'sx');b,db=c.primitive_modes(p,'sy')
    v=.07*a+.04*b;g=.07*da+.04*db
    return np.stack([v*0]+[v]*(NF-1),axis=-1),np.stack([g*0]+[g]*(NF-1),axis=-2)

def boundary(q,fieldfun=fields,material=True):
    nr=len(q.raw);shape=(NF,5,nr) if material else (NF,nr)
    a=[np.zeros(shape+(35,)),np.zeros(shape+(3,)),np.zeros(shape+(3,2)),np.zeros(shape+(35,))]
    for r in np.flatnonzero(q.raw//q.metadata['n']**2>=q.metadata['n']-2):
        v,g=fieldfun(q.boundary_wall_nodes[r])
        if material:
            a[0][:,:,r]=v.transpose(1,2,0);a[3][:,:,r]=np.einsum('qa,qcfa->cfq',q.boundary_wall_normal[r],g)
            v,g=fieldfun(q.boundary_wall_queries[r]);a[1][:,:,r]=v.transpose(1,2,0);a[2][:,:,r]=g[:,:,:,1:].transpose(1,2,0,3)
        else:
            a[0][:,r]=v.T;a[3][:,r]=np.einsum('qa,qca->cq',q.boundary_wall_normal[r],g)
            v,g=fieldfun(q.boundary_wall_queries[r]);a[1][:,r]=v.T;a[2][:,r]=g[:,:,1:].transpose(1,0,2)
    return c.QBoundaryData(*a)

def continuum(q,kappa):
    v,g=fields(q.slot_points[:,2]);p,dp=phi_fields(q.slot_points[:,2])
    g=np.einsum('ra,rcfa->rcf',q.magnetic_b[:,2],g);gp=np.einsum('ra,rca->rc',q.magnetic_b[:,2],dp)
    n,te,ti,vi,ve=np.moveaxis(v,-1,0);gn,gt,gi,gvi,gve=np.moveaxis(g,-1,0)
    k=kappa[:,None];dj=(vi-ve)*gn+n*(gvi-gve)+n*(vi-ve)*k
    pe=te*gn+n*gt;press=(te+TAU*ti)*gn+n*(gt+TAU*gi)
    ec=np.stack((-ve*gve,-MU*pe/n,-.71*MU*gt,-MU*TAU*gi,MU*(gp+TAU*gi),MU*gp,MU*TAU*gi,gi*0),axis=-1)
    R=np.stack((-(ve*gn+n*gve+n*ve*k),-ve*gt+2*te/(3*n)*(.71*dj-n*(gve+k*ve)),-vi*gi+2*ti/(3*n)*(dj-n*(gvi+k*vi)),-vi*gvi-press/n,-ve*gve-MU*pe/n-.71*MU*gt+MU*gp),axis=-1)
    return R.transpose(1,0,2),ec.transpose(1,0,2)

def electron(stencil,phi,q,correction):
    v=stencil[:,:,[1,3,2]];n,te,ti,vi,ve=np.moveaxis(v,-1,0)
    n0,te0,ti0,vi0,ve0=np.moveaxis(stencil[:,:,2],-1,0)
    G=lambda x:q.magnetic_b[None,:,2,2]/(2*(2*np.pi/q.metadata['n']/64))*(x[:,:,1]-x[:,:,0])
    gi=G(ti);gp=G(phi)
    return np.stack((-ve0*G(ve),-MU*G(n*te)/n0,-.71*MU*G(te),-MU*TAU*gi,MU*(gp+TAU*gi),MU*gp,MU*TAU*gi,correction[:,:,4]),axis=-1)

def stack(a,z):return np.stack((a[:,:,0],z[:,:,0],z[:,:,2],z[:,:,1],a[:,:,1]),axis=-2)
def project(q,x):return np.sum(np.asarray(x)[:,q.owner_raw,:]*q.owner_weight[None,:,:,None],axis=-2)


def six_fields(points):
    value, grad = fields(points)
    return (np.concatenate((value, .2+value[...,1:2]-1.1),axis=-1),
            np.concatenate((grad,grad[...,1:2,:]),axis=-2))


def six_boundary(q):
    nr=len(q.raw)
    a=[np.zeros((NF,6,nr,35)),np.zeros((NF,6,nr,3)),
       np.zeros((NF,6,nr,3,2)),np.zeros((NF,6,nr,35))]
    for r in np.flatnonzero(q.raw//q.metadata['n']**2>=q.metadata['n']-2):
        v,g=six_fields(q.boundary_wall_nodes[r]);a[0][:,:,r]=v.transpose(1,2,0)
        a[3][:,:,r]=np.einsum('qa,qcfa->cfq',q.boundary_wall_normal[r],g)
        v,g=six_fields(q.boundary_wall_queries[r]);a[1][:,:,r]=v.transpose(1,2,0)
        a[2][:,:,r]=g[:,:,:,1:].transpose(1,2,0,3)
    return c.QBoundaryData(*a)
