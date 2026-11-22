
def cumulative_gauss(x):
    out=np.empty((len(x),len(x)))
    for j in range(len(x)):
        p=np.polynomial.Polynomial.fromroots(np.delete(x,j));p/=p(x[j]);p=p.integ()
        out[:,j]=p(x)-p(0)
    return out

def integrate(ev,z,extent,source_flux,order):
    """Seed flux times exp integral(div B/Beta) d eta / |Beta|.

    This is the change-of-variables formula for the traced cross section,
    including the nonzero divergence of the actual interpolated field.
    """
    x,w=roots_legendre(order);x=(x+1)/2;w=w/2;C=cumulative_gauss(x)
    n=len(extent);pos=[];arc=[]
    for t in x:
        xyz=sample(z,extent*t)
        arc.append(xyz[:,2])
        pos.append(np.column_stack([np.linalg.norm(xyz[:,:2],axis=1),np.arctan2(xyz[:,1],xyz[:,0])%TAU,
                                  (z['seed'][:,2]+z['delta']*extent*t)%TAU]))
    q=np.stack(pos,axis=1);flat=q.reshape(-1,3);mm={}
    for lo in range(0,len(flat),512):
        m=metric(ev,flat[lo:lo+512]);
        for k,v in m.items():mm.setdefault(k,[]).append(v)
    mm={k:np.concatenate(v).reshape(n,order,*v[0].shape[1:]) for k,v in mm.items()}
    signed=z['delta']*extent
    divergence=mm['divB']/mm['B'][:,:,2]
    exponent=signed[:,None]*(divergence@C.T)
    weights=source_flux[:,None]*abs(signed[:,None])*w[None,:]*np.exp(exponent)/abs(mm['B'][:,:,2])
    vol=weights.sum(1)
    # Physical centroid and per-physical-slab contribution, without folding wedges.
    xy=np.stack([q[:,:,0]*np.cos(q[:,:,1]),q[:,:,0]*np.sin(q[:,:,1])],axis=-1)
    cent=np.sum(weights[:,:,None]*np.concatenate([xy,(signed[:,None]*x)[...,None]],axis=2),axis=1)
    # Integrate the smooth per-ray volume density to the physical slab boundary.
    # Binning Gauss nodes would incorrectly integrate a discontinuous indicator.
    start_slab=np.floor(z['seed'][:,2]/HOP).astype(int)%N
    boundary=(start_slab+(signed>0))*HOP
    cut=np.clip((boundary-z['seed'][:,2])/signed,0,1)
    cw=np.empty((n,order))
    for j in range(order):
        poly=np.polynomial.Polynomial.fromroots(np.delete(x,j));poly/=poly(x[j]);poly=poly.integ();cw[:,j]=poly(cut)-poly(0)
    first=np.sum(weights/w[None,:]*cw,axis=1)
    assert np.min(first)>-1e-12 and np.min(vol-first)>-1e-12
    slab_by_ray=np.zeros((n,N));slab_by_ray[np.arange(n),start_slab]=first
    slab_by_ray[np.arange(n),(start_slab+np.sign(signed).astype(int))%N]+=vol-first
    slab=slab_by_ray.sum(0)
    arc_integral=np.sum(weights*np.stack(arc,axis=1),axis=1)
    return dict(volume=vol,centroid_integral=cent,arc_centroid_integral=arc_integral,
                transport_ratio=np.exp(signed*(divergence@w)),slab_volume=slab,slab_by_ray=slab_by_ray,divergence_exponent_max=np.max(abs(exponent)))

def source_rule(cells,order):
    x,w=roots_legendre(order);x=(x+1)/2;w=w/2
    seeds=[];weights=[];owners=[]
    for j,(r0,r1,t0,t1) in enumerate(cells):
        for a,wa in zip(x,w):
            for b,wb in zip(x,w):
                # Gauss in r (J is in polar coordinates), not disk area.
                seeds.append([r0+(r1-r0)*a,t0+(t1-t0)*b]);weights.append((r1-r0)*(t1-t0)*wa*wb);owners.append(j)
    return np.array(seeds),np.array(weights),np.array(owners)
