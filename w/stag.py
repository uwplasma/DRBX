import numpy as np, scipy.linalg as sla
def build(nx=24,nz=24,shear=1.0,vv=0.3,rough=0.0,seed=1):
    rng=np.random.default_rng(seed); dz=1/nz; dx=1/nx; N=nx*nz
    idx=lambda i,k:(i%nx)*nz+(k%nz)
    def shift(sgn):
        M=np.zeros((N,N))
        for i in range(nx):
            for k in range(nz):
                x=i/nx; sl=0.5+shear*0.4*np.sin(2*np.pi*x)+rough*rng.standard_normal()*0.1
                xs=i+sgn*sl*dz/dx; i0=int(np.floor(xs)); w=xs-i0
                M[idx(i,k),idx(i0,k+sgn)]+=1-w; M[idx(i,k),idx(i0+1,k+sgn)]+=w
        return M
    Ip,Im=shift(1),shift(-1)
    V=np.array([dx*dz*(1+vv*np.cos(2*np.pi*i/nx)) for i in range(nx) for k in range(nz)])
    return Ip,Im,V,dz
for shear,vv,rough in ((0,0,0),(1,0,0),(2,0,2),(1,0.3,0)):
    Ip,Im,V,dz=build(shear=shear,vv=vv,rough=rough)
    N=len(V); I=np.eye(N)
    # collocated pair from central D0
    D0=(Ip-Im)/(2*dz); H=-(D0.T*V[None])/V[:,None]; G=H-np.diag(H.sum(1)); D=-(G.T*V[None])/V[:,None]
    # staggered: G+ f=(Ip f - f)/dz, D+=-V^-1 G+^T V, face value A f=(Ip f+f)/2
    Gp=(Ip-I)/dz; Dp=-(Gp.T*V[None])/V[:,None]; A=0.5*(Ip+I)
    cd=-D; st=-Dp@A
    f=lambda M: sla.eigvals(M).real.max()*dz
    print(f"shear={shear} vv={vv} rough={rough}: collocated -D {f(cd):+.2e}   staggered -D+A {f(st):+.2e}   mass(col) {abs(V@cd).max():.1e} mass(stag) {abs(V@st).max():.1e}")
