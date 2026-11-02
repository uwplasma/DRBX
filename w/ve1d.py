import sys
sys.path.insert(0,"tests")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from fci_braginskii.test_fci_wall_closure import flux_tube_bulk_rates, TAU
from drbx.fci_braginskii.native.fci_wall_closure import SheathClosure, bohm_wall_state
from drbx.fci_braginskii.native.fci_parallel_production_flux import parallel_short_wall_material_data
cells=12; mu=1836.0; dx=2.0/cells
ge,gi=float(SheathClosure().gamma_e(TAU,mu)),2.5
bulk=flux_tube_bulk_rates(cells,dx,mu,slave_ve=False)
bw=jnp.zeros(cells,bool).at[0].set(True); fw=jnp.zeros(cells,bool).at[-1].set(True)
def rate(x,variant,u0):
    u=u0.at[:,1].set(x[:cells]).at[:,4].set(x[cells:])
    r,bst,fst=bulk(u,ge,gi)
    minus=jnp.concatenate([u[:1],u[:-1]]); plus=jnp.concatenate([u[1:],u[-1:]])
    if variant!="none":
        _,_,info=parallel_short_wall_material_data(u,minus,plus,dx,dx,TAU,mu,backward_wall=bw,forward_wall=fw,backward_wall_state=bst,forward_wall_state=fst)
        ve=info["backward_residual"][:,4]*bw+info["forward_residual"][:,4]*fw
        if variant=="char": r=r.at[:,4].add(ve)
        if variant=="relax":
            wv=jnp.where(bw,bst[:,4],0.)+jnp.where(fw,fst[:,4],0.)
            r=r.at[:,4].add(jnp.where(bw|fw,-(u[:,4]-wv)*jnp.sqrt(mu*u[:,1])/dx*0.5,0.))
    return jnp.concatenate([r[:,1],r[:,4]])
u0=jnp.stack([jnp.ones(cells)]*3+[jnp.zeros(cells)]*2,-1)
# state with sonic outflow near walls
V=jnp.linspace(-1,1,cells)*0.7; u0=u0.at[:,3].set(V).at[:,4].set(V)
x0=jnp.concatenate([u0[:,1],u0[:,4]])
for variant in ("none","char","relax"):
    J=np.asarray(jax.jacfwd(lambda x:rate(x,variant,u0))(x0)); w=np.linalg.eigvals(J)
    print(variant,"max Re %.3f"%w.real.max(),"n(Re>1e-9)",int((w.real>1e-9).sum()))
