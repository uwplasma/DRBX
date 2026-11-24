"""Continuum-sourced moving-wall six-field MMS; no discrete RHS fit.

Lead time law: rest + (base field-rest)*(1+.2*sin(2*pi*t/.02)).
A*=.01*recorded zero_wall_value_profile with the same factor. Flutter is off because
saved exact jets do not include the transverse geometry derivatives required
for D F, DF G and DF F continuum diffusion.
"""
from ..options import CONDUCTING_WALL, FLOATING_WALL
from typing import NamedTuple
from dataclasses import replace
import jax
import jax.numpy as jnp
from .continuum import source as continuum_source_terms
from ..drb.operator import WallData
from ..drb.walls import Residual, make_wall_data, perpendicular_wall_relations, read_wall_foot_to_hit_area_ratio
from ..operators.parallel import linear_ops as qo
from ..operators.perp.rhs_options import NodalWallData

class ExactJet(NamedTuple):
    """Values n,Te,Ti,Vi,Ve,Omega,phi, logical derivatives, and exact time rates."""
    value: object
    gradient: object
    hessian: object
    dt: object


class Stage(NamedTuple):
    exact:object;rate:object;source:object;wall:object

REST=jnp.array([1.,1.,1.,0.,0.,0.,0.])
PERIOD=.02

def scaled(jets,a,ap,*,nonlinear_phi=False,grounded_phi=False):
    dev=jets.value-REST
    V=REST+a*dev;G=a*jets.gradient;H=a*jets.hessian;dt=ap*dev
    if nonlinear_phi:
        n=V[:,0];gn=G[:,0];hn=H[:,0]
        if grounded_phi:V=V.at[:,6].set(jnp.log(n)+1.71*(V[:,1]-1))
        else:V=V.at[:,6].set(.5*jnp.log(1836/(4*jnp.pi))+jnp.log(n)+1.71*(V[:,1]-1))
        G=G.at[:,6].set(gn/n[:,None]+1.71*G[:,1])
        H=H.at[:,6].set(hn/n[:,None,None]-jnp.einsum('pi,pj->pij',gn,gn)/n[:,None,None]**2+1.71*H[:,1])
        dt=dt.at[:,6].set(dt[:,0]/n+1.71*dt[:,1])
    return ExactJet(V,G,H,dt)

def parallel_continuum(U,ds,dphi,divb,B,tau,mu):
    """JAX port of independent continuum chain rule pkg.references.continuum's independent chain rule."""
    n,t,i,v,e,o=U.T;dn,dt,di,dv,de,do=ds.T
    dj=dn*(v-e)+n*(dv-de)+divb*n*(v-e)
    return jnp.stack([-(dn*e+n*de+divb*n*e),
        -e*dt+2*t/(3*n)*(.71*dj-n*(de+divb*e)),
        -v*di+2*i/(3*n)*(dj-n*(dv+divb*v)),
        -v*dv-(dn*(t+tau*i)+n*(dt+tau*di))/n,
        -e*de-mu*(dn*t+n*dt)/n-.71*mu*dt+mu*dphi,
        -v*do+B*B/n*dj],1)

class StageParameters(NamedTuple):
    rho_star:object;D:object;chi:object;tau:object;mu:object;nu:object;beta_hat:object


class ManufacturedInputs(NamedTuple):
    exact:object;rate:object;source:object
    physical:object;psi:object;wall:WallData;q_flux:object;p_flux:object


def manufactured_inputs(t,params,data,*,wall_shape,psi_mode):
    amplitude=data.get('amplitude',1.)
    a=amplitude*(1+.2*jnp.sin(2*jnp.pi*t/PERIOD));ap=amplitude*.2*2*jnp.pi/PERIOD*jnp.cos(2*jnp.pi*t/PERIOD)
    nonlinear_potential='uses_nonlinear_manufactured_potential' in data
    grounded='uses_grounded_wall_potential' in data
    nodal=scaled(data['nodal'],a,ap,nonlinear_phi=nonlinear_potential,grounded_phi=grounded);pwall=scaled(data['pwall'],a,ap,nonlinear_phi=nonlinear_potential,grounded_phi=grounded)
    V,G,HH,dV=nodal;U=V[:,:6];phi=V[:,6];tau,mu=params.tau,params.mu
    bracket=params.rho_star*continuum_source_terms.point_bracket(data['h'],data['jac'],G[:,6],G[:,:6])
    slots=[0,1,2,5,6];v5,g5,h5=V[:,slots],G[:,slots],HH[:,slots]
    curv=params.rho_star*continuum_source_terms.continuum_curvature_rhs(jnp.moveaxis(v5,-1,0),jnp.moveaxis(g5,-2,0),data['K'],data['B'],tau,psi_mode)
    perp=bracket+jnp.zeros_like(U).at[:,[0,1,2,5]].set(curv)
    perp=perp+params.D*continuum_source_terms.laplacian_cont(data['metric'],data['divA'],data['jac'],G[:,:6],HH[:,:6])
    psi,gpsi,hpsi=continuum_source_terms.psi_derivatives(psi_mode,tau,v5,g5,h5)
    lap_psi=continuum_source_terms.laplacian_cont(data['metric'],data['divA'],data['jac'],gpsi[:,None],hpsi[:,None])[:,0]
    sigma=U[:,5]-params.rho_star**2*lap_psi
    ds=a*data['ds']
    if nonlinear_potential:ds=ds.at[:,6].set(ds[:,0]/U[:,0]+1.71*ds[:,1])
    q=parallel_continuum(U,ds[:,:6],ds[:,6],data['divb'],data['B'],tau,mu)+params.chi[None,:]*a*data['lap']
    current=U[:,0]*(U[:,3]-U[:,4]);local=jnp.zeros_like(U).at[:,4].set(mu*params.nu*current)
    local=local.at[:,1].set(2*params.nu*current**2/(3*U[:,0]))
    A=a*data['A'].value;dA=ap*data['A'].value
    lapA=a*continuum_source_terms.laplacian_cont(data['metric'],data['divA'],data['jac'],data['A'].gradient[:,None],data['A'].hessian[:,None])[:,0]
    ampere_source=-lapA/params.beta_hat-current
    canonical=U.at[:,4].set(U[:,4]-mu*A);rate=dV[:,:6].at[:,4].add(-mu*dA)
    source=rate-(perp+q+local)
    E,N=wall_shape
    wv,wg=pwall.value.reshape(E,N,7),pwall.gradient.reshape(E,N,7,3)
    normal=data['wall_ginv']/jnp.sqrt(data['wall_ginv'][...,:1])
    normal_d=jnp.einsum('enj,enfj->enf',normal,wg[...,:6,:])
    g=REST[:6]+a*(data['g']-REST[:6]);zero=jnp.zeros_like(U)
    wall=WallData(NodalWallData(wv[...,:6],normal_d,wv[...,6]+tau*wv[...,0]*wv[...,2]),
        g,zero,zero,jnp.array([True,True,True,False,False,True]),sigma.reshape(E,-1),None,ampere_source)
    return ManufacturedInputs(canonical,rate,source,U,psi,wall,
                              a*data['parallel_wall_flux_base'] if nonlinear_potential else None,
                              a*data['perpendicular_wall_flux_base'] if nonlinear_potential else None)


def manufacture_t1_wall(U,psi,exact_g,plan,base,*,owner_grid=None,hit_active=None,p_active=None):
    """Manufacture discrete floating_wall wall residuals on the gathered or owner layout."""
    w=plan.wall_ops;p=plan.q;tw=w.T@U;zext=tw[:,:3]
    zero=make_wall_data(plan,base,wall_law='floating_wall',z_ext=zext)
    pr=perpendicular_wall_relations(U,plan,zero.closure,owner_grid=owner_grid);r=p['R']@U;arr=qo.trace(p,U)
    sound=jnp.sqrt(exact_g[:,1]+w.gi*plan.params.tau*exact_g[:,2])
    target=jnp.where(p['sg']*arr[:,3]>sound,p['sg']*arr[:,3],sound)
    flux=(w.F@U[:,:3])*plan.params.D[:3]/w.Js[:,None]
    leg=p['leg'] if hit_active is None else jnp.where(hit_active,p['leg'],1.)

    def masked(value, mask):
        return value if mask is None else jnp.where(mask,value,0.)

    residual=Residual(
        masked(plan.chi[:3]*(exact_g[:,:3]-r[:,:3])/leg[:,None],
               None if hit_active is None else hit_active[:,None]),
        masked(p['sg']*exact_g[:,3]-target,hit_active),
        exact_g[:,0]*(exact_g[:,3]-exact_g[:,4]),jnp.zeros((len(arr),2)),
        masked((w.T@psi[:,None])[:,0]-pr['psi'],p_active),tw[:,5],jnp.zeros(len(tw)),
        masked(flux+plan.params.rho_star*jnp.einsum('hij,hj->hi',pr['An_plus'],tw[:,:3]-zext),
               None if p_active is None else p_active[:,None]),
        masked(tw[:,3:5]-pr['velocity'],None if p_active is None else p_active[:,None]))
    return make_wall_data(plan,base,wall_law='floating_wall',z_ext=zext,residual=residual)


def manufacture_t0_wall(U,psi,exact_g,plan,base):
    """Discrete G(U*) residual data for the unchanged conducting_wall traction/strong rows."""
    w=plan.wall_ops;p=plan.q;tw=w.T@U;r=p['R']@U
    exact_g=exact_g.at[:,5].set(qo.trace(p,U)[:,5])
    z=exact_g[:,:3];dz=z-1.;sg=p['sg'];ell=jnp.abs(p['leg'])
    L=jnp.array([[1.,0.],[1.,-1.]])
    physicalL=jnp.broadcast_to(L,(len(r),2,2)).at[:,1,1].set(-1/z[:,0])
    wd=plan.chi[3:5]*jnp.array([1.,1/plan.qparams[1]])/ell[:,None]
    wd=wd*read_wall_foot_to_hit_area_ratio(plan)[:,None]
    stress=L.T[None]@(wd[:,:,None]*physicalL)
    psil=qo.trace(p,psi[:,None])[:,0]-1.
    force=sg[:,None]*jnp.stack([dz@jnp.array([2.,1.,1.]),-dz@jnp.array([2.,1.71,1.])+psil],1)
    force=force+jnp.einsum('ij,hj->hi',L.T,wd*r[:,3:5])
    v=jnp.stack([exact_g[:,3],z[:,0]*(exact_g[:,3]-exact_g[:,4])],1)
    law=jnp.einsum('hij,hj->hi',jnp.eye(2)[None]+w.Y@stress,v)-jnp.einsum('hij,hj->hi',w.Y,force)
    flux=(w.F@U[:,:3])*plan.params.D[:3]/w.Js[:,None]
    flux=flux+plan.params.rho_star*jnp.einsum('hij,hj->hi',w.A0plus/w.Js[:,None,None],tw[:,:3]-1.)
    res=Residual(plan.chi[:3]*(exact_g[:,:3]-r[:,:3])/p['leg'][:,None],
        jnp.zeros(len(r)),jnp.zeros(len(r)),law,
        (w.T@psi[:,None])[:,0]-plan.params.tau*tw[:,0]*tw[:,2],tw[:,5],tw[:,1]-1.,flux,tw[:,3:5])
    return make_wall_data(plan,base,wall_law='conducting_wall',z_ext=tw[:,:3],residual=res)


def owner_stage(resolution, data, *, wall_law):
    """Build stage source and strong-wall residuals from owner manufactured field data."""
    grid, plan = resolution.wall_grid, resolution.plan
    q = resolution.q
    mesh = q.mesh
    p_active = resolution.p_walls.put({i:ids>=0 for i,ids in resolution.p_walls.ids_by_owner.items()},mesh)
    shape = (q.mesh.size*resolution.slabs.capacity,grid.wall_per_plane)

    def build(t,plan,data,hit_active,p_active,node_active):
        params = StageParameters(plan.params.rho_star,plan.params.D,plan.chi,*plan.qparams,plan.nu,plan.beta_hat)
        data = dict(data,wall_ginv=data['wall_ginv'].reshape(*shape,3))
        inputs = manufactured_inputs(t,params,data,wall_shape=shape,psi_mode=plan.opts.psi)
        canonical,rate,source,U,psi,wall,q_flux,p_flux = inputs
        g = wall.g
        if q_flux is not None:
            g = g.at[:,:3].set((plan.q['R']@U)[:,:3]+plan.q['leg'][:,None]*q_flux/plan.chi[None,:3])
        g = jnp.where(hit_active[:,None],g,REST[:6])
        wall = wall._replace(g=g,polarization_source=wall.polarization_source.reshape(-1))
        if wall_law == CONDUCTING_WALL:
            pp = dict(plan.q,leg=jnp.where(hit_active,plan.q['leg'],1.))
            wall = manufacture_t0_wall(U,psi,g,replace(plan,q=pp),wall)
        elif wall_law == FLOATING_WALL:
            wall = manufacture_t1_wall(U,psi,g,plan,wall,owner_grid=grid,
                                       hit_active=hit_active,p_active=p_active)
        else:
            raise ValueError(wall_law)
        if q_flux is not None:
            w = plan.wall_ops
            discrete_flux = (w.F@U[:,:3])*plan.params.D[:3]/w.Js[:,None]
            residual = wall.closure.residual._replace(q_flux=q_flux,
                p_flux=wall.closure.residual.p_flux+p_flux-discrete_flux)
            wall = wall._replace(closure=replace(wall.closure,residual=residual))
        canonical = jnp.where(node_active[:,None],canonical,REST[:6])
        rate, source = (jnp.where(node_active[:,None],v,0.) for v in (rate,source))
        return Stage(canonical,rate,source,wall)

    execute = jax.jit(build)
    if len(mesh.local_devices)!=mesh.size:
        from ..drb.owner_route_store import replicate
        return lambda t: execute(replicate(mesh,t),plan,data,resolution.hit_active,p_active,resolution.node_active)
    return lambda t: execute(jnp.asarray(t),plan,data,resolution.hit_active,p_active,resolution.node_active)


_owner_stage_kernel=jax.jit(manufactured_inputs,static_argnames=('wall_shape','psi_mode'))


def load_owner_stage(prepared,owner,t,params,*,wall_law,psi_mode,device=None):
    """Return per-owner stage inputs before the coupled discrete wall closure."""
    data=prepared.read_manufactured_field_data(owner,wall_law,device=device)
    planes=prepared.edges[owner+1]-prepared.edges[owner]
    if device is not None:
        t,params=jax.device_put((t,params),device)
    return _owner_stage_kernel(t,params,data,
                    wall_shape=(planes,prepared.manifest['wall_per_plane']),psi_mode=psi_mode)
