"""Device-resident strong-wall parallel and perpendicular evaluation and nonautonomous SSPRK3."""
from ..options import CONDUCTING_WALL
import jax
from jax.experimental.sparse import BCSR
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from ..operators.perp import sbp_owner as sbp
from ..operators.perp.sbp_boundary import SatBoundaryData
from ..operators.perp.sbp_curvature import curvature_flux
from ..operators.parallel.owner_runtime import coupled_action
from ..operators.parallel.owner_sparse import _expand
from . import afc, walls
from .owner_load import perpendicular_partition_specs
from .stepper import ssprk3_step


def _sparse_local(matrix, arrays, value):
    # Same owner sparse apply as operators.parallel.owner_sparse.csr_kernel, inside the PCG map.
    data, index, ptr, send, _, order, _ = (x[0] for x in (*arrays[:3], *arrays[6:]))
    expanded = _expand(value,send,matrix.delta,matrix.mesh)[order]
    shape = (matrix.rows, matrix.cols+len(matrix.delta)*matrix.width)
    return BCSR((data,index,ptr),shape=shape,indices_sorted=True,unique_indices=True) @ expanded


def owner_actions(resolution, *, switches):
    """Compile raw/forced evaluation and stage projection on the shared owner mesh."""
    switches.require_implemented()
    if switches.flutter != 'off':
        raise NotImplementedError('The owner evaluator requires flutter off under the scheme lock')
    plan, q, slabs = resolution.plan, resolution.q, resolution.slabs
    mesh, grid, opts = q.mesh, resolution.wall_grid, plan.opts
    sh = P(sbp.AXIS)
    spec = perpendicular_partition_specs(plan.ctx)
    shape = (slabs.capacity, plan.ctx.nodal.structure.P)
    N = grid.wall_per_plane
    tau, mu = q.params
    q_action = coupled_action(mesh,q.params,q.rhs_chunk)

    def unpack(ctx):
        return jax.tree.map(lambda a,s:a[0] if s else a,ctx,spec)

    def ampere(ctx,U,source,beta):
        ctx = unpack(ctx)
        u = U.reshape(*shape,6)
        return sbp.ampere_owner(
            ctx.laplacian,ctx.factors,slabs,u[...,0],u[...,3],u[...,4],beta=beta,mu=mu,
            rtol=opts.phi_rtol,c_kappa=opts.laplacian_c_kappa,
            source=None if source is None else source.reshape(shape))

    def ampere_flat(ctx,U,source,beta):
        A,j,ve,info = ampere(ctx,U,source,beta)
        return A.ravel(),j.ravel(),ve.ravel(),info

    ampere_map = jax.shard_map(ampere_flat,mesh=mesh,in_specs=(spec,sh,sh,P()),
                              out_specs=(sh,sh,sh,P()),check_vma=False)
    maps = plan.wall_ops

    def p_stage(ctx,U,datum,gram,H,sigma,own,T,TT,K,rho):
        ctx = unpack(ctx)
        u = U.reshape(*shape,6)
        def lift(x):
            return _sparse_local(maps.TT,TT,(x.ravel()/gram)[:,None]).reshape(shape)
        def project(x):
            return x-lift(_sparse_local(maps.T,T,x.reshape(-1,1)))
        def positive(x):
            return _sparse_local(maps.K,K,x.reshape(-1,1)).reshape(shape)
        psi,info = sbp.strong_psi_owner(u[...,5],H.reshape(shape),datum,
            None if sigma is None else sigma.reshape(shape),ctx.factors,slabs,
            rho_star=rho,rtol=opts.phi_rtol,positive=positive,project=project,lift=lift)
        phi = psi-tau*u[...,0]*u[...,2]
        br = rho*sbp.sbp_bracket_owner(ctx.nodal,phi,u,SatBoundaryData((own.reshape(-1,N,6),)),
                                      1.,slabs,opts.bracket_c_kappa)
        fields = jnp.concatenate((u[...,:3],u[...,5:6],phi[...,None]),axis=-1)
        curv = sbp.curvature_derivative_owner(ctx.nodal,curvature_flux(ctx.nodal),fields,slabs)
        return psi.ravel(),br.reshape(-1,6),curv.reshape(-1,5),ctx.nodal.B.ravel(),info

    p_map = jax.shard_map(p_stage,mesh=mesh,in_specs=(spec,)+(sh,)*9+(P(),),
                         out_specs=(sh,sh,sh,sh,P()),check_vma=False)

    def evaluate(U,plan,q_arrays,wall,node_active,hit_active):
        cl = wall.closure
        if cl is None or switches.wall!=cl.wall_law:
            raise ValueError('The owner operator needs the selected strong-wall closure')
        aux = {}
        if switches.em == 'on':
            A,j,ve,info = ampere_map(plan.ctx,U,wall.ampere_source,plan.beta_hat)
            physical = U.at[:,4].set(ve)
            aux = dict(A=A,current=j,ampere=info)
        else:
            physical = U
        w = plan.wall_ops
        pd = walls.perpendicular_wall_relations(physical,plan,cl,owner_grid=grid)
        own = w.Tc @ physical
        psi,br,curv,B,pol = p_map(plan.ctx,physical,pd['psi'],w.trace_gram,w.H,
                wall.polarization_source,own,w.T.arrays,w.TT.arrays,w.K.arrays,plan.params.rho_star)
        qd = walls.parallel_wall_relations(physical,psi,plan,cl,hit_active=hit_active)
        p,bn,bd = walls.owner_wall_rows(physical,psi,plan,cl,pd,qd,br,curv,B,hit_active=hit_active)
        parallel,status,_ = q_action(*q_arrays,physical,psi,qd['g'],bn,bd,plan.chi,
                                     plan.mass_ratio,plan.dirichlet_mass_correction)
        current = physical[:,0]*(physical[:,3]-physical[:,4])
        local = jnp.zeros_like(U).at[:,4].set(mu*plan.nu*current)
        local = local.at[:,1].set(2*plan.nu*current**2/(3*physical[:,0]))
        hard = [(5,jnp.ones_like(w.Js))]
        if cl.wall_law == CONDUCTING_WALL:
            hard.append((1,(w.curvature_flux!=0).astype(U.dtype)))
        value = walls.project_mass(w,p+parallel+local,hard)
        if switches.limiter == 'afc':
            delta,stats = afc.owner_correction(plan.afc,physical,plan.chi,w.H,w,wall_law=cl.wall_law)
            value = value+delta
            aux['limiter'] = stats
        return jnp.where(node_active[:,None],value,0), dict(aux,psi=psi,polarization=pol,records=status)

    def forced(U,plan,q_arrays,stage,node_active,hit_active):
        raw,aux = evaluate(U,plan,q_arrays,stage.wall,node_active,hit_active)
        return walls.moving_rate(raw+stage.source,stage.rate,plan,stage.wall),aux

    raw_ex, force_ex = jax.jit(evaluate), jax.jit(forced)
    project_ex = jax.jit(walls.compatible_state)
    args = (plan,q.arrays()[:-1])
    active = (resolution.node_active,resolution.hit_active)

    def raw(U,wall):
        return raw_ex(U,*args,wall,*active)

    def force(U,stage):
        return force_ex(U,*args,stage,*active)

    def project(U,stage):
        return project_ex(U,plan,stage.wall)

    return raw,force,project


def owner_timestep(resolution):
    """Locked global AFC bound and step size."""
    p = resolution.plan
    bound = afc.owner_positivity_bound(p.afc,p.wall_ops.H,p.chi)
    return bound,.9*bound['minimum']


def owner_step(rhs,U,t,h,*,prepare,project,observe,m0):
    """Advance the owner state with the same SSPRK3 stage and projection order."""
    return ssprk3_step(rhs,U,t,h,prepare=prepare,project=project,observe=observe,m0=m0)
