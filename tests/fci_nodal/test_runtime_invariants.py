"""Small scheme-locked runtime checks."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

def test_supported_operator_options(runtime_module):
    RuntimeOptions = runtime_module().RuntimeOptions

    options = RuntimeOptions()
    assert (options.tau, options.mu) == (1., 1836.)

    for name, alternatives in {
        'tau': (0.,),
        'mu': (1.,),
    }.items():
        for value in alternatives:
            with pytest.raises(ValueError, match='Supported model parameters require tau=1 and mu=1836'):
                replace(options, **{name: value})


def test_ssprk3_scalar_third_order(runtime_module):
    import math

    step = runtime_module('drb.stepper').ssprk3_step

    errors = []
    for steps in (8, 16, 32, 64):
        u, h, initial = 1., 1. / steps, 0.
        for k in range(steps):
            u, _, initial = step(lambda u, t: (-u, None), u, k*h, h,
                                 prepare=lambda t: t, project=lambda u, t: u,
                                 observe=lambda u: None, m0=initial)
        errors.append(abs(u-math.exp(-1.)))
    orders = [math.log2(coarse/fine) for coarse, fine in zip(errors, errors[1:])]
    print(f'SSPRK3 errors {errors}, orders {orders}')
    assert all(2.95 < order < 3.15 for order in orders)


def test_parallel_divergence_wall_balance(runtime_module):
    import jax
    import jax.numpy as jnp
    import numpy as np

    corrected_center_wall_band = runtime_module('operators.parallel.linear_ops').corrected_center_wall_band

    H = jnp.array([1., 2., 4.])
    C = jnp.array([[-1., 1., 0.], [0., -2., 2.], [1., 0., -1.]])
    R = jnp.array([[.75, .25, 0.], [0., .5, .5]])
    mu, sg = jnp.array([.5, 1.5]), jnp.array([-1., 1.])
    n, vi, ve = jnp.array([[1., 2., 4.], [2., 2., 4.], [.5, 1., 2.]])
    gn, gvi, gve = jnp.array([[1., 2.], [2., 4.], [.5, 2.]])
    p = dict(C=C, G=C/H[:, None], D=-C.T/H[:, None], R=R, RT=R.T, H=H,
             mu=mu, sg=sg, leg=.5*sg, tau=1., mu_model=1836., B2=n,
             blocks=jax.tree_util.Partial(lambda h,rhs: rhs/h[:,None]))
    U = jnp.column_stack([n,jnp.ones(3),jnp.ones(3),vi,ve,jnp.zeros(3)])
    g = jnp.column_stack([gn,jnp.ones(2),jnp.ones(2),gvi,gve,jnp.zeros(2)])
    jw = gn*(gvi-gve)
    center = corrected_center_wall_band(p,U,g,jw=jw)
    # Zero Omega removes its transport. B²/n = 1 extracts the current
    # divergence from the Omega row, and the density row is -D(n Ve).
    volume = H@jnp.column_stack([-center[:,0],center[:,5]])
    wall = (mu*sg)@jnp.column_stack([gn*gve,jw])
    np.testing.assert_array_equal(np.asarray(volume), np.asarray(wall))
    print(f'Weighted parallel divergence and signed wall flux {np.asarray(wall).tolist()}')


def test_wall_laws_conductor_limit(runtime_module):
    import jax
    import jax.numpy as jnp
    import numpy as np

    RuntimeOptions = runtime_module().RuntimeOptions
    admittance = runtime_module('drb.wall_traction').admittance
    walls = runtime_module('drb.walls')
    Closure, Residual = walls.Closure, walls.Residual
    perpendicular_wall_relations, parallel_wall_relations = walls.perpendicular_wall_relations, walls.parallel_wall_relations

    E, N = 4, 4
    count = E*N
    U = jnp.tile(jnp.array([1., 1., 1., 0., 0., 0.]), (count, 1))
    T = jnp.eye(count)
    R = T[jnp.array([0, count-1])]
    sg = jnp.array([1., -1.])
    bn = jnp.tile(sg,count//2)
    q = dict(R=R, C=jnp.zeros_like(T), H=jnp.ones(count), leg=.5*sg, sg=sg,
             blocks=lambda H, rhs: rhs/H[:, None])
    wall = SimpleNamespace(T=T, gi=1., Js=jnp.ones(count), hw=jnp.ones((count,3)),
                           curvature_flux=jnp.zeros(count), A0plus=jnp.zeros((count,3,3)),
                           bn=bn, Y=jnp.asarray(np.stack([admittance(s,0.)[0] for s in (1,-1)])),
                           wall_foot_to_hit_area_ratio=jnp.ones(2))
    options=RuntimeOptions()
    plan=SimpleNamespace(wall_ops=wall,q=q,params=SimpleNamespace(tau=options.tau,rho_star=.00045),
                         qparams=(options.tau,options.mu),chi=jnp.ones(6),runtime_options=options,
                         ctx=SimpleNamespace(plan=SimpleNamespace(structure=SimpleNamespace(n_eta=E,n=N,deta=.5))))
    res=Residual(jnp.zeros((2,3)),jnp.zeros(2),jnp.zeros(2),jnp.zeros((2,2)),
                 jnp.zeros(count),jnp.zeros(count),jnp.zeros(count),
                 jnp.zeros((count,3)),jnp.zeros((count,2)))
    t0=Closure(res,U[:,:3],wall_law='conducting_wall')
    # Cancel the resting floating/Bohm offsets along a prescribed affine
    # residual path while keeping conducting_wall admittance and the thermal rest state fixed.
    floating=.5*np.log(options.mu/(4*np.pi))
    sound=np.sqrt(2.)

    def differences(eps):
        p0=perpendicular_wall_relations(U,plan,t0)
        q0=parallel_wall_relations(U,jnp.ones(count),plan,t0)['g']
        residual=res._replace(q_vi=res.q_vi-(1.-eps)*sound,
                              p_psi=res.p_psi-(1.-eps)*floating,
                              p_velocity=res.p_velocity-(1.-eps)*sound*bn[:,None])
        t1=replace(t0,wall_law='floating_wall',residual=residual)
        p1=perpendicular_wall_relations(U,plan,t1)
        g1=parallel_wall_relations(U,jnp.ones(count),plan,t1)['g']
        return jnp.stack([jnp.max(jnp.abs(p1[key]-p0[key])) for key in ('psi','flux','velocity')]
                         + [jnp.max(jnp.abs(g1-q0))])

    epsilons=(1., .1, .01, .001, 0.)
    measured=np.asarray(jax.jit(jax.vmap(differences))(jnp.asarray(epsilons)))
    for eps,numbers in zip(epsilons,measured):
        print(f'Conductor residual eps={eps:g}, max differences '
              f'P psi/flux/velocity and Q g {numbers.tolist()}')
    assert np.isfinite(measured).all()
