"""``absolute_method`` of the P06 q3 absolute-matrix action: ``"lapack4"`` (default), ``"block_lapack"``, ``"closed_form"``.

Per-node agreement with the 4x4 LAPACK action and with a 60-digit mpmath reference on random physical states
(extreme ratios, ``Ti -> 0``), the validity / fallback semantics, JVPs against finite differences and against the 4x4
action's JVP, and the bitwise-unchanged default.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from pathlib import Path

from drbx.native.fci_curvature_production_flux import curvature_principal_matrix
from drbx.native.fci_perpendicular_face_corrections import (
    ABSOLUTE_METHODS, _absolute_action_closed_form, _curvature_negative_root, _p06_absolute_action,
    _p06_absolute_action_block, p06_characteristic_face_correction)

DATA = Path(__file__).parent/'data/p_shared_face_rows'
FLOOR = 1e-12


def _states(count, seed, n=(-6, 6), te=(-3, 3), r=(-4, 3), tau=(-1, 1)):
    """Physical ``(n, Te, Ti, B, tau, scale, jump)`` with ``r = tau Ti/Te`` log-uniform over ``r``."""
    rng = np.random.default_rng(seed)
    n_, te_ = 10**rng.uniform(*n, count), 10**rng.uniform(*te, count)
    r_, tau_ = 10**rng.uniform(*r, count), 10**rng.uniform(*tau, count)
    ti_ = r_*te_/tau_
    b_ = 10**rng.uniform(-1, 1, count)
    scale = rng.normal(size=count)*10**rng.uniform(-2, 2, count)
    jump = rng.normal(size=(count, 4))*np.stack([n_, te_, te_, np.ones(count)], axis=-1)
    return n_, te_, ti_, b_, tau_, scale, jump


def _matrix(n, te, ti, b, tau, scale):
    return jnp.asarray(scale)[..., None, None]*curvature_principal_matrix(n, te, ti, b, tau)


def _three(state, floor=FLOOR):
    n, te, ti, b, tau, scale, jump = (jnp.asarray(x) for x in state)
    matrix = _matrix(n, te, ti, b, tau, scale)
    lapack, lapack_invalid = _p06_absolute_action(matrix, jump)
    block, block_invalid = _p06_absolute_action_block(matrix, jump)
    closed, closed_invalid = _absolute_action_closed_form(n, te, ti, b, tau, scale, matrix, jump, floor)
    return {'lapack4': (np.asarray(lapack), np.asarray(lapack_invalid)),
            'block_lapack': (np.asarray(block), np.asarray(block_invalid)),
            'closed_form': (np.asarray(closed), np.asarray(closed_invalid))}


def _mp_reference(n, te, ti, b, tau, scale, jump):
    """``|M| jump`` with ``M = scale P`` to 60 digits (``sign(0) = 0``)."""
    mp.mp.dps = 60
    n, te, ti, b, tau = (mp.mpf(float(x)) for x in (n, te, ti, b, tau))
    A = mp.matrix([[2*te, 2*n, 2*n*tau], [4*te**2/(3*n), 14*te/3, 4*tau*te/3], [4*ti*te/(3*n), 4*ti/3, -2*tau*ti]])
    c = [2*b*b*(te+tau*ti)/n, 2*b*b, 2*tau*b*b]
    values, vectors = mp.eig(A)
    inverse = vectors**-1
    u = mp.matrix([mp.mpf(float(x)) for x in jump[:3]])
    lam = [mp.re(x) for x in values]
    zero = mp.mpf(10)**-40*max(abs(x) for x in lam)
    sgn = [0 if abs(x) < zero else mp.sign(x) for x in lam]
    modal = inverse*u
    absolute = vectors*mp.matrix([abs(x)*modal[i] for i, x in enumerate(lam)])
    sign_u = vectors*mp.matrix([sgn[i]*modal[i] for i in range(3)])
    omega = sum(c[i]*sign_u[i] for i in range(3))
    return abs(float(scale))*np.array([float(mp.re(x)) for x in (*absolute, omega)])


def _reference(state):
    n, te, ti, b, tau, scale, jump = state
    return np.array([_mp_reference(n[i], te[i], ti[i], b[i], tau[i], scale[i], jump[i]) for i in range(len(n))])


def _relative(action, reference):
    """Per-node max error over the largest component of the reference."""
    return np.max(np.abs(action-reference), axis=-1)/np.maximum(np.max(np.abs(reference), axis=-1), 1e-300)


def test_methods_listed_and_unknown_rejected():
    assert ABSOLUTE_METHODS == ('lapack4', 'block_lapack', 'closed_form')
    with pytest.raises(ValueError, match='absolute_method'):
        p06_characteristic_face_correction(*(np.zeros((1, 1, 4)),)*3, np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)),
                                           np.zeros(1, bool), np.zeros(1, bool), absolute_method='eig')


def test_physical_states_agree_with_the_mpmath_reference_and_the_4x4_action():
    # moderate ratios: the 4x4 action is itself accurate to ~1e-13 here
    state = _states(120, 0, n=(-2, 2), te=(-1.5, 1.5), r=(-2, 2), tau=(-.5, .5))
    reference = _reference(state)
    out = _three(state)
    assert not out['lapack4'][1].any() and not out['block_lapack'][1].any() and not out['closed_form'][1].any()
    for name, (action, _) in out.items():
        assert np.max(_relative(action, reference)) <= 1e-12, name
    for name in ('block_lapack', 'closed_form'):
        diff = _relative(out[name][0], out['lapack4'][0])
        assert np.max(diff) <= 1e-12, (name, np.max(diff), np.median(diff))


def test_wide_ranges_the_closed_form_is_closer_to_the_reference_than_the_4x4_action():
    # n, Te over 6 decades, r in [1e-4, 1e3]: the 4x4 test admits eigenvector conditions up to 1e8 (errors ~1e-11)
    state = _states(120, 0)
    reference = _reference(state)
    out = _three(state)
    assert not out['closed_form'][1].any()
    valid4, valid3 = ~out['lapack4'][1], ~out['block_lapack'][1]
    assert valid4.sum() >= 100 and valid3.sum() >= 100       # the eigenvector-condition test drops a few nodes of each
    error = {name: _relative(action, reference) for name, (action, _) in out.items()}
    assert np.max(error['closed_form']) <= 1e-12
    assert np.max(error['block_lapack'][valid3]) <= 1e-10
    assert np.median(error['closed_form']) <= 2*np.median(error['lapack4'][valid4])
    assert np.max(_relative(out['closed_form'][0][valid4], out['lapack4'][0][valid4])) <= 2*np.max(error['lapack4'][valid4])+1e-12


def test_closed_form_over_extreme_ratios():
    """``r`` from 1e-14 to 1e12, ``n`` over 24 decades: the cubic root, its Vieta pair and the projector stay accurate."""
    state = _states(100, 1, n=(-12, 12), te=(-8, 4), r=(-14, 12), tau=(-2, 2))
    reference = _reference(state)
    action, invalid = _three(state)['closed_form']
    assert not invalid.any()
    assert np.max(_relative(action, reference)) <= 1e-12


def test_negative_root_solves_the_cubic():
    r = jnp.asarray(np.concatenate(([0.], 10.**np.arange(-16, 14, .25))))
    mu = _curvature_negative_root(r)
    assert float(mu[0]) == 0.
    assert bool(jnp.all(mu[1:] < 0))
    residual = ((9*mu+18*r-60)*mu+60-160*r)*mu+200*r
    scale = 9*jnp.abs(mu)**3+jnp.abs(18*r-60)*mu**2+jnp.abs(60-160*r)*jnp.abs(mu)+200*r
    assert float(jnp.max(jnp.abs(residual)/jnp.maximum(scale, 1e-300))) <= 1e-14
    # the implicit derivative -p_r/p_mu
    _, tangent = jax.jvp(_curvature_negative_root, (r,), (jnp.ones_like(r),))
    step = 1e-7*(1+r)
    finite = (_curvature_negative_root(r+step)-_curvature_negative_root(r-step))/(2*step)
    np.testing.assert_allclose(tangent[1:], finite[1:], rtol=1e-5)


def test_ti_zero_uses_sign_zero_and_stays_finite():
    state = list(_states(6, 2))
    state[2] = np.zeros(6)                           # Ti = 0 exactly
    reference = _reference(tuple(state))
    out = _three(tuple(state))
    for name in ('closed_form', 'block_lapack'):
        action, invalid = out[name]
        assert not invalid.any(), name
        np.testing.assert_allclose(_relative(action, reference), 0, atol=1e-12, err_msg=name)
    # tau = 0 gives the same r = 0 (the third column of the block vanishes)
    state = list(_states(6, 3)); state[4] = np.zeros(6)
    assert np.max(_relative(_three(tuple(state))['closed_form'][0], _reference(tuple(state)))) <= 1e-12


def test_non_physical_nodes_take_the_frobenius_fallback_and_are_counted():
    n, te, ti, b, tau, scale, jump = (x[:8].copy() for x in _states(8, 4))
    n[0] = FLOOR/2; te[1] = -1.; ti[2] = -.2; tau[3] = -1.; n[4] = np.nan; te[5] = FLOOR
    state = (n, te, ti, b, tau, scale, jump)
    matrix = np.asarray(_matrix(*(jnp.asarray(x) for x in state[:6])))
    action, invalid = _three(state)['closed_form']
    np.testing.assert_array_equal(invalid, [True]*6+[False]*2)
    fallback = np.linalg.norm(matrix, axis=(-2, -1))[:, None]*jump
    np.testing.assert_allclose(action[[0, 1, 2, 3, 5]], fallback[[0, 1, 2, 3, 5]], rtol=1e-14)
    assert np.isfinite(action[6:]).all()
    # a node that fails the 4x4 test as well (complex spectrum, Ti < 0) has the very same fallback in both methods
    four = _three(state)['lapack4']
    for i in np.flatnonzero(four[1][:4] & invalid[:4]):
        np.testing.assert_allclose(four[0][i], action[i], rtol=1e-14)
    # reverse-mode derivatives stay finite through the safe-state substitution, M = 0 and Ti = 0 included
    ti0 = ti.copy(); ti0[6] = 0.; scale0 = scale.copy(); scale0[7] = 0.

    def fn(n_, te_, ti_, b_, tau_, scale_, jump_):
        m = _matrix(n_, te_, ti_, b_, tau_, scale_)
        return jnp.sum(_absolute_action_closed_form(n_, te_, ti_, b_, tau_, scale_, m, jump_, FLOOR)[0])
    grads = jax.grad(fn, argnums=range(7))(*(jnp.asarray(x) for x in (n, te, ti0, b, tau, scale0, jump)))
    for g in grads:
        assert np.isfinite(np.asarray(g)[[0, 1, 2, 3, 5, 6, 7]]).all()


def _function(method, state):
    """``(n, Te, Ti, jump) -> |M| jump`` of one method at fixed ``B, tau, scale``."""
    n, te, ti, b, tau, scale, jump = (jnp.asarray(x) for x in state)

    def fn(n_, te_, ti_, jump_):
        m = _matrix(n_, te_, ti_, b, tau, scale)
        if method == 'closed_form':
            return _absolute_action_closed_form(n_, te_, ti_, b, tau, scale, m, jump_, FLOOR)[0]
        if method == 'block_lapack':
            return _p06_absolute_action_block(m, jump_)[0]
        return _p06_absolute_action(m, jump_)[0]
    return fn, (n, te, ti, jump)


@pytest.mark.parametrize('method', ('block_lapack', 'closed_form'))
def test_jvp_matches_finite_differences_and_the_4x4_jvp(method):
    state = _states(30, 5, n=(-1, 1), te=(-1, 1), r=(-2, 1.5), tau=(-.3, .3))
    fn, primals = _function(method, state)
    ref_fn, _ = _function('lapack4', state)
    tangents = tuple(jnp.asarray(t) for t in (np.random.default_rng(6).normal(size=p.shape)*np.abs(p).mean()
                                               for p in primals))
    _, jvp = jax.jvp(fn, primals, tangents)
    _, jvp4 = jax.jvp(ref_fn, primals, tangents)
    eps = 1e-6
    plus = fn(*(p+eps*t for p, t in zip(primals, tangents)))
    minus = fn(*(p-eps*t for p, t in zip(primals, tangents)))
    finite = (plus-minus)/(2*eps)
    scale = np.max(np.abs(np.asarray(jvp)), axis=-1, keepdims=True)
    assert np.max(np.abs(np.asarray(jvp)-np.asarray(finite))/scale) <= 2e-6
    ok = ~np.asarray(_p06_absolute_action(_matrix(*(jnp.asarray(x) for x in state[:6])), jnp.asarray(state[6]))[1])
    assert ok.sum() >= 25
    assert np.max(np.abs(np.asarray(jvp)-np.asarray(jvp4))[ok]/scale[ok]) <= 1e-8


def test_block_lapack_fallback_flag_and_tangent():
    matrix = np.zeros((1, 1, 4, 4)); matrix[0, 0, 0, 1] = -1; matrix[0, 0, 1, 0] = 1       # complex spectrum
    jump = np.array([[[.3, -.2, .1, .4]]])
    action, fallback = _p06_absolute_action_block(jnp.asarray(matrix), jnp.asarray(jump))
    assert int(np.sum(np.asarray(fallback))) == 1
    np.testing.assert_allclose(action, np.linalg.norm(matrix[0, 0])*jump, rtol=0, atol=1e-14)
    tangent = np.zeros_like(matrix); tangent[0, 0, 0, 0] = .1
    _, dot = jax.jvp(lambda m: _p06_absolute_action_block(m, jnp.asarray(jump))[0], (jnp.asarray(matrix),),
                     (jnp.asarray(tangent),))
    eps = 1e-6
    finite = (_p06_absolute_action_block(jnp.asarray(matrix+eps*tangent), jnp.asarray(jump))[0]
              - _p06_absolute_action_block(jnp.asarray(matrix-eps*tangent), jnp.asarray(jump))[0])/(2*eps)
    np.testing.assert_allclose(dot, finite, rtol=1e-6, atol=1e-9)


def _hsx_args(n, state=0):
    with np.load(DATA/f'P06_N{n}.npz', allow_pickle=False) as z:
        a = {name: z[name] for name in z.files}
    return a, (a['central'][state], a['lower'][state], a['upper'][state], a['bmag'], a['normal'],
               a['quadrature_weight'], a['wall_mask'], a['collapsed_mask'])


@pytest.mark.parametrize('method', ABSOLUTE_METHODS)
def test_real_hsx_face_rows_for_every_method(method):
    a, args = _hsx_args(32)
    lo, hi, spectral, positivity, wall = p06_characteristic_face_correction(*args, absolute_method=method)
    np.testing.assert_allclose(lo, a['expected_lower'][0], rtol=0, atol=2e-9)
    np.testing.assert_allclose(hi, a['expected_upper'][0], rtol=0, atol=2e-9)
    assert int(spectral) == int(positivity) == 0


def test_default_method_is_bitwise_the_4x4_action():
    _, args = _hsx_args(32, state=1)
    default = p06_characteristic_face_correction(*args)
    explicit = p06_characteristic_face_correction(*args, absolute_method='lapack4')
    for x, y in zip(default, explicit):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    # the 4x4 helper is untouched by the selector
    state = _states(10, 7)
    matrix = _matrix(*(jnp.asarray(x) for x in state[:6]))
    action, _ = _p06_absolute_action(matrix, jnp.asarray(state[6]))
    assert np.array_equal(np.asarray(action), _three(state)['lapack4'][0])


@pytest.mark.parametrize('method', ('block_lapack', 'closed_form'))
def test_methods_agree_with_the_4x4_action_through_the_face_correction(method):
    rng = np.random.default_rng(8)
    faces, nodes = 7, 9
    central = np.concatenate([.6+rng.random((faces, nodes, 3)), .1*rng.normal(size=(faces, nodes, 1))], axis=-1)
    lower, upper = central*(1+.05*rng.normal(size=central.shape)), central*(1+.05*rng.normal(size=central.shape))
    B = 1.+.4*rng.random((faces, nodes)); normal = .2*rng.normal(size=(faces, nodes)); weight = rng.random((faces, nodes))/nodes
    wall = np.zeros(faces, bool); wall[[1, 4]] = True
    collapsed = np.zeros(faces, bool); collapsed[0] = True
    run = lambda m: jax.jit(lambda c, lo, up: p06_characteristic_face_correction(
        c, lo, up, B, normal, weight, wall, collapsed, absolute_method=m))(central, lower, upper)
    reference, result = run('lapack4'), run(method)
    for x, y in zip(reference[:2], result[:2]):
        np.testing.assert_allclose(np.asarray(y), np.asarray(x), rtol=0, atol=1e-12*np.max(np.abs(np.asarray(x))))
    assert int(reference[2]) == int(result[2]) == 0
    assert int(reference[3]) == int(result[3]) and int(reference[4]) == int(result[4])
    # JVP in the state, against the 4x4 JVP
    tangent = rng.normal(size=central.shape)*.01
    jvp = lambda m: jax.jit(lambda c, t: jax.jvp(lambda cc: p06_characteristic_face_correction(
        cc, lower, upper, B, normal, weight, wall, collapsed, absolute_method=m)[:2], (c,), (t,))[1])(central, tangent)
    for x, y in zip(jvp('lapack4'), jvp(method)):
        np.testing.assert_allclose(np.asarray(y), np.asarray(x), rtol=0, atol=1e-8*np.max(np.abs(np.asarray(x))))
