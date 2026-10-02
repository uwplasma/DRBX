"""Algebra/JAX verification; real-HSX qualification is a separate research audit."""
import math
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry.Bfield_evaluator import ComponentSplineBFieldEvaluator
from drbx.geometry.jax_bfield_evaluator import JaxComponentSplineBFieldEvaluator
from drbx.geometry.compact_toroidal import hermite_coefficients, compact_weights


def build(m, n=24, perturb=None):
    r = np.linspace(1, 2, 7); z = np.linspace(-.5, .5, 8)
    phi = .17 + np.arange(n) * (np.pi / 2) / n
    pp, zz, rr = np.meshgrid(phi, z, r, indexing='ij')
    values = np.stack((np.sin(4*pp), np.cos(8*pp), np.ones_like(pp)), axis=-1)
    if perturb is not None:
        values[perturb] += 1e4
    return ComponentSplineBFieldEvaluator(r, phi, z, values, nfp=4,
                                          toroidal_method=f'compact_c{m}'), values


@pytest.mark.parametrize('m', [2, 3])
def test_polynomial_reproduction_and_shared_jets(m):
    nodes = np.arange(-m, m+2, dtype=float)
    x = np.linspace(0, 1, 31)
    for degree in range(2*m+1):
        np.testing.assert_allclose(compact_weights(x, m) @ nodes**degree, x**degree, atol=3e-12)
    coef = hermite_coefficients(m)
    # Compare left t=1 and right t=0 jets in one common donor numbering.
    for d in range(m+1):
        left = np.zeros(2*m+3); right = left.copy()
        left[:-1] = sum(math.factorial(k)/math.factorial(k-d)*coef[k] for k in range(d, len(coef)))
        right[1:] = math.factorial(d)*coef[d]
        np.testing.assert_allclose(left, right, atol=2e-12)


@pytest.mark.parametrize('m', [2, 3])
def test_raw_plane_locality_nodes_and_unchanged_rz(m):
    host, values = build(m)
    changed, _ = build(m, perturb=12)
    points = np.array([[1.32, host.phi[2]+.4*host._dphi, .12], [1.61, host.phi[0]-.1*host._dphi, -.27]])
    np.testing.assert_array_equal(host.evaluate_cylindrical(points), changed.evaluate_cylindrical(points))
    grid = np.stack(np.meshgrid(host.phi, host.Z, host.R, indexing='ij'), axis=-1)[...,[2,0,1]]
    np.testing.assert_allclose(host.evaluate_cylindrical(grid), values, atol=5e-13)
    # Arbitrary R/Z values: exact parity on every source phi plane.
    rng = np.random.default_rng(19)
    arbitrary = rng.normal(size=values.shape)
    old = ComponentSplineBFieldEvaluator(host.R, host.phi, host.Z, arbitrary, nfp=4)
    new = ComponentSplineBFieldEvaluator(host.R, host.phi, host.Z, arbitrary, nfp=4, toroidal_method=f'compact_c{m}')
    p = np.column_stack((rng.uniform(1,2,70), rng.choice(host.phi,70), rng.uniform(-.5,.5,70)))
    np.testing.assert_allclose(old.evaluate_cylindrical(p), new.evaluate_cylindrical(p), atol=2e-12)


@pytest.mark.parametrize('m', [2, 3])
def test_jax_parity_periodicity_derivatives_and_pytree(m):
    host, _ = build(m)
    state = JaxComponentSplineBFieldEvaluator.from_evaluator(host)
    rng = np.random.default_rng(12)
    p = np.column_stack((rng.uniform(1,2,31),rng.uniform(-5,5,31),rng.uniform(-.5,.5,31)))
    out = jax.jit(lambda s,p:s.evaluate_cylindrical(p))(state,p)
    np.testing.assert_allclose(out, host.evaluate_cylindrical(p), atol=3e-13)
    f = lambda phi:state.evaluate_cylindrical(jnp.array([1.4,phi,.1]))
    for d in range(m+1):
        vals = jax.jit(jax.vmap(f))(jnp.array([host.phi[0]-.013,host.phi[0]-.013+host.period]))
        np.testing.assert_allclose(vals[0],vals[1],rtol=1e-10,atol=1e-9)
        for knot in (host.phi[0], host.phi[6]):
            # Smooth limits across interior knots and the periodic seam.
            vals = jax.jit(jax.vmap(f))(jnp.array([knot-1e-11,knot+1e-11]))
            np.testing.assert_allclose(vals[0],vals[1],rtol=0,atol=5e-6)
        f = jax.jacfwd(f)
    grad = jax.jacfwd(state.evaluate_cylindrical)(jnp.array([1.4,.48,.1]))
    eps=1e-6
    fd=(host.evaluate_cylindrical([1.4,.48+eps,.1])-host.evaluate_cylindrical([1.4,.48-eps,.1]))/(2*eps)
    np.testing.assert_allclose(grad[:,1],fd,rtol=2e-8,atol=1e-8)
    xyz = np.stack((p[:,0]*np.cos(p[:,1]),p[:,0]*np.sin(p[:,1]),p[:,2]),axis=-1)
    np.testing.assert_allclose(jax.jit(state.evaluate_cartesian)(xyz),host.evaluate_cartesian(xyz),atol=2e-12)


@pytest.mark.parametrize('m', [2,3])
def test_smooth_toroidal_refinement(m):
    errors=[]
    for n in (24,48,96):
        host,_=build(m,n)
        phi=host.phi[0]+(np.arange(n)+.37)*host._dphi
        p=np.column_stack((np.full(n,1.4),phi,np.full(n,.1)))
        truth=np.stack((np.sin(4*phi),np.cos(8*phi),np.ones(n)),axis=-1)
        errors.append(np.max(abs(host.evaluate_cylindrical(p)-truth)))
    assert min(np.log2(np.array(errors[:-1])/errors[1:])) > 2*m+.5


def test_invalid_modes_and_bounds():
    h,v=build(2)
    for kwargs in ({'toroidal_method':'bad'},{'toroidal_method':'compact_c2','method':'linear'},
                   {'toroidal_method':'compact_c2','extrapolate':True}):
        with pytest.raises(ValueError):
            ComponentSplineBFieldEvaluator(h.R,h.phi,h.Z,v,nfp=4,**kwargs)
    with pytest.raises(ValueError): h.evaluate_cylindrical([.9,.2,0])

