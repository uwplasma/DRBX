"""One exact adjoint pair for every parallel term of the HSX FCI backend.

The production recipe (``build_support_pair`` from the scatter of a mapped
divergence, as ``LocalFciDrbEBRhs._fci_support_core_pair``) on a small sheared FCI
map: periodic ``x``, parallel ``z`` with forward/backward maps shifting ``x`` by
``s(x) dz`` (linear interpolation).  Rows whose map leaves the ``z`` range are walls.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_braginskii.native.fci_drb_EB_rhs import LocalFciDrbEBRhs
from drbx.fci_braginskii.native.fci_support_pair import build_support_pair

jax.config.update("jax_enable_x64", True)


def _pair(nx=12, nz=14, *, walls=True, volume_variation=0.3, slope=lambda x: 0.5 + 0.4 * np.sin(2 * np.pi * x)):
    x = (np.arange(nx) + 0.5) / nx
    z = (np.arange(nz) + 0.5) / nz
    dz = 1.0 / nz
    columns = np.arange(nx)
    legs = {}
    for sign in (1, -1):
        position = columns + sign * slope(x) * dz * nx
        i0 = np.floor(position).astype(int)
        legs[sign] = (i0 % nx, (i0 + 1) % nx, position - i0)

    def interpolate(q, sign):
        i0, i1, w = legs[sign]
        shifted = jnp.roll(q, -sign, axis=1)
        return (1.0 - w)[:, None] * shifted[i0] + w[:, None] * shifted[i1]

    core = np.ones((nx, nz), dtype=bool)
    if walls:  # wall rows close with the owner value (zero-gradient ghost), as on HSX
        core[:, 0] = core[:, -1] = True

    def divergence0(q):
        plus, minus = interpolate(q, 1), interpolate(q, -1)
        if walls:
            plus = plus.at[:, -1].set(q[:, -1])
            minus = minus.at[:, 0].set(q[:, 0])
        return (plus - minus) / (2.0 * dz)

    mass = jnp.asarray((1.0 + volume_variation * np.cos(2 * np.pi * x))[:, None] * np.ones((1, nz)) / (nx * nz))
    gradient, divergence = build_support_pair(
        divergence0, mass, active=jnp.ones(mass.shape, bool), core=jnp.asarray(core)
    )
    return gradient, divergence, mass, x, z


def _dense(operator, shape):
    n = int(np.prod(shape))
    return np.asarray(jax.vmap(lambda e: operator(e.reshape(shape)).reshape(-1))(jnp.eye(n))).T


@pytest.mark.parametrize("walls", [True, False])
def test_pair_is_an_exact_adjoint_with_sum_free_divergence(walls):
    gradient, divergence, mass, _, _ = _pair(walls=walls)
    g, d = _dense(gradient, mass.shape), _dense(divergence, mass.shape)
    v = np.asarray(mass).reshape(-1)
    assert np.linalg.norm(d + (g.T * v[None, :]) / v[:, None]) <= 1e-13 * np.linalg.norm(d)
    assert np.max(np.abs(v @ d)) <= 1e-13 * np.max(np.abs(v) @ np.abs(d))
    assert np.max(np.abs(g.sum(axis=1))) <= 1e-13 * np.max(np.abs(g).sum(axis=1))


def test_conduction_and_parallel_diffusion_are_symmetric_negative_semidefinite():
    gradient, divergence, mass, x, z = _pair()
    kappa = jnp.asarray(1.0 + 0.5 * np.sin(2 * np.pi * x)[:, None] * np.cos(2 * np.pi * z)[None, :] ** 2)
    v = np.asarray(mass).reshape(-1)
    for coefficient in (1.0, kappa):
        vl = v[:, None] * _dense(lambda f: divergence(coefficient * gradient(f)), mass.shape)
        assert np.linalg.norm(vl - vl.T) <= 1e-13 * np.linalg.norm(vl)
        assert np.max(np.linalg.eigvalsh(0.5 * (vl + vl.T))) <= 1e-13 * np.linalg.norm(vl)
        assert np.max(np.abs(vl.sum(axis=0))) <= 1e-13 * np.linalg.norm(vl)


@pytest.mark.parametrize("which", ["gradient", "divergence", "conduction"])
def test_gradient_divergence_and_conduction_mms_converge_at_second_order(which):
    slope = 0.45
    u = lambda x, z: 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x)
    kappa = lambda x, z: 1.0 + 0.3 * jnp.sin(2 * jnp.pi * x)
    f = lambda x, z: jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * z) + 0.3 * jnp.cos(2 * jnp.pi * (x - z))

    def d_par(g):
        return lambda x, z: slope * jax.grad(g, 0)(x, z) + jax.grad(g, 1)(x, z)

    errors = []
    for n in (16, 32, 64):
        gradient, divergence, _, x, z = _pair(n, n, walls=False, volume_variation=0.0, slope=lambda x: slope + 0 * x)
        grid = lambda g: np.asarray(jax.vmap(jax.vmap(g, (None, 0)), (0, None))(jnp.asarray(x), jnp.asarray(z)))
        field = jnp.asarray(grid(f))
        if which == "gradient":
            numeric, exact = gradient(field), grid(d_par(f))
        elif which == "divergence":
            numeric, exact = divergence(jnp.asarray(grid(u)) * field), grid(d_par(lambda x, z: u(x, z) * f(x, z)))
        else:
            numeric = divergence(jnp.asarray(grid(kappa)) * gradient(field))
            exact = grid(d_par(lambda x, z: kappa(x, z) * d_par(f)(x, z)))
        errors.append(float(np.max(np.abs(np.asarray(numeric) - exact))))
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(orders > 1.8), (which, errors, orders)


def test_closed_box_conserves_mass_momentum_and_thermal_energy_up_to_reported_sources():
    tau, mu = 0.8, 25.0
    gradient, divergence, mass, _, _ = _pair(walls=False)
    rng = np.random.default_rng(4)
    fld = lambda lo, hi: jnp.asarray(rng.uniform(lo, hi, mass.shape))
    primitive = {"density": 1.5 + 0.3 * fld(0, 1), "Te": 1.0 + 0.2 * fld(0, 1), "Ti": 0.8 + 0.2 * fld(0, 1),
                 "Vi": 0.3 * fld(-1, 1), "Ve": 0.5 * fld(-1, 1)}
    host = SimpleNamespace(parameters=SimpleNamespace(tau=tau, mi_over_me=mu))
    rates, src = LocalFciDrbEBRhs._conservative_parallel_rates(host, primitive, gradient, divergence)
    n, te, ti, vi, ve = (np.asarray(primitive[k]) for k in ("density", "Te", "Ti", "Vi", "Ve"))
    dn, dte, dti, dvi, dve = (np.asarray(rates[..., k]) for k in range(5))
    v, s = np.asarray(mass), np.asarray(src)
    assert abs(np.sum(v * dn)) <= 1e-13 * np.sum(v * n)
    np.testing.assert_allclose(np.sum(v * 1.5 * (te * dn + n * dte)), np.sum(v * s[..., 0]), atol=1e-12)
    np.testing.assert_allclose(np.sum(v * 1.5 * tau * (ti * dn + n * dti)), np.sum(v * s[..., 1]), atol=1e-12)
    np.testing.assert_allclose(np.sum(v * (vi * dn + n * dvi)), np.sum(v * s[..., 2]), atol=1e-12)
    g_pe, g_te = (np.asarray(a) for a in gradient(jnp.stack((jnp.asarray(n * te), jnp.asarray(te)))))
    np.testing.assert_allclose(np.sum(v * (ve * dn + n * dve)), -mu * np.sum(v * (g_pe + 0.71 * n * g_te)), atol=1e-11)
