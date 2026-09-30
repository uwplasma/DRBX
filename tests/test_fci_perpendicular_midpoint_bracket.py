"""Synthetic checks of the JAX centered midpoint bracket against the host reference."""
from __future__ import annotations

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native.fci_perpendicular_midpoint_bracket import (
    pair_actions, point_bracket, project_raw_to_owners, validate_bracket_inputs)

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
host = pytest.importorskip("p05_direct_midpoint_global.direct_operator")

PAIRS = ((0, 1), (1, 0), (2, 5), (3, 3), (4, 1), (0, 1), (5, 2))


def _data(seed=0, points=200, fields=6):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(points, 3))
    jac = rng.normal(size=points) * 2.0 + np.sign(rng.normal(size=points)) * 0.5   # both signs
    grads = rng.normal(size=(points, 3, fields))
    return h, jac, grads


def _rel(a, b):
    return np.max(np.abs(np.asarray(a) - b)) / np.max(np.abs(b))


def test_point_bracket_matches_host():
    h, jac, grads = _data()
    got = point_bracket(h, jac, grads[:, :, 0], grads[:, :, 1])
    ref = host.point_bracket(h, jac, grads[:, :, 0], grads[:, :, 1])
    assert _rel(got, ref) <= 1e-14
    with pytest.raises(ValueError):
        point_bracket(h, jac, grads[:, :, 0], grads[:, :2, 1])
    with pytest.raises(ValueError):
        point_bracket(h, jac[:-1], grads[:, :, 0], grads[:, :, 1])


def test_pair_actions_and_antisymmetry_match_host():
    h, jac, grads = _data(1)
    actions, defect = pair_actions(h, jac, grads, PAIRS)
    ref, ref_defect = host.direct_pair_actions(h, jac, grads, PAIRS)
    assert actions.shape == ref.shape == (200, len(PAIRS))
    assert _rel(actions, ref) <= 1e-14
    assert abs(float(defect) - ref_defect) <= 1e-15 * max(1.0, np.max(np.abs(ref)))
    assert float(defect) < 1e-13
    # the self pair (3, 3) is exactly antisymmetric
    assert np.max(np.abs(np.asarray(actions)[:, 3])) < 1e-13
    empty, d0 = pair_actions(h, jac, grads, ())
    assert empty.shape == (200, 0) and float(d0) == 0.0
    with pytest.raises(ValueError):
        pair_actions(h, jac, grads, ((0, 6),))
    with pytest.raises(ValueError):
        pair_actions(h, jac, grads[:, :2], PAIRS)


def test_projection_matches_host():
    rng = np.random.default_rng(2)
    raw, owners, k = 200, 37, len(PAIRS)
    owner = np.concatenate([np.arange(owners), rng.integers(0, owners, raw - owners)])
    rng.shuffle(owner)
    action = rng.normal(size=(raw, k))
    volume = rng.uniform(0.1, 2.0, size=raw)
    owner_volume = np.zeros(owners)
    np.add.at(owner_volume, owner, volume)
    ref = host.project_raw_to_owners(action, volume, owner, owner_volume)
    got = project_raw_to_owners(action, volume, owner, owner_volume)
    assert np.bincount(owner).max() > 3
    assert _rel(got, ref) <= 1e-14
    # jnp int32 owners and explicit owner_count work too
    got2 = project_raw_to_owners(action, volume, owner.astype(np.int32), owner_volume, owner_count=owners)
    assert _rel(got2, ref) <= 1e-14
    with pytest.raises(ValueError):
        project_raw_to_owners(action, volume, owner, owner_volume, owner_count=owners + 1)
    with pytest.raises(ValueError):
        project_raw_to_owners(action, volume[:-1], owner, owner_volume)
    # projecting a constant raw action returns that constant
    const = project_raw_to_owners(np.full((raw, 1), 3.0), volume, owner, owner_volume)
    assert np.max(np.abs(np.asarray(const) - 3.0)) < 1e-14


def test_eager_equals_jit_bitwise():
    h, jac, grads = _data(3)
    owner = np.arange(200) % 40
    volume = np.random.default_rng(3).uniform(0.5, 1.5, 200)
    owner_volume = np.bincount(owner, weights=volume, minlength=40)

    # Arrays enter the outer jit as arguments (as the plan does in production): XLA may
    # rewrite division by a *baked-in constant* owner volume, which is not bitwise-stable.
    def full(h, jac, grads, volume, owner, owner_volume):
        a, d = pair_actions(h, jac, grads, PAIRS)
        return a, d, project_raw_to_owners(a, volume, owner, owner_volume)

    args = (h, jac, grads, volume, owner, owner_volume)
    eager = full(*args)
    jitted = jax.jit(full)(*args)
    for e, j in zip(eager, jitted):
        assert np.array_equal(np.asarray(e), np.asarray(j))
    pb = point_bracket(h, jac, grads[:, :, 0], grads[:, :, 1])
    pbj = jax.jit(point_bracket)(h, jac, grads[:, :, 0], grads[:, :, 1])
    assert np.array_equal(np.asarray(pb), np.asarray(pbj))


def test_jvp_is_bilinear_derivative():
    h, jac, grads = _data(4)
    tangent = np.random.default_rng(5).normal(size=grads.shape)
    ia = np.array([p[0] for p in PAIRS]); ib = np.array([p[1] for p in PAIRS])
    _, jvp = jax.jvp(lambda g: pair_actions(h, jac, g, PAIRS)[0], (jnp.asarray(grads),), (jnp.asarray(tangent),))
    expected = np.empty_like(np.asarray(jvp))
    for i, (a, b) in enumerate(PAIRS):
        expected[:, i] = (host.point_bracket(h, jac, tangent[:, :, a], grads[:, :, b])
                          + host.point_bracket(h, jac, grads[:, :, a], tangent[:, :, b]))
    assert _rel(jvp, expected) <= 1e-13
    # the antisymmetry diagnostic carries no derivative
    _, djvp = jax.jvp(lambda g: pair_actions(h, jac, g, PAIRS)[1], (jnp.asarray(grads),), (jnp.asarray(tangent),))
    assert float(djvp) == 0.0
    grad = jax.grad(lambda g: pair_actions(h, jac, g, PAIRS)[1])(jnp.asarray(grads))
    assert float(jnp.max(jnp.abs(grad))) == 0.0


def test_grad_through_projection():
    rng = np.random.default_rng(6)
    raw, owners = 60, 11
    owner = np.concatenate([np.arange(owners), rng.integers(0, owners, raw - owners)])
    action = rng.normal(size=(raw, 3)); weight = rng.normal(size=(owners, 3))
    volume = rng.uniform(0.5, 2.0, raw)
    owner_volume = np.bincount(owner, weights=volume, minlength=owners)

    def loss(a, v):
        return jnp.sum(weight * project_raw_to_owners(a, v, owner, owner_volume))

    ga, gv = jax.grad(loss, argnums=(0, 1))(jnp.asarray(action), jnp.asarray(volume))
    assert _rel(ga, (volume / owner_volume[owner])[:, None] * weight[owner]) <= 1e-14
    # linear in the action: directional derivative equals the loss difference
    t = rng.normal(size=action.shape)
    d = float(loss(action + t, volume) - loss(action, volume))
    assert abs(d - float(jnp.sum(ga * t))) <= 1e-12 * max(1.0, abs(d))
    assert np.isfinite(np.asarray(gv)).all()


def test_validate_bracket_inputs():
    h, jac, grads = _data(7, points=10)
    validate_bracket_inputs(h, jac, grads)
    with pytest.raises(ValueError):
        validate_bracket_inputs(jacobian=np.zeros(3))
    with pytest.raises(FloatingPointError):
        validate_bracket_inputs(actions=np.array([np.nan]))
    with pytest.raises(ValueError):
        validate_bracket_inputs(raw_owner=np.array([0, 3]), owner_volume=np.ones(3))
    with pytest.raises(ValueError):
        validate_bracket_inputs(owner_volume=np.array([1.0, 0.0]))
