from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

PyTree = object


def tree_add(a: PyTree, b: PyTree, scale: float = 1.0) -> PyTree:
    def add(x, y):
        if x is None or y is None:
            return None
        return x + scale * y

    return jax.tree_util.tree_map(add, a, b, is_leaf=lambda x: x is None)


def rk4_step(rhs: Callable[[float, PyTree], PyTree], t: float, y: PyTree, dt: float) -> PyTree:
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, tree_add(y, k1, 0.5 * dt))
    k3 = rhs(t + 0.5 * dt, tree_add(y, k2, 0.5 * dt))
    k4 = rhs(t + dt, tree_add(y, k3, dt))
    acc = tree_add(k1, k2, 2.0)
    acc = tree_add(acc, k3, 2.0)
    acc = tree_add(acc, k4, 1.0)
    return tree_add(y, acc, dt / 6.0)


def rk4_step_with_phi(
    rhs_with_phi: Callable[[float, PyTree, PyTree | None], tuple[PyTree, PyTree]],
    t: float,
    y: PyTree,
    dt: float,
    phi_guess: PyTree | None,
) -> tuple[PyTree, PyTree]:
    k1, phi1 = rhs_with_phi(t, y, phi_guess)
    k2, phi2 = rhs_with_phi(t + 0.5 * dt, tree_add(y, k1, 0.5 * dt), phi1)
    k3, phi3 = rhs_with_phi(t + 0.5 * dt, tree_add(y, k2, 0.5 * dt), phi2)
    k4, phi4 = rhs_with_phi(t + dt, tree_add(y, k3, dt), phi3)
    acc = tree_add(k1, k2, 2.0)
    acc = tree_add(acc, k3, 2.0)
    acc = tree_add(acc, k4, 1.0)
    return tree_add(y, acc, dt / 6.0), phi4


def build_rk4_scan(
    rhs: Callable[[float, PyTree], PyTree],
    dt: float,
    steps: int,
    save_every: int,
    diag_fn: Callable[[float, PyTree], PyTree],
    *,
    rhs_remat: bool = False,
) -> Tuple[Callable[[PyTree], Tuple[PyTree, PyTree]], int, int]:
    """Return a JIT-compiled runner and output counts.

    Returns (runner, nsave, rem) where:
      - runner(state) -> (final_state, diag_series)
      - diag_series is a pytree with leading dimension nsave
    """
    dt = float(dt)
    steps = int(steps)
    save_every = int(save_every)
    if save_every <= 0:
        raise ValueError("save_every must be > 0")
    if steps < 0:
        raise ValueError("steps must be >= 0")

    nblocks = steps // save_every
    rem = steps % save_every
    nsave = nblocks + 1 + (1 if rem > 0 else 0)

    rhs_fn = jax.checkpoint(rhs) if rhs_remat else rhs

    def inner(carry, _):
        t, y = carry
        y = rk4_step(rhs_fn, t, y, dt)
        return (t + dt, y), None

    def block(carry, _):
        t, y = carry
        (t, y), _ = jax.lax.scan(inner, (t, y), None, length=save_every)
        diag = diag_fn(t, y)
        return (t, y), diag

    def _concat_diag(diag0, diags):
        return jax.tree_util.tree_map(
            lambda d0, ds: jnp.concatenate([d0[jnp.newaxis, ...], ds], axis=0),
            diag0,
            diags,
        )

    def _append_diag(diags, diag_last):
        return jax.tree_util.tree_map(
            lambda ds, dl: jnp.concatenate([ds, dl[jnp.newaxis, ...]], axis=0),
            diags,
            diag_last,
        )

    def run(state: PyTree):
        t0 = jnp.asarray(0.0)
        diag0 = diag_fn(t0, state)
        if nblocks > 0:
            (t, y), diags = jax.lax.scan(block, (t0, state), None, length=nblocks)
            diags = _concat_diag(diag0, diags)
        else:
            t, y = t0, state
            diags = jax.tree_util.tree_map(lambda d0: d0[jnp.newaxis, ...], diag0)
        if rem > 0:
            (t, y), _ = jax.lax.scan(inner, (t, y), None, length=rem)
            diag_last = diag_fn(t, y)
            diags = _append_diag(diags, diag_last)
        return y, diags

    return jax.jit(run), nsave, rem


def build_rk4_scan_cached(
    rhs_with_phi: Callable[[float, PyTree, PyTree | None], tuple[PyTree, PyTree]],
    dt: float,
    steps: int,
    save_every: int,
    diag_fn: Callable[[float, PyTree], PyTree],
    *,
    rhs_remat: bool = False,
) -> Tuple[Callable[[PyTree], Tuple[PyTree, PyTree]], int, int]:
    """Return a JIT-compiled runner with Poisson warm-start caching."""

    dt = float(dt)
    steps = int(steps)
    save_every = int(save_every)
    if save_every <= 0:
        raise ValueError("save_every must be > 0")
    if steps < 0:
        raise ValueError("steps must be >= 0")

    nblocks = steps // save_every
    rem = steps % save_every
    nsave = nblocks + 1 + (1 if rem > 0 else 0)

    rhs_fn = jax.checkpoint(rhs_with_phi) if rhs_remat else rhs_with_phi

    def inner(carry, _):
        t, y, phi_guess = carry
        y, phi_guess = rk4_step_with_phi(rhs_fn, t, y, dt, phi_guess)
        return (t + dt, y, phi_guess), None

    def block(carry, _):
        t, y, phi_guess = carry
        (t, y, phi_guess), _ = jax.lax.scan(inner, (t, y, phi_guess), None, length=save_every)
        diag = diag_fn(t, y)
        return (t, y, phi_guess), diag

    def _concat_diag(diag0, diags):
        return jax.tree_util.tree_map(
            lambda d0, ds: jnp.concatenate([d0[jnp.newaxis, ...], ds], axis=0),
            diag0,
            diags,
        )

    def _append_diag(diags, diag_last):
        return jax.tree_util.tree_map(
            lambda ds, dl: jnp.concatenate([ds, dl[jnp.newaxis, ...]], axis=0),
            diags,
            diag_last,
        )

    def run(state: PyTree):
        t0 = jnp.asarray(0.0)
        phi_guess0 = jnp.zeros_like(state.omega)
        diag0 = diag_fn(t0, state)
        if nblocks > 0:
            (t, y, phi_guess), diags = jax.lax.scan(
                block, (t0, state, phi_guess0), None, length=nblocks
            )
            diags = _concat_diag(diag0, diags)
        else:
            t, y, phi_guess = t0, state, phi_guess0
            diags = jax.tree_util.tree_map(lambda d0: d0[jnp.newaxis, ...], diag0)
        if rem > 0:
            (t, y, phi_guess), _ = jax.lax.scan(inner, (t, y, phi_guess), None, length=rem)
            diag_last = diag_fn(t, y)
            diags = _append_diag(diags, diag_last)
        return y, diags

    return jax.jit(run), nsave, rem
