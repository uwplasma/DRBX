from __future__ import annotations

from dataclasses import dataclass

import diffrax
import jax
import jax.numpy as jnp
from jaxopt import linear_solve
import numpy as np


def _pack_complex_pytree_to_real(x):
    leaves, treedef = jax.tree_util.tree_flatten(x)
    specs = []
    parts = []
    for leaf in leaves:
        leaf = jnp.asarray(leaf)
        if jnp.iscomplexobj(leaf):
            parts.append(jnp.ravel(leaf.real))
            parts.append(jnp.ravel(leaf.imag))
            specs.append((leaf.shape, leaf.dtype, True))
        else:
            parts.append(jnp.ravel(leaf))
            specs.append((leaf.shape, leaf.dtype, False))
    return jnp.concatenate(parts), (treedef, specs)


def _unpack_real_to_complex_pytree(v: jnp.ndarray, meta):
    treedef, specs = meta
    leaves = []
    i = 0
    for shape, dtype, is_complex in specs:
        size = int(np.prod(shape))
        if is_complex:
            re = v[i : i + size].reshape(shape)
            im = v[i + size : i + 2 * size].reshape(shape)
            leaves.append((re + 1j * im).astype(dtype))
            i += 2 * size
        else:
            leaves.append(v[i : i + size].reshape(shape).astype(dtype))
            i += size
    return jax.tree_util.tree_unflatten(treedef, leaves)


@dataclass
class GrowthRateResult:
    gamma: float
    omega: float
    t: np.ndarray
    log_norm: np.ndarray


def estimate_growth_rate(
    matvec,
    y0,
    *,
    tmax: float = 200.0,
    dt0: float = 0.05,
    nsave: int = 200,
    fit_window: float = 0.5,
    method: str = "renormalized",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 500000,
) -> GrowthRateResult:
    """Estimate growth rate from initial-value evolution dv/dt = A v.

    Returns gamma from a least-squares fit of log(||v||) vs t over the last `fit_window`
    fraction of the time history.
    """

    y0_vec, meta = _pack_complex_pytree_to_real(y0)

    def matvec_real(v):
        v_c = _unpack_real_to_complex_pytree(v, meta)
        dv_c = matvec(v_c)
        dv, _ = _pack_complex_pytree_to_real(dv_c)
        return dv

    if method not in {"direct", "renormalized"}:
        raise ValueError("method must be 'direct' or 'renormalized'")

    ts = jnp.linspace(0.0, tmax, nsave)
    saveat = diffrax.SaveAt(ts=ts)
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
    if method == "direct":
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: matvec_real(y)),
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=tmax,
            dt0=dt0,
            y0=y0_vec,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
        norms = jnp.linalg.norm(sol.ys, axis=-1)
        logn = jnp.log(jnp.maximum(norms, 1e-300))
        phase = jnp.zeros_like(logn)
    else:
        nrm0 = jnp.linalg.norm(y0_vec)
        u0 = y0_vec / nrm0
        a0 = jnp.array(0.0, dtype=u0.dtype)
        b0 = jnp.array(0.0, dtype=u0.dtype)

        def rayleigh_parts(u, Au):
            treedef, specs = meta
            _ = treedef  # structure is fixed; only specs matter here.
            re = jnp.array(0.0, dtype=u.dtype)
            im = jnp.array(0.0, dtype=u.dtype)
            denom = jnp.array(0.0, dtype=u.dtype)
            i = 0
            for shape, _dtype, is_complex in specs:
                size = int(np.prod(shape))
                if is_complex:
                    ur = u[i : i + size]
                    ui = u[i + size : i + 2 * size]
                    Ar = Au[i : i + size]
                    Ai = Au[i + size : i + 2 * size]
                    re = re + jnp.dot(ur, Ar) + jnp.dot(ui, Ai)
                    im = im + jnp.dot(ur, Ai) - jnp.dot(ui, Ar)
                    denom = denom + jnp.dot(ur, ur) + jnp.dot(ui, ui)
                    i = i + 2 * size
                else:
                    uu = u[i : i + size]
                    AA = Au[i : i + size]
                    re = re + jnp.dot(uu, AA)
                    denom = denom + jnp.dot(uu, uu)
                    i = i + size
            return re / denom, im / denom

        def rhs(t, state, args):
            u, a, b = state
            Au = matvec_real(u)
            gamma_inst, omega_inst = rayleigh_parts(u, Au)
            du = Au - gamma_inst * u
            da = gamma_inst
            db = omega_inst
            return (du, da, db)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=tmax,
            dt0=dt0,
            y0=(u0, a0, b0),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
        logn = sol.ys[1]
        phase = sol.ys[2]

    i0 = int((1.0 - fit_window) * int(ts.size))
    tt = ts[i0:]
    yy = logn[i0:]
    A = jnp.stack([tt, jnp.ones_like(tt)], axis=1)

    AtA = A.T @ A + 1e-14 * jnp.eye(2, dtype=A.dtype)
    Atb = A.T @ yy
    gamma, _c = linear_solve.solve_cholesky(lambda x: AtA @ x, Atb)

    pp = phase[i0:]
    Atb2 = A.T @ pp
    omega, _c2 = linear_solve.solve_cholesky(lambda x: AtA @ x, Atb2)

    return GrowthRateResult(
        gamma=float(gamma),
        omega=float(omega),
        t=np.asarray(ts),
        log_norm=np.asarray(logn),
    )
