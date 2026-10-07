"""Preflight of the P10 evolved MMS (chunk C5): spectral radius of the stage Jacobian, the RK4 step, term shares, the bracket
energy abscissa and the potential-solve cost for one ``(arm, N, mode, pattern)``; writes ``preflight.json``.

* :func:`spectral_radius`     largest ``|lambda|`` of a real linear operator given as a flat-vector matvec (ARPACK ``eigs`` 'LM',
                              power iteration with a two-dimensional Krylov (Rayleigh-Ritz) estimate as the fallback);
* :func:`rightmost_eigenvalue` the eigenvalue of largest real part of the same operator (ARPACK ``eigs`` 'LR'): the growth
                              rate of the linearized dynamics (``lambda_max`` bounds the step, ``Re lambda`` the growth);
* :func:`stage_linearization` the matvec ``v -> J v`` of the stage RHS (``model.make_stage_rhs``) at the exact state
                              ``q_bar(t)`` (``jax.jvp`` with respect to the state; ``t`` and the carry fixed). State vectors are
                              the flat ``(E, P, 4)`` arrays (C order), fields ``(n, Te, Ti, Omega)``;
* :func:`dt_rule`             ``dt = T / ceil(T / (0.5 * 2.6 / lambda_max))``: half the RK4 stability radius ``2.6``;
* :func:`term_shares`         ``||term||_H / ||d_t q_bar||_H`` per field, term and region of the mode's term set;
* :func:`max_half_c`          ``max (c / 2)`` of the E x B compressibility ``c = -2 T(1)`` of the (scaled) bracket, the
                              abscissa of the bracket's energy identity (``||g(t)|| <= exp(max(c/2) t) ||g(0)||``);
* :func:`cg_iterations`       iterations of the potential solve at ``q_bar(t)`` (cold and warm started; coupled mode);
* :func:`preflight`           the assembly (``preflight.json``, schema ``drbx.p10-preflight-v1``).

    python -m p10_evolved_mms.preflight (--bundle DIR | --arm ARM --n N) --mode MODE --pattern PATTERN [--T T] --out DIR

(run from ``scripts/``).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla

jax.config.update("jax_enable_x64", True)
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native.fci_nodal_perpendicular_rhs import solve_potential_jit                    # noqa: E402
from drbx.native.fci_perpendicular_sbp_bracket import compressibility, velocity_flux      # noqa: E402
from drbx.native.fci_perpendicular_sbp_norms import h_weights, ring_region_masks           # noqa: E402
from p10_evolved_mms import source as src                                                    # noqa: E402
from p10_evolved_mms.fields import CONFIG, FIELDS, MmsParams, default_params               # noqa: E402

SCHEMA = "drbx.p10-preflight-v1"
FIELD_LABELS = ("n", "Te", "Ti", "Omega")
MODES = ("diffusion", "hyperbolic", "coupled")
#: the terms of each stage mode (the contract of ``model.py``)
MODE_TERMS = {"diffusion": ("diffusion",), "hyperbolic": ("bracket", "curvature"), "coupled": ("bracket", "curvature", "diffusion")}
RK4_STABILITY_RADIUS = 2.6
RK4_SAFETY = 0.5
DT_RULE = "T / ceil(T / (0.5 * 2.6 / lambda_max))"


# ---------------------------------------------------------------------------------------------------------------------
# spectral radius
# ---------------------------------------------------------------------------------------------------------------------
def _sorted_eigs(ev) -> np.ndarray:
    ev = np.asarray(ev, dtype=np.complex128).reshape(-1)
    return ev[np.argsort(-np.abs(ev), kind="stable")]


def _eig_pairs(ev, k) -> list:
    return [[float(e.real), float(e.imag)] for e in _sorted_eigs(ev)[:k]]


def _power_ritz(mv, x0, iters: int, tol: float):
    """Power iteration with a two-dimensional Krylov estimate.

    With unit vectors ``x_k`` and ``s_k = ||A x_k||`` (``A x_k = s_k x_{k+1}``), the least-squares fit of
    ``A^2 x_{k-1} = a A x_{k-1} - b x_{k-1}`` gives the Ritz values of ``span{x_{k-1}, A x_{k-1}}``: the roots of
    ``z^2 - a z + b``. Their largest modulus is the estimate (``sqrt(b)`` for a dominant complex pair, which a plain power
    iteration cannot resolve). Converged when the estimate changes by ``<= tol`` (relative) over three successive iterations.
    Returns ``(estimate, converged, ritz_values)``.
    """
    x = np.asarray(x0, dtype=np.float64).reshape(-1)
    x = x / np.linalg.norm(x)
    prev_x, prev_s = None, None
    est, last, streak, ritz = float("nan"), None, 0, np.zeros(0, dtype=np.complex128)
    for _ in range(iters):
        y = mv(x)
        s = float(np.linalg.norm(y))
        if not np.isfinite(s):
            return float("nan"), False, ritz
        if s == 0.0:
            return 0.0, True, np.zeros(1, dtype=np.complex128)
        x_next = y / s
        if prev_x is not None:
            # s_{k-1} s_k x_{k+1} = a s_{k-1} x_k - b x_{k-1}
            M = np.stack([prev_s * x, -prev_x], axis=1)
            r = prev_s * s * x_next
            G = M.T @ M
            sv = np.linalg.svd(M, compute_uv=False)
            if sv[1] > 1e-7 * sv[0]:
                a, b = np.linalg.solve(G, M.T @ r)
                disc = np.sqrt(complex(a * a - 4.0 * b))
                ritz = np.array([(a + disc) / 2.0, (a - disc) / 2.0])
                est = float(np.abs(ritz).max())
            else:                                                                    # collinear: a dominant real eigenvalue
                est, ritz = s, np.array([s], dtype=np.complex128)
            if last is not None and abs(est - last) <= tol * max(abs(est), 1e-300):
                streak += 1
                if streak >= 3:
                    return est, True, ritz
            else:
                streak = 0
            last = est
        prev_x, prev_s, x = x, s, x_next
    return est, False, ritz


def spectral_radius(matvec, x0, *, method: str = "arpack", k: int = 6, tol: float = 1e-6, maxiter: int = 4000,
                    power_iters: int = 2000, ncv: int = 40) -> dict:
    """Largest ``|lambda|`` of a real linear operator given as a flat-vector ``matvec`` (a host function of a ``(n,)`` array,
    for example a wrapped jitted JVP); ``x0 (n,)`` is the (nonzero) start vector.

    ``method="arpack"``: ``scipy.sparse.linalg.eigs(which="LM")`` for the ``k`` eigenvalues of largest modulus (complex pairs
    included), ``ncv`` Lanczos vectors, relative ``tol``. If ARPACK does not converge or fails, falls back to the power
    iteration (:func:`_power_ritz`, at most ``power_iters`` matvecs, estimate tolerance ``tol``); ``method="power"`` uses the
    power iteration directly. Operators with ``n <= k + 2`` (too small for ARPACK) are assembled densely (``method="dense"``).
    Returns ``{"lambda_max", "eigenvalues" (top ``k`` as ``[re, im]``, descending modulus; the power iteration reports its
    Ritz values), "method", "matvecs", "converged"}`` (+ ``"arpack_error"`` when the fallback was used).
    """
    if method not in ("arpack", "power"):
        raise ValueError(f"method must be 'arpack' or 'power', got {method!r}")
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    n = x0.size
    if not np.any(x0):
        raise ValueError("x0 must be nonzero")
    count = [0]

    def mv(x):
        count[0] += 1
        return np.asarray(matvec(np.asarray(x, dtype=np.float64).reshape(-1)), dtype=np.float64).reshape(-1)

    def result(ev, used, converged, **extra):
        ev = _sorted_eigs(ev)
        lam = float(np.abs(ev).max()) if ev.size else float("nan")
        return {"lambda_max": lam, "eigenvalues": _eig_pairs(ev, k), "method": used, "matvecs": count[0],
                "converged": bool(converged and np.isfinite(lam)), **extra}

    if n <= k + 2:
        J = np.stack([mv(e) for e in np.eye(n)], axis=1)
        return result(np.linalg.eigvals(J), "dense", True)
    error = None
    if method == "arpack":
        op = spla.LinearOperator((n, n), matvec=mv, dtype=np.float64)
        try:
            ev = spla.eigs(op, k=k, which="LM", ncv=min(max(ncv, 2 * k + 1), n - 1), tol=tol, maxiter=maxiter, v0=x0,
                           return_eigenvectors=False)
            return result(ev, "arpack", True)
        except spla.ArpackNoConvergence as exc:
            error = f"ArpackNoConvergence ({len(exc.eigenvalues)} of {k} eigenvalues)"
        except Exception as exc:                                                    # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
    est, conv, ritz = _power_ritz(mv, x0, power_iters, tol)
    return result(ritz if ritz.size else np.array([est]), "power", conv, **({"arpack_error": error} if error else {}))


def rightmost_eigenvalue(matvec, x0, *, k: int = 6, ncv: int = 60, tol: float = 1e-6, maxiter: int = 20000) -> dict:
    """The eigenvalue of largest real part of a real linear operator given as a flat-vector ``matvec`` (``x0 (n,)`` the nonzero
    start vector): ``scipy.sparse.linalg.eigs(which="LR")`` for the ``k`` eigenvalues of largest real part, ``ncv`` Lanczos
    vectors, relative ``tol`` (operators with ``n <= k + 2`` are assembled densely). A positive real part is a growth rate.

    Returns ``{"rightmost": [re, im], "eigenvalues" (the ``k`` found, ``[re, im]``, descending real part), "matvecs",
    "converged"}``. On an ARPACK failure ``converged`` is false, ``"error"`` records it, and ``eigenvalues`` / ``rightmost``
    hold whatever ARPACK had converged (``rightmost`` is ``None`` if nothing did); there is no fallback.
    """
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    n = x0.size
    if not np.any(x0):
        raise ValueError("x0 must be nonzero")
    count = [0]

    def mv(x):
        count[0] += 1
        return np.asarray(matvec(np.asarray(x, dtype=np.float64).reshape(-1)), dtype=np.float64).reshape(-1)

    def result(ev, converged, **extra):
        ev = np.asarray(ev, dtype=np.complex128).reshape(-1)
        ev = ev[np.lexsort((-ev.imag, -ev.real))]                                  # descending real part
        pairs = [[float(e.real), float(e.imag)] for e in ev[:k]]
        return {"rightmost": pairs[0] if pairs else None, "eigenvalues": pairs, "matvecs": count[0],
                "converged": bool(converged and pairs and np.isfinite(pairs[0][0])), **extra}

    if n <= k + 2:
        J = np.stack([mv(e) for e in np.eye(n)], axis=1)
        return result(np.linalg.eigvals(J), True)
    op = spla.LinearOperator((n, n), matvec=mv, dtype=np.float64)
    try:
        ev = spla.eigs(op, k=k, which="LR", ncv=min(max(ncv, 2 * k + 1), n - 1), tol=tol, maxiter=maxiter, v0=x0,
                       return_eigenvectors=False)
        return result(ev, True)
    except spla.ArpackNoConvergence as exc:
        return result(exc.eigenvalues, False, error=f"ArpackNoConvergence ({len(exc.eigenvalues)} of {k} eigenvalues)")
    except Exception as exc:                                                        # noqa: BLE001
        return result([], False, error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------------------------------------------------
# time step
# ---------------------------------------------------------------------------------------------------------------------
def dt_rule(lambda_max: float, T: float) -> tuple[float, int]:
    """``dt = T / ceil(T / (0.5 * 2.6 / lambda_max))``: ``(dt, nsteps)`` with ``nsteps`` integer, ``dt * nsteps = T``
    (``2.6`` is the RK4 stability radius on the imaginary and negative real axes; ``0.5`` the safety factor)."""
    lambda_max, T = float(lambda_max), float(T)
    if not (np.isfinite(lambda_max) and lambda_max > 0.0):
        raise ValueError(f"lambda_max must be positive and finite, got {lambda_max}")
    if not T > 0.0:
        raise ValueError(f"T must be positive, got {T}")
    dt_max = RK4_SAFETY * RK4_STABILITY_RADIUS / lambda_max
    nsteps = max(int(math.ceil(T / dt_max)), 1)
    return T / nsteps, nsteps


# ---------------------------------------------------------------------------------------------------------------------
# the stage linearization
# ---------------------------------------------------------------------------------------------------------------------
def exact_state(bundle, p: MmsParams, t) -> jax.Array:
    """``q_bar(t) (E, P, 4)`` at the nodes (``n, Te, Ti, Omega``)."""
    return jnp.asarray(FIELDS.values(bundle.ref.points, t, p)[..., :4])


def stage_linearization(bundle, mode: str, source: str, p: MmsParams, pattern: str, t, *, opts_override=None):
    """Jitted flat matvec ``v -> J v`` of the stage RHS at ``q_bar(t)``.

    The stage RHS is ``model.make_stage_rhs(bundle, mode, source, p, pattern)``; the state is ``q_bar(t)`` and the carry is
    zero (a cold-start potential solve with ``phi_rtol = 1e-13`` in the coupled mode, so the forward-mode tangent through
    the CG iterations is accurate). ``J`` is the forward-mode JVP with respect to the state (``t`` and the carry fixed; the
    source term is state independent, so ``J`` is the Jacobian of the nodal RHS, including the ``psi`` solve in the coupled mode). ``v`` and the
    result are flat ``(E * P * 4,)`` arrays (C order of ``(E, P, 4)``). The returned function carries the attributes
    ``size``, ``shape``, ``x0`` (the flat ``q_bar(t)``) and ``rhs`` (the flat state -> flat RHS map).
    """
    from p10_evolved_mms import model                                               # lazily: written in parallel (chunk C3)

    E, P = bundle.E, bundle.P
    # Forward-mode AD through the CG while_loop differentiates the iterations actually taken: a warm start at (or near) the
    # solution gives a truncated tangent. Linearize with a cold start (zero carry) and a tight solve instead.
    override = {"phi_rtol": 1e-13, "phi_maxit": 600} if mode == "coupled" else {}
    override.update(opts_override or {})
    kw = {"opts_override": override} if override else {}
    rhs_fn = model.make_stage_rhs(bundle, mode, source, p, pattern, **kw)
    tt = jnp.asarray(t, dtype=jnp.float64)
    carry = jnp.zeros_like(model.initial_carry(bundle, mode, p, t))
    x0 = exact_state(bundle, p, t).reshape(-1)

    def flat_rhs(x):
        out, _carry, _info = rhs_fn(model.NodalState.from_array(x.reshape(E, P, 4)), tt, carry)
        return out.array().reshape(-1)

    @jax.jit
    def matvec(v):
        return jax.jvp(flat_rhs, (x0,), (jnp.asarray(v, dtype=x0.dtype),))[1]

    matvec.size, matvec.shape, matvec.x0, matvec.rhs = x0.size, (E, P, 4), x0, flat_rhs
    return matvec


# ---------------------------------------------------------------------------------------------------------------------
# term shares
# ---------------------------------------------------------------------------------------------------------------------
def region_masks(bundle) -> dict:
    """``ring_region_masks(layout, E)`` (``(E, P)`` boolean; includes ``"all"``)."""
    return ring_region_masks(bundle.layout, bundle.E)


def norm_weights(bundle) -> np.ndarray:
    """The H weights ``(E, P)`` (``h_weights(plan)``)."""
    H = np.asarray(h_weights(bundle.ctx.plan), dtype=np.float64)
    return np.broadcast_to(H, (bundle.E, bundle.P)).copy()


def _h_norm(H, a, mask):
    """``sqrt(sum_mask H a^2)`` for ``a (E, P)`` (or ``(E, P, F)`` per field)."""
    w = H[mask]
    a = np.asarray(a, dtype=np.float64)[mask]
    return np.sqrt(np.sum(w.reshape((-1,) + (1,) * (a.ndim - 1)) * a ** 2, axis=0))


def term_shares(bundle, p: MmsParams, mode: str, t) -> dict:
    """``{field: {term: {region: ||term||_H,region / ||d_t q_bar||_H,region}}}`` at time ``t``.

    Fields ``("n", "Te", "Ti", "Omega")``; terms the mode's term set (``MODE_TERMS``) from the continuum RHS
    (:func:`~p10_evolved_mms.source.continuum`: ``bracket``, ``curvature``, ``diffusion`` at the exact fields); regions the
    ``ring_region_masks`` names (``"all"`` is every node). ``H`` is ``h_weights(plan)`` ``(E, P)``. ``None`` where the region's
    ``||d_t q_bar||`` vanishes.
    """
    if mode not in MODE_TERMS:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    cont = src.continuum_jit(bundle, jnp.asarray(t, dtype=jnp.float64), p)
    H = norm_weights(bundle)
    masks = region_masks(bundle)
    dq = {name: _h_norm(H, np.asarray(cont.dq), m) for name, m in masks.items()}
    out = {}
    for f, label in enumerate(FIELD_LABELS):
        out[label] = {}
        for term in MODE_TERMS[mode]:
            arr = np.asarray(getattr(cont, term))
            out[label][term] = {}
            for name, m in masks.items():
                den = float(dq[name][f])
                out[label][term][name] = float(_h_norm(H, arr, m)[f]) / den if den > 0.0 else None
    return out


# ---------------------------------------------------------------------------------------------------------------------
# bracket energy abscissa
# ---------------------------------------------------------------------------------------------------------------------
def potential(bundle, p: MmsParams, t) -> jax.Array:
    """``phi_bar(t) (E, P)`` (including ``a_phi``)."""
    return jnp.asarray(FIELDS.values(bundle.ref.points, t, p)[..., 4])


def max_half_c(bundle, p: MmsParams, t, T=None) -> dict:
    """``max (c / 2)`` of the E x B velocity of ``phi_bar(t)`` for the scaled bracket, and ``max (c / 2) * T``.

    The harness bracket is ``rho_star * sbp_bracket(plan, phi, f, bcd, 1.0, c_kappa)``
    (``fci_nodal_perpendicular_rhs._bracket_term``): the velocity flux of the unit divisor ``F_1 = velocity_flux(plan, phi, 1.0)``
    scaled by ``rho_star``. The operator is positively homogeneous of degree 1 in the flux (transport, the wall product
    correction and the inflow / upwind penalties are linear in ``F``, the SAT strengths are ``max(-sigma v, 0)``), so the
    compressibility ``c = -2 T(1)`` of the scaled bracket is ``rho_star * compressibility(plan, F_1)`` (checked against the
    flux of divisor ``1 / rho_star`` in the tests). Returns ``{"max_half_c", "max_half_c_T"}`` (``None`` if ``T`` is ``None``)
    and the minimum ``"min_half_c"``.
    """
    plan = bundle.ctx.plan
    F1 = velocity_flux(plan, potential(bundle, p, t), 1.0)
    c1 = np.asarray(compressibility(plan, F1))
    rho = float(p.rho_star)
    hi, lo = rho * 0.5 * float(c1.max()), rho * 0.5 * float(c1.min())
    return {"max_half_c": hi, "max_half_c_T": None if T is None else hi * float(T), "min_half_c": lo}


# ---------------------------------------------------------------------------------------------------------------------
# potential solve cost
# ---------------------------------------------------------------------------------------------------------------------
def cg_iterations(bundle, p: MmsParams, pattern: str, t, dt) -> dict:
    """CG iterations of the polarization solve ``solve_potential`` at ``q_bar(t)`` with ``sigma`` of the continuum source:
    cold (``x0 = None``) and warm (``x0 = psi_bar(t - dt)``). Returns ``{"cold", "warm"}`` plus the final relative residuals
    and ``converged`` flags (``"cold_relative_residual"``, ``"warm_relative_residual"``, ``"cold_converged"``,
    ``"warm_converged"``)."""
    opts = src.nodal_options(pattern, phi_mode="solve")
    tt = jnp.asarray(t, dtype=jnp.float64)
    cont = src.continuum_jit(bundle, tt, p)
    wall = src.wall_data_jit(bundle, tt, p, pattern)
    prev = src.continuum_jit(bundle, jnp.asarray(float(t) - float(dt), dtype=jnp.float64), p)
    q = cont.q
    args = (bundle.ctx, opts, p.nodal(), q[..., 3], q[..., 0], q[..., 2], wall.psi)
    out = {}
    for label, x0 in (("cold", None), ("warm", prev.psi)):
        _psi, _phi, info = solve_potential_jit(*args, sigma=cont.sigma, x0=x0)
        out[label] = int(info["iterations"])
        out[f"{label}_relative_residual"] = float(info["relative_residual"])
        out[f"{label}_converged"] = bool(info["converged"])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------------------------------------------------
_IDENTITY_KEYS = ("schema", "arm", "n", "n_eta", "P", "bundle_sha256", "git_commit", "arm_identity", "layout_sha256",
                  "nodal_plan_sha256", "laplacian_plan_sha256", "nodal_metric_identity", "laplacian_metric_identity",
                  "sidecar_sha256", "synthetic")


def _tkey(t) -> str:
    return repr(float(t))


def _params_json(p: MmsParams) -> dict:
    return {"rho_star": float(p.rho_star), "tau": float(p.tau), "D": [float(x) for x in np.asarray(p.D).reshape(-1)],
            "a_phi": float(p.a_phi), "time_scale": float(p.time_scale), "w1": float(p.w1), "a_omega": float(p.a_omega)}


def preflight(bundle, mode: str, pattern: str, p: MmsParams, T, *, times=None, source: str = "continuum", out=None,
              k: int = 6, tol: float = 1e-6, method: str = "arpack", seed: int = 0, opts_override=None, log=None,
              rightmost: bool | None = None) -> dict:
    """The preflight of ``(bundle, mode, pattern)`` as the ``preflight.json`` dict; written to ``out`` (a directory, or a
    ``.json`` path) if given.

    ``lambda_max`` is the list over ``times`` (default ``(0, T/2, T)``) of the spectral radius of the stage Jacobian at
    ``q_bar(t)``; ``dt``, ``nsteps`` follow :func:`dt_rule` of their maximum. ``term_shares`` at ``t = 0`` and ``T`` (keys
    ``repr(float(t))``); ``max_half_c`` the maximum over ``times`` of :func:`max_half_c` (``max_half_c_T`` times ``T``); for the
    coupled mode ``cg_iterations`` ``{"cold", "warm"}`` is the maximum over ``times`` of :func:`cg_iterations` with
    ``dt`` (details in ``cg_detail``). ``rightmost`` is the list over ``times`` of the eigenvalue of largest real part
    (:func:`rightmost_eigenvalue`, ``[re, im]``) of the same Jacobian, ``rightmost_eigenvalues`` / ``rightmost_matvecs`` /
    ``rightmost_converged`` its provenance, ``max_rightmost_real`` the maximum real part over ``times`` and
    ``max_rightmost_real_T`` that times ``T`` (the e-folding exponent of the linearized growth over the run). Extra provenance:
    ``identity``, ``mode``, ``pattern``, ``source``, ``params``, ``T``, ``times``, ``eigenvalues``, ``matvecs``, ``converged``,
    ``seconds``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    log = (lambda *a: None) if log is None else log
    T = float(T)
    times = (0.0, T / 2.0, T) if times is None else tuple(float(t) for t in times)
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    # The rightmost search is skipped by default for "diffusion": its near-null Neumann modes cluster at 0, so ARPACK 'LR'
    # needs ~1e4 matvecs, and the parabolic operator has no growth to report beyond the accepted Neumann semidefiniteness.
    want_right = (mode != "diffusion") if rightmost is None else bool(rightmost)
    rng_right = np.random.default_rng(seed + 1)                                        # a separate stream: lambda_max starts unchanged
    lam, eigs, methods, matvecs, conv, secs = [], [], [], [], [], []
    right, right_ev, right_mv, right_conv = [], [], [], []
    for t in times:
        t0 = time.perf_counter()
        mv = stage_linearization(bundle, mode, source, p, pattern, t, opts_override=opts_override)
        res = spectral_radius(lambda v: np.asarray(mv(v)), rng.standard_normal(mv.size), method=method, k=k, tol=tol)
        lam.append(res["lambda_max"]), eigs.append(res["eigenvalues"]), methods.append(res["method"])
        matvecs.append(res["matvecs"]), conv.append(res["converged"]), secs.append(time.perf_counter() - t0)
        if want_right:
            rr = rightmost_eigenvalue(lambda v: np.asarray(mv(v)), rng_right.standard_normal(mv.size), k=k, tol=tol)
        else:
            rr = {"rightmost": None, "eigenvalues": [], "matvecs": 0, "converged": None}
        right.append(rr["rightmost"]), right_ev.append(rr["eigenvalues"]), right_mv.append(rr["matvecs"])
        right_conv.append(rr["converged"])
        secs[-1] = time.perf_counter() - t0
        log(f"t = {t:g}: lambda_max = {res['lambda_max']:.6g} ({res['method']}, {res['matvecs']} matvecs, "
            f"converged {res['converged']}); rightmost = {rr['rightmost']} ({rr['matvecs']} matvecs, converged "
            f"{rr['converged']}{', ' + rr['error'] if 'error' in rr else ''}); {secs[-1]:.1f} s")
    lam_used = max(lam)
    dt, nsteps = dt_rule(lam_used, T)
    shares = {_tkey(t): term_shares(bundle, p, mode, t) for t in (0.0, T)}
    halves = [max_half_c(bundle, p, t, T) for t in times]
    real_parts = [r[0] for r in right if r is not None]
    max_right = max(real_parts) if real_parts else None
    result = {
        "schema": SCHEMA,
        "identity": {k_: bundle.identity[k_] for k_ in _IDENTITY_KEYS if k_ in bundle.identity},
        "mode": mode, "pattern": pattern, "source": source, "params": _params_json(p), "T": T, "times": list(times),
        "lambda_max": lam, "lambda_max_used": lam_used, "eigenvalues": eigs,
        "method": methods[0] if len(set(methods)) == 1 else "+".join(sorted(set(methods))),
        "matvecs": matvecs, "converged": bool(all(conv)),
        "rightmost": right, "rightmost_eigenvalues": right_ev, "rightmost_matvecs": right_mv,
        "rightmost_converged": bool(all(right_conv)) if want_right else None, "rightmost_skipped": not want_right,
        "max_rightmost_real": max_right, "max_rightmost_real_T": None if max_right is None else max_right * T,
        "dt_rule": DT_RULE, "dt": dt, "nsteps": nsteps,
        "term_shares": shares,
        "max_half_c": max(h["max_half_c"] for h in halves), "max_half_c_T": max(h["max_half_c_T"] for h in halves),
        "max_half_c_per_time": [h["max_half_c"] for h in halves], "min_half_c_per_time": [h["min_half_c"] for h in halves],
        "bracket_in_mode": "bracket" in MODE_TERMS[mode],
    }
    if mode == "coupled":
        cg = [cg_iterations(bundle, p, pattern, t, dt) for t in times]
        result["cg_iterations"] = {"cold": max(c["cold"] for c in cg), "warm": max(c["warm"] for c in cg)}
        result["cg_detail"] = [{"t": t, **c} for t, c in zip(times, cg)]
        log(f"CG iterations cold {result['cg_iterations']['cold']}, warm {result['cg_iterations']['warm']}")
    result["seconds"] = time.perf_counter() - t_start
    if out is not None:
        out = Path(out)
        path = out if out.suffix == ".json" else out / "preflight.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=1, sort_keys=False))
        tmp.replace(path)
    return result


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--bundle", type=Path, help="a saved bundle directory (bundle.save_bundle)")
    grp.add_argument("--arm", choices=tuple(CONFIG["arms"]), help="build the bundle of this arm (needs --n)")
    ap.add_argument("--n", type=int)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--pattern", choices=tuple(CONFIG["patterns"]), required=True)
    ap.add_argument("--source", choices=("continuum", "discrete"), default="continuum")
    ap.add_argument("--T", type=float, default=float(CONFIG["T"]))
    ap.add_argument("--rho-star", type=float)
    ap.add_argument("--a-phi", type=float)
    ap.add_argument("--a-omega", type=float)
    ap.add_argument("--time-scale", type=float)
    ap.add_argument("--w1", type=float)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--method", choices=("arpack", "power"), default="arpack")
    ap.add_argument("--out", type=Path, required=True, help="output directory (writes preflight.json) or a .json path")
    args = ap.parse_args(argv)
    from p10_evolved_mms import bundle as bundle_mod

    if args.bundle is not None:
        bundle = bundle_mod.load_bundle(args.bundle)
    else:
        if args.n is None:
            ap.error("--arm needs --n")
        bundle = bundle_mod.build_bundle(args.arm, args.n)
    over = {k_: v for k_, v in (("rho_star", args.rho_star), ("a_phi", args.a_phi), ("a_omega", args.a_omega),
                                ("time_scale", args.time_scale), ("w1", args.w1)) if v is not None}
    res = preflight(bundle, args.mode, args.pattern, default_params(**over), args.T, source=args.source, out=args.out,
                    k=args.k, tol=args.tol, method=args.method, log=lambda s: print(s, flush=True))
    print(json.dumps({k_: res[k_] for k_ in ("lambda_max", "method", "dt", "nsteps", "max_half_c", "max_half_c_T",
                                              "max_rightmost_real", "max_rightmost_real_T", "cg_iterations") if k_ in res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
