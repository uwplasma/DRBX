"""P10 evolved MMS, chunk C3 (stage RHS glue): constant states, the discrete-source pairing at the RK stage times, the continuum /
discrete truncation identity, the state JVP, the compile strategy and :class:`StageInfo`, on the n = 16 family-A synthetic bundle."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from p10_evolved_mms import fields as F                           # noqa: E402
from p10_evolved_mms import model as M                            # noqa: E402
from p10_evolved_mms import source as S                           # noqa: E402
from p10_evolved_mms import synthetic as syn                      # noqa: E402

# the provisional configuration (rho* = 4.5e-4, time_scale = 1, T = 1/30) and a "fast" set with O(1) advection, curvature and
# polarization (rho* = 0.7); the diffusion coefficients stay at the configured 1e-5 for "config" and O(1e-2) for "fast"
P_CONFIG = F.default_params()
# a_omega * rho_star^2 = 1: the O(1) vorticity of the fast sets (the config set has the consistent rho_star^2 amplitude)
P_FAST = F.default_params(rho_star=0.7, time_scale=1.0, a_phi=1.0, a_omega=1.0 / 0.7 ** 2, D=np.array([0.01, 0.012, 0.014, 0.008]))
P_SLOW = F.default_params(rho_star=0.7, time_scale=0.01, a_phi=1.0, a_omega=1.0 / 0.7 ** 2, D=np.array([0.01, 0.012, 0.014, 0.008]))
CASES = {"config": (P_CONFIG, 0.01, 0.01), "fast": (P_FAST, 0.2, 0.02)}      # config: stage times inside T = 1/30
MODES = ("diffusion", "hyperbolic", "coupled")
PATTERNS = ("NNN-D", "DDDD")
CG_RTOL = float(S.CONFIG["solve"]["rtol"])


@pytest.fixture(scope="module")
def bundle():
    return syn.synthetic_bundle()


@lru_cache(maxsize=None)
def jit_stage(mode, source, pattern, **override):
    """One jitted stage per configuration, ``f(bundle, p, state, t, carry)`` (bundle and parameters are jit arguments, so the
    config / fast parameter sets share the compilation)."""
    cfg = M.stage_config(mode, source, pattern, opts_override=override)
    return jax.jit(M.stage_rhs_unbound(cfg)), cfg


def stage(bundle, mode, source, pattern, p, q, t, carry, **override):
    fn, _ = jit_stage(mode, source, pattern, **override)
    return fn(bundle, M.as_jax_params(p), M.NodalState.from_array(q), t, carry)


def dq_exact(bundle, p, t):
    return F.FIELDS.values_dt(bundle.ref.points, t, p)[1][..., :4]


def max_rel(a, ref):
    return float(jnp.abs(a - ref).max() / jnp.abs(ref).max())


# ----------------------------------------------------------------------------------------------------- the state
def test_nodal_state_roundtrip_and_pytree():
    a = jnp.asarray(np.random.default_rng(0).normal(size=(3, 5, 4)))
    s = M.NodalState.from_array(a)
    assert s.field_names() == ("n", "Te", "Ti", "omega")
    assert all(f.shape == (3, 5) for f in s.field_values())
    np.testing.assert_array_equal(np.asarray(s.array()), np.asarray(a))
    np.testing.assert_array_equal(np.asarray(s.n), np.asarray(a[..., 0]))
    np.testing.assert_array_equal(np.asarray(s.omega), np.asarray(a[..., 3]))
    leaves, tree = jax.tree_util.tree_flatten(s)
    assert len(leaves) == 4 and isinstance(jax.tree_util.tree_unflatten(tree, leaves), M.NodalState)
    both = s.axpy(s, scale=2.0)                                             # FciModelState algebra
    np.testing.assert_allclose(np.asarray(both.array()), 3.0 * np.asarray(a), rtol=1e-15)
    with pytest.raises(ValueError):
        M.NodalState.from_array(jnp.zeros((3, 5, 6)))


def test_stage_config_validation_and_mode_terms():
    for mode, terms in (("diffusion", ("diffusion",)), ("hyperbolic", ("bracket", "curvature")),
                        ("coupled", ("bracket", "curvature", "diffusion"))):
        cfg = M.stage_config(mode, "discrete", "NNN-D")
        assert cfg.opts.terms == terms and cfg.opts.phi_mode == ("solve" if mode == "coupled" else "prescribed")
        assert cfg.opts.psi == "phi_plus_tau_pi"
    with pytest.raises(ValueError):
        M.stage_config("nope", "discrete", "NNN-D")
    with pytest.raises(ValueError):
        M.stage_config("diffusion", "other", "NNN-D")
    with pytest.raises(ValueError):
        M.stage_config("diffusion", "discrete", "NNN-D", opts_override={"terms": ("bracket",)})
    with pytest.raises(TypeError, match="rho_star_convention"):                # the selector no longer exists
        M.stage_config("diffusion", "discrete", "NNN-D", opts_override={"rho_star_convention": "single-length"})


# ----------------------------------------------------------------------------------------------------- 1. constants
CONST = np.array([1.3, 0.9, 1.1, 0.4, 0.25])                                # (n, Te, Ti, Omega, phi)
CONST_FIELDS = F.MmsFields(post=lambda p, t, v, mp: 0.0 * v + jnp.asarray(CONST))


@pytest.mark.parametrize("source", ["continuum", "discrete"])
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("mode", MODES)
def test_constant_state_with_constant_wall_data_has_zero_rhs(bundle, mode, pattern, source):
    """All fields constant in space and time, matching constant wall data: ``R_h = 0`` (every term annihilates a constant, the
    SAT penalties vanish), the source is zero, and the coupled polarization solve returns the constant ``psi`` (``sigma = Omega``)."""
    p = P_FAST
    rhs = jax.jit(M.make_stage_rhs(bundle, mode, source, p, pattern, fields=CONST_FIELDS))
    q = M.exact_state(bundle, p, 0.3, CONST_FIELDS)
    np.testing.assert_array_equal(np.asarray(q), np.broadcast_to(CONST[:4], q.shape))
    carry = M.initial_carry(bundle, mode, p, 0.3, CONST_FIELDS)
    for t in (0.3, 0.37):
        out, new_carry, info = rhs(M.NodalState.from_array(q), t, carry)
        # the stencil sums of a constant cancel to round-off (measured <= 1e-13 with D = 1e-2, vs O(1) terms of the exact fields)
        assert float(jnp.abs(out.array()).max()) < 1e-12, float(jnp.abs(out.array()).max())
        assert bool(info.cg_converged) and bool(info.finite)
        np.testing.assert_allclose(np.asarray(new_carry), np.asarray(carry), rtol=0, atol=1e-13)


# ----------------------------------------------------------------------------------------------------- 2. pairing
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ["config", "fast"])
def test_pairing_discrete_source_reproduces_dt_q_at_the_rk_stage_times(bundle, case, mode, pattern):
    """At the exact state ``q(t)``, ``R_h(q; wall(t)) + S_h(t) = d_t q`` to round-off at the four RK stage times for
    ``"diffusion"`` and ``"hyperbolic"`` (prescribed ``phi``); the coupled mode adds the ``psi`` solve with ``sigma_h`` and the
    exact ``psi`` as warm start (the solver must find it already converged; the solve is then exact up to round-off)."""
    p, t0, dt = CASES[case]
    for t in (t0, t0 + dt / 2, t0 + dt / 2, t0 + dt):
        q = M.exact_state(bundle, p, t)
        dq = dq_exact(bundle, p, t)
        carry = M.initial_carry(bundle, mode, p, t)
        out, new_carry, info = stage(bundle, mode, "discrete", pattern, p, q, t, carry)
        assert max_rel(out.array(), dq) <= 1e-13, (mode, pattern, case, t, max_rel(out.array(), dq))
        assert bool(info.cg_converged)
        if mode == "coupled":
            psi = np.asarray(carry)
            # the warm start is the exact psi, so the returned psi is the exact psi up to the (round-off sized) CG correction
            assert np.abs(np.asarray(new_carry) - psi).max() <= 1e-9 * np.abs(psi).max()
        else:
            np.testing.assert_array_equal(np.asarray(new_carry), np.asarray(carry))


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("case", ["config", "fast"])
def test_pairing_coupled_cold_start_is_limited_by_the_cg_tolerance(bundle, case, pattern):
    """Coupled mode from a cold (zero) warm start: ``psi`` is only known to the CG tolerance. The solve conditions
    ``Omega - sigma_h = rho*^2 L_h psi`` against ``Omega`` (round-off ``~ 1e-16 Omega / (rho*^2 L psi)``, amplified further by
    ``L^-1``), and CG stops at ``|r|_{H^-1} / |b| <= rtol = 1e-11``; the previous agent saw ``psi`` round-off ~3e-11 at
    ``rho* = 4.5e-4``. The error reaches the RHS only through ``phi`` in the bracket and curvature, i.e. times ``rho*`` and a
    gradient, and was measured at ~2.4e-9 of ``max|d_t q|`` for both parameter sets (about 240 rtol; ``rho*`` enters
    ``1/rho*^2`` in the data and ``rho*`` in the terms). The bound used, 2.5e-8, is 10x the measurement."""
    p, t0, dt = CASES[case]
    t = t0 + dt / 2
    q = M.exact_state(bundle, p, t)
    dq = dq_exact(bundle, p, t)
    zeros = jnp.zeros(q.shape[:2])
    out, new_carry, info = stage(bundle, "coupled", "discrete", pattern, p, q, t, zeros)
    assert bool(info.cg_converged) and int(info.cg_iterations) > 0
    assert float(info.cg_relative_residual) <= CG_RTOL
    err = max_rel(out.array(), dq)
    assert err <= 2.5e-8, err
    assert err > 1e-13                                   # the cold start really is limited by the tolerance (not an exact solve)
    psi = M.initial_carry(bundle, "coupled", p, t)
    assert max_rel(new_carry, psi) <= 1e-8


# ----------------------------------------------------------------------------------------------------- 3. continuum source
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ["config", "fast"])
def test_continuum_source_stage_minus_dt_q_is_the_truncation_of_the_mode_terms(bundle, case, mode):
    """With the continuum source, ``rhs(q) - d_t q = R_h(q; phi_h) - R_cont(q; phi)`` summed over the mode's terms only
    (identity to round-off). Prescribed modes: ``phi_h`` is the exact ``phi``. Coupled: ``phi_h`` is the potential of the solve
    with the *continuum* ``sigma`` (it differs from the exact ``phi`` by the Laplacian truncation, which the reference below
    reproduces independently with :func:`solve_potential`, same warm start)."""
    import dataclasses

    from drbx.native import fci_nodal_perpendicular_rhs as rhs_mod

    p, t0, dt = CASES[case]
    t = t0 + dt
    pattern = "NNN-D"
    q = M.exact_state(bundle, p, t)
    dq = dq_exact(bundle, p, t)
    carry = M.initial_carry(bundle, mode, p, t)
    cfg = M.stage_config(mode, "discrete", pattern)
    d = S.discrete_jit(bundle, None, cfg.opts, None, t, p)
    c = S.continuum_jit(bundle, t, p)
    Rc = sum(getattr(c, term) for term in cfg.terms)
    Rh = d.R
    if mode == "coupled":
        params, wall = p.nodal(), S.wall_data_jit(bundle, t, p, pattern)
        _, phi_h, _ = rhs_mod.solve_potential(bundle.ctx, cfg.opts, params, q[..., 3], q[..., 0], q[..., 2], wall.psi,
                                              sigma=c.sigma, x0=carry)
        Rh = rhs_mod.nodal_perpendicular_rhs(bundle.ctx, dataclasses.replace(cfg.opts, phi_mode="prescribed"),
                                             params, q, wall, phi=phi_h).total
        assert float(jnp.abs(phi_h - d.phi).max()) > 1e-6 * float(jnp.abs(d.phi).max())     # phi_h != exact phi
    tau = Rh - Rc
    out, _, _ = stage(bundle, mode, "continuum", pattern, p, q, t, carry)
    scale = max(float(jnp.abs(x).max()) for x in (dq, Rh, Rc))
    err = float(jnp.abs(out.array() - dq - tau).max()) / scale
    assert err <= 1e-13, err
    # the truncation is visible relative to the mode's own terms (at physical rho_star and T = 1/30 they are small next to
    # d_t q, so ``scale`` is the wrong yardstick here)
    assert float(jnp.abs(tau).max()) > 1e-6 * float(jnp.abs(Rc).max())
    # the selected terms only: the unselected continuum terms do not enter
    full = sum(getattr(c, term) for term in M.MODES["coupled"])
    if mode != "coupled" and case == "fast":
        assert float(jnp.abs(Rc - full).max()) > 1e-3 * scale
    # the snapshot diagnostic tau = R_h(exact phi) - R_cont of the same terms
    diag = M.snapshot_diagnostics_jit(cfg)(bundle, M.as_jax_params(p), t, q)
    np.testing.assert_allclose(np.asarray(diag["tau"]), np.asarray(d.R - Rc), rtol=1e-12, atol=1e-13 * scale)
    np.testing.assert_allclose(np.asarray(diag["q_exact"]), np.asarray(q), rtol=1e-14, atol=1e-14)

# ----------------------------------------------------------------------------------------------------- 4. JVP

@pytest.mark.parametrize("mode", ["hyperbolic", "coupled"])
def test_jvp_of_the_stage_rhs_matches_central_finite_differences(bundle, mode):
    """``d rhs / d state`` along a random direction vs a central difference. ``eps = 1e-6``: the finite-difference truncation
    error is ``~ eps^2`` times the third derivative (quadratic curvature term: 6e-11 measured at 1e-6, 6e-7 at 1e-4) and the
    round-off ``~ 1e-16 |f| / eps`` stays below 1e-9. The coupled stage uses a *cold* carry and ``phi_rtol = 1e-13``: forward-mode
    AD differentiates the CG iterations, so with a warm start already at the solution (zero iterations) the tangent of ``psi`` is
    zero (measured: 4e-3 relative error); from a cold start the tangent of the converged Krylov solution is the derivative of the
    solve (3e-10)."""
    p, t = P_FAST, 0.22
    fn, cfg = jit_stage(mode, "discrete", "NNN-D", phi_rtol=1e-13, phi_maxit=100)
    jp = M.as_jax_params(p)
    q = M.exact_state(bundle, p, t)
    carry = jnp.zeros(q.shape[:2]) if mode == "coupled" else M.initial_carry(bundle, mode, p, t)
    v = jnp.asarray(np.random.default_rng(3).normal(size=q.shape)) * jnp.abs(q).max(axis=(0, 1))
    f = lambda x: fn(bundle, jp, M.NodalState.from_array(x), t, carry)[0].array()
    _, jv = jax.jvp(f, (q,), (v,))
    eps = 1e-6
    fd = (f(q + eps * v) - f(q - eps * v)) / (2 * eps)
    assert float(jnp.abs(jv).max()) > 1e-3
    assert float(jnp.abs(jv - fd).max() / jnp.abs(jv).max()) <= 1e-6

# ----------------------------------------------------------------------------------------------------- 5. no retrace
@pytest.mark.parametrize("mode", MODES)
def test_no_retrace_across_stage_times_states_and_carries(bundle, mode):
    """``t``, the state and the carry are traced: the stage compiles once for any stage time, state and carry value (the
    bundle and the parameters are jit arguments of the driver programs; closed-over here)."""
    p = P_FAST
    rhs_fn = M.make_stage_rhs(bundle, mode, "continuum", p, "NNN-D")
    count = []

    @jax.jit
    def f(state, t, carry):
        count.append(1)
        return rhs_fn(state, t, carry)

    q = M.exact_state(bundle, p, 0.2)
    carry = M.initial_carry(bundle, mode, p, 0.2)
    outs = [f(M.NodalState.from_array(q * (1.0 + 0.01 * k)), t, carry + 0.1 * k) for k, t in enumerate((0.2, 0.21, 0.22, 0.2200001))]
    assert len(count) == 1 and f._cache_size() == 1
    assert float(jnp.abs(outs[0][0].array() - outs[1][0].array()).max()) > 0.0     # the time / state really enter


# ----------------------------------------------------------------------------------------------------- 6. StageInfo
def test_stage_info_fields_for_every_mode(bundle):
    p, t = P_FAST, 0.25
    q = M.exact_state(bundle, p, t)
    zeros = jnp.zeros(q.shape[:2])
    for mode in ("diffusion", "hyperbolic"):
        out, carry, info = stage(bundle, mode, "discrete", "DDDD", p, q, t, zeros + 3.0)
        assert isinstance(info, M.StageInfo) and isinstance(out, M.NodalState)
        assert int(info.cg_iterations) == 0 and float(info.cg_relative_residual) == 0.0 and bool(info.cg_converged)
        assert float(info.t) == t and bool(info.finite)
        np.testing.assert_array_equal(np.asarray(carry), np.asarray(zeros + 3.0))      # prescribed modes pass the carry through
        assert float(info.min_n) == float(q[..., 0].min()) and float(info.min_Te) == float(q[..., 1].min())
        assert float(info.min_Ti) == float(q[..., 2].min())
    out, carry, info = stage(bundle, "coupled", "discrete", "DDDD", p, q, t, zeros)
    assert bool(info.cg_converged) and int(info.cg_iterations) > 0 and 0.0 < float(info.cg_relative_residual) <= CG_RTOL
    assert bool(info.finite) and float(info.min_n) > 0.0
    # the carry is the psi of the solve (not the input): the solution of the elliptic problem at the state
    psi = M.initial_carry(bundle, "coupled", p, t)
    assert max_rel(carry, psi) < 1e-8 and float(jnp.abs(carry).max()) > 0.0
    # an exhausted iteration budget is reported
    _, _, bad = stage(bundle, "coupled", "discrete", "DDDD", p, q, t, zeros, phi_maxit=1)
    assert not bool(bad.cg_converged) and int(bad.cg_iterations) == 1
    # non-finite and non-positive states are reported
    qn = q.at[0, 0, 0].set(jnp.nan)
    assert not bool(stage(bundle, "diffusion", "discrete", "DDDD", p, qn, t, zeros)[2].finite)
    qm = q.at[1, 2, 1].set(-0.5)
    info_m = stage(bundle, "diffusion", "discrete", "DDDD", p, qm, t, zeros)[2]
    assert float(info_m.min_Te) == -0.5 and bool(info_m.finite)


def test_stage_rhs_plugs_into_rk4_stepper(bundle):
    """:func:`make_stage_rhs` plugs into :class:`Rk4Stepper`; a step returns four :class:`StageInfo` at the stage times and,
    with the discrete source and a slowly varying exact solution (``time_scale = 0.01``: the RK4 time error ``~ (omega dt)^5``
    is then below round-off), reproduces ``q(t + dt)``."""
    from drbx.native.fci_time_integrator import Rk4Stepper

    p, t, dt = P_SLOW, 0.2, 5e-4
    mode = "coupled"
    rhs_fn = M.make_stage_rhs(bundle, mode, "discrete", p, "NNN-D")
    state = M.NodalState.from_array(M.exact_state(bundle, p, t))
    res = jax.jit(lambda s, c: Rk4Stepper(rhs_fn)(s, time=t, timestep=dt, carry=c))(state, M.initial_carry(bundle, mode, p, t))
    assert len(res.stage_aux) == 4 and all(isinstance(a, M.StageInfo) for a in res.stage_aux)
    np.testing.assert_allclose([float(a.t) for a in res.stage_aux], [t, t + dt / 2, t + dt / 2, t + dt], rtol=1e-15)
    exact = M.exact_state(bundle, p, t + dt)
    change = max_rel(exact, M.exact_state(bundle, p, t))
    assert change > 1e-6                                                          # the exact solution moves
    assert max_rel(res.state.array(), exact) < 1e-3 * change and max_rel(res.state.array(), exact) < 1e-10
