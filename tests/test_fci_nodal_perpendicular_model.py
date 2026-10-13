"""Package model adapter of the nodal SBP perpendicular stack (``drbx.native.fci_nodal_perpendicular_model``): state, stage info, the
accepted defaults, the stage RHS factory, its compile strategy, JVP and the discrete-source exactness (m6), on the n = 16 family-A
synthetic bundle of the P10 harness (no HSX files).

The P10 harness ``model.py`` is a thin layer over this module (its callbacks are the manufactured wall data / source / potential);
``test_p10_evolved_model.py`` (unchanged) covers it through the harness API.
"""
from __future__ import annotations

import dataclasses
import sys
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

from drbx.native import fci_nodal_perpendicular_model as NM                       # noqa: E402
from drbx.native.fci_nodal_perpendicular_model import (NODAL_PERPENDICULAR_DEFAULTS as DEFAULTS, NodalState,   # noqa: E402
                                                       make_nodal_rhs, nodal_stage_rhs)
from drbx.native.fci_nodal_perpendicular_rhs import nodal_perpendicular_rhs, solve_potential   # noqa: E402
from drbx.native.fci_time_integrator import Rk4Stepper                           # noqa: E402
from p10_evolved_mms import bundle as P10B                                        # noqa: E402
from p10_evolved_mms import fields as F                                           # noqa: E402
from p10_evolved_mms import model as M                                            # noqa: E402
from p10_evolved_mms import source as S                                           # noqa: E402
from p10_evolved_mms import synthetic as syn                                      # noqa: E402

PATTERN = "NNN-D"
KINDS = S.pattern_kinds(PATTERN)
# O(1) advection, curvature and polarization (rho* = 0.7; a_omega rho*^2 = 1) so that every term is active
P_FAST = F.default_params(rho_star=0.7, time_scale=1.0, a_phi=1.0, a_omega=1.0 / 0.7 ** 2, D=np.array([0.01, 0.012, 0.014, 0.008]))
# slowly varying exact solution (time_scale = 0.05): the RK4 time error at dt = 5e-5 is at round-off
P_SLOW = F.default_params(rho_star=0.7, time_scale=0.05, a_phi=1.0, a_omega=1.0 / 0.7 ** 2, D=np.array([0.01, 0.012, 0.014, 0.008]))
T0 = 0.2


@pytest.fixture(scope="module")
def bundle():
    return syn.synthetic_bundle()


def exact(bundle, p, t):
    """``(q, phi)`` of the manufactured fields at the nodes."""
    V = F.FIELDS.values(bundle.ref.points, t, p)
    return V[..., :4], V[..., 4]


def mms_callbacks(bundle, p, opts, *, discrete: bool):
    """The MMS callbacks of the harness for ``opts`` (prescribed / solve), closing over ``bundle`` and ``p``: manufactured wall
    data, the (discrete or none) source, the exact ``phi`` (prescribed) or the discrete ``sigma_h`` (solve)."""
    params = p.nodal()
    wall_fn = lambda t, q: S.wall_data(bundle, t, p, PATTERN, opts.psi)                                  # noqa: E731
    d_of = lambda t, wall: S.discrete(bundle, bundle.ctx, opts, params, t, p, wall=wall)               # noqa: E731
    source_fn = (lambda t, q, wall: d_of(t, wall).S) if discrete else None                             # noqa: E731
    phi_fn = (lambda t, q: exact(bundle, p, t)[1]) if opts.phi_mode == "prescribed" else None          # noqa: E731
    sigma_fn = ((lambda t, q: d_of(t, wall_fn(t, q)).sigma) if (discrete and opts.phi_mode == "solve") else None)   # noqa: E731
    return wall_fn, source_fn, phi_fn, sigma_fn


# ----------------------------------------------------------------------------------------------------- state and stage info
def test_state_and_stage_info_are_the_package_objects():
    assert M.NodalState is NodalState and M.StageInfo is NM.StageInfo and M.make_stage_info is NM.make_stage_info
    a = jnp.asarray(np.random.default_rng(0).normal(size=(3, 5, 4)))
    s = NodalState.from_array(a)
    assert s.field_names() == ("n", "Te", "Ti", "omega")
    assert NM.NODAL_STATE_FIELDS == ("density", "Te", "Ti", "vorticity")
    np.testing.assert_array_equal(np.asarray(s.array()), np.asarray(a))
    with pytest.raises(ValueError):
        NodalState.from_array(jnp.zeros((3, 5, 6)))
    q, rhs = a, 2.0 * a
    info = NM.make_stage_info(0.25, q, rhs)
    assert float(info.t) == 0.25 and int(info.cg_iterations) == 0 and bool(info.cg_converged) and bool(info.finite)
    assert float(info.min_n) == float(a[..., 0].min()) and float(info.min_Ti) == float(a[..., 2].min())
    solved = NM.make_stage_info(0.5, q, rhs.at[0, 0, 0].set(jnp.inf), {"iterations": 7, "relative_residual": 3e-12, "converged": False})
    assert int(solved.cg_iterations) == 7 and float(solved.cg_relative_residual) == 3e-12
    assert not bool(solved.cg_converged) and not bool(solved.finite)


def test_zero_carry_has_the_plane_shape(bundle):
    c = NM.zero_carry(bundle.ctx)
    assert c.shape == (bundle.E, bundle.P) and c.dtype == jnp.float64 and float(jnp.abs(c).max()) == 0.0


# ----------------------------------------------------------------------------------------------------- accepted defaults
def test_defaults_are_frozen_and_hashable():
    assert dataclasses.is_dataclass(DEFAULTS) and hash(DEFAULTS) == hash(NM.NodalPerpendicularDefaults())
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULTS.tau = 2.0                                                          # type: ignore[misc]
    assert DEFAULTS.layout_family == "A" and DEFAULTS.psi == "phi_plus_tau_pi" and DEFAULTS.neumann_mode == "physical"
    assert DEFAULTS.curvature_variant == "centred" and not DEFAULTS.curvature_jump_dissipation and DEFAULTS.curvature_c_kappa == 0.0
    assert DEFAULTS.preconditioner == {"method": "core_schur", "factor_dtype": "float32"}


@pytest.mark.parametrize("phi_mode", ["prescribed", "solve"])
@pytest.mark.parametrize("pattern", ["NNN-D", "DDDD"])
def test_harness_options_equal_the_accepted_defaults(pattern, phi_mode):
    """``NODAL_PERPENDICULAR_DEFAULTS`` is the record of what ``configuration.json`` and ``source.nodal_options`` set: the harness
    options of every pattern and potential mode are exactly the defaults' options."""
    harness = S.nodal_options(pattern, phi_mode=phi_mode)
    assert harness == DEFAULTS.options(diffusion_kinds=S.pattern_kinds(pattern), phi_mode=phi_mode)
    assert harness.terms == DEFAULTS.terms and harness.fields == DEFAULTS.fields


def test_defaults_equal_the_harness_configuration():
    cfg = F.CONFIG
    assert cfg["family"] == DEFAULTS.layout_family and cfg["psi"] == DEFAULTS.psi and cfg["neumann_mode"] == DEFAULTS.neumann_mode
    assert cfg["tau"] == DEFAULTS.tau == F.default_params().tau
    assert cfg["bracket_c_kappa"] == DEFAULTS.bracket_c_kappa and cfg["laplacian_c_kappa"] == DEFAULTS.laplacian_c_kappa
    curv = cfg["curvature"]
    assert (curv["variant"], curv["jump_dissipation"], curv["c_kappa"], curv["absolute_method"]) == \
        (DEFAULTS.curvature_variant, DEFAULTS.curvature_jump_dissipation, DEFAULTS.curvature_c_kappa, DEFAULTS.absolute_method)
    assert (cfg["solve"]["rtol"], cfg["solve"]["maxit"]) == (DEFAULTS.phi_rtol, DEFAULTS.phi_maxit) == (1e-11, 50)
    assert cfg["solve"]["preconditioner"] == DEFAULTS.preconditioner_method
    assert cfg["solve"]["factor_dtype"] == DEFAULTS.preconditioner_factor_dtype
    assert tuple(cfg["fields"]) == DEFAULTS.fields == NM.NODAL_STATE_FIELDS
    assert P10B.PRECONDITIONER == DEFAULTS.preconditioner                              # the harness bundle's preconditioner
    assert DEFAULTS.options(diffusion_kinds=KINDS, terms=("diffusion",)).terms == ("diffusion",)         # an override replaces


def test_the_bundle_records_the_default_preconditioner(bundle):
    rec = bundle.identity["preconditioner"]
    assert {k: rec[k] for k in ("method", "factor_dtype")} == DEFAULTS.preconditioner


# ----------------------------------------------------------------------------------------------------- the stage factory
def test_prescribed_stage_is_the_direct_composition_and_passes_the_carry_through(bundle):
    p, t = P_FAST, T0
    q, phi = exact(bundle, p, t)
    opts = DEFAULTS.options(diffusion_kinds=KINDS, phi_mode="prescribed")
    wall = S.wall_data(bundle, t, p, PATTERN)
    source = 0.3 * jnp.cos(q)
    rhs_fn = make_nodal_rhs(bundle.ctx, p.nodal(), opts, terms=DEFAULTS.terms, phi_mode="prescribed",
                            wall_fn=lambda tt, qq: wall, source_fn=lambda tt, qq, w: source, phi_fn=lambda tt, qq: phi)
    carry = jnp.full(q.shape[:2], 3.0)
    out, new_carry, info = rhs_fn(NodalState.from_array(q), t, carry)
    direct = nodal_perpendicular_rhs(bundle.ctx, rhs_fn.options, p.nodal(), q, wall, phi=phi, source=source)
    assert isinstance(out, NodalState) and np.array_equal(np.asarray(out.array()), np.asarray(direct.total))
    assert float(jnp.abs(out.array()).max()) > 1e-3
    assert np.array_equal(np.asarray(new_carry), np.asarray(carry))                  # prescribed: the carry is untouched
    assert float(info.t) == t and int(info.cg_iterations) == 0 and bool(info.cg_converged) and bool(info.finite)
    assert rhs_fn.options.phi_mode == "prescribed" and rhs_fn.options.terms == DEFAULTS.terms
    # the pure form is the same program with the context and the parameters as arguments
    u_out, u_carry, _ = rhs_fn.unbound(bundle.ctx, p.nodal(), NodalState.from_array(q), t, carry)
    assert np.array_equal(np.asarray(u_out.array()), np.asarray(out.array())) and np.array_equal(np.asarray(u_carry), np.asarray(carry))
    # ``terms`` and ``phi_mode`` replace the ones of the options
    only = make_nodal_rhs(bundle.ctx, p.nodal(), opts, terms=("diffusion",), phi_mode="prescribed", wall_fn=lambda tt, qq: wall,
                          phi_fn=lambda tt, qq: phi)
    got = only(NodalState.from_array(q), t, carry)[0].array()
    assert np.array_equal(np.asarray(got), np.asarray(direct.diffusion))


def test_solve_stage_is_the_direct_composition_and_carries_psi(bundle):
    p, t = P_FAST, T0
    q, _phi = exact(bundle, p, t)
    opts = DEFAULTS.options(diffusion_kinds=KINDS, phi_mode="solve", phi_rtol=1e-12, phi_maxit=100)
    wall = S.wall_data(bundle, t, p, PATTERN)
    sigma = 0.01 * jnp.sin(q[..., 3] + 1.0)
    source = 0.3 * jnp.cos(q)
    rhs_fn = make_nodal_rhs(bundle.ctx, p.nodal(), opts, terms=DEFAULTS.terms, phi_mode="solve", wall_fn=lambda tt, qq: wall,
                            source_fn=lambda tt, qq, w: source, sigma_fn=lambda tt, qq: sigma)
    cold = NM.zero_carry(bundle.ctx)
    out, psi, info = rhs_fn(NodalState.from_array(q), t, cold)
    direct = nodal_perpendicular_rhs(bundle.ctx, rhs_fn.options, p.nodal(), q, wall, psi_x0=cold, sigma=sigma, source=source)
    assert np.array_equal(np.asarray(out.array()), np.asarray(direct.total)) and np.array_equal(np.asarray(psi), np.asarray(direct.psi))
    assert bool(info.cg_converged) and int(info.cg_iterations) > 3 and 0.0 < float(info.cg_relative_residual) <= 1e-12
    # the carry is the psi of the solve: the polarization problem with these data, whatever the warm start
    psi_ref, _phi_ref, _ = solve_potential(bundle.ctx, rhs_fn.options, p.nodal(), q[..., 3], q[..., 0], q[..., 2], wall.psi, sigma=sigma)
    assert float(jnp.abs(psi - psi_ref).max() / jnp.abs(psi_ref).max()) < 1e-10
    # warm start from that psi: the solve is already converged, the RHS agrees to the solve tolerance
    out_w, psi_w, info_w = rhs_fn(NodalState.from_array(q), t, psi)
    assert int(info_w.cg_iterations) < int(info.cg_iterations)
    np.testing.assert_allclose(np.asarray(out_w.array()), np.asarray(out.array()), rtol=1e-9, atol=1e-10 * float(jnp.abs(out.array()).max()))
    # no sigma_fn: sigma = 0
    plain = make_nodal_rhs(bundle.ctx, p.nodal(), opts, terms=DEFAULTS.terms, phi_mode="solve", wall_fn=lambda tt, qq: wall)
    direct0 = nodal_perpendicular_rhs(bundle.ctx, rhs_fn.options, p.nodal(), q, wall, psi_x0=cold)
    assert np.array_equal(np.asarray(plain(NodalState.from_array(q), t, cold)[0].array()), np.asarray(direct0.total))


def test_factory_validation(bundle):
    p = P_FAST
    opts = DEFAULTS.options(diffusion_kinds=KINDS)
    wall_fn, phi_fn, sigma_fn = (lambda t, q: None), (lambda t, q: None), (lambda t, q: None)
    ctx, params = bundle.ctx, p.nodal()
    with pytest.raises(ValueError, match="phi_fn is required"):
        make_nodal_rhs(ctx, params, opts, terms=DEFAULTS.terms, phi_mode="prescribed", wall_fn=wall_fn)
    with pytest.raises(ValueError, match="phi_fn is given"):
        make_nodal_rhs(ctx, params, opts, terms=DEFAULTS.terms, phi_mode="solve", wall_fn=wall_fn, phi_fn=phi_fn)
    with pytest.raises(ValueError, match="sigma_fn is given"):
        make_nodal_rhs(ctx, params, opts, terms=DEFAULTS.terms, phi_mode="prescribed", wall_fn=wall_fn, phi_fn=phi_fn,
                       sigma_fn=sigma_fn)
    with pytest.raises(ValueError, match="fields"):
        make_nodal_rhs(ctx, params, dataclasses.replace(opts, fields=("density", "Te", "Ti", "Vi", "Ve", "vorticity"),
                                                        diffusion_kinds=("dirichlet",) * 6),
                       terms=("bracket",), phi_mode="solve", wall_fn=wall_fn)
    with pytest.raises(ValueError, match="terms"):
        make_nodal_rhs(ctx, params, opts, terms=("bracket", "bracket"), phi_mode="solve", wall_fn=wall_fn)
    with pytest.raises(ValueError, match="phi_mode"):
        make_nodal_rhs(ctx, params, opts, terms=DEFAULTS.terms, phi_mode="implicit", wall_fn=wall_fn)
    with pytest.raises(ValueError, match="diffusion_kinds"):
        make_nodal_rhs(ctx, params, DEFAULTS.options(), terms=DEFAULTS.terms, phi_mode="solve", wall_fn=wall_fn)
    rhs_fn = make_nodal_rhs(ctx, params, opts, terms=("bracket",), phi_mode="prescribed", wall_fn=wall_fn, phi_fn=phi_fn)
    with pytest.raises(TypeError, match="NodalState"):
        rhs_fn(jnp.zeros((bundle.E, bundle.P, 4)), 0.0, jnp.zeros((bundle.E, bundle.P)))
    with pytest.raises(ValueError, match="phi_fn"):
        nodal_stage_rhs(ctx, params, NodalState.from_array(jnp.zeros((bundle.E, bundle.P, 4))), 0.0, None,
                        options=dataclasses.replace(opts, phi_mode="prescribed"), wall_fn=wall_fn)


@pytest.mark.parametrize("phi_mode", ["prescribed", "solve"])
def test_stage_plugs_into_rk4_stepper(bundle, phi_mode):
    p, t, dt = P_SLOW, T0, 1e-4
    q, _ = exact(bundle, p, t)
    opts = DEFAULTS.options(diffusion_kinds=KINDS, phi_mode=phi_mode)
    wall_fn, source_fn, phi_fn, sigma_fn = mms_callbacks(bundle, p, opts, discrete=True)
    rhs_fn = make_nodal_rhs(bundle.ctx, p.nodal(), opts, terms=DEFAULTS.terms, phi_mode=phi_mode, wall_fn=wall_fn,
                            source_fn=source_fn, phi_fn=phi_fn, sigma_fn=sigma_fn)
    carry = M.initial_carry(bundle, "coupled" if phi_mode == "solve" else "hyperbolic", p, t)
    res = jax.jit(lambda s, c: Rk4Stepper(rhs_fn)(s, time=t, timestep=dt, carry=c))(NodalState.from_array(q), carry)
    assert len(res.stage_aux) == 4 and all(isinstance(a, NM.StageInfo) for a in res.stage_aux)
    np.testing.assert_allclose([float(a.t) for a in res.stage_aux], [t, t + dt / 2, t + dt / 2, t + dt], rtol=1e-15)
    assert all(bool(a.finite) and bool(a.cg_converged) for a in res.stage_aux)
    assert res.carry.shape == carry.shape


# ----------------------------------------------------------------------------------------------------- compile strategy
@pytest.mark.parametrize("phi_mode", ["prescribed", "solve"])
def test_one_compilation_serves_every_stage_time(bundle, phi_mode):
    """``t`` is traced: ``jit(rhs_fn)`` compiles once for any stage time, state and carry (closed-over geometry), and the time
    really enters through the callbacks."""
    p = P_FAST
    opts = DEFAULTS.options(diffusion_kinds=KINDS, phi_mode=phi_mode)
    wall_fn, source_fn, phi_fn, sigma_fn = mms_callbacks(bundle, p, opts, discrete=False)
    rhs_fn = make_nodal_rhs(bundle.ctx, p.nodal(), opts, terms=DEFAULTS.terms, phi_mode=phi_mode, wall_fn=wall_fn,
                            phi_fn=phi_fn, sigma_fn=sigma_fn)
    count = []

    @jax.jit
    def f(state, t, carry):
        count.append(1)
        return rhs_fn(state, t, carry)

    q, _ = exact(bundle, p, T0)
    carry = jnp.zeros(q.shape[:2])
    outs = [f(NodalState.from_array(q * (1.0 + 0.01 * k)), t, carry + 0.1 * k) for k, t in enumerate((0.2, 0.21, 0.22, 0.2200001))]
    assert len(count) == 1 and f._cache_size() == 1
    assert float(jnp.abs(outs[0][0].array() - outs[1][0].array()).max()) > 0.0


@pytest.mark.parametrize("phi_mode", ["prescribed", "solve"])
def test_geometry_and_parameters_are_jit_arguments_not_constants(bundle, phi_mode):
    """``rhs_fn.unbound(context, params, ...)``: the context and the parameters are traced arguments, so the compiled program takes
    new values without recompiling and no geometry value is baked into it (the output follows them)."""
    p, t = P_FAST, T0
    opts = DEFAULTS.options(diffusion_kinds=KINDS, phi_mode=phi_mode)
    q, phi = exact(bundle, p, t)
    wall = S.wall_data(bundle, t, p, PATTERN)
    rhs_fn = make_nodal_rhs(bundle.ctx, p.nodal(), opts, terms=DEFAULTS.terms, phi_mode=phi_mode,
                            wall_fn=lambda tt, qq: wall, phi_fn=(lambda tt, qq: phi) if phi_mode == "prescribed" else None)
    count = []

    def program(ctx, params, state, t_, carry):
        count.append(1)
        return rhs_fn.unbound(ctx, params, state, t_, carry)

    f = jax.jit(program)
    carry = jnp.zeros(q.shape[:2])
    state = NodalState.from_array(q)
    base = f(bundle.ctx, p.nodal(), state, 0.2, carry)[0].array()
    f(bundle.ctx, p.nodal(), state, 0.3, carry)                                       # another stage time
    ctx2 = bundle.ctx._replace(curvature_flux=bundle.ctx.curvature_flux * 1.5)         # another geometry value, same structure
    moved = f(ctx2, p.nodal(), state, 0.2, carry)[0].array()
    params2 = p.nodal()._replace(rho_star=p.nodal().rho_star * 0.5)
    scaled = f(bundle.ctx, params2, state, 0.2, carry)[0].array()
    assert len(count) == 1 and f._cache_size() == 1
    assert float(jnp.abs(moved - base).max()) > 1e-6 * float(jnp.abs(base).max())
    assert float(jnp.abs(scaled - base).max()) > 1e-6 * float(jnp.abs(base).max())


# ----------------------------------------------------------------------------------------------------- JVP
def _smooth_inputs(bundle, mode):
    """``(p, q, phi, wall, opts, v, sigma)``: the exact smooth state at ``t = 0.22``, a random direction ``v`` and the discrete
    polarization source ``sigma_h`` (so the solve has a smooth solution, as in the evolved MMS)."""
    p, t = P_FAST, 0.22
    q, phi = exact(bundle, p, t)
    wall = S.wall_data(bundle, t, p, PATTERN)
    opts = DEFAULTS.options(diffusion_kinds=KINDS, phi_mode=mode, phi_rtol=1e-13, phi_maxit=100)
    sigma = S.discrete(bundle, bundle.ctx, opts, p.nodal(), t, p, wall=wall).sigma
    v = jnp.asarray(np.random.default_rng(3).normal(size=q.shape)) * jnp.abs(q).max(axis=(0, 1))
    return p, q, phi, wall, opts, v, sigma


@pytest.mark.parametrize("mode", ["prescribed", "solve"])
def test_jvp_of_nodal_perpendicular_rhs_matches_central_finite_differences(bundle, mode):
    """``d total / d state`` along a random direction vs a central difference (``eps = 1e-6``: truncation ``~ eps^2``, round-off
    ``~ 1e-16 / eps``). Solve mode: a *cold* start and ``phi_rtol = 1e-13``; forward mode differentiates the CG iterations
    (see ``test_jvp_in_solve_mode_is_limited_to_a_cold_start_and_forward_mode``)."""
    p, q, phi, wall, opts, v, sigma = _smooth_inputs(bundle, mode)
    params = p.nodal()
    if mode == "prescribed":
        f = lambda x: nodal_perpendicular_rhs(bundle.ctx, opts, params, x, wall, phi=phi).total          # noqa: E731
    else:
        f = lambda x: nodal_perpendicular_rhs(bundle.ctx, opts, params, x, wall, sigma=sigma).total       # noqa: E731
    _, jv = jax.jvp(f, (q,), (v,))
    eps = 1e-6
    fd = (f(q + eps * v) - f(q - eps * v)) / (2 * eps)
    assert float(jnp.abs(jv).max()) > 1e-3
    assert float(jnp.abs(jv - fd).max() / jnp.abs(jv).max()) <= 1e-6


def test_prescribed_mode_is_reverse_differentiable(bundle):
    """``jax.vjp`` through the prescribed-phi RHS: the dot-product test ``<w, J v> = <J^T w, v>``."""
    p, q, phi, wall, opts, v, _sigma = _smooth_inputs(bundle, "prescribed")
    f = lambda x: nodal_perpendicular_rhs(bundle.ctx, opts, p.nodal(), x, wall, phi=phi).total              # noqa: E731
    _, jv = jax.jvp(f, (q,), (v,))
    out, pullback = jax.vjp(f, q)
    w = jnp.asarray(np.random.default_rng(4).normal(size=out.shape))
    (jtw,) = pullback(w)
    lhs, rhs = float(jnp.sum(w * jv)), float(jnp.sum(jtw * v))
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), abs(rhs))


def test_jvp_in_solve_mode_is_limited_to_a_cold_start_and_forward_mode(bundle):
    """The documented limits of the *bare* CG of ``solve_potential`` (it is a ``lax.while_loop``; ``solve_dirichlet_implicit`` is the
    differentiable solve and is not used by the RHS): (1) reverse mode fails: ``jax.vjp`` builds, its pullback raises the ``while_loop`` error; (2)
    forward mode differentiates the iterations, so a warm start that already solves the system (zero iterations) has a zero
    tangent for ``psi`` and the JVP of the RHS misses the polarization response (the cold start of the previous test is exact)."""
    p, q, phi, wall, opts, v, sigma = _smooth_inputs(bundle, "solve")
    params = p.nodal()
    f = lambda x, x0=None: nodal_perpendicular_rhs(bundle.ctx, opts, params, x, wall, psi_x0=x0, sigma=sigma).total   # noqa: E731
    out_v, pullback = jax.vjp(f, q)                                                                  # the forward pass is fine ...
    with pytest.raises(ValueError, match="Reverse-mode differentiation does not work for lax.while_loop"):
        pullback(jnp.ones_like(out_v))                                                               # ... the transpose is not
    out = nodal_perpendicular_rhs(bundle.ctx, opts, params, q, wall, sigma=sigma)
    assert int(out.solve_info["iterations"]) > 3
    psi = out.psi
    g = lambda x: f(x, psi)                                                                          # noqa: E731
    assert int(nodal_perpendicular_rhs(bundle.ctx, opts, params, q, wall, psi_x0=psi, sigma=sigma).solve_info["iterations"]) == 0
    _, jv = jax.jvp(g, (q,), (v,))
    eps = 1e-6
    fd = (f(q + eps * v) - f(q - eps * v)) / (2 * eps)                                               # cold-start FD: the true derivative
    err = float(jnp.abs(jv - fd).max() / jnp.abs(fd).max())
    assert err > 1e-4, err                                                                           # the polarization response is missing


# ----------------------------------------------------------------------------------------------------- m6: discrete-source exactness
def _run_exact(bundle, phi_mode, nsteps=3, dt=5e-5):
    """RK4 from the exact state with the discrete source: ``(max relative error of q at each step, cg convergence)``."""
    p = P_SLOW
    opts = DEFAULTS.options(diffusion_kinds=KINDS, phi_mode=phi_mode, phi_rtol=1e-12, phi_maxit=100)
    wall_fn, source_fn, phi_fn, sigma_fn = mms_callbacks(bundle, p, opts, discrete=True)
    rhs_fn = make_nodal_rhs(bundle.ctx, p.nodal(), opts, terms=DEFAULTS.terms, phi_mode=phi_mode, wall_fn=wall_fn,
                            source_fn=source_fn, phi_fn=phi_fn, sigma_fn=sigma_fn)
    stepper = Rk4Stepper(rhs_fn)
    step = jax.jit(lambda s, c, t: stepper(s, time=t, timestep=dt, carry=c))
    q0, _ = exact(bundle, p, 0.0)
    state = NodalState.from_array(q0)
    carry = M.initial_carry(bundle, "coupled" if phi_mode == "solve" else "hyperbolic", p, 0.0)
    errs, ok = [], True
    for k in range(nsteps):
        res = step(state, carry, k * dt)
        state, carry = res.state, res.carry
        ok = ok and all(bool(a.cg_converged) for a in res.stage_aux)
        qe, _ = exact(bundle, p, (k + 1) * dt)
        errs.append(float(jnp.abs(state.array() - qe).max() / jnp.abs(qe).max()))
    moved = float(jnp.abs(exact(bundle, p, nsteps * dt)[0] - q0).max())
    return errs, ok, moved


def test_m6_exact_nodal_state_solves_the_discrete_source_system_prescribed(bundle):
    """With ``S_h = d_t q - R_h(q)`` the exact nodal state solves the semi-discrete system: the RK4 state stays on it to round-off
    (at dt = 5e-5 the RK4 time error is below it; existing harness test: ``test_p10_evolved_evolve``, 20 steps, 4e-16 / 6e-15)."""
    errs, ok, moved = _run_exact(bundle, "prescribed")
    assert ok and max(errs) <= 1e-12, errs
    assert moved > 1e-6                                                                              # the exact solution moves


def test_m6_exact_nodal_state_solves_the_discrete_source_system_coupled(bundle):
    """Coupled (psi solve with ``sigma_h``): the solve is only exact to the CG tolerance (1e-12 here), so the state stays on the
    exact one to a few times that tolerance times the RHS size (see the harness test for the budget)."""
    errs, ok, moved = _run_exact(bundle, "solve")
    assert ok and max(errs) <= 1e-10, errs
    assert moved > 1e-6
