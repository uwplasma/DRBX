"""Regression tests of the mechanisms found while qualifying the nodal SBP perpendicular stack for the P10 evolved MMS.

Each test pins one mechanism on a small synthetic case (no HSX files); where an earlier test file already covers a mechanism the
mapping is recorded in the module table below instead of duplicating it.

====  ==========================================================================  ==============================================
 m    mechanism                                                                   test
====  ==========================================================================  ==============================================
 m1   float-byte shas are not portable: recorded, never rejected; tolerance        ``test_fci_nodal_geometry_bundle.py::test_m1_*``
 m2   split-form bracket: energy identity, rightmost eigenvalue <= max c/2,       here: ``test_m2_*``
      upwinding / dissipation removes the centred compression mode
 m3   curvature: energy identity, W2 interchange; wall rows are the inflow SAT    ``test_fci_perpendicular_sbp_curvature.py``
      on incoming characteristics (never a Neumann closure)                       (``test_antisymmetry_identity_*``,
                                                                                  ``test_wall_sat_energy_sign_frozen_matrix``),
                                                                                  here: ``test_m3_w2_interchange_*``
 m4   Laplacian: H-symmetric; Dirichlet definite; Neumann near-null bounded        ``test_fci_perpendicular_sbp_laplacian.py``
                                                                                  (``test_probed_jax_operator_is_h_symmetric_*``,
                                                                                  ``test_dirichlet_definiteness_needs_the_shell_*``)
 m5   MMS vorticity scales as a_omega rho*^2 Omega_shape                          ``test_p10_evolved_fields.py::
                                                                                  test_a_omega_scales_the_vorticity_*``
 m6   exact nodal state solves the discrete-source system                         ``test_fci_nodal_perpendicular_model.py::
                                                                                  test_m6_*``; ``test_p10_evolved_evolve.py::
                                                                                  test_discrete_source_run_stays_*`` (slow)
 m7   eta-parity (W1): the eta-harmonic-2 vorticity has the error order of         here: ``test_m7_*`` (diagnostic)
      the main field
====  ==========================================================================  ==============================================
"""
from __future__ import annotations

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

from drbx.geometry.nodal_layout import build_nodal_layout                         # noqa: E402
from drbx.native import fci_perpendicular_sbp_bracket as br                       # noqa: E402
from drbx.native import fci_perpendicular_sbp_curvature as cv                     # noqa: E402
from drbx.native import fci_perpendicular_sbp_ops as ops                          # noqa: E402
from drbx.stencils.nodal_plan import build_nodal_plan, nodal_metric_from_callable   # noqa: E402

TWO_PI = 2.0 * np.pi
HALO = 3


def metric_fn(p):
    """Smooth ``(h, |J|, B, K)`` with an eta-dependent jacobian (the metric of the bracket / curvature test files)."""
    u, th, et = p.T
    h = np.stack([0.1 * np.sin(th), 0.2 * np.cos(et), 1.0 + 0.1 * u], -1)
    K = np.stack([0.3 * np.cos(th), 0.1 * u + 0.05, 0.2 * np.sin(et) + 0.1], -1)
    return h, u * (1.0 + 0.1 * np.cos(th) * np.cos(et)), 1.0 + 0.2 * u, K


@pytest.fixture(scope="module")
def case():
    """n = 24, two ring levels (8 and 24 nodes, one level face), inner wall, 4 planes: P = 256, dense operators of size 1024."""
    n, E = 24, 4
    lay = build_nodal_layout(n, [(8, 16, 8), (16, 24, 24)], n_eta=E, inner="wall")
    return lay, build_nodal_plan(lay, nodal_metric_from_callable(lay, metric_fn)), E


def hsum(plan, a, b):
    Hp = np.asarray(plan.Hp)
    return float(np.sum(Hp.reshape(Hp.shape + (1,) * (np.ndim(a) - 2)) * np.asarray(a) * np.asarray(b)))


def wall_traces(plan, F, *gs):
    out = []
    for blk, side, sign, N in plan.structure.walls:
        a = [np.asarray(ops.trace(plan, blk, side, g)) for g in gs]
        v = np.asarray(ops.flux_trace(plan, blk, side, F[..., 0], F[..., 1]))
        out.append((sign, N, a, v))
    return out


# ----------------------------------------------------------------------------------------------------- m2: the bracket
def _potential(lay, E):
    u, th = lay.node_u[None, :], lay.node_theta[None, :]
    et = (np.arange(E)[:, None] + 0.5) * lay.deta
    phi = ((1.0 + u + 3.0 * u ** 4) * (np.cos(2 * th + 0.3) + 0.5 * np.sin(3 * th)) + np.sin(5 * u) * np.cos(th)) * (1.0 + 0.3 * np.cos(et))
    return np.asarray(phi)


def test_m2_bracket_energy_identity_to_round_off(case):
    """Split form: ``(g, L g)_H = wall + 1/2 (c, g^2)_H + inflow + upwind`` for random ``g``, the residual relative to the size of
    the terms below 1e-15 (the energy identity of the SBP bracket; the existing bracket tests pin it to 1e-13 of a norm product)."""
    lay, plan, E = case
    rng = np.random.default_rng(7)
    F = np.asarray(br.velocity_flux(plan, jnp.asarray(_potential(lay, E)), 0.05))
    g = rng.standard_normal((E, lay.P))
    L = np.asarray(br.linear_operator(plan, F, g, None))
    c = np.asarray(br.compressibility(plan, F))
    walls = wall_traces(plan, F, g)
    wall = sum(-0.5 * (TWO_PI / N) * np.sum(sign * v * a[0] ** 2) for sign, N, a, v in walls)
    inflow = -sum((TWO_PI / N) * np.sum(np.maximum(-sign * v, 0.0) * a[0] ** 2) for sign, N, a, v in walls)
    (a_blk, b_blk, NA, NB, X), = plan.structure.faces
    assert X == "B"
    face = plan.faces[0]
    a, b = np.asarray(ops.trace(plan, a_blk, "outer", g)), np.asarray(ops.trace(plan, b_blk, "inner", g))
    vA = np.asarray(ops.flux_trace(plan, a_blk, "outer", F[..., 0], F[..., 1]))
    vB = np.asarray(ops.flux_trace(plan, b_blk, "inner", F[..., 0], F[..., 1]))
    jump = b - a @ np.asarray(face.Iab).T
    gamma = 0.5 * np.abs(0.5 * (vA @ np.asarray(face.Iab).T + vB))
    upwind = -np.sum((TWO_PI / NB) * gamma * jump * jump)
    terms = (wall, 0.5 * hsum(plan, c, g * g), inflow, upwind)
    assert min(abs(t) for t in terms) > 0.0                                          # every term is active
    residual = abs(hsum(plan, g, L) - sum(terms)) / sum(abs(t) for t in terms)
    assert residual <= 1e-15, residual


def _dense(apply, E, P):
    return np.asarray(jax.jit(jax.jacfwd(lambda x: apply(x.reshape(E, P)).ravel()))(jnp.zeros(E * P)))


def test_m2_rightmost_eigenvalue_is_bounded_by_max_half_c_and_upwinding_removes_the_compression_mode(case):
    """Frozen-potential operator on a compressive flow: the centred operator ``T + c/2 + inflow`` has its numerical abscissa (hence its
    rightmost eigenvalue) at most ``max c/2`` and a positive rightmost eigenvalue (the compression mode); the full bracket (level-face
    upwind SAT and jump dissipation, ``bracket_c_kappa = 1``) keeps the bound and has no growing mode on this case."""
    lay, plan, E = case
    P = lay.P
    F = br.velocity_flux(plan, jnp.asarray(_potential(lay, E)), 0.05)
    c = np.asarray(br.compressibility(plan, F))
    Fe = ops.extend_periodic(F, HALO)
    max_half_c = 0.5 * c.max()
    assert max_half_c > 1.0 and c.min() < 0.0                                         # compressive and expansive regions

    def centred(g):
        return br.transport_ext(plan, Fe, ops.extend_periodic(g, HALO)) + 0.5 * c * g + br.wall_inflow(plan, F, g, None)

    def full(g):
        return br.sbp_bracket_ext(plan, Fe, ops.extend_periodic(g, HALO), None, 1.0)

    Hh = np.sqrt((np.asarray(plan.Hp) * plan.structure.deta).ravel())
    out = {}
    for name, op in (("centred", centred), ("full", full)):
        M = _dense(op, E, P)
        S = Hh[:, None] * M / Hh[None, :]                                             # H^{1/2} M H^{-1/2}: symmetric part = energy form
        out[name] = (np.linalg.eigvals(M).real.max(), np.linalg.eigvalsh(0.5 * (S + S.T)).max())
    (rc, ac), (rf, af) = out["centred"], out["full"]
    print(f"max c/2 = {max_half_c:.4f}; centred: rightmost {rc:.4f}, abscissa {ac:.4f}; full: rightmost {rf:.4f}, abscissa {af:.4f}")
    tol = 1e-9 * max_half_c
    assert rc <= ac + tol and ac <= max_half_c + tol                                  # eigenvalue <= numerical abscissa <= max c/2
    assert rf <= af + tol and af <= max_half_c + tol
    assert rc > 0.1 * max_half_c                                                      # the centred bracket has a compression mode ...
    assert rf < 0.0                                                                   # ... which upwinding and dissipation remove
    assert rf < rc


# ----------------------------------------------------------------------------------------------------- m3: curvature interchange
def test_m3_w2_interchange_identity_to_round_off(case):
    """W2 (linear coupling pair ``C(p)`` in the vorticity equation against ``C(phi)`` in the pressure equations): the discrete
    derivative along the curvature flux satisfies ``(phi, C p)_H + (p, C phi)_H = wall - (c, phi p)_H`` exactly (the pair is adjoint
    in H up to the divergence ``c`` and the wall term), residual relative to the terms at round-off (roadmap: <= 3e-17)."""
    lay, plan, E = case
    rng = np.random.default_rng(11)
    F = rng.standard_normal((E, lay.P, 3))
    phi = rng.standard_normal((E, lay.P))
    pres = 1.0 + 0.3 * rng.standard_normal((E, lay.P))
    g = np.stack([phi, pres], axis=-1)
    Fe = ops.extend_periodic(jnp.asarray(F), HALO)
    c = np.asarray(cv.compressibility_flux(plan, Fe))
    Cg = np.asarray(cv.curvature_derivative_ext(plan, Fe, ops.extend_periodic(jnp.asarray(g), HALO)))
    lhs = hsum(plan, phi, Cg[..., 1]) + hsum(plan, pres, Cg[..., 0])
    wall = sum((TWO_PI / N) * np.sum(sign * v * a[0][..., 0] * a[0][..., 1]) for sign, N, a, v in wall_traces(plan, F, g))
    c_term = hsum(plan, c, phi * pres)
    scale = abs(wall) + abs(c_term) + abs(lhs)
    assert min(abs(wall), abs(c_term)) > 0.0
    assert abs(lhs - (wall - c_term)) <= 1e-13 * scale, abs(lhs - (wall - c_term)) / scale


# ----------------------------------------------------------------------------------------------------- m7: eta parity (W1)
@pytest.mark.diagnostic
def test_m7_w1_eta_harmonic_2_vorticity_has_the_error_order_of_the_main_field():
    """Diagnostic (reported-only case of the evolved MMS). The step-6 vorticity has eta harmonic 1, which cannot exercise the E x B
    energy exchange (W1); the variant ``w1 = 1`` adds an eta-harmonic-2 term to ``Omega`` (``phi``, ``n``, ``Te``, ``Ti`` are
    unchanged). The vorticity row of the RHS with a prescribed ``phi`` is linear in ``Omega`` (bracket and diffusion; the curvature
    row has no ``Omega`` column), so the difference of the ``w1 = 1`` and ``w1 = 0`` rows isolates the added term. Its discrete
    truncation error must be of the same order as that of the main field's row (same order of magnitude at each ``n``) and must
    decrease with ``n`` for both, on the synthetic bundles ``n = 16`` and ``n = 24``."""
    from p10_evolved_mms import fields as F
    from p10_evolved_mms import source as S
    from p10_evolved_mms import synthetic as syn

    base = F.default_params(rho_star=0.05, time_scale=1.0, a_phi=0.5, a_omega=1.0 / 0.05 ** 2)
    t, opts = 0.3, S.nodal_options("NNN-D")

    def errors(n, n_eta):
        b = syn.synthetic_bundle() if n == 16 else syn.synthetic_bundle(n, n_eta, "raw", False)
        c0, c1 = (S.continuum_jit(b, t, base._replace(w1=w1)) for w1 in (0.0, 1.0))
        d0, d1 = (S.discrete_jit(b, None, opts, None, t, base._replace(w1=w1)) for w1 in (0.0, 1.0))
        norm = lambda a: float(jnp.sqrt(jnp.sum(b.H * a ** 2)))                      # noqa: E731
        main = norm(d0.R[..., 3] - c0.R[..., 3]) / norm(c0.R[..., 3])
        dc, dd = c1.R[..., 3] - c0.R[..., 3], d1.R[..., 3] - d0.R[..., 3]
        w1 = norm(dd - dc) / norm(dc)
        assert norm(dc) > 0.1 * norm(c0.R[..., 3])                                    # the added term really enters the row
        return main, w1

    (m16, w16), (m24, w24) = errors(16, 8), errors(24, 12)
    print(f"vorticity-row truncation: main {m16:.4f} -> {m24:.4f}, w1 term {w16:.4f} -> {w24:.4f}")
    assert m24 < m16 / 1.3 and w24 < w16 / 1.3                                       # both converge
    assert 0.1 < w16 / m16 < 10.0 and 0.1 < w24 / m24 < 10.0                         # the same order of error
