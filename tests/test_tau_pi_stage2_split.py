"""tau p_i stage 2: the curvature and parallel splits use psi = phi + tau*n*Ti.

Stage 1 changed only the polarization relation to ``omega = L_perp(phi + tau*p_i)``,
``p_i = n*Ti``.  Stage 2 re-splits the two internal pairs in lockstep:

* curvature: material ``M(psi) C(q)`` plus remainder ``-c C(psi)``;
* parallel: the Ve principal matrix ``A(psi)`` plus the composite force ``mu G(psi)``.

Each pair cancels in the continuum; the matrices are what the upwinding and the
characteristic dissipation are built on.  The matrix functions take ``psi=`` with the
legacy default (the frozen campaigns import them), the model passes
``FciDrbEBParameters.polarization_variable``.  These tests pin the matrix entries of both
selectors, the legacy matrices bitwise, the cancellation of each pair for both selectors, and
the closed-form / polynomial characteristic evaluations that were derived from the matrices.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from drbx.native.fci_curvature_production_flux import (
    curvature_face_linearized_fluctuations,
    curvature_flux_jacobian,
    curvature_principal_matrix,
    curvature_strict_principal_matrix,
)
from drbx.native.fci_parallel_production_flux import (
    parallel_characteristic_split,
    parallel_matrix_from_state,
    parallel_production_principal_matrix,
)
from drbx.native.fci_perpendicular_face_corrections import (
    _absolute_action_closed_form,
    _absolute_matrix_action,
    _absolute_primal,
    p06_midpoint_material_remainder,
)
from drbx.native.q_characteristic_polynomial import (
    polynomial_basis,
    polynomial_characteristic_split,
)
from drbx.native.q_parallel_material import material_from_slots

LEGACY = "phi_plus_tau_ti"
PI = "phi_plus_tau_pi"
MU = 1836.0


def _states(count=64, seed=0, *, tau_values=(0.0, 0.1, 1.0, 3.0)):
    rng = np.random.default_rng(seed)
    n = np.exp(rng.uniform(np.log(0.1), np.log(10.0), count))
    te = np.exp(rng.uniform(np.log(0.1), np.log(10.0), count))
    ti = np.exp(rng.uniform(np.log(0.1), np.log(10.0), count))
    vi = rng.uniform(-3.0, 3.0, count)
    ve = rng.uniform(-10.0, 10.0, count)
    b = rng.uniform(0.5, 2.0, count)
    tau = rng.choice(np.asarray(tau_values), count)
    return n, te, ti, vi, ve, b, tau


# ---------------------------------------------------------------------------
# independent reference matrices (the pre-stage-2 formulas, written out)
# ---------------------------------------------------------------------------

def _legacy_curvature_matrix(n, te, ti, b, tau):
    shape = np.broadcast(n, te, ti, b, tau).shape
    M = np.zeros(shape + (4, 4))
    M[..., 0, 0] = 2 * te
    M[..., 0, 1] = 2 * n
    M[..., 0, 2] = 2 * n * tau
    M[..., 1, 0] = 4 * te * te / (3 * n)
    M[..., 1, 1] = 14 * te / 3
    M[..., 1, 2] = 4 * tau * te / 3
    M[..., 2, 0] = 4 * ti * te / (3 * n)
    M[..., 2, 1] = 4 * ti / 3
    M[..., 2, 2] = -2 * tau * ti
    M[..., 3, 0] = 2 * b * b * (te + tau * ti) / n
    M[..., 3, 1] = 2 * b * b
    M[..., 3, 2] = 2 * tau * b * b
    return M


def _legacy_parallel_matrix(n, te, ti, vi, ve, tau, mu):
    shape = np.broadcast(n, te, ti, vi, ve, tau).shape
    dv = vi - ve
    A = np.zeros(shape + (5, 5))
    A[..., 0, 0] = ve
    A[..., 0, 4] = n
    A[..., 1, 0] = -1.42 * te * dv / (3 * n)
    A[..., 1, 1] = ve
    A[..., 1, 3] = -1.42 * te / 3
    A[..., 1, 4] = 3.42 * te / 3
    A[..., 2, 0] = -2 * ti * dv / (3 * n)
    A[..., 2, 2] = vi
    A[..., 2, 4] = 2 * ti / 3
    A[..., 3, 0] = (te + tau * ti) / n
    A[..., 3, 1] = 1.0
    A[..., 3, 2] = tau
    A[..., 3, 3] = vi
    A[..., 4, 0] = mu * te / n
    A[..., 4, 1] = 1.71 * mu
    A[..., 4, 2] = mu * tau
    A[..., 4, 4] = ve
    return A


# ---------------------------------------------------------------------------
# matrix entries
# ---------------------------------------------------------------------------

def test_curvature_matrix_entries_both_selectors() -> None:
    n, te, ti, _, _, b, tau = _states()
    legacy_ref = _legacy_curvature_matrix(n, te, ti, b, tau)
    default = np.asarray(curvature_principal_matrix(n, te, ti, b, tau))
    legacy = np.asarray(curvature_principal_matrix(n, te, ti, b, tau, psi=LEGACY))
    pi = np.asarray(curvature_principal_matrix(n, te, ti, b, tau, psi=PI))
    # The default is the legacy form (frozen campaign oracles import it); bitwise.
    np.testing.assert_array_equal(default, legacy)
    np.testing.assert_allclose(legacy, legacy_ref, rtol=1e-14, atol=0)
    # psi = phi + tau n Ti: d psi/d n = tau Ti, d psi/d Ti = tau n (instead of 0, tau).
    c = np.stack((2 * n, 4 * te / 3, 4 * ti / 3), axis=-1)
    expected = legacy_ref.copy()
    expected[..., 0:3, 0] += c * (tau * ti)[..., None]
    expected[..., 0:3, 2] += c * (tau * (n - 1.0))[..., None]
    np.testing.assert_allclose(pi, expected, rtol=1e-13, atol=0)
    # Written directly: column 2 is the old diamagnetic part plus c * dpsi/dTi.
    np.testing.assert_allclose(pi[..., 0, 2], 2 * n * tau * n, rtol=1e-13)
    np.testing.assert_allclose(pi[..., 1, 2], 4 * tau * te * n / 3, rtol=1e-13)
    np.testing.assert_allclose(
        pi[..., 2, 2], -2 * tau * ti + (4 * ti / 3) * tau * (n - 1.0), rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(pi[..., 0, 0], 2 * te + 2 * n * tau * ti, rtol=1e-13)
    # The omega row is unchanged.
    np.testing.assert_array_equal(pi[..., 3, :], legacy[..., 3, :])
    # Columns 1 and 3 are unchanged.
    np.testing.assert_array_equal(pi[..., :, 1], legacy[..., :, 1])
    np.testing.assert_array_equal(pi[..., :, 3], legacy[..., :, 3])
    # At n = 1 and Ti = 0 the two forms coincide on the thermodynamic rows.
    one = np.ones(4)
    zero = np.zeros(4)
    np.testing.assert_allclose(
        np.asarray(curvature_principal_matrix(one, one, zero, one, one, psi=PI)),
        np.asarray(curvature_principal_matrix(one, one, zero, one, one, psi=LEGACY)),
        rtol=1e-14,
    )


def test_curvature_psi_threads_through_wrappers_and_validates() -> None:
    state = np.array([[1.3, 0.8, 1.2, 0.0], [0.7, 1.1, 0.9, 0.2]])
    b, tau = 1.4, 0.8
    for psi in (LEGACY, PI):
        np.testing.assert_array_equal(
            np.asarray(curvature_strict_principal_matrix(state, b, tau, psi=psi)),
            np.asarray(curvature_principal_matrix(*state[..., :3].T, b, tau, psi=psi)),
        )
        np.testing.assert_array_equal(
            np.asarray(curvature_flux_jacobian(state, b, tau, normal=-0.5, psi=psi)),
            0.5 * np.asarray(curvature_strict_principal_matrix(state, b, tau, psi=psi)),
        )
    default = curvature_flux_jacobian(state, b, tau, normal=1.0)
    np.testing.assert_array_equal(
        np.asarray(default),
        np.asarray(curvature_flux_jacobian(state, b, tau, normal=1.0, psi=LEGACY)),
    )
    assert not np.array_equal(
        np.asarray(default),
        np.asarray(curvature_flux_jacobian(state, b, tau, normal=1.0, psi=PI)),
    )
    left, right, face = state[0] * 0.9, state[0] * 1.1, state[0]
    legacy = curvature_face_linearized_fluctuations(left, right, face, b, tau)
    explicit = curvature_face_linearized_fluctuations(left, right, face, b, tau, psi=LEGACY)
    pi = curvature_face_linearized_fluctuations(left, right, face, b, tau, psi=PI)
    for a, c in zip(legacy, explicit):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(c))
    assert not np.allclose(np.asarray(legacy[0]), np.asarray(pi[0]))
    with pytest.raises(ValueError, match="psi"):
        curvature_principal_matrix(1.0, 1.0, 1.0, 1.0, 1.0, psi="phi_plus_tau_te")


def test_parallel_matrix_entries_both_selectors() -> None:
    n, te, ti, vi, ve, _, tau = _states()
    legacy_ref = _legacy_parallel_matrix(n, te, ti, vi, ve, tau, MU)
    default = np.asarray(parallel_production_principal_matrix(n, te, ti, vi, ve, tau, MU))
    legacy = np.asarray(
        parallel_production_principal_matrix(n, te, ti, vi, ve, tau, MU, psi=LEGACY)
    )
    pi = np.asarray(
        parallel_production_principal_matrix(n, te, ti, vi, ve, tau, MU, psi=PI)
    )
    np.testing.assert_array_equal(default, legacy)
    np.testing.assert_allclose(legacy, legacy_ref, rtol=1e-14, atol=0)
    # Legacy: A[4,2] = mu tau.  p_i: A[4,0] = mu Te/n + mu tau Ti, A[4,2] = mu tau n.
    np.testing.assert_allclose(legacy[..., 4, 2], MU * tau, rtol=1e-14)
    np.testing.assert_allclose(pi[..., 4, 0], MU * te / n + MU * tau * ti, rtol=1e-13)
    np.testing.assert_allclose(pi[..., 4, 2], MU * tau * n, rtol=1e-14)
    changed = np.zeros((5, 5), dtype=bool)
    changed[4, 0] = changed[4, 2] = True
    other = np.where(changed, 0.0, 1.0)
    np.testing.assert_array_equal(pi * other, legacy * other)
    # Ion row keeps tau (the ion momentum equation is not split).
    np.testing.assert_array_equal(pi[..., 3, :], legacy[..., 3, :])
    state = np.stack((n, te, ti, vi, ve), axis=-1)
    for psi in (LEGACY, PI):
        np.testing.assert_array_equal(
            np.asarray(parallel_matrix_from_state(state, tau, MU, psi=psi)),
            np.asarray(
                parallel_production_principal_matrix(n, te, ti, vi, ve, tau, MU, psi=psi)
            ),
        )
    np.testing.assert_array_equal(
        np.asarray(parallel_matrix_from_state(state, tau, MU)), legacy
    )
    with pytest.raises(ValueError, match="psi"):
        parallel_production_principal_matrix(1.0, 1.0, 1.0, 0.0, 0.0, 1.0, MU, psi="x")


# ---------------------------------------------------------------------------
# each split pair cancels in the continuum, for both selectors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("psi", [LEGACY, PI])
def test_curvature_material_plus_remainder_is_the_unsplit_term(psi) -> None:
    rng = np.random.default_rng(3)
    raw = 12
    n, te, ti, _, _, b, tau_a = _states(raw, seed=4)
    tau = 0.8
    values = np.stack((n, te, ti, rng.normal(size=raw), rng.normal(size=raw)), axis=-1)[:, None, :]
    gradients = rng.normal(size=(raw, 1, 5, 3))
    k = rng.normal(size=(raw, 3))
    weight = np.ones((raw, 1))
    material, remainder = p06_midpoint_material_remainder(
        values, gradients, b[:, None], k[:, None, :], weight, tau, psi=psi
    )
    material, remainder = np.asarray(material), np.asarray(remainder)
    curv = np.einsum("rd,rfd->rf", k, gradients[:, 0])  # C(n), C(Te), C(Ti), C(omega), C(phi)
    cn, cte, cti, com, cphi = curv.T
    unsplit = np.stack(
        (
            (2 * te * cn + 2 * n * cte - 2 * n * cphi) / b,
            (4 * te * te / (3 * n) * cn + 14 * te / 3 * cte - 4 * te / 3 * cphi) / b,
            (4 * ti * te / (3 * n) * cn + 4 * ti / 3 * cte - 10 * tau * ti / 3 * cti
             - 4 * ti / 3 * cphi) / b,
            (2 * b * b * (te + tau * ti) / n * cn + 2 * b * b * cte
             + 2 * tau * b * b * cti) / b,
        ),
        axis=-1,
    )
    # The omega row has no remainder; rows n, Te, Ti: material + remainder is the same
    # unsplit expression whichever psi is used (product rule C(n Ti) = Ti C(n) + n C(Ti)).
    np.testing.assert_allclose(material + remainder, unsplit, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(remainder[:, 3], 0.0)


def test_curvature_split_pieces_differ_but_total_agrees_between_selectors() -> None:
    rng = np.random.default_rng(5)
    raw = 8
    n, te, ti, _, _, b, _ = _states(raw, seed=6)
    values = np.stack((n, te, ti, np.zeros(raw), rng.normal(size=raw)), axis=-1)[:, None, :]
    gradients = rng.normal(size=(raw, 1, 5, 3))
    k = rng.normal(size=(raw, 3))
    args = (values, gradients, b[:, None], k[:, None, :], np.ones((raw, 1)), 1.0)
    m_old, r_old = (np.asarray(x) for x in p06_midpoint_material_remainder(*args, psi=LEGACY))
    m_default, r_default = (np.asarray(x) for x in p06_midpoint_material_remainder(*args))
    m_new, r_new = (np.asarray(x) for x in p06_midpoint_material_remainder(*args, psi=PI))
    np.testing.assert_array_equal(m_old, m_default)
    np.testing.assert_array_equal(r_old, r_default)
    assert np.max(np.abs(m_new[:, :3] - m_old[:, :3])) > 1e-3
    assert np.max(np.abs(r_new[:, :3] - r_old[:, :3])) > 1e-3
    np.testing.assert_allclose(m_new + r_new, m_old + r_old, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("psi", [LEGACY, PI])
def test_parallel_material_plus_composite_force_is_the_unsplit_term(psi) -> None:
    rng = np.random.default_rng(7)
    n, te, ti, vi, ve, _, tau = _states(32, seed=8)
    grad = rng.normal(size=(32, 5))  # G(n), G(Te), G(Ti), G(Vi), G(Ve)
    gphi = rng.normal(size=32)
    A = np.asarray(parallel_production_principal_matrix(n, te, ti, vi, ve, tau, MU, psi=psi))
    material_ve = -np.einsum("rj,rj->r", A[:, 4, :], grad)
    gq = ti * grad[:, 0] + n * grad[:, 2] if psi == PI else grad[:, 2]  # G(n Ti) or G(Ti)
    force = MU * (gphi + tau * gq)
    unsplit = -ve * grad[:, 4] - MU * te / n * grad[:, 0] - 1.71 * MU * grad[:, 1] + MU * gphi
    np.testing.assert_allclose(material_ve + force, unsplit, rtol=1e-11, atol=1e-9)


# ---------------------------------------------------------------------------
# Q-path force split (material -mu tau G(q) and +mu tau G(q) share one G(q))
# ---------------------------------------------------------------------------

def test_q_material_slots_pi_split_cancels_and_defaults_to_legacy() -> None:
    rng = np.random.default_rng(9)
    raw = 5
    # (raw, 5 eta slots, 5 fields n, Te, Ti, Vi, Ve)
    stencil = np.concatenate(
        (
            np.exp(rng.uniform(-0.3, 0.3, (raw, 5, 3))),
            rng.uniform(-0.5, 0.5, (raw, 5, 2)),
        ),
        axis=-1,
    )
    phi = rng.normal(size=(raw, 3))
    L = rng.uniform(0.2, 0.4, (raw, 3))
    args = (stencil, phi, L, 1.1, 0.01)
    kwargs = dict(tau=0.9, mu=MU)
    old = material_from_slots(*args, **kwargs)
    default = material_from_slots(*args, **kwargs, psi=LEGACY)
    new = material_from_slots(*args, **kwargs, psi=PI)
    np.testing.assert_array_equal(np.asarray(old.material), np.asarray(default.material))
    np.testing.assert_array_equal(
        np.asarray(old.generalized_force), np.asarray(default.generalized_force)
    )
    # The pair material + force is independent of the split variable.
    np.testing.assert_allclose(
        np.asarray(new.centered), np.asarray(old.centered), rtol=1e-12, atol=1e-9
    )
    assert np.max(np.abs(np.asarray(new.material)[..., 4] - np.asarray(old.material)[..., 4])) > 1e-3
    np.testing.assert_array_equal(
        np.asarray(new.material)[..., :4], np.asarray(old.material)[..., :4]
    )


# ---------------------------------------------------------------------------
# closed forms derived from the matrices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("psi", [LEGACY, PI])
def test_p06_closed_form_absolute_action_matches_lapack4(psi) -> None:
    count = 400
    n, te, ti, _, _, b, tau = _states(count, seed=11, tau_values=(0.0, 0.1, 1.0, 5.0))
    ti = np.where(np.arange(count) % 25 == 0, 1e-6 * te, ti)  # r -> 0 (exactly 0: lapack4 takes its fallback)
    rng = np.random.default_rng(12)
    jump = rng.normal(size=(count, 4))
    scale = -rng.uniform(0.2, 2.0, count) * np.where(rng.random(count) < 0.5, 1.0, -1.0)
    matrix = jnp.asarray(scale)[:, None, None] * curvature_principal_matrix(
        n, te, ti, b, tau, psi=psi
    )
    closed, invalid = _absolute_action_closed_form(
        n, te, ti, b, tau, scale, matrix, jump, 1e-12, psi=psi
    )
    # |scale| * |P| is |matrix|: the lapack4 evaluation of the same matrix.
    reference = _absolute_primal(matrix, jnp.asarray(jump))[0]
    assert not np.any(np.asarray(invalid))
    error = np.max(np.abs(np.asarray(closed) - np.asarray(reference)), axis=-1)
    norm = np.max(np.abs(np.asarray(reference)), axis=-1)
    assert np.max(error / np.maximum(norm, 1e-300)) < 1e-11


def test_p06_closed_form_default_is_the_legacy_matrix() -> None:
    n, te, ti, _, _, b, tau = _states(40, seed=13)
    rng = np.random.default_rng(14)
    jump = rng.normal(size=(40, 4))
    matrix = -curvature_principal_matrix(n, te, ti, b, tau)
    default = _absolute_action_closed_form(n, te, ti, b, tau, -np.ones(40), matrix, jump, 1e-12)
    explicit = _absolute_action_closed_form(
        n, te, ti, b, tau, -np.ones(40), matrix, jump, 1e-12, psi=LEGACY
    )
    np.testing.assert_array_equal(np.asarray(default[0]), np.asarray(explicit[0]))


def test_p06_closed_form_pi_is_differentiable() -> None:
    b, tau = 1.2, 0.7
    jump = jnp.asarray([0.3, -0.2, 0.5, 0.1])

    def closed(x):
        n, te, ti = x
        matrix = -curvature_principal_matrix(n, te, ti, b, tau, psi=PI)
        return _absolute_action_closed_form(
            n, te, ti, b, tau, -1.0, matrix, jump, 1e-12, psi=PI
        )[0]

    def lapack(x):
        n, te, ti = x
        return _absolute_matrix_action(-curvature_principal_matrix(n, te, ti, b, tau, psi=PI), jump)

    x = jnp.asarray([1.3, 0.9, 1.1])
    # lapack4's custom jvp is the Frechet derivative of the qualified action.
    np.testing.assert_allclose(
        np.asarray(jax.jacfwd(closed)(x)), np.asarray(jax.jacfwd(lapack)(x)), rtol=1e-8, atol=1e-9
    )


@pytest.mark.parametrize("psi", [LEGACY, PI])
@pytest.mark.parametrize("tau", [0.0, 0.1, 1.0])
def test_q_polynomial_characteristic_decomposition_reconstructs_the_matrix(psi, tau) -> None:
    n, te, ti, vi, ve, _, _ = _states(300, seed=21)
    state = jnp.asarray(np.stack((n, te, ti, vi, ve), axis=-1))
    values, right, left, valid = polynomial_basis(state, tau, MU, psi=psi)
    A = parallel_matrix_from_state(state, tau, MU, psi=psi)
    valid = np.asarray(valid)
    assert valid.mean() > 0.9
    reconstructed = jnp.einsum("...ik,...k,...kj->...ij", right, values, left)
    error = jnp.linalg.norm(reconstructed - A, axis=(-2, -1)) / jnp.linalg.norm(A, axis=(-2, -1))
    assert np.max(np.asarray(error)[valid]) < 1e-8
    identity = jnp.einsum("...ik,...kj->...ij", left, right)
    np.testing.assert_allclose(
        np.asarray(identity)[valid], np.broadcast_to(np.eye(5), identity.shape)[valid], atol=1e-8
    )
    # The right-going / left-going actions equal the generic eigensolver split.
    plus, minus, _, _, split_valid = polynomial_characteristic_split(state, tau, MU, 1.0, psi=psi)
    matrix_plus, matrix_minus, _, _, generic_valid = parallel_characteristic_split(
        A, normal=1.0
    )
    both = np.asarray(split_valid) & np.asarray(generic_valid)
    assert both.mean() > 0.85
    scale = np.asarray(jnp.linalg.norm(A, axis=(-2, -1)))
    for polynomial, generic in ((plus, matrix_plus), (minus, matrix_minus)):
        difference = np.asarray(jnp.linalg.norm(polynomial - generic, axis=(-2, -1))) / scale
        assert np.max(difference[both]) < 1e-6


def test_q_polynomial_default_psi_is_legacy() -> None:
    n, te, ti, vi, ve, _, _ = _states(20, seed=22)
    state = jnp.asarray(np.stack((n, te, ti, vi, ve), axis=-1))
    default = polynomial_basis(state, 0.7, MU)
    legacy = polynomial_basis(state, 0.7, MU, psi=LEGACY)
    for a, c in zip(default, legacy):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(c))
