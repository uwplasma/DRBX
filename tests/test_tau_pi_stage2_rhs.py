"""tau p_i stage 2 at the RHS level: periodic slab around the full ``LocalFciDrbEBRhs``.

* the legacy selector (``polarization_variable="phi_plus_tau_ti"``) is bitwise the pre-stage-2
  RHS: the two principal matrices are replaced by frozen copies of the old formulas and the
  complete ``evaluate_stage`` term fields must not move by one bit;
* the default selector re-splits both pairs in lockstep.  About the uniform background (an
  exact steady state, so ``jax.jvp`` is the exact linearization) the curvature material +
  remainder total (upwinding switched off, which is not a continuum term) and the parallel
  composite force equal the unsplit expressions to round-off.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

import jax.numpy as jnp

import drbx.native.fci_curvature_production_flux as curvature_flux
import drbx.native.fci_parallel_production_flux as parallel_flux
from drbx.native.fci_curvature_production_flux import STATE_SIZE as _CURVATURE_STATE_SIZE

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from tau_pi_slab_helpers import (  # noqa: E402
    SlabModel,
    make_parameters,
    term_slot,
)

LEGACY = "phi_plus_tau_ti"
PI = "phi_plus_tau_pi"
SHAPE = (4, 6, 6)
LENGTHS = (0.4, 2.0 * np.pi, 2.0 * np.pi)
MU = 1836.0


def _waves(seed, amplitude):
    """Seven smooth zero-mean fields (n, Te, Ti, Vi, Ve, omega, phi)."""
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*(np.arange(n) / n for n in SHAPE), indexing="ij")
    fields = []
    for _ in range(7):
        phase = rng.uniform(0, 2 * np.pi, 3)
        fields.append(
            amplitude
            * (
                np.sin(2 * np.pi * x + phase[0]) * np.cos(2 * np.pi * y + phase[1])
                + 0.5 * np.cos(2 * np.pi * z + phase[2])
            )
        )
    return fields


def _smooth_state(model, amplitude=0.1, seed=0):
    """A positive, nonuniform state about the uniform background."""
    f = _waves(seed, amplitude)
    base = model.uniform()
    return [
        base[0] * np.exp(f[0]), base[1] * np.exp(f[1]), base[2] * np.exp(f[2]),
        3.0 * f[3], 3.0 * f[4], f[5], f[6],
    ]


# frozen pre-stage-2 matrices (the old function bodies, ignoring ``psi``) -----------------

def _old_curvature_principal_matrix(density, Te, Ti, bmag, tau, *, k_perp_squared=None, psi="ignored"):
    n, te, ti, b, tau_value = tuple(
        jnp.asarray(value, dtype=jnp.float64) for value in (density, Te, Ti, bmag, tau)
    )
    shape = jnp.broadcast_shapes(n.shape, te.shape, ti.shape, b.shape, tau_value.shape)
    n = jnp.broadcast_to(n, shape)
    te = jnp.broadcast_to(te, shape)
    ti = jnp.broadcast_to(ti, shape)
    b = jnp.broadcast_to(b, shape)
    tau_value = jnp.broadcast_to(tau_value, shape)
    n_safe = jnp.maximum(n, 1.0e-30)
    matrix = jnp.zeros(shape + (_CURVATURE_STATE_SIZE, _CURVATURE_STATE_SIZE), dtype=jnp.float64)
    matrix = matrix.at[..., 0, 0].set(2.0 * te)
    matrix = matrix.at[..., 0, 1].set(2.0 * n)
    matrix = matrix.at[..., 0, 2].set(2.0 * n * tau_value)
    matrix = matrix.at[..., 1, 0].set(4.0 * te * te / (3.0 * n_safe))
    matrix = matrix.at[..., 1, 1].set(14.0 * te / 3.0)
    matrix = matrix.at[..., 1, 2].set(4.0 * tau_value * te / 3.0)
    matrix = matrix.at[..., 2, 0].set(4.0 * ti * te / (3.0 * n_safe))
    matrix = matrix.at[..., 2, 1].set(4.0 * ti / 3.0)
    matrix = matrix.at[..., 2, 2].set(-2.0 * tau_value * ti)
    matrix = matrix.at[..., 3, 0].set(2.0 * b * b * (te + tau_value * ti) / n_safe)
    matrix = matrix.at[..., 3, 1].set(2.0 * b * b)
    matrix = matrix.at[..., 3, 2].set(2.0 * tau_value * b * b)
    return matrix


def _old_parallel_principal_matrix(density, Te, Ti, Vi, Ve, tau, mu, *, psi="ignored"):
    density, Te, Ti, Vi, Ve, tau, mu = [
        jnp.asarray(x, dtype=jnp.float64) for x in (density, Te, Ti, Vi, Ve, tau, mu)
    ]
    density, Te, Ti, Vi, Ve, tau, mu = jnp.broadcast_arrays(density, Te, Ti, Vi, Ve, tau, mu)
    n_safe = jnp.maximum(density, 1.0e-30)
    dV = Vi - Ve
    matrix = jnp.zeros(density.shape + (5, 5), dtype=jnp.float64)
    matrix = matrix.at[..., 0, 0].set(Ve)
    matrix = matrix.at[..., 0, 4].set(density)
    matrix = matrix.at[..., 1, 0].set(-1.42 * Te * dV / (3.0 * n_safe))
    matrix = matrix.at[..., 1, 1].set(Ve)
    matrix = matrix.at[..., 1, 3].set(-1.42 * Te / 3.0)
    matrix = matrix.at[..., 1, 4].set(3.42 * Te / 3.0)
    matrix = matrix.at[..., 2, 0].set(-2.0 * Ti * dV / (3.0 * n_safe))
    matrix = matrix.at[..., 2, 2].set(Vi)
    matrix = matrix.at[..., 2, 4].set(2.0 * Ti / 3.0)
    matrix = matrix.at[..., 3, 0].set((Te + tau * Ti) / n_safe)
    matrix = matrix.at[..., 3, 1].set(1.0)
    matrix = matrix.at[..., 3, 2].set(tau)
    matrix = matrix.at[..., 3, 3].set(Vi)
    matrix = matrix.at[..., 4, 0].set(mu * Te / n_safe)
    matrix = matrix.at[..., 4, 1].set(1.71 * mu)
    matrix = matrix.at[..., 4, 2].set(mu * tau)
    matrix = matrix.at[..., 4, 4].set(Ve)
    return matrix


def _zero_curvature_absolute_action(matrix, vector, *, return_fallback=False, **kwargs):
    zero = jnp.zeros_like(jnp.asarray(vector, dtype=jnp.float64))
    if return_fallback:
        return zero, jnp.zeros(zero.shape[:-1], dtype=bool)
    return zero


@pytest.mark.parametrize("path", ["coord", "prod"])
def test_legacy_selector_rhs_is_bitwise_the_pre_stage2_rhs(path, monkeypatch) -> None:
    parameters = make_parameters(1.0, LEGACY)
    state_model = SlabModel(SHAPE, LENGTHS, parameters, path=path)
    state = _smooth_state(state_model)
    new = state_model.terms(state)
    # Same model, matrices replaced by the frozen pre-stage-2 formulas.
    monkeypatch.setattr(curvature_flux, "curvature_principal_matrix", _old_curvature_principal_matrix)
    monkeypatch.setattr(
        parallel_flux, "parallel_production_principal_matrix", _old_parallel_principal_matrix
    )
    old = SlabModel(SHAPE, LENGTHS, parameters, path=path).terms(state)
    assert np.all(np.isfinite(new))
    np.testing.assert_array_equal(new, old)
    # The default selector is the other form: it must actually differ at nonuniform n.
    default = SlabModel(SHAPE, LENGTHS, make_parameters(1.0, PI), path=path)
    monkeypatch.undo()
    changed = default.terms(state)
    curvature_n = term_slot("density", "curvature")
    assert np.max(np.abs(changed[curvature_n] - new[curvature_n])) > 1e-8


def _y_mode_tangent(ny):
    """Seven zero-mean fields varying along y only (the direction of the test curvature)."""
    y = np.arange(ny) / ny
    amplitudes = (0.7, -0.4, 0.5, 0.0, 0.0, 0.3, 0.6)  # n, Te, Ti, Vi, Ve, omega, phi
    return [
        np.broadcast_to(a * np.sin(2 * np.pi * y + 0.4)[None, :, None], (4, ny, 4)).copy()
        for a in amplitudes
    ]


def test_curvature_split_cancels_in_the_continuum_about_the_uniform_background(monkeypatch) -> None:
    """Material + remainder total (upwinding off) is the same for both splits, up to truncation.

    The material block is a face-flux discretization and the remainder the conservative
    curvature of psi, so the pair cancels in the continuum (it is the same field-line operator
    at the discrete level only up to O(h^2)): the difference between the splits must shrink
    with refinement while each total stays O(1).  A re-split of only one half of the pair
    (continuum change) would not converge.
    """
    monkeypatch.setattr(
        curvature_flux, "curvature_characteristic_absolute_action", _zero_curvature_absolute_action
    )
    gaps, scales = [], []
    for ny in (8, 16):
        shape = (4, ny, 4)
        totals = {}
        for psi in (LEGACY, PI):
            model = SlabModel(shape, (0.4, 2.0 * np.pi, 0.4), make_parameters(1.0, psi), path="coord")
            out = model.jvp_terms(model.uniform(), _y_mode_tangent(ny))
            totals[psi] = np.stack(
                [out[term_slot(f, "curvature")] for f in ("density", "Te", "Ti")]
            )
        gaps.append(np.max(np.abs(totals[PI] - totals[LEGACY])))
        scales.append(np.max(np.abs(totals[LEGACY])))
    assert min(scales) > 1e-2
    assert gaps[1] < 0.05 * scales[1]
    assert gaps[1] < gaps[0] / 1.8, (gaps, scales)


def test_parallel_composite_force_is_the_unsplit_force_about_the_uniform_background() -> None:
    """Ve electrostatic term about ``n = Ti = 1``: ``mu G(phi + tau (dn + dTi))`` for p_i.

    The composite force ``mu G(phi + tau n Ti)`` linearizes to ``mu G(dphi + tau (Ti dn + n dTi))``
    (the unsplit expression ``mu G(dphi) + tau mu (Ti G(dn) + n G(dTi))``).  The legacy split
    ``mu G(dphi + tau dTi)`` with ``dTi -> dn + dTi`` is therefore the reference: both are the
    same linear support-paired operator, so the two agree to round-off, while for a pure
    density perturbation the legacy force vanishes and the p_i force does not.
    """
    tau = 0.7
    x, y, z = np.meshgrid(*(np.arange(n) / n for n in SHAPE), indexing="ij")
    dn = np.sin(2 * np.pi * z + 0.3) * np.cos(2 * np.pi * y) + 0.2 * np.cos(2 * np.pi * x)
    dti = np.cos(2 * np.pi * z + 1.1) + 0.3 * np.sin(2 * np.pi * y + 0.5) * np.sin(2 * np.pi * z)
    dphi = np.sin(4 * np.pi * z + 0.7) + 0.5 * np.cos(2 * np.pi * x + 0.2) * np.cos(2 * np.pi * z)
    zero = np.zeros(SHAPE)
    electrostatic = term_slot("Ve", "electrostatic")
    pi_model = SlabModel(SHAPE, LENGTHS, make_parameters(tau, PI), path="prod")
    legacy_model = SlabModel(SHAPE, LENGTHS, make_parameters(tau, LEGACY), path="prod")
    pi_force = pi_model.jvp_terms(
        pi_model.uniform(), [dn, zero, dti, zero, zero, zero, dphi]
    )[electrostatic]
    legacy_reference = legacy_model.jvp_terms(
        legacy_model.uniform(), [zero, zero, dn + dti, zero, zero, zero, dphi]
    )[electrostatic]
    scale = np.max(np.abs(legacy_reference))
    assert scale > 1.0
    np.testing.assert_allclose(pi_force, legacy_reference, rtol=0, atol=1e-11 * scale)
    # A pure density perturbation: no legacy force, a p_i force of mu*tau*G(dn).
    density_only = [dn, zero, zero, zero, zero, zero, zero]
    assert np.max(np.abs(legacy_model.jvp_terms(legacy_model.uniform(), density_only)[electrostatic])) < 1e-9 * scale
    assert np.max(np.abs(pi_model.jvp_terms(pi_model.uniform(), density_only)[electrostatic])) > 1e-3 * scale
