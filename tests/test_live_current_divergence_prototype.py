"""Static operator/assembly tests. No integrator or time-advance calls."""
from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/boundary_load_audit"
sys.path.insert(0, str(WORK))

from root_cause_small_probe import ProbeConfig, build_representative_fixture
from drbx.native.fci_drb_EB_rhs import build_local_fci_drb_eb_operator_boundary_bundle
from drbx.native.fci_boundaries import build_local_boundary_face_trace_from_halo
from drbx.native.fci_operators import (
    local_grad_parallel_op_fci_compatible_from_q,
    local_parallel_div_b_fci_from_q_op,
)


@pytest.fixture(scope="module")
def case():
    fixture = build_representative_fixture(ProbeConfig(shape=(5, 4, 3)))
    reference = fixture.model
    prototype = replace(reference, parallel_current_pairing="live-gradient-prototype")
    face = reference._face_bcs(fixture.state)
    context = reference._stencil_builder_context()
    return reference, prototype, fixture.state, face, context


def test_prototype_requires_explicit_compatible_configuration(case):
    reference, prototype, _, _, _ = case
    assert reference.parallel_current_pairing == "reference"
    for change in (
        {"parallel_current_pairing": "typo"},
        {"parallel_boundary_pairing": "current-phi"},
        {"physical_wall_model_name": "simple-conducting-sheath"},
        {"parallel_material_scheme": "legacy"},
    ):
        with pytest.raises(ValueError):
            replace(prototype, **change)


def test_matrix_free_pair_matches_captured_dense_candidate_and_differentiates(case):
    reference, model, state, face, context = case
    gradient, divergence, target = model._fci_current_phi_boundary_pair(
        face_bc=face, context=context
    )
    with np.load(WORK / "current_potential_sat_reconciliation.npz") as saved:
        dense_g, dense_d, mass = saved["G_live"], saved["D_candidate"], saved["mass"]
    shape = state.phi.shape
    rng = np.random.default_rng(847)
    potential, current, direction = [jnp.asarray(rng.normal(size=shape)) for _ in range(3)]
    g, d = jax.jit(lambda p, j: (gradient(p), divergence(j)))(potential, current)
    np.testing.assert_allclose(g.ravel(), dense_g @ potential.ravel(), atol=2e-12)
    np.testing.assert_allclose(d.ravel(), dense_d @ current.ravel(), atol=2e-12)
    np.testing.assert_array_equal(target, model.geometry.active_cell_mask_owned)
    np.testing.assert_allclose(divergence(jnp.zeros(shape)), 0.0, atol=2e-14)
    np.testing.assert_allclose(gradient(jnp.ones(shape)), 0.0, atol=2e-12)
    np.testing.assert_allclose(np.sum(mass * d.ravel()), 0.0, atol=2e-12)
    np.testing.assert_allclose(
        np.sum(mass * (potential.ravel() * d.ravel() + current.ravel() * g.ravel())),
        0.0, atol=2e-12,
    )
    _, tangent = jax.jit(lambda j, v: jax.jvp(divergence, (j,), (v,)))(current, direction)
    np.testing.assert_allclose(tangent.ravel(), dense_d @ direction.ravel(), atol=2e-12)
    batch = jax.jit(divergence)(jnp.stack((current, direction)))
    np.testing.assert_allclose(batch[0], d, atol=2e-12)
    np.testing.assert_allclose(batch[1], tangent, atol=2e-12)
    _, old, _ = reference._fci_current_phi_boundary_pair(
        face_bc=face, context=context,
        wall_endpoint_current_values=(jnp.zeros(shape), jnp.zeros(shape)),
        build_adjoint=False,
    )
    assert np.linalg.norm(np.asarray(old(current) - d)) > 1.0


def test_canonical_endpoint_lift_added_once_and_endpoint_jvp_preserved(case):
    reference, model, state, face, context = case
    zero = jnp.zeros_like(state.phi)
    rng = np.random.default_rng(121)
    current, backward, forward, dj, db, df = [
        jnp.asarray(rng.normal(size=zero.shape)) for _ in range(6)
    ]
    _, homogeneous, _ = model._fci_current_phi_boundary_pair(face_bc=face, context=context)

    def physical(j, b, f):
        _, divergence, _ = model._fci_current_phi_boundary_pair(
            face_bc=face, context=context, wall_endpoint_current_values=(b, f),
            build_adjoint=False,
        )
        return divergence(j)

    def reference_lift(b, f):
        _, divergence, _ = reference._fci_current_phi_boundary_pair(
            face_bc=face, context=context, wall_endpoint_current_values=(b, f),
            build_adjoint=False,
        )
        return divergence(zero)

    value, derivative = jax.jit(lambda j, b, f, vj, vb, vf: jax.jvp(
        physical, (j, b, f), (vj, vb, vf)
    ))(current, backward, forward, dj, db, df)
    lift = reference_lift(backward, forward)
    assert np.max(np.abs(lift)) > 0.1
    np.testing.assert_allclose(value, homogeneous(current) + lift, atol=2e-12)
    np.testing.assert_allclose(physical(zero, backward, forward), lift, atol=2e-12)
    np.testing.assert_allclose(derivative, homogeneous(dj) + reference_lift(db, df), atol=2e-12)
    epsilon = 1e-5
    secant = (physical(current + epsilon*dj, backward + epsilon*db, forward + epsilon*df)
              - physical(current - epsilon*dj, backward - epsilon*db, forward - epsilon*df)) / (2*epsilon)
    np.testing.assert_allclose(derivative, secant, atol=2e-9, rtol=2e-9)


def _parallel_outputs(model, state, face, context):
    halo = model._prepare_state_halo(state, face)
    boundary = build_local_fci_drb_eb_operator_boundary_bundle(
        halo, model.geometry, model.domain, face, tau=model.parameters.tau
    )
    parallel = model._parallel_operator_boundary(state_halo=halo, operator_boundary=boundary)
    terms = model._fci_parallel_terms(
        state_halo=halo, face_bc=face, operator_boundary=boundary,
        parallel_boundary=parallel, context=context,
    )
    wall = terms["parallel_characteristic_wall_data"]
    return tuple(terms[name] for name in (
        "vorticity_current_flux_div", "characteristic_sat_homogeneous_current_divergence",
        "characteristic_sat_affine_current_divergence", "grad_phi_plus_tau_Ti",
        "parallel_material_residual",
    )) + tuple(wall[f"{side}_wall_characteristic_current"] for side in ("backward", "forward"))


def _independent_legacy_gradient(model, value, scalar_bc, face, context):
    halo = model._prepare_fine_storage_halo(value, scalar_bc)
    trace = build_local_boundary_face_trace_from_halo(halo, model.geometry, model.domain, scalar_bc)
    q, forward, backward = model._fci_prepare_flux_q(value, trace, context)
    inverse, inv_f, inv_b = model._fci_prepare_inverse_b(face, context)
    div_b = local_parallel_div_b_fci_from_q_op(
        inverse, model.geometry, context=context,
        forward_remote_q_values=inv_f, backward_remote_q_values=inv_b,
    )
    return local_grad_parallel_op_fci_compatible_from_q(
        q, model.geometry, context=context, field_owned=value, div_b=div_b,
        forward_remote_q_values=forward, backward_remote_q_values=backward,
    )


def test_static_parallel_assembly_uses_prototype_and_retains_generalized_affine_force(case):
    reference, model, state, face, context = case
    current = state.density * (state.Vi - state.Ve)
    zero = jnp.zeros_like(current)
    # Both are static RHS component evaluations, with short-leg timestep zero.
    old = jax.jit(lambda: _parallel_outputs(reference, state, face, context))()
    live = jax.jit(lambda: _parallel_outputs(model, state, face, context))()
    total, homogeneous, lift, force, material, backward, forward = live
    gradient, divergence, _ = model._fci_current_phi_boundary_pair(face_bc=face, context=context)
    _, physical, _ = model._fci_current_phi_boundary_pair(
        face_bc=face, context=context, wall_endpoint_current_values=(backward, forward),
        build_adjoint=False,
    )
    np.testing.assert_allclose(total, physical(current), atol=2e-11)
    np.testing.assert_allclose(homogeneous, divergence(current), atol=2e-11)
    np.testing.assert_allclose(total, homogeneous + lift, atol=2e-11)
    np.testing.assert_allclose(lift, old[2], atol=2e-11)
    np.testing.assert_allclose(force, old[3], atol=2e-11)
    np.testing.assert_allclose(material, old[4], atol=2e-11)
    np.testing.assert_allclose(backward, old[5], atol=2e-11)
    np.testing.assert_allclose(forward, old[6], atol=2e-11)
    assert np.linalg.norm(np.asarray(total - old[0])) > 1e-3
    _, _, core = model._fci_support_core_pair(face_bc=face, context=context)
    affine = jnp.where(core, 0.0,
        _independent_legacy_gradient(model, zero, face.phi, face, context)
        + model.parameters.tau * _independent_legacy_gradient(model, zero, face.Ti, face, context)
    )
    assert np.max(np.abs(affine)) > 1e-4
    psi = state.phi + model.parameters.tau * state.Ti
    np.testing.assert_allclose(force, gradient(psi) + affine, atol=2e-11)
    # The full affine Green work is the retained current lift plus the
    # generalized-potential normal-data contribution; neither is discarded.
    mass = model._fci_pair_cell_mass()
    work = jnp.sum(mass * (psi * total + current * force))
    boundary_work = jnp.sum(mass * (psi * lift + current * affine))
    np.testing.assert_allclose(work, boundary_work, atol=2e-10)
