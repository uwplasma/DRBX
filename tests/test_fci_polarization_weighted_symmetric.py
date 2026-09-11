"""Integration contracts for the weighted-adjoint polarization operator."""

from __future__ import annotations

from dataclasses import replace
import os
from types import SimpleNamespace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from drbx.native import fci_drb_EB_rhs as rhs_module  # noqa: E402
from drbx.native.fci_boundaries import BC_NEUMANN  # noqa: E402
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_operators import LocalPerpLaplacianInverseSolver  # noqa: E402
from drbx.native.fci_sharding import assemble_local_fci_geometry  # noqa: E402
from fci_drb_eb_test_helpers import (  # noqa: E402
    _build_rhs,
    _context_and_sharded_inputs,
)


def test_rhs_routes_one_selected_operator_to_phi_and_ti() -> None:
    source = (
        _TESTS.parent / "src/drbx/native/fci_drb_EB_rhs.py"
    ).read_text()
    assert 'polarization_operator_form: str = "conservative"' in source
    assert '"support-paired"' in source
    assert 'physical_wall_model_name == "simplified-gbs-mpe"' in source
    assert "solve_augmented_neumann(" in source
    assert "positive_ti_action = self._positive_polarization_action(" in source
    assert "phi_action = self._positive_polarization_action(" in source
    assert "ti_action = self._positive_polarization_action(" in source
    assert "derived_vorticity_face_bc = self._derived_vorticity_face_bc_from_polarization(" in source
    assert "vorticity_halo = self._prepare_scalar_halo(" in source
    assert "state_halo = state_halo_without_phi.replace(" in source


def test_simplified_gbs_mpe_routes_phi_through_augmented_neumann(monkeypatch) -> None:
    with jax.disable_jit(False):
        context, mesh, local, partition, fields, cell_fields = (
            _context_and_sharded_inputs()
        )

    captured: dict[str, object] = {}

    def fake_solve(
        self,
        rhs_owned,
        *,
        gauge_weights_owned,
        gauge_target,
        gauge_affine_offset=0.0,
        guess_owned=None,
        phi_guess_owned=None,
        return_diagnostics=False,
    ):
        del rhs_owned, guess_owned, phi_guess_owned, return_diagnostics
        captured["regularization_epsilon"] = float(self.config.regularization_epsilon)
        captured["gauge_weights_sum"] = float(jnp.sum(gauge_weights_owned))
        captured["gauge_target"] = float(gauge_target)
        captured["gauge_affine_offset"] = float(gauge_affine_offset)
        return jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)

    monkeypatch.setattr(
        LocalPerpLaplacianInverseSolver,
        "solve_augmented_neumann",
        fake_solve,
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells):
        geometry = assemble_local_fci_geometry(local, cells)
        model = replace(
            _build_rhs(context, local, geometry),
            physical_wall_model_name="simplified-gbs-mpe",
            polarization_operator_form="weighted-symmetric",
        )
        model = replace(
            model,
            gmres_config=replace(model.gmres_config, regularization_epsilon=0.0),
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        face_bc = model._face_bcs(state)
        state_halo = model._prepare_state_halo(state, face_bc)
        _ = model._reconstruct_phi_from_prepared(state, state_halo, face_bc)
        kinds_ok = jnp.logical_and(
            jnp.all(face_bc.phi.kind_x[face_bc.phi.mask_x] == BC_NEUMANN),
            jnp.logical_and(
                jnp.all(face_bc.phi.kind_y[face_bc.phi.mask_y] == BC_NEUMANN),
                jnp.all(face_bc.phi.kind_z[face_bc.phi.mask_z] == BC_NEUMANN),
            ),
        )
        return kinds_ok.astype(jnp.float64)

    mapped = shard_map(
        kernel,
        mesh=mesh,
        in_specs=(*((partition,) * 7), partition),
        out_specs=P(),
    )
    kinds_ok = float(np.asarray(mapped(*fields, cell_fields)))
    assert kinds_ok == 1.0
    assert captured["regularization_epsilon"] == 0.0
    assert np.isclose(float(captured["gauge_weights_sum"]), 1.0)
    assert np.isfinite(float(captured["gauge_target"]))
    assert np.isfinite(float(captured["gauge_affine_offset"]))


def test_simplified_gbs_mpe_vorticity_helper_uses_polarization_action(monkeypatch) -> None:
    # Keep fixture construction compiled even when the targeted test is run
    # with JAX_DISABLE_JIT=1; the mapped test kernel itself remains isolated.
    with jax.disable_jit(False):
        context, mesh, local, partition, fields, cell_fields = (
            _context_and_sharded_inputs()
        )
    captured: dict[str, list[tuple[object, object]]] = {"calls": []}

    def fake_apply(
        self,
        values_owned,
        *,
        face_bc=None,
        control_volume_boundary_bc=None,
        project_mean_zero=False,
    ):
        del self, control_volume_boundary_bc, project_mean_zero
        captured["calls"].append((values_owned, face_bc))
        return jnp.asarray(values_owned, dtype=jnp.float64)

    monkeypatch.setattr(
        LocalPerpLaplacianInverseSolver,
        "apply_positive_operator",
        fake_apply,
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells):
        geometry = assemble_local_fci_geometry(local, cells)
        model = replace(
            _build_rhs(context, local, geometry),
            physical_wall_model_name="simplified-gbs-mpe",
            polarization_operator_form="weighted-symmetric",
        )
        model = replace(
            model,
            gmres_config=replace(model.gmres_config, regularization_epsilon=0.0),
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        face_bc = model._face_bcs(state)
        captured["calls"] = []
        baseline_halo = model._prepare_state_halo(state, face_bc)
        derived_bc = model._derived_vorticity_face_bc_from_polarization(
            state,
            state.phi,
            face_bc,
        )
        vorticity_halo = model._prepare_scalar_halo(
            state.vorticity,
            derived_bc,
        )
        owned = model.domain.layout.owned_slices_cell
        owner_ok = jnp.all(vorticity_halo[owned] == baseline_halo.vorticity[owned])
        balance = model.polarization_balance_terms(
            state,
            phi_owned=state.phi,
        )
        omega_pol = model._vorticity_from_polarization(
            state.phi,
            state.Ti,
            face_bc.phi,
            face_bc.Ti,
        )
        identity_ok = jnp.all(
            jnp.isclose(
                omega_pol,
                -(balance[0] - balance[1]),
                rtol=0.0,
                atol=0.0,
            )
        )
        final_calls = captured["calls"][-2:]
        raw_fields_ok = (
            jnp.allclose(final_calls[0][0], state.phi, rtol=0.0, atol=0.0)
            & jnp.allclose(final_calls[1][0], state.Ti, rtol=0.0, atol=0.0)
        )
        bc_routing_ok = (
            final_calls[0][1] is face_bc.phi
            and final_calls[1][1] is face_bc.Ti
        )
        routing_ok = jax.lax.pmin(
            (raw_fields_ok & bc_routing_ok).astype(jnp.float64),
            axis_name=("x", "y", "z"),
        )
        owner_ok = jax.lax.pmin(owner_ok.astype(jnp.float64), axis_name=("x", "y", "z"))
        identity_ok = jax.lax.pmin(identity_ok.astype(jnp.float64), axis_name=("x", "y", "z"))
        return (
            owner_ok,
            identity_ok,
            routing_ok,
        )

    mapped = shard_map(
        kernel,
        mesh=mesh,
        in_specs=(*((partition,) * 7), partition),
        out_specs=(P(), P(), P()),
    )
    owner_ok, identity_ok, routing_ok = np.asarray(mapped(*fields, cell_fields))
    np.testing.assert_allclose(
        owner_ok,
        1.0,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        identity_ok,
        1.0,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(routing_ok, 1.0, rtol=0.0, atol=0.0)


def test_simplified_gbs_mpe_phi_gauge_ignores_inactive_nan_traces(monkeypatch) -> None:
    context, mesh, local, partition, fields, cell_fields = (
        _context_and_sharded_inputs()
    )

    def fake_trace_builder(_phi_halo, geometry, _domain, face_bc):
        del geometry
        return SimpleNamespace(
            value_x=jnp.where(
                face_bc.mask_x,
                jnp.ones_like(face_bc.value_x, dtype=jnp.float64),
                jnp.nan,
            ),
            value_y=jnp.where(
                face_bc.mask_y,
                2.0 * jnp.ones_like(face_bc.value_y, dtype=jnp.float64),
                jnp.nan,
            ),
            value_z=jnp.where(
                face_bc.mask_z,
                3.0 * jnp.ones_like(face_bc.value_z, dtype=jnp.float64),
                jnp.nan,
            ),
            mask_x=face_bc.mask_x,
            mask_y=face_bc.mask_y,
            mask_z=face_bc.mask_z,
        )

    monkeypatch.setattr(
        rhs_module,
        "build_local_boundary_face_trace_from_halo",
        fake_trace_builder,
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells):
        geometry = assemble_local_fci_geometry(local, cells)
        model = replace(
            _build_rhs(context, local, geometry),
            physical_wall_model_name="simplified-gbs-mpe",
            polarization_operator_form="weighted-symmetric",
        )
        model = replace(
            model,
            gmres_config=replace(model.gmres_config, regularization_epsilon=0.0),
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        face_bc = model._face_bcs(state)
        gauge_weights, gauge_affine_offset, gauge_target = (
            model._simplified_gbs_mpe_phi_gauge_data(state, face_bc)
        )
        finite_ok = (
            jnp.isfinite(jnp.sum(gauge_weights))
            & jnp.isfinite(gauge_affine_offset)
            & jnp.isfinite(gauge_target)
        )
        return finite_ok.astype(jnp.float64)

    mapped = shard_map(
        kernel,
        mesh=mesh,
        in_specs=(*((partition,) * 7), partition),
        out_specs=P(),
    )
    finite_ok = float(np.asarray(mapped(*fields, cell_fields)))
    np.testing.assert_allclose(finite_ok, 1.0, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    os.environ.get("DRBX_RUN_SLOW_POLARIZATION_TESTS") != "1",
    reason="full weighted polarization RHS compile is an opt-in slow gate",
)
def test_weighted_symmetric_form_is_shared_by_phi_and_ti_balance() -> None:
    context, mesh, local, partition, fields, cell_fields = (
        _context_and_sharded_inputs()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells):
        geometry = assemble_local_fci_geometry(local, cells)
        model = replace(
            _build_rhs(context, local, geometry),
            polarization_operator_form="weighted-symmetric",
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        reconstructed, info = model.reconstruct_phi(
            state,
            return_diagnostics=True,
        )
        terms = model.polarization_balance_terms(
            state,
            phi_owned=reconstructed,
        )
        face_bc = model._face_bcs(state)
        solver = model._polarization_solver(face_bc.phi)
        phi_action = model._positive_polarization_action(
            solver,
            reconstructed,
            face_bc.phi,
        )
        ti_action = model._positive_polarization_action(
            solver,
            state.Ti,
            face_bc.Ti,
        )
        residual = model.polarization_residual(
            state,
            phi_owned=reconstructed,
        )
        return (
            terms,
            phi_action,
            ti_action,
            residual,
            info.failed,
        )

    mapped = shard_map(
        kernel,
        mesh=mesh,
        in_specs=(*((partition,) * 7), partition),
        out_specs=(
            P(None, "x", "y", "z"),
            partition,
            partition,
            partition,
            P(),
        ),
        check_rep=False,
    )
    terms, phi_action, ti_action, residual, failed = mapped(
        *fields,
        cell_fields,
    )

    np.testing.assert_allclose(
        np.asarray(terms[0]),
        np.asarray(phi_action),
        rtol=0.0,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        np.asarray(terms[1]),
        -float(context.parameters.tau) * np.asarray(ti_action),
        rtol=0.0,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        np.asarray(residual),
        np.asarray(terms[0] - terms[1] - terms[2]),
        rtol=0.0,
        atol=2.0e-11,
    )
    assert not bool(np.asarray(failed))
    assert np.all(np.isfinite(np.asarray(residual)))
