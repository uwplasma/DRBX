"""Focused contracts for the opt-in Rung-3 startup wall layer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


DRIVER = Path(__file__).resolve().parents[1] / "simulate_hsx_blob.py"


def _driver_module():
    spec = importlib.util.spec_from_file_location(
        "simulate_hsx_blob_rung3_wall_layer", DRIVER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fci_wall_layer_uses_directional_targets_and_handles_double_hits():
    driver = _driver_module()
    shape = (5, 1, 1)
    vi = np.zeros(shape)
    ve = np.zeros(shape)
    backward_target = np.zeros(shape + (2,))
    forward_target = np.zeros(shape + (2,))
    backward_target[-1, 0, 0] = (4.0, 6.0)
    forward_target[-1, 0, 0] = (-4.0, -6.0)
    selected_backward = np.zeros(shape, dtype=bool)
    selected_forward = np.zeros(shape, dtype=bool)
    selected_backward[-1, 0, 0] = True
    selected_forward[-1, 0, 0] = True
    active = np.ones(shape, dtype=bool)
    args = (
        vi, ve, backward_target, forward_target,
        selected_backward, selected_forward,
        np.ones(shape), np.ones(shape), np.ones(shape), np.ones(shape),
        active, 3,
    )
    adjusted_vi, adjusted_ve, diagnostics = driver._blend_rung3_fci_wall_layer(*args)
    # Equal stiffness gives the exact symmetric compromise for a conflicting
    # double hit, while the inward ring is smoothly moved toward it.
    np.testing.assert_allclose(adjusted_vi[-1, 0, 0], 0.0)
    np.testing.assert_allclose(adjusted_ve[-1, 0, 0], 0.0)
    assert bool(np.asarray(diagnostics["double_hit"])[-1, 0, 0])
    assert bool(np.asarray(diagnostics["double_hit_conflict"])[-1, 0, 0])
    np.testing.assert_array_equal(adjusted_vi[:2], vi[:2])


def test_fci_wall_layer_preserves_inactive_aliases_and_uses_forward_only_hit():
    driver = _driver_module()
    shape = (5, 1, 1)
    vi = np.zeros(shape)
    ve = np.zeros(shape)
    backward_target = np.zeros(shape + (2,))
    forward_target = np.zeros(shape + (2,))
    forward_target[-2, 0, 0] = (8.0, 12.0)
    selected_backward = np.zeros(shape, dtype=bool)
    selected_forward = np.zeros(shape, dtype=bool)
    selected_forward[-2, 0, 0] = True
    active = np.ones(shape, dtype=bool)
    active[-2, 0, 0] = False
    out_vi, out_ve, diagnostics = driver._blend_rung3_fci_wall_layer(
        vi, ve, backward_target, forward_target,
        selected_backward, selected_forward,
        np.ones(shape), np.ones(shape), np.ones(shape), np.ones(shape),
        active, 3,
    )
    np.testing.assert_array_equal(out_vi, vi)
    np.testing.assert_array_equal(out_ve, ve)
    assert int(np.asarray(diagnostics["modified_owner"]).sum()) == 0


def test_local_initializer_executes_operator_boundary_builder(monkeypatch):
    """Exercise the production initializer path, including its import wiring."""
    driver = _driver_module()
    operator_builder = getattr(
        driver, "build_local_fci_drb_eb_operator_boundary_bundle"
    )
    called = {"builder": False, "compatibility": False}

    def fake_operator_builder(*args, **kwargs):
        called["builder"] = True
        return object()

    monkeypatch.setattr(
        driver, "build_local_fci_drb_eb_operator_boundary_bundle", fake_operator_builder
    )

    def fake_compatibility(model, adjusted, *, layer_count):
        del model
        assert layer_count == 2
        called["compatibility"] = True
        return adjusted, np.zeros(
            len(driver.BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES), dtype=np.float64
        )

    monkeypatch.setattr(
        driver, "initialize_boundary_compatible_rung3", fake_compatibility
    )
    shape = (3, 1, 1)
    zero = np.zeros(shape)
    state = driver.FciDrbEBState(
        density=np.ones(shape), phi=zero, Te=np.ones(shape), Ti=np.ones(shape),
        Vi=zero, Ve=zero, vorticity=zero,
    )
    endpoint = np.zeros(shape + (5,))
    wall = {
        "backward_endpoint_state": endpoint,
        "forward_endpoint_state": endpoint,
        "selected_backward_wall": np.zeros(shape, dtype=bool),
        "selected_forward_wall": np.ones(shape, dtype=bool),
        "backward_alpha": np.ones(shape), "forward_alpha": np.ones(shape),
    }
    fake_face_bc = SimpleNamespace(phi=object(), Ti=object())
    fake_geometry = SimpleNamespace(active_cell_mask_owned=np.ones(shape, dtype=bool))
    fake_parameters = SimpleNamespace(tau=1.0)
    fake_stencil = SimpleNamespace(dx_min=np.ones(shape), dx_plus=np.ones(shape))

    class FakeModel:
        control_volume_geometry = None

        def __init__(self):
            self.geometry = fake_geometry
            self.domain = SimpleNamespace(mesh_axis_names=(None, None, None))
            self.parameters = fake_parameters

        def _owner_state(self, value):
            return value

        def _face_bcs(self, value):
            return fake_face_bc

        def _prepare_state_halo(self, value, face_bc):
            return value

        def _parallel_operator_boundary(self, *, state_halo, operator_boundary):
            return object()

        def _stencil_builder_context(self):
            return object()

        def _fci_parallel_characteristic_wall_data(self, **kwargs):
            return {"wall_data": wall, "primitive_stencils": [fake_stencil]}

        def _vorticity_from_polarization(self, phi, ti, face_phi, face_ti):
            self.vorticity_args = (phi, ti, face_phi, face_ti)
            return zero

    fake_model = FakeModel()
    result, diagnostics = driver._initialize_rung3_wall_layer_local(
        fake_model, state, timestep=0.01, layer_count=2
    )
    assert called["builder"]
    assert called["compatibility"]
    assert operator_builder is not None
    np.testing.assert_array_equal(np.asarray(result.vorticity), zero)
    assert np.asarray(diagnostics).shape == (
        6 + len(driver.BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES),
    )


def test_factored_wall_data_executes_phi_plasma_stencil(monkeypatch):
    """Exercise the real factored helper with all six live state fields.

    The production helper builds plasma-side stencils for ``phi`` in addition
    to the five material fields.  Keep the geometry and stencil operations
    mocked here so this remains a fast unit test, but invoke the actual helper
    rather than a driver-level stand-in; this catches missing state-field keys.
    """
    driver = _driver_module()
    from drbx.native import fci_drb_EB_rhs as rhs_module
    from drbx.native.fci_boundaries import LocalStencil1D

    shape = (2, 1, 1)
    zero = np.zeros(shape)
    state = driver.FciDrbEBState(
        density=np.ones(shape), phi=np.full(shape, 2.0),
        Te=np.ones(shape), Ti=np.ones(shape), Vi=zero, Ve=zero,
        vorticity=zero,
    )
    stencil = LocalStencil1D(
        center=np.ones(shape), minus=np.ones(shape), plus=np.ones(shape),
        dx_min=np.ones(shape), dx_plus=np.ones(shape),
    )
    endpoint_kind = np.full(shape, 2, dtype=np.int32)
    direction = SimpleNamespace(
        endpoint_kind=endpoint_kind,
        endpoint_b_contra_x=np.ones(shape), endpoint_bmag=np.ones(shape),
    )
    model = SimpleNamespace(
        parallel_material_scheme="production-path",
        physical_wall_model_name="simplified-gbs-mpe",
        geometry=SimpleNamespace(
            owned_shape=shape,
            maps=SimpleNamespace(backward=direction, forward=direction),
        ),
        domain=SimpleNamespace(
            layout=SimpleNamespace(owned_slices_cell=(slice(None),) * 3)
        ),
        parameters=SimpleNamespace(
            tau=1.0, mi_over_me=1.0,
            parallel_characteristic_wall_law="primitive-least-residual",
        ),
        parallel_short_leg_treatment="none",
        parallel_short_leg_cfl_limit=2.0,
        parallel_short_leg_selection="all-physical-walls",
        conducting_sheath_wall_potential=None,
    )

    # Bind only the lightweight pieces around the production helper.  The
    # plasma-side callback records all requests, including the phi request.
    requested_plasma_fields = []

    def prepare_q(q_owned, q_trace, context):
        del q_owned, q_trace, context
        return None, None, None

    def plasma_side(q_owned, template_bc, context):
        del template_bc, context
        requested_plasma_fields.append(q_owned)
        return stencil

    model._fci_prepare_q = prepare_q
    model._fci_plasma_side_stencil = plasma_side
    face_bc = SimpleNamespace(
        density=object(), Te=object(), Ti=object(), Vi=object(),
        Ve=object(), phi=object(),
    )
    parallel_boundary = SimpleNamespace(
        density=object(), Te=object(), Ti=object(), Vi=object(), Ve=object(),
    )

    monkeypatch.setattr(
        rhs_module,
        "build_local_fci_stencil_from_field",
        lambda *args, **kwargs: stencil,
    )
    monkeypatch.setattr(
        rhs_module,
        "resolve_fci_material_wall_endpoint_state",
        lambda *args, **kwargs: np.ones(shape + (5,)),
    )
    monkeypatch.setattr(
        rhs_module,
        "parallel_characteristic_wall_data",
        lambda *args, **kwargs: {},
    )
    model._effective_physical_wall_model_name = lambda: "simple-conducting-sheath"

    result = rhs_module.LocalFciDrbEBRhs._fci_parallel_characteristic_wall_data(
        model,
        state_halo=state,
        face_bc=face_bc,
        parallel_boundary=parallel_boundary,
        context=None,
    )

    assert result["wall_data"] == {}
    # Six plasma-side requests prove that the phi field was available to the
    # real helper; the five material requests alone would not catch the bug.
    assert len(requested_plasma_fields) == 6
    assert any(np.allclose(field, state.phi) for field in requested_plasma_fields)


def test_parser_exposes_opt_in_rung3_startup_and_restriction():
    driver = _driver_module()
    parser = driver._build_parser()
    defaults = parser.parse_args([])
    assert defaults.initialize_rung3_wall_layer is False
    assert defaults.rung3_wall_layer_cells == 8

    with pytest.raises(SystemExit):
        driver.main(["--initialize-rung3-wall-layer", "--geometry-only"])


def test_driver_wires_live_wall_targets_and_polarization_derived_vorticity():
    source = DRIVER.read_text(encoding="utf-8")
    assert "initialize_rung3_wall_layer: bool = False" in source
    assert "rung3_wall_layer_cells: int = 8" in source
    assert 'wall_data["backward_endpoint_state"]' in source
    assert 'wall_data["forward_endpoint_state"]' in source
    assert "_fci_parallel_characteristic_wall_data" in source
    assert "initialize_boundary_compatible_rung3(" in source
    assert "boundary_compatibility_acceptance_failures(" in source
    assert "model._vorticity_from_polarization(" in source
    assert "--initialize-rung3-wall-layer" in source
    assert "--rung3-wall-layer-cells" in source
