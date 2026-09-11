"""Small algebraic tests for the coupled boundary model adapter."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import drbx.native.fci_boundary_imex_model as adapter
from drbx.native.fci_drb_EB_rhs import FciDrbEBState
from drbx.native.fci_boundary_imex import CoupledStageConfig
from drbx.native.fci_boundaries import BC_NEUMANN
from drbx.geometry import FCI_DEP_PHYSICAL_BOUNDARY, FCI_DEP_FIELD_INTERIOR


def _state(value=0.0):
    fields = [jnp.full((2,), value, dtype=jnp.float64) for _ in range(7)]
    return FciDrbEBState(*fields)


class _ToyModel:
    physical_wall_model_name = "simplified-gbs-mpe"

    def __init__(self):
        self.geometry = SimpleNamespace(active_cell_mask_owned=jnp.array([True, False]))
        self.domain = SimpleNamespace(mesh_axis_names=())

    def _face_bcs(self, state):
        del state
        return SimpleNamespace(phi=object())

    def _polarization_solver(self, face_bc):
        del face_bc
        return SimpleNamespace(_operator_mass_weights=lambda: (
            jnp.array([True, False]), jnp.array([1.0, 3.0])
        ))

    def _simplified_gbs_mpe_phi_gauge_data(self, state, face_bc):
        del state, face_bc
        return jnp.array([1.0, 0.0]), jnp.asarray(0.25), jnp.asarray(0.5)

    def polarization_residual(self, state, *, phi_owned):
        del state
        return phi_owned

    def polarization_balance_terms(self, state, *, phi_owned):
        zero = jnp.zeros_like(phi_owned)
        return jnp.stack((phi_owned, zero, zero))


def test_context_residual_keeps_inactive_owner_identity_and_augmented_gauge(monkeypatch):
    base = _state(1.0)
    model = _ToyModel()
    implicit = _state(0.5)
    monkeypatch.setattr(adapter, "_spmd_sum", lambda value, domain: value)
    monkeypatch.setattr(
        adapter,
        "evaluate_boundary_imex_split",
        lambda *args, **kwargs: SimpleNamespace(implicit=implicit),
    )
    context = adapter.CoupledBoundaryStageContext(model, base, 2.0)
    trial = base.replace(phi=jnp.array([0.25, 3.0]))
    residual = context.residual(context.pack(trial, 0.75))
    parts = context.residual_components(context.pack(trial, 0.75))
    # Active material residual is trial-base-dt*implicit; inactive is identity.
    np.testing.assert_allclose(parts["material"][:, 0], -1.0)
    np.testing.assert_allclose(parts["material"][:, 1], 0.0)
    np.testing.assert_allclose(parts["polarization"], [1.0, 2.0])
    np.testing.assert_allclose(parts["gauge"], 0.0)
    assert residual.size == 6 * 2 * 7 // 7 + 2 + 1


def test_nonaugmented_context_has_zero_lambda_and_no_gauge(monkeypatch):
    model = _ToyModel()
    model.physical_wall_model_name = "simple-conducting-sheath"
    base = _state()
    monkeypatch.setattr(adapter, "_spmd_sum", lambda value, domain: value)
    monkeypatch.setattr(
        adapter,
        "evaluate_boundary_imex_split",
        lambda *args, **kwargs: SimpleNamespace(implicit=_state()),
    )
    context = adapter.CoupledBoundaryStageContext(model, base, 1.0)
    parts = context.residual_components(context.pack(base, 4.0))
    np.testing.assert_allclose(parts["polarization"], 0.0)
    np.testing.assert_allclose(parts["gauge"], 4.0)


def test_end_to_end_toy_stage_solves_material_phi_and_multiplier(monkeypatch):
    model = _ToyModel()
    base = _state(1.0).replace(phi=jnp.array([0.0, 0.0]))
    target = jnp.array([0.25, 0.0])
    implicit = _state(0.0).replace(density=jnp.array([0.2, 0.2]))
    monkeypatch.setattr(adapter, "_spmd_sum", lambda value, domain: value)
    monkeypatch.setattr(
        adapter,
        "evaluate_boundary_imex_split",
        lambda *args, **kwargs: SimpleNamespace(implicit=implicit),
    )
    model.polarization_residual = lambda state, *, phi_owned: phi_owned - target
    model._simplified_gbs_mpe_phi_gauge_data = lambda state, face_bc: (
        jnp.array([1.0, 0.0]), 0.0, 0.25
    )
    model.gmres_config = SimpleNamespace(acceptance_tol=1e-8, acceptance_atol=1e-10)
    model.recover_polarization_multiplier = lambda state: jnp.asarray(0.0)
    monkeypatch.setattr(
        adapter.CoupledBoundaryStageContext,
        "admissibility",
        lambda self, vector: {"admissible": True, "reason": "ok"},
    )
    cfg = CoupledStageConfig(newton_maxiter=5, linear_maxiter=100, linear_restart=50, backtracking_maxiter=20)
    state, multiplier, info = adapter.solve_coupled_boundary_stage(
        model, base, solve_dt=1.0, config=cfg
    )
    assert info.converged, info
    np.testing.assert_allclose(state.density, [1.2, 1.0], atol=1e-7)
    np.testing.assert_allclose(state.phi, target, atol=1e-7)
    np.testing.assert_allclose(multiplier, 0.0, atol=1e-7)


def test_admissibility_rejects_active_negative_but_ignores_alias(monkeypatch):
    monkeypatch.setattr(adapter, "_spmd_sum", lambda value, domain: value)
    model = _ToyModel()
    base = _state(1.0)
    context = adapter.CoupledBoundaryStageContext(model, base, 1.0)
    bad = base.replace(density=jnp.array([-1.0, -99.0]))
    result = context.admissibility(context.pack(bad, 0.0))
    assert not result["admissible"]
    assert result["reason"] == "nonpositive_density"


def test_admissibility_rejects_nonfinite_active_state(monkeypatch):
    monkeypatch.setattr(adapter, "_spmd_sum", lambda value, domain: value)
    model = _ToyModel()
    base = _state(1.0)
    context = adapter.CoupledBoundaryStageContext(model, base, 1.0)
    bad = base.replace(phi=jnp.array([jnp.nan, 0.0]))
    result = context.admissibility(context.pack(bad, 0.0))
    assert not result["admissible"]
    assert result["reason"] == "nonfinite_phi"


def test_admissibility_rejects_nonfinite_multiplier(monkeypatch):
    monkeypatch.setattr(adapter, "_spmd_sum", lambda value, domain: value)
    model = _ToyModel()
    context = adapter.CoupledBoundaryStageContext(model, _state(1.0), 1.0)
    result = context.admissibility(context.pack(_state(1.0), jnp.nan))
    assert result["reason"] == "nonfinite_multiplier"


def test_polarization_diagnostics_separates_target_and_acceptance(monkeypatch):
    monkeypatch.setattr(adapter, "_spmd_sum", lambda value, domain: value)
    model = _ToyModel()
    model.gmres_config = SimpleNamespace(
        tol=0.1, atol=0.01, acceptance_tol=0.001, acceptance_atol=0.0001
    )
    model.polarization_residual = lambda state, *, phi_owned: jnp.full_like(phi_owned, 0.05)
    model.polarization_balance_terms = lambda state, *, phi_owned: jnp.stack((jnp.zeros_like(phi_owned), jnp.ones_like(phi_owned), jnp.zeros_like(phi_owned)))
    context = adapter.CoupledBoundaryStageContext(model, _state(1.0), 1.0)
    diagnostics = context.polarization_diagnostics(
        context.pack(_state(1.0), 0.0),
        residual={"polarization": jnp.full((2,), 0.05), "gauge": jnp.asarray(0.0)},
    )
    assert bool(diagnostics["target_met"])
    assert not bool(diagnostics["accepted"])


def test_weighted_norm_matches_component_metric_for_tiny_volume(monkeypatch):
    monkeypatch.setattr(adapter, "_spmd_sum", lambda value, domain: value)
    model = _ToyModel()
    model._polarization_solver = lambda face_bc: SimpleNamespace(
        _operator_mass_weights=lambda: (jnp.array([True, False]), jnp.array([1.0e-12, 3.0e-12]))
    )
    context = adapter.CoupledBoundaryStageContext(model, _state(1.0), 1.0)
    vector = jnp.arange(15, dtype=jnp.float64)
    size = 2
    parts = {
        "material": vector[: 6 * size].reshape((6, size)),
        "polarization": vector[6 * size: 7 * size],
        "gauge": vector[7 * size],
    }
    norms = context._norms_from_parts(parts)
    expected = jnp.sqrt(jnp.sum(norms["material"] ** 2) + norms["polarization"] ** 2 + norms["gauge"] ** 2)
    np.testing.assert_allclose(context.norm(vector), expected, rtol=1e-12, atol=1e-12)
    inactive_changed = vector.at[1].set(1.0e12).at[7].set(-1.0e12)
    np.testing.assert_allclose(context.norm(inactive_changed), context.norm(vector), rtol=1e-12, atol=1e-12)


def _wall_model(*, phi=(0.0, 0.0), te=(1.0, 1.0), ti=(1.0, 1.0), n=(1.0, 1.0), b=(1.0, 1.0), fci=False):
    def bc(values):
        return SimpleNamespace(mask_x=jnp.ones((2,), dtype=bool), mask_y=jnp.zeros((2,), dtype=bool), mask_z=jnp.zeros((2,), dtype=bool), value_x=jnp.zeros((2,)), value_y=jnp.zeros((2,)), value_z=jnp.zeros((2,)), kind_x=jnp.full((2,), BC_NEUMANN), kind_y=jnp.zeros((2,), dtype=int), kind_z=jnp.zeros((2,), dtype=int))
    fields = SimpleNamespace(phi=bc(phi), Te=bc(te), Ti=bc(ti), density=bc(n), Vi=bc(n), Ve=bc(n), vorticity=bc(n))
    axes = tuple(SimpleNamespace(B_contra_owned=jnp.asarray(b), Bmag_owned=jnp.ones((2,))) for _ in range(3))
    maps = SimpleNamespace(
        backward=SimpleNamespace(endpoint_kind=jnp.full((2,), FCI_DEP_PHYSICAL_BOUNDARY if fci else FCI_DEP_FIELD_INTERIOR), endpoint_b_contra_x=jnp.asarray(b), endpoint_bmag=jnp.ones((2,))),
        forward=SimpleNamespace(endpoint_kind=jnp.full((2,), FCI_DEP_FIELD_INTERIOR), endpoint_b_contra_x=jnp.asarray(b), endpoint_bmag=jnp.ones((2,))),
    )
    model = _ToyModel()
    model.parallel_operator_scheme = "fci" if fci else "coordinate"
    model.conducting_sheath_wall_potential = 0.0
    model.parameters = SimpleNamespace(Te0=1.0, Ti0=1.0, tau=1.0, mi_over_me=100.0)
    model.domain.layout = SimpleNamespace(owned_slices_cell=(slice(None),))
    model.geometry = SimpleNamespace(active_cell_mask_owned=jnp.ones((2,), dtype=bool), face_bfield=SimpleNamespace(axes=axes), maps=maps)
    if fci:
        model.geometry.active_cell_mask = None
    model.face_fields = fields
    model._face_bcs = lambda state: model.face_fields
    model._prepare_scalar_halo = lambda value, face: value
    model._prepare_state_halo = lambda state, face: state
    model._stencil_builder_context = lambda: None
    model._fci_plasma_side_stencil = lambda value, face, context: SimpleNamespace(minus=value, plus=value)
    return model


@pytest.mark.parametrize("phi, b, expected", [
    ((-1.0, -1.0), (1.0, 1.0), "negative_sheath_drop_x"),
    ((0.0, 0.0), (1.0, 1.0), "ok"),
    ((-1.0, -1.0), (0.0, 0.0), "ok"),
])
def test_regular_sheath_admissibility_branches(monkeypatch, phi, b, expected):
    model = _wall_model(phi=phi, b=b)
    monkeypatch.setattr(adapter, "_spmd_sum", lambda value, domain: value)
    def trace(*args):
        value = jnp.asarray(phi if args[3] is model.face_fields.phi else (1.0, 1.0))
        return SimpleNamespace(value_x=value, value_y=jnp.zeros((2,)), value_z=jnp.zeros((2,)))
    monkeypatch.setattr(adapter, "build_local_boundary_face_trace_from_halo", trace)
    context = adapter.CoupledBoundaryStageContext(model, _state(1.0), 1.0)
    result = context.admissibility(context.pack(_state(1.0), 0.0))
    assert result["reason"] == expected


def test_fci_negative_nongrazing_and_grazing_paths(monkeypatch):
    monkeypatch.setattr(adapter, "_spmd_sum", lambda value, domain: value)
    model = _wall_model(phi=(0.0, 0.0), b=(1.0, 0.0), fci=True)
    for name in ("density", "Te", "Ti", "phi", "Vi", "Ve", "vorticity"):
        getattr(model.face_fields, name).mask_x = jnp.zeros((2,), dtype=bool)
    monkeypatch.setattr(adapter, "build_local_boundary_face_trace_from_halo", lambda *args: SimpleNamespace(value_x=jnp.asarray((0.0, 0.0) if args[3] is model.face_fields.phi else (1.0, 1.0)), value_y=jnp.zeros((2,)), value_z=jnp.zeros((2,))))
    model.conducting_sheath_wall_potential = 2.0
    context = adapter.CoupledBoundaryStageContext(model, _state(1.0), 1.0)
    result = context.admissibility(context.pack(_state(1.0), 0.0))
    assert result["reason"] == "negative_backward_sheath_drop"
