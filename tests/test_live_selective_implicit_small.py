"""Lightweight tests for the live selective split helpers."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "work" / "boundary_load_audit"))
import run_live_selective_implicit_small as runner  # noqa: E402
from run_live_selective_implicit_small import (  # noqa: E402
    _combined_residual_jax,
    _final_comparison_metrics,
    _legacy_final_state_delta,
    _public_case_result,
    _resolve_case_names,
    _rebuild_adapter_from_state,
    _stage_initial_coordinates,
    CASES,
    EventLog,
    FIELDS,
    partition_rhs,
    selected_rows,
)


def test_reduced_packing_has_exact_selected_complement():
    mask = selected_rows(CASES["minimal"], 3)
    assert mask.shape == (6, 3)
    assert mask.sum() == 12
    assert (~mask).sum() == 6
    assert mask[FIELDS.index("Te")].all()
    assert not mask[FIELDS.index("density")].any()
    assert not mask[FIELDS.index("Ti")].any()


def test_candidate_partition_preserves_physical_rhs_field_by_field():
    physical = {field: np.arange(4, dtype=float) + i for i, field in enumerate(FIELDS)}
    explicit = {field: 0.25 * physical[field] for field in FIELDS}
    implicit = {field: physical[field] - explicit[field] for field in FIELDS}
    candidate_explicit, candidate_implicit, defects = partition_rhs(
        physical, explicit, implicit, CASES["minimal"]
    )
    for field in FIELDS:
        np.testing.assert_allclose(candidate_explicit[field] + candidate_implicit[field], physical[field])
        assert defects[field] == 0.0
    for field in ("density", "Ti"):
        np.testing.assert_allclose(candidate_implicit[field], 0.0)
        np.testing.assert_allclose(candidate_explicit[field], physical[field])
    for field in CASES["minimal"]:
        np.testing.assert_allclose(candidate_implicit[field], implicit[field])
        np.testing.assert_allclose(candidate_explicit[field], explicit[field])


def test_case_field_sets_are_nested_and_full_is_reference():
    assert set(CASES["minimal"]) < set(CASES["conservative"]) < set(CASES["full"])


def test_event_log_provenance_distinguishes_small_state_from_production_trajectory(tmp_path):
    log = EventLog(tmp_path / "events.json", {"mode": "test"})
    assert log.record["production_trajectory_advanced"] is False
    assert log.record["small_physical_state_advanced"] is True


def test_stage_initial_coordinates_use_supplied_warm_start():
    supplied = np.asarray([1.25, -0.75, 0.5])

    class Adapter:
        def _initial_coordinates(self, _stage_base):
            raise AssertionError("warm start should bypass base reconstruction")

    result = _stage_initial_coordinates(Adapter(), np.zeros((6, 2)), supplied)
    np.testing.assert_array_equal(result, supplied)


def test_final_metrics_use_full_accumulated_increment_and_handle_zero_denominator():
    state0 = SimpleNamespace(**{
        **{field: np.zeros(2) for field in FIELDS},
        "phi": np.zeros(2),
    })
    full = SimpleNamespace(**{
        **{field: np.ones(2) for field in FIELDS},
        "phi": np.ones(2),
    })
    candidate = SimpleNamespace(**{
        **{field: np.ones(2) for field in FIELDS},
        "phi": np.ones(2),
    })

    class Adapter:
        active_flat = np.asarray([0, 1])

    metrics = _final_comparison_metrics(Adapter(), state0, 0.0, full, 1.0, candidate, 1.0)
    assert metrics["aggregate_increment_relative"] == 0.0
    assert metrics["fields"]["density"]["increment_relative"] == 0.0

    stationary = _final_comparison_metrics(Adapter(), state0, 0.0, state0, 0.0, full, 1.0)
    assert stationary["fields"]["density"]["increment_relative"] is None
    assert stationary["aggregate_increment_relative"] is None


def test_sequential_context_rebuild_uses_accepted_state(monkeypatch):
    import reduced_implicit_benchmark

    captured = {}

    class Replacement:
        def __init__(self, model, base, dt, source, *, mode):
            captured.update(model=model, base=base, dt=dt, source=source, mode=mode)

    monkeypatch.setattr(reduced_implicit_benchmark, "ReducedActualAdapter", Replacement)
    state = object()
    old = SimpleNamespace(model="model", dt=0.25, source="source")
    rebuilt = _rebuild_adapter_from_state(old, state)
    assert isinstance(rebuilt, Replacement)
    assert captured == {"model": "model", "base": state, "dt": 0.25, "source": "source", "mode": "small"}


def test_public_case_result_strips_internal_arrays_for_inconclusive_serialization():
    internal_state = object()
    public = _public_case_result({
        "status": "completed",
        "final_material": np.zeros((6, 2)),
        "final_phi_active": np.zeros(2),
        "final_coordinates_internal": np.zeros(3),
        "final_state_internal": internal_state,
    })
    assert public == {"status": "completed"}


def test_legacy_delta_flattens_material_before_concatenating_phi():
    reference = {"final_material": np.zeros((6, 2)), "final_phi_active": np.zeros(2)}
    candidate = {"final_material": np.ones((6, 2)), "final_phi_active": np.ones(2)}
    absolute, relative = _legacy_final_state_delta(candidate, reference)
    assert absolute == np.sqrt(14.0)
    assert relative == absolute / 1.0e-300


def test_mocked_run_live_two_steps_rebuilds_each_candidate_context(tmp_path, monkeypatch):
    """Exercise sequential control flow without constructing the physical model."""
    import json
    import reduced_implicit_benchmark

    def make_state(material):
        values = {field: np.asarray(material[i], dtype=float).copy() for i, field in enumerate(FIELDS)}
        values["phi"] = np.asarray(material[0], dtype=float).copy()
        return SimpleNamespace(**values)

    initial_state = make_state(np.zeros((6, 2)))
    records = {}

    class FakeAdapter:
        active_flat = np.asarray([0, 1])
        shape = (2,)
        n_active = 2
        algebraic_size = 3
        augmented = True
        model = object()
        source = object()
        dt = 0.1

        def __init__(self, model=None, base=None, dt=0.1, source=None, *, mode="small"):
            self.model = model if model is not None else object()
            self.base = initial_state if base is None else base
            self.dt = dt
            self.source = source
            self.base_material = self.reduce_material(self.base)

        def reduce_material(self, state):
            return np.stack([np.asarray(getattr(state, field)) for field in FIELDS])

        def reconstruct(self, material, *, verify=True):
            del material, verify
            return SimpleNamespace(
                state=self.base,
                z=np.asarray([0.0, 0.0, 0.0]),
                diagnostics={"admissibility": {"admissible": True}},
                algebraic_residual=np.zeros(3),
            )

    monkeypatch.setattr(reduced_implicit_benchmark, "ReducedActualAdapter", FakeAdapter)
    monkeypatch.setattr(reduced_implicit_benchmark, "_small_adapter", lambda **kwargs: FakeAdapter())

    offsets = {"minimal": 1.0, "conservative": 2.0, "full": 3.0}

    def fake_run_case(adapter, current_material, fields, *, dt, stage_iterations, tolerance,
                      damping, initial_coordinates, event_log, case_name, timestep):
        del fields, dt, stage_iterations, tolerance, damping, event_log
        material = np.asarray(current_material, dtype=float)
        records[(case_name, timestep)] = {
            "base_material": material.copy(),
            "initial_coordinates": np.asarray(initial_coordinates, dtype=float).copy(),
            "adapter_base": adapter.base,
        }
        final_material = material + offsets[case_name] + timestep
        final_state = make_state(final_material)
        final_coordinates = np.asarray(initial_coordinates, dtype=float) + offsets[case_name] + timestep
        stage = {"converged": True, "final_residual_l2": 0.0}
        return {
            "case": case_name,
            "status": "completed",
            "stage1": stage,
            "stage2": stage,
            "elapsed_seconds": 0.0,
            "final_material": final_material,
            "final_phi_active": final_state.phi.copy(),
            "final_gauge": float(final_coordinates[-1]),
            "final_coordinates_internal": final_coordinates,
            "final_state_internal": final_state,
        }

    monkeypatch.setattr(runner, "run_case", fake_run_case)
    output = tmp_path / "mocked_steps.json"
    result = runner.run_live(output=output, case="all", cases="minimal,full", steps=2)

    assert result["status"] == "completed"
    assert result["steps_completed"] == 2
    assert "final_comparison_metrics" in result["cases"]["minimal"]
    assert json.loads(output.read_text())["status"] == "completed"
    sidecar = np.load(tmp_path / "mocked_steps.npz")
    np.testing.assert_allclose(sidecar["minimal_final_material"], records[("minimal", 2)]["base_material"] + 1.0 + 2.0)
    assert list(result["cases"]) == ["minimal", "full"]
    assert "conservative" not in sidecar.files
    for name in ("minimal", "full"):
        np.testing.assert_allclose(
            records[(name, 2)]["base_material"],
            records[(name, 1)]["base_material"] + offsets[name] + 1.0,
        )
        np.testing.assert_allclose(
            records[(name, 2)]["initial_coordinates"],
            records[(name, 1)]["initial_coordinates"] + offsets[name] + 1.0,
        )


def test_case_subset_selection_preserves_order_and_rejects_invalid_combinations():
    assert _resolve_case_names("all", "full,minimal") == ["full", "minimal"]
    assert _resolve_case_names("minimal", None) == ["minimal"]
    with pytest.raises(ValueError, match="unknown case"):
        _resolve_case_names("all", "minimal,missing")
    with pytest.raises(ValueError, match="duplicates"):
        _resolve_case_names("all", "minimal,minimal")
    with pytest.raises(ValueError, match="conflicts"):
        _resolve_case_names("minimal", "minimal,full")


def test_combined_material_phi_gauge_residual_has_newton_root(monkeypatch):
    """The live combined residual couples material and algebraic unknowns."""
    import drbx.native.fci_boundary_imex_split as split_module

    n_active = 2
    base = np.zeros((6, n_active), dtype=float)
    selected = np.asarray([2, 3], dtype=int)  # Te active rows
    mask = np.zeros(6 * n_active, dtype=bool)
    mask[selected] = True

    class FakeAdapter:
        model = object()
        source = None
        active_flat = np.arange(n_active)
        augmented = True

        def full_vector(self, material, coordinates):
            values = {field: material[i] for i, field in enumerate(FIELDS)}
            values["phi"] = jnp.zeros(n_active)
            return None, SimpleNamespace(**values)

        def _algebraic_residual(self, material, coordinates):
            del material
            return jnp.asarray([coordinates[0] - 1.0, coordinates[1] + 0.5, coordinates[2]])

        def admissibility(self, material, coordinates):
            del material, coordinates
            return {"admissible": True}

    def fake_split(_model, state, *, source_owned, polarization_multiplier):
        del source_owned, polarization_multiplier
        zeros = jnp.zeros(n_active)
        implicit = SimpleNamespace(
            **{field: (state.Te ** 2 if field == "Te" else zeros) for field in FIELDS}
        )
        return SimpleNamespace(implicit=implicit, explicit=implicit, physical=implicit)

    monkeypatch.setattr(split_module, "evaluate_boundary_imex_split", fake_split)
    adapter = FakeAdapter()
    stage_dt = 0.1
    unknowns = np.asarray([0.5, 0.5, 0.0, 0.0, 0.0])

    def residual(values):
        return _combined_residual_jax(
            adapter, base, selected, mask, ("Te",), stage_dt, values
        )

    for _ in range(8):
        value = jnp.asarray(unknowns)
        residual_value = np.asarray(residual(value), dtype=float)
        if np.linalg.norm(residual_value) < 1.0e-10:
            break
        jacobian = np.asarray(jax.jacfwd(residual)(value), dtype=float)
        unknowns += np.linalg.solve(jacobian, -residual_value)
    np.testing.assert_allclose(unknowns, [0.0, 0.0, 1.0, -0.5, 0.0], atol=1.0e-9)
    np.testing.assert_allclose(np.asarray(residual(jnp.asarray(unknowns))), 0.0, atol=1.0e-9)
