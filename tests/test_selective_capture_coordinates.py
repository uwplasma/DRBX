"""Fast coordinate tests for the selective captured-material runner."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "work/boundary_load_audit"))
from run_reduced_implicit_capture import (
    CaptureReducedProblem, FIELDS, _problem_record_metadata,
)


def make_problem(fields=FIELDS):
    problem = object.__new__(CaptureReducedProblem)
    problem.selected_fields = tuple(fields)
    problem.selected_indices = tuple(FIELDS.index(name) for name in fields)
    problem.excluded_fields = tuple(name for name in FIELDS if name not in fields)
    problem.adapter = SimpleNamespace(
        n_active=2,
        base_material=np.arange(12, dtype=float).reshape(6, 2),
        embed_material_direction=lambda value: np.asarray(value, dtype=float),
    )
    problem.nu = len(fields) * 2
    problem.diagnostic_arrays = {}
    return problem


def test_default_six_field_map_is_backward_compatible():
    problem = make_problem()
    material = problem.adapter.base_material.copy()
    np.testing.assert_array_equal(problem.reduce_material(material), material)
    np.testing.assert_array_equal(problem.expand_material(material), material)
    assert problem.nu == 12
    assert problem.excluded_fields == ()


def test_five_field_map_has_expected_dimensions_and_fixed_density():
    fields = ("Te", "Ti", "Vi", "Ve", "vorticity")
    problem = make_problem(fields)
    selected = np.full((5, 2), -3.0)
    expanded = problem.expand_material(selected)
    np.testing.assert_array_equal(expanded[0], problem.adapter.base_material[0])
    np.testing.assert_array_equal(expanded[1:], selected)
    np.testing.assert_array_equal(problem.reduce_material(expanded), selected)
    assert problem.nu == 10
    assert problem.zero_leakage_check()["passed"]


def test_selected_tangent_embeds_zero_density_and_round_trips():
    problem = make_problem(("Te", "Ti", "Vi", "Ve", "vorticity"))
    direction = np.arange(problem.nu, dtype=float)
    embedded = problem.embed_material_direction(direction)
    full = embedded.reshape(6, 2)
    np.testing.assert_array_equal(full[0], 0.0)
    np.testing.assert_array_equal(full[1:].ravel(), direction)
    assert problem.diagnostic_arrays["last_excluded_material_tangent_max_abs"] == 0.0
    output_check = problem.restricted_output_check(direction)
    assert output_check["excluded_output_coordinates"] == 2
    assert output_check["excluded_output_max_abs"] == 0.0


def test_primitive_split_rejects_missing_or_nonterminal_vorticity():
    with pytest.raises(ValueError, match="requires the vorticity"):
        CaptureReducedProblem._validate_material_selection(
            ("Te", "Ti", "Vi", "Ve"), "primitive-vorticity-lower"
        )

    with pytest.raises(ValueError, match="final selected lane"):
        CaptureReducedProblem._validate_material_selection(
            ("Te", "vorticity", "Ve"), "primitive-vorticity-lower"
        )


def test_selective_convergence_ignores_excluded_density_residual():
    problem = make_problem(("Te", "Ti", "Vi", "Ve", "vorticity"))
    problem.adapter.n_active = 1
    problem.material_rows = np.array([1, 2, 3, 4, 5])
    problem.algebraic_rows = np.array([6, 7])
    problem.material_metric = np.ones(5)
    problem.algebraic_metric = np.ones(2)
    problem.jnp = np
    problem._initial_full_residual = np.zeros(8)
    problem._initial_full_residual[0] = 1.e6  # excluded density
    problem._initial_full_residual[[1, 2, 3, 4, 5]] = [1.e-6, 1.e6, 1.e6, 1.e6, 1.e6]
    problem.ctx = SimpleNamespace(
        config=SimpleNamespace(material_atol=1.e-8, material_rtol=1.e-8),
        polarization_diagnostics=lambda vector, residual: {
            "raw_rhs_l2": 1., "polarization_l2": 0.,
        },
    )
    problem._algebraic_gate = lambda vector, residual: {
        "polarization_violation": 0., "gauge_violation": 0.,
    }
    # Density row is large, while all selected rows are solved.
    residual = np.zeros(8)
    residual[0] = 1.e6  # excluded density remains large but is ignored
    problem.adapter.full_vector = lambda material, z: (np.zeros(8), None)
    problem.adapter.full_residual = lambda vector: residual
    problem._full_cache = None
    problem.full_eval = lambda material, z: (np.zeros(14), None, residual)
    assert problem.full_convergence(np.zeros(5), np.zeros(2), None, None)
    residual[1] = 1.e-4  # one small-initial-norm lane fails its own target
    assert not problem.full_convergence(np.zeros(5), np.zeros(2), None, None)
    assert problem._last_selection_convergence["lane_passed"] == [False, True, True, True, True]


def test_setup_metadata_requires_constructed_problem_and_preserves_dimensions():
    adapter = SimpleNamespace(
        shape=(4, 4, 4), n_active=7, algebraic_size=8,
        model=SimpleNamespace(gmres_config=SimpleNamespace(preconditioner="coarse-additive")),
    )
    problem = SimpleNamespace(
        selected_fields=("Te", "Ti", "Vi", "Ve", "vorticity"),
        excluded_fields=("density",), nu=35,
    )
    args = SimpleNamespace(
        material_fields=problem.selected_fields,
        inner_preconditioner="multiplicative-line-u-coarse-1",
        material_preconditioner="primitive-vorticity-lower",
        max_dense_unknowns=512,
    )
    metadata = {"setup_seconds": 1.5, "coarse_seconds": 0.5, "coarse_rank": 3}
    effective = {"maxiter": 1000}
    result = _problem_record_metadata(adapter, problem, args, metadata, effective)
    assert result["material_unknowns"] == 35
    assert result["coupled_unknowns"] == 43
    assert result["selected_material_fields"][-1] == "vorticity"
    assert result["dense_guard"]["problem_unknowns"] == 43
