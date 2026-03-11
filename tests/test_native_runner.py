from __future__ import annotations

from pathlib import Path

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native import run_config_case
from jax_drb.parity.arrays import (
    build_array_payload_from_summary_payload,
    compare_array_payloads,
    load_portable_array_payload,
)
from jax_drb.parity.compare import compare_summary_payloads, load_summary_json


_EVOLVE_DENSITY_INPUT = """
nout = 5
timestep = 20

[mesh]
nx = 10
ny = 10
nz = 10

yn = y / (2π)
zn = z / (2π)

J = 1

[solver]
mxstep = 1000

[model]
components = e

[e]
type = evolve_density
charge = -1
AA = 1/1836

[Ne]
function = exp(-(x - 0.5)^2 - (mesh:yn - 0.5)^2 - (mesh:zn - 0.5)^2)
"""

_DIFFUSION_INPUT = """
nout = 5
timestep = 1000

[mesh]
nx = 10
ny = 10
nz = 1

dx = 0.0075 + 0.005*x
dy = 0.01
dz = 0.01

J = 1

[solver]
mxstep = 1000

[model]
components = h

[h]
type = evolve_density, evolve_pressure, anomalous_diffusion
AA = 1
charge = 1
anomalous_D = 2
thermal_conduction = false

[Nh]
function = 1 + H(x - 0.25) * H(0.75-x) * exp(-(y-π)^2)
bndry_all = neumann

[Ph]
function = Nh:function
bndry_all = neumann
"""


def test_native_runner_matches_committed_smallest_case_baseline() -> None:
    config = parse_bout_input(_EVOLVE_DENSITY_INPUT)
    result = run_config_case(
        config,
        case_name="evolve_density_rhs",
        parity_mode="one_rhs",
        compare_variables=("Ne",),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/evolve_density_rhs.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=1e-12, scalar_atol=1e-12)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_committed_diffusion_baseline() -> None:
    config = parse_bout_input(_DIFFUSION_INPUT)
    result = run_config_case(
        config,
        case_name="diffusion_one_step",
        parity_mode="one_step",
        compare_variables=("Nh", "Ph"),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/diffusion_one_step.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=1e-3, scalar_atol=2e-6)
    assert comparison.ok, comparison.issues
    assert result.time_points == (0.0, 1000.0)


def test_native_runner_tracks_diffusion_short_window_summary_baseline() -> None:
    config = parse_bout_input(_DIFFUSION_INPUT)
    result = run_config_case(
        config,
        case_name="diffusion_short_window",
        parity_mode="short_window",
        compare_variables=("Nh", "Ph"),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/diffusion_short_window.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=2e-3, scalar_atol=2e-6)
    assert comparison.ok, comparison.issues
    assert result.time_points == (0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0)


def test_native_runner_tracks_diffusion_one_step_array_baseline() -> None:
    config = parse_bout_input(_DIFFUSION_INPUT)
    result = run_config_case(
        config,
        case_name="diffusion_one_step",
        parity_mode="one_step",
        compare_variables=("Nh", "Ph"),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/diffusion_one_step.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=2e-4, array_atol=2e-6)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_diffusion_short_window_array_baseline() -> None:
    config = parse_bout_input(_DIFFUSION_INPUT)
    result = run_config_case(
        config,
        case_name="diffusion_short_window",
        parity_mode="short_window",
        compare_variables=("Nh", "Ph"),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/diffusion_short_window.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=2e-4, array_atol=2e-6)
    assert comparison.ok, comparison.issues
