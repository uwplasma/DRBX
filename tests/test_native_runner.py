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
from jax_drb.reference.cases import ReferenceCase


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

_FLUID_1D_MMS_INPUT = """
nout = 50
timestep = 0.1

MXG = 0

[mesh]
nx = 1
ny = 128
nz = 1
Ly = 10
dy = Ly / ny
J = 1

[solver]
mxstep = 10000
rtol = 1e-7
mms = true

[model]
components = i
normalise_metric = false
Nnorm = 1e18
Bnorm = 1
Tnorm = 5

[i]
type = evolve_density, evolve_pressure, evolve_momentum
charge = 1.0
AA = 2.0
thermal_conduction = false

[Ni]
solution = 1 - 0.1*sin(t - 2.0*y)
source = -0.1*cos(t - 2.0*y) + 0.0628318530717959*cos(2*t + y)

[Pi]
solution = 0.1*cos(t + 3.0*y) + 1
source = (0.0628318530717959*cos(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.0125663706143592*sin(2*t + y)*cos(t - 2.0*y)/(1 - 0.1*sin(t - 2.0*y))^2)*(0.0666666666666667*cos(t + 3.0*y) + 0.666666666666667) - 0.1*sin(t + 3.0*y) + 0.0628318530717959*(0.1*cos(t + 3.0*y) + 1)*cos(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.0188495559215388*sin(t + 3.0*y)*sin(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.0125663706143592*(0.1*cos(t + 3.0*y) + 1)*sin(2*t + y)*cos(t - 2.0*y)/(1 - 0.1*sin(t - 2.0*y))^2

[NVi]
solution = 0.2*sin(2*t + y)
source = -0.188495559215388*sin(t + 3.0*y) + 0.4*cos(2*t + y) + 0.0251327412287183*sin(2*t + y)*cos(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.00251327412287184*sin(2*t + y)^2*cos(t - 2.0*y)/(1 - 0.1*sin(t - 2.0*y))^2
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


def test_native_runner_tracks_fluid_rhs_summary_baseline() -> None:
    config = parse_bout_input(_FLUID_1D_MMS_INPUT)
    result = run_config_case(
        config,
        case_name="fluid_1d_mms_rhs",
        parity_mode="one_rhs",
        compare_variables=("ddt(Ni)", "ddt(Pi)", "ddt(NVi)"),
        reference_case=ReferenceCase(
            name="fluid_1d_mms_rhs",
            stage="stage4",
            reference_path="tests/integrated/1D-fluid/data/BOUT.inp",
            parity_mode="one_rhs",
            rationale="RHS parity",
            compare_variables=("ddt(Ni)", "ddt(Pi)", "ddt(NVi)"),
            extra_overrides=("i:diagnose=true",),
            trim_y_guards=True,
        ),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/fluid_1d_mms_rhs.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=1e-6, scalar_atol=1e-8)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_fluid_rhs_array_baseline() -> None:
    config = parse_bout_input(_FLUID_1D_MMS_INPUT)
    result = run_config_case(
        config,
        case_name="fluid_1d_mms_rhs",
        parity_mode="one_rhs",
        compare_variables=("ddt(Ni)", "ddt(Pi)", "ddt(NVi)"),
        reference_case=ReferenceCase(
            name="fluid_1d_mms_rhs",
            stage="stage4",
            reference_path="tests/integrated/1D-fluid/data/BOUT.inp",
            parity_mode="one_rhs",
            rationale="RHS parity",
            compare_variables=("ddt(Ni)", "ddt(Pi)", "ddt(NVi)"),
            extra_overrides=("i:diagnose=true",),
            trim_y_guards=True,
        ),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/fluid_1d_mms_rhs.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=1e-6, array_atol=1e-8)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_fluid_one_step_summary_baseline() -> None:
    config = parse_bout_input(_FLUID_1D_MMS_INPUT)
    result = run_config_case(
        config,
        case_name="fluid_1d_mms_one_step",
        parity_mode="one_step",
        compare_variables=("Ni", "Pi", "NVi"),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/fluid_1d_mms_one_step.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=5e-5, scalar_atol=1e-8)
    assert comparison.ok, comparison.issues
    assert result.time_points == (0.0, 0.1)


def test_native_runner_tracks_fluid_one_step_array_baseline() -> None:
    config = parse_bout_input(_FLUID_1D_MMS_INPUT)
    result = run_config_case(
        config,
        case_name="fluid_1d_mms_one_step",
        parity_mode="one_step",
        compare_variables=("Ni", "Pi", "NVi"),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/fluid_1d_mms_one_step.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=2e-6, array_atol=5e-7)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_fluid_short_window_summary_baseline() -> None:
    config = parse_bout_input(_FLUID_1D_MMS_INPUT)
    result = run_config_case(
        config,
        case_name="fluid_1d_mms",
        parity_mode="short_window",
        compare_variables=("Ni", "Pi", "NVi"),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/fluid_1d_mms.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=5e-5, scalar_atol=1e-8)
    assert comparison.ok, comparison.issues
    assert result.time_points == tuple(0.1 * index for index in range(51))


def test_native_runner_tracks_fluid_short_window_array_baseline() -> None:
    config = parse_bout_input(_FLUID_1D_MMS_INPUT)
    result = run_config_case(
        config,
        case_name="fluid_1d_mms",
        parity_mode="short_window",
        compare_variables=("Ni", "Pi", "NVi"),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/fluid_1d_mms.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=3e-6, array_atol=5e-7)
    assert comparison.ok, comparison.issues
