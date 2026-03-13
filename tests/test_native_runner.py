from __future__ import annotations

from pathlib import Path

import numpy as np

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native import run_config_case
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.arrays import (
    build_array_payload_from_summary_payload,
    compare_array_payloads,
    load_portable_array_payload,
)
from jax_drb.parity.compare import compare_summary_payloads, load_summary_json
from jax_drb.reference.cases import ReferenceCase
from jax_drb.validation import (
    analyze_blob2d_array_payload,
    analyze_drift_wave_array_payload,
    compare_blob2d_analysis_results,
    load_blob2d_analysis_json,
)


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

_VORTICITY_INPUT = """
nout = 10
timestep = 20

MYG = 0

[mesh]
nx = 10
ny = 1
nz = 10

zn = z / (2π)
J = 1

[mesh:paralleltransform]
type = identity

[solver]
mxstep = 1000

[model]
components = vorticity

[vorticity]
diamagnetic = false
diamagnetic_polarisation = false
average_atomic_mass = 2
bndry_flux = false
poloidal_flows = false
split_n0 = false
phi_dissipation = false

[Vort]
function = exp(-((x-0.5)^2 + (mesh:zn - 0.5)^2)/(0.2^2))
"""

_BLOB2D_INPUT = """
nout = 50
timestep = 50

MYG = 0

[mesh]
nx = 260
ny = 1
nz = 256

Lrad = 0.05
Lpol = 0.05

Bpxy = 0.35
Rxy = 1.5

dx = Lrad * Rxy * Bpxy / (nx - 4)
dz = Lpol / Rxy / nz

hthe = 1
sinty = 0
Bxy = Bpxy
Btxy = 0
bxcvz = 1./Rxy^2

[mesh:paralleltransform]
type = identity

[solver]
mxstep = 10000

[model]
components = e, vorticity, sheath_closure

recalculate_metric = true

Nnorm = 2e18
Bnorm = mesh:Bxy
Tnorm = 5

[e]
type = evolve_density, isothermal

charge = -1
AA = 1./1836

poloidal_flows = false

temperature = 5

[Ne]
height = 0.5
width = 0.05

function = 1 + height * exp(-((x-0.25)/width)^2 - ((z/(2*pi) - 0.5)/width)^2)

[vorticity]

diamagnetic = true
diamagnetic_polarisation = false
average_atomic_mass = 1.0
bndry_flux = false
poloidal_flows = false
split_n0 = false
phi_dissipation = false

[sheath_closure]
connection_length = 10
"""

_DRIFT_WAVE_INPUT = """
nout = 50
timestep = 10

[mesh]
nx = 5
ny = 32
nz = 27

Lx = 0.01
Ly = 10
Lz = 0.01

B = 0.2
inv_Ln = 10
ixseps1 = nx
ixseps2 = nx

dr = Lx / (nx - 4)
dx = dr * B
dy = Ly / ny
dz = Lz / nz

g11 = B^2
g22 = 1
g33 = 1
J = 1 / B

[mesh:paralleltransform]
type = identity

[solver]
mxstep = 10000

[model]
components = (i, e, vorticity, sound_speed, braginskii_collisions, braginskii_friction, braginskii_heat_exchange)

[vorticity]
diamagnetic = false
diamagnetic_polarisation = false
average_atomic_mass = 1
bndry_flux = false
poloidal_flows = false

[vorticity:laplacian]
inner_boundary_flags = 2
outer_boundary_flags = 2

[i]
type = evolve_density, fixed_velocity, fixed_temperature
charge = 1
AA = 1
velocity = 0
temperature = 100

[Ni]
function = 1 + 1e-3 * sin(z - y)
bndry_xin = neumann(mesh:inv_Ln * units:meters^2 * units:Tesla / mesh:B)
bndry_xout = neumann(mesh:inv_Ln * units:meters^2 * units:Tesla / mesh:B)

[e]
type = quasineutral, evolve_momentum, fixed_temperature
charge = -1
AA = 1/1836
temperature = 100
"""

_NEUTRAL_MIXED_INPUT = """
nout = 15
timestep = 20

[mesh]
nx = 10
ny = 10
nz = 10

dx = 1e-3
dy = 1e-3
dz = 1e-3

yn = y / (2π)
zn = z / (2π)

J = 1

[solver]
mxstep = 1000

[model]
components = h

[h]
type = neutral_mixed

[Nh]
function = exp(-(x - 0.5)^2 - (mesh:yn - 0.5)^2 - (mesh:zn - 0.5)^2)

[Ph]
function = 0.1 * Nh:function
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


def test_native_runner_tracks_vorticity_rhs_summary_baseline() -> None:
    config = parse_bout_input(_VORTICITY_INPUT)
    result = run_config_case(
        config,
        case_name="vorticity_rhs",
        parity_mode="one_rhs",
        compare_variables=("ddt(Vort)",),
        reference_case=ReferenceCase(
            name="vorticity_rhs",
            stage="stage4",
            reference_path="tests/integrated/vorticity/data/BOUT.inp",
            parity_mode="one_rhs",
            rationale="Electrostatic vorticity RHS parity",
            compare_variables=("ddt(Vort)",),
            extra_overrides=("vorticity:diagnose=true",),
        ),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/vorticity_rhs.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=1e-9, scalar_atol=1e-12)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_vorticity_rhs_array_baseline() -> None:
    config = parse_bout_input(_VORTICITY_INPUT)
    result = run_config_case(
        config,
        case_name="vorticity_rhs",
        parity_mode="one_rhs",
        compare_variables=("ddt(Vort)",),
        reference_case=ReferenceCase(
            name="vorticity_rhs",
            stage="stage4",
            reference_path="tests/integrated/vorticity/data/BOUT.inp",
            parity_mode="one_rhs",
            rationale="Electrostatic vorticity RHS parity",
            compare_variables=("ddt(Vort)",),
            extra_overrides=("vorticity:diagnose=true",),
        ),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/vorticity_rhs.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=1e-9, array_atol=1e-12)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_vorticity_one_step_summary_baseline() -> None:
    config = parse_bout_input(_VORTICITY_INPUT)
    result = run_config_case(
        config,
        case_name="vorticity_one_step",
        parity_mode="one_step",
        compare_variables=("Vort", "phi"),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/vorticity_one_step.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=1e-3, scalar_atol=1e-6)
    assert comparison.ok, comparison.issues
    assert result.time_points == (0.0, 20.0)


def test_native_runner_tracks_vorticity_one_step_array_baseline() -> None:
    config = parse_bout_input(_VORTICITY_INPUT)
    result = run_config_case(
        config,
        case_name="vorticity_one_step",
        parity_mode="one_step",
        compare_variables=("Vort", "phi"),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/vorticity_one_step.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=2e-3, array_atol=1e-5)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_drift_wave_rhs_summary_baseline() -> None:
    config = parse_bout_input(_DRIFT_WAVE_INPUT)
    result = run_config_case(
        config,
        case_name="drift_wave_rhs",
        parity_mode="one_rhs",
        compare_variables=("Ni", "Ne", "Pe", "ddt(Ni)", "ddt(NVe)", "ddt(Vort)"),
        reference_case=ReferenceCase(
            name="drift_wave_rhs",
            stage="stage6",
            reference_path="tests/integrated/drift-wave/data/BOUT.inp",
            parity_mode="one_rhs",
            rationale="Coupled drift-wave RHS parity",
            compare_variables=("Ni", "Ne", "Pe", "ddt(Ni)", "ddt(NVe)", "ddt(Vort)"),
            extra_overrides=("i:diagnose=true", "e:diagnose=true", "vorticity:diagnose=true"),
            trim_x_guards=True,
            trim_y_guards=True,
        ),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/drift_wave_rhs.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=1e-6, scalar_atol=1e-6)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_drift_wave_rhs_array_baseline() -> None:
    config = parse_bout_input(_DRIFT_WAVE_INPUT)
    result = run_config_case(
        config,
        case_name="drift_wave_rhs",
        parity_mode="one_rhs",
        compare_variables=("Ni", "Ne", "Pe", "ddt(Ni)", "ddt(NVe)", "ddt(Vort)"),
        reference_case=ReferenceCase(
            name="drift_wave_rhs",
            stage="stage6",
            reference_path="tests/integrated/drift-wave/data/BOUT.inp",
            parity_mode="one_rhs",
            rationale="Coupled drift-wave RHS parity",
            compare_variables=("Ni", "Ne", "Pe", "ddt(Ni)", "ddt(NVe)", "ddt(Vort)"),
            extra_overrides=("i:diagnose=true", "e:diagnose=true", "vorticity:diagnose=true"),
            trim_x_guards=True,
            trim_y_guards=True,
        ),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/drift_wave_rhs.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=1e-6, array_atol=1e-6)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_drift_wave_one_step_summary_baseline() -> None:
    config = parse_bout_input(_DRIFT_WAVE_INPUT)
    result = run_config_case(
        config,
        case_name="drift_wave_one_step",
        parity_mode="one_step",
        compare_variables=("Ni", "Ne", "NVe", "Vort", "phi"),
        reference_case=ReferenceCase(
            name="drift_wave_one_step",
            stage="stage6",
            reference_path="tests/integrated/drift-wave/data/BOUT.inp",
            parity_mode="one_step",
            rationale="Single-output drift-wave parity",
            compare_variables=("Ni", "Ne", "NVe", "Vort", "phi"),
            trim_x_guards=True,
            trim_y_guards=True,
        ),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/drift_wave_one_step.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=5e-3, scalar_atol=5e-6)
    assert comparison.ok, comparison.issues
    assert result.time_points == (0.0, 10.0)


def test_native_runner_tracks_drift_wave_one_step_array_baseline() -> None:
    config = parse_bout_input(_DRIFT_WAVE_INPUT)
    result = run_config_case(
        config,
        case_name="drift_wave_one_step",
        parity_mode="one_step",
        compare_variables=("Ni", "Ne", "NVe", "Vort", "phi"),
        reference_case=ReferenceCase(
            name="drift_wave_one_step",
            stage="stage6",
            reference_path="tests/integrated/drift-wave/data/BOUT.inp",
            parity_mode="one_step",
            rationale="Single-output drift-wave parity",
            compare_variables=("Ni", "Ne", "NVe", "Vort", "phi"),
            trim_x_guards=True,
            trim_y_guards=True,
        ),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/drift_wave_one_step.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=5e-2, array_atol=5e-6)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_drift_wave_short_window_benchmark_scalars() -> None:
    config = parse_bout_input(_DRIFT_WAVE_INPUT)
    result = run_config_case(
        config,
        case_name="drift_wave_short_window",
        parity_mode="short_window",
        compare_variables=("Ni", "Ne", "NVe", "Vort", "phi"),
        reference_case=ReferenceCase(
            name="drift_wave_short_window",
            stage="stage6",
            reference_path="tests/integrated/drift-wave/data/BOUT.inp",
            parity_mode="short_window",
            rationale="Short-window drift-wave parity",
            compare_variables=("Ni", "Ne", "NVe", "Vort", "phi"),
            trim_x_guards=True,
            trim_y_guards=True,
        ),
    )
    analysis = analyze_drift_wave_array_payload(
        {
            "time_points": list(result.time_points),
            "variables": {name: np.asarray(value, dtype=np.float64) for name, value in result.variables.items()},
        },
        config=config,
        dataset_scalars=resolved_dataset_scalars(result.run_config),
        fit_points=10,
    )

    assert np.isclose(analysis.measured_gamma_over_wstar, 0.27478899792606437, rtol=1e-2, atol=2e-3)
    assert np.isclose(analysis.measured_omega_over_wstar, 0.23224315136107215, rtol=2e-2, atol=3e-3)
    assert result.time_points == tuple(10.0 * index for index in range(51))


def test_native_runner_tracks_drift_wave_short_window_arrays_with_documented_tolerances() -> None:
    config = parse_bout_input(_DRIFT_WAVE_INPUT)
    result = run_config_case(
        config,
        case_name="drift_wave_short_window",
        parity_mode="short_window",
        compare_variables=("Ni", "Ne", "NVe", "Vort", "phi"),
        reference_case=ReferenceCase(
            name="drift_wave_short_window",
            stage="stage6",
            reference_path="tests/integrated/drift-wave/data/BOUT.inp",
            parity_mode="short_window",
            rationale="Short-window drift-wave parity",
            compare_variables=("Ni", "Ne", "NVe", "Vort", "phi"),
            trim_x_guards=True,
            trim_y_guards=True,
        ),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/drift_wave_short_window.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    for name, max_allowed in {
        "Ni": 1.6e-3,
        "Ne": 1.6e-3,
        "NVe": 1.8e-4,
        "Vort": 2.2e-2,
        "phi": 4.5e-4,
    }.items():
        diff = np.asarray(actual["variables"][name], dtype=np.float64) - np.asarray(expected["variables"][name], dtype=np.float64)
        assert float(np.max(np.abs(diff))) < max_allowed, name


def test_native_runner_tracks_vorticity_short_window_summary_baseline() -> None:
    config = parse_bout_input(_VORTICITY_INPUT)
    result = run_config_case(
        config,
        case_name="vorticity_short_window",
        parity_mode="short_window",
        compare_variables=("Vort", "phi"),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/vorticity_short_window.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=2e-3, scalar_atol=1e-6)
    assert comparison.ok, comparison.issues
    assert result.time_points == tuple(20.0 * index for index in range(11))


def test_native_runner_tracks_vorticity_short_window_array_baseline() -> None:
    config = parse_bout_input(_VORTICITY_INPUT)
    result = run_config_case(
        config,
        case_name="vorticity_short_window",
        parity_mode="short_window",
        compare_variables=("Vort", "phi"),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/vorticity_short_window.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=2e-3, array_atol=1e-5)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_blob2d_rhs_summary_baseline() -> None:
    config = parse_bout_input(_BLOB2D_INPUT)
    result = run_config_case(
        config,
        case_name="blob2d_rhs",
        parity_mode="one_rhs",
        compare_variables=("Ne", "Pe", "phi", "ddt(Ne)", "ddt(Vort)"),
        reference_case=ReferenceCase(
            name="blob2d_rhs",
            stage="stage6",
            reference_path="examples/other/blob2d/BOUT.inp",
            parity_mode="one_rhs",
            rationale="Blob RHS parity",
            compare_variables=("Ne", "Pe", "phi", "ddt(Ne)", "ddt(Vort)"),
            extra_overrides=("e:diagnose=true", "vorticity:diagnose=true"),
            trim_x_guards=True,
        ),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/blob2d_rhs.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=1e-12, scalar_atol=1e-12)
    assert comparison.ok, comparison.issues
    assert result.time_points == (0.0,)


def test_native_runner_tracks_blob2d_rhs_array_baseline() -> None:
    config = parse_bout_input(_BLOB2D_INPUT)
    result = run_config_case(
        config,
        case_name="blob2d_rhs",
        parity_mode="one_rhs",
        compare_variables=("Ne", "Pe", "phi", "ddt(Ne)", "ddt(Vort)"),
        reference_case=ReferenceCase(
            name="blob2d_rhs",
            stage="stage6",
            reference_path="examples/other/blob2d/BOUT.inp",
            parity_mode="one_rhs",
            rationale="Blob RHS parity",
            compare_variables=("Ne", "Pe", "phi", "ddt(Ne)", "ddt(Vort)"),
            extra_overrides=("e:diagnose=true", "vorticity:diagnose=true"),
            trim_x_guards=True,
        ),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/blob2d_rhs.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=1e-12, array_atol=1e-12)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_blob2d_one_step_summary_baseline() -> None:
    config = parse_bout_input(_BLOB2D_INPUT)
    result = run_config_case(
        config,
        case_name="blob2d_one_step",
        parity_mode="one_step",
        compare_variables=("Ne", "Vort", "phi"),
        reference_case=ReferenceCase(
            name="blob2d_one_step",
            stage="stage6",
            reference_path="examples/other/blob2d/BOUT.inp",
            parity_mode="one_step",
            rationale="Single-output blob parity",
            compare_variables=("Ne", "Vort", "phi"),
            trim_x_guards=True,
        ),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/blob2d_one_step.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=2e-3, scalar_atol=2e-6)
    assert comparison.ok, comparison.issues
    assert result.time_points == (0.0, 50.0)


def test_native_runner_tracks_blob2d_one_step_array_baseline() -> None:
    config = parse_bout_input(_BLOB2D_INPUT)
    result = run_config_case(
        config,
        case_name="blob2d_one_step",
        parity_mode="one_step",
        compare_variables=("Ne", "Vort", "phi"),
        reference_case=ReferenceCase(
            name="blob2d_one_step",
            stage="stage6",
            reference_path="examples/other/blob2d/BOUT.inp",
            parity_mode="one_step",
            rationale="Single-output blob parity",
            compare_variables=("Ne", "Vort", "phi"),
            trim_x_guards=True,
        ),
    )
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/blob2d_one_step.npz")
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=2e-3, array_atol=2e-6)
    assert comparison.ok, comparison.issues


def test_native_runner_tracks_blob2d_short_window_summary_and_blob_metrics() -> None:
    config = parse_bout_input(_BLOB2D_INPUT)
    result = run_config_case(
        config,
        case_name="blob2d_short_window",
        parity_mode="short_window",
        compare_variables=("Ne", "Vort", "phi"),
        reference_case=ReferenceCase(
            name="blob2d_short_window",
            stage="stage6",
            reference_path="examples/other/blob2d/BOUT.inp",
            parity_mode="short_window",
            rationale="2D blob convection with recalc-metric path and sheath closure.",
            compare_variables=("Ne", "Vort", "phi"),
            trim_x_guards=True,
        ),
    )
    expected_summary = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/blob2d_short_window.json")
    )
    summary_comparison = compare_summary_payloads(expected_summary, result.payload, scalar_rtol=2e-2, scalar_atol=2e-6)
    assert summary_comparison.ok, summary_comparison.issues
    assert result.time_points == tuple(50.0 * index for index in range(51))

    expected_analysis = load_blob2d_analysis_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_metrics/blob2d_short_window_metrics.json")
    )
    actual_arrays = build_array_payload_from_summary_payload(result.payload, result.variables)
    actual_analysis = analyze_blob2d_array_payload(actual_arrays)
    blob_metrics = compare_blob2d_analysis_results(expected_analysis, actual_analysis)

    assert blob_metrics.peak_max_abs_error < 1.5e-2
    assert blob_metrics.peak_rms_error < 5.0e-3
    assert blob_metrics.center_of_mass_x_max_abs_error < 0.7
    assert blob_metrics.center_of_mass_z_max_abs_error < 0.8


def test_native_runner_tracks_neutral_mixed_rhs_summary_baseline() -> None:
    config = parse_bout_input(_NEUTRAL_MIXED_INPUT)
    result = run_config_case(
        config,
        case_name="neutral_mixed_rhs",
        parity_mode="one_rhs",
        compare_variables=("Nh", "Ph", "NVh", "ddt(Nh)", "ddt(Ph)", "ddt(NVh)"),
        reference_case=ReferenceCase(
            name="neutral_mixed_rhs",
            stage="stage7",
            reference_path="tests/integrated/neutral_mixed/data/BOUT.inp",
            parity_mode="one_rhs",
            rationale="Mixed-neutral RHS parity before transient neutral transport checks.",
            compare_variables=("Nh", "Ph", "NVh", "ddt(Nh)", "ddt(Ph)", "ddt(NVh)"),
            trim_y_guards=True,
        ),
    )
    expected = load_summary_json(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference/neutral_mixed_rhs.json")
    )

    comparison = compare_summary_payloads(expected, result.payload, scalar_rtol=5e-2, scalar_atol=2e-6)
    assert comparison.ok, comparison.issues
    assert result.time_points == (0.0,)


def test_native_runner_tracks_neutral_mixed_rhs_array_baseline() -> None:
    config = parse_bout_input(_NEUTRAL_MIXED_INPUT)
    result = run_config_case(
        config,
        case_name="neutral_mixed_rhs",
        parity_mode="one_rhs",
        compare_variables=("Nh", "Ph", "NVh", "ddt(Nh)", "ddt(Ph)", "ddt(NVh)"),
        reference_case=ReferenceCase(
            name="neutral_mixed_rhs",
            stage="stage7",
            reference_path="tests/integrated/neutral_mixed/data/BOUT.inp",
            parity_mode="one_rhs",
            rationale="Mixed-neutral RHS parity before transient neutral transport checks.",
            compare_variables=("Nh", "Ph", "NVh", "ddt(Nh)", "ddt(Ph)", "ddt(NVh)"),
            trim_y_guards=True,
        ),
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    expected = load_portable_array_payload(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/neutral_mixed_rhs.npz")
    )
    metadata_comparison = compare_array_payloads(
        expected,
        actual,
        scalar_rtol=5e-2,
        scalar_atol=2e-6,
        array_rtol=0.0,
        array_atol=0.0,
    )
    metadata_issues = tuple(issue for issue in metadata_comparison.issues if not issue.field.startswith("variables."))
    assert not metadata_issues, metadata_issues

    density_error = np.asarray(actual["variables"]["ddt(Nh)"]) - np.asarray(expected["variables"]["ddt(Nh)"])
    pressure_error = np.asarray(actual["variables"]["ddt(Ph)"]) - np.asarray(expected["variables"]["ddt(Ph)"])
    momentum_error = np.asarray(actual["variables"]["ddt(NVh)"]) - np.asarray(expected["variables"]["ddt(NVh)"])

    assert np.max(np.abs(density_error)) < 8.0e-3
    assert np.sqrt(np.mean(np.square(density_error))) < 3.5e-3
    assert np.max(np.abs(pressure_error)) < 1.5e-3
    assert np.sqrt(np.mean(np.square(pressure_error))) < 6.0e-4
    assert np.max(np.abs(momentum_error)) < 1.0e-12
