from __future__ import annotations

from pathlib import Path

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native import run_config_case
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
