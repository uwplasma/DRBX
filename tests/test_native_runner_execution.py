from __future__ import annotations

from types import SimpleNamespace

from jax_drb.native.runner_execution import (
    effective_output_steps,
    effective_overrides,
    restart_variable_names,
)


def test_effective_output_steps_respects_parity_mode() -> None:
    assert effective_output_steps("one_rhs", configured_nout=9) == 0
    assert effective_output_steps("one_step", configured_nout=9) == 1
    assert effective_output_steps("short_window", configured_nout=9) == 9


def test_effective_overrides_merges_default_and_case_specific_values() -> None:
    reference_case = SimpleNamespace(extra_overrides=("mesh:nx=64",))
    overrides = effective_overrides("one_step", reference_case=reference_case)
    assert "mesh:nx=64" in overrides
    assert any("nout=" in override for override in overrides)


def test_restart_variable_names_cover_promoted_placeholder_families() -> None:
    diffusion = SimpleNamespace(components=[SimpleNamespace(types=("evolve_density", "evolve_pressure", "anomalous_diffusion"), section="h")])
    mms = SimpleNamespace(components=[SimpleNamespace(types=("evolve_density", "evolve_pressure", "evolve_momentum"), section="i")])
    vort = SimpleNamespace(components=[SimpleNamespace(types=("vorticity",), section="vorticity")])
    blob = SimpleNamespace(components=[SimpleNamespace(types=("blob2d",), section="blob2d")])
    drift = SimpleNamespace(components=[SimpleNamespace(types=("drift_wave",), section="drift_wave")])

    assert restart_variable_names(diffusion) == ("Nh", "Ph")
    assert restart_variable_names(mms) == ("Ni", "Pi", "NVi")
    assert restart_variable_names(vort) == ("Vort",)
    assert restart_variable_names(blob) == ("Ne", "Vort")
    assert restart_variable_names(drift) == ("Ni", "NVe", "Vort")
