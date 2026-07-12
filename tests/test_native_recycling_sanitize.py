from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.recycling_1d import _build_recycling_runtime_model, _build_recycling_state_fields
from jax_drb.native.recycling_sanitize import sanitize_recycling_fields
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.reference.paths import default_reference_root


_REFERENCE_ROOT = default_reference_root()
_REFERENCE_BASE = _REFERENCE_ROOT if _REFERENCE_ROOT is not None else Path("/nonexistent-reference-root")
_INPUT_1D = _REFERENCE_BASE / "tests/integrated/1D-recycling/data/BOUT.inp"


def test_neutral_pressure_default_floor_is_zero_without_override() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    fields = _build_recycling_state_fields(runtime_model)
    fields["Nd"][:] = 1.7e-2
    fields["Pd"][:] = 1.0e-5
    fields["Nd+"][:] = 1.7e-2
    fields["Pd+"][:] = 1.0e-5

    sanitized = sanitize_recycling_fields(config, fields)

    assert np.allclose(sanitized["Pd"], 1.0e-5)
    assert np.allclose(sanitized["Pd+"], 1.7e-3)


def test_electron_pressure_floor_uses_charge_weighted_ion_density() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = apply_bout_overrides(load_bout_input(_INPUT_1D), ("e:temperature_floor=0.25",))
    shape = (1, 3, 1)
    fields = {
        "Nd+": np.full(shape, 2.0e-2, dtype=np.float64),
        "Pd+": np.full(shape, 1.0e-6, dtype=np.float64),
        "Ne": np.full(shape, 1.0e-8, dtype=np.float64),
        "Pe": np.full(shape, 1.0e-6, dtype=np.float64),
    }

    sanitized = sanitize_recycling_fields(config, fields)

    np.testing.assert_allclose(sanitized["Pe"], 0.25 * np.full(shape, 2.0e-2, dtype=np.float64))
