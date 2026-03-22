from __future__ import annotations

from pathlib import Path

import numpy as np

from jax_drb.native import run_curated_case
from jax_drb.parity.arrays import (
    build_array_payload_from_summary_payload,
    compare_array_payloads,
    load_portable_array_payload,
)
from jax_drb.parity.compare import compare_summary_payloads, load_summary_json
from jax_drb.native.recycling_1d import _load_amjuel_rate


_REFERENCE_ROOT = Path("/Users/rogerio/local/hermes-3")
_BASELINE_DIR = Path("/Users/rogerio/local/jax_drb/references/baselines/reference")
_ARRAY_BASELINE_DIR = Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays")


def test_amjuel_rate_tables_are_packaged_for_recycling_branch() -> None:
    hydrogen_iz_coeffs, hydrogen_iz_energy_coeffs, hydrogen_iz_heating = _load_amjuel_rate("d", "iz")
    helium_rec_coeffs, helium_rec_energy_coeffs, helium_rec_heating = _load_amjuel_rate("he", "rec")

    assert hydrogen_iz_coeffs.shape == (9, 9)
    assert helium_rec_coeffs.shape == (9, 9)
    assert hydrogen_iz_energy_coeffs.shape == (9, 9)
    assert helium_rec_energy_coeffs.shape == (9, 9)
    assert np.isfinite(hydrogen_iz_heating)
    assert np.isfinite(helium_rec_heating)


def test_recycling_1d_rhs_matches_summary_baseline() -> None:
    expected = load_summary_json(_BASELINE_DIR / "recycling_1d_rhs.json")
    actual = run_curated_case("recycling_1d_rhs", reference_root=_REFERENCE_ROOT).payload

    comparison = compare_summary_payloads(expected, actual, scalar_rtol=1.0e-6, scalar_atol=1.0e-9)

    assert comparison.ok, comparison.issues


def test_recycling_1d_rhs_matches_array_baseline() -> None:
    expected = load_portable_array_payload(_ARRAY_BASELINE_DIR / "recycling_1d_rhs.npz")
    result = run_curated_case("recycling_1d_rhs", reference_root=_REFERENCE_ROOT)
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=1.0e-6, array_atol=1.0e-9)

    assert comparison.ok, comparison.issues
