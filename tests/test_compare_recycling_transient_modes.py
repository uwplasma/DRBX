from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def _load_compare_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_recycling_transient_modes.py"
    spec = importlib.util.spec_from_file_location("compare_recycling_transient_modes", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare_script = _load_compare_script()


def test_parser_accepts_and_documents_fixed_full_field_jvp_mode() -> None:
    args = compare_script._parse_args(
        [
            "--reference-root",
            "/tmp/reference",
            "--mode",
            "bdf_fixed_full_field_jvp",
        ]
    )

    assert args.reference_root == Path("/tmp/reference")
    assert args.modes == ["bdf_fixed_full_field_jvp"]
    help_text = compare_script._build_parser().format_help()
    normalized_help = " ".join(help_text.split()).replace("full- field", "full-field")
    assert "bdf_fixed_full_field_jvp" in help_text
    assert "fixed full-field JVP BDF path" in normalized_help


def test_default_modes_include_fixed_full_field_jvp_after_bdf() -> None:
    one_step_modes = compare_script._default_modes("recycling_1d_one_step")
    dthe_modes = compare_script._default_modes("recycling_dthe_one_step")

    assert one_step_modes == (
        "continuation",
        "bdf",
        "bdf_fixed_full_field_jvp",
        "adaptive_be",
        "adaptive_bdf",
    )
    assert dthe_modes == ("bdf", "bdf_fixed_full_field_jvp", "adaptive_be", "adaptive_bdf")


def test_bdf_pairwise_delta_report_formats_worst_field_first() -> None:
    mode_variables = {
        "bdf": {
            "Nd+": np.asarray([1.0, 2.0]),
            "Pd+": np.asarray([10.0, 15.0]),
        },
        "bdf_fixed_full_field_jvp": {
            "Nd+": np.asarray([1.25, 2.5]),
            "Pd+": np.asarray([13.0, 15.0]),
        },
    }

    lines = compare_script._format_bdf_pairwise_delta_report(
        mode_variables,
        fields=("Nd+", "Pd+"),
    )

    assert lines == [
        "pairwise_delta=bdf_vs_bdf_fixed_full_field_jvp",
        "  Pd+: max_abs_delta=3.00000000e+00",
        "  Nd+: max_abs_delta=5.00000000e-01",
        "  worst=Pd+ delta=3.00000000e+00",
    ]


def test_mode_diagnostics_report_formats_sorted_values() -> None:
    lines = compare_script._format_mode_diagnostics_report(
        "bdf_fixed_full_field_jvp",
        {
            "bdf_rhs_backend": "fixed_full_field_array",
            "bdf_jacobian_callback_seconds": 0.125,
            "bdf_jacobian_base_rhs_evaluation_count": 0,
        },
    )

    assert lines == [
        "diagnostics mode=bdf_fixed_full_field_jvp",
        "  bdf_jacobian_base_rhs_evaluation_count=0",
        "  bdf_jacobian_callback_seconds=1.25000000e-01",
        "  bdf_rhs_backend=fixed_full_field_array",
    ]


def test_bdf_pairwise_delta_report_crops_both_outputs_to_active_mesh() -> None:
    mesh = SimpleNamespace(xstart=1, xend=2, ystart=0, yend=1)
    bdf = np.zeros((1, 4, 3, 1))
    fixed_jvp = np.zeros((1, 4, 3, 1))
    fixed_jvp[:, 0, 0, :] = 99.0
    fixed_jvp[:, 2, 1, :] = 0.75

    lines = compare_script._format_bdf_pairwise_delta_report(
        {
            "bdf": {"Nd+": bdf},
            "bdf_fixed_full_field_jvp": {"Nd+": fixed_jvp},
        },
        fields=("Nd+",),
        mesh=mesh,
    )

    assert lines == [
        "pairwise_delta=bdf_vs_bdf_fixed_full_field_jvp",
        "  Nd+: max_abs_delta=7.50000000e-01",
        "  worst=Nd+ delta=7.50000000e-01",
    ]


def test_bdf_pairwise_delta_report_is_omitted_without_both_modes() -> None:
    lines = compare_script._format_bdf_pairwise_delta_report(
        {"bdf": {"Nd+": np.asarray([1.0])}},
        fields=("Nd+",),
    )

    assert lines == []
