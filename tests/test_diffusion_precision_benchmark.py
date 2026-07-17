from __future__ import annotations

import pytest

from drbx.config import rewrite_input_precision


def test_rewrite_input_precision_rewrites_runtime_precision_only() -> None:
    template = """
    [time]
    nout = 3
    timestep = 5.0

    [runtime]
    precision = "float64"

    [mesh]
    nx = 16
    """.strip()

    updated = rewrite_input_precision(template, "float32")

    assert 'precision = "float32"' in updated
    assert 'precision = "float64"' not in updated
    assert "nx = 16" in updated


def test_rewrite_input_precision_requires_a_precision_entry() -> None:
    with pytest.raises(ValueError, match="precision entry"):
        rewrite_input_precision("[time]\nnout = 1\n", "float32")


def test_precision_benchmark_example_uses_the_config_helper() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "examples" / "diffusion_precision_benchmark.py"
    ).read_text(encoding="utf-8")
    assert "rewrite_input_precision" in source
