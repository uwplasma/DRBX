from __future__ import annotations

from jax_drb.config.boutinp import parse_toml_input
from jax_drb.runtime import resolve_runtime_precision


def test_resolve_runtime_precision_reads_runtime_section() -> None:
    config = parse_toml_input(
        """
        [time]
        nout = 1
        timestep = 1.0

        [runtime]
        precision = "float32"
        """
    )

    assert resolve_runtime_precision(config=config) == "float32"


def test_resolve_runtime_precision_request_overrides_config() -> None:
    config = parse_toml_input(
        """
        [time]
        nout = 1
        timestep = 1.0

        [runtime]
        precision = "float32"
        """
    )

    assert resolve_runtime_precision(requested="float64", config=config) == "float64"
