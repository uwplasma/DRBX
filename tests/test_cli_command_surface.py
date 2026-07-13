from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from jax_drb.cli import (
    _default_command,
    _inspect_command,
    _normalize_cli_argv,
    main,
)


def test_default_command_and_argv_normalization_errors_are_explicit() -> None:
    assert _normalize_cli_argv([]) == []
    assert _normalize_cli_argv(["inspect", "case.inp"]) == ["inspect", "case.inp"]
    assert _normalize_cli_argv(["run", "case.inp"]) == ["run", "case.inp"]
    assert _normalize_cli_argv(["--help"]) == ["--help"]
    assert _normalize_cli_argv(["case.inp", "--dry-run"]) == [
        "run",
        "case.inp",
        "--dry-run",
    ]

    assert (
        _default_command(argparse.Namespace(subcommand="demo", command=lambda args: 42))
        == 42
    )
    with pytest.raises(SystemExit):
        _default_command(argparse.Namespace(subcommand=None))
    with pytest.raises(SystemExit):
        main([])


def test_inspect_command_reports_resolved_normalization(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "normalized.inp"
    input_path.write_text(
        """
        nout = 1
        timestep = 1

        [mesh]
        nx = 2
        ny = 2
        nz = 1
        dx = 1
        dy = 1
        dz = 1
        J = 1

        [model]
        components = e
        Nnorm = 1e19
        Tnorm = 100
        Bnorm = 2

        [e]
        type = evolve_density
        charge = -1
        AA = 1

        [Ne]
        function = 1
        """,
        encoding="utf-8",
    )

    assert _inspect_command(argparse.Namespace(input_file=input_path)) == 0
    output = capsys.readouterr().out
    assert "normalization:" in output
    assert "Nnorm=1e+19" in output
