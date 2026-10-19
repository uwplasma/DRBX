"""Every model runs from a TOML deck: example decks, outputs, schema errors, restart."""

import json
from pathlib import Path

import numpy as np
import pytest

from drbx.cli import main as drbx_main
from drbx.deck import BACKENDS, load_deck
from tests.docs.test_deck_reference import documented_outputs

INPUTS = Path(__file__).resolve().parents[1] / "examples" / "inputs"
REDUCED = sorted(set(BACKENDS) - {"fci_braginskii"})


def deck(tmp_path, body, name="deck.toml"):
    path = tmp_path / name
    path.write_text(body)
    return path


@pytest.mark.parametrize("backend", REDUCED)
def test_example_deck_runs_and_writes_the_documented_outputs(backend, tmp_path):
    example = INPUTS / f"{backend}.toml"
    assert drbx_main(["run", str(example), "--output-dir", str(tmp_path), "--quiet"]) == 0
    summary = json.loads((tmp_path / f"{backend}_summary.json").read_text())
    assert summary["backend"] == backend and summary["case_name"] == backend
    arrays_documented, results_documented = documented_outputs(backend)
    with np.load(tmp_path / f"{backend}_arrays.npz") as data:
        assert set(data.files) == arrays_documented
        assert all(np.all(np.isfinite(data[name])) for name in data.files if data[name].dtype.kind == "f")
    assert set(summary["results"]) == results_documented


def test_hsx_example_deck_resolves_and_inspect_prints_it(capsys):
    assert drbx_main(["inspect", str(INPUTS / "hsx_fci_blob.toml")]) == 0
    assert "gmres_preconditioner = 'line-u'" in capsys.readouterr().out


def test_hsx_gmres_preconditioner_is_a_deck_key(tmp_path):
    body = '[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\ngeometry = "g"\n'
    assert load_deck(deck(tmp_path, body + 'gmres_preconditioner = "none"\n'))[1]["fci_braginskii"].gmres_preconditioner == "none"
    with pytest.raises(ValueError, match="gmres_preconditioner must be"):
        load_deck(deck(tmp_path, body + 'gmres_preconditioner = "jacobi"\n'))


@pytest.mark.parametrize(
    "body, message",
    [
        ('[model]\nbackend = "nope"\n', "backend must be one of"),
        ('[model]\nbackend = "hasegawa_wakatani"\n[hasegawa_wakatani]\nalpha = 1\n', r"unknown \[hasegawa_wakatani\] key.*valid keys.*adiabaticity"),
        ('[model]\nbackend = "hasegawa_wakatani"\n[bogus]\n', "unknown table"),
        ('[model]\nbackend = "hasegawa_wakatani"\n[grid]\nn = 2.5\n', "n must be int"),
        ('[model]\nbackend = "sol_flux_tube"\n', r"\[open_slab\] shape is required"),
        ('[model]\nbackend = "stellarator_turbulence"\n[run]\nsteps = 1\ndt = 1\n', "exactly one of"),
        ('[model]\nbackend = "sol_recycling"\n[sol_recycling.normalization]\nTnorm = "hot"\n', "Tnorm must be float"),
    ],
)
def test_schema_errors(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        load_deck(deck(tmp_path, body))


def test_cli_reports_deck_errors_without_a_traceback(tmp_path):
    path = deck(tmp_path, '[model]\nbackend = "hasegawa_wakatani"\n[grid]\nm = 1\n')
    with pytest.raises(SystemExit, match="unknown"):
        drbx_main(["run", str(path)])


def test_restart_from_continues_the_run(tmp_path):
    base = (
        '[model]\nbackend = "hasegawa_wakatani"\n[grid]\nn = 16\n'
        "[run]\nsteps_per_block = 20\n"
    )
    full = deck(tmp_path, base + "blocks = 4\n", "full.toml")
    first = deck(tmp_path, base + "blocks = 2\n", "first.toml")
    drbx_main(["run", str(full), "--output-dir", str(tmp_path), "--quiet"])
    drbx_main(["run", str(first), "--output-dir", str(tmp_path), "--quiet"])
    second = deck(tmp_path, base + 'blocks = 2\nrestart_from = "first_arrays.npz"\n', "second.toml")
    drbx_main(["run", str(second), "--output-dir", str(tmp_path), "--quiet"])
    with np.load(tmp_path / "full_arrays.npz") as a, np.load(tmp_path / "second_arrays.npz") as b:
        np.testing.assert_allclose(b["vorticity"], a["vorticity"], rtol=1e-9, atol=1e-12)
        assert b["time"][-1] == pytest.approx(a["time"][-1])
