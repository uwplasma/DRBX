"""The TOML deck and the script options resolve to one HsxBlobConfig."""

from dataclasses import asdict
from pathlib import Path

import pytest

from drbx.cli import main as drbx_main
from drbx.fci_braginskii.run import HsxBlobConfig, _build_parser, load_hsx_blob_deck

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "inputs" / "hsx_fci_blob.toml"


def test_parser_defaults_are_the_config_defaults():
    parsed = vars(_build_parser().parse_args(()))
    assert parsed == asdict(HsxBlobConfig())


def test_example_deck_matches_equivalent_command_line():
    deck = load_hsx_blob_deck(EXAMPLE)
    argv = (
        "--geometry", str(deck.geometry), "--final-time", "0.00375",
        "--num-steps", "5", "--save-every", "5", "--output", str(deck.output),
    )
    assert deck == HsxBlobConfig(**vars(_build_parser().parse_args(argv)))


@pytest.mark.parametrize(
    "body, message",
    [
        ('geometry = "g"\nbogus = 1\n', "unknown"),
        ('geometry = "g"\nnum_steps = 2.5\n', "num_steps must be int"),
        ('geometry = "g"\nblob_center = [0.1]\n', "blob_center"),
        ("num_steps = 2\n", "geometry is required"),
        ('geometry = "g"\nwall_closure = "mpe"\n', "wall_closure"),
        ('geometry = "g"\nion_heat_transmission = 1.0\n', "ion_heat_transmission"),
    ],
)
def test_deck_errors(tmp_path, body, message):
    deck = tmp_path / "deck.toml"
    deck.write_text('[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\n' + body)
    with pytest.raises(ValueError, match=message):
        load_hsx_blob_deck(deck)


def test_drbx_inspect_prints_resolved_config(capsys):
    assert drbx_main(["inspect", str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "backend: fci_braginskii" in out
    assert "num_steps = 5" in out and "rho_star = 0.0005" in out


def test_wall_closure_deck_keys(tmp_path):
    from drbx.fci_braginskii.run import sheath_closure

    assert HsxBlobConfig().wall_closure == "bohm"
    deck = tmp_path / "deck.toml"
    deck.write_text(
        '[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\ngeometry = "g"\n'
        'wall_closure = "zero-flow"\n'
    )
    assert sheath_closure(load_hsx_blob_deck(deck)) is None
    deck.write_text(
        '[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\ngeometry = "g"\n'
        "electron_heat_transmission = 5.0\nsecondary_electron_emission = 0.1\n"
    )
    closure = sheath_closure(load_hsx_blob_deck(deck))
    assert closure.electron_heat_transmission == 5.0
    assert closure.secondary_electron_emission == 0.1
    args = _build_parser().parse_args(("--wall-closure", "zero-flow"))
    assert sheath_closure(args) is None
