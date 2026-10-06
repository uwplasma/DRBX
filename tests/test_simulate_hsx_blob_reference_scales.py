"""rho-star and reference-scale options of the HSX blob driver."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drbx.runtime.reference_scales import reference_rho_star

REPOSITORY = Path(__file__).resolve().parents[1]
DRIVER_PATH = REPOSITORY / "simulate_hsx_blob.py"
B0 = 1.02


def _driver_module():
    spec = importlib.util.spec_from_file_location("hsx_driver_reference_scales", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hsx():
    return _driver_module()


def _stub_main_environment(hsx, monkeypatch, metadata):
    artifact = SimpleNamespace(
        global_geometry=SimpleNamespace(shape=(4, 8, 12)),
        cell_positions=np.zeros((4, 8, 12, 3), dtype=np.float64),
        nfp=1,
        topology=SimpleNamespace(name="square"),
        curvature_edge_one_form=None,
        owner_geometry=None,
        owner_overlap=None,
        metadata=metadata,
    )
    monkeypatch.setattr(hsx, "load_fci_simulation_geometry", lambda path: artifact)
    monkeypatch.setattr(hsx, "make_shard_mesh", lambda counts: object())
    monkeypatch.setattr(
        hsx,
        "build_local_fci_geometries",
        lambda *args, **kwargs: SimpleNamespace(
            global_shape=(4, 8, 12),
            shard_counts=(1, 1, 1),
            maps_valid=True,
            map_fields=None,
            domain=SimpleNamespace(
                layout=SimpleNamespace(owned_shape=(4, 8, 12), cell_halo_shape=(6, 10, 14)),
                periodic_axes=(False, False, True),
                axis_regular_axes=(False, False, False),
            ),
        ),
    )


def _parse(hsx, *argv):
    return hsx._build_parser().parse_args(list(argv))


def test_parser_defaults_keep_explicit_rho_star_behaviour(hsx):
    args = _parse(hsx)
    assert args.rho_star == 1.0
    assert args.rho_star_explicit is False
    assert args.reference_te_ev is None
    assert args.ion_mass_amu == 1.00728


def test_explicit_rho_star_is_recorded_as_explicit(hsx):
    args = _parse(hsx, "--rho-star", "0.05")
    assert args.rho_star == 0.05 and args.rho_star_explicit is True
    rho_star, record = hsx._resolve_reference_scales(
        args, {"reference_magnetic_field_tesla": B0, "length_unit": "m"}
    )
    assert rho_star == 0.05
    assert record["rho_star_source"] == "explicit"
    assert record["reference_te_ev"] is None and record["c_s0_m_per_s"] is None
    assert record["reference_magnetic_field_tesla"] == B0
    # Te_ref that an explicit rho_star implies on the artifact's B0 scale.
    implied = record["implied_reference_te_ev"]
    assert reference_rho_star(implied, B0).rho_star == pytest.approx(0.05, rel=1e-9)


def test_default_rho_star_one_resolves_without_b0(hsx):
    rho_star, record = hsx._resolve_reference_scales(_parse(hsx), {})
    assert rho_star == 1.0
    assert record["rho_star_source"] == "default"
    assert record["reference_magnetic_field_tesla"] is None
    assert record["implied_reference_te_ev"] is None
    assert record["length_unit_source"] == "assumed-legacy-artifact"


@pytest.mark.parametrize("ion_args,mass", [((), 1.00728), (("--ion-mass-amu", "2.013553"), 2.013553)])
def test_derivation_from_te_and_artifact_b0(hsx, ion_args, mass):
    args = _parse(hsx, "--reference-te-ev", "20", *ion_args)
    metadata = {"reference_magnetic_field_tesla": B0, "length_unit": "m"}
    rho_star, record = hsx._resolve_reference_scales(args, metadata)
    expected = reference_rho_star(20.0, B0, ion_mass_amu=mass)
    assert rho_star == pytest.approx(expected.rho_star, rel=1e-14)
    assert record["rho_star_source"] == "derived"
    assert record["reference_te_ev"] == 20.0
    assert record["reference_magnetic_field_tesla"] == B0
    assert record["ion_mass_amu"] == mass
    assert record["l_ref_m"] == 1.0
    assert record["c_s0_m_per_s"] == pytest.approx(expected.c_s0)
    assert record["t_ref_s"] == pytest.approx(expected.t_ref)
    assert record["rho_s0_m"] == pytest.approx(expected.rho_s0_m)
    if not ion_args:
        assert rho_star == pytest.approx(4.5e-4, rel=0.02)


def test_derivation_requires_artifact_b0(hsx):
    args = _parse(hsx, "--reference-te-ev", "20")
    for metadata in (
        {},
        {"reference_magnetic_field": None},
        {"reference_magnetic_field_tesla": None},
        {"reference_magnetic_field_tesla": float("nan")},
        {"reference_magnetic_field_tesla": -1.0},
    ):
        with pytest.raises(ValueError, match="reference_magnetic_field_tesla"):
            hsx._resolve_reference_scales(args, metadata)


def test_derivation_rejects_non_metre_artifacts(hsx):
    args = _parse(hsx, "--reference-te-ev", "20")
    with pytest.raises(ValueError, match="length_unit"):
        hsx._resolve_reference_scales(
            args, {"reference_magnetic_field_tesla": B0, "length_unit": "cm"}
        )


@pytest.mark.parametrize("flag,value", [("--reference-te-ev", "0"), ("--reference-te-ev", "-3"), ("--ion-mass-amu", "0")])
def test_nonpositive_reference_options_are_rejected(hsx, flag, value):
    args = _parse(hsx, flag, value)
    with pytest.raises(ValueError, match="positive"):
        hsx._validate_rho_star_options(args)


def test_rho_star_and_reference_te_are_mutually_exclusive(hsx):
    # Even an explicit value equal to the default conflicts.
    for rho in ("1.0", "0.01"):
        args = _parse(hsx, "--rho-star", rho, "--reference-te-ev", "20")
        with pytest.raises(ValueError, match="mutually exclusive"):
            hsx._validate_rho_star_options(args)
        with pytest.raises(ValueError, match="mutually exclusive"):
            hsx._resolve_reference_scales(args, {"reference_magnetic_field_tesla": B0})


def test_main_rejects_conflicting_options_before_loading_geometry(hsx, monkeypatch, tmp_path, capsys):
    def forbidden(path):
        raise AssertionError("geometry was loaded despite conflicting options")

    monkeypatch.setattr(hsx, "load_fci_simulation_geometry", forbidden)
    with pytest.raises(SystemExit) as error:
        hsx.main(
            ["--geometry", str(tmp_path), "--rho-star", "0.1", "--reference-te-ev", "20"]
        )
    assert error.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_main_fails_clearly_when_artifact_lacks_b0(hsx, monkeypatch, tmp_path, capsys):
    _stub_main_environment(hsx, monkeypatch, metadata={"reference_magnetic_field": None})
    with pytest.raises(SystemExit) as error:
        hsx.main(["--geometry", str(tmp_path), "--geometry-only", "--reference-te-ev", "20"])
    assert error.value.code == 2
    err = capsys.readouterr().err
    assert "reference_magnetic_field_tesla" in err and "B0" in err


def test_main_logs_derived_scales(hsx, monkeypatch, tmp_path, capsys):
    _stub_main_environment(
        hsx,
        monkeypatch,
        metadata={"reference_magnetic_field_tesla": B0, "length_unit": "m"},
    )
    hsx.main(
        [
            "--geometry",
            str(tmp_path),
            "--geometry-only",
            "--reference-te-ev",
            "20",
        ]
    )
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("[simulation] reference scales:"))
    expected = reference_rho_star(20.0, B0)
    assert f"rho_star={expected.rho_star:.6e}" in line
    assert "source=derived" in line and "convention" not in line
    assert "Te_ref=2.000000e+01 eV" in line
    assert f"B0={B0:.6e} T" in line
    assert f"c_s0={expected.c_s0:.6e} m/s" in line
    assert f"t_ref={expected.t_ref:.6e} s" in line
    assert "L_ref=1.000000e+00 m" in line and "m_i=1.007280 u" in line


def test_main_default_run_logs_explicit_default_without_b0(hsx, monkeypatch, tmp_path, capsys):
    _stub_main_environment(hsx, monkeypatch, metadata={})
    hsx.main(["--geometry", str(tmp_path), "--geometry-only"])
    line = next(
        l for l in capsys.readouterr().out.splitlines()
        if l.startswith("[simulation] reference scales:")
    )
    assert "rho_star=1.000000e+00 (source=default)" in line
    assert "B0=unknown" in line and "Te_ref=unknown" in line


def test_parameters_and_run_metadata_are_plumbed_by_name():
    source = DRIVER_PATH.read_text()
    assert "rho_star=float(rho_star_value)" in source
    assert '"rho_star_normalization": "single-length"' in source
    assert '"reference_scales": dict(reference_scales_record)' in source
    assert "args.rho_star)" not in source.split("def main(")[1]
