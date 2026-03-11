from __future__ import annotations

from pathlib import Path

from jax_drb.reference.cases import load_reference_cases, resolve_reference_cases


def test_reference_case_manifest_loads_and_resolves(tmp_path: Path) -> None:
    manifest = tmp_path / "cases.toml"
    reference_root = tmp_path / "source-checkout"
    case_dir = reference_root / "tests" / "integrated" / "toy" / "data"
    case_dir.mkdir(parents=True)
    manifest.write_text(
        """
        [[case]]
        name = "toy_case"
        stage = "stage2"
        reference_path = "tests/integrated/toy/data/BOUT.inp"
        parity_mode = "one_rhs"
        rationale = "Minimal reference."
        extra_overrides = ["e:diagnose=true"]
        trim_y_guards = true
        """,
        encoding="utf-8",
    )
    (case_dir / "BOUT.inp").write_text(
        """
        nout = 1
        timestep = 0.1

        [mesh]
        nx = 4
        ny = 8
        nz = 2

        [solver]
        mxstep = 10

        [model]
        components = e
        Nnorm = 1e18
        Tnorm = 5
        Bnorm = 1

        [e]
        type = evolve_density
        """,
        encoding="utf-8",
    )

    cases = load_reference_cases(manifest)
    resolved = resolve_reference_cases(reference_root, manifest_path=manifest)

    assert cases[0].name == "toy_case"
    assert cases[0].extra_overrides == ("e:diagnose=true",)
    assert cases[0].trim_y_guards is True
    assert resolved[0].exists is True
    assert resolved[0].run_config is not None
    assert resolved[0].run_config.time.nout == 1
    assert resolved[0].run_config.components[0].label == "e:evolve_density"
