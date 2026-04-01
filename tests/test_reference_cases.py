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
        process_count = 4
        artifact_bundle_url = "file:///tmp/reference-bundle.zip"
        artifact_bundle_sha256 = "abc123"
        artifact_bundle_files = ["grid_test2.nc"]
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
    assert cases[0].process_count == 4
    assert cases[0].artifact_bundle_url == "file:///tmp/reference-bundle.zip"
    assert cases[0].artifact_bundle_sha256 == "abc123"
    assert cases[0].artifact_bundle_files == ("grid_test2.nc",)
    assert resolved[0].exists is True
    assert resolved[0].run_config is not None
    assert resolved[0].run_config.time.nout == 1
    assert resolved[0].run_config.components[0].label == "e:evolve_density"


def test_default_manifest_stages_integrated_2d_recycling_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "integrated_2d_recycling_one_step")

    assert case.reference_path == "tests/integrated/2D-recycling/data/BOUT.inp"
    assert case.process_count == 10
    assert case.artifact_bundle_url is not None
    assert case.artifact_bundle_sha256 == "167410a1768c2805acdd28895d4327fa448bc742107ddf82b9062c02800b0cbe"
    assert case.artifact_bundle_files == ("grid_test2.nc",)


def test_integrated_2d_production_case_includes_anomalous_diffusion_components() -> None:
    resolved = resolve_reference_cases(Path("/Users/rogerio/local/hermes-3"))
    resolved_case = next(case for case in resolved if case.case.name == "integrated_2d_production_rhs")
    assert resolved_case.run_config is not None
    labels = tuple(component.label for component in resolved_case.run_config.components)
    assert "d+:anomalous_diffusion" in labels
    assert "e:anomalous_diffusion" in labels


def test_default_manifest_stages_integrated_2d_recycling_medium_window_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "integrated_2d_recycling_medium_window")

    assert case.reference_path == "tests/integrated/2D-recycling/data/BOUT.inp"
    assert case.parity_mode == "short_window"
    assert case.extra_overrides == ("nout=20",)
    assert case.trim_x_guards is True
    assert case.trim_y_guards is True
    assert case.process_count == 10
    assert case.artifact_bundle_url is not None
    assert case.artifact_bundle_sha256 == "167410a1768c2805acdd28895d4327fa448bc742107ddf82b9062c02800b0cbe"
    assert case.artifact_bundle_files == ("grid_test2.nc",)


def test_default_manifest_stages_alfven_wave_cases() -> None:
    cases = load_reference_cases()
    rhs_case = next(case for case in cases if case.name == "alfven_wave_rhs")
    one_step_case = next(case for case in cases if case.name == "alfven_wave_one_step")
    short_window_case = next(case for case in cases if case.name == "alfven_wave_short_window")
    medium_window_case = next(case for case in cases if case.name == "alfven_wave_medium_window")

    assert rhs_case.reference_path == "tests/integrated/alfven-wave/data/BOUT.inp"
    assert rhs_case.parity_mode == "one_rhs"
    assert rhs_case.extra_overrides == ("e:diagnose=true", "vorticity:diagnose=true")
    assert rhs_case.compare_variables == ("Apar", "Ajpar", "phi", "Vort", "NVe", "ddt(NVe)", "ddt(Vort)")

    assert one_step_case.reference_path == "tests/integrated/alfven-wave/data/BOUT.inp"
    assert one_step_case.parity_mode == "one_step"
    assert one_step_case.compare_variables == ("Apar", "Ajpar", "phi", "Vort", "NVe")

    assert short_window_case.reference_path == "tests/integrated/alfven-wave/data/BOUT.inp"
    assert short_window_case.parity_mode == "short_window"
    assert short_window_case.compare_variables == ("Apar", "Ajpar", "phi", "Vort", "NVe")
    assert short_window_case.extra_overrides == ("nout=20",)

    assert medium_window_case.reference_path == "tests/integrated/alfven-wave/data/BOUT.inp"
    assert medium_window_case.parity_mode == "short_window"
    assert medium_window_case.compare_variables == ("Apar", "Ajpar", "phi", "Vort", "NVe")
    assert medium_window_case.extra_overrides == ()


def test_default_manifest_stages_annulus_he_emag_rhs_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "annulus_he_emag_rhs")

    assert case.reference_path == "examples/other/linear/annulus-isothermal-he-emag/BOUT.inp"
    assert case.parity_mode == "one_rhs"
    assert case.compare_variables == ("Apar", "alpha_em", "ddt(Ne)", "ddt(NVe)", "ddt(Vort)")
    assert case.extra_overrides == (
        "nout=0",
        "e:diagnose=true",
        "vorticity:diagnose=true",
        "electromagnetic:diagnose=true",
    )


def test_default_manifest_stages_annulus_he_emag_one_step_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "annulus_he_emag_one_step")

    assert case.reference_path == "examples/other/linear/annulus-isothermal-he-emag/BOUT.inp"
    assert case.parity_mode == "one_step"
    assert case.compare_variables == ("Apar", "Ne", "NVe", "phi", "Vort")
    assert case.extra_overrides == ("timestep=10",)


def test_default_manifest_stages_annulus_he_emag_short_window_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "annulus_he_emag_short_window")

    assert case.reference_path == "examples/other/linear/annulus-isothermal-he-emag/BOUT.inp"
    assert case.parity_mode == "short_window"
    assert case.compare_variables == ("Apar", "Ne", "phi")
    assert case.extra_overrides == ("timestep=10", "nout=5")
