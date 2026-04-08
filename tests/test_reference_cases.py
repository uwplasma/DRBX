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


def test_default_manifest_stages_tokamak_recycling_cases() -> None:
    cases = load_reference_cases()
    rhs_case = next(case for case in cases if case.name == "tokamak_recycling_rhs")
    one_step_case = next(case for case in cases if case.name == "tokamak_recycling_one_step")
    dthe_rhs_case = next(case for case in cases if case.name == "tokamak_recycling_dthe_rhs")
    dthe_one_step_case = next(case for case in cases if case.name == "tokamak_recycling_dthe_one_step")
    dthene_rhs_case = next(case for case in cases if case.name == "tokamak_recycling_dthene_rhs")

    assert rhs_case.reference_path == "examples/tokamak-2D/recycling/BOUT.inp"
    assert rhs_case.parity_mode == "one_rhs"
    assert rhs_case.process_count == 6
    assert rhs_case.extra_overrides == (
        "mesh:file={reference_root}/examples/tokamak-2D/recycling/tokamak.nc",
        "hermes:components=(d+, d, e, sheath_boundary_simple, braginskii_collisions, braginskii_friction, braginskii_heat_exchange, sound_speed, reactions, electron_force_balance, braginskii_conduction, recycling)",
    )

    assert one_step_case.reference_path == "examples/tokamak-2D/recycling/BOUT.inp"
    assert one_step_case.parity_mode == "one_step"
    assert one_step_case.process_count == 6
    assert one_step_case.extra_overrides == (
        "timestep=1",
        "mesh:file={reference_root}/examples/tokamak-2D/recycling/tokamak.nc",
        "hermes:components=(d+, d, e, sheath_boundary_simple, braginskii_collisions, braginskii_friction, braginskii_heat_exchange, sound_speed, reactions, electron_force_balance, braginskii_conduction, recycling)",
    )

    assert dthe_rhs_case.reference_path == "examples/tokamak-2D/recycling-dthe/BOUT.inp"
    assert dthe_rhs_case.parity_mode == "one_rhs"
    assert dthe_rhs_case.process_count == 6
    assert dthe_rhs_case.extra_overrides == (
        "mesh:file={reference_root}/examples/tokamak-2D/recycling-dthe/tokamak.nc",
        "he+:diagnose=false",
        "input:error_on_unused_options=false",
    )

    assert dthe_one_step_case.reference_path == "examples/tokamak-2D/recycling-dthe/BOUT.inp"
    assert dthe_one_step_case.parity_mode == "one_step"
    assert dthe_one_step_case.process_count == 6
    assert dthe_one_step_case.extra_overrides == (
        "timestep=0.1",
        "mesh:file={reference_root}/examples/tokamak-2D/recycling-dthe/tokamak.nc",
        "he+:diagnose=false",
        "input:error_on_unused_options=false",
    )

    assert dthene_rhs_case.reference_path == "examples/tokamak-2D/recycling-dthene/BOUT.inp"
    assert dthene_rhs_case.parity_mode == "one_rhs"
    assert dthene_rhs_case.process_count == 6
    assert dthene_rhs_case.extra_overrides == (
        "mesh:file={reference_root}/examples/tokamak-2D/tokamak.nc",
        "json_database_dir={reference_root}/json_database",
        "he+:diagnose=false",
        "ne+:diagnose=false",
        "input:error_on_unused_options=false",
    )


def test_default_manifest_stages_tokamak_diffusion_flow_one_step_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "tokamak_diffusion_flow_one_step")

    assert case.reference_path == "examples/tokamak-2D/diffusion-flow-evolveT/BOUT.inp"
    assert case.parity_mode == "one_step"
    assert case.compare_variables == ("Nh", "Ph", "NVh")
    assert case.trim_x_guards is True
    assert case.trim_y_guards is True
    assert case.process_count == 6


def test_default_manifest_stages_tokamak_diffusion_one_step_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "tokamak_diffusion_one_step")

    assert case.reference_path == "examples/tokamak-2D/diffusion/BOUT.inp"
    assert case.parity_mode == "one_step"
    assert case.compare_variables == ("Nh",)
    assert case.trim_x_guards is True
    assert case.trim_y_guards is True
    assert case.process_count == 6


def test_default_manifest_stages_tokamak_diffusion_transport_one_step_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "tokamak_diffusion_transport_one_step")

    assert case.reference_path == "examples/tokamak-2D/diffusion-transport/BOUT.inp"
    assert case.parity_mode == "one_step"
    assert case.compare_variables == ("Nh+", "Ph+", "NVh+", "Pe")
    assert case.trim_x_guards is True
    assert case.trim_y_guards is True
    assert case.process_count == 6


def test_default_manifest_stages_tokamak_diffusion_transport_short_window_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "tokamak_diffusion_transport_short_window")

    assert case.reference_path == "examples/tokamak-2D/diffusion-transport/BOUT.inp"
    assert case.parity_mode == "short_window"
    assert case.compare_variables == ("Nh+", "Ph+", "NVh+", "Pe")
    assert case.extra_overrides == ("nout=5",)
    assert case.trim_x_guards is True
    assert case.trim_y_guards is True
    assert case.process_count == 6


def test_default_manifest_stages_tokamak_heat_transport_one_step_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "tokamak_heat_transport_one_step")

    assert case.reference_path == "examples/tokamak-2D/heat-transport/BOUT.inp"
    assert case.parity_mode == "one_step"
    assert case.compare_variables == ("Pe",)
    assert case.trim_x_guards is True
    assert case.trim_y_guards is True
    assert case.process_count == 6


def test_default_manifest_stages_tokamak_heat_transport_short_window_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "tokamak_heat_transport_short_window")

    assert case.reference_path == "examples/tokamak-2D/heat-transport/BOUT.inp"
    assert case.parity_mode == "short_window"
    assert case.compare_variables == ("Pe",)
    assert case.extra_overrides == ("nout=2", "e:diagnose=false")
    assert case.trim_x_guards is True
    assert case.trim_y_guards is True
    assert case.process_count == 6


def test_default_manifest_stages_tokamak_diffusion_conduction_one_step_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "tokamak_diffusion_conduction_one_step")

    assert case.reference_path == "examples/tokamak-2D/diffusion-conduction/BOUT.inp"
    assert case.parity_mode == "one_step"
    assert case.compare_variables == ("Nh+", "Ph+", "Pe")
    assert case.extra_overrides == ("h+:diagnose=false", "e:diagnose=false")
    assert case.trim_x_guards is True
    assert case.trim_y_guards is True
    assert case.process_count == 6


def test_default_manifest_stages_tokamak_linear_transport_one_step_case() -> None:
    cases = load_reference_cases()
    case = next(case for case in cases if case.name == "tokamak_linear_transport_one_step")

    assert case.reference_path == "examples/tokamak-2D/linear-transport/BOUT.inp"
    assert case.parity_mode == "one_step"
    assert case.compare_variables == ("Pe",)
    assert case.extra_overrides == ("e:diagnose=false",)
    assert case.trim_x_guards is True
    assert case.trim_y_guards is True
    assert case.process_count == 6


def test_tokamak_diffusion_flow_case_includes_momentum_and_anomalous_diffusion_components() -> None:
    resolved = resolve_reference_cases(Path("/Users/rogerio/local/hermes-3"))
    resolved_case = next(case for case in resolved if case.case.name == "tokamak_diffusion_flow_one_step")
    assert resolved_case.run_config is not None
    labels = tuple(component.label for component in resolved_case.run_config.components)
    assert "h:evolve_density" in labels
    assert "h:evolve_pressure" in labels
    assert "h:evolve_momentum" in labels
    assert "h:anomalous_diffusion" in labels


def test_tokamak_diffusion_case_includes_density_and_anomalous_diffusion_components() -> None:
    resolved = resolve_reference_cases(Path("/Users/rogerio/local/hermes-3"))
    resolved_case = next(case for case in resolved if case.case.name == "tokamak_diffusion_one_step")
    assert resolved_case.run_config is not None
    labels = tuple(component.label for component in resolved_case.run_config.components)
    assert "h:evolve_density" in labels
    assert "h:anomalous_diffusion" in labels


def test_tokamak_diffusion_transport_case_includes_coupled_transport_components() -> None:
    resolved = resolve_reference_cases(Path("/Users/rogerio/local/hermes-3"))
    resolved_case = next(case for case in resolved if case.case.name == "tokamak_diffusion_transport_one_step")
    assert resolved_case.run_config is not None
    labels = tuple(component.label for component in resolved_case.run_config.components)
    assert "h+:evolve_density" in labels
    assert "h+:evolve_pressure" in labels
    assert "h+:evolve_momentum" in labels
    assert "h+:anomalous_diffusion" in labels
    assert "e:quasineutral" in labels
    assert "e:evolve_pressure" in labels
    assert "e:zero_current" in labels
    assert "e:anomalous_diffusion" in labels


def test_tokamak_diffusion_transport_short_window_case_includes_coupled_transport_components() -> None:
    resolved = resolve_reference_cases(Path("/Users/rogerio/local/hermes-3"))
    resolved_case = next(case for case in resolved if case.case.name == "tokamak_diffusion_transport_short_window")
    assert resolved_case.run_config is not None
    labels = tuple(component.label for component in resolved_case.run_config.components)
    assert "h+:evolve_density" in labels
    assert "h+:evolve_pressure" in labels
    assert "h+:evolve_momentum" in labels
    assert "h+:anomalous_diffusion" in labels
    assert "e:quasineutral" in labels
    assert "e:evolve_pressure" in labels
    assert "e:zero_current" in labels
    assert "e:anomalous_diffusion" in labels


def test_tokamak_heat_transport_case_includes_heat_transport_components() -> None:
    resolved = resolve_reference_cases(Path("/Users/rogerio/local/hermes-3"))
    resolved_case = next(case for case in resolved if case.case.name == "tokamak_heat_transport_one_step")
    assert resolved_case.run_config is not None
    labels = tuple(component.label for component in resolved_case.run_config.components)
    assert "e:fixed_density" in labels
    assert "e:evolve_pressure" in labels
    assert "e:anomalous_diffusion" in labels
    assert "h+:quasineutral" in labels
    assert "h+:set_temperature" in labels
    assert "sheath_boundary_simple" in labels
    assert "braginskii_conduction" in labels


def test_tokamak_heat_transport_short_window_case_includes_heat_transport_components() -> None:
    resolved = resolve_reference_cases(Path("/Users/rogerio/local/hermes-3"))
    resolved_case = next(case for case in resolved if case.case.name == "tokamak_heat_transport_short_window")
    assert resolved_case.run_config is not None
    labels = tuple(component.label for component in resolved_case.run_config.components)
    assert "e:fixed_density" in labels
    assert "e:evolve_pressure" in labels
    assert "e:anomalous_diffusion" in labels
    assert "h+:quasineutral" in labels
    assert "h+:set_temperature" in labels
    assert "sheath_boundary_simple" in labels
    assert "braginskii_conduction" in labels


def test_tokamak_diffusion_conduction_case_includes_conduction_components() -> None:
    resolved = resolve_reference_cases(Path("/Users/rogerio/local/hermes-3"))
    resolved_case = next(case for case in resolved if case.case.name == "tokamak_diffusion_conduction_one_step")
    assert resolved_case.run_config is not None
    labels = tuple(component.label for component in resolved_case.run_config.components)
    assert "h+:evolve_density" in labels
    assert "h+:evolve_pressure" in labels
    assert "h+:anomalous_diffusion" in labels
    assert "e:quasineutral" in labels
    assert "e:evolve_pressure" in labels
    assert "e:anomalous_diffusion" in labels
    assert "sheath_boundary" in labels
    assert "braginskii_conduction" in labels


def test_tokamak_linear_transport_case_includes_fixed_density_transport_components() -> None:
    resolved = resolve_reference_cases(Path("/Users/rogerio/local/hermes-3"))
    resolved_case = next(case for case in resolved if case.case.name == "tokamak_linear_transport_one_step")
    assert resolved_case.run_config is not None
    labels = tuple(component.label for component in resolved_case.run_config.components)
    assert "e:fixed_density" in labels
    assert "e:evolve_pressure" in labels
    assert "e:anomalous_diffusion" in labels
    assert "e:simple_conduction" in labels
