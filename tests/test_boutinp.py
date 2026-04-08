from __future__ import annotations

import math
from pathlib import Path

from jax_drb.config.boutinp import ROOT_SECTION, NumericResolver, apply_bout_overrides, load_bout_input, parse_bout_input, parse_toml_input


def test_parser_preserves_section_and_key_order() -> None:
    config = parse_bout_input(
        """
        nout = 5
        timestep = 20

        [mesh]
        nx = 10
        ny = 10
        file = "grid.nc"  # quoted string

        [solver]
        mxstep = 1000
        """
    )

    assert config.section_names(include_root=True) == (ROOT_SECTION, "mesh", "solver")
    assert tuple(config.section(ROOT_SECTION).keys()) == ("nout", "timestep")
    assert tuple(config.section("mesh").keys()) == ("nx", "ny", "file")
    assert config.parsed(ROOT_SECTION, "nout") == 5
    assert config.parsed("mesh", "file") == "grid.nc"


def test_parser_handles_multiline_component_lists_and_types() -> None:
    config = parse_bout_input(
        """
        [model]
        components = (e, i, sound_speed,
                      vorticity, sheath_boundary)

        [e]
        type = evolve_density, evolve_pressure, evolve_momentum
        """
    )

    assert config.parsed("model", "components") == (
        "e",
        "i",
        "sound_speed",
        "vorticity",
        "sheath_boundary",
    )
    assert config.parsed("e", "type") == (
        "evolve_density",
        "evolve_pressure",
        "evolve_momentum",
    )


def test_numeric_resolver_handles_local_names_section_references_and_power_syntax() -> None:
    config = parse_bout_input(
        """
        [mesh]
        ny = 128
        Ly = 10
        dy = Ly / ny
        Bxy = 0.35
        Rxy = 1.5
        bxcvz = 1./Rxy^2

        [model]
        Bnorm = mesh:Bxy

        [i]
        AA = 1/1836
        """
    )
    resolver = NumericResolver(config)

    assert math.isclose(resolver.resolve("mesh", "dy"), 10.0 / 128.0)
    assert math.isclose(resolver.resolve("model", "Bnorm"), 0.35)
    assert math.isclose(resolver.resolve("mesh", "bxcvz"), 1.0 / (1.5**2))
    assert math.isclose(resolver.resolve("i", "AA"), 1.0 / 1836.0)


def test_non_numeric_expressions_remain_symbolic_until_resolved() -> None:
    config = parse_bout_input(
        """
        [Ne]
        function = exp(-(x - 0.5)^2 - (mesh:yn - 0.5)^2)
        """
    )

    value = config.get("Ne", "function")
    assert value.kind == "expression"
    assert value.raw == "exp(-(x - 0.5)^2 - (mesh:yn - 0.5)^2)"


def test_apply_bout_overrides_replaces_existing_values_and_adds_new_ones() -> None:
    config = parse_bout_input(
        """
        nout = 5

        [mesh]
        file = "grid.nc"
        """
    )

    updated = apply_bout_overrides(
        config,
        (
            "nout=1",
            "mesh:file=/tmp/tokamak.nc",
            "mesh:nx=64",
        ),
    )

    assert updated.parsed(ROOT_SECTION, "nout") == 1
    assert updated.parsed("mesh", "file") == "/tmp/tokamak.nc"
    assert updated.parsed("mesh", "nx") == 64


def test_parse_toml_input_supports_runtime_time_species_and_expression_wrappers() -> None:
    config = parse_toml_input(
        """
        [time]
        nout = 3
        timestep = 5.0

        [runtime]
        precision = "float32"

        [mesh]
        nx = 16
        ny = 24
        nz = 1
        dx = { expr = "0.0075 + 0.005*x" }
        dy = 0.01
        dz = 0.01
        J = 1

        [model]
        components = ["h"]

        [species.h]
        type = ["evolve_density", "evolve_pressure", "anomalous_diffusion"]
        AA = 1
        charge = 1
        anomalous_D = 2

        [fields.Nh]
        function = { expr = "1 + H(x - 0.25) * H(0.75-x)" }
        bndry_all = "neumann"
        """
    )

    assert config.parsed(ROOT_SECTION, "nout") == 3
    assert math.isclose(NumericResolver(config).resolve(ROOT_SECTION, "timestep"), 5.0)
    assert config.parsed("runtime", "precision") == "float32"
    assert config.parsed("model", "components") == ("h",)
    assert config.parsed("h", "type") == ("evolve_density", "evolve_pressure", "anomalous_diffusion")
    assert config.get("mesh", "dx").kind == "expression"
    assert config.get("Nh", "function").kind == "expression"


def test_load_bout_input_dispatches_to_toml_by_suffix(tmp_path: Path) -> None:
    input_path = tmp_path / "input.toml"
    input_path.write_text(
        """
        [time]
        nout = 1
        timestep = 2.5

        [mesh]
        nx = 4
        ny = 4
        nz = 1
        dx = 0.1
        dy = 0.1
        dz = 0.1
        J = 1
        """,
        encoding="utf-8",
    )

    config = load_bout_input(input_path)
    assert config.parsed(ROOT_SECTION, "nout") == 1
    assert math.isclose(NumericResolver(config).resolve(ROOT_SECTION, "timestep"), 2.5)
