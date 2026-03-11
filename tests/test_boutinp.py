from __future__ import annotations

import math

from jax_drb.config.boutinp import ROOT_SECTION, NumericResolver, parse_bout_input


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
        [hermes]
        components = (e, i, sound_speed,
                      vorticity, sheath_boundary)

        [e]
        type = evolve_density, evolve_pressure, evolve_momentum
        """
    )

    assert config.parsed("hermes", "components") == (
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

        [hermes]
        Bnorm = mesh:Bxy

        [i]
        AA = 1/1836
        """
    )
    resolver = NumericResolver(config)

    assert math.isclose(resolver.resolve("mesh", "dy"), 10.0 / 128.0)
    assert math.isclose(resolver.resolve("hermes", "Bnorm"), 0.35)
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
