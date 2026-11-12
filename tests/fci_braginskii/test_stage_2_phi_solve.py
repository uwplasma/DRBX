"""IMEX stage 2 solves phi for its base only under the sheath-current wall."""

from __future__ import annotations

import ast
from types import SimpleNamespace

from drbx.fci_braginskii import run


def test_only_the_sheath_current_wall_solves_the_stage_2_base_phi():
    assert not run._solves_stage_2_phi(None)
    assert not run._solves_stage_2_phi(SimpleNamespace(potential="ambipolar"))
    assert run._solves_stage_2_phi(SimpleNamespace(potential="current"))


def test_stage_infos_count_the_solves_that_run():
    source = open(run.__file__).read()
    stages = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "ssp222_stages"
    )
    text = ast.get_source_segment(source, stages)
    assert "if solve_stage_2_phi:" in text
    assert "(gmres_info_1, *base_infos, gmres_info_2, gmres_info_next)" in text
    assert "phi_solve_count = 3 + solve_stage_2_phi" in source
