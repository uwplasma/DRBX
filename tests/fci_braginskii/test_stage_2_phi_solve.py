"""IMEX stage 2 solves phi for its base only under the sheath-current wall."""

from __future__ import annotations

import ast

import numpy as np
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
        if isinstance(node, ast.FunctionDef) and node.name == "_ssp222_stages"
    )
    text = ast.get_source_segment(source, stages)
    assert "if solve_stage_2_phi:" in text
    assert "(gmres_info_1, *base_infos, gmres_info_2, gmres_info_next)" in text
    assert "phi_solve_count = 3 + solve_stage_2_phi" in source


def _linear_ssp222_run(a, b, dt, steps):
    """Advance y' = -a y (explicit) - b y (implicit) with the production stage algebra."""

    import jax.numpy as jnp
    from drbx.fci_braginskii.native.fci_drb_EB_rhs import FciDrbEBState

    zero = jnp.zeros((1, 1, 1))
    state = FciDrbEBState(density=jnp.ones((1, 1, 1)), phi=zero, Te=zero, Ti=zero, Vi=zero, Ve=zero, vorticity=zero)
    info = jnp.zeros(7)

    def implicit(base, gamma_dt):
        stage = base.replace(density=base.density / (1.0 + b * gamma_dt))
        return stage, stage.axpy(base, scale=-1.0), info

    def explicit(stage):
        return stage.replace(density=-a * stage.density)

    for _ in range(steps):
        stages, _, _ = run._ssp222_stages(
            state, implicit, explicit, lambda s: (s.phi, info), dt, solve_stage_2_phi=False
        )
        state = stages[-1]
    return float(state.density[0, 0, 0])


def test_ssp222_stage_algebra_is_second_order_on_a_linear_split_problem():
    # Pareschi and Russo, IMEX-SSP2(2,2,2); exact solution exp(-(a + b) t).
    a, b, t_end = 2.0, 3.0, 1.0
    errors = [abs(_linear_ssp222_run(a, b, t_end / n, n) - np.exp(-(a + b) * t_end)) for n in (8, 16, 32)]
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert all(1.9 < order < 2.5 for order in orders), orders
