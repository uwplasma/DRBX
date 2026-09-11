import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

from drbx.native.fci_boundary_imex import (
    CoupledStageConfig,
    _fd_jvp,
    solve_damped_newton,
    advance_ssp222_coupled,
)


def test_matrix_free_newton_solves_linear_system():
    A = jnp.asarray([[2.0, 0.3], [-0.4, 1.5]])
    target = jnp.asarray([0.7, -1.2])
    residual = lambda x: A @ (x - target)
    x, info = solve_damped_newton(residual, jnp.zeros(2))
    np.testing.assert_allclose(x, target, rtol=2e-6, atol=2e-7)
    assert info.converged
    assert info.linear_iterations > 0
    assert info.linear_diagnostics
    assert info.linear_diagnostics[0]["num_steps"] > 0
    assert "final_residual_rel_l2" in info.linear_diagnostics[0]


def test_damped_newton_backtracks_inadmissible_trial():
    target = jnp.asarray([1.0]); calls = []
    residual = lambda x: x * x - target
    def admissible(x):
        calls.append(np.asarray(x).copy())
        return bool(np.all((np.asarray(x) > 0) & (np.asarray(x) <= 2.0)))
    x, info = solve_damped_newton(
        residual, jnp.asarray([0.1]),
        config=CoupledStageConfig(backtracking_maxiter=16),
        admissible=admissible,
    )
    np.testing.assert_allclose(x, jnp.asarray([1.0]), rtol=2e-6, atol=2e-7)
    assert info.converged
    assert len(calls) > 4


def test_nonfinite_or_inadmissible_problem_reports_failure():
    residual = lambda x: x + 1.0
    x, info = solve_damped_newton(
        residual, jnp.asarray([-0.5]),
        admissible=lambda v: bool(np.asarray(v)[0] > 0.0),
        config=CoupledStageConfig(backtracking_maxiter=2),
    )
    assert not info.converged
    assert info.reason in ("inadmissible_initial_state", "backtracking_failed", "max_newton_iterations")


def test_initial_invalid_zero_residual_is_rejected_before_residual_work():
    calls = []
    x, info = solve_damped_newton(
        lambda v: calls.append(v) or v * 0.0,
        jnp.asarray([-1.0]),
        admissible=lambda v: bool(np.asarray(v)[0] > 0.0),
    )
    assert not calls
    assert info.reason == "inadmissible_initial_state"
    assert not info.admissible


def test_progress_callback_reports_initial_and_newton_events():
    events = []
    x, info = solve_damped_newton(
        lambda v: v - 1.0, jnp.asarray([0.0]),
        progress=events.append,
    )
    assert info.converged
    assert events[0]["event"] == "initial"
    assert any(event["event"] == "linear_start" for event in events)
    assert any(event["event"] == "linear" for event in events)
    assert any(event["event"] == "correction" for event in events)


def test_convergence_on_final_allowed_newton_iteration():
    target = jnp.asarray([0.75])
    x, info = solve_damped_newton(
        lambda v: v - target, jnp.asarray([0.0]),
        config=CoupledStageConfig(newton_maxiter=1),
    )
    np.testing.assert_allclose(x, target, rtol=2e-6, atol=2e-7)
    assert info.converged
    assert info.iterations == 1


def test_fd_jvp_refines_for_linear_residual():
    A = jnp.asarray([[2.0, -0.5], [0.25, 1.5]])
    x = jnp.asarray([0.7, -0.4]); v = jnp.asarray([1.1, 0.3])
    expected = A @ v
    coarse = _fd_jvp(lambda q: A @ q, x, v, relative_step=1e-4)
    fine = _fd_jvp(lambda q: A @ q, x, v, relative_step=1e-6)
    assert float(jnp.linalg.norm(fine - expected)) < 1e-8
    assert float(jnp.linalg.norm(fine - expected)) <= float(jnp.linalg.norm(coarse - expected)) + 1e-8


def test_linear_failure_is_reported(monkeypatch):
    import drbx.native.fci_boundary_imex as engine
    def failed(*args, **kwargs):
        return jnp.zeros(1), SimpleNamespace(num_steps=0, failed=jnp.asarray(True), converged=jnp.asarray(False))
    monkeypatch.setattr(engine, "solvax_gmres_pytree_solve", failed)
    x, info = engine.solve_damped_newton(lambda v: v - 1.0, jnp.asarray([0.0]))
    np.testing.assert_array_equal(x, jnp.asarray([0.0]))
    assert info.reason == "linear_solve_failed"
    assert not info.converged
    assert info.linear_diagnostics[0]["failed"]


def test_coupled_forwarder_passes_residual_kernel(monkeypatch):
    import drbx.native.fci_boundary_imex as engine
    seen = {}
    def adapter(model, base, **kwargs):
        seen.update(kwargs)
        return base, 0.0, SimpleNamespace(converged=True)
    import drbx.native.fci_boundary_imex_model as model_module
    monkeypatch.setattr(model_module, "solve_coupled_boundary_stage", adapter)
    marker = object()
    engine.solve_coupled_boundary_stage("model", "base", solve_dt=1.0,
                                       residual_kernel=marker)
    assert seen["residual_kernel"] is marker


def test_adapter_preserves_specific_initial_admission_reason(monkeypatch):
    import drbx.native.fci_boundary_imex_model as mm
    class FakeContext:
        augmented = False
        def __init__(self, model, base, solve_dt, source):
            pass
        def pack(self, state, multiplier):
            return jnp.asarray([-1.0])
        def residual(self, value):
            raise AssertionError("residual must not run for invalid initial state")
        def norm(self, value):
            return jnp.linalg.norm(value)
        def inner_product(self, a, b):
            return jnp.vdot(a, b)
        def admissibility(self, value):
            return {"admissible": False, "reason": "negative_density"}
        def convergence(self, x, r, r0):
            return False
    monkeypatch.setattr(mm, "CoupledBoundaryStageContext", FakeContext)
    state, multiplier, info = mm.solve_coupled_boundary_stage(object(), object(), solve_dt=1.0)
    assert not info.converged
    assert info.reason == "inadmissible_initial_state:negative_density"


def test_convergence_callback_is_consulted_and_failure_restores_initial():
    seen = []
    x, info = solve_damped_newton(
        lambda v: v - 1.0, jnp.asarray([0.0]),
        convergence=lambda x, r, r0: seen.append((float(np.linalg.norm(np.asarray(r))), np.shape(r0))) or False,
        config=CoupledStageConfig(newton_maxiter=1),
    )
    np.testing.assert_array_equal(x, jnp.asarray([0.0]))
    assert seen and seen[0][1] == (1,) and not info.converged


def test_custom_norm_requires_matching_krylov_inner_product():
    with np.testing.assert_raises(ValueError):
        solve_damped_newton(lambda v: v - 1.0, jnp.asarray([0.0]),
                            norm=lambda v: jnp.sqrt(2.0 * jnp.vdot(v, v)))


def test_ssp222_manufactured_nonlinear_global_second_order():
    # Split y'=-y^2 into explicit and implicit parts.  Each implicit stage is
    # solved by the actual matrix-free Newton engine, then repeated to T=1.
    def one_step(y0, dt):
        def solve_stage(base, stage_dt):
            def residual(v):
                return v - base - stage_dt * (-0.7 * v * v)
            stage, stage_info = solve_damped_newton(
                residual, base,
                admissible=lambda v: bool(np.asarray(v)[0] > 0.0),
            )
            return stage, -0.7 * stage * stage, stage_info
        result, info = advance_ssp222_coupled(
            jnp.asarray([y0]), dt, solve_stage=solve_stage,
            explicit=lambda state, source: -0.3 * state * state,
            reconstruct_final=lambda x: x,
        )
        assert info["converged"]
        return float(result[0])
    def integrate(dt):
        y = 1.0
        for _ in range(round(1.0 / dt)):
            y = one_step(y, dt)
        return y
    errors = [abs(integrate(h) - 0.5) for h in (0.125, 0.0625, 0.03125)]
    ratios = [errors[0] / errors[1], errors[1] / errors[2]]
    assert all(3.2 < ratio < 5.2 for ratio in ratios)


def test_ssp222_fixed_final_time_noncommuting_linear_split_is_second_order():
    """The actual Newton/GMRES stage engine preserves SSP222 order."""
    E = jnp.asarray([[0.15, 0.8], [-0.35, -0.10]], dtype=jnp.float64)
    I = jnp.asarray([[-0.45, 0.25], [0.55, -0.30]], dtype=jnp.float64)
    initial = jnp.asarray([0.7, -0.4], dtype=jnp.float64)
    final_exact = jax.scipy.linalg.expm(E + I) @ initial
    config = CoupledStageConfig(
        material_rtol=2.0e-11, material_atol=2.0e-13,
        linear_maxiter=40, linear_restart=10, newton_maxiter=5,
    )

    def integrate(steps):
        dt = 1.0 / steps
        def solve_stage(base, stage_dt):
            residual = lambda value: value - base - stage_dt * (I @ value)
            stage, info = solve_damped_newton(
                residual, base, config=config,
                admissible=lambda value: bool(np.all(np.isfinite(np.asarray(value)))),
            )
            assert info.converged
            return stage, I @ stage, info
        state = initial
        for _ in range(steps):
            state, info = advance_ssp222_coupled(
                state, dt, solve_stage=solve_stage,
                explicit=lambda value, source: E @ value,
                reconstruct_final=lambda value: value,
            )
            assert info["converged"]
        return state

    errors = [float(jnp.linalg.norm(integrate(n) - final_exact)) for n in (8, 16, 32)]
    assert errors[0] > errors[1] > errors[2]
    assert 3.5 < errors[0] / errors[1] < 4.5
    assert 3.5 < errors[1] / errors[2] < 4.5


def test_ssp222_orchestrator_commits_only_successful_final_state():
    def solve_stage(base, stage_dt):
        return base, jnp.ones_like(base), {"converged": True, "dt": stage_dt}
    explicit = lambda state, source: jnp.ones_like(state) * 2.0
    result, info = advance_ssp222_coupled(
        jnp.asarray([3.0]), 0.1, solve_stage=solve_stage,
        explicit=explicit, reconstruct_final=lambda x: (x, {"converged": True}),
    )
    np.testing.assert_allclose(result, jnp.asarray([3.3]))
    assert info["converged"]


def test_ssp222_orchestrator_rolls_back_failed_stage():
    def solve_stage(base, stage_dt):
        return base + 1.0, jnp.ones_like(base), {"converged": False}
    result, info = advance_ssp222_coupled(
        jnp.asarray([3.0]), 0.1, solve_stage=solve_stage,
        explicit=lambda state, source: state,
        reconstruct_final=lambda x: x,
    )
    np.testing.assert_array_equal(result, jnp.asarray([3.0]))
    assert not info["converged"]
