"""Eager, matrix-free nonlinear coupled-boundary stage engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
import jax
import jax.numpy as jnp
import numpy as np
from .fci_gmres import SolvaxGmresConfig, solvax_gmres_pytree_solve


@dataclass(frozen=True)
class CoupledStageUnknown:
    state: Any
    multiplier: float = 0.0


@dataclass(frozen=True)
class CoupledStageConfig:
    material_rtol: float = 1e-8
    material_atol: float = 1e-10
    newton_maxiter: int = 12
    linear_maxiter: int = 300
    linear_restart: int = 50
    backtracking_maxiter: int = 12
    gauge_tol: float = 1e-10


@dataclass(frozen=True)
class CoupledStageInfo:
    converged: bool
    residual_norm: float
    initial_residual_norm: float
    iterations: int
    reason: str
    admissible: bool
    multiplier: float = 0.0
    residual_history: tuple[float, ...] = ()
    linear_iterations: int = 0
    rejected_residual_norm: float | None = None
    linear_diagnostics: tuple[dict[str, object], ...] = ()


def _failed_info(info, reason):
    if isinstance(info, dict):
        out = dict(info); out["converged"] = False
        out["reason"] = f"{reason}:{out.get('reason', 'unknown')}"
        return out
    if hasattr(info, "__dataclass_fields__"):
        from dataclasses import replace
        return replace(info, converged=False,
                       reason=f"{reason}:{getattr(info, 'reason', 'unknown')}")
    return CoupledStageInfo(False, float("nan"), float("nan"), 0, reason, False)


def advance_ssp222_coupled(current, dt, *, solve_stage, explicit,
                           reconstruct_final, source_stages=None):
    """Transactional host SSP222 orchestration for six evolved fields.

    ``solve_stage`` owns the coupled implicit solve and returns
    ``(stage_state, implicit_rate, info)``.  No state is committed after a
    failed stage or final algebraic reconstruction.
    """
    gamma = 1.0 - 1.0 / np.sqrt(2.0)
    zero_source = source_stages
    def source(i):
        if zero_source is None:
            return None
        return jax.tree_util.tree_map(lambda x: x[i], zero_source)
    def ok(info):
        return bool(info.get("converged", True)) if isinstance(info, dict) else bool(getattr(info, "converged", True))
    try:
        stage1, i1, info1 = solve_stage(current, gamma * dt)
        if not ok(info1):
            return current, _failed_info(info1, "implicit_stage1_failed")
        e1 = explicit(stage1, source(0))
        stage2_base = jax.tree_util.tree_map(
            lambda y, a, b: y + dt * a + (1.0 - 2.0 * gamma) * dt * b,
            current, e1, i1,
        )
        stage2, i2, info2 = solve_stage(stage2_base, gamma * dt)
        if not ok(info2):
            return current, _failed_info(info2, "implicit_stage2_failed")
        e2 = explicit(stage2, source(1))
        final = jax.tree_util.tree_map(
            lambda y, a, b, c, d: y + 0.5 * dt * (a + b + c + d),
            current, e1, e2, i1, i2,
        )
        reconstructed = reconstruct_final(final)
        result, final_info = reconstructed if isinstance(reconstructed, tuple) else (reconstructed, None)
        if final_info is not None and not ok(final_info):
            return current, _failed_info(final_info, "final_reconstruction_failed")
        return result, {"converged": True, "stage1": info1, "stage2": info2, "final": final_info}
    except Exception as exc:
        return current, {"converged": False, "reason": "stage_exception", "error": repr(exc)}


def _fd_jvp(residual: Callable, x, v, relative_step=None):
    x, v = jnp.asarray(x), jnp.asarray(v)
    step = (jnp.cbrt(jnp.finfo(x.dtype).eps) if relative_step is None else relative_step) * (1 + jnp.linalg.norm(x)) / jnp.maximum(jnp.linalg.norm(v), 1e-30)
    return (residual(x + step * v) - residual(x - step * v)) / (2 * step)


def solve_damped_newton(residual: Callable, initial, *, config=None, norm=None,
                        inner_product=None, admissible=None, preconditioner=None,
                        preconditioner_factory=None, convergence=None,
                        relative_step=None, residual_kernel=None,
                        residual_kernel_args=(), progress=None):
    """Matrix-free central-FD Newton with FGMRES linear corrections."""
    cfg = CoupledStageConfig() if config is None else config
    default_norm = norm is None
    norm = (lambda x: jnp.linalg.norm(x)) if norm is None else norm
    admissible = (lambda x: bool(np.isfinite(np.asarray(x)).all())) if admissible is None else admissible
    x = jnp.asarray(initial, dtype=jnp.float64)
    if not admissible(x):
        if progress is not None:
            progress({"event": "initial", "iteration": 0,
                      "residual_norm": float("inf"), "reason": "inadmissible_initial_state"})
        return x, CoupledStageInfo(False, float("inf"), float("inf"), 0,
                                   "inadmissible_initial_state", False)
    eval_residual = residual if residual_kernel is None else (
        lambda value: residual_kernel(value, *residual_kernel_args)
    )
    r = eval_residual(x); initial_r = r; r0 = float(norm(r)); history = [r0]; linear_total = 0
    if progress is not None:
        progress({"event": "initial", "iteration": 0, "residual_norm": r0,
                  "reason": "initial_residual"})
    rejected_norm = None; initial_x = x; linear_diagnostics = []
    reason = "max_newton_iterations"; converged = False
    for iteration in range(int(cfg.newton_maxiter)):
        rn = float(norm(r))
        if not np.isfinite(rn):
            reason = "nonfinite_residual"; break
        is_converged = (convergence(x, r, initial_r) if convergence is not None else
                        rn <= max(cfg.material_atol, cfg.material_rtol * max(r0, 1.0)))
        if is_converged:
            converged, reason = True, "converged"; break
        apply = lambda v: _fd_jvp(eval_residual, x, v, relative_step)
        gcfg = SolvaxGmresConfig(tol=cfg.material_rtol, atol=cfg.material_atol,
                                 maxiter=cfg.linear_maxiter,
                                 restart=min(cfg.linear_restart, cfg.linear_maxiter))
        if inner_product is None and not default_norm:
            raise ValueError("a custom norm requires a matching inner_product")
        ip = (lambda a, b: jnp.vdot(a, b)) if inner_product is None else inner_product
        prec = preconditioner_factory(x) if preconditioner_factory is not None else preconditioner
        if progress is not None:
            progress({"event": "linear_start", "iteration": iteration + 1,
                      "residual_norm": rn, "reason": "linear_solve_start"})
        dx, lin_info = solvax_gmres_pytree_solve(
            apply, -r, jnp.zeros_like(x), gcfg,
            inner_product=ip, norm=norm,
            all_finite=lambda a: jnp.all(jnp.isfinite(a)), preconditioner=prec)
        linear_total += int(np.asarray(lin_info.num_steps))
        def _scalar(name, default=np.nan):
            value = getattr(lin_info, name, default)
            try:
                return float(np.asarray(value))
            except (TypeError, ValueError):
                return default
        linear_diagnostics.append({
            "initial_residual_l2": _scalar("initial_residual_l2"),
            "final_residual_l2": _scalar("final_residual_l2"),
            "final_residual_rel_l2": _scalar("final_residual_rel_l2"),
            "converged": bool(np.asarray(getattr(lin_info, "converged", False))),
            "failed": bool(np.asarray(getattr(lin_info, "failed", False))),
            "num_steps": int(np.asarray(lin_info.num_steps)),
            "target_tolerance": float(cfg.material_rtol),
            "absolute_tolerance": float(cfg.material_atol),
        })
        if progress is not None:
            progress({"event": "linear", "iteration": iteration + 1,
                      "residual_norm": rn,
                      "linear_iterations": int(np.asarray(lin_info.num_steps)),
                      "reason": "linear_solve_failed" if (
                          bool(np.asarray(lin_info.failed)) or
                          not bool(np.asarray(lin_info.converged))
                      ) else "linear_solve_complete"})
        if bool(np.asarray(lin_info.failed)) or not bool(np.asarray(lin_info.converged)):
            reason = "linear_solve_failed"
            break
        accepted = False
        last_trial_norm = None
        for k in range(int(cfg.backtracking_maxiter) + 1):
            trial = x + (0.5 ** k) * dx
            if not admissible(trial):
                continue
            trial_r = eval_residual(trial); trial_norm = float(norm(trial_r))
            last_trial_norm = trial_norm
            if np.isfinite(trial_norm) and trial_norm < rn:
                x, r, accepted = trial, trial_r, True
                history.append(trial_norm)
                if progress is not None:
                    progress({"event": "correction", "iteration": iteration + 1,
                              "residual_norm": trial_norm, "linear_iterations": linear_total,
                              "reason": "accepted"})
                break
        if not accepted:
            rejected_norm = last_trial_norm
            reason = "backtracking_failed"; break
    # Check the post-correction iterate even when the correction consumed the
    # final permitted Newton iteration.  A user callback is authoritative.
    final_norm = float(norm(r))
    if not converged and reason == "max_newton_iterations":
        final_check = (convergence(x, r, initial_r) if convergence is not None else
                       final_norm <= max(cfg.material_atol, cfg.material_rtol * max(r0, 1.0)))
        if final_check:
            converged, reason = True, "converged"
    result_x = x if converged else initial_x
    return result_x, CoupledStageInfo(converged, final_norm, r0, len(history)-1,
                               reason, bool(admissible(result_x)), 0.0,
                               tuple(history), linear_total, rejected_norm,
                               tuple(linear_diagnostics))


def solve_coupled_boundary_stage(model, base, *, solve_dt, source_owned=None,
                                 config=None, preconditioner_factory=None,
                                 residual_kernel=None, progress=None):
    """Forward to the separately owned production model adapter."""
    from .fci_boundary_imex_model import solve_coupled_boundary_stage as adapter
    return adapter(model, base, solve_dt=solve_dt, source_owned=source_owned,
                   config=config, preconditioner_factory=preconditioner_factory,
                   residual_kernel=residual_kernel, progress=progress)


__all__ = ["CoupledStageUnknown", "CoupledStageConfig", "CoupledStageInfo",
           "solve_damped_newton", "solve_coupled_boundary_stage",
           "advance_ssp222_coupled"]
