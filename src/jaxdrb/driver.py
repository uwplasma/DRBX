from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import jax
import numpy as np

from jaxdrb.bc import BC2D, bc2d_from_strings
from jaxdrb.core.compat import coerce_system_params
from jaxdrb.core.geometry_registry import build_geometry
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.system import DRBSystem
from jaxdrb.integrators import (
    build_rk4_scan,
    build_rk4_scan_cached,
    build_rk4_scan_cached_iters,
)
from jaxdrb.normalization import NormalizationInfo, apply_normalization


@dataclass(frozen=True)
class BuiltSystem:
    system: DRBSystem
    state: DRBSystemState
    normalization: NormalizationInfo | None = None


@dataclass(frozen=True)
class RunResult:
    times: Any
    diagnostics: dict[str, Any]
    final_state: DRBSystemState


def _merge_params(*sections: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sec in sections:
        out.update(sec)
    return out


def _parse_bc_value(val: Any) -> BC2D | None:
    if val is None or isinstance(val, BC2D):
        return val
    if isinstance(val, str):
        return bc2d_from_strings(bc_x=val, bc_y=val)
    if isinstance(val, dict):
        bc_x = str(val.get("bc_x", val.get("x", val.get("bc", "periodic"))))
        bc_y = str(val.get("bc_y", val.get("y", val.get("bc", bc_x))))
        value_x = float(val.get("value_x", val.get("x_value", val.get("value", 0.0))))
        value_y = float(val.get("value_y", val.get("y_value", val.get("value", value_x))))
        grad_x = float(val.get("grad_x", val.get("x_grad", val.get("grad", 0.0))))
        grad_y = float(val.get("grad_y", val.get("y_grad", val.get("grad", grad_x))))
        return bc2d_from_strings(
            bc_x=bc_x,
            bc_y=bc_y,
            value_x=value_x,
            value_y=value_y,
            grad_x=grad_x,
            grad_y=grad_y,
        )
    raise TypeError(f"Unsupported BC value type: {type(val)}")


def _parse_bc_section(bc: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "n": "bc_n",
        "omega": "bc_omega",
        "vpar_e": "bc_vpar_e",
        "vpar_i": "bc_vpar_i",
        "Te": "bc_Te",
        "Ti": "bc_Ti",
        "psi": "bc_psi",
        "phi": "bc_phi",
        "perp": "perp_bc",
    }
    out: dict[str, Any] = {}
    for key, val in bc.items():
        tgt = mapping.get(key, key)
        if tgt in (
            "bc_n",
            "bc_omega",
            "bc_vpar_e",
            "bc_vpar_i",
            "bc_Te",
            "bc_Ti",
            "bc_psi",
            "bc_phi",
            "perp_bc",
        ):
            out[tgt] = _parse_bc_value(val)
        else:
            out[tgt] = val
    return out


def build_system_from_config(cfg: dict[str, Any]) -> BuiltSystem:
    cfg, norm_info = apply_normalization(cfg)

    physics = cfg.get("physics", {})
    closures = cfg.get("closures", {})
    numerics = cfg.get("numerics", {})
    terms = cfg.get("terms", {})
    if isinstance(terms, dict) and "schedule" in terms and "term_schedule" not in terms:
        terms = dict(terms)
        terms["term_schedule"] = terms.pop("schedule")
    bc = cfg.get("bc", {})
    if isinstance(bc, dict) and bc:
        bc = _parse_bc_section(bc)
    geometry = cfg.get("geometry", {})
    boundary_policy = cfg.get("boundary_policy", {})
    if isinstance(boundary_policy, dict) and boundary_policy:
        geometry = dict(geometry)
        geometry["boundary_policy"] = boundary_policy
    init = cfg.get("initial", {})

    params = DRBSystemParams()
    params = update_params_from_dict(params, _merge_params(physics, closures, numerics, terms, bc))

    geom = build_geometry(params, geometry)

    sys_params = coerce_system_params(params)
    system = DRBSystem(params=sys_params, geom=geom)

    shape = geom.shape()
    state = DRBSystemState.zeros(
        shape,
        hot_ion=bool(sys_params.hot_ion_on),
        em=bool(sys_params.em_on),
        neutrals=bool(sys_params.neutrals_on),
    )

    amp = float(init.get("amplitude", 0.0))
    if amp != 0.0:
        seed = int(init.get("seed", 0))
        key = jax.random.PRNGKey(seed)
        noise = amp * jax.random.normal(key, shape=state.n.shape, dtype=state.n.dtype)
        state = DRBSystemState(
            n=noise,
            omega=noise,
            vpar_e=jnp.zeros_like(state.n),
            vpar_i=jnp.zeros_like(state.n),
            Te=jnp.zeros_like(state.n),
            Ti=state.Ti,
            psi=state.psi,
            N=state.N,
        )

    return BuiltSystem(system=system, state=state, normalization=norm_info)


def _diagnostic_fn(
    system: DRBSystem,
    point_idx: tuple[int, ...],
    *,
    mode: str = "full",
    phi_every: int = 1,
    dt_save: float = 1.0,
) -> Callable[[float, DRBSystemState], tuple]:
    def diag(t, y, args=None):
        _ = args
        n_phys = system._phys_n(y.n)
        Te_phys = system._phys_Te(y.Te)
        if n_phys.ndim == 2:
            x0, y0 = point_idx
            point_n = n_phys[x0, y0]
            point_Te = Te_phys[x0, y0]
        else:
            z0, x0, y0 = point_idx
            point_n = n_phys[z0, x0, y0]
            point_Te = Te_phys[z0, x0, y0]
        if mode == "basic":
            zero = jnp.asarray(0.0)
            return (
                jnp.asarray(t),
                jnp.sqrt(jnp.mean(n_phys ** 2)),
                jnp.sqrt(jnp.mean(Te_phys ** 2)),
                jnp.sqrt(jnp.mean(y.omega ** 2)),
                zero,
                point_n,
                point_Te,
                zero,
            )
        if phi_every <= 1:
            phi = system._phi_from_omega(y.omega, n=n_phys)
            rms_phi = jnp.sqrt(jnp.mean(phi ** 2))
            point_phi = phi[x0, y0] if n_phys.ndim == 2 else phi[z0, x0, y0]
        else:
            idx = jnp.round(t / dt_save).astype(jnp.int32)
            use_phi = (idx % phi_every) == 0

            def _compute(_):
                phi = system._phi_from_omega(y.omega, n=n_phys)
                if n_phys.ndim == 2:
                    return jnp.sqrt(jnp.mean(phi ** 2)), phi[x0, y0]
                return jnp.sqrt(jnp.mean(phi ** 2)), phi[z0, x0, y0]

            def _skip(_):
                zero = jnp.asarray(0.0)
                return zero, zero

            rms_phi, point_phi = jax.lax.cond(use_phi, _compute, _skip, operand=None)
        return (
            jnp.asarray(t),
            jnp.sqrt(jnp.mean(n_phys ** 2)),
            jnp.sqrt(jnp.mean(Te_phys ** 2)),
            jnp.sqrt(jnp.mean(y.omega ** 2)),
            rms_phi,
            point_n,
            point_Te,
            point_phi,
        )

    return diag


def run_simulation(cfg: dict[str, Any]) -> RunResult:
    """Run a production jax_drb simulation with JIT scan or Diffrax.

    Config sections:
      - [time] or [integrator]: method, dt, nsteps, save_every, t_end,
        adaptive, rtol, atol, solver, progress, remat
    """
    cfg, norm_info = apply_normalization(cfg)
    built = build_system_from_config(cfg)
    system = built.system
    state = built.state

    time_cfg = cfg.get("time", {})
    integrator_cfg = cfg.get("integrator", {})
    if isinstance(integrator_cfg, dict):
        time_cfg = {**time_cfg, **integrator_cfg}

    method = str(time_cfg.get("method", "diffrax")).lower()
    dt = float(time_cfg.get("dt", 1e-3))
    nsteps = int(time_cfg.get("nsteps", 1000))
    save_every = int(time_cfg.get("save_every", 10))
    t_end = time_cfg.get("t_end", None)
    adaptive = bool(time_cfg.get("adaptive", True))
    rtol = float(time_cfg.get("rtol", 1e-5))
    atol = float(time_cfg.get("atol", 1e-7))
    solver_name = str(time_cfg.get("solver", "dopri8")).lower()
    progress = bool(time_cfg.get("progress", True))
    remat = bool(time_cfg.get("remat", False))
    diag_mode = str(time_cfg.get("diag_mode", "full")).lower()
    diag_phi_every = int(time_cfg.get("diag_phi_every", 1))
    warm_start = bool(time_cfg.get("poisson_warm_start", True))
    track_iters = bool(time_cfg.get("poisson_track_iters", False))
    return_numpy = bool(time_cfg.get("return_numpy", False))

    shape = state.n.shape
    if len(shape) == 2:
        nx, ny = shape
        point_idx = tuple(time_cfg.get("point_idx", (nx // 2, ny // 2)))
    else:
        nz, nx, ny = shape
        point_idx = tuple(time_cfg.get("point_idx", (nz // 2, nx // 2, ny // 2)))

    if method in ("rk4", "rk4_scan", "fixed"):
        dt_save = float(dt * save_every) if save_every > 0 else float(dt)
        diag_fn = _diagnostic_fn(
            system, point_idx, mode=diag_mode, phi_every=diag_phi_every, dt_save=dt_save
        )
        if remat:
            diag_fn = jax.checkpoint(diag_fn)
        if track_iters:
            runner, nsave, rem = build_rk4_scan_cached_iters(
                system.rhs_with_phi_iters,
                dt,
                nsteps,
                save_every,
                diag_fn,
                rhs_remat=remat,
                warm_start=warm_start,
            )
        elif warm_start:
            runner, nsave, rem = build_rk4_scan_cached(
                system.rhs_with_phi, dt, nsteps, save_every, diag_fn, rhs_remat=remat
            )
        else:
            runner, nsave, rem = build_rk4_scan(
                system.rhs, dt, nsteps, save_every, diag_fn, rhs_remat=remat
            )
        final_state, diag_series = runner(state)
        times = np.arange(nsave, dtype=np.float64) * (save_every * dt)
        if rem > 0:
            times[-1] = nsteps * dt
        if track_iters:
            (
                t,
                rms_n,
                rms_Te,
                rms_omega,
                rms_phi,
                point_n,
                point_Te,
                point_phi,
                iters_mean,
                iters_max,
            ) = diag_series
        else:
            t, rms_n, rms_Te, rms_omega, rms_phi, point_n, point_Te, point_phi = diag_series
    elif method in ("diffrax", "dopri8", "tsit5"):
        import diffrax as dfx
        if track_iters:
            track_iters = False

        solver_map = {
            "dopri8": dfx.Dopri8,
            "dopri5": dfx.Dopri5,
            "tsit5": dfx.Tsit5,
            "euler": dfx.Euler,
        }
        solver_cls = solver_map.get(solver_name, dfx.Dopri8)
        solver = solver_cls()

        if t_end is None:
            t_end = nsteps * dt
        if save_every <= 0:
            save_every = 1
        nsave = int(np.floor((t_end / dt) / save_every)) + 1
        times = jnp.asarray(np.linspace(0.0, float(t_end), nsave))

        dt_save = float(times[1] - times[0]) if nsave > 1 else float(dt)
        diag_fn = _diagnostic_fn(
            system, point_idx, mode=diag_mode, phi_every=diag_phi_every, dt_save=dt_save
        )
        if remat:
            diag_fn = jax.checkpoint(diag_fn)

        saveat = dfx.SaveAt(
            subs={
                "diag": dfx.SubSaveAt(ts=times, fn=diag_fn),
                "state": dfx.SubSaveAt(t1=True),
            }
        )

        controller = dfx.PIDController(rtol=rtol, atol=atol) if adaptive else dfx.ConstantStepSize()

        def vf(t, y, args):
            _ = args
            return system.rhs(t, y)

        term = dfx.ODETerm(vf)
        adjoint = dfx.DirectAdjoint()
        adjoint_mode = str(time_cfg.get("adjoint", "")).lower()
        if remat or adjoint_mode in ("checkpoint", "recursive_checkpoint"):
            adjoint = dfx.RecursiveCheckpointAdjoint()
        jit = bool(time_cfg.get("jit", False))
        progress_meter = (
            dfx.ProgressMeter() if (progress and not jit) else dfx.NoProgressMeter()
        )

        def _solve():
            return dfx.diffeqsolve(
                term,
                solver,
                t0=0.0,
                t1=float(t_end),
                dt0=dt,
                y0=state,
                saveat=saveat,
                stepsize_controller=controller,
                adjoint=adjoint,
                progress_meter=progress_meter,
                args=None,
            )

        solve_fn = jax.jit(_solve) if jit else _solve
        sol = solve_fn()
        diag_series = sol.ys["diag"]
        final_state = sol.ys["state"]
        t, rms_n, rms_Te, rms_omega, rms_phi, point_n, point_Te, point_phi = diag_series
    else:
        raise ValueError(f"Unknown integrator method: {method}")

    diagnostics = {
        "t": t,
        "rms_n": rms_n,
        "rms_Te": rms_Te,
        "rms_omega": rms_omega,
        "rms_phi": rms_phi,
        "point_n": point_n,
        "point_Te": point_Te,
        "point_phi": point_phi,
        "times": times,
    }
    if track_iters:
        diagnostics["poisson_iters_mean"] = iters_mean
        diagnostics["poisson_iters_max"] = iters_max
        diagnostics["poisson_iters_mean_all"] = jnp.mean(iters_mean)
        diagnostics["poisson_iters_max_all"] = jnp.max(iters_max)

    if return_numpy:
        diagnostics = {k: np.asarray(jax.device_get(v)) for k, v in diagnostics.items()}
        times = np.asarray(times)

    return RunResult(times=times, diagnostics=diagnostics, final_state=final_state)
