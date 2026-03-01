from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jspl
import numpy as np

from jaxdrb.bc import BC2D, bc2d_from_strings
from jaxdrb.core.compat import coerce_system_params
from jaxdrb.core.geometry_registry import build_geometry
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.system import DRBSystem
from jaxdrb.core.terms.fields import phys_Te, phys_n
from jaxdrb.core.terms.sol import sol_masks
from jaxdrb.core.terms.bcs import resolve_bcs
from jaxdrb.integrators import (
    build_rk4_scan,
    build_rk4_scan_cached_iters_phi,
    build_rk4_scan_cached_iters_split_phi,
    build_rk4_scan_cached_iters_split,
    build_rk4_scan_cached_split,
    build_rk4_scan_cached_phi,
    build_rk4_scan_cached_split_phi,
    build_rk4_scan_cached,
    build_rk4_scan_cached_iters,
    build_rk4_scan_imex_strang,
    build_rk4_scan_split,
)
from jaxdrb.normalization import NormalizationInfo, apply_normalization
from jaxdrb.operators.fd2d import enforce_bc_relaxation_implicit, implicit_diffusion_fd_fft


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


def _apply_bc_relaxation_implicit(
    system: DRBSystem, y: DRBSystemState, dt: float, phi_guess: jnp.ndarray | None = None
) -> DRBSystemState:
    """Implicit operator-split update for stiff BC relaxation terms."""

    params = system.params
    geom = system.geom
    grid = getattr(geom, "grid", None)
    if grid is None or getattr(geom, "ndim", None) != 2:
        return y

    def _nu(val: float | None) -> float:
        base = float(params.bc_enforce_nu)
        if val is None:
            return base
        return float(val)

    nu_n = _nu(params.bc_enforce_nu_n)
    nu_omega = _nu(params.bc_enforce_nu_omega)
    nu_vpar_e = _nu(params.bc_enforce_nu_vpar_e)
    nu_vpar_i = _nu(params.bc_enforce_nu_vpar_i)
    nu_Te = _nu(params.bc_enforce_nu_Te)
    nu_Ti = _nu(params.bc_enforce_nu_Ti)
    nu_psi = _nu(params.bc_enforce_nu_psi)

    if (
        (nu_n == 0.0)
        and (nu_omega == 0.0)
        and (nu_vpar_e == 0.0)
        and (nu_vpar_i == 0.0)
        and (nu_Te == 0.0)
        and (nu_Ti == 0.0)
        and (nu_psi == 0.0)
        and (params.bc_enforce_nu_phi is None or float(params.bc_enforce_nu_phi) == 0.0)
    ):
        return y

    bcs = resolve_bcs(params, geom)
    dt = float(dt)
    dx = float(grid.dx)
    dy = float(grid.dy)

    n = enforce_bc_relaxation_implicit(y.n, dx=dx, dy=dy, bc=bcs.n, nu=nu_n, dt=dt)
    omega = enforce_bc_relaxation_implicit(y.omega, dx=dx, dy=dy, bc=bcs.omega, nu=nu_omega, dt=dt)
    vpar_e = enforce_bc_relaxation_implicit(
        y.vpar_e, dx=dx, dy=dy, bc=bcs.vpar_e, nu=nu_vpar_e, dt=dt
    )
    vpar_i = enforce_bc_relaxation_implicit(
        y.vpar_i, dx=dx, dy=dy, bc=bcs.vpar_i, nu=nu_vpar_i, dt=dt
    )
    Te = enforce_bc_relaxation_implicit(y.Te, dx=dx, dy=dy, bc=bcs.Te, nu=nu_Te, dt=dt)
    Ti = (
        None
        if y.Ti is None
        else enforce_bc_relaxation_implicit(y.Ti, dx=dx, dy=dy, bc=bcs.Ti, nu=nu_Ti, dt=dt)
    )
    psi = (
        None
        if y.psi is None
        else enforce_bc_relaxation_implicit(y.psi, dx=dx, dy=dy, bc=bcs.psi, nu=nu_psi, dt=dt)
    )

    return DRBSystemState(
        n=n,
        omega=omega,
        vpar_e=vpar_e,
        vpar_i=vpar_i,
        Te=Te,
        Ti=Ti,
        psi=psi,
        N=y.N,
    )


def _apply_phi_boundary_relaxation_implicit(
    system: DRBSystem, y: DRBSystemState, dt: float, phi_guess: jnp.ndarray | None = None
) -> DRBSystemState:
    """Implicit operator-split update that relaxes phi on the boundaries."""

    params = system.params
    if bool(params.phi_relax_in_rhs):
        return y
    nu_phi = params.bc_enforce_nu if params.bc_enforce_nu_phi is None else params.bc_enforce_nu_phi
    if nu_phi is None or float(nu_phi) == 0.0:
        return y

    geom = system.geom
    grid = getattr(geom, "grid", None)
    if grid is None or getattr(geom, "ndim", None) != 2:
        return y

    bcs = resolve_bcs(params, geom)
    dt = float(dt)
    dx = float(grid.dx)
    dy = float(grid.dy)

    phi = system._phi_from_omega(y.omega, n=y.n, Ti=y.Ti, Te=y.Te, phi_guess=phi_guess)

    if phi.ndim == 2:
        phi_rel = enforce_bc_relaxation_implicit(
            phi, dx=dx, dy=dy, bc=bcs.phi, nu=float(nu_phi), dt=dt
        )
    else:
        phi_rel = jax.vmap(
            lambda p: enforce_bc_relaxation_implicit(
                p, dx=dx, dy=dy, bc=bcs.phi, nu=float(nu_phi), dt=dt
            )
        )(phi)

    omega = system._omega_from_phi(phi_rel, n=y.n, Ti=y.Ti, Te=y.Te)

    return DRBSystemState(
        n=y.n,
        omega=omega,
        vpar_e=y.vpar_e,
        vpar_i=y.vpar_i,
        Te=y.Te,
        Ti=y.Ti,
        psi=y.psi,
        N=y.N,
    )


def _apply_sol_sheath_phi_implicit(
    system: DRBSystem, y: DRBSystemState, dt: float, phi_guess: jnp.ndarray | None = None
) -> DRBSystemState:
    params = system.params
    if not (
        params.sol_sheath_phi_on
        and params.sol_sheath_phi_dissipation_on
        and params.sol_sheath_phi_implicit
    ):
        return y
    if float(params.sol_parallel_loss_q) <= 0.0:
        return y
    model = str(params.sol_sheath_phi_model).lower()
    if model not in ("linear", "lin"):
        return y

    _, mask_open, _ = sol_masks(params, system.geom)
    if mask_open is None:
        return y

    n_phys = phys_n(params, y.n)
    Te_phys = phys_Te(params, y.Te)
    Te_floor = max(float(params.sol_sheath_phi_Te_floor), float(params.sol_Te_floor))
    Te_eff = jnp.maximum(Te_phys, Te_floor)
    n_pos = jnp.maximum(n_phys, float(params.sol_n_floor))
    cs = jnp.sqrt(Te_eff)
    gamma = float(params.sol_sheath_phi_coeff) / (2.0 * jnp.pi * float(params.sol_parallel_loss_q))
    lam = float(params.sol_sheath_phi_lambda)

    A = gamma * mask_open * n_pos * cs / Te_eff
    const = gamma * mask_open * n_pos * cs * lam

    dt = float(dt)
    if phi_guess is None:
        phi_guess = system._phi_from_omega(y.omega, n=n_phys, Ti=y.Ti, Te=Te_phys)

    b = y.omega + dt * const

    def matvec(phi):
        return system._omega_from_phi(phi, n=n_phys, Ti=y.Ti, Te=Te_phys) + dt * A * phi

    solver = str(params.sol_sheath_phi_implicit_solver).lower()
    tol = float(params.sol_sheath_phi_implicit_rtol)
    atol = float(params.sol_sheath_phi_implicit_atol)
    maxiter = int(params.sol_sheath_phi_implicit_maxiter)
    if solver == "cg":
        phi_new, _ = jspl.cg(matvec, b, x0=phi_guess, tol=tol, atol=atol, maxiter=maxiter)
    else:
        restart = int(params.sol_sheath_phi_implicit_restart)
        phi_new, _ = jspl.gmres(
            matvec,
            b,
            x0=phi_guess,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
        )

    omega_new = system._omega_from_phi(phi_new, n=n_phys, Ti=y.Ti, Te=Te_phys)

    return DRBSystemState(
        n=y.n,
        omega=omega_new,
        vpar_e=y.vpar_e,
        vpar_i=y.vpar_i,
        Te=y.Te,
        Ti=y.Ti,
        psi=y.psi,
        N=y.N,
    )


def _apply_diffusion_implicit(system: DRBSystem, y: DRBSystemState, dt: float) -> DRBSystemState:
    """Implicit update for perpendicular diffusion/biharmonic and linear drags."""

    params = system.params
    geom = system.geom

    grid = getattr(geom, "grid", None)
    if grid is None:
        return y
    perp = getattr(grid, "perp", grid)
    if not hasattr(perp, "dx") or not hasattr(perp, "dy"):
        return y

    bcs = resolve_bcs(params, geom)
    dx = float(perp.dx)
    dy = float(perp.dy)
    dt = float(dt)

    eigs = getattr(geom, "poisson_fd_fft_eigs", None)
    lam_x = None
    lam_y = None
    if eigs is not None and len(eigs) == 2:
        lam_x, lam_y = eigs

    def solve_plane(u, bc, D, D4, mu):
        return implicit_diffusion_fd_fft(
            u,
            dx=dx,
            dy=dy,
            bc=bc,
            dt=dt,
            D=D,
            D4=D4,
            mu=mu,
            lam_x=lam_x,
            lam_y=lam_y,
        )

    def solve_field(u, bc, D, D4, mu):
        if u is None:
            return None
        if u.ndim == 2:
            return solve_plane(u, bc, D, D4, mu)
        return jax.vmap(lambda p: solve_plane(p, bc, D, D4, mu))(u)

    n = solve_field(y.n, bcs.n, params.Dn, params.Dn4, params.mu_lin_n)
    mu_lin_omega = params.mu_lin_omega if bool(params.core_vorticity_damping_on) else 0.0
    omega = solve_field(y.omega, bcs.omega, params.DOmega, params.DOmega4, mu_lin_omega)
    Te = solve_field(y.Te, bcs.Te, params.DTe, params.DTe4, params.mu_lin_Te)
    Ti = solve_field(y.Ti, bcs.Ti, params.DTi, params.DTi4, 0.0) if y.Ti is not None else None
    psi = solve_field(y.psi, bcs.psi, params.Dpsi, params.Dpsi4, 0.0) if y.psi is not None else None

    # Linear vpar drags and collisional coupling (local 2x2 implicit solve).
    vpar_e = y.vpar_e
    vpar_i = y.vpar_i
    mu_e = float(params.mu_lin_vpar_e)
    mu_i = float(params.mu_lin_vpar_i)
    eta_eff = params.eta_par if params.eta_par != 0.0 else params.eta
    me = max(float(params.me_hat), 1e-12)
    nu_ei = float(eta_eff) / me if float(eta_eff) != 0.0 else 0.0
    if (mu_e != 0.0) or (mu_i != 0.0) or (nu_ei != 0.0):
        denom_i = 1.0 + dt * mu_i
        vpar_i = vpar_i / denom_i
        denom_e = 1.0 + dt * (mu_e + nu_ei)
        vpar_e = (vpar_e + dt * nu_ei * vpar_i) / denom_e

    return DRBSystemState(
        n=n,
        omega=omega,
        vpar_e=vpar_e,
        vpar_i=vpar_i,
        Te=Te,
        Ti=Ti,
        psi=psi,
        N=y.N,
    )


def _apply_parallel_implicit(
    system: DRBSystem,
    y: DRBSystemState,
    dt: float,
    phi_guess: jnp.ndarray | None = None,
) -> DRBSystemState:
    with jax.named_scope("parallel_implicit"):
        params = system.params
        geom = system.geom
        shape = y.n.shape
        if len(shape) != 3:
            return y
        if params.log_n or params.log_Te:
            return y
        grid = getattr(geom, "grid", None)
        if grid is None or getattr(grid, "open_field_line", False):
            return y
        if not bool(getattr(geom, "dpar_factor_const", False)):
            return y
        dpar_scalar = getattr(geom, "dpar_factor_scalar", None)
        if dpar_scalar is None:
            return y
        if phi_guess is None:
            return y

        nz = int(shape[0])
        dz = float(getattr(grid, "dz", 1.0))
        k = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=dz)
        # Match the centered-difference stencil used by geom.dpar:
        # d/dz -> i * sin(k*dz) / dz for periodic grids.
        D = 1j * jnp.sin(k * dz) / max(dz, 1e-12)
        D = D * float(dpar_scalar)
        D = D[:, None, None]
        dt = float(dt)

        n_hat = jnp.fft.fft(y.n, axis=0)
        Te_hat = jnp.fft.fft(y.Te, axis=0)
        v_hat = jnp.fft.fft(y.vpar_e, axis=0)
        phi_hat = jnp.fft.fft(phi_guess, axis=0)
        psi_hat = None
        if params.em_on and y.psi is not None:
            psi_hat = jnp.fft.fft(y.psi, axis=0)

        inv_me = 1.0 / max(float(params.me_hat), 1e-12)
        a = -inv_me
        b = -float(params.alpha_Te_ohm) * inv_me
        s = inv_me

        A = jnp.zeros((nz, 3, 3), dtype=n_hat.dtype)
        A = A.at[:, 0, 0].set(1.0)
        A = A.at[:, 1, 1].set(1.0)
        A = A.at[:, 2, 2].set(1.0)
        A = A.at[:, 0, 2].set(dt * D[:, 0, 0])
        A = A.at[:, 1, 2].set(dt * (2.0 / 3.0) * D[:, 0, 0])
        A = A.at[:, 2, 0].set(-dt * a * D[:, 0, 0])
        A = A.at[:, 2, 1].set(-dt * b * D[:, 0, 0])

        B0 = n_hat
        B1 = Te_hat
        source = dt * s * D * phi_hat
        if psi_hat is not None:
            source = source - dt * D * psi_hat
        B2 = v_hat + source
        B = jnp.stack([B0, B1, B2], axis=-1)

        A = A[:, None, None, :, :]
        U_new = jnp.linalg.solve(A, B[..., None])[..., 0]

        n_new = jnp.fft.ifft(U_new[..., 0], axis=0).real
        Te_new = jnp.fft.ifft(U_new[..., 1], axis=0).real
        v_new = jnp.fft.ifft(U_new[..., 2], axis=0).real

        v_i_new = y.vpar_i
        Ti_new = y.Ti
        if params.hot_ion_on and y.Ti is not None:
            Ti_hat = jnp.fft.fft(y.Ti, axis=0)
            v_i_hat = jnp.fft.fft(y.vpar_i, axis=0)
            tau_i = float(params.tau_i)
            A2 = jnp.zeros((nz, 2, 2), dtype=Ti_hat.dtype)
            A2 = A2.at[:, 0, 0].set(1.0)
            A2 = A2.at[:, 1, 1].set(1.0)
            A2 = A2.at[:, 0, 1].set(dt * (2.0 / 3.0) * D[:, 0, 0])
            A2 = A2.at[:, 1, 0].set(dt * tau_i * D[:, 0, 0])
            n_hat_new = jnp.fft.fft(n_new, axis=0)
            B2_0 = Ti_hat
            B2_1 = v_i_hat + dt * (-D * phi_hat - tau_i * D * n_hat_new)
            B2 = jnp.stack([B2_0, B2_1], axis=-1)
            A2 = A2[:, None, None, :, :]
            U2_new = jnp.linalg.solve(A2, B2[..., None])[..., 0]
            Ti_new = jnp.fft.ifft(U2_new[..., 0], axis=0).real
            v_i_new = jnp.fft.ifft(U2_new[..., 1], axis=0).real
        else:
            v_i_hat = jnp.fft.fft(y.vpar_i, axis=0)
            v_i_hat = v_i_hat - dt * D * phi_hat
            v_i_new = jnp.fft.ifft(v_i_hat, axis=0).real

        omega_hat = jnp.fft.fft(y.omega, axis=0)
        omega_hat = omega_hat + dt * D * (jnp.fft.fft(v_i_new, axis=0) - jnp.fft.fft(v_new, axis=0))
        omega_new = jnp.fft.ifft(omega_hat, axis=0).real

        return DRBSystemState(
            n=n_new,
            omega=omega_new,
            vpar_e=v_new,
            vpar_i=v_i_new,
            Te=Te_new,
            Ti=Ti_new,
            psi=y.psi,
            N=y.N,
        )


def _apply_stiff_implicit(
    system: DRBSystem,
    y: DRBSystemState,
    dt: float,
    phi_guess: jnp.ndarray | None = None,
    *,
    parallel_implicit: bool = True,
) -> DRBSystemState:
    y = _apply_diffusion_implicit(system, y, dt)
    if parallel_implicit:
        y = _apply_parallel_implicit(system, y, dt, phi_guess)
    y = _apply_bc_relaxation_implicit(system, y, dt, phi_guess)
    y = _apply_sol_sheath_phi_implicit(system, y, dt, phi_guess)
    y = _apply_phi_boundary_relaxation_implicit(system, y, dt, phi_guess)
    return y


def _build_diffrax_stiff_step(
    system: DRBSystem,
    time_cfg: dict[str, Any],
) -> Callable[[DRBSystemState, float, jnp.ndarray | None], DRBSystemState]:
    import diffrax as dfx

    solver_name = str(time_cfg.get("stiff_solver", "implicit_euler")).lower()
    if solver_name == "diffrax":
        solver_name = str(time_cfg.get("stiff_solver_name", "implicit_euler")).lower()
    solver_map = {
        "implicit_euler": dfx.ImplicitEuler,
        "kvaerno3": dfx.Kvaerno3,
        "kvaerno4": dfx.Kvaerno4,
        "kvaerno5": dfx.Kvaerno5,
    }
    solver_cls = solver_map.get(solver_name, dfx.ImplicitEuler)
    rtol = float(time_cfg.get("stiff_rtol", time_cfg.get("rtol", 1e-1)))
    atol = float(time_cfg.get("stiff_atol", time_cfg.get("atol", 1e-6)))
    solver_kwargs: dict[str, Any] = {}
    implicit_solvers = {"implicit_euler", "kvaerno3", "kvaerno4", "kvaerno5"}
    if solver_name in implicit_solvers:
        import lineax as lx
        import optimistix as optx

        linear_kind = str(time_cfg.get("stiff_linear_solver", "gmres")).lower()
        linear_max_steps = time_cfg.get("stiff_linear_max_steps", 25)
        linear_rtol = time_cfg.get("stiff_linear_rtol", rtol)
        linear_atol = time_cfg.get("stiff_linear_atol", atol)
        if linear_kind in ("auto", "default"):
            linear_solver = lx.AutoLinearSolver(well_posed=None)
        elif linear_kind in ("lu", "dense"):
            linear_solver = lx.LU()
        elif linear_kind in ("cholesky", "chol"):
            linear_solver = lx.Cholesky()
        elif linear_kind == "gmres":
            linear_solver = lx.GMRES(
                rtol=float(linear_rtol),
                atol=float(linear_atol),
                max_steps=None if linear_max_steps is None else int(linear_max_steps),
                restart=int(time_cfg.get("stiff_linear_restart", 20)),
                stagnation_iters=int(time_cfg.get("stiff_linear_stagnation_iters", 20)),
            )
        elif linear_kind in ("bicgstab", "bicg"):
            linear_solver = lx.BiCGStab(
                rtol=float(linear_rtol),
                atol=float(linear_atol),
                max_steps=None if linear_max_steps is None else int(linear_max_steps),
            )
        elif linear_kind in ("cg", "pcg"):
            linear_solver = lx.CG(
                rtol=float(linear_rtol),
                atol=float(linear_atol),
                max_steps=None if linear_max_steps is None else int(linear_max_steps),
            )
        else:
            raise ValueError(f"Unknown stiff_linear_solver: {linear_kind}")

        root_rtol = time_cfg.get("stiff_root_rtol", rtol)
        root_atol = time_cfg.get("stiff_root_atol", atol)
        root_kind = str(time_cfg.get("stiff_root_solver", "verychord")).lower()
        if root_kind in ("newton", "optimistix_newton"):
            root_finder = optx.Newton(
                rtol=float(root_rtol),
                atol=float(root_atol),
                linear_solver=linear_solver,
                kappa=float(time_cfg.get("stiff_root_kappa", 0.01)),
                cauchy_termination=bool(time_cfg.get("stiff_root_cauchy_termination", True)),
            )
        else:
            root_finder = dfx.VeryChord(
                rtol=float(root_rtol),
                atol=float(root_atol),
                linear_solver=linear_solver,
            )
        solver_kwargs["root_finder"] = root_finder
        solver_kwargs["root_find_max_steps"] = int(time_cfg.get("stiff_root_max_steps", 6))

    solver = solver_cls(**solver_kwargs)
    term = dfx.ODETerm(lambda t, y, args: system.rhs_stiff(t, y))
    saveat = dfx.SaveAt(t1=True)
    controller = dfx.ConstantStepSize()
    max_steps = time_cfg.get("stiff_max_steps", None)
    max_steps = int(max_steps) if max_steps is not None else None

    def step(y: DRBSystemState, dt_step: float, phi_guess: jnp.ndarray | None) -> DRBSystemState:
        _ = phi_guess
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=float(dt_step),
            dt0=float(dt_step),
            y0=y,
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=max_steps,
            adjoint=dfx.DirectAdjoint(),
            progress_meter=dfx.NoProgressMeter(),
            args=None,
        )
        return sol.ys

    return step


def _mixmode_phases(seed: float) -> np.ndarray:
    """Match BOUT++ mixmode phase generation."""

    def gen_rand(val: float) -> float:
        if val < 0.0:
            val = -val
        niter = 11 + (23 + int(round(val))) % 79
        a = 0.01
        b = 1.23456789
        x = (a + (val % b)) / (b + 2.0 * a)
        for _ in range(niter):
            x = 3.99 * x * (1.0 - x)
        return float(x)

    phases = np.zeros(14, dtype=np.float64)
    for i in range(14):
        phases[i] = np.pi * (2.0 * gen_rand(seed + float(i)) - 1.0)
    return phases


def _mixmode(x: jnp.ndarray, seed: float) -> jnp.ndarray:
    phases = jnp.asarray(_mixmode_phases(seed), dtype=x.dtype)
    result = jnp.zeros_like(x)
    for i in range(14):
        weight = 1.0 / (1.0 + abs(i - 4.0)) ** 2
        result = result + weight * jnp.cos(float(i) * x + phases[i])
    return result


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
    transport = cfg.get("transport", {})
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
    params = update_params_from_dict(
        params, _merge_params(physics, transport, closures, numerics, terms, bc)
    )

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

    def _perp_xy(
        *, centered: bool = False, x_mode: str = "grid"
    ) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
        grid = getattr(geom, "grid", None)
        if grid is None:
            return None, None
        if hasattr(grid, "perp"):
            x_arr = jnp.asarray(grid.perp.x)
            y_arr = jnp.asarray(grid.perp.y)
            bc = grid.perp.bc
            nx = int(grid.perp.nx)
        elif hasattr(grid, "x") and hasattr(grid, "y"):
            x_arr = jnp.asarray(grid.x)
            y_arr = jnp.asarray(grid.y)
            bc = getattr(grid, "bc", None)
            nx = int(getattr(grid, "nx", x_arr.size))
        elif hasattr(grid, "nx") and hasattr(grid, "ny"):
            x_arr = jnp.asarray(grid.x0) + jnp.asarray(grid.dx) * jnp.arange(grid.nx)
            y_arr = jnp.asarray(grid.y0) + jnp.asarray(grid.dy) * jnp.arange(grid.ny)
            bc = getattr(grid, "bc", None)
            nx = int(grid.nx)
        else:
            return None, None
        if str(x_mode).lower() in ("bout", "index"):
            endpoint = True
            if bc is not None and getattr(bc, "kind_x", 0) == 0:
                endpoint = False
            x_arr = jnp.linspace(0.0, 1.0, nx, endpoint=endpoint)
        if centered:
            x_center = 0.5 * (x_arr[0] + x_arr[-1])
            x_arr = x_arr - x_center
        if len(shape) == 2:
            return x_arr[:, None], y_arr[None, :]
        return x_arr[None, :, None], y_arr[None, None, :]

    def _par_z(mode: str = "grid") -> jnp.ndarray | None:
        mode = str(mode).lower()
        if len(shape) != 3:
            return None
        if mode in ("bout", "index"):
            nz = int(shape[0])
            grid = getattr(geom, "grid", None)
            endpoint = False
            if grid is not None and getattr(grid, "open_field_line", False):
                endpoint = True
            # BOUT++ mixmode uses normalized z in [0, 1)
            z_arr = jnp.linspace(0.0, 1.0, nz, endpoint=endpoint)
        else:
            grid = getattr(geom, "grid", None)
            if grid is None or not hasattr(grid, "z"):
                return None
            z_arr = jnp.asarray(grid.z)
        return z_arr[:, None, None]

    def _apply_x_profile(
        field: jnp.ndarray,
        *,
        profile: str,
        prefix: str,
        xg: jnp.ndarray | None,
    ) -> jnp.ndarray:
        if xg is None:
            return field
        if profile in ("linear_x", "affine_x", "linear"):
            offset = float(init.get(f"{prefix}_offset", float(jnp.mean(field))))
            slope = float(init.get(f"{prefix}_slope", 0.0))
            x_ref = float(init.get(f"{prefix}_xref", 0.0))
            return offset + slope * (xg - x_ref)
        if profile in ("parabolic_x", "quadratic_x"):
            a0 = float(init.get(f"{prefix}_a0", float(jnp.mean(field))))
            a1 = float(init.get(f"{prefix}_a1", 0.0))
            a2 = float(init.get(f"{prefix}_a2", 0.0))
            x_ref = float(init.get(f"{prefix}_xref", 0.0))
            xr = xg - x_ref
            return a0 + a1 * xr + a2 * xr * xr
        if profile in ("gaussian_x", "gauss_x", "gaussian"):
            amp = float(init.get(f"{prefix}_amp", init.get(f"{prefix}_amplitude", 0.0)))
            width = max(float(init.get(f"{prefix}_width", 1.0)), 1e-12)
            x0 = float(init.get(f"{prefix}_x0", 0.0))
            return field + amp * jnp.exp(-(((xg - x0) / width) ** 2))
        return field

    n0 = float(init.get("n0", init.get("n_base", 0.0)))
    Te0 = float(init.get("Te0", init.get("Te_base", 0.0)))
    omega0 = float(init.get("omega0", 0.0))
    vpar_e0 = float(init.get("vpar_e0", 0.0))
    vpar_i0 = float(init.get("vpar_i0", 0.0))

    n_phys = jnp.full_like(state.n, n0)
    Te_phys = jnp.full_like(state.n, Te0)

    x_centered = bool(init.get("x_centered", init.get("perp_centered", False)))
    x_mode = str(init.get("x_mode", "grid")).lower()
    z_mode = str(init.get("z_mode", "grid")).lower()

    n_profile = str(init.get("n_profile", init.get("profile_n", ""))).lower()
    if n_profile in ("linear_x", "affine_x", "linear"):
        xg, _ = _perp_xy(centered=x_centered, x_mode=x_mode)
        n_phys = _apply_x_profile(n_phys, profile=n_profile, prefix="n_profile", xg=xg)
    elif n_profile in ("parabolic_x", "quadratic_x"):
        xg, _ = _perp_xy(centered=x_centered, x_mode=x_mode)
        n_phys = _apply_x_profile(n_phys, profile=n_profile, prefix="n_profile", xg=xg)
    elif n_profile in ("gaussian_x", "gauss_x", "gaussian"):
        xg, _ = _perp_xy(centered=x_centered, x_mode=x_mode)
        n_phys = _apply_x_profile(n_phys, profile=n_profile, prefix="n_profile", xg=xg)
    elif n_profile in ("gaussian_mixmode", "gaussian_mixmode_z", "gaussian_mixmode_reference"):
        xg, yg = _perp_xy(centered=x_centered, x_mode=x_mode)
        zg = _par_z(z_mode)
        if xg is not None:
            amp = float(init.get("n_profile_amp", init.get("n_profile_amplitude", 0.0)))
            width = float(init.get("n_profile_width", 1.0))
            x0 = float(init.get("n_profile_x0", 0.0))
            if width <= 0.0:
                width = 1.0
            n_phys = n_phys + amp * jnp.exp(-(((xg - x0) / width) ** 2))
        if xg is not None and (zg is not None or yg is not None):
            mix_amp = float(init.get("mixmode_amp", 0.0))
            mix_seed = float(init.get("mixmode_seed", 0.5))
            terms = init.get("mixmode_terms", ["z", "4z-x"])
            if isinstance(terms, str):
                terms = [terms]
            mixmode_mode = str(init.get("mixmode_mode", "jax")).lower()
            z_arg = zg
            y_arg = yg
            if mixmode_mode in ("reference", "bout", "boutpp"):
                z_arg = yg
                y_arg = zg
            mix = jnp.zeros_like(n_phys)
            for term in terms:
                key = str(term).replace(" ", "").lower()
                if key in ("z",) and z_arg is not None:
                    arg = z_arg
                elif key in ("x",):
                    arg = xg
                elif key in ("4z-x", "4*z-x") and z_arg is not None:
                    arg = 4.0 * z_arg - xg
                elif key in ("x-z", "x-zed") and z_arg is not None:
                    arg = xg - z_arg
                elif key in ("z-x",) and z_arg is not None:
                    arg = z_arg - xg
                elif key in ("y",) and y_arg is not None:
                    arg = y_arg
                elif key in ("4y-x", "4*y-x") and y_arg is not None:
                    arg = 4.0 * y_arg - xg
                elif key in ("x-y",) and y_arg is not None:
                    arg = xg - y_arg
                elif key in ("y-x",) and y_arg is not None:
                    arg = y_arg - xg
                else:
                    continue
                mix = mix + _mixmode(arg, seed=mix_seed)
            n_phys = n_phys + mix_amp * mix

    # Optional deterministic mixmode perturbation over any base profile.
    n_mix_amp = float(init.get("n_mixmode_amp", init.get("mixmode_amp_global", 0.0)))
    if n_mix_amp != 0.0:
        xg, yg = _perp_xy(centered=x_centered, x_mode=x_mode)
        zg = _par_z(z_mode)
        if xg is not None and (zg is not None or yg is not None):
            mix_seed = float(init.get("n_mixmode_seed", init.get("mixmode_seed", 0.5)))
            terms = init.get("n_mixmode_terms", ["x-z"])
            if isinstance(terms, str):
                terms = [terms]
            mixmode_mode = str(init.get("n_mixmode_mode", init.get("mixmode_mode", "jax"))).lower()
            z_arg = zg
            y_arg = yg
            if mixmode_mode in ("reference", "bout", "boutpp"):
                z_arg = yg
                y_arg = zg
            mix = jnp.zeros_like(n_phys)
            for term in terms:
                key = str(term).replace(" ", "").lower()
                if key in ("z",) and z_arg is not None:
                    arg = z_arg
                elif key in ("x",):
                    arg = xg
                elif key in ("4z-x", "4*z-x") and z_arg is not None:
                    arg = 4.0 * z_arg - xg
                elif key in ("x-z", "x-zed") and z_arg is not None:
                    arg = xg - z_arg
                elif key in ("z-x",) and z_arg is not None:
                    arg = z_arg - xg
                elif key in ("y",) and y_arg is not None:
                    arg = y_arg
                elif key in ("4y-x", "4*y-x") and y_arg is not None:
                    arg = 4.0 * y_arg - xg
                elif key in ("x-y",) and y_arg is not None:
                    arg = xg - y_arg
                elif key in ("y-x",) and y_arg is not None:
                    arg = y_arg - xg
                else:
                    continue
                mix = mix + _mixmode(arg, seed=mix_seed)
            n_phys = n_phys + n_mix_amp * mix

    xg, _ = _perp_xy(centered=x_centered, x_mode=x_mode)
    p_profile = str(
        init.get("p_profile", init.get("profile_p", init.get("pressure_profile", "")))
    ).lower()
    p_phys = None
    if p_profile:
        p_base = jnp.asarray(n_phys * Te_phys)
        p_phys = _apply_x_profile(p_base, profile=p_profile, prefix="p_profile", xg=xg)

    Te_profile = str(init.get("Te_profile", init.get("profile_Te", ""))).lower()
    Te_from_pressure = bool(init.get("Te_pressure_consistent", False)) or Te_profile in (
        "from_pressure",
        "pressure_consistent",
        "pressure_over_n",
    )
    if Te_from_pressure:
        if p_phys is None:
            p_phys = jnp.asarray(n_phys * Te_phys)
        Te_phys = p_phys / jnp.maximum(n_phys, 1e-12)
    elif Te_profile:
        Te_phys = _apply_x_profile(Te_phys, profile=Te_profile, prefix="Te_profile", xg=xg)

    noise_amp = float(init.get("amplitude", init.get("noise_amplitude", 0.0)))
    noise_mode = str(init.get("noise_mode", "state")).lower()
    noise_fields = init.get("noise_fields", ["n", "omega", "Te"])
    if isinstance(noise_fields, str):
        noise_fields = [noise_fields]

    if noise_amp != 0.0:
        seed = int(init.get("seed", 0))
        key = jax.random.PRNGKey(seed)
        noise = noise_amp * jax.random.normal(key, shape=state.n.shape, dtype=state.n.dtype)
        if "n" in noise_fields:
            if noise_mode == "physical":
                n_phys = n_phys + noise
            else:
                n_phys = n_phys + 0.0
        if "Te" in noise_fields:
            if noise_mode == "physical":
                if not Te_from_pressure:
                    Te_phys = Te_phys + noise
            else:
                Te_phys = Te_phys + 0.0

    if Te_from_pressure:
        if p_phys is None:
            p_phys = jnp.asarray(n_phys * Te_phys)
        Te_phys = p_phys / jnp.maximum(n_phys, 1e-12)

    Ti_phys = None
    if sys_params.hot_ion_on and state.Ti is not None:
        Ti0 = float(init.get("Ti0", init.get("Ti_base", Te0)))
        Ti_phys = jnp.full_like(state.n, Ti0)
        Ti_profile = str(init.get("Ti_profile", "")).lower()
        if Ti_profile in ("from_te", "te", "match_te", "same"):
            Ti_phys = Te_phys
        elif Ti_profile in (
            "from_pressure",
            "pressure_consistent",
            "pressure_over_n",
        ):
            if p_phys is None:
                p_phys = jnp.asarray(n_phys * Te_phys)
            Ti_phys = p_phys / jnp.maximum(n_phys, 1e-12)
        elif Ti_profile:
            xg, _ = _perp_xy(centered=x_centered, x_mode=x_mode)
            Ti_phys = _apply_x_profile(Ti_phys, profile=Ti_profile, prefix="Ti_profile", xg=xg)
        else:
            if bool(init.get("Ti_from_Te", True)):
                Ti_phys = Te_phys

    # Keep all state channels shape-consistent even when profiles are x-only
    # and only a subset of fields receives perturbations.
    n_phys = jnp.broadcast_to(n_phys, state.n.shape)
    Te_phys = jnp.broadcast_to(Te_phys, state.n.shape)
    if Ti_phys is not None:
        Ti_phys = jnp.broadcast_to(Ti_phys, state.n.shape)

    n_floor = max(float(sys_params.n0_min), 1e-12)
    Te_floor = max(float(sys_params.sol_Te_floor), 1e-12)

    if sys_params.log_n:
        n_state = jnp.log(jnp.maximum(n_phys, n_floor))
    else:
        n_state = n_phys
    if sys_params.log_Te:
        Te_state = jnp.log(jnp.maximum(Te_phys, Te_floor))
    else:
        Te_state = Te_phys

    omega_state = jnp.full_like(state.n, omega0)
    vpar_e_state = jnp.full_like(state.n, vpar_e0)
    vpar_i_state = jnp.full_like(state.n, vpar_i0)
    Ti_state = Ti_phys if Ti_phys is not None else None

    if noise_amp != 0.0:
        if noise_mode != "physical":
            if "n" in noise_fields:
                n_state = n_state + noise
            if "Te" in noise_fields:
                Te_state = Te_state + noise
        if "omega" in noise_fields:
            omega_state = omega_state + noise
        if "vpar_e" in noise_fields:
            vpar_e_state = vpar_e_state + noise
        if "vpar_i" in noise_fields:
            vpar_i_state = vpar_i_state + noise

    state = DRBSystemState(
        n=n_state,
        omega=omega_state,
        vpar_e=vpar_e_state,
        vpar_i=vpar_i_state,
        Te=Te_state,
        Ti=Ti_state,
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
    use_phi_guess: bool = True,
    use_phi_guess_only: bool = False,
    trace_stats: bool = False,
    trace_enstrophy: bool = False,
    save_fields: bool = False,
    snapshot_fields: tuple[str, ...] = (),
) -> Callable[[float, DRBSystemState], tuple]:
    def diag(t, y, args=None, *, phi_guess=None):
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

        def _abs_stats(arr):
            return jnp.mean(jnp.abs(arr)), jnp.max(jnp.abs(arr))

        use_phi_rms = not (mode in ("basic", "no_phi", "rms_only", "light") or phi_every <= 0)
        idx = jnp.asarray(0)
        use_phi = jnp.asarray(True)
        if use_phi_rms and phi_every > 1:
            idx = jnp.round(t / dt_save).astype(jnp.int32)
            use_phi = (idx % phi_every) == 0

        need_phi = trace_stats or use_phi_rms or ("phi" in snapshot_fields)
        phi_local = None
        if need_phi:
            if use_phi_guess and phi_guess is not None:
                phi_local = phi_guess
            elif use_phi_guess_only:
                phi_local = jnp.zeros_like(y.omega)
            else:
                phi_local = system._phi_from_omega(
                    y.omega, n=n_phys, Ti=y.Ti, Te=Te_phys, phi_guess=phi_guess
                )

        if not use_phi_rms:
            zero = jnp.asarray(0.0)
            base = (
                jnp.asarray(t),
                jnp.sqrt(jnp.mean(n_phys**2)),
                jnp.sqrt(jnp.mean(Te_phys**2)),
                jnp.sqrt(jnp.mean(y.omega**2)),
                zero,
                point_n,
                point_Te,
                zero,
            )
        else:

            def _compute_phi(_):
                if phi_local is None:
                    zero = jnp.asarray(0.0)
                    return zero, zero
                if n_phys.ndim == 2:
                    return jnp.sqrt(jnp.mean(phi_local**2)), phi_local[x0, y0]
                return jnp.sqrt(jnp.mean(phi_local**2)), phi_local[z0, x0, y0]

            def _skip_phi(_):
                zero = jnp.asarray(0.0)
                return zero, zero

            rms_phi, point_phi = jax.lax.cond(use_phi, _compute_phi, _skip_phi, operand=None)
            base = (
                jnp.asarray(t),
                jnp.sqrt(jnp.mean(n_phys**2)),
                jnp.sqrt(jnp.mean(Te_phys**2)),
                jnp.sqrt(jnp.mean(y.omega**2)),
                rms_phi,
                point_n,
                point_Te,
                point_phi,
            )

        if not trace_stats:
            extras = ()
        else:
            if phi_local is None:
                phi_local = jnp.zeros_like(y.omega)

            n_mean, n_max = _abs_stats(n_phys)
            Te_mean, Te_max = _abs_stats(Te_phys)
            ve_mean, ve_max = _abs_stats(y.vpar_e)
            vi_mean, vi_max = _abs_stats(y.vpar_i)
            om_mean, om_max = _abs_stats(y.omega)
            phi_mean, phi_max = _abs_stats(phi_local)
            extras = (
                n_mean,
                n_max,
                Te_mean,
                Te_max,
                ve_mean,
                ve_max,
                vi_mean,
                vi_max,
                om_mean,
                om_max,
                phi_mean,
                phi_max,
            )
            if trace_enstrophy:
                enstrophy = 0.5 * jnp.mean(y.omega**2)
                extras = extras + (enstrophy,)

        snapshots: tuple = ()
        if save_fields:
            field_map = {
                "n": n_phys,
                "Te": Te_phys,
                "omega": y.omega,
                "vpar_e": y.vpar_e,
                "vpar_i": y.vpar_i,
                "Ti": y.Ti if y.Ti is not None else jnp.zeros_like(y.omega),
                "psi": y.psi if y.psi is not None else jnp.zeros_like(y.omega),
                "N": y.N if y.N is not None else jnp.zeros_like(y.omega),
            }
            if "phi" in snapshot_fields:
                if phi_local is None:
                    phi_local = system._phi_from_omega(
                        y.omega, n=n_phys, Ti=y.Ti, Te=Te_phys, phi_guess=phi_guess
                    )
                field_map["phi"] = phi_local
            snapshots = tuple(field_map[name] for name in snapshot_fields if name in field_map)

        return base + extras + snapshots

    return diag


def run_simulation(cfg: dict[str, Any], *, as_numpy: bool | None = None) -> RunResult:
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
    scan_remat = bool(time_cfg.get("scan_remat", False))
    diag_mode = str(time_cfg.get("diag_mode", "full")).lower()
    diag_phi_every = int(time_cfg.get("diag_phi_every", 1))
    diag_phi_use_guess = bool(time_cfg.get("diag_phi_use_guess", True))
    diag_phi_use_guess_only = bool(time_cfg.get("diag_phi_use_guess_only", False))
    trace_stats = bool(time_cfg.get("trace_stats", False) or time_cfg.get("blowup_trace", False))
    trace_enstrophy = bool(time_cfg.get("trace_enstrophy", False))
    save_fields = bool(time_cfg.get("save_fields", False))
    snapshot_fields_cfg = time_cfg.get("snapshot_fields", ("n", "omega", "Te", "phi"))
    if isinstance(snapshot_fields_cfg, (list, tuple)):
        snapshot_fields = tuple(str(item) for item in snapshot_fields_cfg)
    else:
        snapshot_fields = (str(snapshot_fields_cfg),)
    warm_start = bool(time_cfg.get("poisson_warm_start", True))
    carry_phi = bool(time_cfg.get("carry_phi", False))
    track_iters = bool(time_cfg.get("poisson_track_iters", False))
    if as_numpy is None:
        return_numpy = bool(time_cfg.get("return_numpy", False))
    else:
        return_numpy = bool(as_numpy)

    shape = state.n.shape
    trace_stats_out: dict[str, Any] = {}

    def _unpack_diag(diag_series):
        items = list(diag_series)
        it_mean = None
        it_max = None
        if track_iters:
            it_max = items.pop()
            it_mean = items.pop()
        t, rms_n, rms_Te, rms_omega, rms_phi, point_n, point_Te, point_phi = items[:8]
        extra = items[8:]
        trace = {}
        if trace_stats:
            keys = [
                "trace_mean_abs_n",
                "trace_max_abs_n",
                "trace_mean_abs_Te",
                "trace_max_abs_Te",
                "trace_mean_abs_vpar_e",
                "trace_max_abs_vpar_e",
                "trace_mean_abs_vpar_i",
                "trace_max_abs_vpar_i",
                "trace_mean_abs_omega",
                "trace_max_abs_omega",
                "trace_mean_abs_phi",
                "trace_max_abs_phi",
            ]
            if trace_enstrophy:
                keys.append("trace_enstrophy")
            for key, val in zip(keys, extra):
                trace[key] = val
            extra = extra[len(keys) :]
        snapshots: dict[str, Any] = {}
        if save_fields and snapshot_fields:
            for name in snapshot_fields:
                if not extra:
                    break
                snapshots[f"snapshots_{name}"] = extra.pop(0)
        return (
            t,
            rms_n,
            rms_Te,
            rms_omega,
            rms_phi,
            point_n,
            point_Te,
            point_phi,
            it_mean,
            it_max,
            trace,
            snapshots,
        )

    if len(shape) == 2:
        nx, ny = shape
        point_idx = tuple(time_cfg.get("point_idx", (nx // 2, ny // 2)))
    else:
        nz, nx, ny = shape
        point_idx = tuple(time_cfg.get("point_idx", (nz // 2, nx // 2, ny // 2)))

    if method in ("rk4", "rk4_scan", "fixed"):
        dt_save = float(dt * save_every) if save_every > 0 else float(dt)
        diag_fn = _diagnostic_fn(
            system,
            point_idx,
            mode=diag_mode,
            phi_every=diag_phi_every,
            dt_save=dt_save,
            use_phi_guess=diag_phi_use_guess,
            use_phi_guess_only=diag_phi_use_guess_only,
            trace_stats=trace_stats,
            trace_enstrophy=trace_enstrophy,
            save_fields=save_fields,
            snapshot_fields=snapshot_fields,
        )
        if remat:
            diag_fn = jax.checkpoint(diag_fn)
        if track_iters and carry_phi and not warm_start:
            runner, nsave, rem = build_rk4_scan_cached_iters_phi(
                system.rhs_with_phi_iters,
                dt,
                nsteps,
                save_every,
                diag_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
                warm_start=False,
            )
        elif track_iters:
            runner, nsave, rem = build_rk4_scan_cached_iters(
                system.rhs_with_phi_iters,
                dt,
                nsteps,
                save_every,
                diag_fn,
                rhs_remat=remat,
                warm_start=warm_start,
                scan_remat=scan_remat,
            )
        elif carry_phi and not warm_start:
            runner, nsave, rem = build_rk4_scan_cached_phi(
                system.rhs_with_phi,
                dt,
                nsteps,
                save_every,
                diag_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
                warm_start=False,
            )
        elif warm_start:
            runner, nsave, rem = build_rk4_scan_cached(
                system.rhs_with_phi,
                dt,
                nsteps,
                save_every,
                diag_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
            )
        else:
            runner, nsave, rem = build_rk4_scan(
                system.rhs, dt, nsteps, save_every, diag_fn, rhs_remat=remat, scan_remat=scan_remat
            )
        final_state, diag_series = runner(state)
        times = np.arange(nsave, dtype=np.float64) * (save_every * dt)
        if rem > 0:
            times[-1] = nsteps * dt
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
            trace_stats_out,
            snapshots_out,
        ) = _unpack_diag(diag_series)
    elif method in ("rk4_split_bc", "split_bc", "rk4_bc"):
        dt_save = float(dt * save_every) if save_every > 0 else float(dt)
        diag_fn = _diagnostic_fn(
            system,
            point_idx,
            mode=diag_mode,
            phi_every=diag_phi_every,
            dt_save=dt_save,
            use_phi_guess=diag_phi_use_guess,
            use_phi_guess_only=diag_phi_use_guess_only,
            trace_stats=trace_stats,
            trace_enstrophy=trace_enstrophy,
            save_fields=save_fields,
            snapshot_fields=snapshot_fields,
        )
        if remat:
            diag_fn = jax.checkpoint(diag_fn)

        def bc_fn(y, dt_step, phi_guess):
            return _apply_bc_relaxation_implicit(system, y, dt_step, phi_guess)

        if track_iters and carry_phi and not warm_start:
            runner, nsave, rem = build_rk4_scan_cached_iters_split_phi(
                system.rhs_explicit_with_phi_iters,
                dt,
                nsteps,
                save_every,
                diag_fn,
                bc_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
                warm_start=False,
            )
        elif track_iters:
            runner, nsave, rem = build_rk4_scan_cached_iters_split(
                system.rhs_explicit_with_phi_iters,
                dt,
                nsteps,
                save_every,
                diag_fn,
                bc_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
            )
        elif carry_phi and not warm_start:
            runner, nsave, rem = build_rk4_scan_cached_split_phi(
                system.rhs_explicit_with_phi,
                dt,
                nsteps,
                save_every,
                diag_fn,
                bc_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
                warm_start=False,
            )
        elif warm_start:
            runner, nsave, rem = build_rk4_scan_cached_split(
                system.rhs_explicit_with_phi,
                dt,
                nsteps,
                save_every,
                diag_fn,
                bc_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
            )
        else:
            runner, nsave, rem = build_rk4_scan_split(
                system.rhs_explicit,
                dt,
                nsteps,
                save_every,
                diag_fn,
                bc_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
            )
        final_state, diag_series = runner(state)
        times = np.arange(nsave, dtype=np.float64) * (save_every * dt)
        if rem > 0:
            times[-1] = nsteps * dt
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
            trace_stats_out,
            snapshots_out,
        ) = _unpack_diag(diag_series)
    elif method in ("rk4_imex", "rk4_imex_diffusion", "imex_diffusion"):
        dt_save = float(dt * save_every) if save_every > 0 else float(dt)
        diag_fn = _diagnostic_fn(
            system,
            point_idx,
            mode=diag_mode,
            phi_every=diag_phi_every,
            dt_save=dt_save,
            use_phi_guess=diag_phi_use_guess,
            use_phi_guess_only=diag_phi_use_guess_only,
            trace_stats=trace_stats,
            trace_enstrophy=trace_enstrophy,
            save_fields=save_fields,
            snapshot_fields=snapshot_fields,
        )
        if remat:
            diag_fn = jax.checkpoint(diag_fn)

        parallel_implicit = bool(time_cfg.get("parallel_implicit", True))
        if parallel_implicit:
            from jaxdrb.core.terms.registry import build_scheduler_from_names, split_term_schedule

            explicit_names, _ = split_term_schedule(system.params)
            explicit_names = tuple(name for name in explicit_names if name != "parallel")
            if (
                not bool(system.params.phi_relax_in_rhs)
                and (
                    float(system.params.phi_par_dissipation) != 0.0
                    or float(system.params.vort_par_dissipation) != 0.0
                )
                and "extra_dissipation" not in explicit_names
            ):
                explicit_names = tuple(list(explicit_names) + ["extra_dissipation"])
            object.__setattr__(
                system, "scheduler_explicit", build_scheduler_from_names(explicit_names)
            )
            if not warm_start:
                warm_start = True

        def bc_fn(y, dt_step, phi_guess):
            return _apply_stiff_implicit(
                system, y, dt_step, phi_guess, parallel_implicit=parallel_implicit
            )

        if track_iters and carry_phi and not warm_start:
            runner, nsave, rem = build_rk4_scan_cached_iters_split_phi(
                system.rhs_explicit_with_phi_iters,
                dt,
                nsteps,
                save_every,
                diag_fn,
                bc_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
                warm_start=False,
            )
        elif track_iters:
            runner, nsave, rem = build_rk4_scan_cached_iters_split(
                system.rhs_explicit_with_phi_iters,
                dt,
                nsteps,
                save_every,
                diag_fn,
                bc_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
            )
        elif carry_phi and not warm_start:
            runner, nsave, rem = build_rk4_scan_cached_split_phi(
                system.rhs_explicit_with_phi,
                dt,
                nsteps,
                save_every,
                diag_fn,
                bc_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
                warm_start=False,
            )
        elif warm_start:
            runner, nsave, rem = build_rk4_scan_cached_split(
                system.rhs_explicit_with_phi,
                dt,
                nsteps,
                save_every,
                diag_fn,
                bc_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
            )
        else:
            runner, nsave, rem = build_rk4_scan_split(
                system.rhs_explicit,
                dt,
                nsteps,
                save_every,
                diag_fn,
                bc_fn,
                rhs_remat=remat,
                scan_remat=scan_remat,
            )
        final_state, diag_series = runner(state)
        times = np.arange(nsave, dtype=np.float64) * (save_every * dt)
        if rem > 0:
            times[-1] = nsteps * dt
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
            trace_stats_out,
            snapshots_out,
        ) = _unpack_diag(diag_series)
    elif method in ("rk4_imex_strang", "imex_strang"):
        dt_save = float(dt * save_every) if save_every > 0 else float(dt)
        diag_fn = _diagnostic_fn(
            system,
            point_idx,
            mode=diag_mode,
            phi_every=diag_phi_every,
            dt_save=dt_save,
            use_phi_guess=diag_phi_use_guess,
            use_phi_guess_only=diag_phi_use_guess_only,
            trace_stats=trace_stats,
            trace_enstrophy=trace_enstrophy,
            save_fields=save_fields,
            snapshot_fields=snapshot_fields,
        )
        if remat:
            diag_fn = jax.checkpoint(diag_fn)

        stiff_solver = str(time_cfg.get("stiff_solver", "analytic")).lower()
        use_diffrax_stiff = stiff_solver in (
            "diffrax",
            "implicit_euler",
            "kvaerno3",
            "kvaerno4",
            "kvaerno5",
        )

        parallel_implicit = bool(time_cfg.get("parallel_implicit", True))
        if parallel_implicit:
            from jaxdrb.core.terms.registry import build_scheduler_from_names, split_term_schedule

            explicit_names, stiff_names = split_term_schedule(system.params)
            explicit_names = tuple(name for name in explicit_names if name != "parallel")
            object.__setattr__(
                system, "scheduler_explicit", build_scheduler_from_names(explicit_names)
            )
            if use_diffrax_stiff and "parallel" not in stiff_names:
                stiff_names = tuple(stiff_names) + ("parallel",)
                object.__setattr__(
                    system, "scheduler_stiff", build_scheduler_from_names(stiff_names)
                )
            if not warm_start:
                warm_start = True

        if track_iters:
            track_iters = False

        if use_diffrax_stiff:
            stiff_fn = _build_diffrax_stiff_step(system, time_cfg)
        else:

            def stiff_fn(y, dt_step, phi_guess):
                return _apply_stiff_implicit(
                    system, y, dt_step, phi_guess, parallel_implicit=parallel_implicit
                )

        runner, nsave, rem = build_rk4_scan_imex_strang(
            system.rhs_explicit_with_phi,
            stiff_fn,
            dt,
            nsteps,
            save_every,
            diag_fn,
            rhs_remat=remat,
            scan_remat=scan_remat,
            warm_start=warm_start,
            carry_phi=carry_phi,
            jit=bool(time_cfg.get("jit", True)),
        )
        final_state, diag_series = runner(state)
        times = np.arange(nsave, dtype=np.float64) * (save_every * dt)
        if rem > 0:
            times[-1] = nsteps * dt
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
            trace_stats_out,
            snapshots_out,
        ) = _unpack_diag(diag_series)
    elif method in ("diffrax", "dopri8", "tsit5"):
        import diffrax as dfx

        if track_iters:
            track_iters = False

        solver_map = {
            "dopri8": dfx.Dopri8,
            "dopri5": dfx.Dopri5,
            "tsit5": dfx.Tsit5,
            "euler": dfx.Euler,
            "implicit_euler": dfx.ImplicitEuler,
            "kvaerno3": dfx.Kvaerno3,
            "kvaerno4": dfx.Kvaerno4,
            "kvaerno5": dfx.Kvaerno5,
            "kencarp3": dfx.KenCarp3,
            "kencarp4": dfx.KenCarp4,
            "kencarp5": dfx.KenCarp5,
        }
        solver_cls = solver_map.get(solver_name, dfx.Dopri8)
        solver_kwargs = {}
        implicit_solvers = {
            "implicit_euler",
            "kvaerno3",
            "kvaerno4",
            "kvaerno5",
            "kencarp3",
            "kencarp4",
            "kencarp5",
        }
        if solver_name in implicit_solvers:
            import lineax as lx
            import optimistix as optx

            linear_kind = str(time_cfg.get("imex_linear_solver", "auto")).lower()
            linear_max_steps = time_cfg.get("imex_linear_max_steps", 50)
            linear_rtol = time_cfg.get("imex_linear_rtol", rtol)
            linear_atol = time_cfg.get("imex_linear_atol", atol)
            if linear_kind in ("auto", "default"):
                linear_solver = lx.AutoLinearSolver(well_posed=None)
            elif linear_kind in ("lu", "dense"):
                linear_solver = lx.LU()
            elif linear_kind in ("cholesky", "chol"):
                linear_solver = lx.Cholesky()
            elif linear_kind == "gmres":
                linear_solver = lx.GMRES(
                    rtol=float(linear_rtol),
                    atol=float(linear_atol),
                    max_steps=None if linear_max_steps is None else int(linear_max_steps),
                    restart=int(time_cfg.get("imex_linear_restart", 20)),
                    stagnation_iters=int(time_cfg.get("imex_linear_stagnation_iters", 20)),
                )
            elif linear_kind in ("bicgstab", "bicg"):
                linear_solver = lx.BiCGStab(
                    rtol=float(linear_rtol),
                    atol=float(linear_atol),
                    max_steps=None if linear_max_steps is None else int(linear_max_steps),
                )
            elif linear_kind in ("cg", "pcg"):
                linear_solver = lx.CG(
                    rtol=float(linear_rtol),
                    atol=float(linear_atol),
                    max_steps=None if linear_max_steps is None else int(linear_max_steps),
                )
            else:
                raise ValueError(f"Unknown imex_linear_solver: {linear_kind}")

            root_rtol = time_cfg.get("imex_root_rtol", rtol)
            root_atol = time_cfg.get("imex_root_atol", atol)
            root_kind = str(time_cfg.get("imex_root_solver", "verychord")).lower()
            if root_kind in ("newton", "optimistix_newton"):
                root_finder = optx.Newton(
                    rtol=float(root_rtol),
                    atol=float(root_atol),
                    linear_solver=linear_solver,
                    kappa=float(time_cfg.get("imex_root_kappa", 0.01)),
                    cauchy_termination=bool(time_cfg.get("imex_root_cauchy_termination", True)),
                )
            else:
                root_finder = dfx.VeryChord(
                    rtol=float(root_rtol),
                    atol=float(root_atol),
                    linear_solver=linear_solver,
                )
            solver_kwargs["root_finder"] = root_finder
            solver_kwargs["root_find_max_steps"] = int(time_cfg.get("imex_root_max_steps", 10))

        solver = solver_cls(**solver_kwargs)

        if t_end is None:
            t_end = nsteps * dt
        if save_every <= 0:
            save_every = 1
        nsave = int(np.floor((t_end / dt) / save_every)) + 1
        times = jnp.asarray(np.linspace(0.0, float(t_end), nsave))

        dt_save = float(times[1] - times[0]) if nsave > 1 else float(dt)
        diag_fn = _diagnostic_fn(
            system,
            point_idx,
            mode=diag_mode,
            phi_every=diag_phi_every,
            dt_save=dt_save,
            use_phi_guess=diag_phi_use_guess,
            use_phi_guess_only=diag_phi_use_guess_only,
            trace_stats=trace_stats,
            trace_enstrophy=trace_enstrophy,
            save_fields=save_fields,
            snapshot_fields=snapshot_fields,
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
        max_steps = time_cfg.get("max_steps", None)
        max_steps = int(max_steps) if max_steps is not None else None

        def vf(t, y, args):
            _ = args
            return system.rhs(t, y)

        term: dfx.AbstractTerm
        if solver_name in ("kencarp3", "kencarp4", "kencarp5"):

            def vf_explicit(t, y, args):
                _ = args
                return system.rhs_explicit(t, y)

            def vf_implicit(t, y, args):
                _ = args
                return system.rhs_stiff(t, y)

            term = dfx.MultiTerm(dfx.ODETerm(vf_explicit), dfx.ODETerm(vf_implicit))
        else:
            term = dfx.ODETerm(vf)
        adjoint = dfx.DirectAdjoint()
        adjoint_mode = str(time_cfg.get("adjoint", "")).lower()
        if remat or adjoint_mode in ("checkpoint", "recursive_checkpoint"):
            adjoint = dfx.RecursiveCheckpointAdjoint()
        jit = bool(time_cfg.get("jit", False))
        progress_meter = dfx.NoProgressMeter()
        if progress and not jit:
            progress_cls = getattr(dfx, "ProgressMeter", None)
            if progress_cls is not None:
                progress_meter = progress_cls()

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
                max_steps=max_steps,
                adjoint=adjoint,
                progress_meter=progress_meter,
                args=None,
            )

        solve_fn = jax.jit(_solve) if jit else _solve
        sol = solve_fn()
        diag_series = sol.ys["diag"]
        final_state = sol.ys["state"]
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
            trace_stats_out,
            snapshots_out,
        ) = _unpack_diag(diag_series)
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
    if trace_stats_out:
        diagnostics.update(trace_stats_out)
    if snapshots_out:
        diagnostics.update(snapshots_out)
    if track_iters:
        diagnostics["poisson_iters_mean"] = iters_mean
        diagnostics["poisson_iters_max"] = iters_max
        diagnostics["poisson_iters_mean_all"] = jnp.mean(iters_mean)
        diagnostics["poisson_iters_max_all"] = jnp.max(iters_max)

    # If field snapshots are available, expose fluctuation diagnostics
    # relative to the saved equilibrium (first snapshot).
    for field in ("n", "Te", "omega", "phi"):
        key = f"snapshots_{field}"
        if key not in diagnostics:
            continue
        snaps = jnp.asarray(diagnostics[key])
        if snaps.ndim < 2 or int(snaps.shape[0]) < 1:
            continue
        eq = snaps[0]
        delta = snaps - eq[None, ...]
        axes = tuple(range(1, snaps.ndim))
        diagnostics[f"equilibrium_{field}"] = eq
        diagnostics[f"rms_{field}_fluct"] = jnp.sqrt(jnp.mean(delta**2, axis=axes))

    if return_numpy:
        diagnostics = {k: np.asarray(jax.device_get(v)) for k, v in diagnostics.items()}
        times = np.asarray(times)

    return RunResult(times=times, diagnostics=diagnostics, final_state=final_state)
