from __future__ import annotations

from typing import Callable, Literal

import diffrax as dfx
import jax.numpy as jnp

DiffraxSolverName = Literal[
    "tsit5",
    "dopri5",
    "dopri8",
    "euler",
    "implicit_euler",
    "kvaerno3",
    "kvaerno4",
    "kvaerno5",
    "kencarp3",
    "kencarp4",
    "kencarp5",
]


def solver_from_name(name: DiffraxSolverName) -> dfx.AbstractSolver:
    if name == "tsit5":
        return dfx.Tsit5()
    if name == "dopri5":
        return dfx.Dopri5()
    if name == "dopri8":
        return dfx.Dopri8()
    if name == "euler":
        return dfx.Euler()
    if name == "implicit_euler":
        return dfx.ImplicitEuler()
    if name == "kvaerno3":
        return dfx.Kvaerno3()
    if name == "kvaerno4":
        return dfx.Kvaerno4()
    if name == "kvaerno5":
        return dfx.Kvaerno5()
    if name == "kencarp3":
        return dfx.KenCarp3()
    if name == "kencarp4":
        return dfx.KenCarp4()
    if name == "kencarp5":
        return dfx.KenCarp5()
    raise ValueError(f"Unknown Diffrax solver: {name}")


def diffeqsolve(
    rhs: Callable[[float, object], object],
    *,
    y0,
    t0: float,
    t1: float,
    dt0: float,
    save_ts: jnp.ndarray | None = None,
    solver: DiffraxSolverName = "tsit5",
    adaptive: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_steps: int = 200_000,
    progress: bool = False,
) -> dfx.Solution:
    """Integrate an ODE using Diffrax.

    Parameters
    ----------
    rhs:
        Function ``rhs(t, y) -> dy/dt``.
    adaptive:
        If True, use a PID controller to adapt the time step. If False, use a constant step size.
    solver:
        A short solver name. Implicit solvers (e.g. KenCarp/Kvaerno/ImplicitEuler) can help when
        dissipation or closures make the dynamics stiff.
    save_ts:
        If provided, return the solution sampled at these times (useful for movies/plots).
    """

    term = dfx.ODETerm(lambda t, y, args: rhs(t, y))
    solver_obj = solver_from_name(solver)
    stepsize_controller = (
        dfx.PIDController(rtol=rtol, atol=atol) if adaptive else dfx.ConstantStepSize()
    )
    saveat = dfx.SaveAt(ts=save_ts) if save_ts is not None else dfx.SaveAt(t1=True)
    return dfx.diffeqsolve(
        term,
        solver_obj,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(max_steps),
        progress_meter=dfx.TextProgressMeter() if progress else dfx.NoProgressMeter(),
    )
