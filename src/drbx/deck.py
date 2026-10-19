"""TOML decks for every DRBX model: ``drbx run deck.toml`` equals the Python API.

A deck names its model with ``[model] backend = "<name>"`` and configures it
with one TOML table per Python configuration object.  The tables are the
dataclasses (or function signatures) the Python API already uses, so the
defaults live in exactly one place and an unknown key is an error that lists
the valid ones.  ``BACKENDS`` is the registry; ``docs/decks.md`` documents every
table and a test fails if the page and the schema drift apart.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .config.boutinp import tomllib

# ----------------------------------------------------------------------------
# Schema machinery
# ----------------------------------------------------------------------------


def from_signature(name: str, function: Callable, exclude: tuple[str, ...] = ()) -> type:
    """Dataclass whose fields are ``function``'s parameters (annotations and defaults)."""

    hints = typing.get_type_hints(function)
    items = []
    for parameter in inspect.signature(function).parameters.values():
        if parameter.name in exclude:
            continue
        spec = (
            dataclasses.field()
            if parameter.default is inspect.Parameter.empty
            else dataclasses.field(default=parameter.default)
        )
        items.append((parameter.name, hints[parameter.name], spec))
    items.sort(key=lambda item: item[2].default is not dataclasses.MISSING)  # required first
    return dataclasses.make_dataclass(name, items, frozen=True)


def _coerce(hint: Any, value: object, label: str, base_dir: Path) -> object:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        options = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        return _coerce(options[0], value, label, base_dir)
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a table")
        return build_table(hint, value, label, base_dir)
    if origin is tuple:
        args = typing.get_args(hint)
        variable = len(args) == 2 and args[1] is Ellipsis
        if not isinstance(value, list) or (not variable and len(value) != len(args)):
            raise ValueError(
                f"{label} must be a list of {'any number of' if variable else len(args)} values"
            )
        kinds = [args[0]] * len(value) if variable else args
        return tuple(_coerce(kind, item, label, base_dir) for kind, item in zip(kinds, value))
    if hint is Path:
        path = Path(str(_coerce(str, value, label, base_dir))).expanduser()
        return path if path.is_absolute() else base_dir / path
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, hint) and (hint is bool or not isinstance(value, bool)):
        return value
    raise ValueError(f"{label} must be {hint.__name__}, got {value!r}")


def build_table(cls: type, table: Mapping[str, object], where: str, base_dir: Path) -> Any:
    """Instantiate dataclass ``cls`` from a TOML table; unknown or missing keys are errors."""

    hints = typing.get_type_hints(cls)
    fields = {field.name: field for field in dataclasses.fields(cls)}
    unknown = sorted(set(table) - set(fields))
    if unknown:
        raise ValueError(f"unknown {where} key(s): {unknown}; valid keys: {sorted(fields)}")
    missing = [
        name
        for name, field in fields.items()
        if field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
        and name not in table
    ]
    if missing:
        raise ValueError(f"{where} {missing[0]} is required")
    return cls(**{k: _coerce(hints[k], v, f"{where} {k}", base_dir) for k, v in table.items()})


def schema_rows(cls: type) -> list[tuple[str, str, str]]:
    """``(key, type, default)`` text per scalar key of ``cls`` (nested tables are separate)."""

    hints = typing.get_type_hints(cls)
    rows = []
    for field in dataclasses.fields(cls):
        hint = hints[field.name]
        if dataclasses.is_dataclass(hint):
            continue
        kind = hint.__name__ if isinstance(hint, type) else re.sub(r"\b(typing|pathlib)\.", "", str(hint))
        required = field.default is dataclasses.MISSING
        rows.append((field.name, kind, "required" if required else default_text(field.default)))
    return rows


def default_text(value: object) -> str:
    """Documentation form of a default; repository-relative paths hide the checkout location."""

    if isinstance(value, Path):
        return f"<repo>/{value.name}"
    if isinstance(value, tuple):
        return "[" + ", ".join(default_text(item) for item in value) + "]"
    return repr(value)


def table_specs(backend: str) -> dict[str, type]:
    """Every table of ``backend`` as ``{"table" or "table.sub": dataclass}``, nested ones included."""

    flat: dict[str, type] = {}

    def add(path: str, cls: type) -> None:
        flat[path] = cls
        for name, hint in typing.get_type_hints(cls).items():
            if dataclasses.is_dataclass(hint):
                add(f"{path}.{name}", hint)

    for table, cls in _expected_tables(BACKENDS[backend]).items():
        add(table, cls)
    return flat


# ----------------------------------------------------------------------------
# Generic tables and per-model tables (all defaults come from the Python API)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeConfig:
    """``[runtime]``: numerical precision of the run."""

    precision: str = "float64"


@dataclass(frozen=True)
class OutputConfig:
    """``[output]``: where ``<case_name>_summary.json`` and ``<case_name>_arrays.npz`` go."""

    directory: Path | None = None
    case_name: str | None = None


@dataclass(frozen=True)
class HwGrid:
    """``[grid]`` of the Hasegawa-Wakatani model: ``n x n`` periodic box of side ``length``."""

    n: int = 96
    length: float = 2.0 * np.pi * 8.0


@dataclass(frozen=True)
class HwRun:
    """``[run]`` of the Hasegawa-Wakatani model."""

    dt: float = 5.0e-3
    steps_per_block: int = 800
    blocks: int = 40
    seed: int = 0
    seed_amplitude: float = 5.0e-2
    save_frames: bool = False
    restart_from: Path | None = None


@dataclass(frozen=True)
class Autodiff:
    """``[autodiff]``: gradient of the saturated mean flux w.r.t. one Hasegawa-Wakatani parameter."""

    parameter: str
    fd_step: float = 1.0e-3


@dataclass(frozen=True)
class SolRun:
    """``[run]`` of the open-field-line SOL flux tube."""

    cfl: float = 0.4
    steps: int = 60000
    report_every: int = 10000
    electron_temperature: float = 0.5
    ion_temperature: float = 0.5
    restart_from: Path | None = None


@dataclass(frozen=True)
class RecyclingRun:
    """``[run]`` of the recycling SOL; ``dt`` defaults to ``0.1 / nz`` (CFL 0.3 at 3 c_s)."""

    dt: float | None = None
    steps: int = 60000
    report_every: int = 15000
    restart_from: Path | None = None


@dataclass(frozen=True)
class RecyclingProfile:
    """``[profile]`` of the recycling SOL: prescribed temperature and neutral seed."""

    nz: int = 200
    upstream_ev: float = 30.0
    target_ev: float = 1.5
    neutral_seed: float = 0.05


@dataclass(frozen=True)
class Scan:
    """``[scan]``: extra steady states, one per upstream density (empty = no scan)."""

    upstream_density: tuple[float, ...] = ()


@dataclass(frozen=True)
class FciDiffRun:
    """``[run]`` of the differentiable FCI case: autodiff-vs-finite-difference check."""

    amp0: float = 0.1
    n_steps: int = 24
    dt: float = 1.0e-3
    fd_step: float = 1.0e-5


def _signature_tables() -> dict[str, type]:
    from .geometry import (
        build_island_divertor_geometry,
        build_open_slab_geometry,
        build_rotating_ellipse_geometry,
        build_shifted_torus_geometry,
    )
    from .native.fci_differentiable_case import build_context
    from .native.fci_sheath_recycling import compute_fci_sheath_recycling
    from .native.neutrals.detachment_sol_model import detachment_sol_run
    from .native.stellarator_turbulence import run_stellarator_turbulence

    ignore = ("construct_fci_maps", "B_contravariant")
    return {
        "open_slab": from_signature("OpenSlab", build_open_slab_geometry),
        "rotating_ellipse": from_signature("RotatingEllipse", build_rotating_ellipse_geometry, ignore),
        "shifted_torus": from_signature("ShiftedTorus", build_shifted_torus_geometry, ignore),
        "island_divertor": from_signature("IslandDivertor", build_island_divertor_geometry, ignore),
        "sheath": from_signature(
            "Sheath", compute_fci_sheath_recycling,
            ("density", "electron_temperature", "ion_temperature", "maps"),
        ),
        "detachment_run": from_signature("DetachmentRun", detachment_sol_run, ("params", "state", "source_params", "verbose")),
        "turbulence_run": from_signature("TurbulenceRun", run_stellarator_turbulence, ("geometry",)),
        "fci_differentiable": from_signature("FciDifferentiable", build_context),
    }


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Backend:
    """A model a deck can run: its TOML tables and how to run them."""

    tables: Callable[[], dict[str, type]]  # table name -> dataclass (lazy: imports JAX modules)
    run: Callable[[dict[str, Any], Callable[[str], None]], tuple[dict, dict] | None]
    required: tuple[str, ...] = ()
    one_of: tuple[str, ...] = ()  # exactly one of these tables must be present
    generic: bool = True  # accepts [runtime] and [output]


def _array(value: object) -> np.ndarray:
    return np.asarray(value)


def _load_state(path: Path | None, *names: str) -> tuple[dict[str, np.ndarray], float] | None:
    if path is None:
        return None
    with np.load(path) as data:
        missing = [name for name in names if f"state_{name}" not in data]
        if missing:
            raise ValueError(f"restart file {path} lacks state_{missing[0]}; it is from another model")
        return {name: data[f"state_{name}"] for name in names}, float(data["state_time"])


def _hw_noise(grid, seed: int, amplitude: float):
    # FFT of a real field keeps the spectrum Hermitian, so zeta and n stay real.
    import jax.numpy as jnp

    noise = np.fft.fft2(np.random.default_rng(seed).standard_normal((grid.n, grid.n)))
    noise[0, 0] = 0.0
    noise *= np.exp(-np.asarray(grid.k2)) * np.asarray(grid.dealias)
    noise *= amplitude / np.sqrt(np.mean(np.real(np.fft.ifft2(noise)) ** 2))
    return jnp.array(noise)


def _run_hw(cfg: dict, say: Callable[[str], None]):
    import jax
    import jax.numpy as jnp

    from .native.hasegawa_wakatani import (
        hw_grid,
        hw_run,
        hw_run_flux_history,
        particle_flux,
        potential_from_vorticity,
    )

    params, grid_cfg, run = cfg["hasegawa_wakatani"], cfg["grid"], cfg["run"]
    grid = hw_grid(grid_cfg.n, grid_cfg.length)
    restart = _load_state(run.restart_from, "zeta_hat", "n_hat")
    if restart is None:
        zeta = density = _hw_noise(grid, run.seed, run.seed_amplitude)
        t0 = 0.0
    else:
        (state, t0) = restart
        zeta, density = jnp.asarray(state["zeta_hat"]), jnp.asarray(state["n_hat"])
    times, energies, fluxes, frames = [], [], [], []
    for block in range(run.blocks):
        zeta, density = hw_run(zeta, density, grid, params, dt=run.dt, steps=run.steps_per_block)
        phi = potential_from_vorticity(zeta, grid)
        energies.append(float(jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2)) / grid.n**4)
        fluxes.append(float(particle_flux(zeta, density, grid)))
        times.append(t0 + run.dt * run.steps_per_block * (block + 1))
        if run.save_frames:
            frames.append(np.real(np.asarray(jnp.fft.ifft2(zeta))))
        say(f"block {block + 1}/{run.blocks} t={times[-1]:.4g} energy={energies[-1]:.4e} flux={fluxes[-1]:+.4e}")
    summary = {"final_time": times[-1], "final_energy": energies[-1], "final_particle_flux": fluxes[-1],
               "mean_particle_flux_second_half": float(np.mean(fluxes[len(fluxes) // 2:]))}
    arrays = {
        "time": _array(times), "energy": _array(energies), "particle_flux": _array(fluxes),
        "vorticity": np.real(np.asarray(jnp.fft.ifft2(zeta))), "density": np.real(np.asarray(jnp.fft.ifft2(density))),
        "state_zeta_hat": np.asarray(zeta), "state_n_hat": np.asarray(density), "state_time": _array(times[-1]),
    }
    if run.save_frames:
        arrays["vorticity_frames"] = np.stack(frames)
    if "autodiff" in cfg:
        spec = cfg["autodiff"]
        if spec.parameter not in {f.name for f in dataclasses.fields(params)}:
            raise ValueError(f"[autodiff] parameter must be one of {[f.name for f in dataclasses.fields(params)]}")
        zeta0 = density0 = _hw_noise(grid, run.seed, run.seed_amplitude)
        steps = run.steps_per_block * run.blocks

        def objective(value):
            _, _, flux = hw_run_flux_history(
                zeta0, density0, grid, replace(params, **{spec.parameter: value}),
                dt=run.dt, steps=steps, sample_every=run.steps_per_block,
            )
            return jnp.mean(flux[len(flux) // 2:])

        x0 = float(getattr(params, spec.parameter))
        h = spec.fd_step * max(abs(x0), 1.0)
        grad = float(jax.jit(jax.grad(objective))(x0))
        fd = (float(objective(x0 + h)) - float(objective(x0 - h))) / (2.0 * h)
        summary["autodiff"] = {"parameter": spec.parameter, "value": x0, "gradient": grad,
                               "finite_difference": fd, "relative_error": abs(grad - fd) / max(abs(fd), 1e-30)}
        say(f"d<flux>/d{spec.parameter} = {grad:+.4e} (finite difference {fd:+.4e})")
    return summary, arrays


def _run_sol_flux_tube(cfg: dict, say: Callable[[str], None]):
    import jax.numpy as jnp

    from .geometry import build_open_slab_geometry
    from .native.fci_sheath_recycling import compute_fci_sheath_recycling
    from .native.sol_flux_tube import sol_flux_tube_run, sol_flux_tube_source

    params, run, sheath_cfg = cfg["sol_flux_tube"], cfg["run"], cfg["sheath"]
    geometry = build_open_slab_geometry(**vars_of(cfg["open_slab"]))
    source = sol_flux_tube_source(geometry, params)
    restart = _load_state(run.restart_from, "density", "momentum")
    if restart is None:
        density, momentum, t0 = jnp.ones(geometry.shape), jnp.zeros(geometry.shape), 0.0
    else:
        (state, t0) = restart
        density, momentum = jnp.asarray(state["density"]), jnp.asarray(state["momentum"])
    dz = float(geometry.spacing.dz[0, 0, 0])
    dt = run.cfl * dz / (params.sound_speed + 1.0)
    steps_done, residuals, report_steps = 0, [], []
    while steps_done < run.steps:
        chunk = min(run.report_every, run.steps - steps_done)
        previous = density
        density, momentum = sol_flux_tube_run(density, momentum, geometry, params, source, dt=dt, steps=chunk)
        steps_done += chunk
        residuals.append(float(jnp.max(jnp.abs(density - previous)) / (chunk * dt)))
        report_steps.append(steps_done)
        say(f"step {steps_done}/{run.steps} max|dn/dt|={residuals[-1]:.2e}")
    n_line = np.asarray(density)[0, 0, :]
    mach = (np.asarray(momentum)[0, 0, :] / n_line) / params.sound_speed
    sheath = compute_fci_sheath_recycling(
        density, jnp.full(geometry.shape, run.electron_temperature),
        jnp.full(geometry.shape, run.ion_temperature), geometry.maps, **vars_of(sheath_cfg),
    )
    summary = {
        "mach_at_targets": [float(mach[0]), float(mach[-1])],
        "n_target_over_upstream": float(n_line[-1] / n_line[len(n_line) // 2]),
        "total_target_ion_loss": float(sheath.total_ion_particle_loss),
        "total_recycled_source": float(sheath.total_recycled_particle_source),
        "total_target_heat_load": float(sheath.total_target_heat_load),
        "particle_recycling_residual": float(sheath.particle_recycling_residual),
        "current_balance_residual": float(sheath.current_balance_residual),
        "final_time": t0 + steps_done * dt,
    }
    arrays = {
        "z": np.asarray(geometry.grid.z.centers), "density": n_line, "momentum": np.asarray(momentum)[0, 0, :],
        "mach": mach, "source": np.asarray(source)[0, 0, :], "report_step": _array(report_steps),
        "residual": _array(residuals), "ion_particle_loss": np.asarray(sheath.ion_particle_loss)[0, 0, :],
        "target_heat_load": np.asarray(sheath.target_heat_load)[0, 0, :],
        "recycled_particle_source": np.asarray(sheath.recycled_particle_source)[0, 0, :],
        "state_density": np.asarray(density), "state_momentum": np.asarray(momentum),
        "state_time": _array(summary["final_time"]),
    }
    return summary, arrays


def vars_of(config: Any) -> dict[str, Any]:
    """Shallow field dict of a dataclass (nested dataclasses stay objects)."""

    return {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}


def _run_sol_recycling(cfg: dict, say: Callable[[str], None]):
    import jax.numpy as jnp

    from .native.neutrals import (
        SolRecyclingState,
        linear_target_temperature_profile,
        sol_recycling_run,
        target_ion_flux,
    )

    params, profile, run, scan = cfg["sol_recycling"], cfg["profile"], cfg["run"], cfg["scan"]
    nz = profile.nz
    dt = run.dt if run.dt is not None else 0.3 * (1.0 / nz) / 3.0
    temperature = linear_target_temperature_profile(
        nz, upstream_ev=profile.upstream_ev, target_ev=profile.target_ev, normalization=params.normalization
    )
    sound_speed = np.sqrt(2.0 * np.asarray(temperature) / params.ion_mass)

    def relax(p, state, label):
        done = 0
        while done < run.steps:
            chunk = min(run.report_every, run.steps - done)
            state = sol_recycling_run(state, temperature, p, dt=dt, steps=chunk)
            done += chunk
        mach = float(np.asarray(state.ion_momentum)[-1] / (p.ion_mass * max(float(state.ion_density[-1]), p.density_floor)) / sound_speed[-1])
        flux = float(target_ion_flux(state, temperature, p))
        say(f"{label}: target Mach={mach:.4f} target flux={flux:.4f}")
        return state, mach, flux

    restart = _load_state(run.restart_from, "ion_density", "ion_momentum", "neutral_density")
    if restart is None:
        state = SolRecyclingState(
            ion_density=jnp.full(nz, params.upstream_density), ion_momentum=jnp.zeros(nz),
            neutral_density=jnp.full(nz, profile.neutral_seed),
        )
        t0 = 0.0
    else:
        (loaded, t0) = restart
        state = SolRecyclingState(*(jnp.asarray(loaded[k]) for k in ("ion_density", "ion_momentum", "neutral_density")))
    state, mach, flux = relax(params, state, f"n_up={params.upstream_density:g}")
    density, neutral = np.asarray(state.ion_density), np.asarray(state.neutral_density)
    scan_mach, scan_flux = [], []
    for upstream in scan.upstream_density:
        p = replace(params, upstream_density=upstream)
        fresh = SolRecyclingState(jnp.full(nz, upstream), jnp.zeros(nz), jnp.full(nz, profile.neutral_seed))
        _, m, f = relax(p, fresh, f"n_up={upstream:g}")
        scan_mach.append(m)
        scan_flux.append(f)
    final_time = t0 + run.steps * dt
    summary = {"target_mach": mach, "target_ion_flux": flux, "dt": dt, "final_time": final_time,
               "neutral_cushion_ratio": float(neutral[-1] / max(neutral[nz // 2], 1e-9))}
    arrays = {
        "z_over_L": (np.arange(nz) + 0.5) / nz, "temperature": np.asarray(temperature), "ion_density": density,
        "ion_momentum": np.asarray(state.ion_momentum), "neutral_density": neutral,
        "mach": np.asarray(state.ion_momentum) / (params.ion_mass * np.maximum(density, params.density_floor)) / sound_speed,
        "scan_upstream_density": _array(scan.upstream_density, ), "scan_target_mach": _array(scan_mach),
        "scan_target_ion_flux": _array(scan_flux), "state_ion_density": density,
        "state_ion_momentum": np.asarray(state.ion_momentum), "state_neutral_density": neutral,
        "state_time": _array(final_time),
    }
    return summary, arrays


def _run_detachment(cfg: dict, say: Callable[[str], None]):
    from .native.neutrals import (
        detachment_diagnostics,
        detachment_ledger,
        detachment_sol_run,
    )

    params, run_kwargs, scan = cfg["detachment_sd1d"], vars_of(cfg["run"]), cfg["scan"]
    rows, previous, result = [], None, None
    for upstream in scan.upstream_density or (params.upstream_density,):
        p = replace(params, upstream_density=upstream)
        kwargs = dict(run_kwargs)
        if previous is not None:
            kwargs["source_scale"] = previous.source_scale
        result = detachment_sol_run(p, previous.state if previous else None, **kwargs)
        previous = result
        diag, ledger = detachment_diagnostics(result.state, p, result.source_scale), detachment_ledger(result, p)
        rows.append({
            "upstream_density": float(upstream), "target_temperature_ev": float(diag.target_temperature_ev),
            "target_ion_flux_area": float(diag.target_ion_flux_area), "target_density": float(diag.target_density),
            "target_mach": float(diag.target_mach), "upstream_temperature_ev": float(diag.upstream_temperature_ev),
            "residual": float(result.residual), "iterations": int(result.iterations),
            "converged": bool(result.converged), "particle_imbalance": float(ledger["particle_imbalance"]),
            "power_imbalance": float(ledger["power_imbalance"]),
        })
        say(f"n_up={upstream:.3e}: T_t={rows[-1]['target_temperature_ev']:.2f} eV, {result.iterations} its, residual {result.residual:.1e}")
    arrays = {key: _array([row[key] for row in rows]) for key in rows[0]}
    arrays.update({f"profile_{name}": np.asarray(getattr(result.state, name)) for name in result.state._fields})
    return {"points": rows}, arrays


def _run_stellarator_turbulence(cfg: dict, say: Callable[[str], None]):
    from . import geometry as geometry_module
    from .native.stellarator_turbulence import run_stellarator_turbulence

    builders = {"rotating_ellipse": geometry_module.build_rotating_ellipse_geometry,
                "shifted_torus": geometry_module.build_shifted_torus_geometry,
                "island_divertor": geometry_module.build_island_divertor_geometry}
    kind = next(name for name in builders if name in cfg)
    geometry = builders[kind](**vars_of(cfg[kind]))
    run = run_stellarator_turbulence(geometry, **vars_of(cfg["run"]))
    say(f"{kind} {tuple(geometry.shape)}: {len(run.times)} frames, particle content {run.particle_content[0]:.4f} -> {run.particle_content[-1]:.4f}")
    summary = {"geometry": kind, "shape": list(geometry.shape), "n_frames": len(run.times),
               "initial_particle_content": float(run.particle_content[0]),
               "final_particle_content": float(run.particle_content[-1]),
               "total_target_flux": float(np.sum(run.target_flux)),
               "finite": bool(np.all(np.isfinite(run.density_frames)))}
    arrays = {"density_frames": run.density_frames, "omega_frames": run.omega_frames, "times": run.times,
              "particle_content": run.particle_content, "target_flux": run.target_flux}
    return summary, arrays


def _run_fci_differentiable(cfg: dict, say: Callable[[str], None]):
    from .native.fci_differentiable_case import build_context, differentiability_report

    context = build_context(**vars_of(cfg["fci_differentiable"]))
    report = differentiability_report(context, **vars_of(cfg["run"]))
    state = report.pop("final_state")
    say(f"grad={report['grad']:+.6e} fd={report['fd']:+.6e} rel_error={report['rel_error']:.2e}")
    return report, {"density": np.asarray(state.density), "v_parallel": np.asarray(state.v_parallel)}


def _run_hsx(cfg: dict, say: Callable[[str], None]):
    from .fci_braginskii.run import run

    run(cfg["fci_braginskii"])


def _hsx_tables() -> dict[str, type]:
    from .fci_braginskii.run import HsxBlobConfig

    return {"fci_braginskii": HsxBlobConfig}


def _lazy(module: str, name: str) -> type:
    import importlib

    return getattr(importlib.import_module(module, __package__), name)


def _hw_tables() -> dict[str, type]:
    return {"hasegawa_wakatani": _lazy(".native.hasegawa_wakatani", "HasegawaWakataniParameters"),
            "grid": HwGrid, "run": HwRun, "autodiff": Autodiff}


def _sol_tables() -> dict[str, type]:
    library = _signature_tables()
    return {"sol_flux_tube": _lazy(".native.sol_flux_tube", "SolFluxTubeParameters"),
            "open_slab": library["open_slab"], "sheath": library["sheath"], "run": SolRun}


def _recycling_tables() -> dict[str, type]:
    return {"sol_recycling": _lazy(".native.neutrals", "SolRecyclingParameters"),
            "profile": RecyclingProfile, "run": RecyclingRun, "scan": Scan}


def _detachment_tables() -> dict[str, type]:
    library = _signature_tables()
    return {"detachment_sd1d": _lazy(".native.neutrals", "DetachmentSolParameters"),
            "run": library["detachment_run"], "scan": Scan}


def _turbulence_tables() -> dict[str, type]:
    library = _signature_tables()
    return {"rotating_ellipse": library["rotating_ellipse"], "shifted_torus": library["shifted_torus"],
            "island_divertor": library["island_divertor"], "run": library["turbulence_run"]}


def _fci_diff_tables() -> dict[str, type]:
    return {"fci_differentiable": _signature_tables()["fci_differentiable"], "run": FciDiffRun}


BACKENDS: dict[str, Backend] = {
    "hasegawa_wakatani": Backend(_hw_tables, _run_hw),
    "sol_flux_tube": Backend(_sol_tables, _run_sol_flux_tube, required=("open_slab",)),
    "sol_recycling": Backend(_recycling_tables, _run_sol_recycling),
    "detachment_sd1d": Backend(_detachment_tables, _run_detachment),
    "stellarator_turbulence": Backend(
        _turbulence_tables, _run_stellarator_turbulence, required=("run",),
        one_of=("rotating_ellipse", "shifted_torus", "island_divertor"),
    ),
    "fci_differentiable": Backend(_fci_diff_tables, _run_fci_differentiable),
    "fci_braginskii": Backend(_hsx_tables, _run_hsx, required=("fci_braginskii",), generic=False),
}

def _expected_tables(backend: Backend) -> dict[str, type]:
    tables = dict(backend.tables())
    if backend.generic:
        tables.update(runtime=RuntimeConfig, output=OutputConfig)
    return tables


def load_deck(path: str | Path) -> tuple[str, dict[str, Any]]:
    """Parse ``path`` into ``(backend name, {table name: config object})``."""

    source = Path(path)
    data = tomllib.loads(source.read_text(encoding="utf-8"))
    model = data.get("model", {})
    name = model.get("backend") if isinstance(model, dict) else None
    if name not in BACKENDS:
        raise ValueError(f'[model] backend must be one of {sorted(BACKENDS)}, got {name!r}')
    if set(model) != {"backend"}:
        raise ValueError(f'[model] must contain only backend = "{name}"')
    backend = BACKENDS[name]
    tables = _expected_tables(backend)
    unknown = sorted(set(data) - set(tables) - {"model"})
    if unknown:
        raise ValueError(f"unknown table(s) in {name} deck: {unknown}; valid tables: {sorted(tables)}")
    base_dir = source.resolve().parent
    config: dict[str, Any] = {}
    for table, cls in tables.items():
        body = data.get(table, {})
        if not isinstance(body, dict):
            raise ValueError(f"[{table}] must be a table")
        if table in data or table in backend.required or (
            table not in backend.one_of and not _has_required(cls)
        ):
            config[table] = build_table(cls, body, f"[{table}]", base_dir)
    if backend.one_of and sum(table in config for table in backend.one_of) != 1:
        raise ValueError(f"exactly one of {['[' + t + ']' for t in backend.one_of]} must be given")
    if name == "fci_braginskii":
        _check_hsx(config["fci_braginskii"])
    return name, config


def _has_required(cls: type) -> bool:
    return any(f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING for f in dataclasses.fields(cls))


def _check_hsx(config: Any) -> None:
    if config.geometry is None:
        raise ValueError("[fci_braginskii] geometry is required")
    if config.blob_initialization not in ("field-aligned", "logical"):
        raise ValueError("[fci_braginskii] blob_initialization must be 'field-aligned' or 'logical'")
    if config.advance_execution not in ("compiled", "staged-compiled", "eager"):
        raise ValueError("[fci_braginskii] advance_execution must be 'compiled', 'staged-compiled' or 'eager'")
    if config.gmres_preconditioner not in ("none", "line-u"):
        raise ValueError("[fci_braginskii] gmres_preconditioner must be 'none' or 'line-u'")


def describe(name: str, config: Mapping[str, Any]) -> str:
    """The resolved configuration, one ``table.key = value`` line per key."""

    lines = [f"backend: {name}"]
    for table, value in config.items():
        for key, item in vars_of(value).items():
            lines.append(f"{table}.{key} = {item!r}" if table != "fci_braginskii" else f"{key} = {item!r}")
    return "\n".join(lines)


def run_deck(
    path: str | Path,
    *,
    output_dir: Path | None = None,
    case_name: str | None = None,
    quiet: bool = False,
) -> dict | None:
    """Run a deck; write ``<case>_summary.json`` and ``<case>_arrays.npz``; return the summary."""

    name, config = load_deck(path)
    runtime = config.get("runtime", RuntimeConfig())
    if name != "fci_braginskii":
        from .runtime import configure_jax_runtime

        configure_jax_runtime(precision=runtime.precision)
    output = config.get("output", OutputConfig())
    case = case_name or output.case_name or Path(path).stem
    say = (lambda message: None) if quiet else (lambda message: print(f"[{case}] {message}", flush=True))
    result = BACKENDS[name].run(config, say)
    if result is None:
        return None
    summary, arrays = result
    directory = Path(output_dir or output.directory or Path("output") / case)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_name": case,
        "backend": name,
        "config": json.loads(json.dumps({t: vars_of(c) for t, c in config.items() }, default=str)),
        "results": summary,
    }
    (directory / f"{case}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(directory / f"{case}_arrays.npz", **arrays)
    say(f"wrote {directory / (case + '_summary.json')} and {directory / (case + '_arrays.npz')}")
    return payload
