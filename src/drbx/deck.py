"""TOML decks for every DRBX model: ``drbx run deck.toml`` equals the Python API.

``[model] backend = "<name>"`` selects a model; each other table is one Python
configuration dataclass or function signature, so defaults live in one place and
unknown keys are errors.  ``docs/decks.md`` documents the tables (a test checks it).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import types
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .config.boutinp import tomllib

HSX_INPUTS = Path(__file__).resolve().parents[2] / "examples" / "inputs" / "hsx"


def from_signature(name: str, *functions: Callable, exclude: tuple[str, ...] = ()) -> type:
    """Dataclass whose fields are the parameters of ``functions`` (annotations, defaults)."""

    items = []
    for function in functions:
        hints = typing.get_type_hints(function)
        for p in inspect.signature(function).parameters.values():
            if p.name not in exclude:
                spec = dataclasses.field() if p.default is p.empty else dataclasses.field(default=p.default)
                items.append((p.name, hints[p.name], spec))
    items.sort(key=lambda item: item[2].default is not dataclasses.MISSING)  # required first
    return dataclasses.make_dataclass(name, items, frozen=True)


def _coerce(hint: Any, value: object, label: str, base_dir: Path) -> object:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        options = [a for a in typing.get_args(hint) if a is not type(None)]
        return _coerce(Path if Path in options else options[0], value, label, base_dir)
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a table")
        return build_table(hint, value, label, base_dir)
    if origin in (tuple, Sequence):
        args = typing.get_args(hint)
        variable = origin is Sequence or (len(args) == 2 and args[1] is Ellipsis)
        if not isinstance(value, list) or (not variable and len(value) != len(args)):
            raise ValueError(f"{label} must be a list of {'any number of' if variable else len(args)} values")
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
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(table) - set(fields))
    if unknown:
        raise ValueError(f"unknown {where} key(s): {unknown}; valid keys: {sorted(fields)}")
    for name in _required(cls):
        if name not in table:
            raise ValueError(f"{where} {name} is required")
    return cls(**{k: _coerce(hints[k], v, f"{where} {k}", base_dir) for k, v in table.items()})


def _required(cls: type) -> list[str]:
    return [f.name for f in dataclasses.fields(cls) if f.default is dataclasses.MISSING]


def _type_text(hint: Any) -> str:
    """Python-version-independent spelling of a type hint (``tuple[int, ...]``, ``Path | None``)."""

    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        names = [_type_text(a) for a in args if a is not type(None)]
        return " | ".join(names + ["None"] * (type(None) in args))
    if origin:
        return f"{origin.__name__}[{', '.join('...' if a is Ellipsis else _type_text(a) for a in args)}]"
    return hint.__name__


def default_text(value: object) -> str:
    """Documentation form of a default (paths inside the checkout hide its location)."""

    if isinstance(value, Path):
        return f"<repo>/{value.name}"
    if isinstance(value, tuple):
        return "[" + ", ".join(default_text(item) for item in value) + "]"
    return repr(value)


def schema_rows(cls: type) -> list[tuple[str, str, str]]:
    """``(key, type, default)`` text per scalar key of ``cls`` (nested tables are separate)."""

    rows = []
    for f in dataclasses.fields(cls):
        hint = typing.get_type_hints(cls)[f.name]
        if not dataclasses.is_dataclass(hint):
            rows.append((f.name, _type_text(hint), "required" if f.default is dataclasses.MISSING else default_text(f.default)))
    return rows


def table_specs(backend: str) -> dict[str, type]:
    """Every table of ``backend`` as ``{"table" or "table.sub": dataclass}``."""

    flat: dict[str, type] = {}

    def add(path: str, cls: type) -> None:
        flat[path] = cls
        for name, hint in typing.get_type_hints(cls).items():
            if dataclasses.is_dataclass(hint):
                add(f"{path}.{name}", hint)

    for table, cls in _expected_tables(BACKENDS[backend]).items():
        add(table, cls)
    return flat


def vars_of(config: Any) -> dict[str, Any]:
    return {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}


# ---- tables that are not a single Python signature -------------------------


@dataclass(frozen=True)
class RuntimeConfig:
    precision: str = "float64"


@dataclass(frozen=True)
class OutputConfig:
    directory: Path | None = None
    case_name: str | None = None


@dataclass(frozen=True)
class HwGrid:
    n: int = 96
    length: float = 2.0 * np.pi * 8.0


@dataclass(frozen=True)
class HwRun:
    dt: float = 5.0e-3
    steps_per_block: int = 800
    blocks: int = 40
    seed: int = 0
    seed_amplitude: float = 5.0e-2
    save_frames: bool = False


@dataclass(frozen=True)
class Autodiff:
    parameter: str
    fd_step: float = 1.0e-3


@dataclass(frozen=True)
class SolRun:
    cfl: float = 0.4
    steps: int = 60000
    report_every: int = 10000
    electron_temperature: float = 0.5
    ion_temperature: float = 0.5


@dataclass(frozen=True)
class RecyclingRun:
    dt: float | None = None
    steps: int = 60000
    report_every: int = 15000


@dataclass(frozen=True)
class RecyclingProfile:
    nz: int = 200
    upstream_ev: float = 30.0
    target_ev: float = 1.5
    neutral_seed: float = 0.05


@dataclass(frozen=True)
class Scan:
    upstream_density: tuple[float, ...] = ()


@dataclass(frozen=True)
class FciDiffRun:
    amp0: float = 0.1
    n_steps: int = 24
    dt: float = 1.0e-3
    fd_step: float = 1.0e-5


def _hsx_geometry_table() -> type:
    """``HsxSimulationGeometryConfig`` with the shipped coils and vessel as defaults, plus a cache dir."""

    from .fci_braginskii.geometry_build.hsx_simulation_geometry import HsxSimulationGeometryConfig as C

    default = {"makegrid_path": None, "vessel_path": HSX_INPUTS / "vessel_hsx_flare.txt",
               "coils_path": HSX_INPUTS / "coils.hsx"}
    hints = typing.get_type_hints(C)
    items = [(f.name, Path | None if f.name in default else hints[f.name],
              dataclasses.field(default=default[f.name]) if f.name in default else
              (dataclasses.field() if f.default is dataclasses.MISSING else dataclasses.field(default=f.default)))
             for f in dataclasses.fields(C) if f.name != "output"]
    items.sort(key=lambda item: item[2].default is not dataclasses.MISSING)
    items.append(("cache_dir", Path, dataclasses.field(default=Path("output") / "hsx_geometry")))
    return dataclasses.make_dataclass("HsxGeometry", items, frozen=True)


def _geometry_tables() -> dict[str, type]:
    from . import geometry as g

    ignore = ("construct_fci_maps", "B_contravariant")
    return {
        "open_slab": from_signature("OpenSlab", g.build_open_slab_geometry),
        "rotating_ellipse": from_signature("RotatingEllipse", g.build_rotating_ellipse_geometry, exclude=ignore),
        "shifted_torus": from_signature("ShiftedTorus", g.build_shifted_torus_geometry, exclude=ignore),
        "island_divertor": from_signature("IslandDivertor", g.build_island_divertor_geometry, exclude=ignore),
        "essos": from_signature("Essos", g.build_essos_imported_fci_geometry, g.essos_imported_geometry_to_fci,
                                exclude=("imported", "essos_root")),
    }


# ---- runners: (config tables, say) -> (summary, arrays) --------------------


def _hw(cfg: dict, say: Callable[[str], None]):
    import jax
    import jax.numpy as jnp

    from .native.hasegawa_wakatani import (
        hw_grid, hw_run, hw_run_flux_history, particle_flux, potential_from_vorticity,
    )

    params, run = cfg["hasegawa_wakatani"], cfg["run"]
    grid = hw_grid(cfg["grid"].n, cfg["grid"].length)
    # FFT of a real field keeps the spectrum Hermitian, so zeta and n stay real.
    noise = np.fft.fft2(np.random.default_rng(run.seed).standard_normal((grid.n, grid.n)))
    noise[0, 0] = 0.0
    noise *= np.exp(-np.asarray(grid.k2)) * np.asarray(grid.dealias)
    noise *= run.seed_amplitude / np.sqrt(np.mean(np.real(np.fft.ifft2(noise)) ** 2))
    zeta0 = density0 = zeta = density = jnp.array(noise)
    real = lambda spectrum: np.real(np.asarray(jnp.fft.ifft2(spectrum)))  # noqa: E731
    times, energies, fluxes, frames = [], [], [], []
    for block in range(run.blocks):
        zeta, density = hw_run(zeta, density, grid, params, dt=run.dt, steps=run.steps_per_block)
        phi = potential_from_vorticity(zeta, grid)
        energies.append(float(jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2)) / grid.n**4)
        fluxes.append(float(particle_flux(zeta, density, grid)))
        times.append(run.dt * run.steps_per_block * (block + 1))
        frames.append(real(zeta) if run.save_frames else None)
        say(f"block {block + 1}/{run.blocks} t={times[-1]:.4g} energy={energies[-1]:.4e} flux={fluxes[-1]:+.4e}")
    summary = {"final_time": times[-1], "final_energy": energies[-1], "final_particle_flux": fluxes[-1],
               "mean_particle_flux_second_half": float(np.mean(fluxes[len(fluxes) // 2:]))}
    arrays = {"time": times, "energy": energies, "particle_flux": fluxes, "vorticity": real(zeta),
              "density": real(density)}
    if run.save_frames:
        arrays["vorticity_frames"] = np.stack(frames)
    if "autodiff" in cfg:
        spec, names = cfg["autodiff"], [f.name for f in dataclasses.fields(params)]
        if spec.parameter not in names:
            raise ValueError(f"[autodiff] parameter must be one of {names}")

        def objective(value):
            flux = hw_run_flux_history(zeta0, density0, grid, replace(params, **{spec.parameter: value}),
                                       dt=run.dt, steps=run.steps_per_block * run.blocks,
                                       sample_every=run.steps_per_block)[2]
            return jnp.mean(flux[len(flux) // 2:])

        x0 = float(getattr(params, spec.parameter))
        h = spec.fd_step * max(abs(x0), 1.0)
        grad = float(jax.jit(jax.grad(objective))(x0))
        fd = (float(objective(x0 + h)) - float(objective(x0 - h))) / (2.0 * h)
        summary["autodiff"] = {"parameter": spec.parameter, "value": x0, "gradient": grad,
                               "finite_difference": fd, "relative_error": abs(grad - fd) / max(abs(fd), 1e-30)}
        say(f"d<flux>/d{spec.parameter} = {grad:+.4e} (finite difference {fd:+.4e})")
    return summary, arrays


def _sol_flux_tube(cfg: dict, say: Callable[[str], None]):
    import jax.numpy as jnp

    from .geometry import build_open_slab_geometry
    from .native.fci_sheath_recycling import compute_fci_sheath_recycling
    from .native.sol_flux_tube import sol_flux_tube_run, sol_flux_tube_source

    params, run = cfg["sol_flux_tube"], cfg["run"]
    geometry = build_open_slab_geometry(**vars_of(cfg["open_slab"]))
    source = sol_flux_tube_source(geometry, params)
    density, momentum = jnp.ones(geometry.shape), jnp.zeros(geometry.shape)
    dt = run.cfl * float(geometry.spacing.dz[0, 0, 0]) / (params.sound_speed + 1.0)
    done, report_step, residual = 0, [], []
    while done < run.steps:
        chunk, previous = min(run.report_every, run.steps - done), density
        density, momentum = sol_flux_tube_run(density, momentum, geometry, params, source, dt=dt, steps=chunk)
        done += chunk
        report_step.append(done)
        residual.append(float(jnp.max(jnp.abs(density - previous)) / (chunk * dt)))
        say(f"step {done}/{run.steps} max|dn/dt|={residual[-1]:.2e}")
    n, m = np.asarray(density)[0, 0], np.asarray(momentum)[0, 0]
    mach = m / n / params.sound_speed
    full = lambda value: jnp.full(geometry.shape, value)  # noqa: E731
    sheath = compute_fci_sheath_recycling(density, full(run.electron_temperature), full(run.ion_temperature),
                                          geometry.maps, **vars_of(cfg["sheath"]))
    summary = {"mach_at_targets": [float(mach[0]), float(mach[-1])],
               "n_target_over_upstream": float(n[-1] / n[len(n) // 2]), "final_time": done * dt,
               **{k: float(getattr(sheath, k)) for k in (
                   "total_ion_particle_loss", "total_recycled_particle_source", "total_target_heat_load",
                   "particle_recycling_residual", "current_balance_residual")}}
    arrays = {"z": geometry.grid.z.centers, "density": n, "momentum": m, "mach": mach, "source": np.asarray(source)[0, 0],
              "report_step": report_step, "residual": residual,
              **{k: np.asarray(getattr(sheath, k))[0, 0] for k in (
                  "ion_particle_loss", "target_heat_load", "recycled_particle_source")}}
    return summary, arrays


def _sol_recycling(cfg: dict, say: Callable[[str], None]):
    import jax.numpy as jnp

    from .native.neutrals import (
        SolRecyclingState, linear_target_temperature_profile, sol_recycling_run, target_ion_flux,
    )

    params, profile, run = cfg["sol_recycling"], cfg["profile"], cfg["run"]
    nz = profile.nz
    dt = run.dt if run.dt is not None else 0.1 / nz
    temperature = linear_target_temperature_profile(nz, upstream_ev=profile.upstream_ev, target_ev=profile.target_ev,
                                                    normalization=params.normalization)
    cs = np.sqrt(2.0 * np.asarray(temperature) / params.ion_mass)

    def relax(p):
        state = SolRecyclingState(jnp.full(nz, p.upstream_density), jnp.zeros(nz), jnp.full(nz, profile.neutral_seed))
        for start in range(0, run.steps, run.report_every):
            state = sol_recycling_run(state, temperature, p, dt=dt, steps=min(run.report_every, run.steps - start))
        mach = np.asarray(state.ion_momentum) / (p.ion_mass * np.maximum(np.asarray(state.ion_density), p.density_floor)) / cs
        flux = float(target_ion_flux(state, temperature, p))
        say(f"n_up={p.upstream_density:g}: target Mach={mach[-1]:.4f} target flux={flux:.4f}")
        return state, mach, flux

    state, mach, flux = relax(params)
    scan = [relax(replace(params, upstream_density=n)) for n in cfg["scan"].upstream_density]
    neutral = np.asarray(state.neutral_density)
    summary = {"target_mach": float(mach[-1]), "target_ion_flux": flux, "dt": dt,
               "neutral_cushion_ratio": float(neutral[-1] / max(neutral[nz // 2], 1e-9))}
    arrays = {"z_over_L": (np.arange(nz) + 0.5) / nz, "temperature": temperature, "mach": mach,
              "ion_density": state.ion_density, "ion_momentum": state.ion_momentum, "neutral_density": neutral,
              "scan_upstream_density": cfg["scan"].upstream_density,
              "scan_target_mach": [s[1][-1] for s in scan], "scan_target_ion_flux": [s[2] for s in scan]}
    return summary, arrays


def _detachment(cfg: dict, say: Callable[[str], None]):
    from .native.neutrals import detachment_diagnostics, detachment_ledger, detachment_sol_run

    params, kwargs = cfg["detachment_sd1d"], vars_of(cfg["run"])
    rows, result = [], None
    for upstream in cfg["scan"].upstream_density or (params.upstream_density,):
        p = replace(params, upstream_density=upstream)
        if result is not None:
            kwargs["source_scale"] = result.source_scale
        result = detachment_sol_run(p, result.state if result else None, **kwargs)
        diag, ledger = detachment_diagnostics(result.state, p, result.source_scale), detachment_ledger(result, p)
        rows.append({"upstream_density": float(upstream),
                     **{k: float(getattr(diag, k)) for k in (
                         "target_temperature_ev", "target_ion_flux_area", "target_density", "target_mach",
                         "upstream_temperature_ev")},
                     "residual": float(result.residual), "iterations": int(result.iterations),
                     "converged": bool(result.converged),
                     "particle_imbalance": float(ledger["particle_imbalance"]),
                     "power_imbalance": float(ledger["power_imbalance"])})
        say(f"n_up={upstream:.3e}: T_t={rows[-1]['target_temperature_ev']:.2f} eV, {result.iterations} its")
    arrays = {k: [row[k] for row in rows] for k in rows[0]}
    arrays.update({f"profile_{k}": getattr(result.state, k) for k in result.state._fields})
    return {"points": rows}, arrays


def _turbulence(cfg: dict, say: Callable[[str], None]):
    from . import geometry as g
    from .native.stellarator_turbulence import run_stellarator_turbulence

    kind = next(k for k in ("rotating_ellipse", "shifted_torus", "island_divertor", "essos") if k in cfg)
    if kind == "essos":
        options = vars_of(cfg[kind])
        limiter_rho = options.pop("limiter_rho")
        geometry = g.essos_imported_geometry_to_fci(g.build_essos_imported_fci_geometry(**options), limiter_rho=limiter_rho)
    else:
        geometry = getattr(g, f"build_{kind}_geometry")(**vars_of(cfg[kind]))
    run = run_stellarator_turbulence(geometry, **vars_of(cfg["run"]))
    say(f"{kind} {tuple(geometry.shape)}: {len(run.times)} frames")
    summary = {"geometry": kind, "shape": list(geometry.shape), "n_frames": len(run.times),
               "initial_particle_content": float(run.particle_content[0]),
               "final_particle_content": float(run.particle_content[-1]),
               "total_target_flux": float(np.sum(run.target_flux)),
               "finite": bool(np.all(np.isfinite(run.density_frames)))}
    return summary, run._asdict()


def _fci_differentiable(cfg: dict, say: Callable[[str], None]):
    from .native.fci_differentiable_case import build_context, differentiability_report

    report = differentiability_report(build_context(**vars_of(cfg["fci_differentiable"])), **vars_of(cfg["run"]))
    state = report.pop("final_state")
    say(f"grad={report['grad']:+.6e} fd={report['fd']:+.6e} rel_error={report['rel_error']:.2e}")
    return report, {"density": state.density, "v_parallel": state.v_parallel}


def _hsx(cfg: dict, say: Callable[[str], None]):
    import hashlib

    from .fci_braginskii.geometry_build.fci_simulation_geometry import audit_fci_simulation_geometry_checksums
    from .fci_braginskii.geometry_build.hsx_simulation_geometry import (
        HsxSimulationGeometryConfig, build_hsx_simulation_geometry,
    )
    from .fci_braginskii.run import run

    config = cfg["fci_braginskii"]
    if "geometry" in cfg:
        if config.geometry is not None:
            raise ValueError("give either [fci_braginskii] geometry or a [geometry] table, not both")
        table = vars_of(cfg["geometry"])
        cache = table.pop("cache_dir")
        table["metric_mesh_shape"] = table["metric_mesh_shape"] or table["resolution"]
        table["coils_path"] = None if table["makegrid_path"] else table["coils_path"]
        key = hashlib.sha256(json.dumps(table, default=str, sort_keys=True).encode()).hexdigest()[:10]
        artifact = cache / f"hsx_{'x'.join(map(str, table['resolution']))}_{key}"
        if not (artifact.is_dir() and audit_fci_simulation_geometry_checksums(artifact)["valid"]):
            say(f"building HSX geometry in {artifact} (several minutes at 16^3)")
            build_hsx_simulation_geometry(HsxSimulationGeometryConfig(**table, output=artifact))
        config = replace(config, geometry=artifact)
    run(config)


@dataclass(frozen=True)
class Backend:
    tables: Callable[[], dict[str, type]]  # table name -> dataclass (lazy: imports JAX modules)
    run: Callable[[dict[str, Any], Callable[[str], None]], Any]
    one_of: tuple[str, ...] = ()  # exactly one of these tables is required
    required: tuple[str, ...] = ()
    generic: bool = True  # accepts [runtime] and [output]


def _lazy(module: str, name: str) -> type:
    import importlib

    return getattr(importlib.import_module(module, __package__), name)


def _sig(table: str) -> type:
    return _geometry_tables()[table]


def _hsx_tables() -> dict[str, type]:
    return {"fci_braginskii": _lazy(".fci_braginskii.run", "HsxBlobConfig"), "geometry": _hsx_geometry_table()}


def _turbulence_tables() -> dict[str, type]:
    from .native.stellarator_turbulence import run_stellarator_turbulence

    return {**{k: _sig(k) for k in ("rotating_ellipse", "shifted_torus", "island_divertor", "essos")},
            "run": from_signature("TurbulenceRun", run_stellarator_turbulence, exclude=("geometry",))}


def _detachment_tables() -> dict[str, type]:
    from .native.neutrals.detachment_sol_model import detachment_sol_run

    return {"detachment_sd1d": _lazy(".native.neutrals", "DetachmentSolParameters"), "scan": Scan,
            "run": from_signature("DetachmentRun", detachment_sol_run,
                                  exclude=("params", "state", "source_params", "verbose"))}


def _sol_tables() -> dict[str, type]:
    from .native.fci_sheath_recycling import compute_fci_sheath_recycling

    return {"sol_flux_tube": _lazy(".native.sol_flux_tube", "SolFluxTubeParameters"), "open_slab": _sig("open_slab"),
            "sheath": from_signature("Sheath", compute_fci_sheath_recycling,
                                     exclude=("density", "electron_temperature", "ion_temperature", "maps")),
            "run": SolRun}


BACKENDS: dict[str, Backend] = {
    "hasegawa_wakatani": Backend(lambda: {"hasegawa_wakatani": _lazy(".native.hasegawa_wakatani", "HasegawaWakataniParameters"),
                                          "grid": HwGrid, "run": HwRun, "autodiff": Autodiff}, _hw),
    "sol_flux_tube": Backend(_sol_tables, _sol_flux_tube, required=("open_slab",)),
    "sol_recycling": Backend(lambda: {"sol_recycling": _lazy(".native.neutrals", "SolRecyclingParameters"),
                                      "profile": RecyclingProfile, "run": RecyclingRun, "scan": Scan}, _sol_recycling),
    "detachment_sd1d": Backend(_detachment_tables, _detachment),
    "stellarator_turbulence": Backend(_turbulence_tables, _turbulence, required=("run",),
                                      one_of=("rotating_ellipse", "shifted_torus", "island_divertor", "essos")),
    "fci_differentiable": Backend(
        lambda: {"fci_differentiable": from_signature("FciDifferentiable", _lazy(".native.fci_differentiable_case", "build_context")),
                 "run": FciDiffRun}, _fci_differentiable),
    "fci_braginskii": Backend(_hsx_tables, _hsx, required=("fci_braginskii",), generic=False),
}


def _expected_tables(backend: Backend) -> dict[str, type]:
    tables = backend.tables()
    return {**tables, "runtime": RuntimeConfig, "output": OutputConfig} if backend.generic else tables


def parse_set(items: Iterable[str]) -> dict[str, dict[str, Any]]:
    """``["table.key=value", ...]`` (TOML value syntax; bare words are strings) -> ``{table: {key: value}}``."""

    out: dict[str, dict[str, Any]] = {}
    for item in items:
        target, sep, text = item.partition("=")
        table, dot, key = target.strip().partition(".")
        if not (sep and dot and key):
            raise ValueError(f"--set expects table.key=value, got {item!r}")
        try:
            value = tomllib.loads(f"v = {text.strip()}")["v"]
        except tomllib.TOMLDecodeError:
            value = text.strip()
        out.setdefault(table, {})[key] = value
    return out


def load_deck(path: str | Path | Mapping[str, Any], overrides: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[str, dict[str, Any]]:
    """Parse a deck file (or an equivalent dict) into ``(backend name, {table name: config object})``.

    ``overrides`` is ``{table: {key: value}}`` applied on top before validation."""

    if isinstance(path, Mapping):
        source, data = Path.cwd() / "deck.toml", {t: dict(v) for t, v in path.items()}
    else:
        source = Path(path)
        data = tomllib.loads(source.read_text(encoding="utf-8"))
    for table, keys in (overrides or {}).items():
        if not isinstance(data.get(table, {}), dict):
            raise ValueError(f"[{table}] must be a table")
        data.setdefault(table, {}).update(keys)
    model = data.get("model", {})
    name = model.get("backend") if isinstance(model, dict) else None
    if name not in BACKENDS:
        raise ValueError(f"[model] backend must be one of {sorted(BACKENDS)}, got {name!r}")
    if set(model) != {"backend"}:
        raise ValueError(f'[model] must contain only backend = "{name}"')
    backend, tables = BACKENDS[name], _expected_tables(BACKENDS[name])
    unknown = sorted(set(data) - set(tables) - {"model"})
    if unknown:
        raise ValueError(f"unknown table(s) in {name} deck: {unknown}; valid tables: {sorted(tables)}")
    config: dict[str, Any] = {}
    for table, cls in tables.items():
        if not isinstance(data.get(table, {}), dict):
            raise ValueError(f"[{table}] must be a table")
        if table in data or table in backend.required or (table not in backend.one_of and not _required(cls)):
            config[table] = build_table(cls, data.get(table, {}), f"[{table}]", source.resolve().parent)
    if backend.one_of and sum(table in config for table in backend.one_of) != 1:
        raise ValueError(f"exactly one of {['[' + t + ']' for t in backend.one_of]} must be given")
    if name == "fci_braginskii" and config["fci_braginskii"].geometry is None and "geometry" not in config:
        raise ValueError("[fci_braginskii] geometry is required (or give a [geometry] table)")
    return name, config


def describe(name: str, config: Mapping[str, Any]) -> str:
    """The resolved configuration, one ``table.key = value`` line per key."""

    return "\n".join([f"backend: {name}", *(f"{t}.{k} = {v!r}" for t, c in config.items() for k, v in vars_of(c).items())])


def run_deck(path: str | Path | Mapping[str, Any], *, output_dir: Path | None = None, case_name: str | None = None, quiet: bool = False, overrides: Mapping[str, Mapping[str, Any]] | None = None):
    """Run a deck; write ``<case>_summary.json`` and ``<case>_arrays.npz``; return the summary payload."""

    name, config = load_deck(path, overrides)
    output = config.get("output", OutputConfig())
    case = case_name or output.case_name or (name if isinstance(path, Mapping) else Path(path).stem)
    say = (lambda message: None) if quiet else (lambda message: print(f"[{case}] {message}", flush=True))
    if "runtime" in config:
        from .runtime import configure_jax_runtime

        configure_jax_runtime(precision=config["runtime"].precision)
    result = BACKENDS[name].run(config, say)
    if result is None:
        return None
    summary, arrays = result
    directory = Path(output_dir or output.directory or Path("output") / case)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"case_name": case, "backend": name, "results": summary,
               "config": json.loads(json.dumps({t: vars_of(c) for t, c in config.items()}, default=str))}
    (directory / f"{case}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(directory / f"{case}_arrays.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    paths = [directory / f"{case}_summary.json", directory / f"{case}_arrays.npz"]
    say(f"wrote {paths[0]} and {paths[1]}")
    return {**payload, "paths": [str(p) for p in paths]}


def run(deck: str | Path | Mapping[str, Any], *, output_dir: Path | None = None, case_name: str | None = None, quiet: bool = True, **overrides: Mapping[str, Any]):
    """Run a deck file or dict exactly as ``drbx run`` does; ``table={"key": value}`` kwargs override deck keys.

    Returns the summary payload (``results``, ``config``) plus ``paths`` of the files written."""

    return run_deck(deck, output_dir=output_dir, case_name=case_name, quiet=quiet, overrides=overrides)
