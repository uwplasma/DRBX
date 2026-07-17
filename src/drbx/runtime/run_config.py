from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..config.boutinp import BoutConfig, NumericResolver, ROOT_SECTION
from ..config.model import has_model_section, locate_model_section
from ..config.normalization import ModelNormalization
from .scheduler import ComponentRequest, expand_component_requests


@dataclass(frozen=True)
class TimeConfig:
    nout: int
    timestep: float


@dataclass(frozen=True)
class ParallelTransformConfig:
    type: str
    options: Mapping[str, bool | int | float | str | tuple[str, ...]]


@dataclass(frozen=True)
class MeshScalarConfig:
    nx: int | None
    ny: int | None
    nz: int | None
    mxg: int
    myg: int
    mz: int | None
    zperiod: float | None
    file: str | None
    extrapolate_y: bool | None
    parallel_transform: ParallelTransformConfig
    resolved_scalars: Mapping[str, float]


@dataclass(frozen=True)
class SolverConfig:
    type: str | None
    mxstep: int | None
    rtol: float | None
    atol: float | None
    use_precon: bool | None
    cvode_max_order: int | None
    mms: bool | None
    raw_options: Mapping[str, bool | int | float | str | tuple[str, ...]]


@dataclass(frozen=True)
class RunConfiguration:
    time: TimeConfig
    mesh: MeshScalarConfig
    solver: SolverConfig
    normalization: ModelNormalization | None
    components: tuple[ComponentRequest, ...]
    root_scalars: Mapping[str, float]
    model_scalars: Mapping[str, float]

    @classmethod
    def from_config(cls, config: BoutConfig) -> "RunConfiguration":
        resolver = NumericResolver(config)
        time = TimeConfig(
            nout=int(round(_resolve_required_scalar(config, resolver, ROOT_SECTION, "nout"))),
            timestep=_resolve_required_scalar(config, resolver, ROOT_SECTION, "timestep"),
        )
        mesh = _build_mesh_config(config, resolver)
        solver = _build_solver_config(config, resolver)
        normalization = ModelNormalization.from_config(config) if _has_normalization(config) else None
        model_section = locate_model_section(config) if has_model_section(config) else None
        return cls(
            time=time,
            mesh=mesh,
            solver=solver,
            normalization=normalization,
            components=expand_component_requests(config) if model_section is not None else (),
            root_scalars=_resolved_scalars(config, resolver, ROOT_SECTION),
            model_scalars=_resolved_scalars(config, resolver, model_section) if model_section is not None else {},
        )


def _build_mesh_config(config: BoutConfig, resolver: NumericResolver) -> MeshScalarConfig:
    mesh_scalars = _resolved_scalars(config, resolver, "mesh") if config.has_section("mesh") else {}
    parallel_section = "mesh:paralleltransform"
    parallel_options = dict(_resolved_parsed_options(config, parallel_section)) if config.has_section(parallel_section) else {}
    parallel_type = str(parallel_options.get("type", "identity"))
    file_value = config.parsed("mesh", "file") if config.has_option("mesh", "file") else None
    extrapolate_y = bool(config.parsed("mesh", "extrapolate_y")) if config.has_option("mesh", "extrapolate_y") else None
    return MeshScalarConfig(
        nx=_optional_int(mesh_scalars.get("nx")),
        ny=_optional_int(mesh_scalars.get("ny")),
        nz=_optional_int(mesh_scalars.get("nz")),
        mxg=_with_default(_optional_int(_optional_root_scalar(config, resolver, "MXG")), 2),
        myg=_with_default(_optional_int(_optional_root_scalar(config, resolver, "MYG")), 2),
        mz=_optional_int(_optional_root_scalar(config, resolver, "MZ")),
        zperiod=_optional_root_scalar(config, resolver, "zperiod"),
        file=file_value if isinstance(file_value, str) else None,
        extrapolate_y=extrapolate_y,
        parallel_transform=ParallelTransformConfig(type=parallel_type, options=parallel_options),
        resolved_scalars=mesh_scalars,
    )


def _build_solver_config(config: BoutConfig, resolver: NumericResolver) -> SolverConfig:
    raw_options = dict(_resolved_parsed_options(config, "solver")) if config.has_section("solver") else {}
    return SolverConfig(
        type=str(raw_options["type"]) if "type" in raw_options else None,
        mxstep=_optional_int(raw_options.get("mxstep")),
        rtol=_optional_float(raw_options.get("rtol")),
        atol=_optional_float(raw_options.get("atol")),
        use_precon=_optional_bool(raw_options.get("use_precon")),
        cvode_max_order=_optional_int(raw_options.get("cvode_max_order")),
        mms=_optional_bool(raw_options.get("mms")),
        raw_options={**raw_options, **_resolved_scalars(config, resolver, "solver")} if config.has_section("solver") else {},
    )


def _resolved_parsed_options(config: BoutConfig, section: str) -> Mapping[str, bool | int | float | str | tuple[str, ...]]:
    if not config.has_section(section):
        return {}
    options: dict[str, bool | int | float | str | tuple[str, ...]] = {}
    for key, entry in config.section(section).items():
        options[key] = entry.value.raw if entry.value.kind == "expression" else entry.value.parsed
    return options


def _resolved_scalars(config: BoutConfig, resolver: NumericResolver, section: str) -> Mapping[str, float]:
    if not config.has_section(section):
        return {}
    scalars: dict[str, float] = {}
    for key, entry in config.section(section).items():
        if entry.value.kind in {"string", "list"}:
            continue
        try:
            scalars[key] = resolver.resolve(section, key)
        except (KeyError, TypeError, ValueError, SyntaxError):
            continue
    return scalars


def _resolve_required_scalar(config: BoutConfig, resolver: NumericResolver, section: str, key: str) -> float:
    if not config.has_option(section, key):
        raise KeyError(f"Missing required scalar option {section}:{key}")
    return resolver.resolve(section, key)


def _optional_root_scalar(config: BoutConfig, resolver: NumericResolver, key: str) -> float | None:
    if not config.has_option(ROOT_SECTION, key):
        return None
    return resolver.resolve(ROOT_SECTION, key)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(round(float(value)))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def _with_default(value: Any, default: Any) -> Any:
    if value is None:
        return default
    return value


def _has_normalization(config: BoutConfig) -> bool:
    if not has_model_section(config):
        return False
    model_section = locate_model_section(config)
    return all(config.has_option(model_section, key) for key in ("Nnorm", "Tnorm", "Bnorm"))
