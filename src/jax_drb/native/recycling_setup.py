from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp

from ..config.boutinp import BoutConfig, NumericResolver
from .array_backend import use_jax_backend
from .expression import ArrayExpressionEvaluator
from .mesh import StructuredMesh, broadcast_to_field_shape
from .metrics import StructuredMetrics
from .open_field import apply_noflow_flow_guards, apply_noflow_scalar_guards
from .open_field import TargetBoundaryGeometry, build_target_boundary_geometry
from .recycling_fields import recycling_evolving_variable_names


@dataclass(frozen=True)
class OpenFieldSpecies:
    name: str
    density: np.ndarray
    pressure: np.ndarray
    momentum: np.ndarray
    charge: float
    atomic_mass: float
    density_floor: float
    has_pressure: bool
    has_momentum: bool
    noflow_lower_y: bool
    noflow_upper_y: bool
    target_recycle: bool
    recycle_as: str | None
    target_recycle_multiplier: float
    target_recycle_energy: float
    target_fast_recycle_fraction: float
    target_fast_recycle_energy_factor: float

    @property
    def density_name(self) -> str:
        return f"N{self.name}"

    @property
    def pressure_name(self) -> str:
        return f"P{self.name}"

    @property
    def momentum_name(self) -> str:
        return f"NV{self.name}"


@dataclass(frozen=True)
class DensityFeedbackController:
    species_name: str
    density_upstream: float
    density_controller_p: float
    density_controller_i: float
    density_integral_positive: bool
    density_source_positive: bool
    density_source_shape: np.ndarray
    diagnose: bool


@dataclass(frozen=True)
class RecyclingRuntimeModel:
    species_templates: dict[str, OpenFieldSpecies]
    controllers: dict[str, DensityFeedbackController]
    explicit_pressure_sources: dict[str, np.ndarray]
    density_source_overrides: dict[str, np.ndarray] | None
    pressure_source_overrides: dict[str, np.ndarray] | None
    momentum_source_overrides: dict[str, np.ndarray] | None
    preserve_dump_target_state: bool
    preserve_dump_ion_target_state_only: bool
    field_names: tuple[str, ...]
    feedback_names: tuple[str, ...]
    lower_target_geometry: TargetBoundaryGeometry | None = None
    upper_target_geometry: TargetBoundaryGeometry | None = None


def try_literal_reference(config: BoutConfig, raw_value: str) -> tuple[str, str] | None:
    value = raw_value.strip()
    if not (value.startswith("`") and value.endswith("`")):
        return None
    reference = value[1:-1]
    if ":" not in reference:
        return None
    section, key = reference.split(":", 1)
    if not config.has_section(section) or not config.has_option(section, key):
        return None
    return section, key


def evaluate_field_value(
    config: BoutConfig,
    variable_name: str,
    *,
    mesh: StructuredMesh,
    option_name: str,
) -> np.ndarray:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    field = broadcast_to_field_shape(evaluator.resolve_option(variable_name, option_name), mesh)
    return np.asarray(field, dtype=np.float64)


def evaluate_field_option(config: BoutConfig, variable_name: str, *, mesh: StructuredMesh) -> np.ndarray:
    raw_value = (
        config.raw(variable_name, "function")
        if config.has_option(variable_name, "function")
        else config.raw(variable_name, "solution")
    )
    resolved_reference = try_literal_reference(config, raw_value)
    if resolved_reference is not None:
        return evaluate_field_value(config, resolved_reference[0], mesh=mesh, option_name=resolved_reference[1])
    return evaluate_field_value(
        config,
        variable_name,
        mesh=mesh,
        option_name="function" if config.has_option(variable_name, "function") else "solution",
    )


def evaluate_option_field(
    config: BoutConfig,
    section: str,
    option_name: str,
    *,
    mesh: StructuredMesh,
) -> np.ndarray:
    raw_value = config.raw(section, option_name)
    resolved_reference = try_literal_reference(config, raw_value)
    if resolved_reference is not None:
        return evaluate_field_value(config, resolved_reference[0], mesh=mesh, option_name=resolved_reference[1])
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    field = broadcast_to_field_shape(evaluator.resolve_option(section, option_name), mesh)
    return np.asarray(field, dtype=np.float64)


def resolve_species_numeric_option(config: BoutConfig, section: str, option_name: str) -> float:
    raw_value = config.raw(section, option_name).strip()
    resolved_reference = try_literal_reference(config, raw_value)
    if resolved_reference is None and raw_value.startswith("`") and "`:" in raw_value:
        section_end = raw_value.find("`:", 1)
        if section_end > 1:
            referenced_section = raw_value[1:section_end]
            referenced_option = raw_value[section_end + 2 :]
            if config.has_section(referenced_section) and config.has_option(referenced_section, referenced_option):
                resolved_reference = (referenced_section, referenced_option)
    resolver = NumericResolver(config)
    if resolved_reference is not None:
        return float(resolver.resolve(resolved_reference[0], resolved_reference[1]))
    return float(resolver.resolve(section, option_name))


def explicit_pressure_source(
    config: BoutConfig,
    species_name: str,
    *,
    mesh: StructuredMesh,
    dataset_scalars: dict[str, float],
) -> np.ndarray:
    section = f"P{species_name}"
    if not config.has_section(section) or not config.has_option(section, "source"):
        return np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    raw_value = config.raw(section, "source")
    resolved_reference = try_literal_reference(config, raw_value)
    if resolved_reference is not None:
        field = evaluate_field_value(config, resolved_reference[0], mesh=mesh, option_name=resolved_reference[1])
    else:
        evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
        field = broadcast_to_field_shape(evaluator.resolve_option(section, "source"), mesh)
    source_normalisation = 1.60218e-19 * dataset_scalars["Nnorm"] * dataset_scalars["Tnorm"] * dataset_scalars["Omega_ci"]
    return np.asarray(field, dtype=np.float64) / source_normalisation


def load_explicit_pressure_sources(
    config: BoutConfig,
    *,
    species_templates: dict[str, OpenFieldSpecies],
    mesh: StructuredMesh,
    dataset_scalars: dict[str, float],
) -> dict[str, np.ndarray]:
    return {
        name: explicit_pressure_source(config, name, mesh=mesh, dataset_scalars=dataset_scalars)
        for name in species_templates
        if species_templates[name].has_pressure or name == "e"
    }


def initialize_species(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    dataset_scalars: dict[str, float] | None = None,
    field_overrides: dict[str, np.ndarray] | None = None,
) -> dict[str, OpenFieldSpecies]:
    resolver = NumericResolver(config)
    overrides = field_overrides or {}
    scalars = dataset_scalars or {}
    model_species = []
    for section in config.sections:
        if section == "e":
            model_species.append(section)
            continue
        if not config.has_option(section, "type"):
            continue
        type_values = config.parsed(section, "type")
        type_items = type_values if isinstance(type_values, tuple) else (type_values,)
        if any(str(item).startswith("evolve_") or str(item) in {"quasineutral", "neutral_mixed"} for item in type_items):
            model_species.append(section)

    species: dict[str, OpenFieldSpecies] = {}
    for name in model_species:
        density_name = f"N{name}"
        pressure_name = f"P{name}"
        momentum_name = f"NV{name}"
        density = np.asarray(overrides[density_name], dtype=np.float64, copy=True) if density_name in overrides else (
            evaluate_field_option(config, density_name, mesh=mesh) if config.has_section(density_name) else None
        )
        if name == "e":
            pressure = np.asarray(overrides[pressure_name], dtype=np.float64, copy=True) if pressure_name in overrides else evaluate_field_option(config, pressure_name, mesh=mesh)
            momentum = np.zeros_like(pressure, dtype=np.float64)
        else:
            if density is None:
                raise KeyError(f"Missing density section for {name}.")
            pressure = np.asarray(overrides[pressure_name], dtype=np.float64, copy=True) if pressure_name in overrides else (
                evaluate_field_option(config, pressure_name, mesh=mesh) if config.has_section(pressure_name) else density.copy()
            )
            momentum = np.asarray(overrides[momentum_name], dtype=np.float64, copy=True) if momentum_name in overrides else (
                evaluate_field_option(config, momentum_name, mesh=mesh) if config.has_section(momentum_name) else np.zeros_like(density, dtype=np.float64)
            )

        type_values = config.parsed(name, "type")
        components = tuple(str(item) for item in (type_values if isinstance(type_values, tuple) else (type_values,)))
        noflow = "noflow_boundary" in components
        species[name] = OpenFieldSpecies(
            name=name,
            density=np.array(density if density is not None else pressure, dtype=np.float64, copy=True),
            pressure=np.array(pressure, dtype=np.float64, copy=True),
            momentum=np.array(momentum, dtype=np.float64, copy=True),
            charge=float(resolver.resolve(name, "charge")) if config.has_option(name, "charge") else (-1.0 if name == "e" else 0.0),
            atomic_mass=float(resolver.resolve(name, "AA")) if config.has_option(name, "AA") else (1.0 / 1836.0),
            density_floor=float(resolver.resolve(name, "density_floor")) if config.has_option(name, "density_floor") else 1.0e-7,
            has_pressure="evolve_pressure" in components or name == "e" or "neutral_mixed" in components,
            has_momentum="evolve_momentum" in components or "neutral_mixed" in components,
            noflow_lower_y=bool(config.parsed(name, "noflow_lower_y")) if config.has_option(name, "noflow_lower_y") else noflow,
            noflow_upper_y=bool(config.parsed(name, "noflow_upper_y")) if config.has_option(name, "noflow_upper_y") else noflow,
            target_recycle=bool(config.parsed(name, "target_recycle")) if config.has_option(name, "target_recycle") else False,
            recycle_as=str(config.parsed(name, "recycle_as")) if config.has_option(name, "recycle_as") else None,
            target_recycle_multiplier=float(resolver.resolve(name, "target_recycle_multiplier")) if config.has_option(name, "target_recycle_multiplier") else 0.0,
            target_recycle_energy=(
                float(resolver.resolve(name, "target_recycle_energy")) / float(scalars.get("Tnorm", 1.0))
                if config.has_option(name, "target_recycle_energy")
                else 0.0
            ),
            target_fast_recycle_fraction=float(resolver.resolve(name, "target_fast_recycle_fraction")) if config.has_option(name, "target_fast_recycle_fraction") else 0.0,
            target_fast_recycle_energy_factor=float(resolver.resolve(name, "target_fast_recycle_energy_factor")) if config.has_option(name, "target_fast_recycle_energy_factor") else 0.0,
        )

    for name, sp in tuple(species.items()):
        density = sp.density
        pressure = sp.pressure
        momentum = sp.momentum
        if sp.noflow_lower_y and mesh.has_lower_y_target:
            density = np.asarray(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64)
            pressure = np.asarray(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64)
            momentum = np.asarray(apply_noflow_flow_guards(momentum, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64)
        if sp.noflow_upper_y and mesh.has_upper_y_target:
            density = np.asarray(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=False, upper_y=True), dtype=np.float64)
            pressure = np.asarray(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=False, upper_y=True), dtype=np.float64)
            momentum = np.asarray(apply_noflow_flow_guards(momentum, mesh=mesh, lower_y=False, upper_y=True), dtype=np.float64)
        species[name] = OpenFieldSpecies(**{**sp.__dict__, "density": density, "pressure": pressure, "momentum": momentum})
    return species


def override_species_fields(
    species_templates: dict[str, OpenFieldSpecies],
    *,
    fields: dict[str, np.ndarray],
    mesh: StructuredMesh,
) -> dict[str, OpenFieldSpecies]:
    species: dict[str, OpenFieldSpecies] = {}
    dynamic_values = tuple(
        fields.get(field_name)
        for template in species_templates.values()
        for field_name in (template.density_name, template.pressure_name, template.momentum_name)
        if field_name in fields
    )
    use_jax = use_jax_backend(*dynamic_values)

    def _array(value):
        return jnp.asarray(value, dtype=jnp.float64) if use_jax else np.asarray(value, dtype=np.float64)

    for name, template in species_templates.items():
        density = _array(fields.get(template.density_name, template.density))
        pressure = _array(fields.get(template.pressure_name, template.pressure))
        momentum = _array(fields.get(template.momentum_name, template.momentum))
        if template.noflow_lower_y and mesh.has_lower_y_target:
            density = _array(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=True, upper_y=False))
            pressure = _array(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=True, upper_y=False))
            momentum = _array(apply_noflow_flow_guards(momentum, mesh=mesh, lower_y=True, upper_y=False))
        if template.noflow_upper_y and mesh.has_upper_y_target:
            density = _array(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=False, upper_y=True))
            pressure = _array(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=False, upper_y=True))
            momentum = _array(apply_noflow_flow_guards(momentum, mesh=mesh, lower_y=False, upper_y=True))
        species[name] = OpenFieldSpecies(**{**template.__dict__, "density": density, "pressure": pressure, "momentum": momentum})
    return species


def load_density_feedback_controllers(
    config: BoutConfig,
    *,
    species: dict[str, OpenFieldSpecies],
    mesh: StructuredMesh,
    dataset_scalars: dict[str, float],
) -> dict[str, DensityFeedbackController]:
    resolver = NumericResolver(config)
    nnorm = float(dataset_scalars["Nnorm"])
    omega_ci = float(dataset_scalars["Omega_ci"])
    controllers: dict[str, DensityFeedbackController] = {}
    for name, sp in species.items():
        if name == "e" or sp.charge <= 0.0 or not config.has_option(name, "type"):
            continue
        type_values = config.parsed(name, "type")
        components = tuple(str(item).strip() for item in (type_values if isinstance(type_values, tuple) else (type_values,)))
        if "upstream_density_feedback" not in components:
            continue
        density_section = f"N{name}"
        if config.has_option(density_section, "source_shape"):
            raw_value = config.raw(density_section, "source_shape")
            resolved_reference = try_literal_reference(config, raw_value)
            if resolved_reference is not None:
                source_shape = evaluate_field_value(config, resolved_reference[0], mesh=mesh, option_name=resolved_reference[1])
            else:
                evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
                source_shape = broadcast_to_field_shape(evaluator.resolve_option(density_section, "source_shape"), mesh)
            source_shape = np.asarray(source_shape, dtype=np.float64) / (nnorm * omega_ci)
        else:
            source_shape = np.zeros_like(sp.density, dtype=np.float64)
        controllers[name] = DensityFeedbackController(
            species_name=name,
            density_upstream=float(resolver.resolve(name, "density_upstream")) / nnorm,
            density_controller_p=float(resolver.resolve(name, "density_controller_p")) if config.has_option(name, "density_controller_p") else 1.0e-2,
            density_controller_i=float(resolver.resolve(name, "density_controller_i")) if config.has_option(name, "density_controller_i") else 1.0e-3,
            density_integral_positive=bool(config.parsed(name, "density_integral_positive")) if config.has_option(name, "density_integral_positive") else False,
            density_source_positive=bool(config.parsed(name, "density_source_positive")) if config.has_option(name, "density_source_positive") else True,
            density_source_shape=source_shape,
            diagnose=bool(config.parsed(name, "diagnose")) if config.has_option(name, "diagnose") else False,
        )
    return controllers


def build_recycling_runtime_model(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics | None = None,
    dataset_scalars: dict[str, float],
    field_overrides: dict[str, np.ndarray] | None = None,
    field_template_overrides: dict[str, np.ndarray] | None = None,
    density_source_overrides: dict[str, np.ndarray] | None = None,
    pressure_source_overrides: dict[str, np.ndarray] | None = None,
    momentum_source_overrides: dict[str, np.ndarray] | None = None,
    preserve_dump_target_state: bool = False,
    preserve_dump_ion_target_state_only: bool = False,
) -> RecyclingRuntimeModel:
    species_templates = initialize_species(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=field_template_overrides if field_template_overrides is not None else field_overrides,
    )
    controllers = load_density_feedback_controllers(
        config,
        species=species_templates,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
    )
    lower_target_geometry = (
        build_target_boundary_geometry(
            J=np.asarray(metrics.J, dtype=np.float64),
            dy=np.asarray(metrics.dy, dtype=np.float64),
            dx=np.asarray(metrics.dx, dtype=np.float64),
            dz=np.asarray(metrics.dz, dtype=np.float64),
            g_22=np.asarray(metrics.g_22, dtype=np.float64),
            y_index=mesh.ystart,
            guard_index=mesh.ystart - 1,
        )
        if metrics is not None and mesh.has_lower_y_target and mesh.myg > 0
        else None
    )
    upper_target_geometry = (
        build_target_boundary_geometry(
            J=np.asarray(metrics.J, dtype=np.float64),
            dy=np.asarray(metrics.dy, dtype=np.float64),
            dx=np.asarray(metrics.dx, dtype=np.float64),
            dz=np.asarray(metrics.dz, dtype=np.float64),
            g_22=np.asarray(metrics.g_22, dtype=np.float64),
            y_index=mesh.yend,
            guard_index=mesh.yend + 1,
        )
        if metrics is not None and mesh.has_upper_y_target and mesh.myg > 0
        else None
    )
    field_names = recycling_evolving_variable_names(species_templates)
    return RecyclingRuntimeModel(
        species_templates=species_templates,
        controllers=controllers,
        explicit_pressure_sources=load_explicit_pressure_sources(
            config,
            species_templates=species_templates,
            mesh=mesh,
            dataset_scalars=dataset_scalars,
        ),
        density_source_overrides=None if density_source_overrides is None else {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in density_source_overrides.items()},
        pressure_source_overrides=None if pressure_source_overrides is None else {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in pressure_source_overrides.items()},
        momentum_source_overrides=None if momentum_source_overrides is None else {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in momentum_source_overrides.items()},
        preserve_dump_target_state=preserve_dump_target_state,
        preserve_dump_ion_target_state_only=preserve_dump_ion_target_state_only,
        field_names=field_names,
        feedback_names=tuple(sorted(controllers)),
        lower_target_geometry=lower_target_geometry,
        upper_target_geometry=upper_target_geometry,
    )
