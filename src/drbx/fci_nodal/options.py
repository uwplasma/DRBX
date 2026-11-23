"""Caller-owned settings for the supported runtime and builders."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .evaluators.eta_filtered_field import DEFAULT_TABLE_GRID, EtaFilterSpec


CONDUCTING_WALL = 'conducting_wall'
FLOATING_WALL = 'floating_wall'
WALL_LAW_NAMES = (CONDUCTING_WALL, FLOATING_WALL)
WALL_FLAT_TEMPERATURE = 'wall_flat_temperature'
PRESCRIBED_WALL_HEAT_FLUX = 'prescribed_wall_heat_flux'
FIELD_PROFILE_NAMES = (WALL_FLAT_TEMPERATURE, PRESCRIBED_WALL_HEAT_FLUX)


def validate_face_record_chunk(value):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("[mms] face_record_chunk must be a positive integer")


@dataclass(frozen=True)
class RuntimeOptions:
    tau: float = 1.
    mu: float = 1836.
    owner_count: int = 1
    resolution: int | None = None
    symmetry: str | None = None
    field_periods: int | None = None
    cpu_device_count: int = 1
    single_owner_from_resolution: int | None = 32
    first_milestone_resolution: int | None = 16

    def __post_init__(self):
        if (self.tau, self.mu) != (1., 1836.):
            raise ValueError('Supported model parameters require tau=1 and mu=1836')
        if self.owner_count < 1:
            raise ValueError('owner_count must be positive')
        if self.symmetry not in (None, 'nfp', 'none'):
            raise ValueError('symmetry must be nfp or none')
        if self.resolution is not None and self.resolution < 1:
            raise ValueError('resolution must be positive')
        if self.field_periods is not None and self.field_periods < 1:
            raise ValueError('field_periods must be positive')
        if type(self.cpu_device_count) is not int or self.cpu_device_count < 1:
            raise ValueError('cpu_device_count must be a positive integer')
        for name in ('single_owner_from_resolution', 'first_milestone_resolution'):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f'{name} must be a positive integer or None')

    def check_cpu_device_capacity(self, available):
        if self.owner_count > available:
            raise ValueError(f'owner_count={self.owner_count} needs that many CPU devices; '
                             f'set cpu_device_count={self.owner_count} for the case worker or '
                             f'XLA_FLAGS=--xla_force_host_platform_device_count={self.owner_count} before importing JAX')

    def check_mms_owner_count(self, resolution, owners):
        threshold = self.single_owner_from_resolution
        if threshold is not None and resolution >= threshold and owners != 1:
            raise ValueError(f'N{threshold} and higher use one owner only')

    def check_first_milestone(self, resolution):
        allowed = self.first_milestone_resolution
        if allowed is not None and resolution != allowed:
            raise ValueError(f'First-milestone control is resolution {allowed} only')


# Defaults used by the staged HSX metric builder.
HSX_QHS_MAIN_CURRENT_AMPERES = 10722.0
DEFAULT_HSX_QHS_MAKEGRID_CURRENTS = (
    (HSX_QHS_MAIN_CURRENT_AMPERES,) * 6 + (0.0,) * 6
)
@dataclass(frozen=True, kw_only=True)
class MetricOptions:
    """Locked HSX inputs, with all file paths supplied by the caller.

    For the table, supply a metric_cache_path and field_table_dir. A None
    field_filter uses the raw field. makegrid_sha256 may be supplied from a
    verified input manifest; otherwise the table builder hashes the file.
    The metric uses the fixed spline fit and exact level-set planes.
    """

    makegrid_path: Path
    vessel_path: Path | None = None
    metric_cache_dir: Path | None = None
    makegrid_currents: tuple[float, ...] | None = DEFAULT_HSX_QHS_MAKEGRID_CURRENTS
    resolution: tuple[int, int, int] = (16, 16, 16)
    reference_magnetic_field: float | None = None
    rebuild_metric_cache: bool = False
    metric_cache_path: Path | None = None
    field_table_dir: Path | None = None
    field_filter: EtaFilterSpec | None = EtaFilterSpec()
    table_grid: tuple[int, int, float] = DEFAULT_TABLE_GRID
    bfield_toroidal: str = "compact_hermite_c3"
    makegrid_sha256: str | None = None


def boundary_bisection_count(resolution, override=None):
    count = (12 if resolution == 16 else 10) if override is None else override
    if type(count) is not int or count < 0:
        raise ValueError(f'boundary bisections must be a nonnegative integer, got {count!r}')
    return count


@dataclass(frozen=True)
class PolygonBuildOptions:
    resolution: int = 16
    field_periods: int = 4
    symmetry: str = "nfp"
    rings: int = 9
    field_filter: EtaFilterSpec | None = None
    boundary_bisections: int | None = None
    wall_bisections: int = 12
    ray_budget: int | None = None
    supplemental_ray_budget: int = 2000
    merge_fraction: float = .1
    trace_sizing_workers: int = 127
    workers: int = 1

    def __post_init__(self):
        if self.symmetry not in ("nfp", "none") or self.field_periods < 1:
            raise ValueError("Expected nfp or none and positive physical field periods")
        if self.resolution <= self.rings or self.rings < 2:
            raise ValueError("The source disk needs two certified rings and a positive inner radius")
        if self.resolution % self.copies:
            raise ValueError("Eta planes must divide into whole copies")
        if self.trace_sizing_workers < 1:
            raise ValueError("Trace sizing needs a positive worker count")
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError("Polygon workers must be a positive integer")

    def numerical_options(self):
        settings = asdict(self)
        settings.pop("workers")
        return settings

    @property
    def copies(self):
        return self.field_periods if self.symmetry == "nfp" else 1

    @property
    def source_planes(self):
        return self.resolution // self.copies

    @property
    def period_angle(self):
        return (2 * np.pi) / self.copies

    @property
    def face_subdivision(self):
        return max(4, self.resolution // 4)

    @property
    def disk_angles(self):
        return 4 * self.resolution

    @property
    def bisections(self):
        return boundary_bisection_count(self.resolution, self.boundary_bisections)

    @property
    def max_rays(self):
        if self.ray_budget is not None:
            return self.ray_budget
        return 1_200_000 if self.resolution == 48 else 2_200_000


DEFAULT_POLYROUTE_OPTIONS = PolygonBuildOptions()
