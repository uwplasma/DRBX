"""Build Q products from a saved geometry in the caller's process."""
from dataclasses import dataclass
from pathlib import Path


from ...evaluators.eta_filtered_field import EtaFilterSpec


@dataclass(frozen=True)
class ParallelBuildOptions:
    geometry: Path
    metric_cache: Path
    field_table: Path
    out: Path
    resolution: int = 16
    eta_planes: int = 16
    symmetry: str = "nfp"
    field_periods: int = 4
    filter_spec: EtaFilterSpec | None = None
    arm: str | None = None
    face_chunk: int = 8192
    shard_rows: int = 32000
    support_widths: tuple[int, ...] = (25, 40, 64)
    expand_widths: tuple[int, ...] = (40, 64)
    workers: int = 1
    owners: int | None = None
    rss_limit_bytes: int | None = None

    def __post_init__(self):
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError("Q workers must be a positive integer")
        if self.owners is not None and (type(self.owners) is not int or not 1 <= self.owners <= self.eta_planes):
            raise ValueError("Q owners must be between 1 and eta_planes")

    @property
    def symmetry_period_planes(self):
        if self.symmetry == "none":
            return self.eta_planes
        if self.symmetry != "nfp":
            raise ValueError("symmetry must be 'nfp' or 'none'")
        if self.field_periods < 1 or self.eta_planes % self.field_periods:
            raise ValueError("eta_planes must be divisible by field_periods")
        return self.eta_planes // self.field_periods


def build_parallel_operators(options: ParallelBuildOptions) -> Path:
    geometry = Path(options.geometry)
    if geometry.is_dir() or geometry.name == "parallel_geometry_reader.json":
        from .build_by_owner import build_indexed_parallel_operators

        return build_indexed_parallel_operators(options)
    raise ValueError("Owner-only Q construction requires indexed geometry")
