"""Input contract for the local P geometry build."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

from ...evaluators.eta_filtered_field import DEFAULT_TABLE_GRID, EtaFilterSpec


@dataclass(frozen=True)
class PerpendicularBuildOptions:
    """Supply the inputs and output directory before invoking the staged build.

    The caller configures JAX precision and backend. The cubic layout
    spans a full torus in either symmetry mode. Set map_periods explicitly
    to use a multi-period map with symmetry="none". covered_periods is
    the number of the map's periods spanned by the layout.
    """

    metric_cache: Path
    makegrid: Path
    output_directory: Path
    resolution: int = 16
    n_eta: int = 16
    makegrid_currents: tuple[float, ...] = (10722.0,) * 6 + (0.0,) * 6
    bfield_toroidal: str = "compact_hermite_c3"
    eta_filter: EtaFilterSpec | None = EtaFilterSpec()
    table_grid: tuple[int, int, float] = DEFAULT_TABLE_GRID
    table_cache_directory: Path | None = None
    field_periods: int = EtaFilterSpec.nfp
    symmetry: str = "nfp"
    layout_kind: str = "zernike_core_ring"
    chunk: int = 16384
    source_revision: str = "unknown"
    map_periods: int | None = None
    covered_periods: int | None = None
    workers: int = 1

    def __post_init__(self):
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError("P workers must be a positive integer")

    @property
    def period_counts(self) -> tuple[int, int]:
        map_periods = self.map_periods
        if map_periods is None:
            map_periods = self.field_periods if self.symmetry == "nfp" else 1
        covered_periods = map_periods if self.covered_periods is None else self.covered_periods
        for name, count in (("map_periods", map_periods), ("covered_periods", covered_periods)):
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
                raise ValueError(f"{name} must be a positive integer")
        if covered_periods != map_periods:
            raise ValueError("the full-torus Zernike core-ring layout must cover all map periods")
        return int(map_periods), int(covered_periods)

    @property
    def arm(self) -> str:
        return "raw" if self.eta_filter is None else "filtered"
