"""TOML interface for the nodal parallel and perpendicular manufactured case."""

from __future__ import annotations

import json
from .options import FIELD_PROFILE_NAMES, WALL_LAW_NAMES
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

@dataclass(frozen=True)
class NodalConfig:
    case_dir: Path
    resolution: int
    symmetry: str
    field_periods: int
    initial_state: Path | None = None
    owners: int = 1
    threads: int = 2

    def __post_init__(self):
        if self.initial_state is not None:
            raise ValueError("fci_nodal needs a physical initial-state loader and layout validation "
                             "to accept initial_state. This entry starts from manufactured data.")
        if min(self.resolution, self.field_periods) < 1:
            raise ValueError("resolution and field_periods must be positive")
        if self.owners < 1:
            raise ValueError("[fci_nodal] owners must be a positive integer")
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError("[fci_nodal] threads must be a positive integer")


@dataclass(frozen=True)
class GeometryConfig:
    makegrid_path: Path
    vessel_path: Path
    makegrid_currents: tuple[float, ...]
    rings: int
    resolution: tuple[int, int, int]
    workers: int = 1


@dataclass(frozen=True)
class FieldFilterConfig:
    max_harmonic_per_period: int
    samples_per_period: int
    table_grid: tuple[int, int, float]


@dataclass(frozen=True)
class ParallelConfig:
    workers: int = 1


@dataclass(frozen=True)
class MmsConfig:
    field_profile: str = "wall_flat_temperature"
    wall_law: str = "floating_wall"
    max_steps: int | None = None
    stop_after_first: bool = False
    capture_static_rhs: bool = False
    launch_mode: str = "local"
    rank_rss_limit_bytes: int | None = None
    run_timeout_seconds: float | None = None
    rank_command: tuple[str, ...] | None = None
    rank_index_env: str | None = None
    collectives_interface: str | None = None

    def __post_init__(self):
        if self.field_profile not in FIELD_PROFILE_NAMES or self.wall_law not in WALL_LAW_NAMES:
            raise ValueError("[mms] requires wall_flat_temperature or prescribed_wall_heat_flux and wall_law conducting_wall or floating_wall")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("[mms] max_steps must be positive")
        if self.launch_mode not in ("local", "per_owner", "scheduler"):
            raise ValueError("[mms] launch_mode must be local, per_owner or scheduler")
        if self.rank_rss_limit_bytes is not None and self.rank_rss_limit_bytes <= 0:
            raise ValueError("[mms] rank_rss_limit_bytes must be positive")
        timeout = self.run_timeout_seconds
        if timeout is not None and (timeout <= 0 or not isfinite(timeout)):
            raise ValueError("[mms] run_timeout_seconds must be positive and finite")


def tables():
    return {"fci_nodal": NodalConfig, "geometry": GeometryConfig,
            "field_filter": FieldFilterConfig, "parallel": ParallelConfig, "mms": MmsConfig}


def resolve_case_options(config):
    from .geometry_build import GeometryOptions
    from .operator_build.perp import PerpendicularBuildOptions
    from .operator_build.parallel import ParallelBuildOptions
    from .case import CaseOptions
    from .case._rank_launch import scheduler_options
    from .options import MetricOptions, PolygonBuildOptions, RuntimeOptions
    from .evaluators.eta_filtered_field import EtaFilterSpec

    run, geometry = config["fci_nodal"], config["geometry"]
    field, q_config, mms = config["field_filter"], config["parallel"], config["mms"]
    limits = mms.rank_rss_limit_bytes is not None or mms.run_timeout_seconds is not None
    if limits and mms.launch_mode == "local":
        raise ValueError("[mms] rank RSS and timeout options require launch_mode = 'per_owner'")
    scheduler_options(asdict(mms), owners=True)
    spec = EtaFilterSpec(max_harmonic_per_period=field.max_harmonic_per_period,
                         nfp=run.field_periods, samples_per_period=field.samples_per_period)
    if not geometry.makegrid_currents:
        raise ValueError("[geometry] makegrid_currents must list the device coil currents")
    metric = MetricOptions(makegrid_path=geometry.makegrid_path, vessel_path=geometry.vessel_path,
                           makegrid_currents=geometry.makegrid_currents, resolution=geometry.resolution,
                           field_filter=spec, table_grid=field.table_grid)
    layout = dict(resolution=run.resolution, symmetry=run.symmetry, field_periods=run.field_periods)
    polygon = PolygonBuildOptions(**layout, rings=geometry.rings, field_filter=spec,
                                  workers=geometry.workers)
    case_dir = run.case_dir.resolve()
    # The case driver replaces these output/input slots with receipt-bound products.
    cache, table = case_dir / "metric", case_dir / "field_table"
    partitioned = GeometryOptions.for_polygon(
        polygon, output=case_dir / "geometry", metric_cache=cache, field_table=table,
        polygon_geometry=case_dir / "polygon",
        polygon_merged=case_dir / "polygon",
        workers=geometry.workers,
    )
    q = ParallelBuildOptions(case_dir / "geometry", cache, table, case_dir / "parallel_operators",
                      **layout, eta_planes=run.resolution, filter_spec=spec,
                      workers=q_config.workers, owners=run.owners)
    p = PerpendicularBuildOptions(cache, metric.makegrid_path, case_dir / "perpendicular_operators", **layout,
                     n_eta=run.resolution, makegrid_currents=metric.makegrid_currents,
                     bfield_toroidal=metric.bfield_toroidal, eta_filter=spec, table_grid=field.table_grid,
                     map_periods=run.field_periods, covered_periods=run.field_periods)
    runtime = RuntimeOptions(**layout, owner_count=run.owners, cpu_device_count=run.owners)
    options = CaseOptions(metric=metric, polygon=polygon, geometry=partitioned,
                          parallel=q, perpendicular=p, runtime=runtime, owners=run.owners,
                          stage_threads=run.threads)
    runtime.check_cpu_device_capacity(runtime.cpu_device_count)
    runtime.check_mms_owner_count(run.resolution, runtime.owner_count)
    if mms.stop_after_first:
        runtime.check_first_milestone(run.resolution)
    return options


def _check_case_options(case_dir, options, stages=None):
    from .case._receipts import binding, stage_receipt

    # The indexed geometry and Q receipts also bind their build settings.
    for name, requested_options in (
        ("metric", options.metric),
        ("polygon", options.polygon),
        ("geometry", options.geometry),
        ("parallel_operators", options.parallel),
    ):
        receipt, _ = stage_receipt(case_dir, name) if stages is None else stages[name]
        recorded = binding(receipt)["options"]
        requested = (requested_options.numerical_options() if name == "polygon"
                     else asdict(requested_options))
        requested = json.loads(json.dumps(requested, default=str))
        differing = sorted(key for key in requested if requested[key] != recorded.get(key))
        if differing:
            raise ValueError(f"Built case {name} options differ from the deck: {', '.join(differing)}")


def _check_publication(case_dir, options, *, verify_only=False):
    from .case._receipts import PREDECESSORS, binding, stage_receipt, verify_chain

    if verify_only:
        receipt, stamp = verify_chain(case_dir, "owner_inputs")["owner_inputs"]
        stages = verify_chain(case_dir, "perpendicular_operators")
    else:
        # Read current receipt stamps here. The driver verifies the owner products before running.
        receipt, stamp = stage_receipt(case_dir, "owner_inputs")
        names = tuple(PREDECESSORS)
        stages = {name: stage_receipt(case_dir, name) for name in names[:names.index("perpendicular_operators") + 1]}
    _check_case_options(case_dir, options, stages)
    recorded = binding(receipt)["options"]
    geometry = options.geometry
    layout = [geometry.resolution, geometry.planes, geometry.symmetry, geometry.field_periods]
    if recorded.get("owner_count") != options.owner_parts or recorded.get("layout") != layout:
        raise ValueError("Built owner input layout differs from the deck")
    if recorded.get("build_receipts") != [current for _, current in stages.values()]:
        raise ValueError("Owner input receipt differs from the current build receipts")
    return Path(stamp["receipt"])


def build(config, say, *, verify_only=False):
    from .case import build_case

    case_dir = config["fci_nodal"].case_dir.resolve()
    options = resolve_case_options(config)
    if verify_only:
        manifest = _check_publication(case_dir, options, verify_only=True)
        say(f"verified {manifest}")
        return manifest
    for path in (options.metric.makegrid_path, options.metric.vessel_path):
        if not path.is_file():
            raise ValueError(f"Missing device input: {path}")
    manifest = build_case(options, case_dir)
    say(f"published {manifest}")
    return manifest


def run(config, say):
    from .case import run_mms
    from .case._rank_launch import SCHEDULER_OPTIONS

    case_dir = config["fci_nodal"].case_dir.resolve()
    options = resolve_case_options(config)
    if not (case_dir / "owner_inputs" / "receipt.json").is_file():
        raise ValueError(f"No published case at {case_dir}. Run drbx build with this deck first.")
    _check_publication(case_dir, options)
    mms = config["mms"]
    run_options = dict(runtime=options.runtime, field_profile=mms.field_profile, wall_law=mms.wall_law,
                       max_steps=mms.max_steps, stop_after_first=mms.stop_after_first,
                       capture_static_rhs=mms.capture_static_rhs)
    run_options.update(launch_mode=mms.launch_mode,
                       rank_rss_limit_bytes=mms.rank_rss_limit_bytes,
                       run_timeout_seconds=mms.run_timeout_seconds)
    if mms.launch_mode == "scheduler":
        run_options.update({key: getattr(mms, key) for key in SCHEDULER_OPTIONS})
    result = run_mms(case_dir, run_options, stage_threads=options.stage_threads)
    stage = "mms_run"
    say(f"advanced {result['steps']} MMS steps to t={result['time']:.6g} in {case_dir / stage}")
