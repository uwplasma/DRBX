"""Case-level stage ordering and public entry points."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import fcntl
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import TYPE_CHECKING

from ._adoption import validate_adoption
from ._receipts import PREDECESSORS, binding, read_json, receipt_products, stage, verify_chain

if TYPE_CHECKING:
    from ..geometry_build import GeometryOptions
    from ..operator_build.perp import PerpendicularBuildOptions
    from ..operator_build.parallel import ParallelBuildOptions
    from ..options import MetricOptions, PolygonBuildOptions, RuntimeOptions


@dataclass(frozen=True)
class CaseOptions:
    metric: "MetricOptions"
    polygon: "PolygonBuildOptions"
    geometry: "GeometryOptions"
    parallel: "ParallelBuildOptions"
    perpendicular: "PerpendicularBuildOptions"
    runtime: "RuntimeOptions"
    owners: int = 1
    stage_threads: int = 2

    def __post_init__(self):
        n = self.geometry.resolution
        e = self.geometry.planes
        spec = self.metric.field_filter
        layout = (n, self.geometry.symmetry, self.geometry.field_periods)
        for option in (self.polygon, self.parallel, self.perpendicular):
            if (option.resolution, option.symmetry, option.field_periods) != layout:
                raise ValueError("Stage resolution, symmetry and physical periods must agree")
        if (self.parallel.eta_planes, self.perpendicular.n_eta) != (e, e):
            raise ValueError("Stage plane counts must agree")
        parts = self.owner_parts
        if type(parts) is not int or parts < 1:
            raise ValueError("owners must be a positive integer")
        if type(self.stage_threads) is not int or self.stage_threads < 1:
            raise ValueError("stage_threads must be a positive integer")
        if self.parallel.owners not in (None, parts):
            raise ValueError("q.owners must match owners when specified")
        if self.runtime.owner_count not in (1, parts):
            raise ValueError("Runtime owner_count must be 1 or match owners")
        if self.runtime.cpu_device_count not in (1, parts):
            raise ValueError("Runtime cpu_device_count must be 1 or match owners")
        if e // parts < 4:
            raise ValueError("Owner export requires at least four eta planes per owner")
        if any(value != spec for value in (
            self.polygon.field_filter,
            self.geometry.field_filter, self.parallel.filter_spec, self.perpendicular.eta_filter,
        )):
            raise ValueError("Pass one filter specification to every consumer")
        if (
            self.geometry.polygon_rings, self.geometry.angular_subdivisions, self.geometry.bisections
        ) != (self.polygon.rings, self.polygon.disk_angles // n, self.polygon.bisections):
            raise ValueError("Use GeometryOptions.for_polygon for the production polygon route")
        if any(actual not in (None, expected) for actual, expected in zip(
            (self.runtime.resolution, self.runtime.symmetry, self.runtime.field_periods), layout, strict=True,
        )):
            raise ValueError("Runtime options disagree with the build layout")
        if (self.geometry.support_widths, self.geometry.expand_widths) != (self.parallel.support_widths, self.parallel.expand_widths):
            raise ValueError("Geometry and Q owner support settings must agree")
        if (Path(self.metric.makegrid_path).resolve() != Path(self.perpendicular.makegrid).resolve()
                or self.metric.makegrid_currents != self.perpendicular.makegrid_currents
                or self.metric.bfield_toroidal != self.perpendicular.bfield_toroidal
                or self.metric.table_grid != self.perpendicular.table_grid):
            raise ValueError("Metric/table and P must use the same MAKEGRID, currents, interpolation and table grid")


    @property
    def owner_parts(self) -> int:
        return self.owners


@contextmanager
def _case_lock(case):
    case.mkdir(parents=True, exist_ok=True)
    with (case / ".case.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _settings(name, *, stage_threads=2, host_cpu_devices=1, device="cpu"):
    threads = "1" if name in ("polygon", "geometry") else str(stage_threads)
    settings = {
        **{key: threads for key in ("NPROC", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                  "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                                  "TF_NUM_INTRAOP_THREADS")},
        "TF_NUM_INTEROP_THREADS": "1",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "DRBX_DISABLE_COMPILATION_CACHE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={host_cpu_devices}",
    }
    if name == "mms_run" and device == "gpu":
        settings.update(JAX_PLATFORMS="cuda,cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")
        settings.pop("CUDA_VISIBLE_DEVICES")
    return settings


def _execute(name, options, out, inputs, settings):
    reply = out / "worker_reply.pkl"
    env = {**os.environ, **settings}
    env.pop("DRBX_HOST_DEVICE_COUNT", None)
    subprocess.run(
        [sys.executable, "-B", "-m", f"{__package__}._worker", name, str(out), str(reply)],
        env=env, input=pickle.dumps((options, inputs)), check=True,
    )
    with reply.open("rb") as stream:
        return pickle.load(stream)


def _flow():
    from importlib.resources import files

    flow = files(f"{__package__.rsplit('.', 1)[0]}.mms").joinpath("manufactured_ion_flow_fit.json")
    if flow.is_file():
        return Path(str(flow))
    return Path(__file__).with_name("mms") / "manufactured_ion_flow_fit.json"


def _run_parallel_build_stage(case, options, dependencies, *, adopt=None):
    """Run or adopt Q against the case's four receipted direct inputs."""
    q = replace(options.parallel, owners=options.owner_parts)
    inputs = {role: path for receipt, _ in dependencies for role, path in receipt_products(receipt).items()}
    settings = _settings("parallel_operators", stage_threads=options.stage_threads)
    return stage(
        case, "parallel_operators", settings=settings, options=asdict(q),
        inputs=(), expected=("parallel_construction",), dependencies=dependencies, adopt=adopt,
        verify_adoption=lambda origin: validate_adoption(
            "parallel_operators", origin, (q.resolution, q.eta_planes, q.symmetry, q.field_periods),
            inputs, options.metric,
        ),
        action=lambda out: _execute("parallel_operators", q, out, inputs, settings),
    )


def build_case(options: CaseOptions, case_dir, *, adopt=None):
    """Build or resume a case. Tests can explicitly adopt stage/role paths."""
    case = Path(case_dir).resolve()
    adopt = {} if adopt is None else adopt
    stages = {}

    def run(name, stage_options, expected, *, raw=(), payload=None):
        deps = tuple(stages[key] for key in PREDECESSORS[name])
        inputs = {role: path for receipt, _ in stages.values() for role, path in receipt_products(receipt).items()}
        settings = _settings(name, stage_threads=options.stage_threads)
        stages[name] = stage(
            case, name, settings=settings, options=stage_options, inputs=raw, expected=expected,
            dependencies=deps, adopt=adopt.get(name),
            verify_adoption=lambda origin: validate_adoption(
                name, origin, (geometry.resolution, geometry.planes, geometry.symmetry,
                               geometry.field_periods), inputs, metric,
            ),
            action=lambda out: _execute(
                name, stage_options if payload is None else payload, out, inputs, settings),
        )

    if "owner_inputs" in adopt:
        if set(adopt) != {"owner_inputs"}:
            raise ValueError("Adopt owner_inputs alone with owner stages enabled")
        with _case_lock(case):
            return _publish_owner_inputs(case, options, adopt=adopt["owner_inputs"])
    allowed = {"metric", "field_table", "polygon", "geometry", "parallel_operators", "perpendicular_operators"}
    if unknown := set(adopt) - allowed:
        raise ValueError(f"Unknown adoption stages: {sorted(unknown)}")
    with _case_lock(case):
        metric = options.metric
        geometry = options.geometry
        run("metric", asdict(metric), ("metric_cache",),
            raw=(metric.makegrid_path, metric.vessel_path), payload=metric)
        run("field_table", asdict(metric), ("field_table",), payload=metric)
        run("polygon", options.polygon.numerical_options(), ("polygon_geometry", "polygon_merged", "polygon_case"),
            payload=options.polygon)
        run("geometry", asdict(geometry), ("indexed_geometry",), payload=geometry,
            raw=(geometry.geometry_reference_key_table,))
        stages["parallel_operators"] = _run_parallel_build_stage(
            case, options, tuple(stages[key] for key in PREDECESSORS["parallel_operators"]),
            adopt=adopt.get("parallel_operators"),
        )
        map_periods = options.perpendicular.field_periods if options.perpendicular.map_periods is None else options.perpendicular.map_periods
        covered_periods = map_periods if options.perpendicular.covered_periods is None else options.perpendicular.covered_periods
        p = replace(options.perpendicular, map_periods=map_periods, covered_periods=covered_periods)
        parts = options.owner_parts
        run("perpendicular_operators", dict(asdict(p), owner_count=parts), ("owner_perpendicular_bundle",), payload=(p, parts))
        return _publish_owner_inputs(case, options, stages=stages)


def _publish_owner_inputs(case, options, *, adopt=None, stages=None):
    from ._owner_inputs import ROLES, validate_inputs

    geometry = options.geometry
    layout = (geometry.resolution, geometry.planes, geometry.symmetry, geometry.field_periods)
    config = dict(owner_count=options.owner_parts, layout=layout)
    inputs = {}
    if stages is not None:
        config["build_receipts"] = [stamp for _,stamp in stages.values()]
        inputs = {role:path for receipt,_ in stages.values() for role,path in receipt_products(receipt).items()}
        inputs["manufactured_ion_flow_fit"] = _flow()
    settings = _settings("owner_inputs", stage_threads=options.stage_threads)
    def verify(origin):
        adopted = {role:Path(path) for role,path in origin["products"].items()}
        if set(adopted) != set(ROLES):
            raise ValueError("Unexpected owner input roles")
        validate_inputs(adopted, parts=options.owner_parts, layout=layout)

    _, stamp = stage(case, "owner_inputs", settings=settings, options=config, inputs=(),
                     expected=ROLES, adopt=adopt, verify_adoption=verify,
                     action=lambda out:_execute("owner_inputs",config,out,inputs,settings))
    return Path(stamp["receipt"])


def _run_owner_mms(case, mms_options, *, stage_threads, controls):
    from ..options import validate_mms_device

    initial = verify_chain(case,"owner_inputs")["owner_inputs"]
    parts = binding(initial[0])["options"]["owner_count"]
    launch = mms_options.get("launch_mode", "local")
    device = mms_options.get("device", "cpu")
    validate_mms_device(device, launch, owners=parts)
    if controls and parts != 1:
        raise ValueError("MMS segment and checkpoint controls require one owner")
    requested = mms_options["runtime"]
    if requested.owner_count not in (1, parts):
        raise ValueError("Runtime and published owner counts differ")
    if requested.cpu_device_count not in (1, parts):
        raise ValueError("Runtime cpu_device_count must be 1 or match the published owner count")
    runtime = replace(requested,owner_count=parts,cpu_device_count=parts)
    mms_options = dict(mms_options,runtime=runtime)
    inputs = receipt_products(initial[0])
    checksums = dict(initial[0]["output_files"])

    def run_stage(name, opts, payload, roles, deps):
        ranked = name=="mms_run" and launch in ("per_owner","scheduler")
        settings = _settings(name, stage_threads=stage_threads, host_cpu_devices=1 if ranked else parts,
                             device=device)
        if ranked:
            from ._rank_launch import launch_rank_run
            action = lambda out:launch_rank_run(payload,out,inputs,settings,checksums=checksums)
        else:
            action = lambda out:_execute(name,payload,out,inputs,settings)
        return stage(case,name,settings=settings,options=opts,inputs=(),expected=roles,
                     dependencies=deps,action=action)

    wall = run_stage("wall_data",{"runtime":asdict(runtime)},runtime,("wall_data",),(initial,))
    inputs.update(receipt_products(wall[0]))
    checksums.update(wall[0]["output_files"])
    field_profile = mms_options["field_profile"]
    prep = run_stage("mms_prepare",{"runtime":asdict(runtime),"field_profile":field_profile},
                     dict(runtime=runtime,field_profile=field_profile),("prepared",),(initial,wall))
    inputs.update(receipt_products(prep[0]))
    checksums.update(prep[0]["output_files"])
    dependencies = (initial,wall,prep)
    scientific = dict(mms_options,runtime=asdict(runtime))
    payload = mms_options
    if launch == "local" and parts == 1:
        payload = dict(mms_options,checkpoint=dict(controls=controls, options=scientific,
                       input_receipts=[stamp for _, stamp in dependencies]))
    run = run_stage("mms_run",scientific,payload,("owner_state","summary","mms_errors"),dependencies)
    if run[1] is None:
        return dict(paused=True, time=run[0].time, steps=run[0].steps)
    products = receipt_products(run[0])
    return dict(owner_state=products["owner_state"],**read_json(products["summary"]))


def run_mms(case_dir, mms_options, *, stage_threads=2):
    """Prepare and advance the manufactured case, returning the staged runner's result.

    mms_options supplies runtime (RuntimeOptions), field_profile, wall_law and optionally
    max_steps and stop_after_first, as in the integrated MMS entry points.
    stage_threads sets the worker thread limit for wall data, preparation and advance.
    """
    from ._rank_launch import scheduler_options
    from ..mms.restart import SEGMENT_KEYS, segment_controls
    from ..options import validate_face_record_chunk, validate_mms_device

    if type(stage_threads) is not int or stage_threads < 1:
        raise ValueError("stage_threads must be a positive integer")
    chunk = mms_options.get("face_record_chunk")
    validate_face_record_chunk(chunk)
    case = Path(case_dir).resolve()
    launch = mms_options.get("launch_mode","local")
    if launch not in ("local","per_owner","scheduler"):
        raise ValueError("launch_mode must be local, per_owner or scheduler")
    device = mms_options.get("device", "cpu")
    validate_mms_device(device, launch)
    limits = any(mms_options.get(key) is not None
                 for key in ("rank_rss_limit_bytes", "run_timeout_seconds"))
    with _case_lock(case):
        if limits and launch == "local":
            raise ValueError("Owner rank RSS and timeout options require launch_mode=per_owner")
        scheduler_options(mms_options, owners=True)
        controls = segment_controls(mms_options)
        scientific = {key: value for key, value in mms_options.items() if key not in SEGMENT_KEYS}
        if chunk is None:
            scientific.pop("face_record_chunk", None)
        if device == "cpu":
            scientific.pop("device", None)
        return _run_owner_mms(case, scientific, stage_threads=stage_threads, controls=controls)
