"""Case worker adapters for the existing owner MMS preparation and operator."""

from time import perf_counter

from ._owner_inputs import OwnerParallelInputs, field_manifest, validate_inputs
from ._wall_data import CHI, D
from .._hashing import sha256_file as digest
from ._receipts import read_json, write_json

TARGET_TIMES = [6.905965144396657e-4, .005]


def prepare(options, out, inputs):
    import numpy as np

    from ..mms.prepare import MmsPreparationOptions, prepare_owners

    p, _, _, _ = validate_inputs(inputs, parts=options["runtime"].owner_count)
    wall = inputs["wall_data"]
    if tuple(read_json(wall/"manifest.json")["edges"]) != p.edges:
        raise ValueError("Wall and P owner edges differ")

    def read_q(owner):
        folder = wall/f"owner{owner:04d}"
        with np.load(folder/"mms_wall_hit_reference.npz", allow_pickle=False) as z:
            return tuple(z[key] for key in ("ids", "hit_points"))

    root = out/"prepared"
    prepare_owners(root, p, read_q, field_manifest(inputs,p), flow=inputs["manufactured_ion_flow_fit"], chi=CHI, D=D,
                   options=MmsPreparationOptions(field_profile=options["field_profile"]))
    return {"prepared": root}, 0


def _local(value):
    return value.addressable_data(0) if hasattr(value,"addressable_data") and not value.is_fully_addressable else value


def _solver_scalars(aux):
    import numpy as np

    if int(_local(aux["records"][0])):
        raise ArithmeticError("Invalid coupled owner Q record")
    result = {name: {key: np.asarray(_local(value)).item() for key,value in aux[name].items()}
              for name in ("ampere", "polarization")}
    if any(not value["converged"] for value in result.values()):
        raise ArithmeticError("Coupled owner solve did not converge")
    return result


def _write_rows(out, resolution, arrays, *, name):
    import numpy as np

    q = resolution.q
    shards = {key:{shard.device:shard.data for shard in array.addressable_shards}
              for key,array in arrays.items()}
    for owner,device in enumerate(q.mesh.devices.flat):
        if owner not in q.nodes.ids_by_owner:
            continue
        root = out/f"owner{owner:04d}"
        root.mkdir(exist_ok=True)
        ids = q.nodes.ids_by_owner[owner]
        keep = ids >= 0
        rows = {key:np.asarray(value[device])[keep] for key,value in shards.items()}
        if name == "state":
            np.save(root/"node_ids.npy", ids[keep], allow_pickle=False)
            np.save(root/"nodal_state.npy", rows["U"], allow_pickle=False)
        else:
            np.savez(root/"static_rhs.npz", **rows)


def run(options, out, inputs, *, rank=None):
    started = perf_counter()

    import json
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    from ..operator_build.perp.sbp_setup import _layout
    from ..drb import Switches
    from ..drb.owner_load import load_owner_resolution
    from ..drb.owner_operator import owner_actions, owner_step, owner_timestep
    from ..mms.cache import OwnerPrepared, load_owner_mms
    from ..mms.error_report import merge_reports, owner_masks, owner_partials, owner_weights, target_partials
    from ..mms.evolved import owner_stage
    from ..mms.restart import CHECKPOINT_DIR, Checkpoint, Paused

    runtime = options["runtime"]
    if rank is not None:
        for name,expected in rank["catalogs"].items():
            if digest(name)!=expected:
                raise ValueError(f"Rank input catalog changed: {name}")
    p, meta, construction, indexed = validate_inputs(inputs, parts=runtime.owner_count,
        indexed_identity=None if rank is None else rank["indexed"])
    layout = _layout(dict(p.manifest, n=p.manifest["N"]))
    runtime.check_mms_owner_count(layout.n, p.parts)
    first = options.get("stop_after_first", False)
    if first:
        runtime.check_first_milestone(layout.n)
    max_steps = options.get("max_steps")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    q = OwnerParallelInputs(inputs["owner_parallel_construction"], meta, construction, p,
                     checksums=None if rank is None else rank["mass"])
    wall = inputs["wall_data"]
    if tuple(read_json(wall/"manifest.json")["edges"]) != p.edges:
        raise ValueError("Wall and P owner edges differ")
    with np.load(wall/"constants.npz", allow_pickle=False) as z:
        constants = {name:z[name] for name in ("Om","Bdiv","Wm","gi")}
    routes = None
    devices = None
    if rank is not None:
        from ..drb.owner_route_store import RouteStore
        routes = RouteStore.open(rank["routes"],wall)
        devices = sorted(jax.devices("cpu"),key=lambda d:(d.process_index,d.id))
        if ([d.process_index for d in devices]!=list(range(p.parts))
                or jax.process_index()!=rank["owner"] or jax.local_device_count()!=1):
            raise ValueError("Expected one CPU device per owner in global rank order")
    resolution = load_owner_resolution(
        layout=layout, owner_p=p, q_products=inputs["owner_parallel"], wall_cache=wall,
        wall_constants=constants, perpendicular_identity_sha256=p.manifest["source_identity_sha256"],
        indexed_geometry=indexed, read_parallel_mass=q.mass, read_parallel_face_stiffness=q.K, options=runtime,
        devices=devices,routes=routes,
    )
    prepared = OwnerPrepared(inputs["prepared"])
    if ((prepared.manifest["field_profile"], tuple(prepared.edges)) != (options["field_profile"], p.edges)
            or prepared.manifest["source"]["owner_perpendicular_manifest_sha256"] != digest(p.root/"manifest.json")):
        raise ValueError("Prepared owner MMS lineage differs")
    wall_law = options["wall_law"]
    data = load_owner_mms(resolution, prepared, wall_law=wall_law)
    if rank is not None:
        from ..drb.rank_runtime import verify_replicas
        verify_replicas(resolution.q.mesh,(resolution.plan,resolution.q.arrays(),data),
                        metadata=(runtime,resolution.slabs,resolution.wall_grid,resolution.plan.opts))
    stage = owner_stage(resolution,data,wall_law=wall_law)
    raw,force,project = owner_actions(resolution,switches=Switches(em="on",wall=wall_law,limiter="afc"))
    outputs = out/"owners"
    outputs.mkdir(exist_ok=rank is not None)
    stats = {"stages":[]}
    if options.get("capture_static_rhs"):
        manufactured = stage(TARGET_TIMES[0])
        r,r_aux = jax.block_until_ready(raw(manufactured.exact,manufactured.wall))
        f,f_aux = jax.block_until_ready(force(manufactured.exact,manufactured))
        stats.update(static=_solver_scalars(r_aux), static_forced=_solver_scalars(f_aux))
        _write_rows(outputs,resolution,dict(rhs=r,rhs_manufactured=f),name="rhs")

    def rhs(u,m):
        result, aux = jax.block_until_ready(force(u,m))
        return result, _solver_scalars(aux)

    check = jax.jit(lambda u: jnp.all(jnp.isfinite(u)) & jnp.all(u[:,:3]>0))
    def observe(u):
        if not bool(_local(check(u))):
            raise ArithmeticError("Nonfinite or nonpositive owner MMS state")

    bound,dt = owner_timestep(resolution)
    dt = float(_local(dt))
    bound = {name:np.asarray(_local(value)).tolist() for name,value in bound.items()}
    targets = TARGET_TIMES[:1] if first else TARGET_TIMES
    masks = owner_masks(resolution.q.nodes, wall, q, inputs["owner_parallel"], inputs["indexed_geometry"])
    weights = owner_weights(resolution)
    report = dict(geometry=[owner_partials(h, *masks[owner]) for owner, h in weights.items()],
                  targets=[], seconds=[])
    checkpoint = restored = None
    controls = {}
    if rank is None and p.parts == 1 and (invocation := options.get("checkpoint")) is not None:
        controls = invocation["controls"]
        started = invocation.get("started", started)
        checkpoint = Checkpoint(out.parent/CHECKPOINT_DIR, dict(
            options=invocation["options"], input_receipts=invocation["input_receipts"],
            precision="float64" if jax.config.x64_enabled else "float32",
            jax_version=jax.__version__, threads=os.environ.get("NPROC")))
        restored = checkpoint.load()
    bounds = dict(dt=dt, positivity_bound=bound)
    if restored is None:
        t, cache_time, steps = 0., 0., 0
        m0 = stage(cache_time)
        u = m0.exact
    else:
        saved, progress = restored
        if json.dumps(progress["bounds"], sort_keys=True) != json.dumps(bounds, sort_keys=True):
            raise ValueError("MMS checkpoint time step or positivity bound differs")
        t, cache_time, steps = (progress[key] for key in ("time", "cache_time", "steps"))
        m0 = stage(cache_time)
        if saved.shape != m0.exact.shape or saved.dtype != m0.exact.dtype:
            raise ValueError("MMS checkpoint state shape or precision differs")
        u = jax.device_put(saved, m0.exact.sharding)
        stats["stages"] = progress["stages"]
        report.update(targets=progress["targets"], seconds=progress["seconds"])
    start_step = steps
    longest = max(report["seconds"], default=0.)
    interval = controls.get("checkpoint_interval_seconds", 600.)
    next_checkpoint = started+interval
    wall_budget = controls.get("segment_wall_seconds", float("inf"))
    segment_steps = controls.get("segment_steps", float("inf"))
    while t<targets[-1]-2e-15 and (max_steps is None or steps<max_steps):
        target = next(x for x in targets if x>t+2e-15)
        h = min(dt,target-t)
        begin = perf_counter()
        u,info,m0 = owner_step(rhs,u,t,h,prepare=stage,project=project,observe=observe,m0=m0)
        report["seconds"].append(perf_counter()-begin)
        t += h
        cache_time = t
        steps += 1
        stats["stages"].append(info)
        reached = abs(t-target)<2e-15
        if reached:
            milestone = m0 if t == target else stage(target)
            t = target
            _, aux = jax.block_until_ready(raw(u,milestone.wall))
            _solver_scalars(aux)
            partials = target_partials(resolution,weights,masks,u,milestone,data,t,aux)
            report["targets"].append(dict(time=t,steps=steps,partials=partials))
        if checkpoint is not None:
            now = perf_counter()
            longest = max(longest, report["seconds"][-1])
            finished = t >= targets[-1]-2e-15 or steps == max_steps
            pause = not finished and (steps-start_step >= segment_steps or now-started+1.5*longest > wall_budget)
            if reached or now >= next_checkpoint or pause or finished:
                checkpoint.save(u, dict(time=t, cache_time=cache_time, steps=steps, bounds=bounds,
                                        stages=stats["stages"], targets=report["targets"], seconds=report["seconds"]))
                next_checkpoint = perf_counter()+interval
            if pause:
                return Paused(t, steps), 0
    _write_rows(outputs,resolution,dict(U=u),name="state")
    info = dict(time=t,steps=steps,dt=dt,positivity_bound=bound,
                field_profile=options["field_profile"],solves=stats,edges=p.edges)
    if rank is not None:
        return info, report
    summary = out/"result.json"
    write_json(summary,info)
    errors = out/"mms_errors.json"
    write_json(errors,merge_reports([report],dt=dt,steps=steps))
    return {"owner_state":outputs,"summary":summary,"mms_errors":errors}, 0
