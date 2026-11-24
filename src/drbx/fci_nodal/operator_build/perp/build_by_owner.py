"""Direct production of the perpendicular nodal and Laplacian metric leaves by whole eta slab."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from ...operators.nodal.nodal_layout import eta_planes, wall_points
from ...operators.perp.geometry_bundle import layout_identity
from ..parallel.job_storage import guard_rss, peak_rss
from .eta_filter import arm_identity
from .extraction import _laplacian_values, _nodal_values
from .owner_bundle import _leaf_entry, _plane_edges, _publish_owner_bundle
from ..._hashing import sha256_file as _sha
from .reference import build_reference


def _owner_worker(options, layout, provenance):
    limit = os.environ.get("DRBX_FCI_NODAL_WORKER_RSS_BYTES")
    if limit:
        guard_rss(int(limit))
    t0 = time.perf_counter()
    provenance = dict(provenance)
    provider = build_reference(options, provenance)
    return options, layout, provenance, provider, time.perf_counter() - t0


def _initialize(options, layout, provenance):
    global _worker
    _worker = _owner_worker(options, layout, provenance)


def _sample(job, context):
    owner, root, begin, end = job
    options, layout, provenance, provider, worker_seconds = context
    start = time.perf_counter()
    planes = slice(begin, end)
    eta = eta_planes(layout)[planes]
    E, P = len(eta), layout.P
    points = np.stack([np.broadcast_to(layout.node_u, (E, P)), np.broadcast_to(layout.node_theta, (E, P)),
                       np.broadcast_to(eta[:, None], (E, P))], axis=-1)
    wp = wall_points(layout, layout.walls[0], planes=planes)
    nodal = _nodal_values(provider, points, options.chunk)
    lap = _laplacian_values(provider.reference, points, wp, options.chunk)
    arm = getattr(provider.reference, "eta_filter_arm", None)
    arm_meta = arm.meta() if arm is not None else None
    fields = dict(points=points, wall_points=wp, h=nodal.h, jac=nodal.jac, B=nodal.B,
                  K=nodal.K, A=lap["A"], divA=lap["divA"], ginv_u=lap["ginv_u"],
                  wall_A_row=lap["wall_A"][..., 0, :], wall_ginv_u=lap["wall_ginv_u"], wall_J=lap["wall_J"])
    fields = {key:np.asarray(value, dtype=np.float64) for key,value in fields.items()}
    if (not all(np.isfinite(a).all() for a in fields.values())
            or np.any(nodal.jac <= 0.) or np.any(nodal.B <= 0.)):
        raise ValueError("Nonfinite P owner leaves or nonpositive nodal J/B")
    # Match the gathered normalization checks by reporting local maxima.
    Hp = [np.array(jac, copy=True) for jac in (nodal.jac, lap["J"])]
    core = layout.blocks[0]
    for values in Hp:
        values[:, :core.n_nodes] /= core.u
        values *= layout.wxy[None, :]
    agreement = dict(jac_difference=float(np.max(np.abs(nodal.jac-lap["J"]))),
                     jac_scale=float(np.max(np.abs(nodal.jac))),
                     Hp_difference=float(np.max(np.abs(Hp[0]-Hp[1]))),
                     Hp_scale=float(np.max(np.abs(Hp[0]))))
    fields["nodal_meta_eta_filter"] = np.array(json.dumps(arm_meta, sort_keys=True))
    directory = root/f"owner{owner:04d}"
    directory.mkdir()
    files = {}
    for name, values in fields.items():
        path = directory/f"{name}.npy"
        with path.open("xb") as stream:
            np.save(stream, values, allow_pickle=False)
        files[name] = _leaf_entry(root, path, values.shape, values.dtype)
    info = dict(pid=os.getpid(), seconds=time.perf_counter()-start,
                worker_setup_seconds=worker_seconds, peak_rss_bytes=peak_rss())
    return dict(begin=begin,end=end,files=files), arm_meta, provenance, float(lap["sign"]), agreement, info


def _run_owner(job):
    return _sample(job, _worker)


def _use_table(options, out, provenance, shared_table):
    if options.eta_filter is None:
        return options
    identity = arm_identity(provenance, options)
    directory = options.table_cache_directory or out/"field_table"
    options = replace(options, table_cache_directory=Path(directory).resolve())
    path = options.table_cache_directory/f"field_table_{identity[:16]}.npz"
    if shared_table is not None and (path != Path(shared_table).resolve() or not path.is_file()):
        raise ValueError(f"P owner stage requires the shared field table {path}")
    if not path.is_file():
        # One controller builds an absent table, so workers never race to publish it.
        provider = build_reference(options, dict(provenance))
        provider.reference.eta_filter_arm.table()
    provenance["field_table"] = dict(path=str(path), sha256=_sha(path))
    return options


def _build_owners(options, layout, out, provenance, parts, *, shared_table):
    start = time.perf_counter()
    edges = _plane_edges(layout.n_eta, parts)
    root = out/"owners"
    root.mkdir(parents=True)
    provenance = dict(provenance)
    options = _use_table(options, out, provenance, shared_table)
    counts = list(zip(edges[:-1], edges[1:]))
    jobs = [(owner, root, begin, end) for owner, (begin, end) in enumerate(counts)]
    workers = min(options.workers, parts)
    if workers == 1:
        context = _owner_worker(options, layout, provenance)
        results = [_sample(job, context) for job in jobs]
        del context
    else:
        # Match the geometry scheduler's spawned, private evaluators.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize, initargs=(options,layout,provenance)) as pool:
            results = list(pool.map(_run_owner, jobs, chunksize=1))
    owners, arms, inputs, signs, agreements, measurements = zip(*results, strict=True)
    if any((arm, source, sign) != (arms[0], inputs[0], signs[0])
           for arm, source, sign in zip(arms, inputs, signs, strict=True)):
        raise ValueError("P owner inputs or Jacobian signs disagree")
    agreement = {}
    for key in ("jac", "Hp"):
        maximum = max(row[key+"_difference"] for row in agreements)
        relative = maximum/max(row[key+"_scale"] for row in agreements)
        if relative > 1e-12:
            raise ValueError(f"{key} differs between the two P metric evaluations: {relative:.3e}")
        agreement[key] = relative
    leaves = {key:dict(shape=(layout.n_eta,*entry["shape"][1:]) if entry["shape"] else (),
                       dtype=entry["dtype"]) for key,entry in owners[0]["files"].items()}
    identity = dict(schema="drbx.perpendicular.owner_source.v1", n=layout.n, n_eta=layout.n_eta, P=layout.P,
                    arm=options.arm, arm_identity=arms[0], layout_kind=options.layout_kind,
                    sign=signs[0], agreement=agreement, layout_sha256=layout_identity(layout),
                    source_revision=options.source_revision, created=dict(inputs=inputs[0]))
    identity_path = root/"identity.json"
    with identity_path.open("x") as stream:
        json.dump(identity, stream, sort_keys=True, indent=2)
        stream.write("\n")
    reduce_started = time.perf_counter()
    directories = [root/f"owner{owner:04d}" for owner in range(parts)]
    _publish_owner_bundle(root, identity_path, identity, edges, directories, owners, leaves)
    record = out/"owner_build.json"
    with record.open("x") as stream:
        json.dump(dict(workers=workers, owners=dict(enumerate(measurements)),
                       reduction_seconds=time.perf_counter()-reduce_started,
                       seconds=time.perf_counter()-start, peak_rss_bytes=peak_rss()), stream, indent=2)
        stream.write("\n")
    return dict(owner_perpendicular_bundle=root, owner_p_build=record)
