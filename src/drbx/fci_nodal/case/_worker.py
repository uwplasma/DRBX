"""Numerical entry-point adapters, invoked in a fresh configured interpreter."""

from dataclasses import replace
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

from ._receipts import peak_rss, read_json, worker_code_files


def metric(options, out, inputs):
    from ..evaluators.cache import build_metric_cache

    return {"metric_cache": build_metric_cache(replace(options, metric_cache_dir=out))}, 0


def field_table(options, out, inputs):
    from ..evaluators.field_table import build_field_table

    path = build_field_table(replace(options, metric_cache_path=inputs["metric_cache"], field_table_dir=out))
    if path is None:
        raise NotImplementedError("Polygons, geometry and Q require a field table")
    return {"field_table": path}, 0


def _counted(module):
    function, counts = module.trace, [0]

    def count(evaluator, seeds, *args, **kwargs):
        counts[0] += len(seeds)
        return function(evaluator, seeds, *args, **kwargs)

    return patch.object(module, "trace", count), counts


def polygon(options, out, inputs):
    from ..operator_build.parallel import open_route, shared_polygon
    from ..evaluators.eta_filtered_field import EtaFilterTable
    from ..operators.nodal.nodal_layout_specs import build_zernike_core_ring_layout

    table = EtaFilterTable.load(inputs["field_table"])
    route = open_route(out, metric_cache=inputs["metric_cache"], field_table=table,
                       field_table_path=inputs["field_table"], layout_builder=build_zernike_core_ring_layout,
                       shared_polygon=shared_polygon, options=options)
    products = route.run()
    return {"polygon_geometry": products["geometry"], "polygon_merged": products["merged"],
            "polygon_case": products["case_record"], "polygon_route": out}, route.ray_ledger()["new_rays"]


def geometry(options, out, inputs):
    from ..geometry_build import build_geometry, rays

    trace, count = _counted(rays)
    roles = {name: inputs[name] for name in (
        "metric_cache", "field_table", "polygon_geometry", "polygon_merged",
    )}
    with trace:
        product = build_geometry(replace(options, **roles, output=out / "indexed",
                                         polygon_receipt=inputs["polygon_case"]))
    rays_traced = read_json(product / "sharded_geometry_complete.json")["new_rays"] if options.workers > 1 else count[0]
    return {"indexed_geometry": product}, rays_traced


def parallel_operators(options, out, inputs):
    from ..operator_build.parallel import build_parallel_operators

    geometry = inputs["indexed_geometry"]
    if limit := os.environ.get("DRBX_FCI_NODAL_WORKER_RSS_BYTES"):
        cap = int(limit)
        options = replace(options, rss_limit_bytes=min(cap, options.rss_limit_bytes or cap))
    runtime = build_parallel_operators(replace(
        options, geometry=geometry, metric_cache=inputs["metric_cache"],
        field_table=inputs["field_table"], out=out / "runtime",
    ))
    return {"parallel_construction": runtime}, 0


def perpendicular_operators(options, out, inputs):
    from ..operator_build.perp import build_nodal_bundle

    options, owner_parts = options
    metric = inputs["metric_cache"]
    table = inputs["field_table"]
    options = replace(options, metric_cache=metric, table_cache_directory=table.parent,
                      output_directory=out)
    return build_nodal_bundle(options, owner_parts=owner_parts, shared_table=table), 0
def _export_parallel_application(q, owner_p, index, out, meta):
    from ..operator_build.parallel.application import export_parallel_application
    from ..options import RuntimeOptions

    construction = read_json(q / "owners/manifest.json")
    runtime = RuntimeOptions(owner_count=len(construction["owners"]), resolution=meta["N"],
                             symmetry=meta["symmetry"], field_periods=meta["field_periods"])
    return export_parallel_application(q, index, out / "parallel_application", owner_p=owner_p, options=runtime)


def owner_inputs(options, out, inputs):
    from ._owner_inputs import ROLES, validate_inputs

    index = inputs["indexed_geometry"]
    meta = read_json(index/"geometry.json")
    q = inputs["parallel_construction"]
    owner_parallel = _export_parallel_application(q, inputs["owner_perpendicular_bundle"], index, out, meta)
    products = dict(inputs, owner_parallel=owner_parallel, owner_parallel_construction=q/"owners")
    products = {role:products[role] for role in ROLES}
    validate_inputs(products, parts=options["owner_count"])
    return products, 0


def wall_data(options, out, inputs):
    from ._wall_data import build_wall_products

    return build_wall_products(options, out, inputs)


def mms_prepare(options, out, inputs):
    from ._mms_stages import prepare

    return prepare(options, out, inputs)


def mms_run(options, out, inputs):
    from ._mms_stages import run

    return run(options, out, inputs)


def execute_local(name, options, out, inputs):
    product, rays = globals()[name](options, out, inputs)
    return product, rays, peak_rss(), worker_code_files(name)


def main():
    # Only the caller's pipe carries Python objects. Case and receipt files are JSON.
    name, out, reply = sys.argv[1:]
    limit = os.environ.get("DRBX_FCI_NODAL_WORKER_RSS_BYTES")
    if limit:
        from threading import Event, Thread
        stop = Event()

        def monitor():
            while not stop.wait(.05):
                if peak_rss() >= int(limit):
                    os.write(2, b"Case stage exceeded RSS limit\n")
                    os._exit(97)

        Thread(target=monitor, daemon=True).start()
    options, inputs = pickle.load(sys.stdin.buffer)
    product = execute_local(name, options, Path(out), inputs)
    if limit and peak_rss() >= int(limit):
        raise MemoryError("Case stage exceeded RSS limit")
    with Path(reply).open("xb") as stream:
        pickle.dump(product, stream)


if __name__ == "__main__":
    main()
