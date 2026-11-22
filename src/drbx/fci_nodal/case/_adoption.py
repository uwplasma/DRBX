"""Preserve source provenance and check the contracts an adopted artifact exposes."""

from pathlib import Path
import hashlib
import json

from .._hashing import sha256_file as digest
from ._receipts import file_receipts, read_json, receipt_products


def identify_adopted(adoption):
    products = {role: str(Path(path).resolve(strict=True)) for role, path in adoption.items()
                if role != "origin_receipt"}
    files = file_receipts(products.values())
    result = dict(products=products, files=files)
    if path := adoption.get("origin_receipt"):
        path = Path(path).resolve(strict=True)
        origin = read_json(path)
        if any(origin["products"].get(role) != product for role, product in products.items()):
            raise ValueError(f"Adopted products differ from origin receipt {path}")
        if any(origin["output_files"].get(name) != checksum for name, checksum in files.items()):
            raise ValueError(f"Adopted files differ from origin receipt {path}")
        result.update(receipt=str(path), receipt_sha256=digest(path), identity=origin["identity"])
    return result


def validate_adoption(name, origin, layout, inputs, metric_options):
    import numpy as np

    products = {key: Path(value) for key, value in origin["products"].items()}
    n, planes, symmetry, periods = layout
    metric = inputs.get("metric_cache")
    table = inputs.get("field_table")

    def match(label, value, expected):
        if value is not None and expected is not None and value != expected:
            raise ValueError(f"Adopted {name}: {label} {value!r} differs from {expected!r}")

    def table_meta(path):
        with np.load(path, allow_pickle=False) as archive:
            return json.loads(archive["meta"].item()) if "meta" in archive else {}

    def hashes(meta):
        if table is not None:
            expected = meta.get("field_table_sha256", meta.get("eta_table_sha256"))
            if expected is not None:
                match("field-table hash", expected, digest(table))
        if metric is not None:
            expected = meta.get("metric_cache_sha256", meta.get("map_sha256"))
            if expected is not None:
                match("metric cache hash", expected, digest(metric))

    if "receipt" in origin:
        source = read_json(origin["receipt"])
        recorded_identity = source.get("case_binding") or source["identity"]
        for stamp in recorded_identity.get("input_receipts", []):
            path = Path(stamp["receipt"])
            if path.exists() and digest(path) == stamp["sha256"]:
                predecessor = read_json(path)
                for role, product in receipt_products(predecessor).items():
                    if role in inputs and product.is_file() and inputs[role].is_file():
                        expected = predecessor["output_files"].get(str(product.resolve()))
                        match(f"{role} source hash", expected, digest(inputs[role]))

    if name == "metric":
        from .._metric_contract import (
            BASE_MESH_SHAPE, BASE_POLOIDAL_MODES, BASE_PROJECTIONS, BASE_RADIAL_DEGREE, BASE_TOROIDAL_MODES,
            spline_cache_settings,
        )
        from ..evaluators.exact_plane_metric import require_exact_metric_payload

        with np.load(products["metric_cache"], allow_pickle=False) as archive:
            require_exact_metric_payload(archive, "metric_evaluator_")
            if "cache_spec" not in archive or "nfp" not in archive:
                raise ValueError("Adopted metric: cache specification and map periods are required")
            match("map periods", int(archive["nfp"]), periods)
            spec = json.loads(archive["cache_spec"].item())
            expected = dict(
                spline_cache_settings(), resolution=list(metric_options.resolution), topology="toroidal",
                metric_mesh_shape=list(BASE_MESH_SHAPE), metric_radial_degree=BASE_RADIAL_DEGREE,
                metric_poloidal_modes=BASE_POLOIDAL_MODES, metric_toroidal_modes=BASE_TOROIDAL_MODES,
                eta_projection_iterations=BASE_PROJECTIONS,
            )
            for key, value in expected.items():
                if key not in spec:
                    raise ValueError(f"Adopted metric: cache specification lacks {key}")
                match(key, spec[key], value)
    elif name == "field_table":
        path = products["field_table"]
        with np.load(path, allow_pickle=False) as archive:
            spec = json.loads(archive["spec"].item())
        filter_spec = metric_options.field_filter
        if filter_spec is None:
            raise ValueError("Adopted field table: a raw-field case cannot use a filtered table")
        match("filter specification", spec, filter_spec.as_dict())
        meta = table_meta(path)
        hashes(meta)
        if metric is not None and meta.get("arm_sha256"):
            # Same arm inputs as the metric table writer. Legacy map projections
            # with distinct hashes need explicit compatible provenance.
            core = dict(map_sha256=digest(metric),
                        makegrid_sha256=metric_options.makegrid_sha256 or digest(metric_options.makegrid_path),
                        makegrid_currents=list(metric_options.makegrid_currents),
                        bfield_toroidal=metric_options.bfield_toroidal, eta_filter=spec,
                        table_grid=list(metric_options.table_grid))
            arm = hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()
            match("map/field arm identity", meta["arm_sha256"], arm)
    elif name == "polygon":
        record = products.get("polygon_case")
        if record is not None:
            from ..operator_build.parallel._case import CASE_SCHEMA

            metadata = read_json(record)
            if metadata.get("schema") != CASE_SCHEMA:
                raise ValueError("Unsupported polygon case schema for adoption")
            match("resolution", metadata.get("N"), n)
            match("field periods", metadata.get("field_periods"), periods)
            match("symmetry", metadata.get("symmetry"), symmetry)
            hashes(metadata)
    elif name == "geometry":
        from ..geometry_build.reader import GEOMETRY_SCHEMA

        metadata = read_json(products["indexed_geometry"] / "geometry.json")
        if metadata.get("schema") != GEOMETRY_SCHEMA:
            raise ValueError("Unsupported indexed geometry schema for adoption")
        for key, wanted in (("N", n), ("E", planes), ("symmetry", symmetry), ("field_periods", periods)):
            match(key, metadata.get(key), wanted)
        if metric_options.field_filter is not None:
            match("filter specification", metadata.get("field_filter"), metric_options.field_filter.as_dict())
        hashes(metadata)
    elif name == "parallel_operators":
        from ._geometry_binding import verify_parallel_geometry_binding

        runtime = products["parallel_construction"]
        verify_parallel_geometry_binding(runtime, inputs["indexed_geometry"])
        owner = read_json(runtime / "owners/manifest.json")
        owner_layout = owner["layout"]
        for key, wanted in (("N", n), ("eta_planes", planes), ("symmetry", symmetry),
                            ("field_periods", periods)):
            match(key, owner_layout[key], wanted)
        hashes(owner)
        return
    elif name == "perpendicular_operators":
        from ..operator_build.perp.owner_bundle import OwnerPerpendicularBundle

        owner = OwnerPerpendicularBundle(products["owner_perpendicular_bundle"])
        metadata = read_json(owner.root / "identity.json")
        if "sbp_leaves" not in owner.manifest or owner.manifest.get("source_identity") != "identity.json":
            raise ValueError("Missing direct P owner setup or source identity")
        source = metadata.get("created", {}).get("inputs", {})
        match("resolution", metadata.get("n"), n)
        match("eta planes", metadata.get("n_eta"), planes)
        match("symmetry", source.get("symmetry"), symmetry)
        match("field periods", source.get("field_periods"), periods)
        hashes(source)
        if "P" in metadata:
            from ..operators.nodal.nodal_layout_specs import build_zernike_core_ring_layout

            match("nodes per plane", metadata["P"], build_zernike_core_ring_layout(n, n_eta=planes).P)
        if metric is not None:
            match("metric cache hash", source.get("metric_cache", {}).get("sha256"), digest(metric))
        if table is not None:
            arm = metadata.get("arm_identity")
            if isinstance(arm, dict):
                arm = arm.get("arm_sha256")
            match("field-table arm identity", arm, table_meta(table).get("arm_sha256"))
