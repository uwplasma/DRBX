"""Stage receipts, immutable input checks, and current case lineage."""

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

from .._hashing import sha256_file as digest


# The first predecessor of each stage establishes execution order. Extra
# predecessors name the direct product producers consumed by that stage.
PREDECESSORS = {
    "metric": (),
    "field_table": ("metric",),
    "polygon": ("field_table", "metric"),
    "geometry": ("polygon", "metric", "field_table"),
    "parallel_operators": ("geometry", "metric", "field_table"),
    "perpendicular_operators": ("parallel_operators", "metric", "field_table"),
    "owner_inputs": (),
    "wall_data": ("owner_inputs",),
    "mms_prepare": ("owner_inputs", "wall_data"),
    "mms_run": ("owner_inputs", "wall_data", "mms_prepare"),
}
OWNER_STAGES = (
    "owner_inputs",
    "wall_data",
    "mms_prepare",
    "mms_run",
)
BACKEND_ROOT = Path(__file__).resolve().parents[1]
# Geometry executes its two kernels directly and imports the lane modules in spawned workers.
GEOMETRY_EXECUTED_SOURCES = (
    "geometry_build/dual_cells.py", "geometry_build/_volume_base.py",
    "geometry_build/_shard_jobs.py", "geometry_build/_geometry_worker.py",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def peak_rss():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
                    encoding="utf-8")


def file_receipts(products):
    files = set()
    for value in products:
        path = Path(value).resolve(strict=True)
        if path.is_dir():
            files.update(p for p in path.rglob("*") if p.is_file())
        else:
            files.add(path)
    return {str(path): digest(path) for path in sorted(files)}


def code_files(name, recorded=()):
    sources = {path.resolve() for path in (BACKEND_ROOT / "case").glob("*.py")}
    for source in set(recorded).union(GEOMETRY_EXECUTED_SOURCES if name == "geometry" else ()):
        path = (BACKEND_ROOT / source).resolve()
        if not path.is_relative_to(BACKEND_ROOT):
            raise ValueError(f"Code source is outside the backend: {source}")
        sources.add(path)
    return {path.relative_to(BACKEND_ROOT).as_posix(): digest(path) for path in sorted(sources)}


def worker_code_files(stage_name):
    """Snapshot the loaded modules and explicit sources used by the worker."""
    package = __package__.rsplit(".", 1)[0]
    sources = set()
    for name, module in tuple(sys.modules.items()):
        if name == package or name.startswith(package + "."):
            if filename := getattr(module, "__file__", None):
                path = Path(filename).resolve()
                if path.is_relative_to(BACKEND_ROOT):
                    sources.add(path.relative_to(BACKEND_ROOT).as_posix())
    return code_files(stage_name, sources)


def code_identity(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def verify_code(name, identity):
    files = identity["code_files"]
    if files != code_files(name, files) or identity["code_sha256"] != code_identity(files):
        raise ValueError("code identity changed")


def stage_receipt(case, name):
    path = case / name / "receipt.json"
    raw = path.read_bytes()
    return json.loads(raw), dict(stage=name, receipt=str(path), sha256=hashlib.sha256(raw).hexdigest())


def receipt_products(receipt):
    return {role: Path(path) for role, path in receipt["products"].items()}


def binding(receipt):
    return receipt["case_binding"] if receipt.get("adopted_from") is not None else receipt["identity"]


def stage_identity(name, options, inputs, settings, dependencies, *, sources=None):
    paths = sorted({str(Path(path).resolve()) for path in inputs if path is not None})
    options = json.loads(json.dumps(options, default=str, sort_keys=True, allow_nan=False))
    sources = code_files(name) if sources is None else sources
    return dict(options=options, input_receipts=[stamp for _, stamp in dependencies],
                input_paths=paths, input_files=file_receipts(paths), settings=settings,
                code_files=sources, code_sha256=code_identity(sources))


def verify_chain(case, last):
    """Validate the current case in stage order and name the first stale link."""
    stages = {}
    owner_path = last in OWNER_STAGES
    for name, predecessors in PREDECESSORS.items():
        if (name in OWNER_STAGES) != owner_path:
            continue
        try:
            receipt, stamp = stage_receipt(case, name)
            current = binding(receipt)
            dependencies = [stages[key] for key in predecessors]
            if receipt["stage"] != name or current["input_receipts"] != [s for _, s in dependencies]:
                raise ValueError("predecessor receipts differ")
            verify_code(name, current)
            if file_receipts(current["input_paths"]) != current["input_files"]:
                raise ValueError("input files changed")
            if file_receipts(receipt_products(receipt).values()) != receipt["output_files"]:
                raise ValueError("output files changed")
            if receipt["test_case"] != is_test_case(dependencies, receipt.get("adopted_from")):
                raise ValueError("test case lineage changed")
        except (OSError, KeyError, ValueError) as error:
            raise ValueError(f"Stale stage {name}: {error}") from error
        stages[name] = receipt, stamp
        if name == last:
            return stages
    raise ValueError(f"Unknown stage {last}")


def is_test_case(dependencies, adoption=None):
    return adoption is not None or any(receipt["test_case"] for receipt, _ in dependencies)


def stage(case, name, *, settings, options, inputs, expected, action, adopt=None,
          dependencies=(), verify_adoption=None):
    from ._adoption import identify_adopted

    directory = case / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "receipt.json"
    dependencies = tuple(dependencies)
    identity = stage_identity(name, options, inputs, settings, dependencies)
    prior = read_json(path) if path.exists() else {}
    origin = None if adopt is None else identify_adopted(adopt)
    previous_origin = prior.get("adopted_from")
    if origin is None:
        origin = previous_origin

    start = time.monotonic()
    if (prior and previous_origin == origin and prior.get("test_case") == is_test_case(dependencies, origin)):
        try:
            previous = binding(prior)
            verify_code(name, previous)
            expected_identity = dict(identity, code_files=previous["code_files"], code_sha256=previous["code_sha256"])
            if (set(expected).issubset(prior["products"])
                    and previous == expected_identity
                    and prior["output_files"] == file_receipts(receipt_products(prior).values())):
                return stage_receipt(case, name)
        except (OSError, ValueError, KeyError):
            pass
    if origin is not None and adopt is None:
        raise ValueError(f"Stale stage {name}: adoption changed. Supply explicit adoption again.")
    if origin is not None:
        verify_adoption(origin)

    number = 1
    while (attempt := directory / f"attempt-{number:04d}").exists():
        number += 1
    attempt.mkdir()
    if origin is None:
        products, rays, rss, sources = action(attempt)
        identity.update(code_files=sources, code_sha256=code_identity(sources))
        verify_code(name, identity)
    else:
        products, rays, rss = origin["products"], 0, peak_rss()
    if not set(expected).issubset(products):
        raise ValueError(f"{name}: missing products {sorted(set(expected) - set(products))}")
    products = {role: str(Path(value).resolve(strict=True)) for role, value in products.items()}
    receipt = dict(stage=name, identity=identity if origin is None else origin.get("identity"), adopted_from=origin,
                   test_case=is_test_case(dependencies, origin), products=products,
                   output_files=file_receipts(products.values()), rays_traced=rays,
                   elapsed_seconds=time.monotonic() - start, peak_rss_bytes=rss,
                   setting_scope="no producer called" if origin is not None else "stage worker")
    if origin is not None:
        receipt["case_binding"] = identity
    pending = attempt / "receipt.pending.json"
    write_json(pending, receipt)
    pending.replace(path)
    return stage_receipt(case, name)
