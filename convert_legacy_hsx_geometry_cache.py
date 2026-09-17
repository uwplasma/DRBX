#!/usr/bin/env python3
"""Convert explicitly named legacy HSX geometry caches to an artifact.

This is a one-time producer-side migration aid.  It is intentionally not
imported by either simulation driver: consumers accept only a published
``FciSimulationGeometry3D`` directory.  The two cache files are copied into
an isolated temporary cache directory so the producer cannot discover an
unrelated cache beside either explicitly named input, and the legacy files
are never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

from drbx.geometry.hsx_simulation_geometry import (
    HsxSimulationGeometryConfig,
    build_hsx_simulation_geometry,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-metric-cache",
        type=Path,
        required=True,
        help="One existing legacy MetricEvaluator cache file.",
    )
    parser.add_argument(
        "--legacy-map-cache",
        type=Path,
        required=True,
        help="One existing legacy FCI map cache file.",
    )
    parser.add_argument("--makegrid", type=Path, required=True, help="HSX MAKEGRID input file.")
    parser.add_argument("--vessel", type=Path, required=True, help="HSX vessel input file.")
    parser.add_argument(
        "--resolution",
        nargs=3,
        type=int,
        required=True,
        metavar=("NU", "NTHETA", "NETA"),
        help="Target full-torus cell resolution.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Published artifact directory.")
    parser.add_argument("--fit-sample-shape", nargs=3, type=int, default=(8, 9, 8), metavar=("NR", "NPHI", "NETA"))
    parser.add_argument("--metric-mesh-shape", nargs=3, type=int, required=True, metavar=("NU", "NTHETA", "NETA_PERIOD"))
    parser.add_argument("--makegrid-currents", type=lambda value: tuple(float(part) for part in value.split(",")), default=None)
    parser.add_argument("--rebuild-metric-cache", action="store_true", help="Refit if the named legacy cache is not compatible.")
    parser.add_argument("--trace-tolerance-cells", type=float, default=1.0e-3)
    parser.add_argument("--status", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    return parser


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} must be an existing file: {path}")
    return resolved


def convert_legacy_cache(
    *,
    legacy_metric_cache: Path,
    legacy_map_cache: Path,
    makegrid: Path,
    vessel: Path,
    resolution: tuple[int, int, int],
    output: Path,
    fit_sample_shape: tuple[int, int, int] = (8, 9, 8),
    metric_mesh_shape: tuple[int, int, int] | None = None,
    makegrid_currents: tuple[float, ...] | None = None,
    rebuild_metric_cache: bool = False,
    trace_tolerance_cells: float = 1.0e-3,
    status: Path | None = None,
    log: Path | None = None,
):
    """Convert two explicitly named caches by invoking the producer once.

    The temporary cache directory contains only copies of the supplied metric
    and map files.  This preserves the producer's normal cache reader while
    making cache selection deterministic and preventing writes to legacy
    inputs.  Any missing or incompatible arrays are rebuilt by the producer
    and must still pass producer qualification before publication.
    """

    metric_source = _require_file(legacy_metric_cache, "legacy metric cache")
    map_source = _require_file(legacy_map_cache, "legacy map cache")
    makegrid_path = _require_file(makegrid, "MAKEGRID input")
    vessel_path = _require_file(vessel, "vessel input")
    if metric_mesh_shape is None:
        raise ValueError(
            "metric_mesh_shape must explicitly specify "
            "(NU, NTHETA, NETA_PER_PERIOD)"
        )
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{target.name}.legacy-convert-", dir=target.parent) as workdir:
        cache_dir = Path(workdir) / "metric-cache"
        cache_dir.mkdir()
        # The producer's legacy reader intentionally recognizes only its
        # historical cache prefix.  Give the explicitly selected file that
        # prefix inside the isolated directory instead of depending on the
        # user's original filename.
        staged_metric = cache_dir / "hsx_metric_legacy_input.npz"
        staged_map = Path(workdir) / map_source.name
        shutil.copy2(metric_source, staged_metric)
        shutil.copy2(map_source, staged_map)
        config = HsxSimulationGeometryConfig(
            makegrid_path=makegrid_path,
            vessel_path=vessel_path,
            resolution=tuple(int(value) for value in resolution),
            fit_sample_shape=tuple(int(value) for value in fit_sample_shape),
            metric_mesh_shape=tuple(int(value) for value in metric_mesh_shape),
            makegrid_currents=makegrid_currents,
            output=target,
            metric_cache_dir=cache_dir,
            map_cache_path=staged_map,
            rebuild_metric_cache=bool(rebuild_metric_cache),
            trace_tolerance_cells=float(trace_tolerance_cells),
        )
        return build_hsx_simulation_geometry(config, status_path=status, log_path=log)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    convert_legacy_cache(
        legacy_metric_cache=args.legacy_metric_cache,
        legacy_map_cache=args.legacy_map_cache,
        makegrid=args.makegrid,
        vessel=args.vessel,
        resolution=tuple(args.resolution),
        output=args.output,
        fit_sample_shape=tuple(args.fit_sample_shape),
        metric_mesh_shape=tuple(args.metric_mesh_shape),
        makegrid_currents=args.makegrid_currents,
        rebuild_metric_cache=args.rebuild_metric_cache,
        trace_tolerance_cells=args.trace_tolerance_cells,
        status=args.status,
        log=args.log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
