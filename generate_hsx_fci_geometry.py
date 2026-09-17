#!/usr/bin/env python3
"""Generate a resumable HSX FCI simulation-geometry artifact.

This command only constructs geometry.  It never launches a simulation run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from drbx.geometry.hsx_simulation_geometry import (
    HsxSimulationGeometryConfig,
    build_hsx_simulation_geometry,
)


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", nargs=3, type=int, required=True, metavar=("NU", "NTHETA", "NETA"), help="Toroidal cell counts.")
    parser.add_argument("--output", type=Path, required=True, help="Output geometry artifact path.")
    parser.add_argument("--makegrid", type=Path, default=root / "mgrid_res2p5cm_180pln.nc")
    parser.add_argument("--vessel", type=Path, default=root / "vessel_hsx_flare.txt")
    parser.add_argument("--makegrid-currents", type=lambda value: tuple(float(part) for part in value.split(",")), default=None)
    parser.add_argument("--fit-sample-shape", nargs=3, type=int, default=(8, 9, 8), metavar=("NR", "NPHI", "NZ"))
    parser.add_argument("--radial-degree", type=int, default=3)
    parser.add_argument("--vertical-degree", type=int, default=3)
    parser.add_argument("--toroidal-modes", type=int, default=2)
    parser.add_argument("--metric-spline-degree", type=int, default=1)
    parser.add_argument("--mmpde-iterations", type=int, default=0)
    parser.add_argument("--metric-mesh-shape", nargs=3, type=int, required=True, metavar=("NU", "NTHETA", "NETA_PERIOD"))
    parser.add_argument("--metric-radial-degree", type=int, default=17)
    parser.add_argument("--metric-poloidal-modes", type=int, default=15)
    parser.add_argument("--metric-toroidal-modes", type=int, default=16)
    parser.add_argument("--eta-projection-iterations", type=int, default=0)
    parser.add_argument("--axis-core-radius", type=float, default=0.03)
    parser.add_argument("--reference-magnetic-field", type=float, default=None)
    parser.add_argument("--include-curvature-edge-one-form", action="store_true")
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument("--map-cache", type=Path, default=None, help="Optional independent FCI map cache/checkpoint path.")
    parser.add_argument("--status", type=Path, default=None, help="Resumable status JSON path.")
    parser.add_argument("--log", type=Path, default=None, help="Append-only producer log path.")
    parser.add_argument("--rebuild-metric-cache", action="store_true")
    parser.add_argument("--trace-tolerance-cells", type=float, default=1.0e-3, help="64-vs-128 transverse endpoint tolerance in cell units.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = HsxSimulationGeometryConfig(
        makegrid_path=args.makegrid,
        vessel_path=args.vessel,
        resolution=tuple(args.resolution),
        fit_sample_shape=tuple(args.fit_sample_shape),
        radial_degree=args.radial_degree,
        vertical_degree=args.vertical_degree,
        toroidal_modes=args.toroidal_modes,
        metric_spline_degree=args.metric_spline_degree,
        mmpde_iterations=args.mmpde_iterations,
        metric_mesh_shape=tuple(args.metric_mesh_shape),
        metric_radial_degree=args.metric_radial_degree,
        metric_poloidal_modes=args.metric_poloidal_modes,
        metric_toroidal_modes=args.metric_toroidal_modes,
        eta_projection_iterations=args.eta_projection_iterations,
        axis_core_radius=args.axis_core_radius,
        reference_magnetic_field=args.reference_magnetic_field,
        makegrid_currents=args.makegrid_currents,
        metric_cache_dir=args.metric_cache_dir,
        map_cache_path=args.map_cache,
        output=args.output,
        rebuild_metric_cache=args.rebuild_metric_cache,
        include_curvature_edge_one_form=args.include_curvature_edge_one_form,
        trace_tolerance_cells=args.trace_tolerance_cells,
    )
    build_hsx_simulation_geometry(config, status_path=args.status, log_path=args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
