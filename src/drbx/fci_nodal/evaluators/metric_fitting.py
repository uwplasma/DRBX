"""Metric fitting entry point from hsx_fci_builder.py:1459–1732."""

from __future__ import annotations

from pathlib import Path
import time

from .magnetic_field_evaluator import bfield_evaluator_from_makegrid
from .metric_evaluator import build_metric_evaluator as build_base_metric
from .exact_plane_metric import ExactPlaneMetricEvaluator
from .._metric_contract import (
    BASE_MESH_SHAPE, BASE_POLOIDAL_MODES, BASE_PROJECTIONS, BASE_RADIAL_DEGREE, BASE_TOROIDAL_MODES,
    SPLINE_COARSE_FIT, SPLINE_FIT,
)
from .spline_scalar_potential import spline_scalar_potential_evaluator_from_bfield

from ._wall import WallEvaluator


def build_metric_evaluator(
    *,
    makegrid_path: Path,
    vessel_path: Path,
    resolution: tuple[int, int, int],
    makegrid_currents: object | None = None,
) -> tuple[ExactPlaneMetricEvaluator, object, object, int]:
    """Fit the spline eta, construct its Fourier–Zernike base and wrap exact planes."""
    nu, nv, neta = (int(value) for value in resolution)
    if nu < 3 or nv < 3 or neta < 4:
        raise ValueError("resolution must satisfy NU >= 3, NV >= 3, NETA >= 4")
    makegrid_path = Path(makegrid_path).resolve()
    vessel_path = Path(vessel_path).resolve()

    def load_bfield():
        stage_start = time.perf_counter()
        print(
            f"[geometry] loading MAKEGRID magnetic field ({makegrid_path})",
            flush=True,
        )
        loaded = bfield_evaluator_from_makegrid(
            makegrid_path,
            currents=makegrid_currents,
            method="cubic",
        )
        print(
            f"[geometry] MAKEGRID loaded in "
            f"{time.perf_counter() - stage_start:.3f} s",
            flush=True,
        )
        return loaded

    # The fit releases MAKEGRID during the solve and reloads it for metric work.
    stage_start = time.perf_counter()
    print(f"[geometry] loading vessel ({vessel_path})", flush=True)
    wall = WallEvaluator.from_file(vessel_path)
    print(
        f"[geometry] vessel loaded in "
        f"{time.perf_counter() - stage_start:.3f} s",
        flush=True,
    )
    nfp = int(wall.nfp)
    if neta % nfp:
        raise ValueError(
            f"full-torus NETA={neta} must be divisible by HSX nfp={nfp}"
        )
    neta_per_period = neta // nfp
    if neta_per_period < 2:
        raise ValueError(
            "full-torus resolution must provide at least two eta cells "
            f"per field period; got NETA={neta}, nfp={nfp}"
        )

    stage_start = time.perf_counter()
    eta_evaluator = spline_scalar_potential_evaluator_from_bfield(
        load_bfield, wall, **SPLINE_FIT, coarse=dict(SPLINE_COARSE_FIT),
        log=lambda message: print(message, flush=True),
    )
    print(f"[geometry] spline fit completed in {time.perf_counter() - stage_start:.3f} s", flush=True)
    bfield = load_bfield()
    if wall.nfp != bfield.nfp:
        raise ValueError(
            "MAKEGRID and vessel field-period counts disagree: "
            f"{bfield.nfp} != {wall.nfp}"
        )
    stage_start = time.perf_counter()
    metric_evaluator = build_base_metric(
        eta_evaluator, topology="toroidal", wall_evaluator=wall, mesh_shape=BASE_MESH_SHAPE,
        radial_degree=BASE_RADIAL_DEGREE, poloidal_modes=BASE_POLOIDAL_MODES,
        toroidal_modes=BASE_TOROIDAL_MODES, projection_iterations=BASE_PROJECTIONS,
    )
    print(f"[geometry] Fourier–Zernike base built in {time.perf_counter() - stage_start:.3f} s", flush=True)
    stage_start = time.perf_counter()
    print(
        "[geometry] constructing exact level-set planes of the spline eta",
        flush=True,
    )
    metric_evaluator = ExactPlaneMetricEvaluator(metric_evaluator, eta_evaluator)
    metric_evaluator.validate()
    print(
        f"[geometry] exact level-set planes constructed in "
        f"{time.perf_counter() - stage_start:.3f} s",
        flush=True,
    )
    return metric_evaluator, eta_evaluator, bfield, nfp
