"""Dense HSX metric checkpoint from hsx_fci_builder.py:1920–2652."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import jax.numpy as jnp
import numpy as np

from .metric_evaluator import MetricEvaluator, require_metric_construction
from .._metric_contract import (
    BASE_MESH_SHAPE, BASE_POLOIDAL_MODES, BASE_PROJECTIONS, BASE_RADIAL_DEGREE, BASE_TOROIDAL_MODES,
    spline_cache_settings,
)

from ._dense_geometry import BFieldGeometry, MetricGeometry
from ._hsx import (
    METRIC_BUILDER_REVISION,
    METRIC_CACHE_FORMAT_VERSION,
    METRIC_CACHE_IDENTITY_VERSION,
    METRIC_FIELDS,
    _metric_input_content_identity,
    _mode_cutoff,
    _repeat_field_period_cells,
    _repeat_field_period_eta_faces,
    _rotate_field_period_positions,
    _write_npz_atomic,
    topology_descriptor,
)
from .metric_fitting import build_metric_evaluator
from ..options import MetricOptions


def build_metric_cache(options: MetricOptions) -> Path:
    """Return a validated or newly written dense, full-torus metric cache."""
    if options.vessel_path is None or options.metric_cache_dir is None:
        raise ValueError("vessel_path and metric_cache_dir are required for a metric cache")
    makegrid_path = options.makegrid_path
    vessel_path = options.vessel_path
    metric_cache_dir = options.metric_cache_dir
    makegrid_currents = options.makegrid_currents
    resolution = options.resolution
    reference_magnetic_field = options.reference_magnetic_field
    rebuild_metric_cache = options.rebuild_metric_cache
    descriptor = topology_descriptor("toroidal")
    topology = descriptor.name
    nu, nv, neta = (int(value) for value in resolution)
    if nu < 3 or nv < 3 or neta < 4:
        raise ValueError("resolution must satisfy NU >= 3, NV >= 3, NETA >= 4")
    if topology == "toroidal" and nv % 2:
        raise ValueError("toroidal global NTHETA must be even for axis half-turn parity")

    makegrid_path = makegrid_path.resolve()
    vessel_path = vessel_path.resolve()
    if makegrid_currents is None:
        makegrid_current_array = None
    else:
        makegrid_current_array = np.asarray(
            makegrid_currents, dtype=np.float64
        ).reshape(-1)
        if (
            makegrid_current_array.size == 0
            or not np.all(np.isfinite(makegrid_current_array))
        ):
            raise ValueError("makegrid_currents must contain finite values")
    cache_spec = {
        "format_version": METRIC_CACHE_FORMAT_VERSION,
        "identity_version": METRIC_CACHE_IDENTITY_VERSION,
        "metric_builder_revision": METRIC_BUILDER_REVISION,
        "makegrid": _metric_input_content_identity(makegrid_path),
        "makegrid_currents": (
            None
            if makegrid_current_array is None
            else makegrid_current_array.tolist()
        ),
        "vessel": _metric_input_content_identity(vessel_path),
        "resolution": [nu, nv, neta],
        "topology": descriptor.name,
        "coordinate_names": list(descriptor.coordinate_names),
        "periodic_axes": list(descriptor.periodic_axes),
        "axis_regular_axes": list(descriptor.axis_regular_axes),
        "logical_extents": [list(extent) for extent in descriptor.logical_extents],
        "toroidal_domain": "full_2pi",
        # Serialized placeholders keep the exact-cache key of the pinned cold run.
        "fit_sample_shape": [8, 9, 8],
        "radial_degree": 3,
        "vertical_degree": 3,
        "toroidal_modes": 2,
        "metric_spline_degree": 1,
        "mmpde_iterations": 0,
        "axis_core_radius": 0.03,
        "metric_mesh_shape": list(BASE_MESH_SHAPE),
        "metric_radial_degree": BASE_RADIAL_DEGREE,
        "metric_poloidal_modes": BASE_POLOIDAL_MODES,
        "metric_toroidal_modes": BASE_TOROIDAL_MODES,
        "eta_projection_iterations": BASE_PROJECTIONS,
        **spline_cache_settings(),
        "reference_magnetic_field": (
            None
            if reference_magnetic_field is None
            else float(reference_magnetic_field)
        ),
    }
    filtered_identity = options.filtered_map_identity()
    cache_spec.update(filtered_identity)
    cache_spec_json = json.dumps(
        cache_spec,
        sort_keys=True,
        separators=(",", ":"),
    )
    cache_key = hashlib.sha256(cache_spec_json.encode()).hexdigest()[:24]
    metric_cache_dir = metric_cache_dir.resolve()
    metric_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = metric_cache_dir / f"hsx_metric_{cache_key}.npz"
    locations = ("cell", "x_face", "y_face", "z_face")
    expected_shapes = {
        "cell": (nu, nv, neta),
        "x_face": (nu + 1, nv, neta),
        "y_face": (nu, nv + 1, neta),
        "z_face": (nu, nv, neta + 1),
    }
    if cache_path.is_file() and not rebuild_metric_cache:
        try:
            cache_load_start = time.perf_counter()
            print(
                f"[metric-cache] loading {cache_path}",
                flush=True,
            )
            with np.load(cache_path, allow_pickle=False) as cached:
                cache_payload = {
                    name: np.array(cached[name], copy=True)
                    for name in cached.files
                }
            if (
                int(cache_payload["format_version"].item())
                != METRIC_CACHE_FORMAT_VERSION
            ):
                raise ValueError("cache format version mismatch")
            cached_spec = json.loads(str(cache_payload["cache_spec"].item()))
            if cached_spec != cache_spec:
                raise ValueError("cache specification mismatch")
            cached_metadata = {
                "topology": str(cache_payload["topology"].item()),
                "coordinate_names": tuple(
                    json.loads(str(cache_payload["coordinate_names_json"].item()))
                ),
                "periodic_axes": tuple(
                    bool(value) for value in cache_payload["periodic_axes"]
                ),
                "axis_regular_axes": tuple(
                    bool(value) for value in cache_payload["axis_regular_axes"]
                ),
                "logical_extents": tuple(
                    tuple(float(value) for value in extent)
                    for extent in np.asarray(cache_payload["logical_extents"])
                ),
            }
            if cached_metadata != {
                "topology": descriptor.name,
                "coordinate_names": descriptor.coordinate_names,
                "periodic_axes": descriptor.periodic_axes,
                "axis_regular_axes": descriptor.axis_regular_axes,
                "logical_extents": descriptor.logical_extents,
            }:
                raise ValueError("cached topology descriptor is inconsistent")
            for name, expected_shape in (
                ("u_faces", (nu + 1,)),
                ("v_faces", (nv + 1,)),
                ("eta_faces", (neta + 1,)),
                ("cell_position", (nu, nv, neta, 3)),
            ):
                if cache_payload[name].shape != expected_shape:
                    raise ValueError(
                        f"cached {name} has shape {cache_payload[name].shape}, "
                        f"expected {expected_shape}"
                    )
            cached_nfp = int(cache_payload["nfp"].item())
            if cached_nfp < 1 or neta % cached_nfp:
                raise ValueError("cached nfp is incompatible with full-torus NETA")
            require_metric_construction(cache_payload, options)
            cached_metric_evaluator = MetricEvaluator.from_cache_payload(
                cache_payload,
                prefix="metric_evaluator_",
            )
            if cached_metric_evaluator.nfp != cached_nfp:
                raise ValueError("cached evaluator nfp is inconsistent")
            if cached_metric_evaluator.topology != topology:
                raise ValueError("cached evaluator topology is inconsistent")
            if not np.isclose(cached_metric_evaluator.u[0], 0.0) or not np.isclose(
                cached_metric_evaluator.u[-1], 1.0
            ):
                raise ValueError("cached evaluator u extent is inconsistent")
            if not np.isclose(cached_metric_evaluator.v[0], 0.0):
                raise ValueError("cached evaluator theta origin is inconsistent")
            if not np.isclose(
                cached_metric_evaluator.v.size * np.diff(cached_metric_evaluator.v)[0],
                2.0 * np.pi,
            ):
                raise ValueError("cached evaluator theta extent is inconsistent")
            if cached_metric_evaluator.radial_degree != BASE_RADIAL_DEGREE:
                raise ValueError("cached evaluator radial degree is inconsistent")
            if _mode_cutoff(cached_metric_evaluator.poloidal_modes) != BASE_POLOIDAL_MODES:
                raise ValueError("cached evaluator poloidal modes are inconsistent")
            if _mode_cutoff(cached_metric_evaluator.toroidal_modes) != cache_spec["metric_toroidal_modes"]:
                raise ValueError("cached evaluator toroidal modes are inconsistent")
            for location in locations:
                expected_shape = expected_shapes[location]
                for field in METRIC_FIELDS:
                    name = f"{location}_metric_{field}"
                    if cache_payload[name].shape != expected_shape:
                        raise ValueError(
                            f"cached {name} has shape "
                            f"{cache_payload[name].shape}, expected "
                            f"{expected_shape}"
                        )
                if cache_payload[f"{location}_Bmag"].shape != expected_shape:
                    raise ValueError(f"cached {location} Bmag shape mismatch")
                if (
                    cache_payload[f"{location}_B_contra"].shape
                    != expected_shape + (3,)
                ):
                    raise ValueError(
                        f"cached {location} B_contra shape mismatch"
                    )
            eta_faces = np.asarray(cache_payload["eta_faces"], dtype=np.float64)
            reference_magnetic_field = float(cache_payload["reference_magnetic_field"].item())
            if not np.isclose(
                eta_faces[-1] - eta_faces[0],
                2.0 * np.pi,
                rtol=2.0e-10,
                atol=2.0e-12,
            ):
                raise ValueError("cached eta axis does not span the full 2π torus")
            print(
                f"[metric-cache] loaded in "
                f"{time.perf_counter() - cache_load_start:.3f} s",
                flush=True,
            )
            return cache_path
        except (
            EOFError,
            KeyError,
            OSError,
            ValueError,
            zipfile.BadZipFile,
        ) as error:
            print(
                f"[metric-cache] ignored ({error}); rebuilding {cache_path}",
                flush=True,
            )

    print(f"[metric-cache] miss: {cache_path}", flush=True)
    metric_evaluator, _, bfield, nfp = build_metric_evaluator(
        makegrid_path=makegrid_path, vessel_path=vessel_path,
        resolution=(nu, nv, neta), makegrid_currents=makegrid_current_array,
    )
    if filtered_identity:
        from .map_filter import filter_solved_map

        metric_evaluator, sample_fit_error = filter_solved_map(metric_evaluator, options.field_filter)
    nfp = int(nfp)
    if nfp < 1:
        raise ValueError(f"HSX nfp must be a positive integer; got {nfp}")
    if neta % nfp:
        raise ValueError(
            f"full-torus NETA={neta} must be divisible by HSX nfp={nfp}"
        )
    neta_per_period = neta // nfp

    # The full-torus PDE grid is independent of the Fourier–Zernike base grid.
    u_faces = np.linspace(0.0, 1.0, nu + 1, dtype=np.float64)
    v_faces = (
        np.linspace(0.0, 2.0 * np.pi, nv + 1, dtype=np.float64)
    )
    eta_faces_period = (
        float(metric_evaluator.eta[0])
        + np.arange(neta_per_period + 1, dtype=np.float64)
        * float(metric_evaluator.period)
        / float(neta_per_period)
    )
    u_centers = 0.5 * (u_faces[:-1] + u_faces[1:])
    v_centers = 0.5 * (v_faces[:-1] + v_faces[1:])
    eta_centers_period = 0.5 * (
        eta_faces_period[:-1] + eta_faces_period[1:]
    )
    cell_points = np.stack(
        np.meshgrid(u_centers, v_centers, eta_centers_period, indexing="ij"),
        axis=-1,
    )
    positive_u_faces = u_faces[1:]
    x_face_points = np.stack(
        np.meshgrid(
            positive_u_faces, v_centers, eta_centers_period, indexing="ij"
        ),
        axis=-1,
    )
    y_face_points = np.stack(
        np.meshgrid(u_centers, v_faces, eta_centers_period, indexing="ij"),
        axis=-1,
    )
    z_face_points = np.stack(
        np.meshgrid(u_centers, v_centers, eta_faces_period, indexing="ij"),
        axis=-1,
    )
    metric_results = []
    for location, points in (
        ("cell centers", cell_points),
        ("u faces", x_face_points),
        ("v faces", y_face_points),
        ("eta faces", z_face_points),
    ):
        stage_start = time.perf_counter()
        print(
            f"[geometry] evaluating metric coefficients at {location}",
            flush=True,
        )
        metric_results.append(metric_evaluator.evaluate(points))
        print(
            f"[geometry] metric coefficients at {location} completed in "
            f"{time.perf_counter() - stage_start:.3f} s",
            flush=True,
        )
    cell_metric_eval, *metric_evaluations = metric_results
    cell_positions = _rotate_field_period_positions(
        np.asarray(cell_metric_eval.position, dtype=np.float64),
        nfp,
    )
    eta_faces = (
        float(eta_faces_period[0])
        + np.arange(neta + 1, dtype=np.float64)
        * 2.0
        * np.pi
        / float(neta)
    )
    stage_start = time.perf_counter()
    print(
        "[geometry] evaluating magnetic coefficients at cell centers",
        flush=True,
    )
    cell_b_eval = metric_evaluator.evaluate_magnetic_field(
        cell_points,
        bfield,
    )
    print(
        f"[geometry] magnetic coefficients at cell centers completed in "
        f"{time.perf_counter() - stage_start:.3f} s",
        flush=True,
    )
    b_evaluations = []
    for location, points in (
        ("u faces", x_face_points),
        ("v faces", y_face_points),
        ("eta faces", z_face_points),
    ):
        stage_start = time.perf_counter()
        print(
            f"[geometry] evaluating magnetic coefficients at {location}",
            flush=True,
        )
        b_evaluations.append(
            metric_evaluator.evaluate_magnetic_field(points, bfield)
        )
        print(
            f"[geometry] magnetic coefficients at {location} completed "
            f"in {time.perf_counter() - stage_start:.3f} s",
            flush=True,
        )

    if topology == "toroidal":
        # The u=0 face is a finite-volume collapsed-face representation,
        # not an ordinary pointwise polar metric.  Build it from the first
        # positive-u radial layer using (-u,theta,eta)=(u,theta+pi,eta)
        # and T=diag(-1,1,1).  NTHETA is even, so the half-turn is exact.
        half_turn = nv // 2
        proxy_metric = metric_evaluator.evaluate(
            np.stack(
                np.meshgrid(
                    [u_centers[0]], v_centers, eta_centers_period, indexing="ij"
                ),
                axis=-1,
            )
        )
        proxy_b = metric_evaluator.evaluate_magnetic_field(
            np.stack(
                np.meshgrid(
                    [u_centers[0]], v_centers, eta_centers_period, indexing="ij"
                ),
                axis=-1,
            ),
            bfield,
        )
        def _axis_tensor(values):
            mirrored = np.roll(np.asarray(values), half_turn, axis=1)
            signs = np.asarray((-1.0, 1.0, 1.0))
            transformed = mirrored * signs.reshape(1, 1, 1, 3, 1)
            transformed = transformed * signs.reshape(1, 1, 1, 1, 3)
            return 0.5 * (np.asarray(values) + transformed)


        axis_metric = {
            "signed_J": np.zeros((1, nv, neta_per_period), dtype=np.float64),
            "g_contra": _axis_tensor(proxy_metric.g_contra),
            "g_cov": _axis_tensor(proxy_metric.g_cov),
        }
        values = proxy_b.B_contravariant
        mirrored = np.roll(np.asarray(values), half_turn, axis=1)
        axis_b = {
            "B_contravariant": 0.5 * (
                np.asarray(values)
                + mirrored * np.asarray((-1.0, 1.0, 1.0)).reshape(1, 1, 1, 3)
            ),
            "magnitude": np.mean(
                np.stack(
                    (np.asarray(proxy_b.magnitude),
                     np.roll(np.asarray(proxy_b.magnitude), half_turn, axis=1)),
                    axis=0,
                ),
                axis=0,
            ),
        }
        if any(
            not np.all(np.isfinite(values))
            for values in (*axis_metric.values(), *axis_b.values())
        ):
            raise ValueError("collapsed toroidal axis-face representation is not finite")
        metric_evaluations[0] = SimpleNamespace(
            **{
                key: np.concatenate((axis_metric[key], np.asarray(value)), axis=0)
                for key, value in {
                "signed_J": metric_evaluations[0].signed_J,
                "g_contra": metric_evaluations[0].g_contra,
                "g_cov": metric_evaluations[0].g_cov,
                }.items()
            }
        )
        b_evaluations[0] = SimpleNamespace(
            **{
                key: np.concatenate((axis_b[key], np.asarray(value)), axis=0)
                for key, value in {
                "B_contravariant": b_evaluations[0].B_contravariant,
                "magnitude": b_evaluations[0].magnitude,
                }.items()
            }
        )

    if reference_magnetic_field is None:
        reference_magnetic_field = float(
            np.median(
                np.asarray(cell_b_eval.magnitude, dtype=np.float64)
            )
        )
    reference_magnetic_field = float(reference_magnetic_field)
    if (
        not np.isfinite(reference_magnetic_field)
        or reference_magnetic_field <= 0.0
    ):
        raise ValueError(
            "reference magnetic field must be positive and finite"
        )

    metric_geometries = []
    for location, evaluation in zip(
        locations,
        (cell_metric_eval, *metric_evaluations),
    ):
        repeat_periods = (
            _repeat_field_period_eta_faces
            if location == "z_face"
            else _repeat_field_period_cells
        )
        signed_J = repeat_periods(
            np.asarray(evaluation.signed_J, dtype=np.float64),
            nfp,
        )
        g_contra = repeat_periods(
            np.asarray(evaluation.g_contra, dtype=np.float64),
            nfp,
        )
        g_cov = repeat_periods(
            np.asarray(evaluation.g_cov, dtype=np.float64),
            nfp,
        )
        metric_geometries.append(
            MetricGeometry(
                J=jnp.asarray(signed_J),
                g11=jnp.asarray(g_contra[..., 0, 0]),
                g22=jnp.asarray(g_contra[..., 1, 1]),
                g33=jnp.asarray(g_contra[..., 2, 2]),
                g12=jnp.asarray(g_contra[..., 0, 1]),
                g13=jnp.asarray(g_contra[..., 0, 2]),
                g23=jnp.asarray(g_contra[..., 1, 2]),
                g_11=jnp.asarray(g_cov[..., 0, 0]),
                g_22=jnp.asarray(g_cov[..., 1, 1]),
                g_33=jnp.asarray(g_cov[..., 2, 2]),
                g_12=jnp.asarray(g_cov[..., 0, 1]),
                g_13=jnp.asarray(g_cov[..., 0, 2]),
                g_23=jnp.asarray(g_cov[..., 1, 2]),
            )
        )
    bfield_geometries = []
    for location, evaluation in zip(
        locations,
        (cell_b_eval, *b_evaluations),
    ):
        repeat_periods = (
            _repeat_field_period_eta_faces
            if location == "z_face"
            else _repeat_field_period_cells
        )
        bfield_geometries.append(
            BFieldGeometry(
                B_contra=(
                    jnp.asarray(
                        repeat_periods(
                            np.asarray(
                                evaluation.B_contravariant,
                                dtype=np.float64,
                            ),
                            nfp,
                        )
                    )
                    / reference_magnetic_field
                ),
                Bmag=(
                    jnp.asarray(
                        repeat_periods(
                            np.asarray(
                                evaluation.magnitude,
                                dtype=np.float64,
                            ),
                            nfp,
                        )
                    )
                    / reference_magnetic_field
                ),
            )
        )
    bfield_geometries = tuple(bfield_geometries)

    cache_write_start = time.perf_counter()
    print(
        f"[metric-cache] serializing evaluated geometry to "
        f"{cache_path}",
        flush=True,
    )
    cache_payload = {
        "format_version": np.asarray(
            METRIC_CACHE_FORMAT_VERSION,
            dtype=np.int64,
        ),
        "cache_spec": np.asarray(cache_spec_json),
        "topology": np.asarray(descriptor.name),
        "coordinate_names_json": np.asarray(
            json.dumps(descriptor.coordinate_names)
        ),
        "periodic_axes": np.asarray(descriptor.periodic_axes, dtype=bool),
        "axis_regular_axes": np.asarray(
            descriptor.axis_regular_axes, dtype=bool
        ),
        "logical_extents": np.asarray(
            descriptor.logical_extents, dtype=np.float64
        ),
        "u_faces": u_faces,
        "v_faces": v_faces,
        "eta_faces": eta_faces,
        "reference_magnetic_field": np.asarray(
            reference_magnetic_field,
            dtype=np.float64,
        ),
        "nfp": np.asarray(nfp, dtype=np.int64),
        "cell_position": cell_positions,
    }
    cache_payload.update(
        metric_evaluator.to_cache_payload(
            prefix="metric_evaluator_"
        )
    )
    if filtered_identity:
        cache_payload["map_filter_fit_error_metres"] = np.asarray(json.dumps(sample_fit_error))
    for location, metric in zip(locations, metric_geometries):
        for field in METRIC_FIELDS:
            cache_payload[f"{location}_metric_{field}"] = np.asarray(
                getattr(metric, field),
                dtype=np.float64,
            )
    for location, magnetic_field in zip(
        locations,
        bfield_geometries,
    ):
        cache_payload[f"{location}_B_contra"] = np.asarray(
            magnetic_field.B_contra,
            dtype=np.float64,
        )
        cache_payload[f"{location}_Bmag"] = np.asarray(
            magnetic_field.Bmag,
            dtype=np.float64,
        )
    _write_npz_atomic(cache_path, cache_payload)
    print(
        f"[metric-cache] written in "
        f"{time.perf_counter() - cache_write_start:.3f} s "
        f"({cache_path.stat().st_size / 2**20:.1f} MiB)",
        flush=True,
    )
    return cache_path
