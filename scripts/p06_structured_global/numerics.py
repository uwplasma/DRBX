#!/usr/bin/env python3
"""Complete-owner curvature algebra around the shared structured reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp


HERE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("HSX_DEPLOYMENT_ROOT", HERE.parents[1])).resolve()
ROOT = Path(os.environ.get("HSX_SOURCE_ROOT", WORKSPACE / "DRBX")).resolve()
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, ROOT / "src", SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


from p07_diffusion_global import numerics as portable  # noqa: E402

from drbx.native.fci_curvature_production_flux import curvature_principal_matrix  # noqa: E402
from drbx.native.fci_operators import _curvature_bc_characteristic_wall_states  # noqa: E402


SCHEMA = "drbx.p06-structured-global-case-v1"
PREPARE_SCHEMA = "drbx.p06-structured-global-prepare-v1"
CHUNK_SCHEMA = "drbx.p06-structured-global-chunk-v1"
PREFLIGHT_SCHEMA = "drbx.p06-structured-global-preflight-v1"
SUMMARY_SCHEMA = "drbx.p06-structured-global-summary-v1"
FIELD_NAMES = ("corrected_frozen_mms", "regular_chart_heldout", "homogeneous_dirichlet", "variable_dirichlet")
EQUATIONS = ("density", "Te", "Ti", "vorticity")
TERMS = ("material", "remainder", "total")
ACTIONS = ("centered", "U")
DIRECTIONS = ("u", "theta", "eta")
TAU = 1.0
FLOOR = 1.0e-12
PRIMARY_ACTION = {
    "density": "U",
    "Te": "U",
    "Ti": "U",
    "vorticity": "centered",
}


def _load_context(geometry_root: Path, baseline_root: Path, n: int) -> Any:
    """Load only tracked portable geometry plus the frozen omega observation."""
    faces, centers, raw_owner, raw_volume, volume, geometry = portable.build_context(int(n), Path(geometry_root))
    with np.load(Path(baseline_root) / f"N{n}.reference.npz", allow_pickle=False) as source:
        raw_omega = np.asarray(source["actual_vorticity"], dtype=np.float64).reshape(-1)
    omega = np.bincount(raw_owner, weights=raw_volume * raw_omega, minlength=len(volume)) / volume
    return SimpleNamespace(
        resolution=int(n), raw_owner=np.asarray(raw_owner), geometry=geometry,
        arrays={"raw_volume":np.asarray(raw_volume), "owner_values":np.stack((np.zeros_like(omega), omega))},
        x_centers=centers[0], y_centers=centers[1], z_centers=centers[2],
        x_faces=faces[0], y_faces=faces[1], z_faces=faces[2],
        eta_period=float(faces[2][-1] - faces[2][0]),
    )


def _load_resolution(geometry_root: Path, baseline_root: Path, n: int) -> Any:
    del baseline_root
    faces, centers, raw_owner, raw_volume, volume, geometry = portable.build_context(int(n), Path(geometry_root))
    del faces, centers, raw_volume
    keys = np.stack(np.unravel_index(geometry.owner_flat_ids, (int(n),)*3), axis=-1)
    masks = portable.masks(geometry)
    masks["axis_core"] = keys[:,0] == 0
    masks["first_ring"] = keys[:,0] == 1
    masks["physical_wall"] = masks["boundary"]
    masks["aggregate_interface"] = masks["transition"]
    theta = np.zeros((n,n,n),bool); theta[:,(0,n-1),:]=True
    eta = np.zeros((n,n,n),bool); eta[:,:,(0,n-1)]=True
    masks["theta_seam"] = np.bincount(raw_owner,weights=theta.ravel(),minlength=len(volume))>0
    masks["eta_seam"] = np.bincount(raw_owner,weights=eta.ravel(),minlength=len(volume))>0
    return SimpleNamespace(resolution=int(n), owner_keys=keys, owner_volume=volume,
                           raw_owner=raw_owner, masks=masks,
                           artifact_path=Path(geometry_root)/f"{n}x{n}x{n}")


def _compact_statistics(actual: np.ndarray, exact: np.ndarray, data: Any) -> dict[str, Any]:
    weight=np.asarray(data.owner_volume); error=np.asarray(actual)-np.asarray(exact)
    numerator=float(np.sum(weight*error*error)); volume=float(np.sum(weight))
    result={"absolute_l2":math.sqrt(numerator/volume),
            "relative_l2":math.sqrt(numerator/max(float(np.sum(weight*np.asarray(exact)**2)),1e-300)),
            "maximum_absolute_error":float(np.max(np.abs(error))),
            "signed_volume_weighted_mean_error":float(np.sum(weight*error)/volume),
            "owner_count":len(weight),"regions":{}}
    for label,mask in data.masks.items():
        local=float(np.sum(weight[mask]*error[mask]**2)); local_volume=float(np.sum(weight[mask]))
        result["regions"][label]={"absolute_l2":math.sqrt(local/local_volume) if local_volume>0 else None,
            "maximum_absolute_error":float(np.max(np.abs(error[mask]))) if np.any(mask) else None,
            "squared_error_fraction":local/max(numerator,1e-300),
            "volume_fraction":local_volume/volume,"owner_count":int(np.count_nonzero(mask))}
    return result


base = SimpleNamespace(
    _raw_keys=lambda n, indices: np.stack(np.unravel_index(np.asarray(indices,dtype=np.int64),(n,n,n)),axis=-1),
    _face_keys=portable.face_keys,
    _face_index=lambda n,axis,i,j,k: portable.face_indices(n,np.column_stack((np.full(len(i),axis),i,j,k))),
    _face_count=portable.face_count,
    _cell_quadrature=lambda context,keys: portable.quadrature((context.x_faces,context.y_faces,context.z_faces),keys,3,face=False),
    _face_quadrature=lambda context,keys: portable.quadrature((context.x_faces,context.y_faces,context.z_faces),keys,3,face=True),
)
cubic = SimpleNamespace(_load_context=_load_context)
integrated = SimpleNamespace(_reference=portable.reference,_load_resolution=_load_resolution,
                             _compact_statistics=_compact_statistics)


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1024.0**3


def _current_rss_gib() -> float | None:
    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024.0**3
        except (OSError, ValueError, IndexError):
            pass
    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.Process().memory_info().rss) / 1024.0**3
    except (ImportError, OSError):
        return None


def _metric_cache_diagnostics(reference: Any) -> dict[str, Any] | None:
    evaluator = getattr(reference, "metric_evaluator", None)
    if evaluator is None or not hasattr(evaluator, "basis_cache_diagnostics"):
        return None
    return evaluator.basis_cache_diagnostics()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            metadata_json=np.asarray(json.dumps(_json(metadata), sort_keys=True)),
        )
    temporary.replace(path)


def _load_npz(path: Path, schema: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        arrays = {name: np.asarray(source[name]) for name in source.files if name != "metadata_json"}
    if metadata.get("schema") != schema:
        raise ValueError(f"schema mismatch for {path}")
    if metadata.get("array_sha256") != _array_hash(*(arrays[name] for name in sorted(arrays))):
        raise ValueError(f"payload hash mismatch for {path}")
    return arrays, metadata


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _event(stage: str, **details: Any) -> None:
    print(json.dumps({"event": "p06_curvature_global", "stage": stage, **_json(details)}, sort_keys=True), flush=True)


def _config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "drbx.p06-structured-global-runtime-v1":
        raise ValueError("unsupported P06 structured runtime configuration")
    return payload


def _path(config: Mapping[str, Any], name: str) -> Path:
    return Path(config["paths"][name]).resolve()


def _source_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "implementation": Path(__file__),
        "parallel_runner": HERE / "parallel_runner.py",
        "observations": HERE / "observations.py",
        "campaign": HERE / "campaign.py",
        "configuration": HERE / "configuration.json",
        "input_manifest": HERE / "input_manifest.json",
        "portable_geometry_reference": SCRIPTS / "p07_diffusion_global/numerics.py",
        "structured_kernels": SCRIPTS / "p07_combined_global/kernels.py",
        "continuum_reference": ROOT / "hsx_mms_continuum_reference.py",
        "owner_geometry": ROOT / "src/drbx/geometry/fci_perpendicular_bracket.py",
        "boundary_rows": ROOT / "src/drbx/geometry/fci_boundary_functional_reconstruction.py",
        "curvature_matrix": ROOT / "src/drbx/native/fci_curvature_production_flux.py",
        "curvature_wall": ROOT / "src/drbx/native/fci_operators.py",
        "reference_sidecar": _path(config, "reference_sidecar"),
    }
    paths.update({f"shared:{p.relative_to(SCRIPTS)}": p for p in sorted((SCRIPTS / "perpendicular_structured").rglob("*.py"))})
    if not any(name.startswith("shared:") for name in paths):
        raise ValueError("shared perpendicular_structured service is required")
    return {name: _identity(path) for name, path in paths.items()}


def _raw_points(context: Any) -> np.ndarray:
    return np.stack(
        np.meshgrid(context.x_centers, context.y_centers, context.z_centers, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)


def _raw_owner(context: Any) -> np.ndarray:
    result = np.asarray(context.raw_owner, dtype=np.int64)
    if np.any(result < 0):
        raise ValueError("raw topology contains an inactive owner")
    return result


def _owner_observations(context: Any, raw_values: np.ndarray) -> np.ndarray:
    raw_owner = _raw_owner(context)
    raw_volume = np.asarray(context.arrays["raw_volume"], dtype=np.float64)
    count = len(context.geometry.owner_flat_ids)
    volume = np.bincount(raw_owner, weights=raw_volume, minlength=count)
    result = np.empty((raw_values.shape[0], count), dtype=np.float64)
    for field in range(raw_values.shape[0]):
        result[field] = np.bincount(
            raw_owner, weights=raw_volume * raw_values[field], minlength=count
        ) / volume
    return result


def _heldout_fields(points: np.ndarray, eta_period: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(points, dtype=np.float64)
    u, theta, eta = q.T
    x = u * np.cos(theta)
    y = u * np.sin(theta)
    zeta = 2.0 * np.pi * eta / float(eta_period)
    wave = 2.0 * np.pi / float(eta_period)
    radius = x * x + y * y
    a = 1.0 - radius
    envelope = a * a
    ex = -4.0 * x * a
    ey = -4.0 * y * a
    specifications = (
        (0.08*x + 0.03*y*np.sin(zeta), 0.08+0*x, 0.03*np.sin(zeta), 0.03*y*np.cos(zeta)),
        (0.06*y + 0.02*x*y*np.cos(zeta), 0.02*y*np.cos(zeta), 0.06+0.02*x*np.cos(zeta), -0.02*x*y*np.sin(zeta)),
        (0.05*x*np.cos(zeta) + 0.02*(x*x-y*y), 0.05*np.cos(zeta)+0.04*x, -0.04*y, -0.05*x*np.sin(zeta)),
        (0.05*(x*x-y*y) + 0.02*x*np.sin(zeta), 0.10*x+0.02*np.sin(zeta), -0.10*y, 0.02*x*np.cos(zeta)),
        (0.04*x*y + 0.03*y*np.cos(zeta), 0.04*y, 0.04*x+0.03*np.cos(zeta), -0.03*y*np.sin(zeta)),
    )
    values = np.empty((5, len(q)), dtype=np.float64)
    gradients = np.empty((5, len(q), 3), dtype=np.float64)
    for field, ((function, fx, fy, fz), background) in enumerate(
        zip(specifications, (1.0, 1.0, 1.0, 0.0, 0.0), strict=True)
    ):
        values[field] = background + envelope * function
        gx = ex * function + envelope * fx
        gy = ey * function + envelope * fy
        gradients[field, :, 0] = gx*np.cos(theta) + gy*np.sin(theta)
        gradients[field, :, 1] = -u*gx*np.sin(theta) + u*gy*np.cos(theta)
        gradients[field, :, 2] = wave * envelope * fz
    return values, gradients


def _dirichlet_fields(points: np.ndarray, eta_period: float, *, variable: bool) -> tuple[np.ndarray, np.ndarray]:
    """Positive thermodynamic controls with a prescribed phi wall trace.

    The phi trace is zero or spatially varying; both have a nonzero radial
    derivative on the wall. The formula is regular in Cartesian x/y at the
    magnetic axis, so the axis is not exempted from the global check.
    """
    values, gradients = _heldout_fields(points, eta_period)
    q = np.asarray(points, dtype=np.float64)
    u, theta, eta = q.T
    x, y = u * np.cos(theta), u * np.sin(theta)
    frequency = 2.0 * np.pi / float(eta_period)
    phase = frequency * eta
    envelope = 1.0 - x*x - y*y
    forcing = 0.03 + 0.02*x*np.sin(phase)
    phi = envelope * forcing
    gx = -2*x*forcing + envelope*0.02*np.sin(phase)
    gy = -2*y*forcing
    gz = envelope*0.02*x*np.cos(phase)
    if variable:
        phi += 0.04*(x*x-y*y)*np.cos(phase)
        gx += 0.08*x*np.cos(phase)
        gy -= 0.08*y*np.cos(phase)
        gz -= 0.04*(x*x-y*y)*np.sin(phase)
    values[4] = phi
    gradients[4, :, 0] = gx*np.cos(theta) + gy*np.sin(theta)
    gradients[4, :, 1] = -u*gx*np.sin(theta) + u*gy*np.cos(theta)
    gradients[4, :, 2] = frequency*gz
    return values, gradients


def _evaluate_fields(name: str, reference: Any, points: np.ndarray, time_value: float) -> tuple[np.ndarray, np.ndarray]:
    if name == "regular_chart_heldout":
        return _heldout_fields(points, reference.eta_period)
    if name == "homogeneous_dirichlet":
        return _dirichlet_fields(points, reference.eta_period, variable=False)
    if name == "variable_dirichlet":
        return _dirichlet_fields(points, reference.eta_period, variable=True)
    if name != "corrected_frozen_mms":
        raise ValueError(name)
    raw = reference._fields_raw(np.asarray(points, dtype=np.float64), float(time_value))
    values = np.empty((5, len(points)), dtype=np.float64)
    gradients = np.empty((5, len(points), 3), dtype=np.float64)
    for field, name_ in enumerate(("density", "Te", "Ti", "vorticity", "phi")):
        if name_ == "vorticity":
            values[field] = 0.0
            gradients[field] = 0.0
        else:
            payload = raw[name_]
            values[field] = np.asarray(payload[0]) + (1.0 if field < 3 else 0.0)
            gradients[field] = np.stack(payload[1:4], axis=-1)
    return values, gradients


def _continuum_terms(values: np.ndarray, gradients: np.ndarray, prepared: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, te, ti, _omega, _phi = values
    bmag = np.asarray(prepared.B)
    curvature = np.einsum("pd,fpd->fp", np.asarray(prepared.K), gradients)
    matrix = np.asarray(curvature_principal_matrix(n, te, ti, bmag, TAU))
    material = np.einsum("pij,jp->pi", matrix, curvature[:4]) / np.maximum(bmag[:, None], 1.0e-30)
    cpsi = curvature[4] + TAU * curvature[2]
    coeff = np.column_stack((-2*n/bmag, -4*te/(3*bmag), -4*ti/(3*bmag), np.zeros_like(n)))
    remainder = coeff * cpsi[:, None]
    directional_curvature = np.einsum("pd,fpd->pdf", np.asarray(prepared.K), gradients)
    material_directional = np.einsum("pij,pdj->pdi", matrix, directional_curvature[..., :4]) / np.maximum(bmag[:, None, None], 1.0e-30)
    remainder_directional = (directional_curvature[..., 4] + TAU*directional_curvature[..., 2])[..., None] * coeff[:, None, :]
    return material, remainder, material + remainder, material_directional, remainder_directional, material_directional + remainder_directional


def _face_geometry(reference: Any, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(points, dtype=np.float64)
    metric = reference._metric(q)
    K = np.empty((len(q), 3), dtype=np.float64)
    wall = np.isclose(q[:, 0], 1.0, rtol=0.0, atol=8*np.finfo(float).eps)
    if np.any(~wall):
        K[~wall] = reference._curvature(q[~wall])
    if np.any(wall):
        qw = q[wall]
        def covariant_over_b(x: np.ndarray) -> np.ndarray:
            local = reference._metric(x)
            return local["bcov"] / local["B"][:, None]
        h = min(float(reference.finite_difference_step), 0.05)
        sample = []
        for offset in range(5):
            shifted = qw.copy(); shifted[:, 0] -= offset*h
            sample.append(covariant_over_b(shifted))
        du = (25*sample[0]-48*sample[1]+36*sample[2]-16*sample[3]+3*sample[4])/(12*h)
        dtheta = reference._derivative(covariant_over_b, qw, 1)
        deta = reference._derivative(covariant_over_b, qw, 2)
        curl = np.stack((dtheta[...,2]-deta[...,1], deta[...,0]-du[...,2], du[...,1]-dtheta[...,0]), axis=-1)
        K[wall] = 0.5 * metric["B"][wall, None] * curl / np.maximum(np.abs(metric["J"])[wall, None], 1.0e-30)
    return np.asarray(metric["J"]), np.asarray(metric["B"]), K


def _principal_matrix(state: np.ndarray, bmag: np.ndarray) -> np.ndarray:
    n, te, ti = np.moveaxis(state[..., :3], -1, 0)
    matrix = np.zeros(state.shape[:-1] + (4, 4), dtype=np.float64)
    matrix[...,0,0]=2*te; matrix[...,0,1]=2*n; matrix[...,0,2]=2*n*TAU
    matrix[...,1,0]=4*te*te/(3*n); matrix[...,1,1]=14*te/3; matrix[...,1,2]=4*TAU*te/3
    matrix[...,2,0]=4*ti*te/(3*n); matrix[...,2,1]=4*ti/3; matrix[...,2,2]=-2*TAU*ti
    matrix[...,3,0]=2*bmag*bmag*(te+TAU*ti)/n; matrix[...,3,1]=2*bmag*bmag; matrix[...,3,2]=2*TAU*bmag*bmag
    return matrix


def _absolute_action(matrix: np.ndarray, jump: np.ndarray) -> tuple[np.ndarray, int]:
    result = np.empty_like(jump)
    fallback = 0
    for row, (operator, vector) in enumerate(zip(matrix.reshape(-1,4,4), jump.reshape(-1,4), strict=True)):
        eigenvalues, eigenvectors = np.linalg.eig(operator)
        if (
            np.max(np.abs(np.imag(eigenvalues))) <= 1.0e-10*(1+np.max(np.abs(np.real(eigenvalues))))
            and np.isfinite(np.linalg.cond(eigenvectors))
            and np.linalg.cond(eigenvectors) <= 1.0e8
        ):
            result.reshape(-1,4)[row] = np.real(eigenvectors @ (np.abs(np.real(eigenvalues)) * (np.linalg.inv(eigenvectors) @ vector)))
        else:
            result.reshape(-1,4)[row] = np.linalg.norm(operator) * vector
            fallback += 1
    return result, fallback


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Reduce complete checkpointed midpoint observations to actual owners."""
    config=_config(args.config); output=_path(config,"output"); n=int(args.resolution)
    plan_path=output/f"N{n}.observation.plan.json"
    receipt_path=output/f"N{n}.observation-receipt.json"
    plan=json.loads(plan_path.read_text()); receipt=json.loads(receipt_path.read_text())
    if (plan.get("resolution")!=n or receipt.get("status")!="complete"
        or receipt.get("identity")!=plan.get("identity")
        or receipt.get("plan_sha256")!=_sha256(plan_path)
        or not receipt.get("data_sha256")):
        raise ValueError("invalid observation plan or receipt")
    target=output/f"N{n}.prepare.npz"
    if target.is_file():
        _arrays,metadata=_load_npz(target,PREPARE_SCHEMA)
        if (metadata.get("sources")==_json(_source_identity(config))
            and metadata.get("observation_identity")==plan["identity"]
            and metadata.get("observation_data_sha256")==receipt["data_sha256"]):
            _event("prepare_cache_hit",resolution=n)
            return metadata
        raise ValueError("stale owner preparation; use a new campaign folder")
    started=time.perf_counter()
    context=cubic._load_context(_path(config,"geometry"),_path(config,"baseline"),n)
    data=integrated._load_resolution(_path(config,"geometry"),_path(config,"baseline"),n)
    owner_count=len(data.owner_volume)
    sums=np.zeros((len(FIELD_NAMES),5,owner_count),dtype=np.float64)
    seen=np.zeros(n**3,dtype=bool)
    chunk_hashes={}
    raw_owner=np.asarray(context.raw_owner,dtype=np.int64)
    raw_volume=np.asarray(context.arrays["raw_volume"],dtype=np.float64)
    volume=np.zeros(owner_count)
    for unit in plan["units"]:
        path=output/f"N{n}.observation.chunks"/f"{unit['id']}.npz"
        arrays,meta=_load_npz(path,"drbx.p06-structured-observation-chunk-v1")
        ids=np.arange(int(unit["first"]),int(unit["last"]),dtype=np.int64)
        if (meta.get("identity")!=plan["identity"] or not np.array_equal(arrays["indices"],ids)
            or arrays["values"].shape!=(len(FIELD_NAMES),5,len(ids))
            or meta.get("array_sha256")!=_array_hash(*(arrays[name] for name in sorted(arrays)))
            or np.any(seen[ids])):
            raise ValueError(f"stale, corrupt or duplicate observation chunk {path}")
        chunk_hashes[unit["id"]]=meta["array_sha256"]
        seen[ids]=True
        owners=raw_owner[ids];weights=raw_volume[ids]
        np.add.at(volume,owners,weights)
        for state in range(len(FIELD_NAMES)):
            for field in range(5):
                np.add.at(sums[state,field],owners,weights*arrays["values"][state,field])
    if not np.all(seen) or np.any(volume<=0):
        raise ValueError("incomplete observation raw-cell coverage")
    digest=hashlib.sha256(json.dumps(dict(sorted(chunk_hashes.items())),sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    if digest!=receipt["data_sha256"]:
        raise ValueError("observation receipt does not match complete checkpoint data")
    mismatch=float(np.max(np.abs(volume-data.owner_volume)/np.maximum(np.abs(data.owner_volume),1e-300)))
    if mismatch>1e-12:
        raise ValueError(f"owner volume differs from immutable topology: {mismatch}")
    states=sums/volume[None,None,:]
    states[0,3]=np.asarray(context.arrays["owner_values"][1])
    arrays={"owner_values":states,"owner_volume":np.asarray(data.owner_volume),
            "owner_keys":np.asarray(data.owner_keys),"raw_owner":np.asarray(data.raw_owner,dtype=np.int32),
            **{f"region:{name}":np.asarray(mask,dtype=bool) for name,mask in data.masks.items()}}
    metadata={"schema":PREPARE_SCHEMA,"status":"complete","resolution":n,
              "candidate":config["candidate"],
              "owner_observation":"raw midpoint values projected with physical raw_volume",
              "observation_identity":plan["identity"],
              "observation_data_sha256":receipt["data_sha256"],
              "observation_plan_sha256":_sha256(plan_path),
              "volume_relative_mismatch_max":mismatch,
              "sources":_source_identity(config),
              "geometry_manifest":_identity(data.artifact_path/"manifest.json"),
              "seconds":time.perf_counter()-started,"maximum_rss_gib":_max_rss_gib()}
    metadata["array_sha256"]=_array_hash(*(arrays[name] for name in sorted(arrays)))
    _write_npz(target,arrays,metadata)
    _event("prepare_complete",resolution=n,seconds=metadata["seconds"])
    return metadata


_STRUCTURED_CACHE: dict[tuple[int, str], Any] = {}
_TRACE_CACHE: dict[tuple[int, str, float], dict[bytes, tuple[np.ndarray, np.ndarray]]] = {}


def _structured(n: int, input_root: Path) -> Any:
    """Use the common geometry-only service; never construct selection-v3 fits."""
    key = (int(n), str(Path(input_root).resolve()))
    if key not in _STRUCTURED_CACHE:
        from perpendicular_structured.reconstruction import StructuredReconstruction, load_context
        _STRUCTURED_CACHE[key] = StructuredReconstruction(load_context(int(n), Path(input_root)))
    return _STRUCTURED_CACHE[key]


def _trace(name: str, reference: Any, time_value: float):
    def analytic(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(points, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("trace points must have shape (Q,3)")
        cache = _TRACE_CACHE.setdefault((id(reference), name, float(time_value)), {})
        if len(cache) > 32768:
            cache.clear()
        keys = [np.ascontiguousarray(point).tobytes() for point in q]
        missing = [(key, point) for key, point in zip(keys, q, strict=True) if key not in cache]
        if missing:
            unique = dict(missing)
            values, gradients = _evaluate_fields(name, reference, np.stack(list(unique.values())), time_value)
            for column, key in enumerate(unique):
                cache[key] = (values[:, column].copy(), gradients[:, column].T.copy())
        return (np.stack([cache[key][0] for key in keys]),
                np.stack([cache[key][1] for key in keys]))
    return analytic


def _apply_rows(rows: Any, owner_values: np.ndarray, trace: Any) -> tuple[np.ndarray, np.ndarray]:
    value, gradient = rows.apply(owner_values.T, trace=trace)
    value = np.asarray(value, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 5 or gradient.shape != (len(value), 3, 5):
        raise ValueError("shared reconstruction returned invalid value/gradient shape")
    if not (np.all(np.isfinite(value)) and np.all(np.isfinite(gradient))):
        raise ValueError("nonfinite shared reconstructed value/gradient")
    return value, gradient


def _face_incidence(n: int, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw-cell face incidence with periodic theta/eta and radial boundaries."""
    lower = np.full(len(keys), -1, dtype=np.int64)
    upper = np.full(len(keys), -1, dtype=np.int64)
    for axis in range(3):
        rows = np.flatnonzero(keys[:, 0] == axis)
        if not len(rows):
            continue
        xyz = np.asarray(keys[rows, 1:], dtype=np.int64)
        m = xyz[:, axis]
        lower_valid = (m > 0) if axis == 0 else np.ones(len(rows), dtype=bool)
        upper_valid = (m < n) if axis == 0 else np.ones(len(rows), dtype=bool)
        lo = xyz.copy(); hi = xyz.copy()
        lo[:, axis] = (m - 1) % n
        hi[:, axis] = m % n
        lower[rows[lower_valid]] = np.ravel_multi_index(lo[lower_valid].T, (n,n,n))
        upper[rows[upper_valid]] = np.ravel_multi_index(hi[upper_valid].T, (n,n,n))
    return lower, lower >= 0, upper, upper >= 0


def _compute_cells(
    context: Any,
    reference: Any,
    prepare: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    time_value: float,
    curl_step: float,
    input_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    del curl_step
    indices = np.asarray(indices, dtype=np.int64)
    keys = base._raw_keys(context.resolution, indices)
    points, weights = base._cell_quadrature(context, keys)
    service = _structured(context.resolution, input_root)
    owner = np.asarray(prepare["owner_values"])
    values = np.empty((len(FIELD_NAMES), 5, len(indices), 27), dtype=np.float64)
    gradients = np.empty((len(FIELD_NAMES), 5, len(indices), 27, 3), dtype=np.float64)
    donor_count = np.empty(len(indices), dtype=np.int32)
    conditioned = np.empty(len(indices), dtype=bool)
    for row, key in enumerate(keys):
        operator = service.rows(tuple(map(int, key)), points[row], location="cell")
        donor_count[row] = len(operator.donor_ids)
        conditioned[row] = bool(operator.boundary_conditioned)
        for state, name in enumerate(FIELD_NAMES):
            v, g = _apply_rows(operator, owner[state], _trace(name, reference, time_value))
            values[state, :, row] = v.T
            gradients[state, :, row] = np.moveaxis(g, -1, 0)
    flat = points.reshape(-1, 3)
    prepared_geometry = reference.prepare(flat)
    metric = reference._metric(flat)
    jacobian = np.asarray(metric["J"]).reshape(len(indices), 27)
    bmag = np.asarray(metric["B"]).reshape(len(indices), 27)
    evolution_weight = weights * jacobian / np.maximum(bmag, 1.0e-30)
    physical_weight = weights * jacobian
    arrays: dict[str, np.ndarray] = {
        "indices": indices,
        "evolution_volume": np.sum(evolution_weight, axis=1),
        "physical_volume": np.sum(physical_weight, axis=1),
        "donor_count": donor_count,
        "boundary_conditioned": conditioned,
    }
    closure = 0.0
    for state_index, field_name in enumerate(FIELD_NAMES):
        candidate = _continuum_terms(
            values[state_index].reshape(5,-1),
            gradients[state_index].reshape(5,-1,3),
            prepared_geometry,
        )
        exact_values, exact_gradients = _evaluate_fields(field_name, reference, flat, time_value)
        exact = _continuum_terms(exact_values, exact_gradients, prepared_geometry)
        for term_index, term in enumerate(TERMS):
            candidate_term = candidate[term_index].reshape(len(indices),27,4)
            exact_term = exact[term_index].reshape(len(indices),27,4)
            arrays[f"candidate:{field_name}:{term}"] = np.sum(evolution_weight[...,None]*candidate_term, axis=1)
            arrays[f"reference_physical:{field_name}:{term}"] = np.sum(physical_weight[...,None]*exact_term, axis=1)
            arrays[f"reference_evolution:{field_name}:{term}"] = np.sum(evolution_weight[...,None]*exact_term, axis=1)
            candidate_directional = candidate[term_index+3].reshape(len(indices),27,3,4)
            arrays[f"candidate_directional:{field_name}:{term}"] = np.sum(evolution_weight[...,None,None]*candidate_directional, axis=1)
        closure = max(closure, float(np.max(np.abs(candidate[0]+candidate[1]-candidate[2]))))
    details = {
        "entity_count": len(indices),
        "boundary_conditioned_count": int(np.count_nonzero(conditioned)),
        "M_plus_R_closure_max": closure,
        "maximum_donor_count": int(np.max(donor_count)),
    }
    return arrays, details


def _compute_faces(
    context: Any,
    reference: Any,
    prepare: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    time_value: float,
    input_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    indices = np.asarray(indices, dtype=np.int64)
    keys = base._face_keys(context.resolution, indices)
    points, weights = base._face_quadrature(context, keys)
    service = _structured(context.resolution, input_root)
    owner = np.asarray(prepare["owner_values"])
    face_count = len(indices)
    central = np.zeros((len(FIELD_NAMES),5,face_count,9), dtype=np.float64)
    left = np.zeros_like(central); right = np.zeros_like(central)
    donor_count = np.zeros(face_count, dtype=np.int32)
    conditioned = np.zeros(face_count, dtype=bool)
    collapsed = (keys[:,0] == 0) & (keys[:,1] == 0)
    for row, key in enumerate(keys):
        if collapsed[row]:
            continue
        face_key = tuple(map(int,key))
        operator = service.rows(face_key, points[row], location="face")
        sides = service.side_rows(face_key, points[row])
        if not isinstance(sides, tuple) or len(sides) != 2:
            raise ValueError("shared side_rows must return (left,right)")
        donor_count[row] = len(operator.donor_ids)
        conditioned[row] = bool(operator.boundary_conditioned)
        for state, name in enumerate(FIELD_NAMES):
            trace = _trace(name, reference, time_value)
            v, _g = _apply_rows(operator, owner[state], trace)
            central[state,:,row] = v.T
            for target, side in ((left, sides[0]), (right, sides[1])):
                if side is None:
                    target[state,:,row] = v.T
                else:
                    sv, _sg = _apply_rows(side, owner[state], trace)
                    target[state,:,row] = sv.T
    wall_rows = np.flatnonzero((keys[:,0]==0)&(keys[:,1]==context.resolution))
    trace_error = 0.0
    wall_normal_gradient_max = {name: 0.0 for name in FIELD_NAMES}
    wall_phi_trace_range = {name: [float('inf'), float('-inf')] for name in FIELD_NAMES}
    for state, name in enumerate(FIELD_NAMES):
        for row in wall_rows:
            exact_value, exact_gradient = _evaluate_fields(name, reference, points[row], time_value)
            trace_error = max(trace_error, float(np.max(np.abs(central[state,4,row] - exact_value[4]))))
            wall_normal_gradient_max[name] = max(
                wall_normal_gradient_max[name], float(np.max(np.abs(exact_gradient[4,:,0]))))
            wall_phi_trace_range[name][0] = min(wall_phi_trace_range[name][0], float(np.min(exact_value[4])))
            wall_phi_trace_range[name][1] = max(wall_phi_trace_range[name][1], float(np.max(exact_value[4])))
    if not len(wall_rows):
        wall_phi_trace_range = {name: [None, None] for name in FIELD_NAMES}
    regular = ~collapsed
    J = np.zeros((face_count,9)); B = np.ones((face_count,9)); K = np.zeros((face_count,9,3))
    if np.any(regular):
        rj, rb, rk = _face_geometry(reference, points[regular].reshape(-1,3))
        J[regular] = rj.reshape(-1,9)
        B[regular] = rb.reshape(-1,9)
        K[regular] = rk.reshape(-1,9,3)
    normal = J * np.take_along_axis(K, keys[:,0,None,None].repeat(9,axis=1), axis=2)[:,:,0] / np.maximum(B*B,1.0e-30)
    correction = np.zeros((len(FIELD_NAMES),face_count,2,4))
    jump_rms = []
    spectral_fallback = 0
    positivity_fallback = 0
    for state in range(len(FIELD_NAMES)):
        state_central = central[state,:4].transpose(1,2,0).copy()
        state_left = left[state,:4].transpose(1,2,0).copy()
        state_right = right[state,:4].transpose(1,2,0).copy()
        state_central[collapsed,:,:3] = 1.0
        state_left[collapsed,:,:3] = 1.0
        state_right[collapsed,:,:3] = 1.0
        for row in wall_rows:
            interior = state_central[row]
            exterior, working, _fallback = _curvature_bc_characteristic_wall_states(
                jnp.asarray(interior), jnp.asarray(interior), jnp.asarray(B[row]), TAU,
                jnp.asarray(normal[row]), interior_on_right=False, positivity_floor=FLOOR,
            )
            state_left[row] = interior
            state_right[row] = np.asarray(exterior)
            state_central[row] = np.asarray(working)
        matrix = _principal_matrix(state_central, B)
        flux_matrix = -normal[...,None,None]*matrix
        jump = state_right-state_left
        absolute, fallback = _absolute_action(flux_matrix,jump)
        spectral_fallback += fallback
        positivity_fallback += int(np.count_nonzero(state_central[...,:3] <= FLOOR))
        material = np.einsum("fqij,fqj->fqi",flux_matrix,jump)
        dplus=0.5*(material+absolute); dminus=0.5*(material-absolute)
        dplus[collapsed]=0.0; dminus[collapsed]=0.0
        correction[state,:,0] = -np.sum(weights[...,None]*dminus,axis=1)
        correction[state,:,1] = -np.sum(weights[...,None]*dplus,axis=1)
        jump_rms.append(float(np.sqrt(np.mean(jump[regular]**2))) if np.any(regular) else 0.0)
    lower_raw, lower_valid, upper_raw, upper_valid = _face_incidence(context.resolution, keys)
    arrays = {
        "indices": indices,
        "keys": np.asarray(keys),
        "correction": correction,
        "lower_raw": lower_raw,
        "lower_valid": lower_valid,
        "upper_raw": upper_raw,
        "upper_valid": upper_valid,
        "donor_count": donor_count,
        "boundary_conditioned": conditioned,
    }
    details = {
        "entity_count": face_count,
        "wall_face_count": len(wall_rows),
        "collapsed_face_count": int(np.count_nonzero(collapsed)),
        "boundary_conditioned_count": int(np.count_nonzero(conditioned)),
        "dirichlet_trace_error_max": trace_error,
        "phi_wall_normal_gradient_max": wall_normal_gradient_max,
        "phi_wall_trace_range": wall_phi_trace_range,
        "jump_rms": jump_rms,
        "spectral_fallback_count": spectral_fallback,
        "positivity_fallback_count": positivity_fallback,
        "maximum_donor_count": int(np.max(donor_count)) if face_count else 0,
    }
    return arrays, details


def _compute_reference_global(
    context: Any,
    reference: Any,
    indices: np.ndarray,
    *,
    time_value: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    indices = np.asarray(indices, dtype=np.int64)
    numerator, volume = _reference_on_raw_cells(context, reference, indices, 5, time_value)
    arrays = {"indices": indices, "numerator_q5": numerator, "volume_q5": volume}
    if not all(np.all(np.isfinite(value)) for value in arrays.values()) or np.any(volume <= 0):
        raise ValueError("invalid q5 global continuous reference")
    return arrays, {"entity_count": len(indices), "physical_reference_rule": "q5", "maximum_rss_gib": _max_rss_gib()}


def _compute_reference_control(
    context: Any,
    reference: Any,
    indices: np.ndarray,
    *,
    time_value: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Independent q3/q5/q7 exact-field volume controls on complete raw cells."""
    indices = np.asarray(indices, dtype=np.int64)
    arrays: dict[str, np.ndarray] = {"indices": indices}
    for order in (3, 5, 7):
        numerator, volume = _reference_on_raw_cells(context, reference, indices, order, time_value)
        arrays[f"numerator_q{order}"] = numerator
        arrays[f"volume_q{order}"] = volume
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("nonfinite bounded reference control")
    return arrays, {"entity_count": len(indices), "orders": [3, 5, 7]}


def _face_count(n: int) -> int:
    return base._face_count(n)


def _raw_keys(n: int, indices: np.ndarray) -> np.ndarray:
    return base._raw_keys(n, indices)


def _face_index(n: int, axis: int, i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
    return base._face_index(n, axis, i, j, k)


def _chunk_valid(path: Path, identity: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        arrays, metadata = _load_npz(path, CHUNK_SCHEMA)
    except Exception:
        return False
    return metadata.get("identity") == _json(identity) and metadata.get("status") == "complete"


def _statistics(actual: np.ndarray, exact: np.ndarray, data: Any) -> dict[str, Any]:
    return integrated._compact_statistics(np.asarray(actual), np.asarray(exact), data)


def _case(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args.config); output = _path(config,"output"); n = int(args.resolution)
    started = time.perf_counter()
    prepare, prepare_meta = _load_npz(output/f"N{n}.prepare.npz", PREPARE_SCHEMA)
    raw_owner = prepare["raw_owner"].astype(np.int64)
    owner_count = len(prepare["owner_volume"])
    evolution_volume=np.zeros(owner_count); physical_volume=np.zeros(owner_count)
    q5_reference_volume=np.zeros(owner_count)
    candidate={field:{term:np.zeros((owner_count,4)) for term in TERMS} for field in FIELD_NAMES}
    candidate_directional={field:{term:np.zeros((owner_count,3,4)) for term in TERMS} for field in FIELD_NAMES}
    reference_physical={field:{term:np.zeros((owner_count,4)) for term in TERMS} for field in FIELD_NAMES}
    reference_q5={field:{term:np.zeros((owner_count,4)) for term in TERMS} for field in FIELD_NAMES}
    reference_evolution={field:{term:np.zeros((owner_count,4)) for term in TERMS} for field in FIELD_NAMES}
    face_correction={field:np.zeros((owner_count,4)) for field in FIELD_NAMES}
    face_directional={field:np.zeros((owner_count,3,4)) for field in FIELD_NAMES}
    chunk_root=output/f"N{n}.chunks"
    face_details=[]; cell_details=[]
    cell_paths=sorted(chunk_root.glob("cell_*.npz"))+sorted(chunk_root.glob(f"N{n}-cell-*.npz"))
    face_paths=sorted(chunk_root.glob("face_*.npz"))+sorted(chunk_root.glob(f"N{n}-face-*.npz"))
    reference_paths=sorted(chunk_root.glob("reference_*.npz"))+sorted(chunk_root.glob(f"N{n}-reference-*.npz"))
    if not cell_paths or not face_paths or not reference_paths:
        raise RuntimeError("global assembly requires validated face, cell and q5 reference chunks")
    for path in cell_paths:
        arrays, metadata=_load_npz(path,CHUNK_SCHEMA); cell_details.append(metadata["details"])
        indices=arrays["indices"].astype(np.int64); owners=raw_owner[indices]
        np.add.at(evolution_volume,owners,arrays["evolution_volume"])
        np.add.at(physical_volume,owners,arrays["physical_volume"])
        for field in FIELD_NAMES:
            for term in TERMS:
                np.add.at(candidate[field][term],owners,arrays[f"candidate:{field}:{term}"])
                np.add.at(candidate_directional[field][term],owners,arrays[f"candidate_directional:{field}:{term}"])
                np.add.at(reference_physical[field][term],owners,arrays[f"reference_physical:{field}:{term}"])
                np.add.at(reference_evolution[field][term],owners,arrays[f"reference_evolution:{field}:{term}"])
    reference_raw_count=0
    for path in reference_paths:
        arrays, _metadata=_load_npz(path,CHUNK_SCHEMA)
        indices=arrays["indices"].astype(np.int64); owners=raw_owner[indices]
        reference_raw_count += len(indices)
        np.add.at(q5_reference_volume,owners,arrays["volume_q5"])
        for state,field in enumerate(FIELD_NAMES):
            for term_index,term in enumerate(TERMS):
                np.add.at(reference_q5[field][term],owners,arrays["numerator_q5"][state,term_index])
    if reference_raw_count != n**3:
        raise RuntimeError("incomplete q5 reference raw-cell coverage")
    for path in face_paths:
        arrays,metadata=_load_npz(path,CHUNK_SCHEMA); face_details.append(metadata["details"])
        for row in range(len(arrays["indices"])):
            axis=int(arrays["keys"][row,0])
            for side,valid_name,raw_name in ((0,"lower_valid","lower_raw"),(1,"upper_valid","upper_raw")):
                if not bool(arrays[valid_name][row]): continue
                owner=raw_owner[int(arrays[raw_name][row])]
                for state,field in enumerate(FIELD_NAMES):
                    value=arrays["correction"][state,row,side]
                    face_correction[field][owner]+=value
                    face_directional[field][owner,axis]+=value
    if np.any(evolution_volume<=0) or np.any(physical_volume<=0) or np.any(q5_reference_volume<=0):
        raise RuntimeError("incomplete complete-owner volume coverage")
    data=integrated._load_resolution(_path(config,"geometry"),_path(config,"baseline"),n)
    if not np.array_equal(data.owner_keys,prepare["owner_keys"]):
        raise ValueError("owner ordering mismatch")
    arrays_out={"owner_keys":prepare["owner_keys"],"owner_volume":physical_volume,
                "evolution_volume":evolution_volume,"q5_reference_volume":q5_reference_volume}
    statistics:dict[str,Any]={}
    closure=0.0; reference_closure=0.0
    for field in FIELD_NAMES:
        statistics[field]={}
        target={term:reference_q5[field][term]/q5_reference_volume[:,None] for term in TERMS}
        reference_closure=max(reference_closure,float(np.max(np.abs(target["material"]+target["remainder"]-target["total"]))))
        centered={term:candidate[field][term]/evolution_volume[:,None] for term in TERMS}
        upwind={term:centered[term].copy() for term in TERMS}
        upwind["material"] += face_correction[field]/evolution_volume[:,None]
        upwind["total"] = upwind["material"]+upwind["remainder"]
        closure=max(closure,float(np.max(np.abs(upwind["material"]+upwind["remainder"]-upwind["total"]))))
        for term in TERMS:
            arrays_out[f"target:{field}:{term}"]=target[term]
            arrays_out[f"target_q3_diagnostic:{field}:{term}"]=reference_physical[field][term]/physical_volume[:,None]
            arrays_out[f"candidate:centered:{field}:{term}"]=centered[term]
            arrays_out[f"candidate:U:{field}:{term}"]=upwind[term]
            arrays_out[f"candidate_directional:centered:{field}:{term}"]=candidate_directional[field][term]/evolution_volume[:,None,None]
            u_directional=arrays_out[f"candidate_directional:centered:{field}:{term}"].copy()
            if term in ("material","total"):
                u_directional += face_directional[field]/evolution_volume[:,None,None]
            arrays_out[f"candidate_directional:U:{field}:{term}"]=u_directional
            arrays_out[f"reference_evolution:{field}:{term}"]=reference_evolution[field][term]/evolution_volume[:,None]
        for action, values in (("centered",centered),("U",upwind)):
            statistics[field][action]={}
            for term in TERMS:
                statistics[field][action][term]={
                    equation:_statistics(values[term][:,eq],target[term][:,eq],data)
                    for eq,equation in enumerate(EQUATIONS)
                }
    output_npz=output/f"N{n}.npz"
    metadata={
        "schema":SCHEMA,"status":"complete","resolution":n,
        "candidate":config["candidate"],"scope":config["scope"],"primary_action":PRIMARY_ACTION,
        "sources":_source_identity(config),"prepare":_identity(output/f"N{n}.prepare.npz"),
        "statistics":statistics,
        "verification":{
            "finite_complete_owner_coverage":all(np.all(np.isfinite(v)) for v in arrays_out.values()),
            "M_plus_R_closure_max":closure,
            "q5_reference_M_plus_R_closure_max":reference_closure,
            "q5_to_q3_physical_volume_relative_max":float(np.max(np.abs(q5_reference_volume-physical_volume)/np.maximum(np.abs(q5_reference_volume),1e-300))),
            "dirichlet_trace_error_max":max((float(d["dirichlet_trace_error_max"]) for d in face_details),default=0.0),
            "physical_wall_model":"frozen curvature characteristic wall state; shared prescribed Dirichlet trace in research reconstruction",
            "reference_rule":"q5 exact-field continuous physical-volume target; q3 retained as diagnostic",
        },
        "timing":{"seconds":time.perf_counter()-started,"maximum_rss_gib":_max_rss_gib()},
    }
    metadata["array_sha256"]=_array_hash(*(arrays_out[name] for name in sorted(arrays_out)))
    _write_npz(output_npz,arrays_out,metadata); _write_json(output/f"N{n}.json",metadata)
    _event("case_complete",resolution=n,seconds=metadata["timing"]["seconds"])
    return metadata


def _selected_complete_owners(data: Any) -> tuple[np.ndarray, dict[str,list[int]]]:
    selected:set[int]=set(); strata:dict[str,list[int]]={}
    for name,mask in data.masks.items():
        candidates=np.flatnonzero(mask)
        if len(candidates):
            pick=[int(candidates[len(candidates)//2])]
            strata[name]=pick; selected.update(pick)
    keys=np.asarray(data.owner_keys)
    seam=np.flatnonzero((keys[:,1]==0)|(keys[:,1]==data.resolution-1)|(keys[:,2]==0)|(keys[:,2]==data.resolution-1))
    seam_pick=sorted(set(map(int,seam[[0,-1]]))) if len(seam) else []
    strata["periodic_seams"]=seam_pick; selected.update(seam_pick)
    frozen_hotspots=json.loads((HERE/"configuration.json").read_text())["preflight_hotspot_owners"].get(str(data.resolution),[])
    if any(owner<0 or owner>=len(data.owner_keys) for owner in frozen_hotspots):
        raise ValueError("preflight hotspot owner outside immutable topology")
    strata["archived_reference_hotspots"]=list(map(int,frozen_hotspots))
    selected.update(map(int,frozen_hotspots))
    return np.asarray(sorted(selected),dtype=np.int64),strata


def _reference_on_raw_cells(context:Any,reference:Any,raw_indices:np.ndarray,order:int,time_value:float) -> tuple[np.ndarray,np.ndarray]:
    keys=base._raw_keys(context.resolution,raw_indices)
    if order==3:
        points,weights=base._cell_quadrature(context,keys)
    else:
        # Same tensor-product logical cell rule, generalized to q5/q7.
        nodes,one=np.polynomial.legendre.leggauss(order)
        points=np.empty((len(keys),order**3,3)); weights=np.empty((len(keys),order**3))
        for row,(i,j,k) in enumerate(keys):
            axes=[]; axis_weights=[]
            for faces,index in ((context.x_faces,i),(context.y_faces,j),(context.z_faces,k)):
                lo=float(faces[index]); hi=float(faces[index+1])
                axes.append(0.5*(lo+hi)+0.5*(hi-lo)*nodes)
                axis_weights.append(0.5*(hi-lo)*one)
            mesh=np.meshgrid(*axes,indexing="ij"); wmesh=np.meshgrid(*axis_weights,indexing="ij")
            points[row]=np.stack(mesh,axis=-1).reshape(-1,3); weights[row]=np.prod(np.stack(wmesh,axis=-1),axis=-1).reshape(-1)
    flat=points.reshape(-1,3); prepared=reference.prepare(flat); metric=reference._metric(flat)
    physical=weights*np.asarray(metric["J"]).reshape(len(keys),-1)
    numerator=np.empty((len(FIELD_NAMES),len(TERMS),len(keys),4))
    for state,field in enumerate(FIELD_NAMES):
        values,gradients=_evaluate_fields(field,reference,flat,time_value)
        terms=_continuum_terms(values,gradients,prepared)
        for term in range(len(TERMS)):
            numerator[state,term]=np.sum(physical[...,None]*terms[term].reshape(len(keys),-1,4),axis=1)
    return numerator,np.sum(physical,axis=1)


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Reduce validated parallel preflight chunks, retaining all seven owners."""
    config = _config(args.config); output = _path(config, "output"); n = int(args.resolution)
    prepare, _ = _load_npz(output / f"N{n}.prepare.npz", PREPARE_SCHEMA)
    data = integrated._load_resolution(_path(config,"geometry"), _path(config,"baseline"), n)
    owners, strata = _selected_complete_owners(data)
    selected_raw = np.flatnonzero(np.isin(prepare["raw_owner"], owners))
    raw_keys = base._raw_keys(n, selected_raw)
    selected_faces: set[int] = set()
    for axis in range(3):
        upper = raw_keys.copy(); upper[:,axis] += 1
        selected_faces.update(map(int,base._face_index(n,axis,*raw_keys.T)))
        selected_faces.update(map(int,base._face_index(n,axis,*upper.T)))
    plan = json.loads((output / f"N{n}.preflight.plan.json").read_text())
    if plan.get("coverage") != "preflight" or plan.get("resolution") != n:
        raise ValueError("incorrect preflight plan")
    receipt = json.loads((output / f"N{n}.preflight.parallel-receipt.json").read_text())
    if receipt.get("numerical_identity") != plan["numerical_identity"] or receipt.get("status") != "complete":
        raise ValueError("missing or stale preflight execution receipt")
    raw_owner = np.asarray(prepare["raw_owner"],dtype=np.int64)
    lookup = {int(owner): row for row, owner in enumerate(owners)}
    shape = (len(FIELD_NAMES),len(TERMS),len(owners),len(EQUATIONS))
    candidate = np.zeros(shape)
    q3_exact = np.zeros(shape)
    controls = {order: np.zeros(shape) for order in (3,5,7)}
    controls_volume = {order: np.zeros(len(owners)) for order in (3,5,7)}
    physical_volume = np.zeros(len(owners)); evolution_volume = np.zeros(len(owners))
    face_correction = np.zeros((len(FIELD_NAMES),len(owners),len(EQUATIONS)))
    seen: dict[str, list[int]] = {kind: [] for kind in ("face","cell","reference_control")}
    face_details=[]; cell_details=[]
    for unit in plan["units"]:
        kind = unit["kind"]
        path = output / f"N{n}.preflight.chunks" / f"{unit['id']}.npz"
        arrays, metadata = _load_npz(path, CHUNK_SCHEMA)
        ids = np.asarray(unit["indices"],dtype=np.int64)
        if not np.array_equal(arrays["indices"],ids) or metadata.get("numerical_identity") != plan["numerical_identity"]:
            raise ValueError(f"stale preflight chunk {path}")
        if metadata.get("array_sha256") != _array_hash(*(arrays[name] for name in sorted(arrays))):
            raise ValueError(f"corrupt preflight chunk {path}")
        seen[kind].extend(map(int,ids))
        if kind == "cell":
            cell_details.append(metadata["details"])
            for position,raw in enumerate(ids):
                row=lookup[int(raw_owner[raw])]
                physical_volume[row] += arrays["physical_volume"][position]
                evolution_volume[row] += arrays["evolution_volume"][position]
                for state,name in enumerate(FIELD_NAMES):
                    for term_index,term in enumerate(TERMS):
                        candidate[state,term_index,row] += arrays[f"candidate:{name}:{term}"][position]
                        q3_exact[state,term_index,row] += arrays[f"reference_physical:{name}:{term}"][position]
        elif kind == "face":
            face_details.append(metadata["details"])
            for position in range(len(ids)):
                for side, valid, raw in ((0,"lower_valid","lower_raw"),(1,"upper_valid","upper_raw")):
                    if bool(arrays[valid][position]):
                        owner=int(raw_owner[int(arrays[raw][position])])
                        if owner in lookup:
                            face_correction[:,lookup[owner]] += arrays["correction"][:,position,side]
        elif kind == "reference_control":
            for position,raw in enumerate(ids):
                row=lookup[int(raw_owner[raw])]
                for order in (3,5,7):
                    controls[order][:,:,row] += arrays[f"numerator_q{order}"][:,:,position]
                    controls_volume[order][row] += arrays[f"volume_q{order}"][position]
        else:
            raise ValueError(kind)
    if sorted(seen["cell"]) != selected_raw.tolist() or sorted(seen["reference_control"]) != selected_raw.tolist() or sorted(seen["face"]) != sorted(selected_faces):
        raise ValueError("preflight chunk coverage is missing or duplicated")
    if min(np.min(physical_volume),np.min(evolution_volume),*(np.min(v) for v in controls_volume.values())) <= 0:
        raise ValueError("nonpositive preflight owner volume")
    centered = candidate / evolution_volume[None,None,:,None]
    upwind = centered.copy()
    upwind[:,0] += face_correction/evolution_volume[None,:,None]
    upwind[:,2] = upwind[:,0]+upwind[:,1]
    references = {order: controls[order]/controls_volume[order][None,None,:,None] for order in (3,5,7)}
    q3_cell = q3_exact/physical_volume[None,None,:,None]
    q3_replay = float(np.max(np.abs(q3_cell-references[3])))
    budget={}; sampled_errors={}
    for state,name in enumerate(FIELD_NAMES):
        budget[name]={}; sampled_errors[name]={}
        for term_index,term in enumerate(TERMS):
            budget[name][term]={}; sampled_errors[name][term]={}
            for equation_index,equation in enumerate(EQUATIONS):
                delta3=references[3][state,term_index,:,equation_index]-references[7][state,term_index,:,equation_index]
                delta5=references[5][state,term_index,:,equation_index]-references[7][state,term_index,:,equation_index]
                weight=controls_volume[7]
                rms=lambda q: float(np.sqrt(np.sum(weight*q*q)/np.sum(weight)))
                sampled_errors[name][term][equation]={
                    "centered_q3_rms":rms(centered[state,term_index,:,equation_index]-references[3][state,term_index,:,equation_index]),
                    "U_q3_rms":rms(upwind[state,term_index,:,equation_index]-references[3][state,term_index,:,equation_index]),
                }
                budget[name][term][equation]={
                    "q3_minus_q7_sample_rms":rms(delta3),
                    "q5_minus_q7_sample_rms":rms(delta5),
                    "q3_minus_q7_max":float(np.max(np.abs(delta3))),
                    "q5_minus_q7_max":float(np.max(np.abs(delta5))),
                    "sample_only_not_global_bound":True,
                }
    trace_error=max((float(d["dirichlet_trace_error_max"]) for d in face_details),default=0.0)
    wall_faces=sum(int(d["wall_face_count"]) for d in face_details)
    normal={name:max((float(d["phi_wall_normal_gradient_max"][name]) for d in face_details),default=0.0) for name in FIELD_NAMES}
    trace_ranges={}
    for name in FIELD_NAMES:
        ranges=[d["phi_wall_trace_range"][name] for d in face_details if d["phi_wall_trace_range"][name][0] is not None]
        trace_ranges[name]=[min(x[0] for x in ranges),max(x[1] for x in ranges)] if ranges else [None,None]
    closure=float(np.max(np.abs(centered[:,0]+centered[:,1]-centered[:,2])))
    finite=all(np.all(np.isfinite(x)) for x in (candidate,q3_exact,face_correction,centered,upwind,*references.values()))
    implementation_pass=bool(finite and wall_faces>0 and trace_error<1e-9 and q3_replay<1e-8 and closure<1e-10 and normal["homogeneous_dirichlet"]>1e-3 and normal["variable_dirichlet"]>1e-3 and trace_ranges["variable_dirichlet"][1]-trace_ranges["variable_dirichlet"][0]>1e-5)
    arrays_out={"owner_ids":owners,"raw_ids":selected_raw,"face_ids":np.asarray(sorted(selected_faces)),
        "physical_volume":physical_volume,"evolution_volume":evolution_volume,
        "candidate_centered":centered,"candidate_U":upwind,"face_correction":face_correction,
        **{f"reference_q{order}":values for order,values in references.items()},
        **{f"reference_volume_q{order}":volume for order,volume in controls_volume.items()}}
    payload={"schema":PREFLIGHT_SCHEMA,"status":"complete","resolution":n,
        "passes_implementation_preflight":implementation_pass,
        "complete_owner_indices":owners,"complete_raw_cell_count":len(selected_raw),"face_count":len(selected_faces),
        "strata":strata,"reference_qualification":budget,"sampled_errors":sampled_errors,
        "checks":{"q3_reference_replay_max":q3_replay,"M_plus_R_closure_max":closure,
                  "dirichlet_trace_error_max":trace_error,"wall_face_count":wall_faces,
                  "phi_wall_normal_gradient_max":normal,"phi_wall_trace_range":trace_ranges,
                  "finite":finite},
        "preflight_plan_sha256":_sha256(output/f"N{n}.preflight.plan.json"),
        "preflight_receipt_sha256":_sha256(output/f"N{n}.preflight.parallel-receipt.json"),
        "sources":_source_identity(config)}
    payload["array_sha256"]=_array_hash(*(arrays_out[name] for name in sorted(arrays_out)))
    _write_npz(output/f"N{n}.preflight.npz",arrays_out,payload)
    _write_json(output/f"N{n}.preflight.json",payload)
    _event("preflight_complete",resolution=n,owners=len(owners),raw_cells=len(selected_raw),implementation_pass=implementation_pass)
    if not implementation_pass:
        raise ValueError("bounded implementation preflight failed; inspect saved evidence")
    return payload


def _orders(errors:Sequence[float]) -> list[float]:
    resolutions=(32,48,64)
    return [float(np.log(errors[i]/errors[i+1])/np.log(resolutions[i+1]/resolutions[i])) for i in range(2)]


def _merge(args:argparse.Namespace) -> dict[str,Any]:
    config=_config(args.config); output=_path(config,"output")
    cases=[json.loads((output/f"N{n}.json").read_text()) for n in (32,48,64)]
    preflight=[json.loads((output/f"N{n}.preflight.json").read_text()) for n in (32,48,64)]
    results={}; gates=[]
    for field in FIELD_NAMES:
        results[field]={}
        for equation in EQUATIONS:
            primary=PRIMARY_ACTION[equation]
            results[field][equation]={"primary_action":primary,"actions":{}}
            for action in ACTIONS:
                results[field][equation]["actions"][action]={}
                for term in TERMS:
                    errors=[case["statistics"][field][action][term][equation]["absolute_l2"] for case in cases]
                    orders=_orders(errors) if min(errors)>0 else [None,None]
                    entry={"errors":errors,"orders":orders,"acceptance_role":"primary" if action==primary else "diagnostic"}
                    if action==primary and not (equation=="vorticity" and term=="remainder"):
                        entry["passes_both_intervals"]=bool(all(order is not None and order>=1.8 for order in orders))
                        gates.append(entry["passes_both_intervals"])
                    results[field][equation]["actions"][action][term]=entry
    reference={}; reference_gate=[]
    for field in FIELD_NAMES:
        reference[field]={}
        for equation in EQUATIONS:
            reference[field][equation]={}
            primary=PRIMARY_ACTION[equation]
            for term in TERMS:
                budgets=[float(item["reference_qualification"][field][term][equation]["q5_minus_q7_sample_rms"]) for item in preflight]
                errors=[float(case["statistics"][field][primary][term][equation]["absolute_l2"]) for case in cases]
                structural_zero=equation=="vorticity" and term=="remainder"
                ratios=[0.0 if structural_zero else budget/max(error,1.0e-300) for budget,error in zip(budgets,errors,strict=True)]
                passes=structural_zero or all(ratio<0.1 for ratio in ratios)
                reference[field][equation][term]={"q5_minus_q7_sample_rms":budgets,
                    "fraction_of_matching_primary_global_error":ratios,
                    "passes_all_10_percent_screens":passes,"structural_zero":structural_zero,
                    "sample_only_not_global_uncertainty_bound":True}
                reference_gate.append(passes)
    payload={
        "schema":SUMMARY_SCHEMA,"status":"computation completed",
        "invariants_checked":all(case["verification"]["finite_complete_owner_coverage"]
            and case["verification"]["M_plus_R_closure_max"]<1.0e-11
            and case["verification"]["q5_reference_M_plus_R_closure_max"]<1.0e-11
            and case["verification"]["dirichlet_trace_error_max"]<1.0e-9 for case in cases),
        "global_order_pass":bool(all(gates)),
        "bounded_reference_controls_passed":bool(all(reference_gate)),
        "complete_qualification_passed":bool(all(gates) and all(reference_gate)),
        "global_accuracy_passed":bool(all(gates) and all(reference_gate)),
        "acceptance":"orders >=1.8 on both intervals for primary M/R/total components; exact-zero vorticity remainder exempt; diagnostic action never hidden in pooled score",
        "results":results,"reference_qualification":reference,"primary_action":PRIMARY_ACTION,
        "reference_rule":"global continuous exact-field q5 physical-volume; q3 candidate-cell replay retained as diagnostic",
        "candidate":config["candidate"],"scope":config["scope"],"sources":_source_identity(config),
    }
    _write_json(output/"summary.json",payload); _event("merge_complete",global_accuracy_passed=payload["global_accuracy_passed"])
    return payload


def _validate(args:argparse.Namespace) -> dict[str,Any]:
    config=_config(args.config); output=_path(config,"output"); stage=args.stage
    if stage=="preflight":
        n=int(args.resolution)
        payload=json.loads((output/f"N{n}.preflight.json").read_text())
        _arrays,archive=_load_npz(output/f"N{n}.preflight.npz",PREFLIGHT_SCHEMA)
        if (payload!=archive or payload.get("schema")!=PREFLIGHT_SCHEMA
            or not payload.get("passes_implementation_preflight")
            or payload.get("sources")!=_json(_source_identity(config))
            or payload.get("preflight_plan_sha256")!=_sha256(output/f"N{n}.preflight.plan.json")
            or payload.get("preflight_receipt_sha256")!=_sha256(output/f"N{n}.preflight.parallel-receipt.json")):
            raise ValueError("invalid preflight")
    elif stage.startswith("prepare_N"):
        n=int(stage.split("N",1)[1]); _arrays,payload=_load_npz(output/f"N{n}.prepare.npz",PREPARE_SCHEMA)
        receipt=json.loads((output/f"N{n}.observation-receipt.json").read_text())
        if (payload.get("status")!="complete" or payload.get("sources")!=_json(_source_identity(config))
            or payload.get("observation_data_sha256")!=receipt.get("data_sha256")):
            raise ValueError("invalid preparation")
    elif stage.startswith("case_N"):
        n=int(stage.split("N",1)[1]); _arrays,payload=_load_npz(output/f"N{n}.npz",SCHEMA)
        if payload!=json.loads((output/f"N{n}.json").read_text()) or payload.get("sources")!=_json(_source_identity(config)):
            raise ValueError("case metadata/source mismatch")
        check=payload.get("verification",{})
        if (payload.get("status")!="complete" or not check.get("finite_complete_owner_coverage")
            or check.get("M_plus_R_closure_max",1)>1e-11
            or check.get("q5_reference_M_plus_R_closure_max",1)>1e-11
            or check.get("dirichlet_trace_error_max",1)>1e-9):
            raise ValueError("invalid case")
    elif stage=="merge":
        payload=json.loads((output/"summary.json").read_text())
        if payload.get("schema")!=SUMMARY_SCHEMA or payload.get("status")!="computation completed" or not payload.get("invariants_checked"):
            raise ValueError("invalid merge")
    else:
        raise ValueError(f"unknown validation stage {stage}")
    return payload
