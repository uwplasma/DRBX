from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import BenchmarkBundle


def _as_np(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _relative_l2(reference: np.ndarray, candidate: np.ndarray, *, eps: float = 1.0e-30) -> float:
    ref = _as_np(reference).reshape(-1)
    cmp = _as_np(candidate).reshape(-1)
    if ref.shape != cmp.shape:
        raise ValueError(f"Shape mismatch in relative_l2: {ref.shape} != {cmp.shape}")
    denom = float(np.linalg.norm(ref))
    numer = float(np.linalg.norm(cmp - ref))
    return numer / max(denom, eps)


def _interp1(
    reference_axis: np.ndarray, candidate_axis: np.ndarray, candidate_values: np.ndarray
) -> np.ndarray:
    ref_x = _as_np(reference_axis).reshape(-1)
    cmp_x = _as_np(candidate_axis).reshape(-1)
    cmp_y = _as_np(candidate_values).reshape(-1)
    if cmp_x.size != cmp_y.size:
        raise ValueError(f"Axis/value size mismatch: {cmp_x.size} != {cmp_y.size}")
    if ref_x.size == 0:
        return np.zeros_like(ref_x)
    if cmp_x.size == 1:
        return np.full_like(ref_x, float(cmp_y[0]))
    return np.interp(ref_x, cmp_x, cmp_y)


def _axis_key_for_diagnostic(bundle: BenchmarkBundle, key: str) -> str | None:
    if key.startswith("rms_") and key in bundle.diagnostics:
        return "__times_norm__"
    if key == "psd_n_f" and "freq_hz" in bundle.diagnostics:
        return "freq_hz"
    if key == "psd_n_ky" and "ky_m-1" in bundle.diagnostics:
        return "ky_m-1"
    if key in ("coh_n_phi", "phase_n_phi") and "coh_freq_hz" in bundle.diagnostics:
        return "coh_freq_hz"
    if key == "pdf_n_y" and "pdf_n_x" in bundle.diagnostics:
        return "pdf_n_x"
    if key == "pdf_Te_y" and "pdf_Te_x" in bundle.diagnostics:
        return "pdf_Te_x"
    if key == "gamma_r_profile" and "x_index" in bundle.axes:
        return "x_index"
    return None


def _lookup_axis(bundle: BenchmarkBundle, axis_key: str) -> np.ndarray:
    if axis_key == "__times_norm__":
        return _as_np(bundle.times_norm)
    if axis_key in bundle.diagnostics:
        return _as_np(bundle.diagnostics[axis_key])
    if axis_key in bundle.axes:
        return _as_np(bundle.axes[axis_key])
    raise KeyError(f"Missing axis '{axis_key}' in diagnostics and axes.")


def _aligned_candidate_values(
    reference: BenchmarkBundle,
    candidate: BenchmarkBundle,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    ref_values = _as_np(reference.diagnostics[key])
    cmp_values = _as_np(candidate.diagnostics[key])
    if ref_values.shape == cmp_values.shape:
        return ref_values, cmp_values
    if ref_values.ndim != 1 or cmp_values.ndim != 1:
        raise ValueError(
            f"Cannot align non-1D diagnostic '{key}' with shapes {ref_values.shape} and {cmp_values.shape}"
        )
    axis_key = _axis_key_for_diagnostic(reference, key)
    if axis_key is not None:
        return ref_values, _interp1(
            _lookup_axis(reference, axis_key),
            _lookup_axis(candidate, axis_key),
            cmp_values,
        )
    raise ValueError(
        f"Cannot align diagnostic '{key}' with shapes {ref_values.shape} and {cmp_values.shape}"
    )


@dataclass(frozen=True)
class DiagnosticComparison:
    per_key_rel_l2: dict[str, float]
    mean_rel_l2: float
    max_rel_l2: float


def compare_bundle_diagnostics(
    reference: BenchmarkBundle,
    candidate: BenchmarkBundle,
    *,
    keys: tuple[str, ...] = (
        "rms_n_fluct",
        "rms_Te_fluct",
        "rms_omega_fluct",
        "rms_phi_fluct",
        "psd_n_f",
        "psd_n_ky",
    ),
) -> DiagnosticComparison:
    per_key: dict[str, float] = {}
    for key in keys:
        if key not in reference.diagnostics or key not in candidate.diagnostics:
            continue
        ref_values, cmp_values = _aligned_candidate_values(reference, candidate, key)
        per_key[key] = _relative_l2(ref_values, cmp_values)
    if not per_key:
        raise ValueError("No comparable diagnostics were found.")
    vals = np.asarray(tuple(per_key.values()), dtype=np.float64)
    return DiagnosticComparison(
        per_key_rel_l2=per_key,
        mean_rel_l2=float(np.mean(vals)),
        max_rel_l2=float(np.max(vals)),
    )
