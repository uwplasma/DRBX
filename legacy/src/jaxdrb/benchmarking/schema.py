from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

QE = 1.602176634e-19
MP = 1.67262192369e-27


def _as_np(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


@dataclass(frozen=True)
class BenchmarkNormalization:
    """Normalization metadata shared by Hermes and jax_drb benchmark bundles."""

    Nnorm: float
    Tnorm_eV: float
    Bnorm_T: float
    m_i_amu: float = 2.0
    Z_i: float = 1.0

    @property
    def cs0_m_s(self) -> float:
        return float(np.sqrt(QE * self.Tnorm_eV / (self.m_i_amu * MP)))

    @property
    def omega_ci_s(self) -> float:
        return float((self.Z_i * QE * self.Bnorm_T) / (self.m_i_amu * MP))

    @property
    def rho_s0_m(self) -> float:
        return float(self.cs0_m_s / max(self.omega_ci_s, 1e-30))

    def to_json(self) -> str:
        obj = {
            "Nnorm": float(self.Nnorm),
            "Tnorm_eV": float(self.Tnorm_eV),
            "Bnorm_T": float(self.Bnorm_T),
            "m_i_amu": float(self.m_i_amu),
            "Z_i": float(self.Z_i),
            "cs0_m_s": float(self.cs0_m_s),
            "omega_ci_s": float(self.omega_ci_s),
            "rho_s0_m": float(self.rho_s0_m),
        }
        return json.dumps(obj, sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "BenchmarkNormalization":
        obj = json.loads(s)
        return BenchmarkNormalization(
            Nnorm=float(obj["Nnorm"]),
            Tnorm_eV=float(obj["Tnorm_eV"]),
            Bnorm_T=float(obj["Bnorm_T"]),
            m_i_amu=float(obj.get("m_i_amu", 2.0)),
            Z_i=float(obj.get("Z_i", 1.0)),
        )


@dataclass
class BenchmarkBundle:
    """Portable benchmark container used by side-by-side tools and CI checks."""

    code: str
    geometry: str
    normalization: BenchmarkNormalization
    times_norm: np.ndarray
    times_si: np.ndarray
    axes: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    snapshots: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_defaults(self) -> "BenchmarkBundle":
        self.times_norm = _as_np(self.times_norm)
        self.times_si = _as_np(self.times_si)
        self.axes = {str(k): _as_np(v) for k, v in self.axes.items()}
        self.diagnostics = {str(k): _as_np(v) for k, v in self.diagnostics.items()}
        self.snapshots = {str(k): _as_np(v) for k, v in self.snapshots.items()}
        return self


def save_bundle_npz(bundle: BenchmarkBundle, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "meta_code": bundle.code,
        "meta_geometry": bundle.geometry,
        "meta_normalization_json": bundle.normalization.to_json(),
        "meta_extra_json": json.dumps(bundle.metadata, sort_keys=True),
        "times_norm": _as_np(bundle.times_norm),
        "times_si": _as_np(bundle.times_si),
    }
    for k, v in bundle.axes.items():
        payload[f"axis__{k}"] = _as_np(v)
    for k, v in bundle.diagnostics.items():
        payload[f"diag__{k}"] = _as_np(v)
    for k, v in bundle.snapshots.items():
        payload[f"snap__{k}"] = _as_np(v)
    np.savez(out, **payload)
    return out


def load_bundle_npz(path: str | Path) -> BenchmarkBundle:
    data = np.load(path, allow_pickle=True)
    code = str(data["meta_code"])
    geometry = str(data["meta_geometry"])
    normalization = BenchmarkNormalization.from_json(str(data["meta_normalization_json"]))
    metadata = json.loads(str(data["meta_extra_json"]))
    axes: dict[str, np.ndarray] = {}
    diagnostics: dict[str, np.ndarray] = {}
    snapshots: dict[str, np.ndarray] = {}
    for key in data.files:
        if key.startswith("axis__"):
            axes[key.split("__", 1)[1]] = _as_np(data[key])
        elif key.startswith("diag__"):
            diagnostics[key.split("__", 1)[1]] = _as_np(data[key])
        elif key.startswith("snap__"):
            snapshots[key.split("__", 1)[1]] = _as_np(data[key])
    return BenchmarkBundle(
        code=code,
        geometry=geometry,
        normalization=normalization,
        times_norm=_as_np(data["times_norm"]),
        times_si=_as_np(data["times_si"]),
        axes=axes,
        diagnostics=diagnostics,
        snapshots=snapshots,
        metadata=metadata,
    ).with_defaults()
