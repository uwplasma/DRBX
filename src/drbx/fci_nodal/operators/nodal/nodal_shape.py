"""Versioned Zernike core-ring layout shape contract; legacy constructors keep their original identity."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import numpy as np


_V1_FIELDS = ("n_r", "n_theta", "n_eta", "version", "layout_kind", "core_policy", "phase_policy")
_V2_FIELDS = _V1_FIELDS + ("field_periods", "replication_period_count")


ZERNIKE_CORE_RING_KIND = "zernike_core_ring"


@dataclass(frozen=True)
class NodalShape:
    """Independent radial, angular and toroidal counts, with explicit v2 replication.

    Physical field periods and geometric replication are distinct. Neither grid
    eligibility nor geometric replication certifies preconditioner bank reuse.
    """

    n_r: int
    n_theta: int
    n_eta: int
    version: int = 1
    layout_kind: str = "zernike_core_ring"
    core_policy: str = "radial-eighth-polar-gauss"
    phase_policy: str = "angular-cell-centre"
    field_periods: int = 4
    replication_period_count: int | None = None

    def __post_init__(self):
        for name in ("n_r", "n_theta", "n_eta", "version", "field_periods", "replication_period_count"):
            value = getattr(self, name)
            if name == "replication_period_count" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if (self.version not in (1, 2) or self.layout_kind != ZERNIKE_CORE_RING_KIND or
                self.core_policy != "radial-eighth-polar-gauss" or self.phase_policy != "angular-cell-centre"):
            raise ValueError("unsupported nodal shape version, layout_kind, core or phase policy")
        if self.n_r < 16 or self.n_r % 8:
            raise ValueError("n_r must be >= 16 and divisible by 8")
        if self.n_theta < 2 * self.p + 1:
            raise ValueError("n_theta must resolve core trace modes 1..p (n_theta >= 2*p+1)")
        if self.version == 1:
            if self.field_periods != 4 or self.replication_period_count is not None:
                raise ValueError("explicit replication or nondefault field periods require version=2")
            if self.n_eta < 4 or self.n_eta % 4:
                raise ValueError("n_eta must be positive and divisible by 4 (HSX field periods)")
        else:
            if self.replication_period_count is None:
                object.__setattr__(self, "replication_period_count", 4)
            if self.field_periods < 1 or self.replication_period_count < 1:
                raise ValueError("field_periods and replication_period_count must be positive")
            if self.field_periods % self.replication_period_count:
                raise ValueError("replication_period_count must divide field_periods")
            if self.n_eta < 4 or self.n_eta % self.replication_period_count:
                raise ValueError("n_eta must be >= 4 and divisible by replication_period_count")

    @property
    def K(self):
        return self.n_r // 8

    @property
    def p(self):
        return self.K + 2

    @property
    def delta(self):
        return np.pi / self.n_theta

    @property
    def source_planes(self):
        replication = 4 if self.version == 1 else self.replication_period_count
        return self.n_eta // replication

    @property
    def period_reuse_eligible(self):
        return self.n_eta % self.field_periods == 0

    def to_dict(self):
        base = {key: getattr(self, key) for key in _V1_FIELDS}
        quantities = dict(K=self.K, p=self.p, core_radius=1.0 / 8.0,
                          core_radial_nodes=self.p // 2 + 1, core_angles=2 * self.p + 2,
                          trace_points=self.n_theta, phase=float(self.delta))
        if self.version == 1:
            return dict(base, **quantities, planes_per_period=self.n_eta // 4, id_order="plane-major-kP+i")
        return dict(base, **quantities, field_periods=self.field_periods,
                    replication_period_count=self.replication_period_count, planes_per_replica=self.source_planes,
                    planes_per_field_period=self.n_eta // self.field_periods if self.period_reuse_eligible else None,
                    period_reuse_eligible=self.period_reuse_eligible, id_order="plane-major-kP+i")

    @classmethod
    def from_dict(cls, metadata):
        if not isinstance(metadata, dict):
            raise ValueError("incomplete nodal shape metadata")
        version = metadata.get("version")
        if isinstance(version, bool) or not isinstance(version, Integral) or version not in (1, 2):
            raise ValueError("unsupported nodal shape version")
        fields = _V1_FIELDS if version == 1 else _V2_FIELDS
        if not set(fields).issubset(metadata):
            raise ValueError("incomplete nodal shape metadata")
        shape = cls(**{key: metadata[key] for key in fields})
        canonical = shape.to_dict()
        if canonical != metadata:
            raise ValueError("inconsistent or unsupported nodal shape metadata")
        if version == 2:
            for key, value in canonical.items():
                if isinstance(value, bool):
                    valid = isinstance(metadata[key], bool)
                elif isinstance(value, int):
                    valid = isinstance(metadata[key], Integral) and not isinstance(metadata[key], bool)
                else:
                    valid = True
                if not valid:
                    raise ValueError(f"inconsistent nodal shape metadata type for {key}")
        return shape
