"""Standalone magnetic-field evaluators for cylindrical MAKEGRID data.

This module deliberately does not depend on the older DRBX magnetic-geometry
stack.  The component-spline implementation follows the common DESC/SIMSOPT
pattern: assemble the coil-group fields on their native ``(R, phi, Z)`` grid,
then expose a fast vectorized evaluator at arbitrary physical points.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import map_coordinates, spline_filter1d
from scipy.interpolate import RegularGridInterpolator


class BFieldEvaluator(ABC):
    """General interface for vectorized magnetic-field evaluation.

    Input arrays have shape ``(..., 3)``.  Cylindrical coordinates and vectors
    use ``(R, phi, Z)`` and ``(B_R, B_phi, B_Z)`` ordering.  Cartesian
    coordinates and vectors use ``(X, Y, Z)`` and ``(B_X, B_Y, B_Z)``.
    ``__call__`` is shorthand for Cartesian evaluation.
    """

    @property
    @abstractmethod
    def R(self) -> np.ndarray:
        """Return the source radial grid."""

    @property
    @abstractmethod
    def phi(self) -> np.ndarray:
        """Return the source toroidal-angle grid over one field period."""

    @property
    @abstractmethod
    def Z(self) -> np.ndarray:
        """Return the source vertical grid."""

    @property
    @abstractmethod
    def nfp(self) -> int:
        """Return the number of field periods."""

    @property
    @abstractmethod
    def period(self) -> float:
        """Return the physical toroidal period in radians."""

    @property
    @abstractmethod
    def currents(self) -> np.ndarray:
        """Return the coil-group multipliers used to assemble the field."""

    @abstractmethod
    def evaluate_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        """Evaluate cylindrical field components at cylindrical points."""

    @abstractmethod
    def evaluate_cartesian(self, points_xyz: Any) -> np.ndarray:
        """Evaluate Cartesian field components at Cartesian points."""

    def __call__(self, points_xyz: Any) -> np.ndarray:
        """Evaluate Cartesian field components at Cartesian points."""

        return self.evaluate_cartesian(points_xyz)


class ComponentSplineBFieldEvaluator(BFieldEvaluator):
    """Component-wise spline evaluator on a uniform cylindrical grid.

    Parameters
    ----------
    R, phi, Z:
        Strictly increasing one-dimensional source axes. ``phi`` must contain
        one field period without a duplicate endpoint.
    field_cylindrical:
        Combined field values with shape ``(nphi, nZ, nR, 3)``.
    nfp:
        Number of toroidal field periods.
    currents:
        Coil-group multipliers already used to assemble
        ``field_cylindrical``. They are retained as provenance metadata.
    method:
        ``"linear"`` or ``"cubic"``. Cubic uses SciPy's local legacy
        tensor-product cubic evaluator, avoiding the prohibitively large
        global coefficient solve of ``RegularGridInterpolator(method="cubic")``
        for production MAKEGRID files.
    extrapolate:
        If false, queries outside the R/Z source box raise ``ValueError``.
        Toroidal angles are always wrapped periodically.
    """

    def __init__(
        self,
        R: Any,
        phi: Any,
        Z: Any,
        field_cylindrical: Any,
        *,
        nfp: int = 1,
        currents: Any = None,
        method: str = "cubic",
        extrapolate: bool = False,
    ) -> None:
        self._R = _axis(R, "R")
        self._phi = _axis(phi, "phi")
        self._Z = _axis(Z, "Z")
        if self._R[0] <= 0:
            raise ValueError("R source grid must be strictly positive")
        if int(nfp) != nfp or int(nfp) < 1:
            raise ValueError("nfp must be a positive integer")
        if method not in {"linear", "cubic"}:
            raise ValueError("method must be 'linear' or 'cubic'")

        minimum = 2 if method == "linear" else 4
        if min(len(self._R), len(self._phi), len(self._Z)) < minimum:
            raise ValueError(
                f"{method} interpolation requires at least {minimum} points per axis"
            )

        values = np.asarray(field_cylindrical, dtype=np.float64)
        expected = (len(self._phi), len(self._Z), len(self._R), 3)
        if values.shape != expected:
            raise ValueError(f"field_cylindrical must have shape {expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError("field values must be finite")

        current_array = (
            np.ones(1, dtype=np.float64)
            if currents is None
            else np.asarray(currents, dtype=np.float64).reshape(-1)
        )
        if current_array.size == 0 or not np.all(np.isfinite(current_array)):
            raise ValueError("currents must be a non-empty finite vector")

        self._currents = current_array.copy()
        self._nfp = int(nfp)
        self._period = 2.0 * np.pi / self._nfp
        self._method = method
        self._extrapolate = bool(extrapolate)

        for axis, label in (
            (self._R, "R"),
            (self._phi, "phi"),
            (self._Z, "Z"),
        ):
            differences = np.diff(axis)
            if not np.allclose(
                differences, differences[0], rtol=2e-7, atol=2e-12
            ):
                raise ValueError(f"{label} source grid must be uniformly spaced")
        dphi = np.diff(self._phi)
        if not np.isclose(
            self._phi[-1] - self._phi[0] + dphi[0],
            self._period,
            rtol=2e-7,
            atol=2e-12,
        ):
            raise ValueError(
                "phi grid must contain one field period without a duplicate endpoint"
            )

        self._dR = self._R[1] - self._R[0]
        self._dphi = self._phi[1] - self._phi[0]
        self._dZ = self._Z[1] - self._Z[0]
        self._pad = 1 if method == "linear" else 4
        self._interpolators: tuple[RegularGridInterpolator, ...] | None = None
        self._coefficients: tuple[np.ndarray, ...] | None = None

        if self._extrapolate:
            # Extrapolation is uncommon in mesh generation. Keep the simpler
            # local RGI path for that explicit opt-in case.
            phi_extended = np.concatenate(
                (
                    self._phi[-2:] - self._period,
                    self._phi,
                    self._phi[:2] + self._period,
                )
            )
            scipy_method = "linear" if method == "linear" else "cubic_legacy"
            interpolators = []
            for component in range(3):
                component_values = values[..., component]
                extended = np.concatenate(
                    (
                        component_values[-2:],
                        component_values,
                        component_values[:2],
                    ),
                    axis=0,
                )
                interpolators.append(
                    RegularGridInterpolator(
                        (self._R, self._Z, phi_extended),
                        np.transpose(extended, (2, 1, 0)),
                        method=scipy_method,
                        bounds_error=False,
                        fill_value=None,
                    )
                )
            self._interpolators = tuple(interpolators)
        else:
            # The default fast path prefilters each component once. Toroidal
            # filtering is exactly periodic, while reflected R/Z ghost cells
            # provide nonperiodic boundary support without a global sparse
            # spline solve.
            coefficients = []
            for component in range(3):
                padded = np.pad(
                    values[..., component],
                    ((0, 0), (self._pad, self._pad), (self._pad, self._pad)),
                    mode="reflect",
                )
                if method == "cubic":
                    padded = spline_filter1d(
                        padded, order=3, axis=0, mode="grid-wrap"
                    )
                    padded = spline_filter1d(
                        padded, order=3, axis=1, mode="mirror"
                    )
                    padded = spline_filter1d(
                        padded, order=3, axis=2, mode="mirror"
                    )
                coefficients.append(padded)
            self._coefficients = tuple(coefficients)

    @property
    def R(self) -> np.ndarray:
        return self._R.copy()

    @property
    def phi(self) -> np.ndarray:
        return self._phi.copy()

    @property
    def Z(self) -> np.ndarray:
        return self._Z.copy()

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def period(self) -> float:
        return self._period

    @property
    def currents(self) -> np.ndarray:
        return self._currents.copy()

    @property
    def method(self) -> str:
        return self._method

    @property
    def extrapolate(self) -> bool:
        return self._extrapolate

    def evaluate_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        points, leading_shape = _points(points_rphiz, "points_rphiz")
        if np.any(points[:, 0] <= 0):
            raise ValueError("cylindrical query points must have R > 0")

        query = points[:, (0, 2, 1)].copy()
        query[:, 2] = self._phi[0] + np.mod(
            query[:, 2] - self._phi[0], self._period
        )
        if self._interpolators is not None:
            result = np.column_stack(
                [interpolator(query) for interpolator in self._interpolators]
            )
        else:
            if np.any(query[:, 0] < self._R[0]) or np.any(
                query[:, 0] > self._R[-1]
            ):
                raise ValueError("R query lies outside the source grid")
            if np.any(query[:, 1] < self._Z[0]) or np.any(
                query[:, 1] > self._Z[-1]
            ):
                raise ValueError("Z query lies outside the source grid")
            coordinates = np.vstack(
                (
                    (query[:, 2] - self._phi[0]) / self._dphi,
                    (query[:, 1] - self._Z[0]) / self._dZ + self._pad,
                    (query[:, 0] - self._R[0]) / self._dR + self._pad,
                )
            )
            order = 1 if self._method == "linear" else 3
            result = np.column_stack(
                [
                    map_coordinates(
                        coefficients,
                        coordinates,
                        order=order,
                        mode="grid-wrap",
                        prefilter=False,
                    )
                    for coefficients in self._coefficients
                ]
            )
        return result.reshape(leading_shape + (3,))

    def evaluate_cartesian(self, points_xyz: Any) -> np.ndarray:
        points, leading_shape = _points(points_xyz, "points_xyz")
        radius = np.hypot(points[:, 0], points[:, 1])
        if np.any(radius <= 0):
            raise ValueError("Cartesian query points must have R > 0")

        phi = np.arctan2(points[:, 1], points[:, 0])
        cylindrical_points = np.column_stack((radius, phi, points[:, 2]))
        field = self.evaluate_cylindrical(cylindrical_points).reshape((-1, 3))
        cosine = np.cos(phi)
        sine = np.sin(phi)
        result = np.empty_like(field)
        result[:, 0] = field[:, 0] * cosine - field[:, 1] * sine
        result[:, 1] = field[:, 0] * sine + field[:, 1] * cosine
        result[:, 2] = field[:, 2]
        return result.reshape(leading_shape + (3,))


def bfield_evaluator_from_makegrid(
    path: str | Path,
    *,
    currents: Any = None,
    method: str = "cubic",
    extrapolate: bool = False,
) -> BFieldEvaluator:
    """Construct a component-spline evaluator from a MAKEGRID NetCDF file.

    MAKEGRID stores one ``br_###``, ``bp_###`` and ``bz_###`` array per coil
    group.  The evaluator assembles

    ``B = sum(currents[group] * B_group)``.

    Explicit ``currents`` take precedence.  If omitted, a finite
    ``raw_coil_cur`` vector with the correct length is used; if that variable
    is absent, unit multipliers are used.  Callers should pass the physical
    current configuration explicitly when ``raw_coil_cur`` is only generator
    metadata or a sign convention.

    Arrays are read and accumulated one component/group at a time, so all
    source coil-group arrays are never retained simultaneously.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with Dataset(str(path), "r") as dataset:
        nfp = _scalar_int(dataset, "nfp")
        nr = _scalar_int(dataset, "ir")
        nz = _scalar_int(dataset, "jz")
        nphi = _scalar_int(dataset, "kp")
        group_count = _scalar_int(dataset, "nextcur")

        R = _coordinate_axis(
            dataset,
            ("rad", "R", "r"),
            nr,
            np.linspace(_scalar(dataset, "rmin"), _scalar(dataset, "rmax"), nr),
            "R",
        )
        Z = _coordinate_axis(
            dataset,
            ("zee", "Z", "z"),
            nz,
            np.linspace(_scalar(dataset, "zmin"), _scalar(dataset, "zmax"), nz),
            "Z",
        )
        period = 2.0 * np.pi / nfp
        phi = _coordinate_axis(
            dataset,
            ("phi",),
            nphi,
            np.arange(nphi, dtype=np.float64) * period / nphi,
            "phi",
        )

        group_suffixes = _group_suffixes(dataset)
        if not group_suffixes:
            raise ValueError("No complete br_###/bp_###/bz_### groups found")
        if len(group_suffixes) != group_count:
            raise ValueError(
                "MAKEGRID nextcur does not match the number of complete field groups"
            )

        if currents is None:
            raw_currents = _optional_vector(dataset, "raw_coil_cur")
            if (
                raw_currents is not None
                and raw_currents.size == group_count
                and np.all(np.isfinite(raw_currents))
            ):
                current_array = raw_currents
            else:
                current_array = np.ones(group_count, dtype=np.float64)
        else:
            current_array = np.asarray(currents, dtype=np.float64).reshape(-1)
        if current_array.size != group_count or not np.all(np.isfinite(current_array)):
            raise ValueError(
                f"currents must contain {group_count} finite coil-group values"
            )

        combined = np.zeros((nphi, nz, nr, 3), dtype=np.float64)
        for suffix, weight in zip(group_suffixes, current_array):
            for component_index, prefix in enumerate(("br", "bp", "bz")):
                variable = dataset.variables[f"{prefix}_{suffix}"]
                combined[..., component_index] += weight * _component(
                    variable, nr, nz, nphi, f"{prefix}_{suffix}"
                )

    return ComponentSplineBFieldEvaluator(
        R,
        phi,
        Z,
        combined,
        nfp=nfp,
        currents=current_array,
        method=method,
        extrapolate=extrapolate,
    )


def _axis(values: Any, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size == 0 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} source grid must be finite and non-empty")
    if np.any(np.diff(axis) <= 0):
        raise ValueError(f"{name} source grid must be strictly increasing")
    return axis


def _points(values: Any, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1:] != (3,):
        raise ValueError(f"{name} must have shape (..., 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points.reshape((-1, 3)), points.shape[:-1]


def _scalar(dataset: Dataset, name: str) -> float:
    if name in dataset.variables:
        value = np.asarray(dataset.variables[name][:]).reshape(-1)
    elif name in dataset.ncattrs():
        value = np.asarray(getattr(dataset, name)).reshape(-1)
    else:
        raise ValueError(f"MAKEGRID file is missing {name}")
    if value.size != 1 or not np.isfinite(value[0]):
        raise ValueError(f"MAKEGRID {name} must be one finite scalar")
    return float(value[0])


def _scalar_int(dataset: Dataset, name: str) -> int:
    value = _scalar(dataset, name)
    if value < 1 or not value.is_integer():
        raise ValueError(f"MAKEGRID {name} must be a positive integer")
    return int(value)


def _coordinate_axis(
    dataset: Dataset,
    candidate_names: tuple[str, ...],
    count: int,
    fallback: np.ndarray,
    label: str,
) -> np.ndarray:
    for name in candidate_names:
        if name not in dataset.variables:
            continue
        value = np.asarray(dataset.variables[name][:], dtype=np.float64)
        if value.ndim == 1 and value.size == count:
            return _axis(value, label)
    return _axis(fallback, label)


def _group_suffixes(dataset: Dataset) -> list[str]:
    return sorted(
        name[3:]
        for name in dataset.variables
        if name.startswith("br_")
        and f"bp_{name[3:]}" in dataset.variables
        and f"bz_{name[3:]}" in dataset.variables
    )


def _optional_vector(dataset: Dataset, name: str) -> np.ndarray | None:
    if name not in dataset.variables:
        return None
    value = np.asarray(dataset.variables[name][:], dtype=np.float64).reshape(-1)
    return value if value.size else None


def _component(
    variable: Any,
    nr: int,
    nz: int,
    nphi: int,
    label: str,
) -> np.ndarray:
    data = np.asarray(variable[:], dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"{label} must be three-dimensional")

    aliases = {
        "r": {"r", "ir", "rad", "radius", "nr"},
        "z": {"z", "jz", "zee", "vertical", "nz"},
        "phi": {"phi", "kp", "tor", "toroidal", "nphi"},
    }
    found: dict[str, int] = {}
    for index, dimension in enumerate(variable.dimensions):
        matches = [
            axis_name
            for axis_name, names in aliases.items()
            if dimension.lower() in names
        ]
        if len(matches) == 1:
            found[matches[0]] = index

    if len(found) != 3:
        if data.shape != (nphi, nz, nr):
            raise ValueError(
                f"{label} dimensions cannot be mapped to (phi, Z, R)"
            )
        found = {"phi": 0, "z": 1, "r": 2}
    if len(set(found.values())) != 3:
        raise ValueError(f"{label} has ambiguous dimension names")

    result = np.transpose(data, (found["phi"], found["z"], found["r"]))
    if result.shape != (nphi, nz, nr):
        raise ValueError(
            f"{label} maps to shape {result.shape}, expected {(nphi, nz, nr)}"
        )
    return result


__all__ = [
    "BFieldEvaluator",
    "ComponentSplineBFieldEvaluator",
    "bfield_evaluator_from_makegrid",
]
