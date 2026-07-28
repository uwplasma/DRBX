"""Tests and a direct-MAKEGRID comparison CLI for ``Bfield_evaluator``."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from drbx.geometry.Bfield_evaluator import bfield_evaluator_from_makegrid


pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated in NumPy 2.5"
    ":DeprecationWarning"
)


def _scalar(dataset: Dataset, name: str) -> float:
    return float(np.asarray(dataset.variables[name][:]).squeeze())


def _coordinate_or_uniform(
    dataset: Dataset,
    name: str,
    count: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    if name in dataset.variables:
        candidate = np.asarray(dataset.variables[name][:], dtype=np.float64)
        if candidate.ndim == 1 and candidate.size == count:
            return candidate
    return np.linspace(lower, upper, count)


def _component_in_phizr_order(
    variable: object,
    nphi: int,
    nz: int,
    nr: int,
) -> np.ndarray:
    data = np.asarray(variable[:], dtype=np.float64)
    expected_names = {"phi": "phi", "zee": "z", "rad": "r"}
    mapped = {
        expected_names[name.lower()]: index
        for index, name in enumerate(variable.dimensions)
        if name.lower() in expected_names
    }
    if len(mapped) == 3:
        data = np.transpose(data, (mapped["phi"], mapped["z"], mapped["r"]))
    if data.shape != (nphi, nz, nr):
        raise ValueError(
            f"reference component has shape {data.shape}, "
            f"expected {(nphi, nz, nr)}"
        )
    return data


def _reference(path: Path, currents: np.ndarray | None = None) -> dict[str, object]:
    """Read and assemble MAKEGRID samples independently of the evaluator."""

    with Dataset(str(path), "r") as dataset:
        nr = int(_scalar(dataset, "ir"))
        nz = int(_scalar(dataset, "jz"))
        nphi = int(_scalar(dataset, "kp"))
        nfp = int(_scalar(dataset, "nfp"))
        R = _coordinate_or_uniform(
            dataset,
            "rad",
            nr,
            _scalar(dataset, "rmin"),
            _scalar(dataset, "rmax"),
        )
        Z = _coordinate_or_uniform(
            dataset,
            "zee",
            nz,
            _scalar(dataset, "zmin"),
            _scalar(dataset, "zmax"),
        )
        period = 2.0 * np.pi / nfp
        if "phi" in dataset.variables:
            phi = np.asarray(dataset.variables["phi"][:], dtype=np.float64)
        else:
            phi = np.arange(nphi, dtype=np.float64) * period / nphi

        groups = sorted(
            match.group(1)
            for name in dataset.variables
            if (match := re.fullmatch(r"br_(\d+)", name))
            and f"bp_{match.group(1)}" in dataset.variables
            and f"bz_{match.group(1)}" in dataset.variables
        )
        if currents is None:
            if "raw_coil_cur" in dataset.variables:
                weights = np.asarray(
                    dataset.variables["raw_coil_cur"][:], dtype=np.float64
                ).reshape(-1)
            else:
                weights = np.ones(len(groups), dtype=np.float64)
        else:
            weights = np.asarray(currents, dtype=np.float64).reshape(-1)
        if weights.size != len(groups):
            raise ValueError("current/group count mismatch")

        fields = []
        for prefix in ("br", "bp", "bz"):
            total = np.zeros((nphi, nz, nr), dtype=np.float64)
            for suffix, weight in zip(groups, weights):
                total += weight * _component_in_phizr_order(
                    dataset.variables[f"{prefix}_{suffix}"], nphi, nz, nr
                )
            fields.append(total)

    return {
        "R": R,
        "Z": Z,
        "phi": phi,
        "BR": fields[0],
        "Bphi": fields[1],
        "BZ": fields[2],
        "currents": weights,
        "period": period,
    }


def _cylindrical_grid_points(reference: dict[str, object]) -> np.ndarray:
    phi, Z, R = np.meshgrid(
        reference["phi"], reference["Z"], reference["R"], indexing="ij"
    )
    return np.stack((R, phi, Z), axis=-1)


def _to_cartesian(points_rphiz: np.ndarray) -> np.ndarray:
    R, phi, Z = np.moveaxis(points_rphiz, -1, 0)
    return np.stack((R * np.cos(phi), R * np.sin(phi), Z), axis=-1)


def _write_synthetic(path: Path) -> dict[str, object]:
    R = np.linspace(1.3, 2.1, 7)
    Z = np.linspace(-0.55, 0.55, 8)
    nfp = 2
    period = 2.0 * np.pi / nfp
    phi = np.arange(9, dtype=np.float64) * period / 9
    RR, PP, ZZ = np.meshgrid(R, phi, Z, indexing="ij")

    group_1 = (
        0.12
        + 0.025 * (RR - 1.7)
        + 0.018 * ZZ
        + 0.007 * np.cos(2 * PP)
        + 0.011 * np.sin(PP) * np.cos(ZZ),
        0.42 + 0.03 * np.cos(PP) + 0.012 * (RR - 1.7) ** 2,
        -0.08 + 0.02 * np.sin(2 * PP) + 0.015 * ZZ,
    )
    group_2 = (
        -0.018 + 0.009 * np.cos(PP + ZZ),
        0.16 + 0.02 * np.sin(PP) + 0.01 * ZZ,
        0.035 * np.cos(2 * PP) + 0.006 * (RR - 1.7),
    )

    with Dataset(str(path), "w", format="NETCDF4") as dataset:
        for name, size in (
            ("rad", len(R)),
            ("zee", len(Z)),
            ("phi", len(phi)),
            ("external_coils", 2),
        ):
            dataset.createDimension(name, size)
        for name, dtype, value in (
            ("ir", "i4", len(R)),
            ("jz", "i4", len(Z)),
            ("kp", "i4", len(phi)),
            ("nfp", "i4", nfp),
            ("nextcur", "i4", 2),
            ("rmin", "f8", R[0]),
            ("rmax", "f8", R[-1]),
            ("zmin", "f8", Z[0]),
            ("zmax", "f8", Z[-1]),
        ):
            dataset.createVariable(name, dtype)[:] = value
        for name, values in (("rad", R), ("zee", Z), ("phi", phi)):
            dataset.createVariable(name, "f8", (name,))[:] = values
        dataset.createVariable(
            "raw_coil_cur", "f8", ("external_coils",)
        )[:] = [1.5, -0.75]
        for group, fields in ((1, group_1), (2, group_2)):
            for component, values in zip(("br", "bp", "bz"), fields):
                variable = dataset.createVariable(
                    f"{component}_{group:03d}",
                    "f8",
                    ("phi", "zee", "rad"),
                )
                variable[:, :, :] = np.ascontiguousarray(
                    np.transpose(values, (1, 2, 0))
                )

    return _reference(path)


@pytest.fixture
def synthetic_makegrid(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    path = tmp_path / "synthetic.mgrid.nc"
    return path, _write_synthetic(path)


def _expected_field(reference: dict[str, object]) -> np.ndarray:
    return np.stack(
        (reference["BR"], reference["Bphi"], reference["BZ"]), axis=-1
    )


def test_componentwise_spline_at_original_nodes(synthetic_makegrid):
    path, reference = synthetic_makegrid
    evaluator = bfield_evaluator_from_makegrid(
        path, currents=reference["currents"]
    )
    np.testing.assert_allclose(
        evaluator.evaluate_cylindrical(_cylindrical_grid_points(reference)),
        _expected_field(reference),
        atol=3e-7,
        rtol=3e-7,
    )


def test_default_currents_come_from_makegrid(synthetic_makegrid):
    path, reference = synthetic_makegrid
    evaluator = bfield_evaluator_from_makegrid(path)
    np.testing.assert_allclose(evaluator.currents, reference["currents"])
    np.testing.assert_allclose(
        evaluator.evaluate_cylindrical(_cylindrical_grid_points(reference)),
        _expected_field(reference),
        atol=3e-7,
        rtol=3e-7,
    )


def test_current_weighting_two_groups(synthetic_makegrid):
    path, _ = synthetic_makegrid
    reference = _reference(path, np.array([2.0, 0.0]))
    evaluator = bfield_evaluator_from_makegrid(path, currents=[2.0, 0.0])
    np.testing.assert_allclose(
        evaluator.evaluate_cylindrical(_cylindrical_grid_points(reference)),
        _expected_field(reference),
        atol=3e-7,
        rtol=3e-7,
    )


def test_periodic_cartesian_and_batch_consistency(synthetic_makegrid):
    path, reference = synthetic_makegrid
    evaluator = bfield_evaluator_from_makegrid(
        path, currents=reference["currents"]
    )
    points = _cylindrical_grid_points(reference)[1:4, 1:4, 1:4]
    field_cylindrical = evaluator.evaluate_cylindrical(points)

    shifted = points + np.array([0.0, reference["period"], 0.0])
    np.testing.assert_allclose(
        evaluator.evaluate_cylindrical(shifted),
        field_cylindrical,
        atol=3e-7,
        rtol=3e-7,
    )

    _, phi, _ = np.moveaxis(points, -1, 0)
    expected_cartesian = np.stack(
        (
            field_cylindrical[..., 0] * np.cos(phi)
            - field_cylindrical[..., 1] * np.sin(phi),
            field_cylindrical[..., 0] * np.sin(phi)
            + field_cylindrical[..., 1] * np.cos(phi),
            field_cylindrical[..., 2],
        ),
        axis=-1,
    )
    points_cartesian = _to_cartesian(points)
    np.testing.assert_allclose(
        evaluator.evaluate_cartesian(points_cartesian),
        expected_cartesian,
        atol=3e-7,
        rtol=3e-7,
    )
    np.testing.assert_allclose(
        evaluator(points_cartesian), expected_cartesian, atol=3e-7, rtol=3e-7
    )
    batch = points.reshape(-1, 3)[:5].reshape(5, 1, 3)
    assert evaluator.evaluate_cylindrical(batch).shape == (5, 1, 3)


def test_out_of_domain_behavior(synthetic_makegrid):
    path, reference = synthetic_makegrid
    evaluator = bfield_evaluator_from_makegrid(
        path, currents=reference["currents"], extrapolate=False
    )
    with pytest.raises(ValueError):
        evaluator.evaluate_cylindrical(
            [reference["R"][0] - 0.1, reference["phi"][0], reference["Z"][0]]
        )


def _cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the spline evaluator with raw MAKEGRID grid values"
    )
    parser.add_argument("makegrid", type=Path)
    parser.add_argument("--currents", nargs="+", type=float)
    parser.add_argument("--method", choices=("linear", "cubic"), default="cubic")
    parser.add_argument("--atol", type=float, default=3e-7)
    parser.add_argument("--rtol", type=float, default=3e-7)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)

    currents = (
        None if args.currents is None else np.asarray(args.currents, dtype=np.float64)
    )
    reference = _reference(args.makegrid, currents)
    evaluator = bfield_evaluator_from_makegrid(
        args.makegrid, currents=currents, method=args.method
    )
    points = _cylindrical_grid_points(reference).reshape((-1, 3))
    expected = _expected_field(reference).reshape((-1, 3))

    if args.max_points is not None:
        if args.max_points < 1:
            parser.error("--max-points must be positive")
        if args.max_points < len(points):
            indices = np.random.default_rng(args.seed).choice(
                len(points), args.max_points, replace=False
            )
            points = points[indices]
            expected = expected[indices]

    actual = np.asarray(evaluator.evaluate_cylindrical(points))
    error = np.abs(actual - expected)
    max_abs = float(np.max(error))
    scale = np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    max_rel = float(np.max(error / scale))
    passed = bool(
        np.allclose(actual, expected, atol=args.atol, rtol=args.rtol)
    )
    print(
        f"points checked: {len(points)}\n"
        f"max absolute error: {max_abs:.6e}\n"
        f"max relative error: {max_rel:.6e}\n"
        f"result: {'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())
