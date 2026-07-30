import io
from pathlib import Path

import numpy as np
import pytest

from drbx.geometry.WallEvaluator import WallEvaluator, parse_kisslinger


def text_wall(degrees=False):
    nt, npol, nfp = 9, 17, 2
    ph = np.linspace(0, 2*np.pi/nfp, nt)
    th = np.linspace(0, 2*np.pi, npol)
    rows=[]
    for p in ph:
        R=3.0+0.35*np.cos(th)+0.1*np.cos(2*th+nfp*p)
        Z=0.8*np.sin(th)+0.08*np.sin(2*th-nfp*p)
        rows.append((np.degrees(p) if degrees else p, np.stack((R,Z),axis=-1)))
    lines=[f"{nt} {npol} {nfp}"]
    for p,rz in rows: lines.append(" ".join([str(p), *[str(x) for x in rz.ravel()]]))
    return "\n".join(lines)


def test_parse_degree_detection_and_strict_closures():
    data=parse_kisslinger(io.StringIO(text_wall(degrees=True)))
    assert data["nfp"] == 2
    np.testing.assert_allclose(data["phi"][-1], np.pi)
    with pytest.raises(ValueError): parse_kisslinger(text_wall().replace("9 17 2", "8 17 2"))
    bad=text_wall().replace("3.45 0.0", "3.45 0.1", 1)
    with pytest.raises(ValueError): parse_kisslinger(bad)


def test_periodic_broadcast_evaluation_and_derivatives():
    w=WallEvaluator(text_wall())
    assert w.phi0 == pytest.approx(0.0)
    assert w.plane_count == 9
    assert w.contour_point_count == 17
    assert w.R_bounds[0] < w.R_bounds[1]
    assert w.Z_bounds[0] < w.Z_bounds[1]
    ph=np.array([[0.1],[0.4]])
    th=np.array([[0.2,1.2,2.5]])
    values=w.evaluate(ph,th,derivatives=True)
    assert all(x.shape==(2,3) for x in values)
    np.testing.assert_allclose(w.evaluate(ph,th+2*np.pi)[0], values[0], atol=1e-10)
    np.testing.assert_allclose(w.evaluate(ph+w.period,th)[0], values[0], atol=1e-10)
    dtheta, dphi = w.derivatives_cylindrical(ph, th)
    assert dtheta.shape == dphi.shape == (2, 3, 3)
    np.testing.assert_allclose(dtheta[..., 1], 0.0)
    np.testing.assert_allclose(dphi[..., 1], 1.0)
    center, zcenter, dcenter, dzcenter = w.centerline(ph[:, 0], derivatives=True)
    assert center.shape == zcenter.shape == dcenter.shape == dzcenter.shape == (2,)
    axis_r, axis_z, axis_dr, axis_dz = w.reference_axis(ph[:, 0])
    assert axis_r.shape == axis_z.shape == axis_dr.shape == axis_dz.shape == (2,)


def test_value_only_evaluation_does_not_request_spline_derivatives():
    wall = WallEvaluator(text_wall())
    calls = []

    class SpySpline:
        def __init__(self, spline):
            self.spline = spline

        def ev(self, *args, **kwargs):
            calls.append(kwargs.copy())
            return self.spline.ev(*args, **kwargs)

    wall._interpolation = tuple(SpySpline(spline) for spline in wall.interpolation)
    phi = np.array([0.1, 0.4])
    theta = np.array([0.2, 1.2])

    wall.cartesian(phi, theta)
    assert len(calls) == 2
    assert all(not kwargs for kwargs in calls)

    calls.clear()
    wall.derivatives_cylindrical(phi, theta)
    assert len(calls) == 6
    assert sum(bool(kwargs) for kwargs in calls) == 4


def test_contains_projection_and_closed_curves():
    w = WallEvaluator(text_wall())
    inside = w.cartesian(0.2, 0.0)
    inside[:2] *= 3.0 / np.hypot(inside[0], inside[1])
    outside = inside.copy()
    outside[:2] *= 1.4
    assert bool(w.contains_cartesian(inside))
    assert not bool(w.contains_cartesian(outside))

    phi = np.array([[0.0, 0.2, 0.4], [0.0, 0.2, 0.4]])
    radius = np.array([[3.0, 2.4, 3.0], [3.0, 2.4, 3.0]])
    height = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    points_rphiz = np.stack(
        (radius, phi, height), axis=-1
    )
    contained = w.contains_cylindrical(points_rphiz)
    assert contained.shape == (2, 3)
    np.testing.assert_array_equal(contained, [[True, False, False], [True, False, False]])

    projected = w.project(inside * 0.98)
    assert np.linalg.norm(projected - inside) < 3e-1
    assert np.allclose(w.wall_boundary_curve(0.2)[0], w.wall_boundary_curve(0.2)[-1])

    class Eta:
        nfp = 2
        period = np.pi

        def evaluate_cylindrical(self, points, *, wrapped=False):
            points = np.asarray(points)
            values = points[..., 1] + 0.2 * points[..., 2]
            return np.mod(values, self.period) if wrapped else values

    curve = w.constant_eta_boundary_curve(Eta(), 0.0, npoints=64)
    assert np.all(np.isfinite(curve)) and np.allclose(curve[0], curve[-1])
    curve_rphiz = np.stack(
        (
            np.hypot(curve[:, 0], curve[:, 1]),
            np.arctan2(curve[:, 1], curve[:, 0]),
            curve[:, 2],
        ),
        axis=-1,
    )
    residual = Eta().evaluate_cylindrical(curve_rphiz)
    assert np.max(np.abs(residual)) < 1e-8
    assert np.min(curve_rphiz[:, 1]) < 0.0 < np.max(curve_rphiz[:, 1])

    with pytest.raises(ValueError, match="nfp"):
        w.constant_eta_boundary_curve(type("BadEta", (), {"nfp": 3, "period": 2*np.pi/3})(), 0.0)
    with pytest.raises(ValueError, match="period"):
        w.constant_eta_boundary_curve(type("BadEta", (), {"nfp": 2, "period": 2.0})(), 0.0)


def test_real_hsx_vessel_when_available():
    path = Path("/Users/yxie/Desktop/HSX drbx/vessel_hsx_flare.txt")
    if not path.exists():
        pytest.skip("real HSX vessel file is not available")
    wall = WallEvaluator(path)
    assert wall.nfp == 4
    assert wall.phi.size == 161
    assert wall.raw["RZ"].shape == (161, 201, 2)
    np.testing.assert_allclose(wall.raw["RZ"][0], wall.raw["RZ"][-1], atol=2e-6)
    np.testing.assert_allclose(
        wall.evaluate(wall.phi[0], 0.25),
        wall.evaluate(wall.phi[0] + wall.period, 0.25),
        atol=1e-8,
    )
    p0 = wall.cartesian(wall.phi[0], 0.25)
    p1 = wall.cartesian(wall.phi[0] + wall.period, 0.25)
    rotation = np.array(
        [
            [np.cos(wall.period), -np.sin(wall.period)],
            [np.sin(wall.period), np.cos(wall.period)],
        ]
    )
    np.testing.assert_allclose(p1[:2], rotation @ p0[:2], atol=1e-8)


def test_malformed_tokens_trailing_and_toroidal_endpoint():
    with pytest.raises(ValueError, match="nonnumeric"):
        parse_kisslinger(text_wall() + " nope")
    with pytest.raises(ValueError, match="expected"):
        parse_kisslinger(text_wall() + " 1")
    bad = text_wall()
    lines = bad.splitlines()
    last = lines[-1].split()
    last[3] = str(float(last[3]) + 0.01)
    lines[-1] = " ".join(last)
    with pytest.raises(ValueError, match="toroidal endpoint"):
        parse_kisslinger("\n".join(lines))


@pytest.mark.skipif(
    not Path("/Users/yxie/Desktop/HSX drbx/vessel_hsx_flare.txt").exists(),
    reason="real HSX vessel fixture unavailable",
)
def test_real_hsx_flare_vessel_loads_and_evaluates():
    w=WallEvaluator.from_file("/Users/yxie/Desktop/HSX drbx/vessel_hsx_flare.txt")
    assert w.nfp == 4 and w.raw["RZ"].shape == (161,201,2)
    points=w.cartesian(w.phi[[0, 40, -1]], np.array([0.0,1.0,2.0]))
    assert np.all(np.isfinite(points))
