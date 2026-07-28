"""Smooth periodic evaluator for Kisslinger/FLARE wall files."""
from __future__ import annotations
from pathlib import Path
from types import MappingProxyType
from typing import Any, TextIO
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize


def parse_kisslinger(source: Any) -> dict[str, Any]:
    if hasattr(source, "read"):
        text = source.read()
    elif isinstance(source, (str, Path)):
        try:
            text = Path(source).read_text() if Path(source).exists() else str(source)
        except OSError:
            text = str(source)
    else:
        text = str(source)
    tokens = []
    for line in text.splitlines():
        tokens.extend(line.split("#", 1)[0].split())
    try:
        nums = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("Kisslinger data contains a nonnumeric token") from exc
    if len(nums) < 3 or any(not np.isfinite(x) or int(x) != x for x in nums[:3]):
        raise ValueError("Kisslinger header must contain three integer values")
    nt, npol, nfp = map(int, nums[:3])
    if nt < 3 or npol < 4 or nfp < 1:
        raise ValueError("invalid Kisslinger dimensions")
    expected = 3 + nt * (1 + 2 * npol)
    if len(nums) != expected:
        raise ValueError(f"expected {expected} numbers, found {len(nums)}")
    rows = np.asarray(nums[3:]).reshape(nt, 1 + 2 * npol)
    phi = rows[:, 0]
    rz = rows[:, 1:].reshape(nt, npol, 2)
    scale = np.pi / 180 if phi[-1] - phi[0] > 2 * np.pi + 1e-8 else 1.0
    phi = phi * scale
    if np.any(~np.isfinite(phi)) or np.any(~np.isfinite(rz)) or np.any(rz[..., 0] <= 0):
        raise ValueError("wall data must be finite and have positive R")
    if np.any(np.diff(phi) <= 0):
        raise ValueError("toroidal angles must be strictly increasing")
    period = phi[-1] - phi[0]
    if not np.isclose(period, 2 * np.pi / nfp, rtol=2e-6, atol=2e-8):
        raise ValueError("toroidal samples must include exactly one field period")
    if not np.allclose(rz[:, 0], rz[:, -1], rtol=2e-8, atol=2e-10):
        raise ValueError("poloidal curves must be closed")
    if not np.allclose(rz[0], rz[-1], rtol=2e-7, atol=2e-6):
        raise ValueError("toroidal endpoint must repeat the first plane")
    return {"phi": phi, "RZ": rz, "nfp": nfp, "ntoroidal": nt, "npoloidal": npol}


class WallEvaluator:
    """Evaluate a smooth, field-periodic cylindrical wall surface."""
    def __init__(self, source: Any, *, nfp: int | None = None):
        data = parse_kisslinger(source) if not isinstance(source, dict) else source
        self._phi = np.asarray(data["phi"], float).copy()
        self._rz = np.asarray(data.get("RZ", data.get("rz")), float).copy()
        self._nfp = int(data["nfp"] if nfp is None else nfp)
        if self._rz.ndim != 3 or self._rz.shape[-1] != 2 or self._rz.shape[0] != self._phi.size:
            raise ValueError("RZ must have shape (ntoroidal, npoloidal, 2)")
        if np.any(np.diff(self._phi) <= 0) or np.any(~np.isfinite(self._rz)) or np.any(self._rz[..., 0] <= 0):
            raise ValueError("invalid wall arrays")
        if not np.allclose(self._rz[:, 0], self._rz[:, -1], rtol=2e-8, atol=2e-10):
            raise ValueError("poloidal curves must be closed")
        if not np.allclose(self._rz[0], self._rz[-1], rtol=2e-7, atol=2e-6):
            raise ValueError("toroidal endpoint must repeat the first plane")
        self._period = float(self._phi[-1] - self._phi[0])
        if not np.isclose(self._period, 2*np.pi/self._nfp, rtol=2e-6, atol=2e-8):
            raise ValueError("phi and nfp are inconsistent")
        self._theta = np.linspace(0, 2*np.pi, self._rz.shape[1])
        self._raw = MappingProxyType({"phi": self._phi.copy(), "RZ": self._rz.copy(), "nfp": self._nfp})
        self._interpolation = tuple(self._make_channel(k) for k in range(2))

    def _make_channel(self, k):
        # Tile an endpoint-exclusive fundamental rectangle.  The extra tiles let
        # RectBivariateSpline evaluate arbitrary wrapped queries without creating
        # an interpolator per query.
        p = self._phi[:-1]
        t = self._theta[:-1]
        pp = np.concatenate((p - self._period, p, p + self._period))
        tt = np.concatenate((t - 2*np.pi, t, t + 2*np.pi))
        base = self._rz[:-1, :-1, k]
        values = np.tile(base, (3, 3))
        return RectBivariateSpline(pp, tt, values, kx=min(3, len(pp)-1), ky=min(3, len(tt)-1), s=0)

    @classmethod
    def from_file(cls, path: str | Path, *, nfp: int | None = None):
        return cls(path, nfp=nfp)

    @classmethod
    def from_stream(cls, stream: TextIO, *, nfp: int | None = None):
        return cls(stream, nfp=nfp)

    @classmethod
    def from_kisslinger(cls, source: Any, *, nfp: int | None = None):
        return cls(source, nfp=nfp)
    @property
    def phi(self): return self._phi.copy()
    @property
    def phi0(self): return float(self._phi[0])
    @property
    def nfp(self): return self._nfp
    @property
    def period(self): return self._period
    @property
    def plane_count(self): return int(self._phi.size)
    @property
    def contour_point_count(self): return int(self._rz.shape[1])
    @property
    def R_bounds(self): return float(np.min(self._rz[..., 0])), float(np.max(self._rz[..., 0]))
    @property
    def Z_bounds(self): return float(np.min(self._rz[..., 1])), float(np.max(self._rz[..., 1]))
    @property
    def raw(self): return self._raw
    @property
    def interpolation(self): return self._interpolation

    def _values(self, phi, theta):
        ph, th = np.broadcast_arrays(np.asarray(phi, float), np.asarray(theta, float))
        x = np.mod(ph - self._phi[0], self._period) + self._phi[0]
        u = np.mod(th, 2*np.pi)
        vals=[]; dth=[]; dph=[]
        for k in range(2):
            spline = self._interpolation[k]
            vals.append(spline.ev(x.ravel(), u.ravel()).reshape(u.shape))
            dth.append(spline.ev(x.ravel(), u.ravel(), dx=0, dy=1).reshape(u.shape))
            dph.append(spline.ev(x.ravel(), u.ravel(), dx=1, dy=0).reshape(u.shape))
        return vals[0], vals[1], dth[0], dth[1], dph[0], dph[1]

    def evaluate(self, phi, theta, *, derivatives=False):
        v = self._values(phi, theta)
        return v if derivatives else (v[0], v[1])

    evaluate_rz = evaluate

    def cylindrical(self, phi, theta):
        radius, height = self._values(phi, theta)[:2]
        wrapped_phi, _ = np.broadcast_arrays(np.asarray(phi, float), np.asarray(theta, float))
        return np.stack((radius, wrapped_phi, height), axis=-1)

    evaluate_cylindrical = cylindrical

    def derivatives_cylindrical(self, phi, theta):
        """Return derivatives of ``(R, phi, Z)`` with respect to theta and phi."""
        radius, height, dr_dtheta, dz_dtheta, dr_dphi, dz_dphi = self._values(
            phi, theta
        )
        zeros = np.zeros_like(radius)
        ones = np.ones_like(height)
        dtheta = np.stack((dr_dtheta, zeros, dz_dtheta), axis=-1)
        dphi = np.stack((dr_dphi, ones, dz_dphi), axis=-1)
        return dtheta, dphi

    def cartesian(self, phi, theta):
        radius, height = self._values(phi, theta)[:2]
        query_phi, _ = np.broadcast_arrays(np.asarray(phi, float), np.asarray(theta, float))
        return np.stack(
            (radius * np.cos(query_phi), radius * np.sin(query_phi), height),
            axis=-1,
        )

    evaluate_cartesian = cartesian

    def centerline(self, phi, *, derivatives=False, ntheta=256):
        """Return the periodic R/Z centerline, averaged over the wall contour."""
        ph = np.asarray(phi, float)
        theta = np.linspace(0.0, 2.0*np.pi, int(ntheta), endpoint=False)
        r, z, _, _, dr, dz = self._values(ph[..., None], theta)
        out = (np.mean(r, axis=-1), np.mean(z, axis=-1))
        if derivatives:
            out += (np.mean(dr, axis=-1), np.mean(dz, axis=-1))
        return out

    def reference_axis(self, phi):
        """Return ``(R, Z, dR/dphi, dZ/dphi)`` for scalar-potential fits."""
        return self.centerline(phi, derivatives=True)

    evaluate_centerline = centerline

    def _polygon(self, phi, n=512):
        theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=True)
        cylindrical = self.cylindrical(phi, theta)
        return cylindrical[..., (0, 2)]

    def contains_cylindrical(self, points_rphiz):
        """Return containment for cylindrical points shaped ``(..., 3)``.

        The point ordering is ``(R, phi, Z)``. Queries are grouped by wrapped
        toroidal angle, so a structured grid reuses one polygon per angle.
        """
        points = np.asarray(points_rphiz, dtype=float)
        if points.shape == (3,):
            points = points.reshape((1, 3))
            scalar = True
        elif points.ndim >= 1 and points.shape[-1] == 3:
            scalar = False
        else:
            raise ValueError("points_rphiz must have shape (..., 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_rphiz must be finite")

        flat = points.reshape((-1, 3))
        wrapped = np.mod(flat[:, 1] - self._phi[0], self._period) + self._phi[0]
        group_keys = np.round(wrapped, decimals=12)
        result = np.zeros(flat.shape[0], dtype=bool)
        for key in np.unique(group_keys):
            indices = np.flatnonzero(group_keys == key)
            polygon = self._polygon(float(np.mean(wrapped[indices])))
            radius = flat[indices, 0]
            height = flat[indices, 2]
            radius0 = polygon[:-1, 0]
            radius1 = polygon[1:, 0]
            height0 = polygon[:-1, 1]
            height1 = polygon[1:, 1]
            crosses = (height0[:, None] > height[None, :]) != (
                height1[:, None] > height[None, :]
            )
            intersections = radius0[:, None] + (
                height[None, :] - height0[:, None]
            ) * (radius1 - radius0)[:, None] / (
                height1 - height0
            )[:, None]
            result[indices] = np.count_nonzero(
                crosses & (radius[None, :] < intersections), axis=0
            ) % 2 == 1
        result = result.reshape(points.shape[:-1])
        return bool(result[0]) if scalar else result

    def contains_cartesian(self, points_xyz):
        points = np.asarray(points_xyz, dtype=float)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError("points_xyz must have shape (..., 3)")
        radius = np.hypot(points[..., 0], points[..., 1])
        phi = np.arctan2(points[..., 1], points[..., 0])
        cylindrical = np.stack((radius, phi, points[..., 2]), axis=-1)
        return self.contains_cylindrical(cylindrical)

    def contains(self, points_xyz):
        return self.contains_cartesian(points_xyz)
    def project(self, points):
        q = np.asarray(points, float)
        shape = q.shape[:-1]
        ans = []
        for point in np.atleast_2d(q).reshape(-1,3):
            phi = np.arctan2(point[1], point[0])
            theta = np.arctan2(
                point[2], np.hypot(point[0], point[1]) - np.mean(self._rz[..., 0])
            )
            objective = lambda values: np.sum(
                (self.cartesian(values[0], values[1]) - point) ** 2
            )
            optimum = minimize(
                objective,
                [phi, theta],
                method="Nelder-Mead",
                options={"maxiter": 300, "xatol": 1e-10},
            )
            ans.append(self.cartesian(*optimum.x))
        return np.asarray(ans).reshape(shape+(3,))

    nearest = project

    def wall_boundary_curve(self, phi, npoints=256):
        theta = np.linspace(0.0, 2.0 * np.pi, int(npoints), endpoint=True)
        return self.cartesian(phi, theta)

    wall_curve = wall_boundary_curve

    def _validate_eta_evaluator(self, eta_evaluator):
        eta_nfp = getattr(eta_evaluator, "nfp", None)
        if eta_nfp is not None and int(eta_nfp) != eta_nfp:
            raise ValueError("eta evaluator nfp must be an integer")
        if eta_nfp is not None and int(eta_nfp) != self._nfp:
            raise ValueError("eta evaluator nfp is inconsistent with the wall")
        eta_period = getattr(eta_evaluator, "period", None)
        if eta_period is not None and not np.isclose(
            float(eta_period), self._period, rtol=2e-7, atol=2e-10
        ):
            raise ValueError("eta evaluator period is inconsistent with the wall")

    def _evaluate_eta(self, eta_evaluator, phi, theta, branch_center):
        cylindrical = self.cylindrical(phi, theta)
        if hasattr(eta_evaluator, "evaluate_cylindrical"):
            try:
                values = eta_evaluator.evaluate_cylindrical(
                    cylindrical, wrapped=False
                )
            except TypeError:
                values = eta_evaluator.evaluate_cylindrical(cylindrical)
        else:
            cartesian = self.cartesian(phi, theta)
            try:
                values = eta_evaluator.evaluate_cartesian(
                    cartesian, wrapped=False
                )
            except TypeError:
                values = eta_evaluator.evaluate_cartesian(cartesian)
            values = branch_center + np.remainder(
                np.asarray(values) - branch_center + np.pi, 2.0 * np.pi
            ) - np.pi
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("eta evaluator returned nonfinite values")
        return values

    def constant_eta_boundary_curve(
        self, eta_evaluator, eta, npoints=256, *, nphi=129, maxiter=60
    ):
        """Return a closed wall curve at a requested value of an eta field.

        For each poloidal wall point, eta is bracketed on a toroidal grid over
        one field period and refined with vectorized bisection.
        """
        self._validate_eta_evaluator(eta_evaluator)
        eta_array = np.asarray(eta, float)
        if eta_array.ndim > 0 and eta_array.size != 1:
            return np.stack(
                [
                    self.constant_eta_boundary_curve(
                        eta_evaluator,
                        value,
                        npoints,
                        nphi=nphi,
                        maxiter=maxiter,
                    )
                    for value in eta_array.ravel()
                ]
            )
        theta = np.linspace(0.0, 2.0 * np.pi, int(npoints), endpoint=True)
        target = float(eta_array.reshape(-1)[0])
        center = self._phi[0] + target
        grid = np.linspace(
            center - 0.5 * self._period,
            center + 0.5 * self._period,
            int(nphi),
        )
        ph, th = np.meshgrid(grid, theta[:-1], indexing="ij")
        values = self._evaluate_eta(eta_evaluator, ph, th, center)
        diff = values - target
        crossings = (diff[:-1] * diff[1:] <= 0)
        idx = np.argmax(crossings, axis=0)
        if not np.all(np.any(crossings, axis=0)):
            raise ValueError("target eta is not bracketed on the wall over one field period")
        lo, hi = grid[idx], grid[idx+1]
        cols = np.arange(theta.size-1)
        flo, fhi = diff[idx, cols], diff[idx+1, cols]
        for _ in range(maxiter):
            mid = 0.5*(lo+hi)
            fm = self._evaluate_eta(eta_evaluator, mid, theta[:-1], center) - target
            left = flo * fm <= 0
            hi = np.where(left, mid, hi); fhi = np.where(left, fm, fhi)
            lo = np.where(left, lo, mid); flo = np.where(left, flo, fm)
        phi = np.r_[0.5 * (lo + hi), 0.5 * (lo[0] + hi[0])]
        return self.cartesian(phi, theta)
    constant_eta_curve = constant_eta_boundary_curve


__all__ = ["WallEvaluator", "parse_kisslinger"]
