"""Fast checks of the uniform B-spline scalar potential and its least-squares fit (synthetic fields, small sizes).

* the cardinal basis (partition of unity, derivatives vs finite differences);
* the sum-factorised least-squares operator (adjoint identity, exact column norms, rows vs direct evaluation);
* recovery of a harmonic synthetic potential (tan-angle misfit and ``G``), stellarator symmetry to round-off, the
  knot-box guard, exact dyadic prolongation;
* the evaluator: phi solve vs an independent Newton, eager vs jit, payload round trip, ``wrapped=`` semantics,
  knot-box bounds.

The real-HSX fit is checked against the prototype product outside the test suite (it needs MAKEGRID).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.geometry import spline_scalar_potential as ssp  # noqa: E402
from drbx.geometry.WallEvaluator import WallEvaluator  # noqa: E402

NFP = 4
PERIOD = 2.0 * math.pi / NFP


def _wall(R0=1.2, a=0.12, b=0.09, d=0.02, nt=33, npol=129):
    """A stellarator-symmetric rotating-ellipse vessel: R(-theta, -phi) = R, Z(-theta, -phi) = -Z."""
    phi = np.linspace(0.0, PERIOD, nt)
    theta = np.linspace(0.0, 2.0 * np.pi, npol)
    P, T = np.meshgrid(phi, theta, indexing="ij")
    rz = np.stack((R0 + a * np.cos(T) + d * np.cos(T - NFP * P), b * np.sin(T) + d * np.sin(T - NFP * P)), -1)
    rz[:, -1] = rz[:, 0]
    rz[-1] = rz[0]
    return WallEvaluator({"phi": phi, "RZ": rz, "nfp": NFP})


class _HarmonicField:
    """B = grad(chi), chi = G0 phi + a (R^2 - 2 Z^2) + b R^4 cos(4 phi) + c ln R (harmonic); MAKEGRID-like attributes."""

    def __init__(self, G0=1.2, a=0.02, b=0.007, c=0.06):
        self.G0, self.a, self.b, self.c = G0, a, b, c
        self.period, self.nfp = PERIOD, NFP
        self.R = np.linspace(0.835, 1.625, 316)
        self.Z = np.linspace(-0.444, 0.444, 356)
        self.phi = np.arange(90) * PERIOD / 90

    def evaluate_cylindrical(self, points):
        R, ph, Z = np.asarray(points, float).T
        return np.column_stack((2 * self.a * R + 4 * self.b * R ** 3 * np.cos(4 * ph) + self.c / R,
                                (self.G0 - 4 * self.b * R ** 4 * np.sin(4 * ph)) / R, -4 * self.a * Z))


class _SymmetricField(_HarmonicField):
    """Stellarator-symmetric harmonic field: Phi~ odd under (phi, Z) -> (-phi, -Z).

    chi = G0 phi + b R^4 sin(4 phi) + c Z ln R + e (Z^3 - 1.5 Z R^2) + f R^8 sin(8 phi)."""

    def __init__(self, G0=1.2, b=0.004, c=0.05, e=0.3, f=2e-4):
        super().__init__(G0=G0)
        self.b, self.c, self.e, self.f = b, c, e, f

    def evaluate_cylindrical(self, points):
        R, ph, Z = np.asarray(points, float).T
        b, c, e, f = self.b, self.c, self.e, self.f
        BR = 4 * b * R ** 3 * np.sin(4 * ph) + c * Z / R - 3 * e * Z * R + 8 * f * R ** 7 * np.sin(8 * ph)
        Bp = (self.G0 + 4 * b * R ** 4 * np.cos(4 * ph) + 8 * f * R ** 8 * np.cos(8 * ph)) / R
        BZ = c * np.log(R) + e * (3 * Z ** 2 - 1.5 * R ** 2)
        return np.column_stack((BR, Bp, BZ))


def _cylindrical_vectors(points_xyz, vectors_xyz):
    phi = np.arctan2(points_xyz[:, 1], points_xyz[:, 0])
    c, s = np.cos(phi), np.sin(phi)
    return np.stack((vectors_xyz[:, 0] * c + vectors_xyz[:, 1] * s, -vectors_xyz[:, 0] * s + vectors_xyz[:, 1] * c,
                     vectors_xyz[:, 2]), -1)


def _tan_angle(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.linalg.norm(np.cross(a, b), axis=-1) / np.abs(np.sum(a * b, -1))


def _interior_points(wall, n, seed, rho=0.9):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-np.pi, np.pi, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    section = wall.cylindrical(phi, theta)
    centre = wall.centerline(phi)
    s = rho * np.sqrt(rng.uniform(0.0, 1.0, n))
    R = centre[0] + s * (section[:, 0] - centre[0])
    Z = centre[1] + s * (section[:, 2] - centre[1])
    return np.column_stack((R * np.cos(phi), R * np.sin(phi), Z))


@pytest.fixture(scope="module")
def wall():
    return _wall()


@pytest.fixture(scope="module")
def harmonic_fit(wall):
    field = _HarmonicField()
    return field, ssp.spline_scalar_potential_evaluator_from_bfield(field, wall, h=0.04, nphi=12,
                                                                    coarse=dict(maxiter=300), maxiter=4000)


@pytest.fixture(scope="module")
def small_system(wall):
    field = _HarmonicField()
    R_lo, Z_lo, nR, nZ = ssp.spline_knot_box(field, wall, 0.05, 0.02, z_snap=0.025)
    system = ssp._SplineFitSystem(wall, R_lo=R_lo, Z_lo=Z_lo, nR=nR, nZ=nZ, h=0.05, p=5, nphi=8, period=PERIOD,
                                  phi0=0.0, gauss=3, rho_data=1.02, lam=1e-2)
    system.assemble_rhs(field, wall)
    return system


# ---------------------------------------------------------------------------------------------------------------------
# basis
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("degree", [3, 5])
def test_cardinal_basis_partition_of_unity_and_derivatives(degree):
    t = jnp.linspace(0.0, 1.0, 21)
    v, d1, d2 = (np.asarray(a) for a in ssp.cardinal_bspline_basis(degree, t, derivatives=2))
    assert v.shape == (21, degree + 1)
    np.testing.assert_allclose(v.sum(axis=1), 1.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(d1.sum(axis=1), 0.0, rtol=0, atol=1e-13)
    assert np.all(v >= -1e-16)
    e = 1e-6
    vp, d1p = (np.asarray(a) for a in ssp.cardinal_bspline_basis(degree, t + e, derivatives=1))
    vm, d1m = (np.asarray(a) for a in ssp.cardinal_bspline_basis(degree, t - e, derivatives=1))
    np.testing.assert_allclose((vp - vm) / (2 * e), d1, rtol=0, atol=1e-8)
    np.testing.assert_allclose((d1p - d1m) / (2 * e), d2, rtol=0, atol=1e-7)
    # autodiff through the tables equals the analytic derivative table
    jac = jax.vmap(jax.jacfwd(lambda s: ssp.cardinal_bspline_basis(degree, s[None], derivatives=0)[0][0]))(t)
    np.testing.assert_allclose(np.asarray(jac), d1, rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------------------------------------------------
# operator
# ---------------------------------------------------------------------------------------------------------------------
def test_operator_adjoint_identity(small_system):
    s = small_system
    rng = np.random.default_rng(0)
    x, y = jnp.asarray(rng.standard_normal(s.n)), jnp.asarray(rng.standard_normal(s.m))
    Ax, ATy = s.matvec(x), s.rmatvec(y)
    lhs, rhs = float(jnp.dot(Ax, y)), float(jnp.dot(x, ATy))
    assert abs(lhs - rhs) <= 1e-12 * float(jnp.linalg.norm(Ax) * jnp.linalg.norm(y))


def test_column_norms_are_exact(small_system):
    s = small_system
    dsq = np.asarray(s.column_norms_squared())
    rng = np.random.default_rng(1)
    columns = np.concatenate([[s.n - 1], rng.choice(s.n - 1, 25, replace=False)])
    for j in columns:
        e = jnp.zeros(s.n).at[j].set(1.0)
        exact = float(jnp.sum(s.matvec(e) ** 2))
        assert abs(exact - dsq[j]) <= 1e-12 * exact


def test_data_rows_equal_direct_spline_evaluation(small_system):
    """Data rows of A x are sqrt(w)/Bs * (Phi_R, (Phi_phi + G)/R, Phi_Z) of the spline with coefficients x."""
    s = small_system
    rng = np.random.default_rng(2)
    x = rng.standard_normal(s.n)
    y = np.asarray(s.matvec(jnp.asarray(x)))
    ev = ssp.SplineScalarPotentialEvaluator(np.asarray(s._dense(jnp.asarray(x))), G=1.0, R_lo=s.R_lo, h_R=s.h,
                                            Z_lo=s.Z_lo, h_Z=s.h, nfp=NFP, degree=s.p)
    points = s.data_points()
    _, dR, dP, dZ = ev.terms_cylindrical(points)
    ir, ip, iz = np.unravel_index(s.d_glob, (s.nRg, s.nPg, s.nZg))
    sw = s._sR[ir] * s._sP[ip] * s._sZ[iz] / s.Bs
    expected = np.concatenate([sw * dR, sw * (dP + x[-1]) / points[:, 0], sw * dZ])
    np.testing.assert_allclose(y[:3 * s.Nd], expected, rtol=0, atol=1e-13 * np.abs(expected).max())


def test_knot_box_guard_rejects_uncentred_box_and_phi_offset(wall):
    field = _HarmonicField()
    R_lo, Z_lo, nR, nZ = ssp.spline_knot_box(field, wall, 0.05, 0.02, z_snap=0.025)
    assert abs(Z_lo + 0.5 * nZ * 0.05) < 1e-15
    common = dict(R_lo=R_lo, nR=nR, nZ=nZ, h=0.05, p=5, nphi=8, period=PERIOD, gauss=3, rho_data=1.02, lam=1e-2)
    with pytest.raises(ValueError, match="mirror-symmetric knot box"):
        ssp._SplineFitSystem(wall, Z_lo=Z_lo + 0.01, phi0=0.0, **common)
    with pytest.raises(ValueError, match="mirror-symmetric knot box"):
        ssp._SplineFitSystem(wall, Z_lo=Z_lo, phi0=0.01, **common)


def test_knot_boxes_nest_for_the_warm_start(wall):
    field = _HarmonicField()
    fine = ssp.spline_knot_box(field, wall, 0.01, 0.02)
    coarse = ssp.spline_knot_box(field, wall, 0.02, 0.02)
    assert fine[0] == coarse[0] and abs(fine[1] - coarse[1]) < 1e-15
    assert fine[1] == pytest.approx(-0.5 * fine[3] * 0.01, abs=1e-15)


def test_prolongation_is_exact():
    rng = np.random.default_rng(4)
    p, nR, nphi, nZ = 5, 4, 6, 5
    coarse = ssp.SplineScalarPotentialEvaluator(rng.standard_normal((nR + p, nphi, nZ + p)), G=1.3, R_lo=1.0, h_R=0.04,
                                                Z_lo=-0.1, h_Z=0.04, nfp=NFP, degree=p)
    fine_coef = ssp.prolong_coefficients(coarse.coefficients, p, (2 * nR + p, 2 * nphi, 2 * nZ + p))
    fine = ssp.SplineScalarPotentialEvaluator(np.asarray(fine_coef), G=1.3, R_lo=1.0, h_R=0.02, Z_lo=-0.1, h_Z=0.02,
                                              nfp=NFP, degree=p)
    q = np.column_stack((rng.uniform(1.0, 1.16, 500), rng.uniform(-4, 4, 500), rng.uniform(-0.1, 0.1, 500)))
    np.testing.assert_allclose(fine.terms_cylindrical(q), coarse.terms_cylindrical(q), rtol=0, atol=1e-11)


# ---------------------------------------------------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------------------------------------------------
def test_harmonic_field_recovery(harmonic_fit, wall):
    field, ev = harmonic_fit
    X = _interior_points(wall, 4000, seed=5)
    _, g = ev.evaluate_and_gradient_cartesian(X)
    R = np.hypot(X[:, 0], X[:, 1])
    B = field.evaluate_cylindrical(np.column_stack((R, np.arctan2(X[:, 1], X[:, 0]), X[:, 2])))
    b = _tan_angle(_cylindrical_vectors(X, g), B)
    assert np.sqrt(np.mean(b ** 2)) < 1e-6 and b.max() < 1e-5
    assert abs(ev.G - field.G0) <= 1e-8 * field.G0
    diag = ev.diagnostics
    assert diag["misfit_tan_angle_all_data"]["rms"] < 1e-6
    assert diag["coarse"]["iterations"] == 300
    assert diag["monitor"]["history"], "the intrinsic monitor recorded no check"
    assert ev.I == 0.0 and ev.nfp == NFP and ev.period == pytest.approx(PERIOD)


def test_fit_is_stellarator_symmetric_to_roundoff(wall):
    field = _SymmetricField()
    ev = ssp.spline_scalar_potential_evaluator_from_bfield(field, wall, h=0.04, nphi=12, coarse=dict(maxiter=200),
                                                           maxiter=600, monitor_every=0)
    X = _interior_points(wall, 3000, seed=6)
    R, phi = np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0])
    Xm = np.column_stack((R * np.cos(-phi), R * np.sin(-phi), -X[:, 2]))
    g = _cylindrical_vectors(X, ev.gradient_cartesian(X))
    gm = _cylindrical_vectors(Xm, ev.gradient_cartesian(Xm))
    asym = np.linalg.norm(gm - g * np.array([-1.0, 1.0, 1.0]), axis=-1) / np.linalg.norm(g, axis=-1)
    # The fit itself is not exact (misfit >> round-off), so symmetry is a property of the knots and the mask.
    # Measured for this setup: an uncentred knot box (Z shifted by a quarter cell) gives ~2e-6 rms / 6e-6 max; the
    # centred box gives LSMR round-off accumulation, ~1e-11 (warm start) to ~1e-9 (cold start, 600 iterations).
    B = field.evaluate_cylindrical(np.column_stack((R, phi, X[:, 2])))
    assert np.sqrt(np.mean(_tan_angle(g, B) ** 2)) > 1e-7
    assert asym.max() < 1e-9
    # eta is odd up to the additive gauge constant of Phi~ (the constant function is in the null space of the data
    # and interior-penalty rows): eta + eta(mirror) = 2 c, c tiny but not set by the symmetry.
    total = ev.evaluate_cartesian(X) + ev.evaluate_cartesian(Xm)
    assert abs(total.mean()) < 1e-9
    np.testing.assert_allclose(total - total.mean(), 0.0, rtol=0, atol=1e-11)


def test_fit_accepts_a_loader_and_rejects_float32_free_inputs(wall):
    calls = []

    def loader():
        calls.append(1)
        return _HarmonicField()

    ev = ssp.spline_scalar_potential_evaluator_from_bfield(loader, wall, h=0.05, nphi=8, coarse=False, maxiter=50,
                                                           monitor_every=0)
    assert calls == [1] and ev.diagnostics["coarse"] is None
    with pytest.raises(ValueError, match="unknown coarse"):
        ssp.spline_scalar_potential_evaluator_from_bfield(_HarmonicField(), wall, h=0.04, nphi=12,
                                                          coarse=dict(nphi_cells=3))


# ---------------------------------------------------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------------------------------------------------
def test_phi_solve_matches_plain_newton_on_the_evaluator(harmonic_fit, wall):
    _, ev = harmonic_fit
    X = _interior_points(wall, 600, seed=7)
    R, phi, Z = np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0]), X[:, 2]
    labels = ev.evaluate_cylindrical(np.column_stack((R, phi, Z)))
    start = phi + 0.02 * np.random.default_rng(8).standard_normal(phi.size)
    solved, residual = ev.solve_phi(R, Z, labels, start, return_residual=True)
    p = start.copy()                       # independent route: Newton with the public value and gradient
    for _ in range(30):
        q = np.column_stack((R, p, Z))
        p = p - (ev.evaluate_cylindrical(q) - labels) / (R * ev.gradient_cylindrical(q)[:, 1])
    assert np.max(np.abs(solved - p)) < 1e-12
    assert np.max(np.abs(solved - phi)) < 1e-12
    assert np.max(np.abs(residual)) < 1e-13


def test_eager_and_jit_parity(harmonic_fit, wall):
    _, ev = harmonic_fit
    spline = ev.jax
    X = _interior_points(wall, 257, seed=9)
    R, phi, Z = (jnp.asarray(a) for a in (np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0]), X[:, 2]))
    eager = spline.terms(R, phi, Z)
    jitted = jax.jit(lambda s, a, b, c: s.terms(a, b, c))(spline, R, phi, Z)
    for a, b in zip(eager, jitted):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-12, atol=2e-12)
    labels = spline.eta(R, phi, Z)
    solve_eager = spline.solve_phi(R, Z, labels, phi + 0.01)
    solve_jit = jax.jit(lambda s, a, b, c, d: s.solve_phi(a, b, c, d))(spline, R, Z, labels, phi + 0.01)
    np.testing.assert_allclose(np.asarray(solve_eager), np.asarray(solve_jit), rtol=0, atol=2e-12)
    np.testing.assert_allclose(np.asarray(solve_jit), np.asarray(phi), rtol=0, atol=1e-12)
    # the NumPy-facing evaluator runs the same kernels
    np.testing.assert_allclose(ev.terms_cylindrical(np.column_stack((R, phi, Z))), np.stack(eager), rtol=2e-12,
                               atol=2e-12)


def test_phi_solve_implicit_derivatives(harmonic_fit, wall):
    """d phi / d(R, Z, eta) and d phi / d coefficients from custom_root equal the implicit-function formula."""
    _, ev = harmonic_fit
    spline = ev.jax
    X = _interior_points(wall, 1, seed=10)[0]
    R, phi, Z = float(np.hypot(X[0], X[1])), float(np.arctan2(X[1], X[0])), float(X[2])
    eta = float(spline.eta(R, phi, Z))
    grad = jax.grad(lambda a, b, c: spline.solve_phi(a, b, c, phi + 0.01), argnums=(0, 1, 2))(R, Z, eta)
    _, eR, ephi, eZ = (float(v) for v in spline.eta_and_partials(R, phi, Z))
    np.testing.assert_allclose([float(g) for g in grad], [-eR / ephi, -eZ / ephi, 1.0 / ephi], rtol=1e-12)

    def phi_of(coefficients):
        leaves, tree = jax.tree_util.tree_flatten(spline)
        return jax.tree_util.tree_unflatten(tree, [coefficients] + leaves[1:]).solve_phi(R, Z, eta, phi + 0.01)

    dc = jax.grad(phi_of)(spline.coefficients)
    deta_dc = jax.grad(lambda c: jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(spline), [c] + jax.tree_util.tree_leaves(spline)[1:]).eta(R, phi, Z))(
        spline.coefficients)
    np.testing.assert_allclose(np.asarray(dc), -np.asarray(deta_dc) / ephi, rtol=0, atol=1e-13)


def test_payload_round_trip(harmonic_fit, tmp_path, wall):
    _, ev = harmonic_fit
    payload = ev.to_payload(prefix="eta_")
    assert all(k.startswith("eta_") for k in payload)
    restored = ssp.SplineScalarPotentialEvaluator.from_payload(payload, prefix="eta_")
    path = tmp_path / "spline.npz"
    ev.save(path)
    loaded = ssp.SplineScalarPotentialEvaluator.load(path)
    X = _interior_points(wall, 300, seed=11)
    for other in (restored, loaded):
        np.testing.assert_array_equal(other.coefficients, ev.coefficients)
        assert other.G == ev.G and other.knot_box == ev.knot_box and other.degree == ev.degree
        e0, g0 = ev.evaluate_and_gradient_cartesian(X)
        e1, g1 = other.evaluate_and_gradient_cartesian(X)
        np.testing.assert_array_equal(e0, e1)
        np.testing.assert_array_equal(g0, g1)
    assert loaded.diagnostics["G"] == ev.diagnostics["G"]
    with pytest.raises(ValueError, match="missing"):
        ssp.SplineScalarPotentialEvaluator.from_payload(payload)


def test_wrapped_semantics_and_branches(harmonic_fit, wall):
    _, ev = harmonic_fit
    X = _interior_points(wall, 200, seed=12)
    cyl = np.column_stack((np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0]), X[:, 2]))
    eta = ev.evaluate_cylindrical(cyl, wrapped=False)
    shifted = cyl + np.array([0.0, PERIOD, 0.0])
    np.testing.assert_allclose(ev.evaluate_cylindrical(shifted, wrapped=False), eta + PERIOD, rtol=0, atol=1e-13)
    np.testing.assert_allclose(ev.evaluate_cylindrical(cyl + np.array([0.0, 2 * np.pi, 0.0])), eta + 2 * np.pi,
                               rtol=0, atol=1e-13)
    wrapped = ev.evaluate_cylindrical(shifted, wrapped=True)
    assert np.all((wrapped >= 0.0) & (wrapped < PERIOD))
    np.testing.assert_allclose(np.angle(np.exp(2j * np.pi * (wrapped - eta) / PERIOD)), 0.0, rtol=0, atol=1e-12)
    # Cartesian queries use the principal branch atan2(y, x)
    np.testing.assert_allclose(ev.evaluate_cartesian(X), eta, rtol=0, atol=1e-15)
    value, gradient = ev.evaluate_and_gradient_cartesian(X, wrapped=True)
    np.testing.assert_allclose(value, np.mod(eta, PERIOD), rtol=0, atol=1e-15)
    np.testing.assert_allclose(gradient, ev.gradient_cartesian(X), rtol=0, atol=0)
    # grad eta is the gradient of eta (central differences)
    step = 1e-6
    fd = np.stack([(ev.evaluate_cartesian(X + step * e) - ev.evaluate_cartesian(X - step * e)) / (2 * step)
                   for e in np.eye(3)], -1)
    np.testing.assert_allclose(fd, gradient, rtol=0, atol=1e-8)
    # magnetic potential / field consistency: grad(chi) = G grad(eta)
    np.testing.assert_allclose(ev.magnetic_field_cartesian(X), ev.G * gradient, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(ev.evaluate_magnetic_potential_cylindrical(cyl), ev.G * eta, rtol=1e-13, atol=1e-13)


def test_queries_outside_the_knot_box_raise(harmonic_fit):
    _, ev = harmonic_fit
    box = ev.knot_box
    inside = np.array([[0.5 * (box["R_lo"] + box["R_hi"]), 0.3, 0.0]])
    ev.evaluate_cylindrical(inside)
    for bad in ([box["R_hi"] + 1e-3, 0.3, 0.0], [box["R_lo"] - 1e-3, 0.3, 0.0], [inside[0, 0], 0.3, box["Z_hi"] + 1e-3]):
        with pytest.raises(ValueError, match="knot box"):
            ev.evaluate_cylindrical(np.array([bad]))
    with pytest.raises(ValueError, match="knot box"):
        ev.solve_phi(np.array([box["R_hi"] + 1e-3]), np.array([0.0]), np.array([0.1]))
