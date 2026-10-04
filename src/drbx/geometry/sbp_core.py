"""Host-side disk-core block of the nodal perpendicular scheme: Zernike basis and projection-type SBP operator.

The core is the disk ``u <= R_c = K / n`` around the magnetic axis, handled in Cartesian block coordinates
``(x, y)`` (``x = u cos(theta)``, ``y = u sin(theta)``) so that the axis is regular. It is a diagonal-norm
multidimensional SBP block exact on ``P_p`` (polynomials of total degree ``<= p``):

- nodes: ``n_s = p // 2 + 1`` Gauss-Legendre radii in ``s = u^2`` times ``2 p + 2`` equispaced angles (offset 0), with the
  dx dy quadrature ``wxy`` exact to total degree ``2 p + 1``;
- basis: Zernike functions, orthonormal on the disk, built with the Jacobi recurrence (the monomial basis spans the same
  space but is ill conditioned at the degrees used);
- operator: ``Q = H Dp + Pi^T E (I - Pi) + (I - Pi^T) E (I - Pi) / 2`` with ``Pi`` the ``H``-orthogonal projector onto ``P_p``
  and ``E`` the boundary form of the extrapolation ``Rx`` to ``N_g = n`` points of the first ring level, so that
  ``Q + Q^T = E`` and ``D = H^-1 Q`` is exact on ``P_p``;
- core velocity gradient (D5c) matrices, frame transforms of ``h``, ``|J|`` and ``K``, and the Gram data of the shell
  damping projector.

Everything here is host NumPy, built once. :class:`CoreBlock` satisfies the ``Block`` protocol of
:mod:`drbx.geometry.nodal_layout` and is the first block of a layout with ``inner="core"``.
"""
from __future__ import annotations

import numpy as np

from drbx.geometry.nodal_layout import Side
from drbx.geometry.sbp_operators import ring_basis

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def polar_gauss_nodes(p: int, R_c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(u, theta, wxy, ring_radii)`` of the polar Gauss nodes of degree ``p`` on the disk of radius ``R_c``.

    ``n_s = p // 2 + 1`` radii ``u_k = sqrt(s_k)`` from Gauss-Legendre in ``s = u^2`` (ring weight ``R_c^2 w_k / 4``) and
    ``n_th = 2 p + 2`` angles ``2 pi j / n_th``; ring-major order; ``wxy`` includes the angular weight ``2 pi / n_th``.
    """
    n_s, n_th = p // 2 + 1, 2 * p + 2
    xi, wx = np.polynomial.legendre.leggauss(n_s)
    s = R_c * R_c * (1.0 + xi) / 2.0
    radii = np.sqrt(s)
    ring_w = R_c * R_c * wx / 4.0
    u = np.repeat(radii, n_th)
    theta = np.tile(TWO_PI * np.arange(n_th) / n_th, n_s)
    wxy = np.repeat(ring_w, n_th) * (TWO_PI / n_th)
    return u, theta, wxy, radii


# ---------------------------------------------------------------------------
# Zernike basis
# ---------------------------------------------------------------------------
def zernike_indices(p: int) -> list[tuple[int, int]]:
    """``(nu, m)`` with ``nu = 0 .. p``, ``m = -nu, -nu + 2, .., nu``, ordered by ``nu`` then ``m``."""
    return [(nu, m) for nu in range(p + 1) for m in range(-nu, nu + 1, 2)]


def jacobi_alpha0(alpha: int, kmax: int, t) -> tuple[np.ndarray, np.ndarray]:
    """``P_k^{(alpha, 0)}(t)`` and ``dP_k/dt`` for ``k = 0 .. kmax`` by the three-term recurrence; shape ``(kmax + 1,) + t.shape``."""
    t = np.asarray(t, dtype=np.float64)
    P = np.zeros((kmax + 1,) + t.shape)
    dP = np.zeros_like(P)
    P[0] = 1.0
    if kmax >= 1:
        P[1] = (alpha + 1.0) + (alpha + 2.0) * (t - 1.0) / 2.0
        dP[1] = (alpha + 2.0) / 2.0
    for k in range(2, kmax + 1):
        s = 2 * k + alpha
        a = 2.0 * k * (k + alpha) * (s - 2)
        b = (s - 1) * (s * (s - 2) * t + alpha * alpha)
        c = 2.0 * (k + alpha - 1) * (k - 1) * s
        P[k] = (b * P[k - 1] - c * P[k - 2]) / a
        dP[k] = ((s - 1) * s * (s - 2) * P[k - 1] + b * dP[k - 1] - c * dP[k - 2]) / a
    return P, dP


def zernike_vandermonde(u, theta, p: int, R_c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal Zernike functions of total degree ``<= p`` and their Cartesian derivatives at ``(u, theta)``.

    Returns ``(V, Vx, Vy, degree)``: arrays ``(Q, dim)`` with ``dim = (p + 1)(p + 2) / 2`` and ``degree`` (``nu`` per column).
    ``Z_{nu,m} = c R_nu^|m|(rho) Theta_m(theta)`` with ``rho = u / R_c``, ``Theta_m = cos(m theta)`` for ``m >= 0`` and
    ``sin(|m| theta)`` for ``m < 0``, normalised so that ``int_{u < R_c} Z_a Z_b dx dy = delta_ab``.
    """
    u = np.asarray(u, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    rho = u / R_c
    t = 1.0 - 2.0 * rho * rho
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    idx = zernike_indices(p)
    V = np.empty((u.size, len(idx)))
    Vx = np.empty_like(V)
    Vy = np.empty_like(V)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for col, (nu, m) in enumerate(idx):
        alpha = abs(m)
        k = (nu - alpha) // 2
        if alpha not in cache:
            cache[alpha] = jacobi_alpha0(alpha, (p - alpha) // 2, t)
        P, dP = cache[alpha][0][k], cache[alpha][1][k]
        sign = -1.0 if k % 2 else 1.0
        R = sign * rho**alpha * P
        dR = sign * (-4.0 * rho ** (alpha + 1) * dP + (alpha * rho ** (alpha - 1) * P if alpha else 0.0))
        norm = np.sqrt((2.0 * nu + 2.0) / (2.0 if m == 0 else 1.0)) / (np.sqrt(np.pi) * R_c)
        if m >= 0:
            Th, dTh = np.cos(m * theta), -m * np.sin(m * theta)
        else:
            Th, dTh = np.sin(alpha * theta), alpha * np.cos(alpha * theta)
        R_over_rho = sign * rho ** (alpha - 1) * P if alpha else 0.0 * rho
        V[:, col] = norm * R * Th
        radial = dR / R_c * Th
        angular = R_over_rho / R_c * dTh
        Vx[:, col] = norm * (cos_t * radial - sin_t * angular)
        Vy[:, col] = norm * (sin_t * radial + cos_t * angular)
    return V, Vx, Vy, np.array([nu for nu, _ in idx])


# ---------------------------------------------------------------------------
# Shell damping data (host helpers)
# ---------------------------------------------------------------------------
def shell_gram_inverse(Vm: np.ndarray, Hk: np.ndarray) -> np.ndarray:
    """``(Vm^T diag(Hk) Vm)^-1`` for one eta plane (``Hk = wxy * jac_b``)."""
    return np.linalg.inv(Vm.T @ (Hk[:, None] * Vm))


def shell_projector(Vm: np.ndarray, Hk: np.ndarray) -> np.ndarray:
    """Dense ``H``-orthogonal projector onto the span of ``Vm`` (``P_{p-1}``) for one plane; ``P_h = I - projector``."""
    return Vm @ shell_gram_inverse(Vm, Hk) @ (Vm.T * Hk[None, :])


def shell_kappa(c_kappa: float, p: int, R_c: float, vxy_max: float) -> float:
    """Shell damping rate ``kappa = c_kappa * max |V_xy| * p / R_c``."""
    return float(c_kappa * vxy_max * p / R_c)


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------
class CoreBlock:
    """Disk core ``u <= R_c = K / n`` with exactness degree ``p`` (see the module docstring).

    Attributes: ``u, theta, wxy`` (node coordinates and quadrature), ``D1, D2`` (Cartesian derivatives), ``V, Vx, Vy``
    (Zernike basis at the nodes), ``VH = (V^T H V)^-1 V^T H``, ``Pi``, ``Qx, Qy, Ex, Ey``, ``Vm`` (the ``nu <= p - 1`` columns),
    the core D5c matrices ``Q_phi, Q_tr`` (polynomial coefficients from the core values and the ring trace) and
    ``G1_phi, G1_tr, G2_phi, G2_tr`` (their Cartesian gradients) and ``sides = {"outer": Side}``.
    """

    kind = "core"
    frame = "cartesian"
    velocity_gradient = None
    dissipation = None

    def __init__(self, n: int, K: int, p: int) -> None:
        n, K, p = int(n), int(K), int(p)
        if K < 1 or K >= n:
            raise ValueError(f"core radius K={K} must satisfy 1 <= K < n={n}")
        if p < 1 or p > (n - 1) // 2:
            raise ValueError(f"core degree p={p} must satisfy 1 <= p <= (n - 1) / 2 = {(n - 1) // 2} (ring modes 1..p)")
        self.n, self.K, self.p = n, K, p
        self.R_c = R = K / n
        self.u, self.theta, self.wxy, self.radii = polar_gauss_nodes(p, R)
        self.n_nodes = self.u.size
        x, y = self.u * np.cos(self.theta), self.u * np.sin(self.theta)
        self.x, self.y = x, y
        V, Vx, Vy, degree = zernike_vandermonde(self.u, self.theta, p, R)
        self.V, self.Vx, self.Vy, self.degree = V, Vx, Vy, degree
        self.dim = V.shape[1]
        H = np.diag(self.wxy)
        VtH = V.T * self.wxy[None, :]
        self.VH = np.linalg.solve(VtH @ V, VtH)
        self.Pi = V @ self.VH
        Dpx, Dpy = Vx @ self.VH, Vy @ self.VH
        # extrapolation to the first level's theta nodes (theta_g = pi / n + 2 pi j / n)
        self.N_g = n
        self.theta_g = np.pi / n + TWO_PI * np.arange(n) / n
        Vg = zernike_vandermonde(np.full(n, R), self.theta_g, p, R)[0]
        self.Vg = Vg
        self.Rx = Rx = Vg @ self.VH
        Bg = R * TWO_PI / n
        cg, sg = np.cos(self.theta_g), np.sin(self.theta_g)
        self.Ex = Ex = Rx.T @ (Bg * cg[:, None] * Rx)
        self.Ey = Ey = Rx.T @ (Bg * sg[:, None] * Rx)
        I = np.eye(self.n_nodes)
        Pi, IP = self.Pi, I - self.Pi
        self.Qx = H @ Dpx + Pi.T @ Ex @ IP + 0.5 * IP.T @ Ex @ IP
        self.Qy = H @ Dpy + Pi.T @ Ey @ IP + 0.5 * IP.T @ Ey @ IP
        self.D1 = self.Qx / self.wxy[:, None]
        self.D2 = self.Qy / self.wxy[:, None]
        self.Vm = V[:, : p * (p + 1) // 2]
        # side
        T = Rx
        self.sides = {"outer": Side(N=n, T=T, TF1=R * cg[:, None] * Rx, TF2=R * sg[:, None] * Rx, u_face=R, rows=())}
        # core D5c: KKT-constrained fit to the core values and to boundary modes 1..p of the ring trace
        _B, Binv, keys, _om = ring_basis(n, np.pi / n)
        sel = [i for i, (m, typ) in enumerate(keys) if typ in ("c", "s") and 1 <= m <= p]
        A = Binv[sel] @ Vg
        nd, nc = self.dim, len(sel)
        KKT = np.zeros((nd + nc, nd + nc))
        KKT[:nd, :nd] = VtH @ V
        KKT[:nd, nd:] = A.T
        KKT[nd:, :nd] = A
        Kinv = np.linalg.inv(KKT)
        Q_phi = Kinv[:nd, :nd] @ VtH
        Q_tr = Kinv[:nd, nd:] @ Binv[sel]
        self.D5c_sel = tuple(sel)
        self.Q_phi, self.Q_tr = Q_phi, Q_tr
        self.G1_phi, self.G1_tr = Vx @ Q_phi, Vx @ Q_tr
        self.G2_phi, self.G2_tr = Vy @ Q_phi, Vy @ Q_tr

    # -- frame transforms -------------------------------------------------
    def to_block_frame(self, h_log, jac_log, u=None, theta=None):
        """Covariant ``h`` and ``|J|`` from logical ``(u, theta, eta)`` to the block frame ``(x, y, eta)``."""
        u = self.u if u is None else np.asarray(u)
        theta = self.theta if theta is None else np.asarray(theta)
        c, s = np.cos(theta), np.sin(theta)
        hu, ht = h_log[..., 0], h_log[..., 1]
        h = np.stack([c * hu - s / u * ht, s * hu + c / u * ht, h_log[..., 2]], axis=-1)
        return h, jac_log / u

    def to_block_frame_K(self, K_log, u=None, theta=None):
        """Contravariant curvature ``K`` from logical to block frame: ``K^x = cos K^u - u sin K^theta``, etc."""
        u = self.u if u is None else np.asarray(u)
        theta = self.theta if theta is None else np.asarray(theta)
        c, s = np.cos(theta), np.sin(theta)
        Ku, Kt = K_log[..., 0], K_log[..., 1]
        return np.stack([c * Ku - u * s * Kt, s * Ku + u * c * Kt, K_log[..., 2]], axis=-1)

    # -- evaluation ---------------------------------------------------------
    def evaluate_at(self, values, u, theta):
        """Projection onto ``P_p`` evaluated at ``(u, theta)``: ``V(u, theta) @ VH @ values``, ``values (E, N_c[, F])``."""
        Vq = zernike_vandermonde(np.asarray(u).ravel(), np.asarray(theta).ravel(), self.p, self.R_c)[0]
        return np.einsum("qn,en...->eq...", Vq @ self.VH, np.asarray(values))


def build_core_block(n: int, K: int, p: int) -> CoreBlock:
    """Core of radius ``K / n`` and polynomial degree ``p`` for the raw grid size ``n``."""
    return CoreBlock(n, K, p)
