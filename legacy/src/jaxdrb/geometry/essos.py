from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EssosTabulatedResult:
    """Results of converting an ESSOS field-line geometry into a TabulatedGeometry file."""

    path: Path
    meta: dict


def _central_diff_periodic(x: np.ndarray, dl: float) -> np.ndarray:
    return (np.roll(x, -1, axis=0) - np.roll(x, 1, axis=0)) / (2.0 * dl)


def _normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-14) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + eps)


def _basis_grads_from_mapping(
    dX_ds: np.ndarray, dX_dtheta: np.ndarray, dX_dphi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute gradients (contravariant basis) from covariant basis vectors.

    For a mapping X(s,theta,phi) with covariant basis vectors e_s = ∂X/∂s, etc:

      ∇s     = (e_theta × e_phi) / sqrt(g)
      ∇theta = (e_phi × e_s)     / sqrt(g)
      ∇phi   = (e_s × e_theta)   / sqrt(g)

    where sqrt(g) = e_s · (e_theta × e_phi).
    """

    e_s = dX_ds
    e_t = dX_dtheta
    e_p = dX_dphi
    cross_tp = np.cross(e_t, e_p)
    sqrtg = np.einsum("ij,ij->i", e_s, cross_tp)
    sqrtg = sqrtg.reshape((-1, 1))
    grad_s = cross_tp / sqrtg
    grad_t = np.cross(e_p, e_s) / sqrtg
    grad_p = np.cross(e_s, e_t) / sqrtg
    return grad_s, grad_t, grad_p


def _resample_uniform_arc_length(
    s: np.ndarray, arrays: dict[str, np.ndarray], *, n: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Resample 1D arrays defined on a monotone arclength s onto a uniform grid."""

    s = np.asarray(s, dtype=float)
    if s.ndim != 1:
        raise ValueError("Expected 1D arclength array.")
    if not np.all(np.diff(s) > 0):
        raise ValueError("Arclength array must be strictly increasing.")

    s_uniform = np.linspace(s[0], s[-1], n, endpoint=False)
    out = {}
    for k, v in arrays.items():
        v = np.asarray(v)
        if v.shape[0] != s.shape[0]:
            raise ValueError(f"Array '{k}' has incompatible leading dimension.")
        if v.ndim == 1:
            out[k] = np.interp(s_uniform, s, v)
        else:
            # Interpolate each component independently.
            out[k] = np.stack([np.interp(s_uniform, s, v[:, c]) for c in range(v.shape[1])], axis=1)
    return s_uniform, out


def vmec_fieldline_to_tabulated(
    *,
    wout_file: str | Path,
    s: float,
    alpha: float = 0.0,
    nphi: int = 256,
    nfield_periods: int = 1,
    out_path: str | Path,
) -> EssosTabulatedResult:
    """Build a `TabulatedGeometry` `.npz` file from an ESSOS VMEC equilibrium along a field line.

    Coordinate choice:
      - VMEC coordinates (s, theta, phi), with a field line defined by alpha = theta - iota * phi.
      - We sample uniformly in phi over `nfield_periods` and then convert to uniform arclength `l`.
      - Perpendicular coordinates are taken as x = s and y = alpha.
    """

    try:
        from essos.fields import Vmec  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("ESSOS is required for vmec_fieldline_to_tabulated().") from e

    wout_file = Path(wout_file)
    if not wout_file.exists():
        raise FileNotFoundError(wout_file)

    vmec = Vmec(str(wout_file))

    # Build a 1D field line in (s,theta,phi) coordinates.
    # iota is inferred from B^theta/B^phi at a representative point.
    phi = np.linspace(0.0, 2.0 * np.pi * nfield_periods / vmec.nfp, nphi, endpoint=False)
    probe = np.array([s, 0.0, 0.0], dtype=float)
    Bsup = np.asarray(vmec.B_contravariant(probe))
    iota = float(Bsup[1] / Bsup[2])
    theta = alpha + iota * phi

    coords = np.stack([np.full_like(phi, s), theta, phi], axis=1)

    # Mapping to xyz and basis vectors via jacobian of to_xyz.
    import jax
    import jax.numpy as jnp

    def to_xyz_jax(p):
        return vmec.to_xyz(p)

    jac = jax.jacfwd(to_xyz_jax)
    jac_v = jax.vmap(jac)(jnp.asarray(coords))
    dX_ds = np.asarray(jac_v[:, :, 0])
    dX_dt = np.asarray(jac_v[:, :, 1])
    dX_dp = np.asarray(jac_v[:, :, 2])
    xyz = np.asarray(jax.vmap(to_xyz_jax)(jnp.asarray(coords)))

    grad_s, grad_t, grad_p = _basis_grads_from_mapping(dX_ds, dX_dt, dX_dp)
    grad_alpha = grad_t - iota * grad_p

    gxx = np.einsum("ij,ij->i", grad_s, grad_s)
    gxy = np.einsum("ij,ij->i", grad_s, grad_alpha)
    gyy = np.einsum("ij,ij->i", grad_alpha, grad_alpha)

    # Curvature: compute b-hat along the line, then kappa = db/dl.
    Bxyz = np.asarray(jax.vmap(lambda p: vmec.B(p))(jnp.asarray(coords)))
    b = _normalize(Bxyz, axis=1)

    # arclength from xyz
    ds_arc = np.linalg.norm(np.diff(xyz, axis=0, append=xyz[:1]), axis=1)
    l_raw = np.cumsum(ds_arc)
    l_raw = np.concatenate([[0.0], l_raw[:-1]])
    # enforce strictly increasing for interpolation by dropping the last wrap step
    keep = np.arange(nphi)
    s_mono = l_raw[keep] + 1e-12 * keep
    total_L = float(s_mono[-1] + ds_arc[-1])

    dl_raw = float(np.mean(np.diff(s_mono)))
    db_dl = _central_diff_periodic(b, dl_raw)
    kappa = db_dl
    bxk = np.cross(b, kappa)

    curv_x = np.einsum("ij,ij->i", bxk, grad_s)
    curv_y = np.einsum("ij,ij->i", bxk, grad_alpha)

    arrays = {
        "gxx": gxx,
        "gxy": gxy,
        "gyy": gyy,
        "curv_x": curv_x,
        "curv_y": curv_y,
        "B": np.linalg.norm(Bxyz, axis=1),
        "dpar_factor": np.ones_like(gxx),
    }

    l_uniform, arrays_u = _resample_uniform_arc_length(s_mono, arrays, n=nphi)
    # Normalize l to start at 0 with periodic endpoint excluded.
    l_uniform = l_uniform - float(l_uniform[0])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, l=l_uniform, **arrays_u)

    return EssosTabulatedResult(
        path=out_path,
        meta={
            "field": "vmec",
            "wout_file": str(wout_file),
            "s": float(s),
            "alpha": float(alpha),
            "iota": iota,
            "L": total_L,
        },
    )


def near_axis_fieldline_to_tabulated(
    *,
    rc: np.ndarray,
    zs: np.ndarray,
    etabar: float,
    nfp: int,
    r: float,
    alpha: float = 0.0,
    nphi: int | None = None,
    out_path: str | Path,
) -> EssosTabulatedResult:
    """Build a `.npz` tabulated geometry from an ESSOS near-axis field line.

    Coordinate/metric choice (pragmatic for a first ESSOS integration):
      - We trace a single field line at fixed `r` and build the parallel coordinate `l` as
        **arclength** in xyz space.
      - We use a **local orthonormal perpendicular basis** (Frenet-like) around that field line,
        so `gxx≈1, gyy≈1, gxy≈0` and `k_perp^2 ≈ kx^2 + ky^2`.
      - Curvature coefficients are computed from the field-line curvature.

    This is enough to run the linear solver on a near-axis configuration and study how growth rates
    change with near-axis parameters. A future extension can replace the local-orthonormal perp
    metric with a true Clebsch-like (r, alpha) metric derived from the near-axis expansion.
    """

    try:
        import jax
        import jax.numpy as jnp
        from essos.fields import near_axis  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("ESSOS is required for near_axis_fieldline_to_tabulated().") from e

    field = near_axis(rc=jnp.asarray(rc), zs=jnp.asarray(zs), etabar=float(etabar), nfp=int(nfp))
    iota = float(np.asarray(field.iota))

    if nphi is None:
        nphi = int(field.nphi)
    nphi = int(nphi)

    phi = np.linspace(0.0, 2.0 * np.pi / nfp, nphi, endpoint=False)
    theta = (alpha + iota * phi) % (2.0 * np.pi)

    # Precompute a (theta,varphi) -> (R,Z,phi0) grid at this r using ESSOS.
    ntheta_grid = 64
    theta_grid = np.linspace(0.0, 2.0 * np.pi, ntheta_grid, endpoint=False)
    R2, Z2, phi02 = field.Frenet_to_cylindrical(float(r), ntheta=ntheta_grid, phi_is_varphi=True)
    R2 = np.asarray(R2)
    Z2 = np.asarray(Z2)
    phi02 = np.asarray(phi02)
    varphi_grid = np.linspace(0.0, 2.0 * np.pi / nfp, int(field.nphi), endpoint=False)

    # Interpolate along varphi if needed.
    if varphi_grid.size != nphi:

        def interp_varphi(arr2):
            out = np.stack(
                [
                    np.interp(phi, varphi_grid, arr2[i], period=2.0 * np.pi / nfp)
                    for i in range(ntheta_grid)
                ],
                axis=0,
            )
            return out

        R2 = interp_varphi(R2)
        Z2 = interp_varphi(Z2)
        phi02 = interp_varphi(phi02)

    # For each varphi sample, interpolate in theta at theta(varphi).
    def interp_theta(arr_theta, th_val):
        # periodic interpolation
        th = float(th_val) % (2.0 * np.pi)
        th_ext = np.concatenate([theta_grid, [2.0 * np.pi]])
        arr_ext = np.concatenate([arr_theta, [arr_theta[0]]])
        return float(np.interp(th, th_ext, arr_ext))

    R_line = np.array([interp_theta(R2[:, j], theta[j]) for j in range(nphi)], dtype=float)
    Z_line = np.array([interp_theta(Z2[:, j], theta[j]) for j in range(nphi)], dtype=float)
    phi0_line = np.array([interp_theta(phi02[:, j], theta[j]) for j in range(nphi)], dtype=float)

    x = R_line * np.cos(phi0_line)
    y = R_line * np.sin(phi0_line)
    xyz = np.stack([x, y, Z_line], axis=1)

    # arclength coordinate and curvature from geometry of xyz curve
    ds_arc = np.linalg.norm(np.diff(xyz, axis=0, append=xyz[:1]), axis=1)
    l_raw = np.cumsum(ds_arc)
    l_raw = np.concatenate([[0.0], l_raw[:-1]])
    s_mono = l_raw + 1e-12 * np.arange(l_raw.size)
    dl_raw = float(np.mean(np.diff(s_mono)))

    t_hat = _normalize(_central_diff_periodic(xyz, dl_raw), axis=1)
    kappa = _central_diff_periodic(t_hat, dl_raw)
    bxk = np.cross(t_hat, kappa)

    # Local orthonormal perpendicular coordinates (x̂, ŷ) from Frenet frame.
    n_hat = _normalize(kappa, axis=1)
    b_hat = _normalize(np.cross(t_hat, n_hat), axis=1)

    grad_x = n_hat
    grad_y = b_hat
    gxx = np.einsum("ij,ij->i", grad_x, grad_x)
    gxy = np.einsum("ij,ij->i", grad_x, grad_y)
    gyy = np.einsum("ij,ij->i", grad_y, grad_y)
    curv_x = np.einsum("ij,ij->i", bxk, grad_x)
    curv_y = np.einsum("ij,ij->i", bxk, grad_y)

    Bmag = np.asarray(
        jax.vmap(lambda p: field.AbsB(p))(
            jnp.asarray(np.stack([np.full_like(phi, r), theta, phi], axis=1))
        )
    )

    arrays_u = {
        "gxx": gxx,
        "gxy": gxy,
        "gyy": gyy,
        "curv_x": curv_x,
        "curv_y": curv_y,
        "B": Bmag,
        "dpar_factor": np.ones_like(gxx),
    }

    l_uniform, arrays_u = _resample_uniform_arc_length(s_mono, arrays_u, n=nphi)
    l_uniform = l_uniform - float(l_uniform[0])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, l=l_uniform, **arrays_u)

    return EssosTabulatedResult(
        path=out_path,
        meta={
            "field": "near_axis",
            "nfp": int(nfp),
            "r": float(r),
            "alpha": float(alpha),
            "iota": iota,
            "note": "perp metric uses local orthonormal frame around the field line",
        },
    )


def biotsavart_fieldline_to_tabulated(
    *,
    coils_json: str | Path,
    R0: float,
    Z0: float = 0.0,
    phi0: float = 0.0,
    nsteps: int = 2000,
    nout: int = 256,
    maxtime: float = 800.0,
    out_path: str | Path,
) -> EssosTabulatedResult:
    """Build a `.npz` tabulated geometry from an ESSOS Biot-Savart field-line trace.

    This uses a *local* perpendicular coordinate model:

      - gxx = 1, gxy = 0, gyy = 1  (so k_perp^2 is constant)

    while still capturing:

      - parallel coordinate as arclength `l`,
      - curvature coefficients from the field-line curvature.
    """

    try:
        import jax.numpy as jnp
        from essos.coils import Coils_from_json  # type: ignore
        from essos.dynamics import Tracing  # type: ignore
        from essos.fields import BiotSavart  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("ESSOS is required for biotsavart_fieldline_to_tabulated().") from e

    coils_json = Path(coils_json)
    if not coils_json.exists():
        raise FileNotFoundError(coils_json)

    coils = Coils_from_json(str(coils_json))
    field = BiotSavart(coils)

    x0 = float(R0 * np.cos(phi0))
    y0 = float(R0 * np.sin(phi0))
    initial = jnp.array([[x0, y0, float(Z0)]])

    tracing = Tracing(
        field=field,
        model="FieldLineAdaptative",
        initial_conditions=initial,
        maxtime=float(maxtime),
        times_to_trace=int(nsteps),
        atol=1e-8,
        rtol=1e-8,
    )
    xyz = np.asarray(tracing.trajectories[0, :, :3], dtype=float)

    ds_arc = np.linalg.norm(np.diff(xyz, axis=0, append=xyz[:1]), axis=1)
    l_raw = np.cumsum(ds_arc)
    l_raw = np.concatenate([[0.0], l_raw[:-1]])
    s_mono = l_raw + 1e-12 * np.arange(l_raw.size)
    dl_raw = float(np.mean(np.diff(s_mono)))

    # b-hat is tangent along the curve
    t_hat = _normalize(_central_diff_periodic(xyz, dl_raw), axis=1)
    kappa = _central_diff_periodic(t_hat, dl_raw)
    bxk = np.cross(t_hat, kappa)

    # Local orthonormal perpendicular coordinates: x̂, ŷ = (n̂, b̂) from Frenet frame.
    n_hat = _normalize(kappa, axis=1)
    b_hat = _normalize(np.cross(t_hat, n_hat), axis=1)

    grad_x = n_hat
    grad_y = b_hat

    gxx = np.einsum("ij,ij->i", grad_x, grad_x)
    gxy = np.einsum("ij,ij->i", grad_x, grad_y)
    gyy = np.einsum("ij,ij->i", grad_y, grad_y)
    curv_x = np.einsum("ij,ij->i", bxk, grad_x)
    curv_y = np.einsum("ij,ij->i", bxk, grad_y)

    arrays = {
        "gxx": gxx,
        "gxy": gxy,
        "gyy": gyy,
        "curv_x": curv_x,
        "curv_y": curv_y,
        "B": np.asarray([float(field.AbsB(p)) for p in xyz]),
        "dpar_factor": np.ones_like(gxx),
    }

    nout = int(nout)
    if nout <= 8:
        raise ValueError("nout must be > 8.")
    if nout > len(s_mono):
        raise ValueError("nout must be <= nsteps.")

    l_uniform, arrays_u = _resample_uniform_arc_length(s_mono, arrays, n=nout)
    l_uniform = l_uniform - float(l_uniform[0])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, l=l_uniform, **arrays_u)

    return EssosTabulatedResult(
        path=out_path,
        meta={
            "field": "biotsavart",
            "coils_json": str(coils_json),
            "R0": float(R0),
            "Z0": float(Z0),
            "phi0": float(phi0),
        },
    )
