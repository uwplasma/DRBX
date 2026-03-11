from __future__ import annotations

import numpy as np


def _as_np(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _flatten_space(a: np.ndarray) -> np.ndarray:
    if a.ndim < 2:
        raise ValueError("Expected array with time + at least one spatial dimension.")
    return a.reshape(a.shape[0], -1)


def _segment_indices(nt: int, nperseg: int, noverlap: int) -> list[tuple[int, int]]:
    nperseg = int(max(8, min(nt, nperseg)))
    step = max(1, nperseg - int(noverlap))
    out: list[tuple[int, int]] = []
    i0 = 0
    while i0 + nperseg <= nt:
        out.append((i0, i0 + nperseg))
        i0 += step
    if not out:
        out.append((0, nt))
    return out


def compute_fluctuation_rms(
    field_t: np.ndarray,
    *,
    equilibrium_mode: str = "t0",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rms_total, rms_fluct, equilibrium_field)."""

    a = _as_np(field_t)
    if a.ndim < 2:
        raise ValueError("compute_fluctuation_rms expects time-resolved field.")
    if equilibrium_mode == "mean":
        eq = np.mean(a, axis=0)
    else:
        eq = a[0]
    delta = a - eq[None, ...]
    axes = tuple(range(1, a.ndim))
    rms_total = np.sqrt(np.mean(a * a, axis=axes))
    rms_fluct = np.sqrt(np.mean(delta * delta, axis=axes))
    return rms_total, rms_fluct, eq


def compute_frequency_psd(
    series_t: np.ndarray,
    *,
    dt: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch-like PSD for a scalar time series."""

    y = _as_np(series_t).reshape(-1)
    nt = y.size
    if nt < 4:
        f = np.fft.rfftfreq(max(nt, 2), d=max(dt, 1e-30))
        return f, np.zeros_like(f)
    if noverlap is None:
        noverlap = nperseg // 2
    segments = _segment_indices(nt, nperseg=nperseg, noverlap=noverlap)
    pxx = None
    win = np.hanning(max(8, min(nt, nperseg)))
    for s0, s1 in segments:
        chunk = y[s0:s1]
        w = np.hanning(chunk.size) if chunk.size != win.size else win
        chunk = (chunk - np.mean(chunk)) * w
        ft = np.fft.rfft(chunk)
        p = (np.abs(ft) ** 2) / max(np.sum(w * w), 1e-30)
        pxx = p if pxx is None else (pxx + p)
    assert pxx is not None
    pxx /= float(len(segments))
    f = np.fft.rfftfreq(segments[0][1] - segments[0][0], d=max(dt, 1e-30))
    return f, pxx


def compute_ky_psd(
    field_2d: np.ndarray,
    *,
    dy: float,
    axis_y: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum over the binormal/poloidal axis."""

    a = _as_np(field_2d)
    if a.ndim < 2:
        raise ValueError("compute_ky_psd expects at least 2D field.")
    ay = np.moveaxis(a, axis_y, -1)
    ny = ay.shape[-1]
    ft = np.fft.rfft(ay - np.mean(ay, axis=-1, keepdims=True), axis=-1)
    power = np.mean(np.abs(ft) ** 2, axis=tuple(range(ay.ndim - 1)))
    ky = 2.0 * np.pi * np.fft.rfftfreq(ny, d=max(dy, 1e-30))
    return ky, power


def compute_pdf(
    field: np.ndarray,
    *,
    bins: int = 100,
    vrange: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = _as_np(field).reshape(-1)
    hist, edges = np.histogram(x, bins=bins, range=vrange, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def compute_cross_coherence_phase(
    x_t: np.ndarray,
    y_t: np.ndarray,
    *,
    dt: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f, coherence, phase) from two scalar time traces."""

    x = _as_np(x_t).reshape(-1)
    y = _as_np(y_t).reshape(-1)
    nt = min(x.size, y.size)
    x = x[:nt]
    y = y[:nt]
    if nt < 8:
        f = np.fft.rfftfreq(max(nt, 2), d=max(dt, 1e-30))
        z = np.zeros_like(f)
        return f, z, z
    if noverlap is None:
        noverlap = nperseg // 2
    segments = _segment_indices(nt, nperseg=nperseg, noverlap=noverlap)
    pxx = None
    pyy = None
    pxy = None
    win = np.hanning(max(8, min(nt, nperseg)))
    for s0, s1 in segments:
        xs = x[s0:s1]
        ys = y[s0:s1]
        w = np.hanning(xs.size) if xs.size != win.size else win
        xs = (xs - np.mean(xs)) * w
        ys = (ys - np.mean(ys)) * w
        xft = np.fft.rfft(xs)
        yft = np.fft.rfft(ys)
        norm = max(np.sum(w * w), 1e-30)
        sx = (np.abs(xft) ** 2) / norm
        sy = (np.abs(yft) ** 2) / norm
        sxy = (xft * np.conj(yft)) / norm
        pxx = sx if pxx is None else (pxx + sx)
        pyy = sy if pyy is None else (pyy + sy)
        pxy = sxy if pxy is None else (pxy + sxy)
    assert pxx is not None and pyy is not None and pxy is not None
    pxx /= float(len(segments))
    pyy /= float(len(segments))
    pxy /= float(len(segments))
    coherence = np.abs(pxy) ** 2 / np.maximum(pxx * pyy, 1e-30)
    phase = np.angle(pxy)
    f = np.fft.rfftfreq(segments[0][1] - segments[0][0], d=max(dt, 1e-30))
    return f, coherence, phase


def compute_radial_particle_flux_profile(
    n: np.ndarray,
    phi: np.ndarray,
    *,
    dy: float,
    B0: float = 1.0,
    axis_y: int = -1,
) -> np.ndarray:
    """Estimate Γ_r(x) = <n vE,r> from vE,r = -∂y phi / B."""

    n_arr = _as_np(n)
    phi_arr = _as_np(phi)
    phi_y = np.moveaxis(phi_arr, axis_y, -1)
    n_y = np.moveaxis(n_arr, axis_y, -1)
    dphi_dy = (np.roll(phi_y, -1, axis=-1) - np.roll(phi_y, 1, axis=-1)) / (2.0 * max(dy, 1e-30))
    ver = -dphi_dy / max(B0, 1e-30)
    gamma = n_y * ver
    # average over all axes except radial (last-1 after move for 2D/3D data)
    if gamma.ndim < 2:
        return gamma
    radial_axis = gamma.ndim - 2
    axes = tuple(i for i in range(gamma.ndim) if i != radial_axis)
    return np.mean(gamma, axis=axes)


def compute_target_fluxes(
    n: np.ndarray,
    vpar_i: np.ndarray,
    Te: np.ndarray,
    *,
    Ti: np.ndarray | None = None,
    gamma_e: float = 5.0,
    gamma_i: float = 3.5,
    axis_par: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate target particle and sheath heat fluxes from end-planes."""

    n_arr = _as_np(n)
    vi_arr = _as_np(vpar_i)
    Te_arr = _as_np(Te)
    Ti_arr = _as_np(Ti) if Ti is not None else np.zeros_like(Te_arr)

    n_p = np.moveaxis(n_arr, axis_par, 1)
    vi_p = np.moveaxis(vi_arr, axis_par, 1)
    Te_p = np.moveaxis(Te_arr, axis_par, 1)
    Ti_p = np.moveaxis(Ti_arr, axis_par, 1)

    left_n = n_p[:, 0]
    right_n = n_p[:, -1]
    left_vi = np.abs(vi_p[:, 0])
    right_vi = np.abs(vi_p[:, -1])
    left_Te = Te_p[:, 0]
    right_Te = Te_p[:, -1]
    left_Ti = Ti_p[:, 0]
    right_Ti = Ti_p[:, -1]

    gamma_t = 0.5 * (left_n * left_vi + right_n * right_vi)
    qe_t = 0.5 * (gamma_e * left_n * left_Te * left_vi + gamma_e * right_n * right_Te * right_vi)
    qi_t = 0.5 * (gamma_i * left_n * left_Ti * left_vi + gamma_i * right_n * right_Ti * right_vi)

    # average over remaining spatial axes, preserving time
    if gamma_t.ndim > 1:
        axes = tuple(range(1, gamma_t.ndim))
        gamma_t = np.mean(gamma_t, axis=axes)
        qe_t = np.mean(qe_t, axis=axes)
        qi_t = np.mean(qi_t, axis=axes)
    return gamma_t, qe_t, qi_t


def finite_run_gate(
    diagnostics: dict[str, np.ndarray],
    *,
    keys: tuple[str, ...] = ("rms_n_fluct", "rms_Te_fluct", "rms_omega_fluct", "rms_phi_fluct"),
    max_growth_factor: float | None = None,
    max_rms_abs: float | None = None,
    ref_index: int = 1,
) -> tuple[bool, str, float, float]:
    """
    Gate run quality using finite values, growth bound, and absolute RMS bound.

    Returns `(passed, reason, growth_max, peak_abs)`.
    """

    growth = 0.0
    peak = 0.0
    for key in keys:
        if key not in diagnostics:
            return False, f"missing:{key}", growth, peak
        arr = _as_np(diagnostics[key]).reshape(-1)
        if arr.size == 0:
            return False, f"empty:{key}", growth, peak
        if not np.all(np.isfinite(arr)):
            return False, f"nonfinite:{key}", growth, peak
        ridx = min(max(ref_index, 0), arr.size - 1)
        ref = max(abs(float(arr[ridx])), 1e-12)
        peak_k = float(np.max(np.abs(arr)))
        growth = max(growth, peak_k / ref)
        peak = max(peak, peak_k)

    if max_growth_factor is not None and growth > float(max_growth_factor):
        return False, "gate_growth", growth, peak
    if max_rms_abs is not None and peak > float(max_rms_abs):
        return False, "gate_peak", growth, peak
    return True, "ok", growth, peak
