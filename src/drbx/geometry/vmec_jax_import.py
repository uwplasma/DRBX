"""Optional ``vmec_jax`` adapter: wout equilibria, surface fields, field lines.

`vmec_jax <https://github.com/rogeriojorge/vmec_jax>`_ owns the VMEC wout
schema (:func:`vmec_jax.core.wout.read_wout`) and the Fourier-synthesis
conventions of VMEC2000. This adapter imports it from an external checkout
(``DRBX_VMEC_JAX_ROOT``, defaulting to ``~/local/vmec_jax``) the same way
:mod:`drbx.geometry.essos_import` imports ESSOS, and adds the small pieces
`drbx` examples need on top of a loaded wout:

- equilibrium summaries (``nfp``, aspect ratio, iota profile, ``B0``);
- contravariant magnetic-field synthesis ``B^theta``/``B^phi`` and ``|B|`` on
  half-mesh flux surfaces from the Nyquist coefficient tables;
- a JAX RK4 field-line tracer in VMEC ``(s, theta, phi)`` coordinates.  A
  VMEC equilibrium has ``B^s = 0`` by construction, so a field line stays on
  its flux surface and obeys ``d theta / d phi = B^theta / B^phi``; the
  average slope over many toroidal transits is the rotational transform;
- cylindrical ``(R, Z)`` mapping of traced lines and of the LCFS boundary
  through the ``rmnc``/``zmns`` (and asymmetric partner) tables.

The adapter keeps `drbx` importable without vmec_jax: everything raises
``ImportError``/``FileNotFoundError`` lazily and
:func:`vmec_jax_runtime_available` reports availability without raising.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

_PRIVATE_DEFAULT_VMEC_JAX_ROOT = Path.home() / "local" / "vmec_jax"


def _resolve_vmec_jax_root(vmec_jax_root: str | Path | None = None) -> Path:
    if vmec_jax_root is not None:
        return Path(vmec_jax_root).expanduser()
    return Path(os.environ.get("DRBX_VMEC_JAX_ROOT", _PRIVATE_DEFAULT_VMEC_JAX_ROOT)).expanduser()


def _import_vmec_jax_modules(*, vmec_jax_root: str | Path | None = None) -> dict[str, Any]:
    jax.config.update("jax_enable_x64", True)
    root = _resolve_vmec_jax_root(vmec_jax_root)
    if root.exists():
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
    wout_module = importlib.import_module("vmec_jax.core.wout")
    plotting_module = importlib.import_module("vmec_jax.core.plotting")
    return {
        "read_wout": wout_module.read_wout,
        "surface_rz": plotting_module.surface_rz,
        "surface_modB": plotting_module.surface_modB,
        "axis_rz": plotting_module.axis_rz,
    }


def vmec_jax_runtime_available(*, vmec_jax_root: str | Path | None = None) -> bool:
    """Return whether vmec_jax can be imported by the optional adapter."""

    try:
        _import_vmec_jax_modules(vmec_jax_root=vmec_jax_root)
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False
    return True


def load_vmec_jax_wout(path: str | Path, *, vmec_jax_root: str | Path | None = None) -> Any:
    """Read a VMEC ``wout_*.nc`` file with vmec_jax's :func:`read_wout`.

    Returns the :class:`vmec_jax.core.wout.WoutData` dataclass, in file
    conventions (no unit conversions).
    """

    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(
            f"VMEC wout file not found: {resolved}. The pedagogical examples point "
            "at an external checkout (ESSOS_test or vmec_jax examples/data); adjust "
            "the WOUT_PATH parameter to a wout NetCDF file that exists locally."
        )
    modules = _import_vmec_jax_modules(vmec_jax_root=vmec_jax_root)
    return modules["read_wout"](resolved)


def vmec_jax_wout_summary(wout: Any) -> dict[str, Any]:
    """Compact scalar summary of a loaded wout equilibrium."""

    iotaf = np.asarray(wout.iotaf, dtype=np.float64)
    return {
        "nfp": int(wout.nfp),
        "ns": int(wout.ns),
        "mnmax": int(wout.mnmax),
        "lasym": bool(wout.lasym),
        "aspect": float(wout.aspect),
        "major_radius": float(wout.Rmajor_p),
        "minor_radius": float(wout.Aminor_p),
        "b0": float(wout.b0),
        "volavgB": float(wout.volavgB),
        "iota_axis": float(iotaf[0]),
        "iota_edge": float(iotaf[-1]),
        "iota_min": float(np.min(iotaf)),
        "iota_max": float(np.max(iotaf)),
    }


def vmec_jax_half_mesh_s(wout: Any) -> np.ndarray:
    """Normalized toroidal flux ``s`` of the half-mesh rows ``1..ns-1``.

    Row ``j`` of the half-mesh tables (``bsupumnc`` et al.) lives at
    ``s = (j - 1/2) / (ns - 1)``; row 0 is unused padding in the wout schema.
    """

    ns = int(wout.ns)
    return (np.arange(1, ns, dtype=np.float64) - 0.5) / float(ns - 1)


def _half_mesh_nyquist_pair(wout: Any, cos_name: str, sin_name: str, s_index: int) -> tuple[np.ndarray, np.ndarray]:
    ns = int(wout.ns)
    if not 1 <= int(s_index) <= ns - 1:
        raise ValueError(f"half-mesh s_index must be in 1..{ns - 1}, got {s_index}")
    cos_table = np.asarray(getattr(wout, cos_name), dtype=np.float64)[int(s_index)]
    sin_table = getattr(wout, sin_name, None)
    if bool(wout.lasym) and sin_table is not None:
        sin_coeff = np.asarray(sin_table, dtype=np.float64)[int(s_index)]
    else:
        sin_coeff = np.zeros_like(cos_table)
    return cos_table, sin_coeff


def _synthesize(cos_coeff: np.ndarray, sin_coeff: np.ndarray, xm: np.ndarray, xn: np.ndarray,
                theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """``sum_mn [c cos(m theta - n phi) + s sin(...)]`` broadcast over theta/phi.

    ``theta`` and ``phi`` may have any common broadcast shape; ``xn`` is the
    file-convention toroidal mode table (``nfp`` factor included).
    """

    theta_b, phi_b = np.broadcast_arrays(np.asarray(theta, dtype=np.float64), np.asarray(phi, dtype=np.float64))
    angle = (
        np.asarray(xm, dtype=np.float64) * theta_b[..., None]
        - np.asarray(xn, dtype=np.float64) * phi_b[..., None]
    )
    return np.sum(cos_coeff * np.cos(angle) + sin_coeff * np.sin(angle), axis=-1)


def evaluate_vmec_jax_surface_field(
    wout: Any,
    *,
    s_index: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Contravariant field components and ``|B|`` on one half-mesh surface.

    ``theta``/``phi`` broadcast to any common shape.  Returns ``b_sup_theta``
    (``B^theta``), ``b_sup_phi`` (``B^phi``), ``mod_b`` and the surface's
    normalized flux ``s``.  The field-line pitch on the surface is
    ``d theta / d phi = B^theta / B^phi``.
    """

    xm_nyq = np.asarray(wout.xm_nyq, dtype=np.float64)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=np.float64)
    bsupu_c, bsupu_s = _half_mesh_nyquist_pair(wout, "bsupumnc", "bsupumns", s_index)
    bsupv_c, bsupv_s = _half_mesh_nyquist_pair(wout, "bsupvmnc", "bsupvmns", s_index)
    bmod_c, bmod_s = _half_mesh_nyquist_pair(wout, "bmnc", "bmns", s_index)
    return {
        "s": float(vmec_jax_half_mesh_s(wout)[int(s_index) - 1]),
        "b_sup_theta": _synthesize(bsupu_c, bsupu_s, xm_nyq, xn_nyq, theta, phi),
        "b_sup_phi": _synthesize(bsupv_c, bsupv_s, xm_nyq, xn_nyq, theta, phi),
        "mod_b": _synthesize(bmod_c, bmod_s, xm_nyq, xn_nyq, theta, phi),
    }


def trace_vmec_jax_field_lines(
    wout: Any,
    *,
    s_index: int,
    theta0: np.ndarray,
    n_transits: int = 40,
    steps_per_transit: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace field lines on one half-mesh flux surface with a JAX RK4 in phi.

    Integrates ``d theta / d phi = B^theta(s, theta, phi) / B^phi(s, theta, phi)``
    from the Nyquist coefficient tables of half-mesh row ``s_index``, starting
    at poloidal angles ``theta0`` (any number of lines, vectorized) and
    advancing ``n_transits`` toroidal transits with ``steps_per_transit``
    fixed RK4 steps each.

    Returns ``(phi_nodes, theta_lines)`` with shapes ``(n_steps + 1,)`` and
    ``(n_lines, n_steps + 1)``; ``theta_lines`` is unwrapped (not reduced
    modulo ``2 pi``), so its secular slope in phi is the rotational transform.
    """

    xm_nyq = jnp.asarray(np.asarray(wout.xm_nyq, dtype=np.float64))
    xn_nyq = jnp.asarray(np.asarray(wout.xn_nyq, dtype=np.float64))
    bsupu_c, bsupu_s = _half_mesh_nyquist_pair(wout, "bsupumnc", "bsupumns", s_index)
    bsupv_c, bsupv_s = _half_mesh_nyquist_pair(wout, "bsupvmnc", "bsupvmns", s_index)
    bsupu_c = jnp.asarray(bsupu_c)
    bsupu_s = jnp.asarray(bsupu_s)
    bsupv_c = jnp.asarray(bsupv_c)
    bsupv_s = jnp.asarray(bsupv_s)

    n_steps = int(n_transits) * int(steps_per_transit)
    h = 2.0 * jnp.pi / float(steps_per_transit)

    def pitch(theta_value: jax.Array, phi_value: jax.Array) -> jax.Array:
        angle = xm_nyq[None, :] * theta_value[:, None] - xn_nyq[None, :] * phi_value
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        b_sup_theta = cos_a @ bsupu_c + sin_a @ bsupu_s
        b_sup_phi = cos_a @ bsupv_c + sin_a @ bsupv_s
        safe = jnp.where(jnp.abs(b_sup_phi) > 1.0e-30, b_sup_phi, 1.0e-30)
        return b_sup_theta / safe

    def step(theta_value: jax.Array, step_index: jax.Array) -> tuple[jax.Array, jax.Array]:
        phi_value = h * step_index
        k1 = pitch(theta_value, phi_value)
        k2 = pitch(theta_value + 0.5 * h * k1, phi_value + 0.5 * h)
        k3 = pitch(theta_value + 0.5 * h * k2, phi_value + 0.5 * h)
        k4 = pitch(theta_value + h * k3, phi_value + h)
        next_theta = theta_value + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_theta, next_theta

    @jax.jit
    def integrate(theta_start: jax.Array) -> jax.Array:
        _, history = jax.lax.scan(step, theta_start, jnp.arange(n_steps, dtype=jnp.float64))
        return jnp.concatenate([theta_start[None, :], history], axis=0).T

    theta_start = jnp.atleast_1d(jnp.asarray(theta0, dtype=jnp.float64))
    theta_lines = np.asarray(jax.block_until_ready(integrate(theta_start)), dtype=np.float64)
    phi_nodes = np.asarray(float(h)) * np.arange(n_steps + 1, dtype=np.float64)
    return phi_nodes, theta_lines


def traced_rotational_transform(phi_nodes: np.ndarray, theta_lines: np.ndarray) -> np.ndarray:
    """Least-squares slope ``d theta / d phi`` per traced line (= iota).

    VMEC's poloidal angle is not a straight-field-line angle, so the pitch
    oscillates within a transit; the secular slope over many transits
    converges to the rotational transform of the surface.
    """

    phi = np.asarray(phi_nodes, dtype=np.float64)
    theta = np.atleast_2d(np.asarray(theta_lines, dtype=np.float64))
    phi_centered = phi - phi.mean()
    denominator = float(np.sum(phi_centered * phi_centered))
    return np.sum(phi_centered[None, :] * (theta - theta.mean(axis=1, keepdims=True)), axis=1) / denominator


def _full_mesh_coeffs_at_s(wout: Any, cos_name: str, sin_name: str, s: float) -> tuple[np.ndarray, np.ndarray]:
    cos_table = np.asarray(getattr(wout, cos_name), dtype=np.float64)
    ns = cos_table.shape[0]
    s_full = np.linspace(0.0, 1.0, ns)
    s_clipped = float(np.clip(s, 0.0, 1.0))
    cos_coeff = np.array([np.interp(s_clipped, s_full, cos_table[:, k]) for k in range(cos_table.shape[1])])
    sin_table = getattr(wout, sin_name, None)
    if bool(wout.lasym) and sin_table is not None:
        sin_table = np.asarray(sin_table, dtype=np.float64)
        sin_coeff = np.array([np.interp(s_clipped, s_full, sin_table[:, k]) for k in range(sin_table.shape[1])])
    else:
        sin_coeff = np.zeros_like(cos_coeff)
    return cos_coeff, sin_coeff


def vmec_jax_surface_rz(
    wout: Any,
    *,
    s: float,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cylindrical ``(R, Z)`` on the flux surface ``s`` at angles ``theta``/``phi``.

    The full-mesh ``rmnc``/``zmns`` (plus asymmetric partner) tables are
    linearly interpolated in ``s`` mode-by-mode, then synthesized at the
    requested angles (any common broadcast shape).  Use this to map traced
    ``(theta, phi)`` field-line points, e.g. Poincare crossings at ``phi = 0``,
    into the lab frame.
    """

    xm = np.asarray(wout.xm, dtype=np.float64)
    xn = np.asarray(wout.xn, dtype=np.float64)
    rmnc, rmns = _full_mesh_coeffs_at_s(wout, "rmnc", "rmns", s)
    zmns, zmnc = _full_mesh_coeffs_at_s(wout, "zmns", "zmnc", s)
    major_radius = _synthesize(rmnc, rmns, xm, xn, theta, phi)
    vertical = _synthesize(zmnc, zmns, xm, xn, theta, phi)
    return major_radius, vertical


def vmec_jax_boundary_rz(
    wout: Any,
    *,
    phi: float = 0.0,
    n_theta: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed ``(R, Z)`` curve of the LCFS (outermost surface) at fixed ``phi``."""

    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta) + 1)
    return vmec_jax_surface_rz(wout, s=1.0, theta=theta, phi=np.full_like(theta, float(phi)))
