"""Field-provider adapter for the HSX geometry builder.

The builder historically read a MAKEGRID ``mgrid`` file through
:class:`~.Bfield_evaluator.ComponentSplineBFieldEvaluator`. This adapter lets
it consume any :class:`drbx.geometry.field_provider.MagneticFieldProvider`
(ESSOS coils, MAKEGRID filaments, later VMEX + vmec-extender) instead.

``R``/``phi``/``Z`` are not a sampling grid here: the builder uses only
their end points, as the box in which the scalar-potential coordinate is
fitted, and ``phi[0]`` as the period origin.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import spline_filter1d

from .Bfield_evaluator import BFieldEvaluator
from .jax_bfield_evaluator import JaxComponentSplineBFieldEvaluator


@jax.tree_util.register_pytree_node_class
class JaxProviderBFieldEvaluator:
    """JAX view used by the compiled tracer (``evaluate_cartesian`` only)."""

    def __init__(self, provider: Any):
        self.provider = provider

    def tree_flatten(self):
        return (self.provider,), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)

    def evaluate_cartesian(self, points_xyz: Any) -> jax.Array:
        return self.provider.B(jnp.asarray(points_xyz, dtype=jnp.float64))

    __call__ = evaluate_cartesian


class ProviderBFieldEvaluator(BFieldEvaluator):
    """NumPy-facing :class:`BFieldEvaluator` backed by a field provider."""

    method = "provider"
    extrapolate = False

    def __init__(
        self,
        provider: Any,
        *,
        R_bounds: tuple[float, float],
        Z_bounds: tuple[float, float],
        phi0: float = 0.0,
    ):
        self.provider = provider
        self._nfp = int(provider.nfp)
        self._period = 2.0 * np.pi / self._nfp
        self._R = np.asarray(R_bounds, dtype=np.float64)
        self._Z = np.asarray(Z_bounds, dtype=np.float64)
        self._phi = np.array([phi0, phi0 + self._period])
        self._B = jax.jit(lambda p, x: p.B(x))

    @property
    def R(self) -> np.ndarray:
        return self._R

    @property
    def phi(self) -> np.ndarray:
        return self._phi

    @property
    def Z(self) -> np.ndarray:
        return self._Z

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def period(self) -> float:
        return self._period

    @property
    def currents(self) -> np.ndarray:
        return np.asarray(getattr(self.provider, "currents", np.zeros(0)), dtype=np.float64)

    def evaluate_cartesian(self, points_xyz: Any) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float64)
        return np.asarray(self._B(self.provider, jnp.asarray(points)), dtype=np.float64)

    def evaluate_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        points = np.asarray(points_rphiz, dtype=np.float64)
        R, phi, Z = points[..., 0], points[..., 1], points[..., 2]
        b = self.evaluate_cartesian(np.stack((R * np.cos(phi), R * np.sin(phi), Z), axis=-1))
        c, s = np.cos(phi), np.sin(phi)
        return np.stack((b[..., 0] * c + b[..., 1] * s, -b[..., 0] * s + b[..., 1] * c, b[..., 2]), axis=-1)

    def jax_evaluator(self) -> JaxProviderBFieldEvaluator:
        return JaxProviderBFieldEvaluator(self.provider)

    def content_identity(self) -> dict[str, object]:
        """Cache identity: provenance plus a hash of every parameter leaf."""

        digest = hashlib.sha256(b"drbx-field-provider-v1\0")
        for leaf in jax.tree_util.tree_leaves(self.provider):
            array = np.ascontiguousarray(np.asarray(leaf, dtype=np.float64))
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        provenance = json.loads(json.dumps(getattr(self.provider, "provenance", {}), default=str))
        return {"scheme": "field-provider-v1", "sha256": digest.hexdigest(), "provenance": provenance}


def _filter_matrix(n: int, mode: str) -> np.ndarray:
    """The cubic B-spline prefilter of SciPy along one axis, as a matrix."""

    return spline_filter1d(np.eye(n), order=3, axis=0, mode=mode)


def _spline_coefficients(values: jax.Array, method: str) -> jax.Array:
    """JAX version of the ``ComponentSplineBFieldEvaluator`` prefilter.

    ``values`` is one component on the ``(phi, Z, R)`` grid. Reflect padding
    and the prefilter are linear, so the coefficients (the leaves read by
    :class:`JaxComponentSplineBFieldEvaluator`) are differentiable in the
    table values, and they match the SciPy producer bit for bit up to
    round-off.
    """

    pad = 1 if method == "linear" else 4
    padded = jnp.pad(values, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    if method == "linear":
        return padded
    m_phi = _filter_matrix(padded.shape[0], "grid-wrap")
    m_z = _filter_matrix(padded.shape[1], "mirror")
    m_r = _filter_matrix(padded.shape[2], "mirror")
    return jnp.einsum("ip,jq,kr,pqr->ijk", m_phi, m_z, m_r, padded)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoilFieldTable:
    """In-memory MGRID-like table of a field provider, interpolated in JAX.

    The provider (coils) is evaluated once on a uniform ``(R, phi, Z)`` grid
    over one field period; stellarator symmetry halves the evaluations. The
    table is then interpolated with the cubic B-spline of
    :class:`JaxComponentSplineBFieldEvaluator` (the MGRID path's consumer).
    Everything is JAX, so ``jax.grad`` through :meth:`from_provider` and
    :meth:`B` gives derivatives in the provider's parameters (coil currents
    and nodes) through the table values.
    """

    spline: JaxComponentSplineBFieldEvaluator
    nfp: int = field(metadata={"static": True}, default=1)
    stellsym: bool = field(metadata={"static": True}, default=False)
    meta: tuple = field(metadata={"static": True}, default=())

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self.meta)

    @classmethod
    def from_provider(
        cls,
        provider: Any,
        *,
        R_bounds: tuple[float, float],
        Z_bounds: tuple[float, float],
        shape: tuple[int, int, int],
        method: str = "cubic",
    ) -> CoilFieldTable:
        """Tabulate ``provider`` on ``shape = (nR, nphi, nZ)`` points per period.

        With ``provider.stellsym`` the Z grid is made symmetric and only
        ``nphi // 2 + 1`` planes are evaluated; the rest follow from
        ``B_R(R, -phi, -Z) = -B_R``, ``B_phi, B_Z`` even.
        """

        n_r, n_phi, n_z = (int(v) for v in shape)
        nfp = int(provider.nfp)
        stellsym = bool(provider.stellsym)
        if stellsym:
            if n_phi % 2:
                raise ValueError("stellarator-symmetric tables need an even nphi")
            z_max = max(abs(float(Z_bounds[0])), abs(float(Z_bounds[1])))
            Z_bounds = (-z_max, z_max)
        R = np.linspace(float(R_bounds[0]), float(R_bounds[1]), n_r)
        Z = np.linspace(float(Z_bounds[0]), float(Z_bounds[1]), n_z)
        phi = 2.0 * np.pi / nfp * np.arange(n_phi) / n_phi
        n_eval = n_phi // 2 + 1 if stellsym else n_phi
        P, ZZ, RR = np.meshgrid(phi[:n_eval], Z, R, indexing="ij")
        xyz = np.stack((RR * np.cos(P), RR * np.sin(P), ZZ), axis=-1)
        b = provider.B(jnp.asarray(xyz))
        c, s = np.cos(P), np.sin(P)
        values = jnp.stack(
            (b[..., 0] * c + b[..., 1] * s, -b[..., 0] * s + b[..., 1] * c, b[..., 2]), axis=-1
        )
        if stellsym:
            mirror = values[1 : n_phi // 2][::-1, ::-1] * jnp.asarray([-1.0, 1.0, 1.0])
            values = jnp.concatenate((values, mirror), axis=0)
        spline = JaxComponentSplineBFieldEvaluator(
            R=R,
            phi=phi,
            Z=Z,
            coefficients=tuple(_spline_coefficients(values[..., k], method) for k in range(3)),
            currents=np.ones(1),
            nfp=nfp,
            period=2.0 * np.pi / nfp,
            dR=float(R[1] - R[0]),
            dphi=float(phi[1] - phi[0]),
            dZ=float(Z[1] - Z[0]),
            pad=1 if method == "linear" else 4,
            method=method,
        )
        meta = tuple(sorted({**dict(getattr(provider, "meta", ())), "table_shape": (n_r, n_phi, n_z), "table_method": method}.items()))
        return cls(spline=spline, nfp=nfp, stellsym=stellsym, meta=meta)

    def B(self, xyz: jnp.ndarray) -> jnp.ndarray:
        return self.spline.evaluate_cartesian(xyz)

    def inside(self, xyz: jnp.ndarray) -> jnp.ndarray:
        xyz = jnp.asarray(xyz)
        r = jnp.hypot(xyz[..., 0], xyz[..., 1])
        sp = self.spline
        return (r >= sp.R[0]) & (r <= sp.R[-1]) & (xyz[..., 2] >= sp.Z[0]) & (xyz[..., 2] <= sp.Z[-1])


def provider_bfield_evaluator(
    provider: Any,
    *,
    R_bounds: tuple[float, float],
    Z_bounds: tuple[float, float],
) -> ProviderBFieldEvaluator:
    return ProviderBFieldEvaluator(provider, R_bounds=R_bounds, Z_bounds=Z_bounds)


__all__ = ["CoilFieldTable", "JaxProviderBFieldEvaluator", "ProviderBFieldEvaluator", "provider_bfield_evaluator"]
