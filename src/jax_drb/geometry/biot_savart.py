from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


MU0_OVER_4PI = 1.0e-7


@dataclass(frozen=True)
class FourierCoilSet:
    """ESSOS-compatible Fourier coil bundle for JAX Biot-Savart evaluation."""

    base_dofs: jnp.ndarray
    base_currents: jnp.ndarray
    n_segments: int
    nfp: int
    stellsym: bool
    gamma: jnp.ndarray
    gamma_dash: jnp.ndarray
    currents: jnp.ndarray
    metadata: dict[str, int | float | bool | str]

    @property
    def n_coils(self) -> int:
        return int(self.gamma.shape[0])

    @property
    def order(self) -> int:
        return int((self.base_dofs.shape[2] - 1) // 2)


def load_essos_biot_savart_json(path: str | Path, *, n_segments: int | None = None) -> FourierCoilSet:
    """Load an ESSOS ``ESSOS_biot_savart_*.json`` coil file.

    The supported format contains Fourier XYZ curve coefficients under
    ``dofs_curves``, base-coil currents under ``dofs_currents``, and symmetry
    metadata ``nfp`` / ``stellsym``. The implementation intentionally mirrors
    the public ESSOS convention without importing ESSOS at runtime.
    """

    resolved = Path(path)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    segments = int(n_segments if n_segments is not None else data["n_segments"])
    return build_fourier_coil_set(
        base_dofs=jnp.asarray(data["dofs_curves"], dtype=jnp.float64),
        base_currents=jnp.asarray(data["dofs_currents"], dtype=jnp.float64),
        n_segments=segments,
        nfp=int(data["nfp"]),
        stellsym=bool(data["stellsym"]),
        source=resolved.name,
    )


def build_fourier_coil_set(
    *,
    base_dofs: jnp.ndarray,
    base_currents: jnp.ndarray,
    n_segments: int,
    nfp: int = 1,
    stellsym: bool = False,
    source: str = "in_memory",
) -> FourierCoilSet:
    """Build all symmetry-expanded coil curves and derivatives."""

    dofs = jnp.asarray(base_dofs, dtype=jnp.float64)
    currents = jnp.asarray(base_currents, dtype=jnp.float64)
    if dofs.ndim != 3 or dofs.shape[1] != 3 or dofs.shape[2] % 2 != 1:
        raise ValueError("base_dofs must have shape (n_base_coils, 3, 2 * order + 1)")
    if currents.ndim != 1 or currents.shape[0] != dofs.shape[0]:
        raise ValueError("base_currents must have one current per base coil")
    if n_segments <= 2:
        raise ValueError("n_segments must be greater than 2")
    base_gamma, base_gamma_dash = _evaluate_base_fourier_curves(dofs, n_segments)
    gamma, gamma_dash, expanded_currents = _apply_essos_symmetries(
        base_gamma,
        base_gamma_dash,
        currents,
        nfp=int(nfp),
        stellsym=bool(stellsym),
    )
    return FourierCoilSet(
        base_dofs=dofs,
        base_currents=currents,
        n_segments=int(n_segments),
        nfp=int(nfp),
        stellsym=bool(stellsym),
        gamma=gamma,
        gamma_dash=gamma_dash,
        currents=expanded_currents,
        metadata={
            "geometry_family": "essos_biot_savart_coils",
            "source": source,
            "n_base_coils": int(dofs.shape[0]),
            "n_coils": int(gamma.shape[0]),
            "n_segments": int(n_segments),
            "nfp": int(nfp),
            "stellsym": bool(stellsym),
            "order": int((dofs.shape[2] - 1) // 2),
        },
    )


def biot_savart_field(coils: FourierCoilSet, points: jnp.ndarray, *, softening: float = 1.0e-10) -> jnp.ndarray:
    """Evaluate the magnetic field at one or more Cartesian points.

    Parameters
    ----------
    coils:
        Symmetry-expanded Fourier coil bundle.
    points:
        Cartesian point array with final dimension three. Leading dimensions are
        preserved.
    softening:
        Small denominator floor used only to avoid singular values if a sample
        point lands exactly on a filament.
    """

    point_array = jnp.asarray(points, dtype=jnp.float64)
    leading_shape = point_array.shape[:-1]
    flat_points = jnp.reshape(point_array, (-1, 3))
    fields = jax.vmap(lambda point: _biot_savart_field_one(coils.gamma, coils.gamma_dash, coils.currents, point, softening))(
        flat_points
    )
    return jnp.reshape(fields, leading_shape + (3,))


def magnetic_field_magnitude(coils: FourierCoilSet, points: jnp.ndarray) -> jnp.ndarray:
    """Return ``|B|`` at Cartesian points."""

    return jnp.linalg.norm(biot_savart_field(coils, points), axis=-1)


def coil_axis_guess(coils: FourierCoilSet) -> tuple[float, float]:
    """Return a robust cylindrical axis guess from base-coil centroids."""

    base_centers = jnp.mean(coils.gamma[: coils.base_dofs.shape[0]], axis=1)
    radius = jnp.mean(jnp.linalg.norm(base_centers[:, :2], axis=1))
    z_axis = jnp.mean(base_centers[:, 2])
    return float(radius), float(z_axis)


def _evaluate_base_fourier_curves(dofs: jnp.ndarray, n_segments: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    quadpoints = jnp.linspace(0.0, 1.0, int(n_segments), endpoint=False, dtype=jnp.float64)
    gamma = jnp.einsum("ca,s->csa", dofs[:, :, 0], jnp.ones_like(quadpoints))
    gamma_dash = jnp.zeros((dofs.shape[0], int(n_segments), 3), dtype=jnp.float64)
    order = (dofs.shape[2] - 1) // 2
    for mode in range(1, order + 1):
        sin_phase = jnp.sin(2.0 * jnp.pi * mode * quadpoints)
        cos_phase = jnp.cos(2.0 * jnp.pi * mode * quadpoints)
        sin_coeff = dofs[:, :, 2 * mode - 1]
        cos_coeff = dofs[:, :, 2 * mode]
        gamma = gamma + jnp.einsum("ca,s->csa", sin_coeff, sin_phase)
        gamma = gamma + jnp.einsum("ca,s->csa", cos_coeff, cos_phase)
        gamma_dash = gamma_dash + jnp.einsum("ca,s->csa", sin_coeff, 2.0 * jnp.pi * mode * cos_phase)
        gamma_dash = gamma_dash - jnp.einsum("ca,s->csa", cos_coeff, 2.0 * jnp.pi * mode * sin_phase)
    return gamma, gamma_dash


def _apply_essos_symmetries(
    base_gamma: jnp.ndarray,
    base_gamma_dash: jnp.ndarray,
    base_currents: jnp.ndarray,
    *,
    nfp: int,
    stellsym: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    gammas = []
    gamma_dashes = []
    currents = []
    flips = (False, True) if stellsym else (False,)
    for period in range(int(nfp)):
        phi = 2.0 * np.pi * float(period) / float(nfp)
        rotation = jnp.asarray(
            [
                [np.cos(phi), -np.sin(phi), 0.0],
                [np.sin(phi), np.cos(phi), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float64,
        ).T
        for flip in flips:
            transform = rotation
            sign = 1.0
            if flip:
                transform = transform @ jnp.diag(jnp.asarray([1.0, -1.0, -1.0], dtype=jnp.float64))
                sign = -1.0
            gammas.append(jnp.einsum("csa,ab->csb", base_gamma, transform))
            gamma_dashes.append(jnp.einsum("csa,ab->csb", base_gamma_dash, transform))
            currents.append(sign * base_currents)
    return (
        jnp.concatenate(gammas, axis=0),
        jnp.concatenate(gamma_dashes, axis=0),
        jnp.concatenate(currents, axis=0),
    )


def _biot_savart_field_one(
    gamma: jnp.ndarray,
    gamma_dash: jnp.ndarray,
    currents: jnp.ndarray,
    point: jnp.ndarray,
    softening: float,
) -> jnp.ndarray:
    displacement = point[None, None, :] - gamma
    radius = jnp.linalg.norm(displacement, axis=-1)
    denominator = jnp.maximum(radius**3, float(softening) ** 3)
    dfield = jnp.cross(gamma_dash, displacement, axis=-1) / denominator[..., None]
    coil_sum = jnp.sum(currents[:, None, None] * dfield, axis=0)
    return MU0_OVER_4PI * jnp.mean(coil_sum, axis=0)
