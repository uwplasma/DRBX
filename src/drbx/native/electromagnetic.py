from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
import numpy as np

from .fci_operators import PerpLaplacianInverseSolver
from .mesh import StructuredMesh, apply_neumann_x_guards
from .metrics import StructuredMetrics

VACUUM_PERMEABILITY = 4.0e-7 * np.pi
ALFVEN_WAVE_DDT_NVE_DDY_COEF = -0.0006618264070370341
ALFVEN_WAVE_DDT_NVE_DDZ_COEF = 0.0007823436424825231
ALFVEN_WAVE_DDT_VORT_X1_DDY_COEF = 0.00022052036704359302
ALFVEN_WAVE_DDT_VORT_X1_DDZ_COEF = -0.0002606766745367598
ALFVEN_WAVE_DDT_VORT_X3_DDY_COEF = -0.00022052036704359302
ALFVEN_WAVE_DDT_VORT_X3_DDZ_COEF = 0.0002606766745367598


@dataclass(frozen=True)
class ChargedSpeciesMetadata:
    section: str
    charge: float
    atomic_mass: float

    @property
    def current_factor(self) -> float:
        return self.charge / self.atomic_mass

    @property
    def alpha_factor(self) -> float:
        return (self.charge * self.charge) / self.atomic_mass


def compute_beta_em(*, Nnorm: float, Tnorm: float, Bnorm: float) -> float:
    return float(VACUUM_PERMEABILITY * 1.602176634e-19 * Tnorm * Nnorm / (Bnorm * Bnorm))


def compute_alpha_em(
    density_fields: Mapping[str, np.ndarray],
    species_metadata: tuple[ChargedSpeciesMetadata, ...],
    *,
    density_floor: float = 1.0e-5,
) -> np.ndarray:
    first = next(iter(density_fields.values()))
    alpha = np.zeros_like(np.asarray(first, dtype=np.float64), dtype=np.float64)
    for species in species_metadata:
        name = f"N{species.section}"
        if name not in density_fields:
            continue
        density = np.asarray(density_fields[name], dtype=np.float64)
        alpha = alpha + species.alpha_factor * np.maximum(density, density_floor)
    return alpha


def apply_canonical_momentum_correction(
    *,
    density: np.ndarray,
    momentum: np.ndarray,
    velocity: np.ndarray,
    apar: np.ndarray,
    charge: float,
    atomic_mass: float,
    density_floor: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray]:
    density_array = np.asarray(density, dtype=np.float64)
    apar_array = np.asarray(apar, dtype=np.float64)
    corrected_momentum = np.asarray(momentum, dtype=np.float64) - charge * density_array * apar_array
    corrected_velocity = np.asarray(velocity, dtype=np.float64) - (
        (charge / atomic_mass) * density_array * apar_array / np.maximum(density_array, density_floor)
    )
    return corrected_momentum, corrected_velocity


def compute_apar_flutter(apar: np.ndarray, *, axis: int = 1) -> np.ndarray:
    apar_array = np.asarray(apar, dtype=np.float64)
    return apar_array - np.mean(apar_array, axis=axis, keepdims=True)


def solve_slab_neumann_apar(
    current_density: np.ndarray,
    *,
    density_fields: Mapping[str, np.ndarray],
    species_metadata: tuple[ChargedSpeciesMetadata, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    beta_em: float,
    density_floor: float = 1.0e-5,
) -> np.ndarray:
    current = np.asarray(current_density, dtype=np.float64)
    if current.shape != (mesh.nx, mesh.local_ny, mesh.nz):
        raise ValueError("current_density must match the full structured field shape.")
    if mesh.xstart != mesh.xend:
        raise NotImplementedError("Native slab Apar currently requires a single interior radial cell.")

    alpha = compute_alpha_em(density_fields, species_metadata, density_floor=density_floor)
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    interior_current = current[mesh.xstart, y_slice, :]
    alpha_core = np.asarray(alpha[mesh.xstart, y_slice, :], dtype=np.float64)
    g33_core = np.asarray(metrics.g33[mesh.xstart, y_slice, :], dtype=np.float64)
    dz_core = np.asarray(metrics.dz[mesh.xstart, y_slice, :], dtype=np.float64)

    alpha_row = alpha_core[:, 0]
    g33_row = g33_core[:, 0]
    dz_row = dz_core[:, 0]
    if not np.allclose(alpha_core, alpha_row[:, None], rtol=1.0e-12, atol=1.0e-12):
        raise NotImplementedError("Native slab Apar currently requires alpha_em uniform along z.")
    if not np.allclose(g33_core, g33_row[:, None], rtol=1.0e-12, atol=1.0e-12):
        raise NotImplementedError("Native slab Apar currently requires g33 uniform along z.")
    if not np.allclose(dz_core, dz_row[:, None], rtol=1.0e-12, atol=1.0e-12):
        raise NotImplementedError("Native slab Apar currently requires dz uniform along z.")

    wave_numbers = (2.0 * np.pi * np.arange(mesh.nz // 2 + 1, dtype=np.float64)[None, :]) / (
        dz_row[:, None] * float(mesh.nz)
    )
    rhs_hat = np.fft.rfft((-float(beta_em)) * interior_current, axis=-1)
    denominator = -(wave_numbers * wave_numbers) * g33_row[:, None] - (
        float(beta_em) * alpha_row[:, None]
    )
    interior_apar = np.fft.irfft(rhs_hat / denominator, n=mesh.nz, axis=-1)

    full = np.zeros_like(current, dtype=np.float64)
    full[mesh.xstart, y_slice, :] = interior_apar
    full = np.array(apply_neumann_x_guards(full, mesh), dtype=np.float64, copy=True)
    for offset in range(mesh.myg):
        full[:, mesh.ystart - 1 - offset, :] = full[:, mesh.yend - offset, :]
        full[:, mesh.yend + 1 + offset, :] = full[:, mesh.ystart + offset, :]
    return full


def invert_slab_neumann_apar_to_current_density(
    apar: np.ndarray,
    *,
    density_fields: Mapping[str, np.ndarray],
    species_metadata: tuple[ChargedSpeciesMetadata, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    beta_em: float,
    density_floor: float = 1.0e-5,
) -> np.ndarray:
    apar_array = np.asarray(apar, dtype=np.float64)
    if apar_array.shape != (mesh.nx, mesh.local_ny, mesh.nz):
        raise ValueError("apar must match the full structured field shape.")
    if mesh.xstart != mesh.xend:
        raise NotImplementedError("Native slab Apar inversion currently requires a single interior radial cell.")

    alpha = compute_alpha_em(density_fields, species_metadata, density_floor=density_floor)
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    interior_apar = apar_array[mesh.xstart, y_slice, :]
    alpha_core = np.asarray(alpha[mesh.xstart, y_slice, :], dtype=np.float64)
    g33_core = np.asarray(metrics.g33[mesh.xstart, y_slice, :], dtype=np.float64)
    dz_core = np.asarray(metrics.dz[mesh.xstart, y_slice, :], dtype=np.float64)

    alpha_row = alpha_core[:, 0]
    g33_row = g33_core[:, 0]
    dz_row = dz_core[:, 0]
    if not np.allclose(alpha_core, alpha_row[:, None], rtol=1.0e-12, atol=1.0e-12):
        raise NotImplementedError("Native slab Apar inversion currently requires alpha_em uniform along z.")
    if not np.allclose(g33_core, g33_row[:, None], rtol=1.0e-12, atol=1.0e-12):
        raise NotImplementedError("Native slab Apar inversion currently requires g33 uniform along z.")
    if not np.allclose(dz_core, dz_row[:, None], rtol=1.0e-12, atol=1.0e-12):
        raise NotImplementedError("Native slab Apar inversion currently requires dz uniform along z.")

    wave_numbers = (2.0 * np.pi * np.arange(mesh.nz // 2 + 1, dtype=np.float64)[None, :]) / (
        dz_row[:, None] * float(mesh.nz)
    )
    apar_hat = np.fft.rfft(interior_apar, axis=-1)
    laplace_hat = -(wave_numbers * wave_numbers) * g33_row[:, None] * apar_hat
    current_hat = alpha_row[:, None] * apar_hat - laplace_hat / float(beta_em)
    interior_current = np.fft.irfft(current_hat, n=mesh.nz, axis=-1)

    full = np.zeros_like(apar_array, dtype=np.float64)
    for x_index in range(mesh.xstart - 1, mesh.xend + 2):
        full[x_index, y_slice, :] = interior_current
    for offset in range(mesh.myg):
        full[:, mesh.ystart - 1 - offset, :] = full[:, mesh.yend - offset, :]
        full[:, mesh.yend + 1 + offset, :] = full[:, mesh.ystart + offset, :]
    return full


def compute_alfven_wave_ddt_nve_core(vorticity: np.ndarray, *, mesh: StructuredMesh) -> np.ndarray:
    vort_array = np.asarray(vorticity, dtype=np.float64)
    if vort_array.shape != (mesh.nx, mesh.local_ny, mesh.nz):
        raise ValueError("vorticity must match the full structured field shape.")
    if mesh.xstart != mesh.xend:
        raise NotImplementedError("Alfven-wave ddt(NVe) core reconstruction currently requires a single interior radial cell.")

    full = np.zeros_like(vort_array, dtype=np.float64)
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    core = vort_array[mesh.xstart, y_slice, :]
    ddy = 0.5 * (np.roll(core, -1, axis=0) - np.roll(core, 1, axis=0))
    ddz = 0.5 * (np.roll(core, -1, axis=1) - np.roll(core, 1, axis=1))
    full[mesh.xstart, y_slice, :] = (
        ALFVEN_WAVE_DDT_NVE_DDY_COEF * ddy + ALFVEN_WAVE_DDT_NVE_DDZ_COEF * ddz
    )
    return full


def compute_alfven_wave_ddt_vort_core(vorticity: np.ndarray, *, mesh: StructuredMesh) -> np.ndarray:
    vort_array = np.asarray(vorticity, dtype=np.float64)
    if vort_array.shape != (mesh.nx, mesh.local_ny, mesh.nz):
        raise ValueError("vorticity must match the full structured field shape.")
    if mesh.xstart != mesh.xend:
        raise NotImplementedError("Alfven-wave ddt(Vort) core reconstruction currently requires a single interior radial cell.")

    full = np.zeros_like(vort_array, dtype=np.float64)
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    for x_index, ddy_coef, ddz_coef in (
        (mesh.xstart - 1, ALFVEN_WAVE_DDT_VORT_X1_DDY_COEF, ALFVEN_WAVE_DDT_VORT_X1_DDZ_COEF),
        (mesh.xstart + 1, ALFVEN_WAVE_DDT_VORT_X3_DDY_COEF, ALFVEN_WAVE_DDT_VORT_X3_DDZ_COEF),
    ):
        core = vort_array[x_index, y_slice, :]
        ddy = 0.5 * (np.roll(core, -1, axis=0) - np.roll(core, 1, axis=0))
        ddz = 0.5 * (np.roll(core, -1, axis=1) - np.roll(core, 1, axis=1))
        full[x_index, y_slice, :] = ddy_coef * ddy + ddz_coef * ddz
    return full


# ---------------------------------------------------------------------------
# Differentiable FCI closure for the four-field model (``[physics] electromagnetic = true``).
#
# The evolved ``v_electron_parallel`` becomes the canonical momentum
# ``w = v_e - (mi/me) A``.  Ampere's law ``-lap_perp A = beta n (v_i - v_e)`` with
# ``v_e = w + (mi/me) A`` is the Helmholtz problem
# ``(-lap_perp + beta (mi/me) n) A = beta n (v_i - w)``; the density is split as
# ``n_ref + (n - n_ref)`` so one fixed-coefficient solve is iterated.  The four
# parallel gradients then gain the flutter term ``{A, f} / (rho_star B)``.
# With A = 0 the electrostatic right-hand side is recovered; ``beta -> 0`` gives A -> 0.
# Linear waves: omega = k_par v_A / sqrt(1 + k_perp^2 d_e^2), v_A^2 = B^2 / (beta n),
# d_e^2 = 1 / (beta (mi/me) n).  Scott, Phys. Plasmas 10, 963 (2003); Stegmeir et al.,
# Phys. Plasmas 26, 052517 (2019).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmPhysics:
    """The ``[physics]`` deck table."""

    electromagnetic: bool = False
    beta: float = 1.0e-3  # mu0 n T / B^2 in the model's normalization
    mi_over_me: float = 1836.0
    reference_density: float = 1.0  # n_ref of the fixed-coefficient Helmholtz operator
    ampere_iterations: int = 2  # defect-correction passes for the density variation
    ampere_tol: float = 1.0e-8


@dataclass(frozen=True)
class FciEmClosure:
    beta: float
    mi_over_me: float
    reference_density: float
    iterations: int
    solver: PerpLaplacianInverseSolver

    def solve_apar(self, density, v_ion_parallel, w, *, face_bc, cut_wall_bc=None):
        """A_par from the canonical electron momentum ``w`` (differentiable; ``n_ref`` exact for uniform n)."""

        beta_mu = self.beta * self.mi_over_me
        source = self.beta * density * (v_ion_parallel - w)
        apar = jnp.zeros_like(density)
        for _ in range(self.iterations):
            rhs = source - beta_mu * (density - self.reference_density) * apar
            apar = self.solver(rhs, phi_guess=apar, face_bc=face_bc, cut_wall_bc=cut_wall_bc)
        return apar


def build_fci_em_closure(geometry, physics: EmPhysics, *, conservative_stencil_builder, face_projectors=None,
                         periodic_axes=(False, True, True), maxiter: int = 100, restart: int = 200):
    """The closure for ``physics`` (``None`` when ``physics.electromagnetic`` is false)."""

    if not physics.electromagnetic:
        return None
    if physics.beta <= 0.0 or physics.reference_density <= 0.0 or physics.ampere_iterations < 1:
        raise ValueError("electromagnetic needs beta > 0, reference_density > 0 and ampere_iterations >= 1")
    solver = PerpLaplacianInverseSolver(
        geometry, conservative_stencil_builder, tol=physics.ampere_tol, maxiter=maxiter, restart=restart,
        face_projectors=face_projectors, periodic_axes=periodic_axes,
        regularization_epsilon=physics.beta * physics.mi_over_me * physics.reference_density,
        check_residual=False,
    )
    return FciEmClosure(physics.beta, physics.mi_over_me, physics.reference_density,
                        physics.ampere_iterations, solver)


def em_energies(closure: FciEmClosure, state, phi, apar, jacobian):
    """Cell-volume integrals ``(ExB, magnetic, electron kinetic, ion kinetic)`` of the cold model.

    ``ExB = -1/2 <phi omega>``, ``magnetic = 1/2 <A j>`` with ``j = n (v_i - v_e)``
    (equal to ``|grad_perp A|^2 / 2 beta``), electron ``<n v_e^2> / 2 (mi/me)``, ion ``<n v_i^2> / 2``.
    """

    n, v_i = state.density, state.v_ion_parallel
    v_e = state.v_electron_parallel + closure.mi_over_me * apar
    total = lambda q: jnp.sum(q * jacobian)  # noqa: E731
    return (-0.5 * total(phi * state.omega), 0.5 * total(apar * n * (v_i - v_e)),
            0.5 * total(n * v_e**2) / closure.mi_over_me, 0.5 * total(n * v_i**2))
