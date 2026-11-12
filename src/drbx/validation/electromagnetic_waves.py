"""Slab harness for the electromagnetic four-field model: waves of the production right-hand side against theory.

Slab with Dirichlet radial walls, periodic ``y`` and ``z``, modes ``sin(kx x) exp(i (ky y + kz z))`` and an optional
background ``n0(x) ~ exp(-x/Ln)``. :func:`measured_frequency` seeds the eigenvector of the 4x4 local dispersion matrix
:func:`dispersion_matrix` for ``(phi, n1/n0, v_i, w)`` with ``w = v_e - (mi/me) A_par``, advances the RK4 step of the
production RHS and reads the phase of ``phi``. Scott, Phys. Plasmas 10, 963 (2003); Stegmeir et al.,
Phys. Plasmas 26, 052517 (2019). Used by ``tests/test_electromagnetic_fci.py`` and ``examples/electromagnetic``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry import (
    ConservativeStencilBuilder,
    LocalStencilBuilder,
    build_conservative_stencil_from_field,
    build_curvature_coefficients,
    build_local_stencil_from_field,
    build_open_slab_geometry,
)
from ..native.electromagnetic import EmPhysics, build_fci_em_closure
from ..native.fci_4_field_rhs import Fci4FieldBlobParameters, Fci4FieldState, compute_4field_blob_rhs
from ..native.fci_boundaries import FourFieldBoundaryConditions
from ..native.fci_operators import (
    PerpLaplacianInverseSolver,
    build_perp_laplacian_face_projectors,
    perp_laplacian_conservative_op,
)
from ..native.stellarator_turbulence import radial_dirichlet_face_bc

PERIODIC = (False, True, True)


def slab(shape, lengths, *, beta, mu, te=0.0, em=True, ln=None, iterations=2):
    """Closed slab (x walls Dirichlet, y/z periodic) and the keyword bundle of ``compute_4field_blob_rhs``."""
    g = build_open_slab_geometry(shape, radial_extent=lengths[0], binormal_extent=lengths[1], parallel_length=lengths[2])
    closed = jnp.zeros(shape, bool)
    g = dataclasses.replace(g, maps=dataclasses.replace(g.maps, forward_boundary=closed, backward_boundary=closed))
    local = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
    conservative = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    projectors = build_perp_laplacian_face_projectors(g)
    wall = radial_dirichlet_face_bc(g)
    if ln:  # the density wall values follow n0(x)
        xf = np.asarray(g.grid.x.faces)
        value = np.asarray(wall.value_x).copy()
        value[0], value[-1] = np.exp(-(xf[0] - lengths[0] / 2) / ln), np.exp(-(xf[-1] - lengths[0] / 2) / ln)
        density_wall = dataclasses.replace(wall, value_x=jnp.asarray(value))
    else:
        density_wall = dataclasses.replace(wall, value_x=jnp.ones_like(wall.value_x))
    bcs = FourFieldBoundaryConditions(phi_face_bc=wall, density_face_bc=density_wall, omega_face_bc=wall,
                                      v_ion_parallel_face_bc=wall, v_electron_parallel_face_bc=wall)
    params = Fci4FieldBlobParameters(
        rho_star=1.0, Te=te, mi_over_me=mu, phi_inversion_tol=1e-12, phi_inversion_maxiter=100, phi_inversion_restart=200,
        density_perp_diffusion=0.0, omega_perp_diffusion=0.0, v_ion_parallel_perp_diffusion=0.0,
        v_electron_parallel_perp_diffusion=0.0)
    phi_solver = PerpLaplacianInverseSolver(g, conservative, tol=1e-12, maxiter=100, restart=200,
                                            face_projectors=projectors, periodic_axes=PERIODIC, check_residual=False)
    closure = build_fci_em_closure(
        g, EmPhysics(electromagnetic=em, beta=beta, mi_over_me=mu, ampere_iterations=iterations, ampere_tol=1e-12),
        conservative_stencil_builder=conservative, face_projectors=projectors, periodic_axes=PERIODIC)
    kw = dict(geometry=g, parameters=params, curvature_coefficients=build_curvature_coefficients(g, periodic_axes=PERIODIC),
              stencil_builder=local, conservative_stencil_builder=conservative, boundary_conditions=bcs,
              phi_face_projectors=projectors, phi_inverse_solver=phi_solver, periodic_axes=PERIODIC, electromagnetic=closure)
    return g, kw, phi_solver, bcs


def laplacian(f, g, kw, bcs):
    """The discrete perpendicular Laplacian of the potential inversion (so seeds are consistent with phi = L^-1 omega)."""
    stencil = kw["conservative_stencil_builder"](f, g, periodic_axes=PERIODIC, face_bc=bcs.phi_face_bc)
    return perp_laplacian_conservative_op(stencil, g, face_projectors=kw["phi_face_projectors"], face_bc=bcs.phi_face_bc,
                                          periodic_axes=PERIODIC)


def rhs(state, kw):
    result, _, phi = compute_4field_blob_rhs(state, return_phi=True, **kw)
    return result.rhs, phi


def rk4(state, kw, dt):
    k1 = rhs(state, kw)[0]
    k2 = rhs(state.axpy(k1, scale=0.5 * dt), kw)[0]
    k3 = rhs(state.axpy(k2, scale=0.5 * dt), kw)[0]
    k4 = rhs(state.axpy(k3, scale=dt), kw)[0]
    mean = Fci4FieldState(*[(getattr(k1, f) + 2 * getattr(k2, f) + 2 * getattr(k3, f) + getattr(k4, f)) / 6
                            for f in ("density", "omega", "v_ion_parallel", "v_electron_parallel")])
    return state.axpy(mean, scale=dt)


def dispersion_matrix(kx, ky, kz, beta, mu, te, ln):
    """``d/dt (phi, n1/n0, v_i, w) = M (...)`` at ``n0 = 1`` for ``exp(i (ky y + kz z))``; ``ln=None``: no gradient."""
    k2, de2, ws = kx**2 + ky**2, 1.0 / (beta * mu), (ky / ln if ln else 0.0)
    a, b = k2 * de2 / (1 + k2 * de2), 1 / (1 + k2 * de2)  # v_e = a w + b v_i
    c = ws / (mu * (1 + k2 * de2))  # A_par = (v_i - w) / (mu (1 + k2 de2)); flutter of n0: grad_par ln n -> i (kz nu + ws A)
    i = 1j * kz
    m = np.zeros((4, 4), complex)
    m[0, 2], m[0, 3] = -i * (1 - b) / k2, i * a / k2
    m[1, 0], m[1, 2], m[1, 3] = -1j * ws, -i * b, -i * a
    m[2, 1], m[2, 2], m[2, 3] = -te * i, -1j * te * c, 1j * te * c
    m[3, 0], m[3, 1], m[3, 2], m[3, 3] = mu * i, -mu * te * i, -1j * mu * te * c, 1j * mu * te * c
    return m


def measured_frequency(kx, ky, kz, beta, mu, te, ln, shape, *, amp=1e-5, fraction=0.15):
    """Seed the analytic eigenvector of the largest real frequency, advance with RK4, read the phase of phi."""
    lengths = (np.pi / kx, 2 * np.pi / ky if ky else 1.0, 2 * np.pi / kz)
    g, kw, _, bcs = slab(shape, lengths, beta=beta, mu=mu, te=te, ln=ln)
    x, y, z = (np.asarray(a.centers) for a in (g.grid.x, g.grid.y, g.grid.z))
    omega_all, vectors = np.linalg.eig(dispersion_matrix(kx, ky, kz, beta, mu, te, ln))
    k = int(np.argmax((1j * omega_all).real))
    omega, (phi_h, nu_h, vi_h, w_h) = (1j * omega_all)[k], vectors[:, k]
    theta, profile = ky * y[None, :, None] + kz * z[None, None, :], np.sin(kx * x)[:, None, None]
    n0 = (np.exp(-(x - lengths[0] / 2) / ln) if ln else np.ones_like(x))[:, None, None]
    mode = lambda c: amp * np.real(c * np.exp(1j * theta)) * profile  # noqa: E731
    omega0 = laplacian(jnp.asarray(mode(phi_h)), g, kw, bcs)
    state = Fci4FieldState(density=jnp.asarray(n0 * (1 + mode(nu_h))), omega=omega0,
                           v_ion_parallel=jnp.asarray(mode(vi_h)), v_electron_parallel=jnp.asarray(mode(w_h)))
    steps = 12
    horizon = fraction * 2 * np.pi / abs(omega.real)
    step = jax.jit(lambda s: rk4(s, kw, horizon / steps))
    phase = lambda s: 2 * np.mean(np.asarray(rhs(s, kw)[1]) * profile * np.exp(-1j * theta)) / np.mean(profile**2)  # noqa: E731
    first = phase(state)
    for _ in range(steps):
        state = step(state)
    return omega.real, -np.angle(phase(state) / first) / horizon
