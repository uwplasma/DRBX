"""SD1D-matched 1D plasma--neutral detachment model with a steady-state solver.

This module reproduces, term by term, the equations and the finite-volume
discretisation of SD1D (Dudson et al., PPCF 61, 065008, 2019; SD1D commit
e4417531, files ``sd1d.cxx``, ``div_ops.cxx``, ``radiation.cxx``) for the
hydrogen-only configuration of that paper's fixed-ionisation-cost scans
(``Eionize`` 13.6/30/60 eV, no excitation, no impurity). Where the paper and
the source differ, the source is followed.

Equation / term ledger (SD1D normalisation: ``Nnorm`` [m^-3], ``Tnorm`` [eV],
velocities in ``Cs0 = sqrt(e Tnorm / m_i)``, time in ``1/Omega_ci``, lengths in
``rho_s0``; ``P = 2 N T`` with ``Te = Ti = T``; ``A`` is the flux-tube area)::

    dN/dt   = -FS(N, V)                         + S_iz - S_rec + s * N_src
    dNV/dt  = -FS(NV, V) - dP/dl                - F
    dP/dt   = -FS(P, V) - 2/3 P dV/dl + 2/3 Div(kappa dT/dl)
              - 2/3 (R + E)                     + P_src
    dNn/dt  = -FV(Nn, Vn) + S + Div(Dn Nn dlnPn/dl)          + recycling
    dNVn/dt = -FV(NVn, Vn) + F - dPn/dl + Div(Dn NVn dlnPn/dl)
    dPn/dt  = -FV(Pn, Vn) - 2/3 Pn dVn/dl + 2/3 E + 2/3 Div(kappa_n dTn/dl)
              + Div(Dn Pn dlnPn/dl)                         + recycling (3.5 eV)

- ``FS``: flux-split upwind divergence ``(1/A) d(A f v)/dl`` with MinMod face
  reconstruction and wave speed ``a = sqrt(2 T)`` (SD1D ``Div_par_FV_FS``);
  ``FV``: MinMod upwind divergence (``Div_par_FV``); ``Div(K df/dl)`` is the
  area-weighted two-point flux with arithmetic face ``K`` (``Div_par_diffusion``)
  and, for electron conduction, upwinded ``K`` (``Div_par_diffusion_upwind``).
- ``dV/dl``, ``dP/dl``, ``dPn/dl``, ``dVn/dl`` are cell-centred central
  differences **without** the area factor (BOUT++ ``Grad_par``/``Div_par``
  with ``B = 1``), exactly as in SD1D.
- Spitzer ``kappa = 3.2 (m_i/m_e) (P/2) tau_e``, ``tau_e`` with the Coulomb
  logarithm frozen at the normalisation values, no flux limiter.
- Atomic rates are SD1D's ``UpdatedRadiatedPower`` fits (Janev ionisation
  polynomial, AMJUEL-form 9x9 recombination plus radiative and three-body terms,
  9x9 CX fit at E = 10 eV), integrated over each cell with SD1D's Simpson rule
  on face-averaged states. ``R = (Eiz/Tnorm) R_iz + (1.09 T - 13.6/Tnorm) R_rc``,
  ``E = 3/2 (T - Tn) R_cx + 3/2 T R_rc - 3/2 Tn R_iz``,
  ``F = (V - Vn) R_cx + V R_rc - Vn R_iz``, ``S = R_iz - R_rc`` (plasma source).
- Neutral diffusion ``Dn = dneut vth^2 / (nu_cx + nu_iz + nu_nn)``,
  ``vth^2 = max(Tn, tn_floor)``, ``nu_nn = vth / min(1/(Nn a0 pi), 0.1 m)``,
  ``kappa_n = dneut Nn vth^2 / nu``; ``Pn`` relaxes to ``T Nn`` wherever
  ``Nn < 1e-5`` (SD1D's low-density switch).
- Target (sheath) boundary: ``V_out = max(sqrt(2T), V)``; ``N``, ``P`` linearly
  extrapolated to the face; no conduction through the face (``T`` zero gradient);
  energy leaves by advection only, which with ``sheath_gamma = 6`` is SD1D's
  total ``gamma = 6`` (``(gamma - 6) T N V`` extra cooling otherwise).
  Neutral velocity is reflected (no neutral flux through either end).
- Recycling: ``frecycle`` times the actual ion flux through the target face,
  times the face area, is injected into the last cell at 3.5 eV.
- Upstream: no-flow symmetry boundary. Sources: uniform particle and power
  sources over ``length_xpt`` (area 1 there). SD1D's PI controller scales the
  particle source to hold ``N(cell 0) = n_up``; at steady state this is the
  algebraic constraint ``N_0 = n_up`` with the source amplitude ``s`` as the
  extra unknown, which is how it is solved here.

Grid: ``dy_j = (L/ny)(1 + (1-dymin)(1 - y_j/pi))``, ``y_j = 2 pi (j+1/2)/ny``;
area ``A = 1 + (A_t - 1) H(y - y_x)(y - y_x)/(2 pi - y_x)``.

Steady state: pseudo-transient continuation, backward-Euler steps solved by
Newton with the exact Jacobian (colored forward-mode, five-cell stencil) in
2-cell block-tridiagonal form bordered by the source/constraint pair, solved
by a block LU in ``lax.scan`` (forward and transposed). Derivatives of steady outputs come from the
implicit-function theorem (``jax.custom_jvp`` + ``lax.custom_linear_solve``
with the transposed block solve), so ``jax.jacfwd`` and ``jax.grad`` both work
outside ``jit``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.core import eval_context as _eval_context
from jax.scipy.linalg import lu_factor, lu_solve

__all__ = [
    "DetachmentDiagnostics",
    "DetachmentSolParameters",
    "DetachmentSolState",
    "DetachmentSteadyResult",
    "detachment_diagnostics",
    "detachment_initial_state",
    "detachment_ledger",
    "detachment_rhs",
    "detachment_sol_run",
    "detachment_target_outputs",
    "sd1d_charge_exchange_rate",
    "sd1d_grid",
    "sd1d_ionisation_rate",
    "sd1d_recombination_rate",
]

_QE = 1.602176634e-19
_MP = 1.672621923e-27
_ME = 9.1093837015e-31
# BOUT++ 099fe819 SI constants used by SD1D for the normalisation.
_BOUT_QE = 1.602176634e-19
_BOUT_MP = 1.672621898e-27
_BOUT_ME = 9.10938356e-31
_NFIELDS = 6
_SOURCE_INERTIA = 1.0e-4


@dataclass(frozen=True)
class DetachmentSolParameters:
    """SD1D deck parameters (defaults: the ``13.6eV`` scan of Dudson 2019)."""

    ny: int = 800
    length: float = 30.0                 # m
    length_xpt: float = 10.0             # m, source region and area kink
    area_expansion: float = 2.0
    dymin: float = 0.1
    upstream_density: float = 3.0e19     # m^-3, held at cell 0
    power_flux: float = 5.0e7            # W m^-2 over length_xpt
    particle_flux: float = 4.0e23        # m^-2 s^-1, shape of the particle source
    ionisation_energy: float = 13.6      # eV per ionisation (Eionize)
    recycling_fraction: float = 0.99
    neutral_diffusion: float = 10.0      # dneut
    neutral_temperature_floor: float = 0.5  # eV (tn_floor)
    recycled_temperature: float = 3.5    # eV (Franck-Condon)
    max_mean_free_path: float = 0.1      # m
    sheath_gamma: float = 6.0
    charge_exchange: bool = True
    recombination: bool = True
    Nnorm: float = 1.0e20
    Tnorm: float = 100.0
    Bnorm: float = 1.0
    AA: float = 2.0

    @property
    def cs0(self) -> float:
        return math.sqrt(_BOUT_QE * self.Tnorm / (self.AA * _BOUT_MP))

    @property
    def omega_ci(self) -> float:
        return _BOUT_QE * self.Bnorm / (self.AA * _BOUT_MP)

    @property
    def rho_s0(self) -> float:
        return self.cs0 / self.omega_ci


class DetachmentSolState(NamedTuple):
    """SD1D evolved fields (normalised), each of shape ``(ny,)``."""

    ion_density: jnp.ndarray
    ion_momentum: jnp.ndarray
    plasma_pressure: jnp.ndarray
    neutral_density: jnp.ndarray
    neutral_momentum: jnp.ndarray
    neutral_pressure: jnp.ndarray


class DetachmentSteadyResult(NamedTuple):
    state: DetachmentSolState
    source_scale: float          # PI source multiplier s (SD1D ``source``)
    residual: float              # max_v ||dU_v/dt||_inf * tau / ||U_v||_inf
    iterations: int
    converged: bool


class _Grid(NamedTuple):
    dl: jnp.ndarray       # cell length, normalised (ny+4 incl. guards)
    area: jnp.ndarray     # J (ny+4)
    y: jnp.ndarray
    source_mask: jnp.ndarray  # H(y_xpt - y) on interior cells


def sd1d_grid(params: DetachmentSolParameters, *, guards: int = 2):
    """SD1D cell lengths [m], areas and centre positions [m] (interior cells)."""

    g = _grid(params)
    sl = slice(guards, guards + params.ny) if guards == 2 else slice(None)
    dl_m = np.asarray(g.dl) * params.rho_s0
    centres = np.cumsum(dl_m[2:-2]) - 0.5 * dl_m[2:-2]
    return dl_m[sl], np.asarray(g.area)[sl], centres


def _grid(params: DetachmentSolParameters) -> _Grid:
    ny = params.ny
    y = 2.0 * np.pi * (np.arange(-2, ny + 2) + 0.5) / ny
    dy = (params.length / ny) * (1.0 + (1.0 - params.dymin) * (1.0 - y / np.pi))
    src = params.length_xpt / params.length
    a = 1.0 - params.dymin
    y_xpt = np.pi * (2.0 - params.dymin - np.sqrt((2.0 - params.dymin) ** 2 - 4.0 * a * src)) / a
    area = 1.0 + (params.area_expansion - 1.0) * np.heaviside(y - y_xpt, 0.0) * (y - y_xpt) / (2.0 * np.pi - y_xpt)
    mask = (y_xpt - y[2:-2] > 0.0).astype(float)
    return _Grid(jnp.asarray(dy / params.rho_s0), jnp.asarray(area), jnp.asarray(y), jnp.asarray(mask))


# --------------------------------------------------------------------------
# SD1D ``UpdatedRadiatedPower`` hydrogen rates [m^3/s] (radiation.cxx)

_IZ = (-3.271397e1, 1.353656e1, -5.739329, 1.563155, -2.877056e-1, 3.482560e-2,
       -2.631976e-3, 1.119544e-4, -2.039150e-6)

_REC = np.array([
    [-2.855728479302E+01, -7.664042607917E-01, -4.930424003280E-03, -5.386830982777E-03, -1.626039237665E-04, 6.080907650243E-06, 2.101102051942E-05, -2.770717597683E-06, 1.038235939800E-07],
    [3.488563234375E-02, -3.583233366133E-03, -3.620245352252E-03, -9.532840484460E-04, 1.888048628708E-04, -1.014890683861E-05, 2.245676563601E-05, -4.695982369246E-06, 2.523166611507E-07],
    [-2.799644392058E-02, -7.452514292790E-03, 6.958711963182E-03, 4.631753807534E-04, 1.288577690147E-04, -1.145028889459E-04, -2.245624273814E-06, 3.250878872873E-06, -2.145390398476E-07],
    [1.209545317879E-02, 2.709299760454E-03, -2.139257298118E-03, -5.371179699661E-04, -1.634580516353E-05, 5.942193980802E-05, -2.944873763540E-06, -9.387290785993E-07, 7.381435237585E-08],
    [-2.436630799820E-03, -7.745129766167E-04, 4.603883706734E-04, 1.543350502150E-04, -9.601036952725E-06, -1.211851723717E-05, 1.002105099354E-06, 1.392391630459E-07, -1.299713684966E-08],
    [2.837893719800E-04, 1.142444698207E-04, -5.991636837395E-05, -2.257565836876E-05, 3.425262385387E-06, 1.118965496365E-06, -1.291320799814E-07, -1.139093288575E-08, 1.265189576423E-09],
    [-1.886511169084E-05, -9.382783518064E-06, 4.729262545726E-06, 1.730782954588E-06, -4.077019941998E-07, -4.275321573501E-08, 7.786155463269E-09, 5.178505597480E-10, -6.854203970018E-11],
    [6.752155602894E-07, 3.902800099653E-07, -1.993485395689E-07, -6.618240780594E-08, 2.042041097083E-08, 3.708616111085E-10, -2.441127783437E-10, -9.452402157390E-12, 1.836615031798E-12],
    [-1.005893858779E-08, -6.387411585521E-09, 3.352589865190E-09, 1.013364275013E-09, -3.707977721109E-10, 7.068450112690E-12, 3.773208484020E-12, -4.672724022059E-14, -1.640492364811E-14],
])  # MATA[j][i]: j = density power, i = temperature power

_CX = np.array([
    [-1.829079582E1, 1.640252721E-1, 3.364564509E-2, 9.530225559E-3, -8.519413900E-4, -1.247583861E-3, 3.014307546E-4, -2.499323170E-5, 6.932627238E-7],
    [2.169137616E-1, -1.106722014E-1, -1.382158680E-3, 7.348786287E-3, -6.343059502E-4, -1.919569450E-4, 4.075019352E-5, -2.850044983E-6, 6.966822400E-8],
    [4.307131244E-2, 8.948693625E-3, -1.209480567E-2, -3.675019470E-4, 1.039643391E-3, -1.553840718E-4, 2.670827249E-6, 7.695300598E-7, -3.783302282E-8],
    [-5.754895093E-4, 6.062141761E-3, 1.075907882E-3, -8.119301728E-4, 8.911036876E-6, 3.175388950E-5, -4.515123642E-6, 2.187439284E-7, -2.911233952E-9],
    [-1.552077120E-3, -1.210431588E-3, 8.297212634E-4, 1.361661817E-4, -1.008928628E-4, 1.080693990E-5, 5.106059414E-7, -1.299275586E-7, 5.117133050E-9],
    [-1.876800283E-4, -4.052878752E-5, -1.907025663E-4, 1.141663042E-5, 1.775681984E-5, -3.149286924E-6, 3.105491555E-8, 2.274394089E-8, -1.130988251E-9],
    [1.125490271E-4, 2.875900436E-5, 1.338839629E-5, -4.340802793E-6, -7.003521917E-7, 2.318308730E-7, -6.030983538E-9, -1.755944926E-9, 1.005189187E-10],
    [-1.238982763E-5, -2.616998140E-6, -1.171762874E-7, 3.517971869E-7, -4.928692833E-8, 1.756388999E-10, -1.446756796E-10, 7.143183138E-11, -3.989884106E-12],
    [4.163596197E-7, 7.558092849E-8, -1.328404104E-8, -9.170850254E-9, 3.208853884E-9, -3.952740759E-10, 2.739558476E-11, -1.693040209E-12, 6.388219930E-14],
])  # cxcoeffs[i][j]: i = ln T power, j = ln E power
_CX_T = tuple(float(c) for c in _CX @ (np.log(10.0) ** np.arange(9)))  # E = 10 eV


def _poly(coeffs, x):
    out = jnp.zeros_like(x)
    for c in reversed(coeffs):
        out = out * x + c
    return out


def sd1d_ionisation_rate(temperature_ev):
    """SD1D ionisation ``<sigma v>`` [m^3/s] (T floored at 0.025 eV)."""
    t = jnp.maximum(temperature_ev, 0.025)
    return jnp.exp(_poly(_IZ, jnp.log(t))) * 1.0e-6


def sd1d_charge_exchange_rate(temperature_ev):
    """SD1D charge-exchange ``<sigma v>`` [m^3/s] at E = 10 eV."""
    t = jnp.maximum(temperature_ev, 0.025)
    return jnp.exp(_poly(_CX_T, jnp.log(t))) * 1.0e-6


def sd1d_recombination_rate(density_m3, temperature_ev):
    """SD1D recombination ``<sigma v>`` [m^3/s] (zero for n < 1e3 m^-3)."""
    t = jnp.maximum(temperature_ev, 0.025)
    n = jnp.maximum(density_m3, 1.0e3)
    rn = jnp.log(n * 1.0e-14)
    rt = jnp.log(t)
    tpows = [jnp.ones_like(rt)]
    for _ in range(8):
        tpows.append(tpows[-1] * rt)
    suma = jnp.zeros_like(rt)
    npow = jnp.ones_like(rn)
    for j in range(9):
        row = sum(float(_REC[j, i]) * tpows[i] for i in range(9))
        suma = suma + row * npow
        npow = npow * rn
    hav = jnp.exp(suma) * 1.0e-6 / (1.0 + 0.125 * t)
    ry, chi, a = 13.60569, 0.35, 3.92e-20
    b = 3.0e-124 * 1.6e-19 ** (-4.5)
    rad = a * ry**1.5 / (jnp.sqrt(t) * (ry + chi * t))
    return jnp.where(density_m3 < 1.0e3, 0.0, hav + rad + b * n * t ** (-5.0))


# --------------------------------------------------------------------------
# Discrete operators (index space with two guard cells per side)


def _minmod_faces(f):
    """MinMod left/right face values for ext cells 1..n-2 (returns arrays on 1..n-2)."""
    c, m, p = f[1:-1], f[:-2], f[2:]
    a, b = p - c, c - m
    slope = jnp.where(a * b <= 0.0, 0.0, jnp.where(jnp.abs(a) < jnp.abs(b), a, b))
    return c - 0.5 * slope, c + 0.5 * slope


def _face_factor(area):
    return 0.5 * (area[1:] + area[:-1])


def _div_fs(f, v, a, grid):
    """SD1D ``Div_par_FV_FS`` on interior cells; returns (div, boundary-face fluxes)."""
    left, right = _minmod_faces(f)            # ext cells 1..ny+2
    # faces between ext cells k, k+1 for k = 1..ny+1
    rk, lk1 = right[:-1], left[1:]
    v1, v2 = v[1:-2], v[2:-1]
    vpar = 0.5 * (v1 + v2)
    amax = jnp.maximum(a[1:-2], a[2:-1])
    right_part = jnp.where(vpar > amax, rk * vpar, jnp.where(vpar < -amax, 0.0, rk * 0.5 * (vpar + amax)))
    left_part = jnp.where(vpar < -amax, lk1 * vpar, jnp.where(vpar > amax, 0.0, lk1 * 0.5 * (vpar - amax)))
    flux = (right_part + left_part) * _face_factor(grid.area)[1:-1]
    div = (flux[1:] - flux[:-1]) / (grid.dl[2:-2] * grid.area[2:-2])
    return div, flux


def _div_fv(f, v, grid):
    """SD1D ``Div_par_FV`` (MinMod upwind) on interior cells."""
    left, right = _minmod_faces(f)
    rk, lk1 = right[:-1], left[1:]
    vpar = 0.5 * (v[1:-2] + v[2:-1])
    flux = jnp.where(vpar > 0.0, rk * vpar, jnp.where(vpar < 0.0, lk1 * vpar, 0.0)) * _face_factor(grid.area)[1:-1]
    return (flux[1:] - flux[:-1]) / (grid.dl[2:-2] * grid.area[2:-2])


def _grad_c(f, grid):
    """BOUT++ C2 ``Grad_par`` / ``Div_par`` (B = 1, no area) on interior cells."""
    return (f[3:-1] - f[1:-3]) / (2.0 * grid.dl[2:-2])


def _div_diffusion(k, f, grid, *, upwind=False):
    """``Div_par_diffusion`` (or ``_upwind``) on interior cells.

    Only interior faces are needed: every use in SD1D has a zero-gradient
    guard (T, Tn, log Pn) so the boundary-face flux vanishes identically.
    """
    ki, fi = k[2:-2], f[2:-2]
    dl = grid.dl[2:-2]
    grad = 2.0 * (fi[1:] - fi[:-1]) / (dl[1:] + dl[:-1])
    if upwind:
        c = jnp.where(grad > 0.0, ki[1:], ki[:-1])
    else:
        c = 0.5 * (ki[1:] + ki[:-1])
    flux = c * _face_factor(grid.area[2:-2]) * grad
    flux = jnp.concatenate([jnp.zeros(1), flux, jnp.zeros(1)])
    return (flux[1:] - flux[:-1]) / (dl * grid.area[2:-2])


def _ext(interior, lo, hi):
    return jnp.concatenate([jnp.stack([lo, lo]), interior, jnp.stack([hi, hi])])


class _Terms(NamedTuple):
    ddt: jnp.ndarray            # (ny, 6)
    target_flux: jnp.ndarray    # Nout * Vout (normalised flux density at the face)
    target_area: jnp.ndarray
    S: jnp.ndarray
    R: jnp.ndarray
    E: jnp.ndarray
    F: jnp.ndarray
    Riz: jnp.ndarray
    Rrec: jnp.ndarray
    Ecx: jnp.ndarray
    ion_flux: jnp.ndarray       # FS face fluxes of N (area-weighted), ny+1 faces
    energy_flux: jnp.ndarray    # FS face fluxes of P (area-weighted)
    nvi_flux: jnp.ndarray
    Te: jnp.ndarray
    Vout: jnp.ndarray
    Nout: jnp.ndarray
    Pout: jnp.ndarray
    recycle: jnp.ndarray


def _terms(u, s, theta, params: DetachmentSolParameters, grid: _Grid) -> _Terms:
    """SD1D ``rhs`` on interior cells for state ``u`` (ny, 6), source scale ``s``.

    ``theta = (n_up [m^-3], power_flux [W/m^2])``; only ``power_flux`` enters here.
    """
    Tn_ = params.Tnorm
    Nn_ = params.Nnorm
    om = params.omega_ci
    mi_me = params.AA * _BOUT_MP / _BOUT_ME
    coulomb = 6.6 - 0.5 * math.log(Nn_ * 1e-20) + 1.5 * math.log(Tn_)
    tau_e0 = 1.0 / (2.91e-6 * (Nn_ / 1e6) * coulomb * Tn_ ** (-1.5))

    Ne = jnp.maximum(u[:, 0], 1e-10)
    NVi = u[:, 1]
    P = jnp.maximum(u[:, 2], 1e-10)
    Nn = jnp.maximum(u[:, 3], 1e-10)
    NVn = u[:, 4]
    Pn = u[:, 5]
    Vi = NVi / Ne
    Te = 0.5 * P / Ne
    Nnlim = jnp.maximum(Nn, 1e-5)
    Nelim = jnp.maximum(Ne, 1e-5)
    Vn = NVn / Nnlim
    Tn = jnp.maximum(Pn / Nnlim, 1e-12)

    # Coefficients
    tau_e = om * tau_e0 * Te**1.5 / Ne
    kappa = 3.2 * mi_me * 0.5 * P * tau_e
    sig_cx = Nelim * Nn_ * sd1d_charge_exchange_rate(Te * Tn_) / om
    sig_iz = Nelim * Nn_ * sd1d_ionisation_rate(Te * Tn_) / om
    vth2 = jnp.maximum(Tn, params.neutral_temperature_floor / Tn_)
    a0 = math.pi * 5.29e-11**2
    lam = jnp.minimum(1.0 / (Nn_ * Nnlim * a0), params.max_mean_free_path) / params.rho_s0
    sigma = sig_cx + sig_iz + jnp.sqrt(vth2) / lam
    Dn = params.neutral_diffusion * vth2 / sigma
    kappa_n = params.neutral_diffusion * Nnlim * vth2 / sigma

    # Sheath (upper) and symmetry (lower) guards
    Vout = jnp.maximum(jnp.sqrt(2.0 * Te[-1]), Vi[-1])
    Nout = jnp.maximum(0.5 * (3.0 * Ne[-1] - Ne[-2]), 0.0)
    Pout = jnp.maximum(0.5 * (3.0 * P[-1] - P[-2]), 0.0)
    Ne_x = _ext(Ne, Ne[0], 2.0 * Nout - Ne[-1])
    Vi_x = _ext(Vi, -Vi[0], 2.0 * Vout - Vi[-1])
    NVi_x = _ext(NVi, -NVi[0], Nout * Vout)
    Te_x = _ext(Te, Te[0], Te[-1])
    P_x = _ext(P, P[0], 2.0 * Pout - P[-1])
    Nn_g = 2.0 * Nn[-1] - Nn[-2]
    Nn_x = _ext(Nn, Nn[0], Nn_g)
    Vn_x = _ext(Vn, -Vn[0], -Vn[-1])
    Tn_x = _ext(Tn, Tn[0], Tn[-1])
    Pn_x = _ext(Pn, Pn[0], Nn_g * Tn[-1])
    NVn_x = _ext(NVn, NVn[0], -NVn[-1])
    a_x = jnp.sqrt(2.0 * Te_x)
    J = grid.area

    # Atomic sources (Simpson rule on face averages)
    def lcr(x):
        return 0.5 * (x[1:-3] + x[2:-2]), x[2:-2], 0.5 * (x[2:-2] + x[3:-1])

    TeL, TeC, TeR = lcr(Te_x)
    NeL, NeC, NeR = lcr(Ne_x)
    ViL, ViC, ViR = lcr(Vi_x)
    TnL, TnC, TnR = lcr(Tn_x)
    NnL, NnC, NnR = lcr(jnp.maximum(Nn_x, 0.0))
    VnL, VnC, VnR = lcr(Vn_x)
    JL, JC, JR = lcr(J)

    def simpson(fn):
        return (JL * fn(0) + 4.0 * JC * fn(1) + JR * fn(2)) / (6.0 * JC)

    Te3, Ne3, Vi3, Tn3, Nn3, Vn3 = (TeL, TeC, TeR), (NeL, NeC, NeR), (ViL, ViC, ViR), (TnL, TnC, TnR), (NnL, NnC, NnR), (VnL, VnC, VnR)
    scale = Nn_ / om
    if params.charge_exchange:
        Rcx = [Ne3[k] * Nn3[k] * sd1d_charge_exchange_rate(Te3[k] * Tn_) * scale for k in range(3)]
    else:
        Rcx = [jnp.zeros_like(TeC)] * 3
    if params.recombination:
        Rrc = [sd1d_recombination_rate(Ne3[k] * Nn_, Te3[k] * Tn_) * Ne3[k] ** 2 * scale for k in range(3)]
    else:
        Rrc = [jnp.zeros_like(TeC)] * 3
    Rizr = [Ne3[k] * Nn3[k] * sd1d_ionisation_rate(Te3[k] * Tn_) * scale for k in range(3)]

    Ecx = 1.5 * simpson(lambda k: (Te3[k] - Tn3[k]) * Rcx[k])
    Fcx = simpson(lambda k: (Vi3[k] - Vn3[k]) * Rcx[k])
    Rrec = simpson(lambda k: (1.09 * Te3[k] - 13.6 / Tn_) * Rrc[k])
    Erec = 1.5 * simpson(lambda k: Te3[k] * Rrc[k])
    Frec = simpson(lambda k: Vi3[k] * Rrc[k])
    Srec = simpson(lambda k: Rrc[k])
    Riz = (params.ionisation_energy / Tn_) * simpson(lambda k: Rizr[k])
    Eiz = -1.5 * simpson(lambda k: Tn3[k] * Rizr[k])
    Fiz = -simpson(lambda k: Vn3[k] * Rizr[k])
    Siz = -simpson(lambda k: Rizr[k])
    R = Rrec + Riz
    E = Ecx + Erec + Eiz
    F = Frec + Fiz + Fcx
    S = Srec + Siz

    # Sources
    ne_src = params.particle_flux / params.length_xpt * grid.source_mask / (Nn_ * om)
    pe_src = theta[1] * (2.0 / 3.0) / params.length_xpt * grid.source_mask / (_BOUT_QE * Nn_ * Tn_ * om)

    # Plasma
    div_n, flux_n = _div_fs(Ne_x, Vi_x, a_x, grid)
    div_nv, flux_nv = _div_fs(NVi_x, Vi_x, a_x, grid)
    div_p, flux_p = _div_fs(P_x, Vi_x, a_x, grid)
    ddt_n = -div_n - S + s * ne_src
    ddt_nv = -div_nv - _grad_c(P_x, grid) - F
    ddt_p = -div_p - (2.0 / 3.0) * P * _grad_c(Vi_x, grid) - (2.0 / 3.0) * (R + E) + pe_src
    ddt_p = ddt_p + (2.0 / 3.0) * _div_diffusion(_ext(kappa, kappa[0], kappa[-1]), Te_x, grid, upwind=True)
    face_t = 0.5 * (J[-3] + J[-2])
    q_extra = (params.sheath_gamma - 6.0) * Te[-1] * Nout * Vout
    ddt_p = ddt_p.at[-1].add(-(2.0 / 3.0) * q_extra * face_t / (grid.dl[-3] * J[-3]))
    low = (Ne < 1e-5) & (ddt_n < 0.0)
    ddt_n = jnp.where(low, 0.0, ddt_n)
    ddt_nv = jnp.where(low, 0.0, ddt_nv)
    ddt_p = jnp.where(low, 0.0, ddt_p)

    # Neutrals
    logPn = jnp.log(jnp.maximum(Pn, 1e-7))
    logPn_x = _ext(logPn, logPn[0], logPn[-1])
    ext = lambda x: _ext(x, x[0], x[-1])
    ddt_nn = -_div_fv(Nn_x, Vn_x, grid) + S + _div_diffusion(ext(Dn * Nn), logPn_x, grid)
    ddt_nvn = -_div_fv(NVn_x, Vn_x, grid) + F - _grad_c(Pn_x, grid) + _div_diffusion(ext(NVn * Dn), logPn_x, grid)
    ddt_pn = (-_div_fv(Pn_x, Vn_x, grid) - (2.0 / 3.0) * Pn * _grad_c(Vn_x, grid) + (2.0 / 3.0) * E
              + (2.0 / 3.0) * _div_diffusion(ext(kappa_n), Tn_x, grid) + _div_diffusion(ext(Dn * Pn), logPn_x, grid))
    ddt_pn = jnp.where(Nn < 1e-5, -1e-2 * (Pn - Te * Nn), ddt_pn)

    # Recycling of the actual target ion flux
    recycle = params.recycling_fraction * Nout * Vout * face_t / (J[-3] * grid.dl[-3])
    ddt_nn = ddt_nn.at[-1].add(recycle)
    ddt_pn = ddt_pn.at[-1].add(recycle * params.recycled_temperature / Tn_)

    ddt = jnp.stack([ddt_n, ddt_nv, ddt_p, ddt_nn, ddt_nvn, ddt_pn], axis=1)
    return _Terms(ddt, Nout * Vout, face_t, S, R, E, F, Riz, Rrec, Ecx, flux_n, flux_p, flux_nv,
                  Te, Vout, Nout, Pout, recycle)


# --------------------------------------------------------------------------
# Block-tridiagonal LU (lax.scan; compile time independent of the block count)


class _BlockLU(NamedTuple):
    lu: jnp.ndarray
    piv: jnp.ndarray
    lower: jnp.ndarray
    upper: jnp.ndarray


def block_thomas_factor(lower, diag, upper):
    """Top-down block LU: ``D'_k = D_k - L_k D'_{k-1}^{-1} U_{k-1}``."""
    first = lu_factor(diag[0])

    def step(carry, inp):
        d_k, l_k, u_prev = inp
        x = lu_solve(carry, u_prev)
        new = lu_factor(d_k - l_k @ x)
        return new, new

    _, (lus, pivs) = jax.lax.scan(step, first, (diag[1:], lower[1:], upper[:-1]))
    return _BlockLU(jnp.concatenate([first[0][None], lus]), jnp.concatenate([first[1][None], pivs]), lower, upper)


def block_thomas_solve(f: _BlockLU, rhs, transpose=False):
    """Solve ``A x = rhs`` (or ``A^T x = rhs``) with the factors; rhs ``(nb, m)``."""
    lus, pivs = f.lu, f.piv
    if not transpose:
        def down(y_prev, inp):
            lu, piv, l_k, r_k = inp
            y = r_k - l_k @ lu_solve((lu, piv), y_prev)
            return y, y

        _, ys = jax.lax.scan(down, rhs[0], (lus[:-1], pivs[:-1], f.lower[1:], rhs[1:]))
        ys = jnp.concatenate([rhs[:1], ys])
        x_last = lu_solve((lus[-1], pivs[-1]), ys[-1])

        def up(x_next, inp):
            lu, piv, u_k, y_k = inp
            x = lu_solve((lu, piv), y_k - u_k @ x_next)
            return x, x

        _, xs = jax.lax.scan(up, x_last, (lus[:-1], pivs[:-1], f.upper[:-1], ys[:-1]), reverse=True)
        return jnp.concatenate([xs, x_last[None]])
    z0 = lu_solve((lus[0], pivs[0]), rhs[0], trans=1)

    def down_t(z_prev, inp):
        lu, piv, u_prev, r_k = inp
        z = lu_solve((lu, piv), r_k - u_prev.T @ z_prev, trans=1)
        return z, z

    _, zs = jax.lax.scan(down_t, z0, (lus[1:], pivs[1:], f.upper[:-1], rhs[1:]))
    zs = jnp.concatenate([z0[None], zs])

    def up_t(x_next, inp):
        lu, piv, l_next, z_k = inp
        x = z_k - lu_solve((lu, piv), l_next.T @ x_next, trans=1)
        return x, x

    _, xs = jax.lax.scan(up_t, zs[-1], (lus[:-1], pivs[:-1], f.lower[1:], zs[:-1]), reverse=True)
    return jnp.concatenate([xs, zs[-1][None]])


# --------------------------------------------------------------------------
# Steady-state system G(x, theta) = [ddt(u); N_0 - n_up] = 0,  x = (u, s)


def _theta(params: DetachmentSolParameters):
    return jnp.asarray([params.upstream_density, params.power_flux], dtype=jnp.float64)


def _residual(u, s, theta, params, grid):
    ddt = _terms(u, s, theta, params, grid).ddt
    return ddt, u[0, 0] - theta[0] / params.Nnorm


def detachment_rhs(state: DetachmentSolState, source_scale, params: DetachmentSolParameters):
    """SD1D time derivatives of the six fields (normalised), shape ``(ny, 6)``."""
    u = jnp.stack(state, axis=1)
    return _terms(u, source_scale, _theta(params), params, _grid(params)).ddt


def _scales(u):
    """Per-field magnitudes; momenta are measured against density x Cs0."""
    m = jnp.max(jnp.abs(u), axis=0)
    floor = jnp.stack([m[0], m[0], m[2], m[3], m[3], m[5]])
    return jnp.maximum(jnp.maximum(m, floor), 1e-30)


def _residual_norm(ddt, u, params, con=0.0):
    """``max_v ||dU_v/dt||_inf tau / ||U_v||_inf`` together with ``|N_0 - n_up| / ||N||``."""
    tau = params.length / params.rho_s0  # transit time at Cs0, normalised
    scales = _scales(u)
    return jnp.maximum(jnp.max(jnp.max(jnp.abs(ddt), axis=0) * tau / scales), jnp.abs(con) / scales[0])


def _jacobian_blocks(u, s, theta, params, grid):
    """Exact Jacobian of ddt w.r.t. u as 2-cell block-tridiagonal + border column."""
    ny = u.shape[0]
    cells = jnp.arange(ny)
    seeds = []
    for c in range(5):
        for b in range(_NFIELDS):
            seeds.append(jnp.zeros((ny, _NFIELDS)).at[:, b].set((cells % 5 == c).astype(u.dtype)))
    seeds = jnp.stack(seeds)
    f = lambda uu: _terms(uu, s, theta, params, grid).ddt
    jt = jax.vmap(lambda t: jax.jvp(f, (u,), (t,))[1])(seeds)  # (30, ny, 6)
    jt = jt.reshape(5, _NFIELDS, ny, _NFIELDS)               # [color, b, i, a]
    b_col = jax.jvp(lambda ss: _terms(u, ss, theta, params, grid).ddt, (s,), (jnp.ones_like(s),))[1]
    # band[d][i, a, b] = d ddt[i, a] / d u[i + d, b]; the column cell i + d has
    # colour (i + d) % 5, so the entry is selected by an elementwise mask.
    band = {}
    for d in range(-2, 3):
        j = cells + d
        valid = (j >= 0) & (j < ny)
        acc = jnp.zeros((ny, _NFIELDS, _NFIELDS))
        for c in range(5):
            sel = (valid & (j % 5 == c))[:, None, None]
            acc = acc + jnp.where(sel, jnp.transpose(jt[c], (1, 2, 0)), 0.0)  # (i, a, b)
        band[d] = acc
    zero = jnp.zeros((ny // 2, _NFIELDS, _NFIELDS))
    ev = {d: band[d][0::2] for d in band}
    od = {d: band[d][1::2] for d in band}

    def blk(r0c0, r0c1, r1c0, r1c1):
        return jnp.concatenate([jnp.concatenate([r0c0, r0c1], axis=2), jnp.concatenate([r1c0, r1c1], axis=2)], axis=1)

    diag = blk(ev[0], ev[1], od[-1], od[0])
    lower = blk(ev[-2], ev[-1], zero, od[-2])
    upper = blk(ev[2], zero, od[1], od[2])
    return lower, diag, upper, b_col


def _bordered_solve(factors, b_col, rhs_u, rhs_c, *, transpose=False, shift=0.0):
    """Solve [[A, b], [e0^T, 0]] [du; ds] = [rhs_u; rhs_c] (or its transpose)."""
    ny = rhs_u.shape[0]
    shape = (ny // 2, 2 * _NFIELDS)
    if not transpose:
        y = block_thomas_solve(factors, rhs_u.reshape(shape)).reshape(ny, _NFIELDS)
        z = block_thomas_solve(factors, b_col.reshape(shape)).reshape(ny, _NFIELDS)
        ds = (y[0, 0] - rhs_c) / (z[0, 0] - shift)
        return y - z * ds, ds
    # [[A^T, e0], [b^T, 0]] [w; mu] = [rhs_u; rhs_c]
    e0 = jnp.zeros((ny, _NFIELDS)).at[0, 0].set(1.0)
    y = block_thomas_solve(factors, rhs_u.reshape(shape), transpose=True).reshape(ny, _NFIELDS)
    z = block_thomas_solve(factors, e0.reshape(shape), transpose=True).reshape(ny, _NFIELDS)
    mu = (jnp.sum(b_col * y) - rhs_c) / jnp.sum(b_col * z)
    return y - z * mu, mu


@partial(jax.jit, static_argnums=(5,))
def _ptc_step(u, s, theta, inv_dt, grid, params):
    # grid arrays are traced arguments: as embedded constants XLA constant-folds
    # the Jacobian assembly and compile time grows with ny.
    ddt, con = _residual(u, s, theta, params, grid)
    lower, diag, upper, b_col = _jacobian_blocks(u, s, theta, params, grid)
    # backward Euler: (J - I/dt) du + b ds = -ddt ; e0.du = -con
    diag = diag - inv_dt * jnp.eye(2 * _NFIELDS)[None]
    factors = block_thomas_factor(lower, diag, upper)
    # the source amplitude relaxes in pseudo-time like SD1D's controller:
    # e0.du + (eps/dt) ds = -con, which becomes the exact constraint as dt -> inf
    du, ds = _bordered_solve(factors, b_col, -ddt, -con, shift=_SOURCE_INERTIA * inv_dt)
    return du, ds, _residual_norm(ddt, u, params, con)


@partial(jax.jit, static_argnums=(4,))
def _norm_at(u, s, theta, grid, params):
    ddt, con = _residual(u, s, theta, params, grid)
    return _residual_norm(ddt, u, params, con)


def detachment_initial_state(params: DetachmentSolParameters, *, upstream_ev: float = 60.0, target_ev: float = 20.0):
    """Two-point-like starting profile (not a solution)."""
    _, _, x = sd1d_grid(params)
    frac = x / params.length
    t = (target_ev**3.5 + (upstream_ev**3.5 - target_ev**3.5) * (1.0 - frac)) ** (2.0 / 7.0) / params.Tnorm
    n = params.upstream_density / params.Nnorm * (upstream_ev / params.Tnorm) / t * (1.0 - 0.5 * frac**4)
    v = np.sqrt(2.0 * t) * frac**2
    nn = n[-1] * np.exp(-(params.length - x) / 0.3) + 1e-6
    tn = params.recycled_temperature / params.Tnorm
    u = np.stack([n, n * v, 2.0 * n * t, nn, 0.0 * nn, nn * tn], axis=1)
    return DetachmentSolState(*[jnp.asarray(c) for c in u.T])


def _interp_state(state: DetachmentSolState, src: DetachmentSolParameters, dst: DetachmentSolParameters):
    _, _, xs = sd1d_grid(src)
    _, _, xd = sd1d_grid(dst)
    return DetachmentSolState(*[jnp.asarray(np.interp(xd, xs, np.asarray(f))) for f in state])


def detachment_sol_run(params: DetachmentSolParameters, state: DetachmentSolState | None = None, *,
                       source_scale: float = 0.1, tol: float = 1e-8, max_iter: int = 400,
                       initial_dt: float = 1.0e2, newton_switch: float = 1e-5, source_params: DetachmentSolParameters | None = None,
                       verbose: bool = False) -> DetachmentSteadyResult:
    """Solve for the SD1D steady state by pseudo-transient Newton continuation.

    ``state`` is a starting guess (on ``source_params``'s grid if given, else on
    ``params``'s grid); ``None`` uses :func:`detachment_initial_state`. Steps are
    backward-Euler with time step ``dt`` (units ``1/Omega_ci``) grown by the
    residual ratio (switched evolution relaxation) and cut on rejection; the run
    stops when the scaled residual ``max_v ||dU_v/dt|| tau / ||U_v||`` (``tau``
    the sound transit time) falls below ``tol``.
    """
    if state is None and params.ny > 100:
        coarse = replace(params, ny=100)
        r0 = detachment_sol_run(coarse, None, source_scale=source_scale, tol=tol, max_iter=max_iter)
        state, source_scale, source_params = r0.state, r0.source_scale, coarse
    if state is None:
        state = detachment_initial_state(params)
    if source_params is not None and source_params.ny != params.ny:
        state = _interp_state(state, source_params, params)
    u = jnp.stack(state, axis=1)
    s = jnp.asarray(float(source_scale))
    theta = _theta(params)
    dt = float(initial_dt)
    grid = _grid(params)
    res = float(_norm_at(u, s, theta, grid, params))
    it = 0
    for it in range(1, max_iter + 1):
        if res < tol:
            it -= 1
            break
        du, ds, _ = _ptc_step(u, s, theta, 1.0 / dt if dt < 1e300 else 0.0, grid, params)
        un, sn = u + du, s + ds
        ok = bool(jnp.all(jnp.isfinite(un))) and float(jnp.min(un[:, 0])) > 0 and float(jnp.min(un[:, 2])) > 0 \
            and float(jnp.min(un[:, 3])) > -1e-12
        rn = float(_norm_at(un, sn, theta, grid, params)) if ok else np.inf
        if ok and ((dt < 1e299 and rn < 1e2 * res) or rn < res):
            growth = min(4.0, max(res / max(rn, 1e-300), 1.3))
            u, s, res = un, sn, rn
            dt = dt * growth if dt < 1e299 else dt
            if res < newton_switch:
                dt = 1e300  # pure Newton near convergence
        else:
            dt = 0.3 * min(dt, 1e30)
        if verbose:
            print(f"  ptc {it:4d} res={res:.3e} dt={dt:.2e} s={float(s):.4f}")
    return DetachmentSteadyResult(DetachmentSolState(*[u[:, k] for k in range(_NFIELDS)]),
                                  float(s), res, it, res < tol)


# --------------------------------------------------------------------------
# Diagnostics, ledger and implicit derivatives


class DetachmentDiagnostics(NamedTuple):
    target_ion_flux: jnp.ndarray        # Gamma_t [m^-2 s^-1] at the sheath face
    target_ion_flux_area: jnp.ndarray   # Gamma_t * A_t (per unit upstream area)
    target_temperature_ev: jnp.ndarray  # last-cell T [eV]
    target_density: jnp.ndarray         # last-cell N [m^-3]
    target_mach: jnp.ndarray            # last-cell V / sqrt(2T)
    upstream_temperature_ev: jnp.ndarray


def detachment_diagnostics(state: DetachmentSolState, params: DetachmentSolParameters,
                           source_scale=0.0) -> DetachmentDiagnostics:
    """Target ion flux, temperature, density and Mach number in SI units."""
    u = jnp.stack(state, axis=1)
    t = _terms(u, source_scale, _theta(params), params, _grid(params))
    gamma = t.target_flux * params.Nnorm * params.cs0
    te = t.Te
    return DetachmentDiagnostics(
        target_ion_flux=gamma,
        target_ion_flux_area=gamma * t.target_area,
        target_temperature_ev=te[-1] * params.Tnorm,
        target_density=state.ion_density[-1] * params.Nnorm,
        target_mach=(state.ion_momentum[-1] / state.ion_density[-1]) / jnp.sqrt(2.0 * te[-1]),
        upstream_temperature_ev=te[0] * params.Tnorm,
    )


def detachment_ledger(result: DetachmentSteadyResult, params: DetachmentSolParameters) -> dict:
    """Steady particle and power balances, integrated over ``A dl`` (per m^2 upstream).

    Particles [s^-1 m^-2]: plasma source + net ionisation = target ion flux;
    total (plasma + neutral): plasma source = (1 - frecycle) target flux.
    Power [W m^-2]: input = target advected energy + ionisation/recombination
    radiation + net energy to neutrals (E, returned through the neutral channel),
    with the compression and kinetic-exchange terms reported separately.
    """
    st = result.state
    u = jnp.stack(st, axis=1)
    grid = _grid(params)
    t = _terms(u, result.source_scale, _theta(params), params, grid)
    vol = np.asarray(grid.dl[2:-2] * grid.area[2:-2]) * params.rho_s0  # m (per m^2)
    om, nn, cs = params.omega_ci, params.Nnorm, params.cs0
    pnorm = _BOUT_QE * params.Nnorm * params.Tnorm * om          # W m^-3 per normalised P rate
    ne_src = params.particle_flux / params.length_xpt * np.asarray(grid.source_mask) * result.source_scale
    source = float(np.sum(ne_src * vol))
    net_iz = float(np.sum(-np.asarray(t.S) * vol)) * nn * om
    target = float(t.target_flux * t.target_area) * nn * cs
    recycled = params.recycling_fraction * target
    flux_p = np.asarray(t.energy_flux)
    p_in = params.power_flux * float(np.sum(np.asarray(grid.source_mask) * vol)) / params.length_xpt
    # (3/2) P advective flux leaving through the target face, in W m^-2
    adv = 1.5 * float(flux_p[-1]) * pnorm * params.rho_s0
    radiation = float(np.sum(np.asarray(t.R) * vol)) * pnorm
    to_neutrals = float(np.sum(np.asarray(t.E) * vol)) * pnorm
    P = np.asarray(st.plasma_pressure)
    Vx = np.asarray(st.ion_momentum / st.ion_density)
    compression = float(np.sum(P * np.asarray(_grad_c(_ext(jnp.asarray(Vx), -Vx[0], 2 * t.Vout - Vx[-1]), grid)) * vol)) * pnorm
    sheath_extra = (params.sheath_gamma - 6.0) * float(t.Te[-1] * t.target_flux * t.target_area) * pnorm * params.rho_s0
    out = dict(
        particle_source=source, net_ionisation=net_iz, target_flux=target, recycled=recycled,
        particle_imbalance=(source + net_iz - target) / target,
        global_particle_imbalance=(source - (1.0 - params.recycling_fraction) * target) / target,
        power_input=p_in, target_advected_power=adv, radiation=radiation, energy_to_neutrals=to_neutrals,
        compression_work=compression, sheath_extra=sheath_extra,
    )
    out["power_imbalance"] = (p_in - adv - radiation - to_neutrals - compression - sheath_extra) / p_in
    return out


def _target_outputs_from(u, s, theta, params, grid):
    t = _terms(u, s, theta, params, grid)
    return jnp.stack([t.Te[-1] * params.Tnorm, t.target_flux * t.target_area * params.Nnorm * params.cs0])


def detachment_target_outputs(theta, params: DetachmentSolParameters, guess: DetachmentSteadyResult | None = None,
                              *, tol: float = 1e-8):
    """``[T_t (eV), Gamma_t A_t]`` at steady state as a differentiable function of
    ``theta = [n_up (m^-3), power_flux (W/m^2)]``.

    The primal is the Newton solve (concrete values, call outside ``jit``); the
    derivative is the implicit-function tangent ``dx = -G_x^{-1} G_theta dtheta``
    through ``lax.custom_linear_solve`` with the transposed bordered block solve,
    so ``jax.jacfwd`` and ``jax.grad`` both apply.
    """
    grid = _grid(params)
    cache = {}

    def solve_primal(th):
        # The Newton loop runs eagerly on concrete values even when a reverse-mode
        # trace is active (grad traces the jvp rule with concrete primals).
        with _eval_context():
            return _solve_primal(th)

    def _solve_primal(th):
        th = np.asarray(th)
        p = replace(params, upstream_density=float(th[0]), power_flux=float(th[1]))
        start = guess.state if guess is not None else None
        s0 = guess.source_scale if guess is not None else 0.1
        r = detachment_sol_run(p, start, source_scale=s0, tol=tol)
        if not r.converged:
            raise RuntimeError(f"steady solve did not converge (residual {r.residual:.2e})")
        cache["r"] = r
        return jnp.stack(r.state, axis=1), jnp.asarray(r.source_scale)

    @jax.custom_jvp
    def steady(th):
        return solve_primal(th)

    @steady.defjvp
    def steady_jvp(primals, tangents):
        (th,), (dth,) = primals, tangents
        u, s = solve_primal(th)
        lower, diag, upper, b_col = _jacobian_blocks(u, s, th, params, grid)
        factors = block_thomas_factor(lower, diag, upper)

        def g_theta(d):
            return jax.jvp(lambda t_: _residual(u, s, t_, params, grid), (th,), (d,))[1]

        rhs = jax.tree_util.tree_map(lambda x: -x, g_theta(dth))

        def matvec(x):
            du, ds = x
            return jax.jvp(lambda uu, ss: _residual(uu, ss, th, params, grid), (u, s), (du, ds))[1]

        def solve(_, r):
            return _bordered_solve(factors, b_col, r[0], r[1])

        def tsolve(_, r):
            return _bordered_solve(factors, b_col, r[0], r[1], transpose=True)

        dx = jax.lax.custom_linear_solve(matvec, rhs, solve, tsolve)
        return (u, s), dx

    u, s = steady(jnp.asarray(theta, dtype=jnp.float64))
    return _target_outputs_from(u, s, jnp.asarray(theta, dtype=jnp.float64), params, grid)
