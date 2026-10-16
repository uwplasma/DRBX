"""Diffusive neutral atoms for the HSX FCI Braginskii backend (exhaust PR 2).

Model (one atomic species, hydrogen, ``m_n = m_i``):

- One evolved field, the neutral density ``N`` on the plasma owner cells.
  The neutral temperature ``T_n`` is prescribed (uniform, ``neutral_temperature_ev``)
  and the neutral flow is the diffusive drift only.  A neutral energy equation
  would add a field without a closure to set it at HSX densities; the energy
  the reactions hand to the neutrals beyond ``3/2 T_n`` per atom (and the
  CX momentum) goes to a neutral reservoir that the ledger reports
  (``thermostat``), so total energy is still closed.
- Isotropic diffusion with the full metric (neutrals are not magnetized):
  ``dN/dt = (1/J) d_i (J D g^ij d_j N)`` on the fine logical grid, summed to
  the owner aggregates.  ``D = v_th^2 / (nu_cx + nu_iz)`` with
  ``v_th^2 = T_n / m_i`` and ``nu = n <sigma v>``, capped by a mean free path
  ``lambda <= mfp_max`` (``D <= v_th mfp_max``).  Wall (``u = 1``) and axis
  faces carry no flux (reflecting) unless a wall influx is supplied
  (recycling, PR 3).
- Ionisation, recombination and charge exchange from
  ``drbx.native.neutrals.compute_hydrogen_reaction_sources`` (AMJUEL 2020
  pinned fits, receiver mixing heat, CX frictional heat, electron energy loss
  with the 13.6 eV recombination add-back).  They enter the plasma rows
  through one volumetric source hook, ``apply_volumetric_source``, which
  updates the conserved variables (n, n Vi, 3/2 n Te, 3/2 tau n Ti + n Vi^2/2)
  so particle and energy transfers are exact to round-off.

Units are the backend's: lengths in metres, n in ``n0`` (``reference_density_m3``),
temperatures in ``Te0`` (from rho*), velocities in ``c_s0 = sqrt(Te0/m_i)``,
time in ``1 m / c_s0``.  The ion temperature field is ``Ti / tau`` of the
physical one (the ion energy is ``3/2 tau n Ti``).

The coupling is an operator split (Lie) after each IMEX step; see
``docs/fci_braginskii_hsx_backend.md`` (Neutrals).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ...native.neutrals import PlasmaNormalization, compute_hydrogen_reaction_sources
from ...native.neutrals.atomic_rates import (
    charge_exchange_rate_coefficient,
    rate_coefficient,
)

__all__ = [
    "NEUTRAL_LEDGER_COLUMNS",
    "NeutralGrid",
    "NeutralParameters",
    "VolumetricSource",
    "apply_volumetric_source",
    "build_neutral_grid",
    "inventories",
    "knudsen_map",
    "neutral_diffusion_rate",
    "neutral_diffusivity",
    "neutral_step",
    "rate_validity_counts",
    "reaction_source",
    "reference_temperature_ev",
    "time_unit_s",
]

_E = 1.602176634e-19
_MP = 1.67262192369e-27
# AMJUEL fit validity (atomic_rates clamps to these; we count what falls outside).
_T_RANGE_EV = (0.1, 1.0e4)
_N_RANGE_M3 = (1.0e14, 1.0e22)

NEUTRAL_LEDGER_COLUMNS = (
    "ionisation",  # sum V I dt (events)
    "recombination",  # sum V R dt
    "charge_exchange",  # sum V C dt
    "electron_energy_loss",  # -sum V S_Ee dt (radiation + ionisation potential, net of 13.6 eV add-back)
    "neutral_reservoir_energy",  # sum V (Q_n - 3/2 T_n S_n) dt: energy the fixed-T_n neutrals give away
    "neutral_reservoir_momentum",  # sum V S_Mn dt
    "diffusion_particle_change",  # sum V dN over the diffusion substep (0 up to round-off)
    "wall_influx",  # sum of the supplied wall influx dt (0 until recycling)
    "diffusion_relres",  # backward-Euler residual / rhs (2-norm)
    "max_dt_nu",  # max dt * (ionisation + CX + recombination) frequency per particle
    "out_of_range_Te",
    "out_of_range_ne",
    "out_of_range_Tcx",
)


def reference_temperature_ev(rho_star: float, ion_mass_amu: float = 1.0) -> float:
    """``Te0`` implied by ``rho* = rho_s / 1 m`` at ``B_ref = 1 T``."""

    return float((rho_star * _E * 1.0) ** 2 / (ion_mass_amu * _MP * _E))


def time_unit_s(reference_temperature: float, ion_mass_amu: float = 1.0) -> float:
    """``1 m / c_s0``."""

    return float(1.0 / np.sqrt(_E * reference_temperature / (ion_mass_amu * _MP)))


@dataclass(frozen=True)
class NeutralParameters:
    reference_density_m3: float
    reference_temperature_ev: float
    neutral_temperature_ev: float = 3.0
    mfp_max: float = 0.05  # metres
    tau: float = 1.0
    reaction_substeps: int = 1
    gmres_tolerance: float = 1.0e-10
    gmres_restart: int = 60
    gmres_maxiter: int = 20

    @property
    def neutral_temperature(self) -> float:
        return self.neutral_temperature_ev / self.reference_temperature_ev

    @property
    def normalization(self) -> PlasmaNormalization:
        # compute_hydrogen_reaction_sources scales rates by Nnorm / Omega with
        # Omega = e Bnorm / m_p; choose Bnorm so that 1/Omega is our time unit.
        t0 = time_unit_s(self.reference_temperature_ev)
        return PlasmaNormalization(
            Nnorm=self.reference_density_m3,
            Tnorm=self.reference_temperature_ev,
            Bnorm=_MP / (_E * t0),
            ion_mass=1.0,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NeutralGrid:
    """Fine-face coefficients of the full-metric FV Laplacian plus the owner map.

    ``a_*``: two-point coefficient ``J g^ii dA / dxi_i`` on the faces normal to
    each axis; ``c_*``: cross coefficients ``J g^ij dA`` (j != i).  Faces are
    indexed by their minus cell (x: face i+1/2 at index i, the last x face is
    the wall and is zero; y, z periodic).  ``owner_flat``: flat owner index of
    each fine cell; ``volume``: owner measure (aggregate chart volume, zero on
    aliases); ``owner``: active-owner mask.
    """

    a_x: jnp.ndarray
    c_xy: jnp.ndarray
    c_xz: jnp.ndarray
    a_y: jnp.ndarray
    c_yx: jnp.ndarray
    c_yz: jnp.ndarray
    a_z: jnp.ndarray
    c_zx: jnp.ndarray
    c_zy: jnp.ndarray
    inv_dx2: jnp.ndarray  # 1/(x_{i+1}-x_{i-1}) style centred-difference weights (per x cell)
    dy: float
    dz: float
    owner_flat: jnp.ndarray
    volume: jnp.ndarray
    owner: jnp.ndarray
    cross_x: jnp.ndarray  # face joins two different aggregates
    cross_y: jnp.ndarray
    cross_z: jnp.ndarray
    cell_length: jnp.ndarray  # min physical cell size per fine cell [m]
    wall_area: jnp.ndarray  # physical area of each wall (u = 1) fine face [m^2]
    g_contra: jnp.ndarray  # cell g^11, g^12, g^13, g^22, g^23, g^33 stacked on axis 0

    _ARRAYS = (
        "a_x", "c_xy", "c_xz", "a_y", "c_yx", "c_yz", "a_z", "c_zx", "c_zy", "inv_dx2",
        "owner_flat", "volume", "owner", "cross_x", "cross_y", "cross_z", "cell_length", "wall_area", "g_contra",
    )

    def tree_flatten(self):
        return tuple(getattr(self, k) for k in self._ARRAYS), (self.dy, self.dz)

    @classmethod
    def tree_unflatten(cls, aux, children):
        kwargs = dict(zip(cls._ARRAYS, children))
        return cls(dy=aux[0], dz=aux[1], **kwargs)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.a_x.shape)


def build_neutral_grid(geometry, owner_geometry=None) -> NeutralGrid:
    """Build the isotropic full-metric operator data from a global ``FciGeometry3D``.

    ``owner_geometry`` is the polar angular agglomeration (``owner_index``,
    ``aggregate_id``, ``aggregate_chart_volume``); without it every cell is
    its own owner with volume ``J dx dy dz``.
    """

    grid = geometry.grid
    xc = np.asarray(grid.x.centers, dtype=np.float64)
    dxw = np.asarray(grid.x.widths, dtype=np.float64)
    dy = float(np.asarray(grid.y.widths)[0])
    dz = float(np.asarray(grid.z.widths)[0])
    nx, ny, nz = geometry.shape
    fm = geometry.face_metric

    def comp(metric, name):
        return np.asarray(metric.J, dtype=np.float64) * np.asarray(getattr(metric, name), dtype=np.float64)

    # x faces: index i is face i+1/2 between cells i and i+1 (face array index i+1).
    fx = fm.x
    jg11 = comp(fx, "g11")[1:]
    jg12 = comp(fx, "g12")[1:]
    jg13 = comp(fx, "g13")[1:]
    dxc = np.empty(nx)
    dxc[:-1] = np.diff(xc)
    dxc[-1] = 1.0
    a_x = jg11 * dy * dz / dxc[:, None, None]
    c_xy = jg12 * dy * dz
    c_xz = jg13 * dy * dz
    for arr in (a_x, c_xy, c_xz):
        arr[-1] = 0.0  # wall face (u = 1): reflecting
    # y faces: index j is face j+1/2 between cells j and j+1 (periodic).
    fy = fm.y
    jg22 = comp(fy, "g22")[:, 1:]
    a_y = jg22 * dxw[:, None, None] * dz / dy
    c_yx = comp(fy, "g12")[:, 1:] * dxw[:, None, None] * dz
    c_yz = comp(fy, "g23")[:, 1:] * dxw[:, None, None] * dz
    fz = fm.z
    a_z = comp(fz, "g33")[:, :, 1:] * dxw[:, None, None] * dy / dz
    c_zx = comp(fz, "g13")[:, :, 1:] * dxw[:, None, None] * dy
    c_zy = comp(fz, "g23")[:, :, 1:] * dxw[:, None, None] * dy
    # centred x-difference weights: 1/(x_{i+1}-x_{i-1}), one-sided at the ends
    inv_dx2 = np.empty(nx)
    inv_dx2[1:-1] = 1.0 / (xc[2:] - xc[:-2])
    inv_dx2[0] = 1.0 / (xc[1] - xc[0])
    inv_dx2[-1] = 1.0 / (xc[-1] - xc[-2])

    cm = geometry.cell_metric
    if owner_geometry is None:
        owner_flat = np.arange(nx * ny * nz).reshape(nx, ny, nz)
        volume = np.asarray(cm.J) * dxw[:, None, None] * dy * dz
        owner = np.ones((nx, ny, nz), dtype=bool)
    else:
        topology = owner_geometry.topology
        owner_index = np.asarray(topology.owner_index, dtype=np.int64)
        owner_flat = np.ravel_multi_index(tuple(np.moveaxis(owner_index, -1, 0)), (nx, ny, nz))
        owner = np.asarray(topology.is_active_owner, dtype=bool)
        volume = np.where(owner, np.asarray(owner_geometry.aggregate_chart_volume, dtype=np.float64), 0.0)
    cross_x = owner_flat != np.roll(owner_flat, -1, axis=0)
    cross_x[-1] = False
    cross_y = owner_flat != np.roll(owner_flat, -1, axis=1)
    cross_z = owner_flat != np.roll(owner_flat, -1, axis=2)
    # physical cell sizes h_i = sqrt(g_ii) dxi (covariant metric)
    hx = np.sqrt(np.asarray(cm.g_11)) * dxw[:, None, None]
    hy = np.sqrt(np.asarray(cm.g_22)) * dy
    hz = np.sqrt(np.asarray(cm.g_33)) * dz
    cell_length = np.minimum(np.minimum(hx, hy), hz)
    # wall face area |J grad u| dy dz = J sqrt(g^11) dy dz
    wall_area = np.asarray(fx.J)[-1] * np.sqrt(np.asarray(fx.g11)[-1]) * dy * dz
    j = lambda v, dt=jnp.float64: jnp.asarray(v, dtype=dt)
    return NeutralGrid(
        a_x=j(a_x), c_xy=j(c_xy), c_xz=j(c_xz),
        a_y=j(a_y), c_yx=j(c_yx), c_yz=j(c_yz),
        a_z=j(a_z), c_zx=j(c_zx), c_zy=j(c_zy),
        inv_dx2=j(inv_dx2), dy=dy, dz=dz,
        owner_flat=jnp.asarray(owner_flat, dtype=jnp.int32), volume=j(volume), owner=jnp.asarray(owner),
        cross_x=jnp.asarray(cross_x), cross_y=jnp.asarray(cross_y), cross_z=jnp.asarray(cross_z),
        cell_length=j(cell_length), wall_area=j(wall_area),
        g_contra=j(np.stack([np.asarray(getattr(cm, k)) for k in ("g11", "g12", "g13", "g22", "g23", "g33")])),
    )


# ------------------------------------------------------------------ diffusion
def _fine(owner_field, grid: NeutralGrid):
    return owner_field.ravel()[grid.owner_flat]


def _cell_gradients(f, grid: NeutralGrid):
    """Cell-centred logical derivatives (x one-sided at the ends, y/z periodic)."""

    w = grid.inv_dx2[:, None, None]
    fx = jnp.concatenate(
        (f[1:2] - f[0:1], f[2:] - f[:-2], f[-1:] - f[-2:-1]), axis=0
    ) * w
    fy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * grid.dy)
    fz = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2.0 * grid.dz)
    return fx, fy, fz


def _face_fluxes(field_fine, d_fine, grid: NeutralGrid):
    """Integrated face fluxes ``-D (J g^ij d_j N) dA`` in the +axis direction."""

    f = field_fine
    gx, gy, gz = _cell_gradients(f, grid)
    avg = lambda v, ax: 0.5 * (v + jnp.roll(v, -1, axis=ax))
    dfx, dfy, dfz = avg(d_fine, 0), avg(d_fine, 1), avg(d_fine, 2)
    jump = lambda ax: jnp.roll(f, -1, axis=ax) - f
    flux_x = -dfx * (grid.a_x * jump(0) + grid.c_xy * avg(gy, 0) + grid.c_xz * avg(gz, 0))
    flux_y = -dfy * (grid.a_y * jump(1) + grid.c_yx * avg(gx, 1) + grid.c_yz * avg(gz, 1))
    flux_z = -dfz * (grid.a_z * jump(2) + grid.c_zx * avg(gx, 2) + grid.c_zy * avg(gy, 2))
    # The last x face is the wall (zero coefficients); faces inside one
    # aggregate cancel in the owner sum below, so they need no masking.
    return flux_x, flux_y, flux_z


def _owner_sum(fine_integrated, grid: NeutralGrid):
    n = fine_integrated.size
    total = jax.ops.segment_sum(fine_integrated.ravel(), grid.owner_flat.ravel(), num_segments=n)
    return total.reshape(grid.shape)


def neutral_diffusion_rate(neutral_density, diffusivity, grid: NeutralGrid, wall_influx=None):
    """Owner ``dN/dt`` of the isotropic diffusion (conservative: ``sum V dN/dt = influx``).

    ``wall_influx``: optional particles per unit time entering each fine wall
    face (shape ``grid.shape[1:]``); zero means a reflecting wall.
    """

    f = _fine(neutral_density, grid)
    d = _fine(diffusivity, grid)
    fx, fy, fz = _face_fluxes(f, d, grid)
    net = (jnp.roll(fx, 1, axis=0).at[0].set(0.0) - fx)
    net = net + jnp.roll(fy, 1, axis=1) - fy + jnp.roll(fz, 1, axis=2) - fz
    if wall_influx is not None:
        net = net.at[-1].add(wall_influx)
    integrated = _owner_sum(net, grid)
    return jnp.where(grid.owner, integrated / jnp.where(grid.owner, grid.volume, 1.0), 0.0)


def _diffusion_diagonal(diffusivity, grid: NeutralGrid):
    d = _fine(diffusivity, grid)
    avg = lambda v, ax: 0.5 * (v + jnp.roll(v, -1, axis=ax))
    kx = jnp.where(grid.cross_x, avg(d, 0) * grid.a_x, 0.0)
    ky = jnp.where(grid.cross_y, avg(d, 1) * grid.a_y, 0.0)
    kz = jnp.where(grid.cross_z, avg(d, 2) * grid.a_z, 0.0)
    per_cell = kx + jnp.roll(kx, 1, axis=0).at[0].set(0.0) + ky + jnp.roll(ky, 1, axis=1) + kz + jnp.roll(kz, 1, axis=2)
    total = _owner_sum(per_cell, grid)
    return jnp.where(grid.owner, total / jnp.where(grid.owner, grid.volume, 1.0), 0.0)


def implicit_diffusion(neutral_density, diffusivity, dt, grid: NeutralGrid, params: NeutralParameters, wall_influx=None):
    """Backward Euler for the diffusion, then the conservative update.

    ``N* = (I - dt L)^-1 (N + dt S_wall)`` by preconditioned GMRES, then
    ``N_new = N + dt (L N* + S_wall)`` so the owner sum changes by exactly
    the wall influx regardless of the GMRES residual.
    """

    source = (
        jnp.zeros_like(neutral_density)
        if wall_influx is None
        else neutral_diffusion_rate(jnp.zeros_like(neutral_density), diffusivity, grid, wall_influx)
    )
    rhs = neutral_density + dt * source
    diag = 1.0 + dt * _diffusion_diagonal(diffusivity, grid)

    def operator(v):
        return v - dt * neutral_diffusion_rate(v, diffusivity, grid)

    solution, _ = jax.scipy.sparse.linalg.gmres(
        operator,
        rhs,
        x0=rhs,
        tol=params.gmres_tolerance,
        atol=0.0,
        restart=params.gmres_restart,
        maxiter=params.gmres_maxiter,
        M=lambda v: v / diag,
        solve_method="batched",
    )
    relres = jnp.linalg.norm(operator(solution) - rhs) / jnp.maximum(jnp.linalg.norm(rhs), 1.0e-300)
    new = neutral_density + dt * (neutral_diffusion_rate(solution, diffusivity, grid) + source)
    return jnp.where(grid.owner, new, 0.0), relres


# ------------------------------------------------------------------ reactions
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VolumetricSource:
    """Typed volumetric source on the owner measure (the plasma source hook).

    Densities per unit volume and time in backend units: particles
    ``particle``, parallel ion momentum ``momentum`` (``m_i n0 c_s0``), ion
    thermal energy ``ion_energy`` and electron thermal energy
    ``electron_energy`` (``n0 Te0``).  Kinetic energy is not a separate
    channel: ``apply_volumetric_source`` adds ``Vi S_M - Vi^2 S_n / 2``.
    """

    particle: jnp.ndarray
    momentum: jnp.ndarray
    ion_energy: jnp.ndarray
    electron_energy: jnp.ndarray

    def tree_flatten(self):
        return (self.particle, self.momentum, self.ion_energy, self.electron_energy), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


def apply_volumetric_source(density, Vi, Te, Ti, source: VolumetricSource, dt, *, tau, owner):
    """Advance (n, Vi, Te, Ti) by ``dt * source`` in conserved variables.

    Particle, momentum and total (thermal + kinetic) ion energy and electron
    thermal energy change by exactly ``dt`` times the source (plus the kinetic
    work ``Vi S_M - Vi^2 S_n / 2`` of the ion channel); the primitive fields
    are recovered from them.
    """

    n = density
    n_new = n + dt * source.particle
    safe = jnp.where(owner, n_new, 1.0)
    mom = n * Vi + dt * source.momentum
    Vi_new = mom / safe
    ion_total = 1.5 * tau * n * Ti + 0.5 * n * Vi**2
    ion_total = ion_total + dt * (source.ion_energy + Vi * source.momentum - 0.5 * Vi**2 * source.particle)
    Ti_new = (ion_total - 0.5 * n_new * Vi_new**2) / (1.5 * tau * safe)
    We = 1.5 * n * Te + dt * source.electron_energy
    Te_new = We / (1.5 * safe)
    keep = lambda new, old: jnp.where(owner, new, old)
    return keep(n_new, n), keep(Vi_new, Vi), keep(Te_new, Te), keep(Ti_new, Ti)


def reaction_source(density, Vi, Te, Ti, neutral_density, params: NeutralParameters, owner):
    """AMJUEL hydrogen reaction channels as a plasma ``VolumetricSource`` plus the neutral side."""

    n = jnp.where(owner, jnp.maximum(density, 0.0), 0.0)
    nn = jnp.where(owner, jnp.maximum(neutral_density, 0.0), 0.0)
    Tn = jnp.full_like(n, params.neutral_temperature)
    out = compute_hydrogen_reaction_sources(
        n, Vi, params.tau * jnp.maximum(Ti, 0.0), jnp.maximum(Te, 0.0), nn, jnp.zeros_like(n), Tn,
        normalization=params.normalization,
    )
    plasma = VolumetricSource(
        particle=out.ion_density, momentum=out.ion_momentum,
        ion_energy=out.ion_energy, electron_energy=out.electron_energy,
    )
    return plasma, out


def rate_validity_counts(density, Te, Ti, params: NeutralParameters, owner):
    """Owner cells whose rate-table inputs fall outside the AMJUEL fit range."""

    te_ev = Te * params.reference_temperature_ev
    ne = density * params.reference_density_m3
    tcx = (params.tau * Ti + params.neutral_temperature) * params.reference_temperature_ev
    out = lambda v, lo, hi: jnp.sum(owner & ((v < lo) | (v > hi)))
    return (
        out(te_ev, *_T_RANGE_EV).astype(jnp.float64),
        out(ne, *_N_RANGE_M3).astype(jnp.float64),
        out(tcx, *_T_RANGE_EV).astype(jnp.float64),
    )


def collision_frequencies(density, Te, Ti, params: NeutralParameters):
    """Per-neutral CX and ionisation frequencies (backend time units)."""

    scale = params.reference_density_m3 * time_unit_s(params.reference_temperature_ev)
    n = jnp.maximum(density, 0.0)
    ne_m3 = n * params.reference_density_m3
    te_ev = jnp.maximum(Te, 0.0) * params.reference_temperature_ev
    tcx = (params.tau * jnp.maximum(Ti, 0.0) + params.neutral_temperature) * params.reference_temperature_ev
    nu_iz = n * rate_coefficient("d", "iz", te_ev, ne_m3) * scale
    nu_cx = n * charge_exchange_rate_coefficient(tcx) * scale
    return nu_cx, nu_iz


def neutral_diffusivity(density, Te, Ti, params: NeutralParameters):
    """``D = v_th min(v_th / nu, mfp_max)``, ``v_th^2 = T_n / m_i``; also returns the mfp."""

    nu_cx, nu_iz = collision_frequencies(density, Te, Ti, params)
    vth = jnp.sqrt(params.neutral_temperature)
    mfp = vth / jnp.maximum(nu_cx + nu_iz, 1.0e-300)
    return vth * jnp.minimum(mfp, params.mfp_max), mfp


def inventories(density, Vi, Te, Ti, neutral_density, grid: NeutralGrid, params: NeutralParameters):
    """(nuclei, plasma thermal+kinetic energy, neutral thermal energy) on the owner measure."""

    v = grid.volume
    nuclei = jnp.sum(v * (density + neutral_density))
    plasma = jnp.sum(v * (1.5 * density * Te + 1.5 * params.tau * density * Ti + 0.5 * density * Vi**2))
    neutral = jnp.sum(v * 1.5 * params.neutral_temperature * neutral_density)
    return nuclei, plasma, neutral


def neutral_step(density, Vi, Te, Ti, neutral_density, dt, grid: NeutralGrid, params: NeutralParameters, wall_influx=None):
    """One split neutral step: reactions (explicit, conserved form), then implicit diffusion.

    Returns ``(density, Vi, Te, Ti, neutral_density, ledger, ionisation_rate)``
    with ``ledger`` ordered as ``NEUTRAL_LEDGER_COLUMNS``.
    """

    owner = grid.owner
    v = grid.volume
    sub = int(params.reaction_substeps)
    h = dt / sub
    Tn = params.neutral_temperature
    zero = jnp.zeros((), dtype=jnp.float64)

    def body(_, carry):
        n, vi, te, ti, nn, acc, max_nu = carry
        plasma, out = reaction_source(n, vi, te, ti, nn, params, owner)
        n2, vi2, te2, ti2 = apply_volumetric_source(n, vi, te, ti, plasma, h, tau=params.tau, owner=owner)
        nn2 = jnp.where(owner, nn + h * out.neutral_density, 0.0)
        increments = jnp.stack((
            jnp.sum(v * out.ionization_rate),
            jnp.sum(v * out.recombination_rate),
            jnp.sum(v * out.charge_exchange_rate),
            -jnp.sum(v * out.electron_energy),
            jnp.sum(v * (out.neutral_energy - 1.5 * Tn * out.neutral_density)),
            jnp.sum(v * out.neutral_momentum),
        )) * h
        safe_n = jnp.where(owner & (n > 0), n, 1.0)
        safe_nn = jnp.where(owner & (nn > 0), nn, 1.0)
        per_particle = jnp.maximum(
            (out.ionization_rate + out.charge_exchange_rate) / safe_nn,
            (out.recombination_rate + out.charge_exchange_rate) / safe_n,
        )
        max_nu = jnp.maximum(max_nu, h * jnp.max(jnp.where(owner, per_particle, 0.0)))
        return n2, vi2, te2, ti2, nn2, acc + increments, max_nu

    n, vi, te, ti, nn, acc, max_nu = jax.lax.fori_loop(
        0, sub, body, (density, Vi, Te, Ti, neutral_density, jnp.zeros(6), zero)
    )
    _, out = reaction_source(density, Vi, Te, Ti, neutral_density, params, owner)
    D, _ = neutral_diffusivity(n, te, ti, params)
    D = jnp.where(owner, D, 0.0)
    before = jnp.sum(v * nn)
    nn_new, relres = implicit_diffusion(nn, D, dt, grid, params, wall_influx)
    influx = zero if wall_influx is None else jnp.sum(wall_influx) * dt
    counts = rate_validity_counts(n, te, ti, params, owner)
    ledger = jnp.concatenate((
        acc,
        jnp.stack((jnp.sum(v * nn_new) - before, influx, relres, max_nu)),
        jnp.stack(counts),
    ))
    return n, vi, te, ti, nn_new, ledger, out.ionization_rate


def knudsen_map(density, Te, Ti, neutral_density, grid: NeutralGrid, params: NeutralParameters):
    """Neutral Knudsen numbers on the fine cells.

    ``Kn_cell = mfp / h`` (``h`` the smallest physical cell size) and
    ``Kn_grad = mfp / L`` with ``L = min(L_N, L_n)``, ``L_X = X / |grad X|``
    (physical gradient with the full metric).  The diffusive closure is
    valid where ``Kn_grad << 1``; ``Kn_cell > 1`` means a neutral crosses
    more than a cell between collisions.  Returns ``(mfp, Kn_cell, Kn_grad)``.
    """

    _, mfp = neutral_diffusivity(density, Te, Ti, params)
    g = grid.g_contra

    def grad_length(field):
        f = _fine(field, grid)
        gx, gy, gz = _cell_gradients(f, grid)
        sq = (g[0] * gx * gx + g[3] * gy * gy + g[5] * gz * gz
              + 2.0 * (g[1] * gx * gy + g[2] * gx * gz + g[4] * gy * gz))
        return jnp.abs(f) / jnp.maximum(jnp.sqrt(jnp.maximum(sq, 0.0)), 1.0e-300)

    length = jnp.minimum(grad_length(neutral_density), grad_length(density))
    mfp_f = _fine(mfp, grid)
    return mfp_f, mfp_f / grid.cell_length, mfp_f / jnp.maximum(length, 1.0e-300)
