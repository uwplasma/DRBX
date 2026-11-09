"""Shared pieces of the FCI neutral gas: grid operators, reaction hook, recycling, Knudsen map.

The transport model is ``fci_neutral_fluid.py`` (advanced fluid neutrals, hydrogen, ``m_n = m_i``).  This module holds
what the fluid model and the run share:

- ``NeutralGrid``: full-metric face coefficients of the isotropic operators (nine-point cross terms), the owner map and
  the field direction ``b``; neutrals are not magnetized, so the pressure-diffusion fluxes use ``g^ij``.
- ``reaction_source`` / ``apply_volumetric_source``: ionisation, recombination and charge exchange from
  ``drbx.native.neutrals.compute_hydrogen_reaction_sources`` (AMJUEL 2020 pinned fits) as one plasma source hook that
  updates the conserved variables ``(n, n Vi, 3/2 n Te, 3/2 tau n Ti + n Vi^2/2)``, so transfers are exact to round-off.
- ``recycled_neutral_source`` and ``RecyclingParameters``: ``R`` times the applied sheath ion wall loss, deposited in the
  wall owner cells with the Franck-Condon or wall energy.
- ``knudsen_map``: ``lambda / L`` per cell; the fluid closure needs ``Kn <~ 0.1-0.3``.

Units are the backend's: lengths in metres, n in ``n0`` (``reference_density_m3``), temperatures in ``Te0`` (from
rho*), velocities in ``c_s0 = sqrt(Te0/m_i)``, time in ``1 m / c_s0``.  The ion temperature field is ``Ti / tau`` of the
physical one (the ion energy is ``3/2 tau n Ti``).  The coupling is an operator split (Lie) after each IMEX step.
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
    "NeutralGrid",
    "NeutralParameters",
    "VolumetricSource",
    "apply_volumetric_source",
    "build_neutral_grid",
    "inventories",
    "knudsen_map",
    "neutral_diffusion_rate",
    "RECYCLING_LEDGER_COLUMNS",
    "RecyclingParameters",
    "recycled_neutral_source",
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
    neutral_temperature_ev: float = 3.0  # initial T_n
    tau: float = 1.0
    reaction_substeps: int = 1
    gmres_tolerance: float = 1.0e-10
    gmres_restart: int = 60
    gmres_maxiter: int = 20
    wall_absorption: float = 0.0  # fluid model: probability that a wall-incident atom is absorbed
    wall_energy_reflection: float = 1.0  # fluid model: energy fraction kept by a reflected atom

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
    pb_x: jnp.ndarray  # J b^i dA on the faces normal to each axis (wall face zero): flux of a parallel flow
    pb_y: jnp.ndarray
    pb_z: jnp.ndarray
    b_cell: jnp.ndarray  # cell unit field direction b^i (contravariant) stacked on axis 0

    _ARRAYS = (
        "a_x", "c_xy", "c_xz", "a_y", "c_yx", "c_yz", "a_z", "c_zx", "c_zy", "inv_dx2",
        "owner_flat", "volume", "owner", "cross_x", "cross_y", "cross_z", "cell_length", "wall_area", "g_contra",
        "pb_x", "pb_y", "pb_z", "b_cell",
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
    fb, cb = geometry.face_bfield, geometry.cell_bfield
    pb_x = np.asarray(fx.J)[1:] * np.asarray(fb.x.b_contra)[1:, :, :, 0] * dy * dz
    pb_x[-1] = 0.0
    pb_y = np.asarray(fy.J)[:, 1:] * np.asarray(fb.y.b_contra)[:, 1:, :, 1] * dxw[:, None, None] * dz
    pb_z = np.asarray(fz.J)[:, :, 1:] * np.asarray(fb.z.b_contra)[:, :, 1:, 2] * dxw[:, None, None] * dy
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
        pb_x=j(pb_x), pb_y=j(pb_y), pb_z=j(pb_z), b_cell=j(np.moveaxis(np.asarray(cb.b_contra), -1, 0)),
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


def grad_norm(f, grid: NeutralGrid):
    """Physical ``|grad f|`` of a fine-cell field (full metric)."""

    g = grid.g_contra
    gx, gy, gz = _cell_gradients(f, grid)
    sq = (g[0] * gx * gx + g[3] * gy * gy + g[5] * gz * gz
          + 2.0 * (g[1] * gx * gy + g[2] * gx * gz + g[4] * gy * gz))
    return jnp.sqrt(jnp.maximum(sq, 0.0))


def _owner_sum(fine_integrated, grid: NeutralGrid):
    n = fine_integrated.size
    total = jax.ops.segment_sum(fine_integrated.ravel(), grid.owner_flat.ravel(), num_segments=n)
    return total.reshape(grid.shape)


def neutral_diffusion_rate(neutral_density, diffusivity, grid: NeutralGrid, wall_influx=None):
    """Owner ``dN/dt`` of the isotropic diffusion (conservative: ``sum V dN/dt = influx``).

    ``wall_influx``: optional particles per unit time entering each fine wall
    face (shape ``grid.shape[1:]``); zero means a reflecting wall.
    """

    fx, fy, fz = _face_fluxes(_fine(neutral_density, grid), _fine(diffusivity, grid), grid)
    return owner_inflow(fx, fy, fz, grid, wall_influx)


def owner_inflow(fx, fy, fz, grid: NeutralGrid, wall_influx=None):
    """Owner rate ``-div F`` of integrated fine-face fluxes (+axis direction); x wall face is the last one."""

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


def reaction_source(density, Vi, Te, Ti, neutral_density, params: NeutralParameters, owner, neutral_temperature=None,
                    neutral_velocity=None):
    """AMJUEL hydrogen reaction channels as a plasma ``VolumetricSource`` plus the neutral side.

    ``neutral_temperature``, ``neutral_velocity``: evolved ``T_n`` and ``u_n`` fields (fluid model); default the
    fixed ``T_n`` and atoms at rest.
    """

    n = jnp.where(owner, jnp.maximum(density, 0.0), 0.0)
    nn = jnp.where(owner, jnp.maximum(neutral_density, 0.0), 0.0)
    Tn = jnp.full_like(n, params.neutral_temperature) if neutral_temperature is None else neutral_temperature
    out = compute_hydrogen_reaction_sources(
        n, Vi, params.tau * jnp.maximum(Ti, 0.0), jnp.maximum(Te, 0.0), nn,
        jnp.zeros_like(n) if neutral_velocity is None else neutral_velocity, Tn,
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


def collision_frequencies(density, Te, Ti, params: NeutralParameters, neutral_temperature=None):
    """Per-neutral CX and ionisation frequencies (backend time units); ``T_n`` field or the fixed one."""

    scale = params.reference_density_m3 * time_unit_s(params.reference_temperature_ev)
    n = jnp.maximum(density, 0.0)
    ne_m3 = n * params.reference_density_m3
    te_ev = jnp.maximum(Te, 0.0) * params.reference_temperature_ev
    tn = params.neutral_temperature if neutral_temperature is None else neutral_temperature
    tcx = (params.tau * jnp.maximum(Ti, 0.0) + tn) * params.reference_temperature_ev
    nu_iz = n * rate_coefficient("d", "iz", te_ev, ne_m3) * scale
    nu_cx = n * charge_exchange_rate_coefficient(tcx) * scale
    return nu_cx, nu_iz


def inventories(density, Vi, Te, Ti, neutral_density, grid: NeutralGrid, params: NeutralParameters, neutral_temperature=None,
                neutral_velocity=0.0):
    """(nuclei, plasma thermal+kinetic energy, neutral thermal+kinetic energy) on the owner measure."""

    v = grid.volume
    nuclei = jnp.sum(v * (density + neutral_density))
    plasma = jnp.sum(v * (1.5 * density * Te + 1.5 * params.tau * density * Ti + 0.5 * density * Vi**2))
    tn = params.neutral_temperature if neutral_temperature is None else neutral_temperature
    neutral = jnp.sum(v * (1.5 * tn + 0.5 * neutral_velocity**2) * neutral_density)
    return nuclei, plasma, neutral


# ------------------------------------------------------------------ recycling
RECYCLING_LEDGER_COLUMNS = (
    "time",
    "wall_ion_loss",  # ion wall particle loss of the IMEX step from the wall-face states (step_particle_face)
    "recycled_particles",  # R * wall_ion_loss, deposited as neutrals in the wall cells
    "recycled_energy",  # recycled_particles * E_rec (energy the wall returns with the atoms)
    "recycled_neutral_thermal",  # recycled_particles * 3/2 T_n (what the fixed-T_n inventory gains)
    "recycled_energy_reservoir",  # recycled_energy - recycled_neutral_thermal
    "nuclei",  # sum V (n + N) after the split
    "nuclei_change",  # over the whole step (IMEX + split)
    "nuclei_change_expected",  # explicit density sum (step_particle_explicit) - (1 - R) wall_ion_loss
)


@dataclass(frozen=True)
class RecyclingParameters:
    """Wall recycling: ``R`` of the applied ion wall flux returns as atoms.

    ``energy`` is ``"franck-condon"`` (each atom carries
    ``franck_condon_energy_ev``, 3 eV) or ``"wall"`` (thermal accommodation,
    ``3/2 wall_temperature_ev``).
    """

    coefficient: float = 0.99
    energy: str = "franck-condon"
    franck_condon_energy_ev: float = 3.0
    wall_temperature_ev: float = 0.025

    def energy_per_atom(self, reference_temperature_ev: float) -> float:
        """Recycled energy per atom in backend units (``Te0``)."""

        if self.energy == "franck-condon":
            ev = self.franck_condon_energy_ev
        elif self.energy == "wall":
            ev = 1.5 * self.wall_temperature_ev
        else:
            raise ValueError(f"recycling energy must be 'franck-condon' or 'wall', got {self.energy!r}")
        return float(ev / reference_temperature_ev)


def recycled_neutral_source(wall_loss_rate, coefficient, owner):
    """Neutral source ``R * (applied ion wall loss rate)`` in each wall owner cell.

    ``wall_loss_rate`` is ``-implicit.n`` of the IMEX step: under the sheath
    the implicit density row is the area-weighted sum of the wall-leg ion
    fluxes of that owner, so the deposit is the flux the implicit stages
    applied.  ``LocalFciDrbEBRhs.wall_step_check`` compares that flux with the
    one computed from the wall-face states.
    """

    return jnp.where(owner, coefficient * wall_loss_rate, 0.0)


def knudsen_map(density, Te, Ti, neutral_density, grid: NeutralGrid, params: NeutralParameters, neutral_temperature=None):
    """Neutral Knudsen numbers on the fine cells.

    ``Kn_cell = mfp / h`` (``h`` the smallest physical cell size) and
    ``Kn_grad = mfp / L`` with ``L = min(L_N, L_n)``, ``L_X = X / |grad X|``
    (physical gradient with the full metric).  The diffusive closure is
    valid where ``Kn_grad << 1``; ``Kn_cell > 1`` means a neutral crosses
    more than a cell between collisions.  Returns ``(mfp, Kn_cell, Kn_grad)``.
    """

    nu_cx, nu_iz = collision_frequencies(density, Te, Ti, params, neutral_temperature)
    vth = jnp.sqrt(params.neutral_temperature if neutral_temperature is None else neutral_temperature)
    mfp = vth / jnp.maximum(nu_cx + nu_iz, 1.0e-300)

    def grad_length(field):
        f = _fine(field, grid)
        return jnp.abs(f) / jnp.maximum(grad_norm(f, grid), 1.0e-300)

    length = jnp.minimum(grad_length(neutral_density), grad_length(density))
    mfp_f = _fine(mfp, grid)
    return mfp_f, mfp_f / grid.cell_length, mfp_f / jnp.maximum(length, 1.0e-300)
