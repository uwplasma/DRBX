"""Stage-local physical wall bundles for the FCI drift-reduced model.

Wall models in this module provide physical traces and operator boundary
conditions.  The parallel material operator is responsible for combining a
model trace with the owner state through its live characteristic numerical
flux.  In particular, these models do not solve residuals in incoming
characteristic amplitudes and do not classify or release characteristic lanes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Mapping, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry import LocalDomain3D, LocalFciGeometry3D, SIDE_PHYSICAL

from .fci_halo import (
    neumann_face_trace_physical_affine,
)
from .fci_boundaries import BC_DIRICHLET, BC_NEUMANN, LocalBoundaryFaceBC3D

if TYPE_CHECKING:
    from .fci_drb_EB_rhs import (
        FciDrbEBRhsParameters,
        FciDrbEBState,
        LocalFciDrbEBPhysicalWallBundle,
    )


PHYSICAL_WALL_MODEL_NAMES = (
    "legacy-velocity-trace",
    "no-flow",
    "simple-conducting-sheath",
    "simplified-gbs-mpe",
)


class LocalFciDrbEBPhysicalWallModel(Protocol):
    """Stage-local physical wall model contract."""

    def __call__(
        self,
        state: FciDrbEBState,
        geometry: LocalFciGeometry3D,
        domain: LocalDomain3D,
        parameters: FciDrbEBRhsParameters,
    ) -> LocalFciDrbEBPhysicalWallBundle: ...


def _context_value(context: Any, name: str, default: Any) -> Any:
    """Read a value from a mapping or a parameter-like object."""

    if context is None:
        return default
    if isinstance(context, Mapping):
        return context.get(name, default)
    return getattr(context, name, default)


def _parameter_value(parameters: Any, names: tuple[str, ...], default: Any) -> Any:
    """Read the first available value from a parameter-like object."""

    for name in names:
        value = _context_value(parameters, name, None)
        if value is not None:
            return value
    return default


def _require_positive_thermodynamics(
    density: Any,
    electron_temperature: Any,
    ion_temperature: Any,
) -> None:
    """Reject invalid eager states before evaluating a sheath exponential.

    Compiled JAX calls cannot raise a Python exception from a traced value.
    They instead receive the same finite-domain check through the returned
    bundle's finite values; eager calls, which are the normal stage-builder
    path and the focused tests, fail immediately with a useful error.
    """

    values = (density, electron_temperature, ion_temperature)
    if any(isinstance(value, jax.core.Tracer) for value in values):
        return
    arrays = [np.asarray(value) for value in values]
    if any(
        not np.all(np.isfinite(value)) or not np.all(value > 0.0)
        for value in arrays
    ):
        raise ValueError(
            "physical wall models require finite positive n, Te, and Ti"
        )


def _require_finite(value: Any, name: str) -> None:
    """Reject nonfinite eager wall outputs without clipping physical values."""

    if isinstance(value, jax.core.Tracer):
        return
    if not np.all(np.isfinite(np.asarray(value))):
        raise ValueError(f"physical wall model produced nonfinite {name}")


def _require_positive_finite(value: Any, name: str) -> None:
    """Reject an invalid positive scalar/array in an eager wall-law call."""

    if isinstance(value, jax.core.Tracer):
        return
    array = np.asarray(value)
    if not np.all(np.isfinite(array)) or not np.all(array > 0.0):
        raise ValueError(f"warm-ion sheath reference requires finite positive {name}")


def _require_nonnegative_finite(value: Any, name: str) -> None:
    """Reject an invalid nonnegative scalar/array in an eager wall-law call."""

    if isinstance(value, jax.core.Tracer):
        return
    array = np.asarray(value)
    if not np.all(np.isfinite(array)) or not np.all(array >= 0.0):
        raise ValueError(
            f"warm-ion sheath reference requires finite nonnegative {name}"
        )


def _base_face_conditions(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
) -> tuple[LocalBoundaryFaceBC3D, LocalBoundaryFaceBC3D]:
    """Return zero-value Dirichlet and zero-normal-gradient face data."""

    empty = LocalBoundaryFaceBC3D.empty(geometry.layout)
    mask_x = (
        empty.mask_x.at[0]
        .set(domain.runtime_has_physical_lower(0))
        .at[-1]
        .set(domain.runtime_has_physical_upper(0))
    )
    mask_y = (
        empty.mask_y.at[:, 0, :]
        .set(domain.runtime_has_physical_lower(1))
        .at[:, -1, :]
        .set(domain.runtime_has_physical_upper(1))
    )
    neumann = replace(
        empty,
        kind_x=empty.kind_x.at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN),
        kind_y=(
            empty.kind_y.at[:, 0, :]
            .set(BC_NEUMANN)
            .at[:, -1, :]
            .set(BC_NEUMANN)
        ),
        mask_x=mask_x,
        mask_y=mask_y,
    )
    dirichlet = replace(
        empty,
        kind_x=empty.kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        kind_y=(
            empty.kind_y.at[:, 0, :]
            .set(BC_DIRICHLET)
            .at[:, -1, :]
            .set(BC_DIRICHLET)
        ),
        mask_x=mask_x,
        mask_y=mask_y,
    )
    return dirichlet, neumann


def _boundary_owner(array: Any, axis: int, side: str) -> jax.Array:
    """Return the plasma-side owner plane for one physical wall side."""

    return jnp.take(array, 0 if side == "lower" else -1, axis=axis)


def _outward_b_normal(
    geometry: LocalFciGeometry3D,
    axis: int,
    side: str,
) -> jax.Array:
    """Return ``B dot n_wall`` on a physical face."""

    face_bfield = geometry.face_bfield.axes[axis]
    face_index = 0 if side == "lower" else -1
    b_normal = jnp.take(
        face_bfield.B_contra_owned[..., axis], face_index, axis=axis
    )
    return -b_normal if side == "lower" else b_normal


def _set_parallel_values(
    base: LocalBoundaryFaceBC3D,
    values: tuple[jax.Array, jax.Array, jax.Array],
) -> LocalBoundaryFaceBC3D:
    """Set x/y/z face values while preserving the base masks and kinds."""

    return replace(
        base,
        value_x=base.value_x.at[0].set(values[0][0]).at[-1].set(values[0][1]),
        value_y=base.value_y.at[:, 0, :].set(values[1][0]).at[:, -1, :].set(values[1][1]),
        value_z=base.value_z.at[:, :, 0].set(values[2][0]).at[:, :, -1].set(values[2][1]),
    )


def _bundle_with_velocity_conditions(
    state: FciDrbEBState,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    ion_velocity: str,
    electron_velocity: str | None = None,
    parameters: FciDrbEBRhsParameters | None = None,
    wall_potential: Any = None,
) -> LocalFciDrbEBPhysicalWallBundle:
    """Build common scalar conditions and stage-local velocity traces."""

    from .fci_drb_EB_rhs import LocalFciDrbEBPhysicalWallBundle

    dirichlet, neumann = _base_face_conditions(geometry, domain)
    if electron_velocity is None:
        electron_velocity = ion_velocity

    ion_values: list[tuple[jax.Array, jax.Array]] = []
    electron_values: list[tuple[jax.Array, jax.Array]] = []
    for axis in range(3):
        axis_ion: list[jax.Array] = []
        axis_electron: list[jax.Array] = []
        for side in ("lower", "upper"):
            vi_owner = _boundary_owner(state.Vi, axis, side)
            ve_owner = _boundary_owner(state.Ve, axis, side)
            if ion_velocity == "neumann":
                vi_face = vi_owner
            elif ion_velocity == "no-flow":
                vi_face = jnp.zeros_like(vi_owner)
            elif ion_velocity == "sheath":
                vi_face = _conducting_sheath_ion_velocity(
                    state,
                    geometry,
                    axis,
                    side,
                    parameters,
                )
            else:
                raise ValueError(f"unknown ion velocity condition {ion_velocity!r}")

            if electron_velocity == "neumann":
                ve_face = ve_owner
            elif electron_velocity == "no-flow":
                ve_face = jnp.zeros_like(ve_owner)
            elif electron_velocity == "sheath":
                ve_face = _conducting_sheath_electron_velocity(
                    state,
                    geometry,
                    axis,
                    side,
                    parameters,
                    wall_potential,
                )
            else:
                raise ValueError(
                    f"unknown electron velocity condition {electron_velocity!r}"
                )
            _require_finite(vi_face, "ion wall velocity")
            _require_finite(ve_face, "electron wall velocity")
            axis_ion.append(vi_face)
            axis_electron.append(ve_face)
        ion_values.append((axis_ion[0], axis_ion[1]))
        electron_values.append((axis_electron[0], axis_electron[1]))

    # A computed sheath target is a prescribed physical face value.  Mark it
    # Dirichlet so downstream trace construction consumes the target rather
    # than replacing it with a Neumann/owner extrapolation.
    velocity_bc = (
        dirichlet if ion_velocity in ("no-flow", "sheath") else neumann
    )
    electron_velocity_bc = (
        dirichlet if electron_velocity in ("no-flow", "sheath") else neumann
    )
    velocity_bc = _set_parallel_values(velocity_bc, tuple(ion_values))
    electron_velocity_bc = _set_parallel_values(
        electron_velocity_bc, tuple(electron_values)
    )

    return LocalFciDrbEBPhysicalWallBundle(
        density=neumann,
        phi=dirichlet,
        Te=neumann,
        Ti=neumann,
        Vi=velocity_bc,
        Ve=electron_velocity_bc,
        vorticity=dirichlet,
    )


def _bundle_with_simplified_gbs_mpe_conditions(
    state: FciDrbEBState,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    parameters: FciDrbEBRhsParameters | None,
    topology_halos: Mapping[str, jax.Array] | None,
    wall_potential: Any = None,
) -> LocalFciDrbEBPhysicalWallBundle:
    """Build topology-halo-driven simplified GBS-MPE wall conditions.

    On active physical faces this enforces the coupled primitive laws
    ``D_n n + sigma*n_face/cB*D_n Vi = 0`` and
    ``D_n phi + sigma*Te_face/cB*D_n Vi = 0``.  Admissibility is checked only
    on active physical faces; inactive topology or masked faces are inert.
    """
    from .fci_drb_EB_rhs import LocalFciDrbEBPhysicalWallBundle

    if parameters is None:
        raise ValueError("simplified GBS wall requires model parameters")
    required = ("density", "phi", "Vi", "Te", "Ti")
    if not isinstance(topology_halos, Mapping):
        raise ValueError("simplified GBS wall requires topology_halos mapping")
    missing = [name for name in required if name not in topology_halos]
    if missing:
        raise ValueError("topology_halos missing fields: " + ", ".join(missing))
    halos = {}
    for name in required:
        value = jnp.asarray(topology_halos[name], dtype=jnp.float64)
        if value.shape != domain.layout.cell_halo_shape:
            raise ValueError(
                f"topology_halos[{name!r}] must have shape {domain.layout.cell_halo_shape}, got {value.shape}"
            )
        halos[name] = value
    dirichlet, neumann = _base_face_conditions(geometry, domain)
    density_values = []
    phi_values = []
    ion_values = []
    electron_values = []
    for axis in range(3):
        di = []
        dp = []
        ii = []
        ee = []
        for side in ("lower", "upper"):
            side_kind = (domain.shard_spec.lower_side_kind(axis) if side == "lower"
                         else domain.shard_spec.upper_side_kind(axis))
            if side_kind != SIDE_PHYSICAL:
                zero = jnp.zeros_like(_boundary_owner(state.density, axis, side))
                di.append(zero)
                dp.append(zero)
                ii.append(zero)
                ee.append(zero)
                continue
            te, _ = neumann_face_trace_physical_affine(halos["Te"], geometry, domain, axis, side)
            ti, _ = neumann_face_trace_physical_affine(halos["Ti"], geometry, domain, axis, side)
            vi_base, vi_response = neumann_face_trace_physical_affine(halos["Vi"], geometry, domain, axis, side)
            n_base, n_response = neumann_face_trace_physical_affine(halos["density"], geometry, domain, axis, side)
            phi_base, phi_response = neumann_face_trace_physical_affine(halos["phi"], geometry, domain, axis, side)
            active = (neumann.mask_x[0 if side == "lower" else -1] if axis == 0 else
                      neumann.mask_y[:, 0 if side == "lower" else -1, :] if axis == 1 else
                      neumann.mask_z[:, :, 0 if side == "lower" else -1])
            raw_values = (te, ti, vi_base, vi_response, n_base, n_response,
                          phi_base, phi_response, _boundary_owner(state.Vi, axis, side),
                          _boundary_owner(state.Ve, axis, side), _outward_b_normal(geometry, axis, side))
            safe_te = jnp.where(active, te, 1.0)
            safe_ti = jnp.where(active, ti, 1.0)
            safe_vi_base = jnp.where(active, vi_base, 0.0)
            safe_vi_response = jnp.where(active, vi_response, 1.0)
            safe_n_base = jnp.where(active, n_base, 1.0)
            safe_n_response = jnp.where(active, n_response, 1.0)
            safe_phi_base = jnp.where(active, phi_base, 0.0)
            safe_phi_response = jnp.where(active, phi_response, 1.0)
            safe_vi_owner = jnp.where(active, raw_values[8], 0.0)
            safe_ve_owner = jnp.where(active, raw_values[9], 0.0)
            safe_bn = jnp.where(active, raw_values[10], 1.0)
            tau = jnp.asarray(_parameter_value(parameters, ("tau",), 1.0), dtype=safe_te.dtype)
            cb = jnp.sqrt(safe_te + tau * safe_ti)
            bn = safe_bn
            sigma = jnp.where(bn >= 1.0e-12, 1.0, jnp.where(bn <= -1.0e-12, -1.0, 0.0))
            grazing = jnp.abs(bn) <= 1.0e-12
            vi_owner = safe_vi_owner
            vi_face = jnp.where(grazing, vi_owner, sigma * jnp.maximum(cb, sigma * vi_owner))
            gvi = (vi_face - safe_vi_base) / safe_vi_response
            k = sigma * gvi / cb
            denominator = 1.0 + safe_n_response * k
            n_face = safe_n_base / denominator
            g_n = -k * n_face
            g_phi = -safe_te * k
            phi_face = safe_phi_base + safe_phi_response * g_phi
            wall = (_default_sheath_wall_potential(parameters, safe_te.dtype)
                    if wall_potential is None else jnp.asarray(wall_potential, dtype=safe_te.dtype))
            mu = jnp.asarray(_parameter_value(parameters, ("mi_over_me", "mu"), 1836.0), dtype=safe_te.dtype)
            ve_owner = safe_ve_owner
            ve_face = jnp.where(grazing, ve_owner, sigma * jnp.sqrt(mu * safe_te / (2.0 * jnp.pi)) * jnp.exp(-(phi_face - wall) / safe_te))
            invalid = ((denominator == 0.0) | ~jnp.isfinite(te) | ~jnp.isfinite(ti)
                       | (te <= 0.0) | (ti <= 0.0) | ~jnp.isfinite(n_face)
                       | ~jnp.isfinite(phi_face) | ~jnp.isfinite(vi_face)
                       | ~jnp.isfinite(ve_face) | ~jnp.isfinite(gvi)
                       | ~jnp.isfinite(vi_response) | ~jnp.isfinite(n_response)
                       | ~jnp.isfinite(phi_response) | (vi_response == 0.0)
                       | (n_response == 0.0) | ~jnp.isfinite(g_n) | ~jnp.isfinite(g_phi)
                       | ~jnp.isfinite(cb) | (cb <= 0.0) | (n_face <= 0.0)) & active
            if not any(isinstance(value, jax.core.Tracer) for value in (n_face, phi_face)) and bool(jnp.any(invalid)):
                raise ValueError("simplified GBS wall produced singular, nonfinite, or nonpositive face state")
            di.append(jnp.where(active, jnp.where(invalid, jnp.nan, g_n), 0.0))
            dp.append(jnp.where(active, jnp.where(invalid, jnp.nan, g_phi), 0.0))
            ii.append(jnp.where(active, jnp.where(invalid, jnp.nan, vi_face), 0.0))
            ee.append(jnp.where(active, jnp.where(invalid, jnp.nan, ve_face), 0.0))
        density_values.append((di[0], di[1]))
        phi_values.append((dp[0], dp[1]))
        ion_values.append((ii[0], ii[1]))
        electron_values.append((ee[0], ee[1]))
    return LocalFciDrbEBPhysicalWallBundle(
        density=_set_parallel_values(neumann, tuple(density_values)),
        phi=_set_parallel_values(neumann, tuple(phi_values)),
        Te=neumann, Ti=neumann,
        Vi=_set_parallel_values(dirichlet, tuple(ion_values)),
        Ve=_set_parallel_values(dirichlet, tuple(electron_values)),
        vorticity=LocalBoundaryFaceBC3D.empty(geometry.layout),
    )


def _conducting_sheath_ion_velocity(
    state: FciDrbEBState,
    geometry: LocalFciGeometry3D,
    axis: int,
    side: str,
    parameters: FciDrbEBRhsParameters | None,
) -> jax.Array:
    """Return weak-sheath target velocity before the material numerical flux."""

    if parameters is None:
        raise ValueError("conducting sheath requires model parameters")
    Te = _boundary_owner(state.Te, axis, side)
    Ti = _boundary_owner(state.Ti, axis, side)
    Vi_owner = _boundary_owner(state.Vi, axis, side)
    _require_positive_thermodynamics(
        _boundary_owner(state.density, axis, side), Te, Ti
    )
    tau = jnp.asarray(_parameter_value(parameters, ("tau",), 1.0), dtype=Te.dtype)
    b_normal = _outward_b_normal(geometry, axis, side)
    tol = jnp.asarray(1.0e-12, dtype=Te.dtype)
    grazing = jnp.abs(b_normal) <= tol
    sigma = jnp.where(b_normal >= tol, 1.0, jnp.where(b_normal <= -tol, -1.0, 0.0))
    c_b = jnp.sqrt(Te + tau * Ti)
    target = sigma * jnp.maximum(c_b, sigma * Vi_owner)
    return jnp.where(grazing, Vi_owner, target)


def equilibrium_warm_ion_floating_sheath_drop(
    electron_temperature: Any,
    ion_temperature: Any,
    *,
    tau: Any = 1.0,
    mi_over_me: Any = 1836.0,
) -> jax.Array:
    """Return the equilibrium Maxwellian floating-sheath potential drop.

    The returned quantity is the plasma-side (sheath-entrance) potential
    minus the material-wall potential, in the simulation's normalized energy
    units.  It is an equilibrium *reference scalar* for selecting the
    additive gauge of an all-Neumann polarization solve; it is not a
    pointwise, dynamically enforced zero-current condition.

    The reference follows from equating the Maxwellian electron loss speed
    with the warm-ion Bohm speed,

    ``sqrt(mu*Te/(2*pi))*exp(-Delta_phi/Te) = sqrt(Te + tau*Ti)``.

    ``electron_temperature`` and ``ion_temperature`` may be broadcastable
    arrays, which allows the same helper to construct a face-shaped target
    when an equilibrium profile rather than scalar reference parameters are
    used.
    """

    Te = jnp.asarray(electron_temperature)
    dtype = Te.dtype
    Ti = jnp.asarray(ion_temperature, dtype=dtype)
    tau_value = jnp.asarray(tau, dtype=dtype)
    mass_ratio = jnp.asarray(mi_over_me, dtype=dtype)
    _require_positive_finite(Te, "electron temperature")
    _require_positive_finite(Ti, "ion temperature")
    # tau=0 is retained as the cold-ion limiting case; negative values are
    # unphysical because they would reduce the squared warm-Bohm speed.
    _require_nonnegative_finite(tau_value, "tau")
    _require_positive_finite(mass_ratio, "mass ratio")
    bohm_argument = Te + tau_value * Ti
    _require_positive_finite(bohm_argument, "warm Bohm temperature")
    electron_half_flux = jnp.sqrt(mass_ratio * Te / (2.0 * jnp.pi))
    c_b = jnp.sqrt(bohm_argument)
    drop = Te * jnp.log(electron_half_flux / c_b)
    _require_finite(drop, "floating-sheath reference drop")
    return drop


def warm_ion_floating_sheath_face_potential_target(
    wall_potential: Any,
    electron_temperature: Any,
    ion_temperature: Any,
    *,
    tau: Any = 1.0,
    mi_over_me: Any = 1836.0,
) -> jax.Array:
    """Return the plasma-side target for a wall-referenced gauge constraint.

    This computes ``phi_face_target = phi_wall + Delta_phi_ref``.  It is one
    scalar constraint after the face weights are applied by the augmented
    polarization solve, rather than a pointwise Dirichlet boundary value.
    The reference drop is calibrated at equilibrium and does not impose local
    floating current during the evolving simulation.
    """

    wall = jnp.asarray(wall_potential)
    _require_finite(wall, "wall potential")
    target = wall + equilibrium_warm_ion_floating_sheath_drop(
        electron_temperature,
        ion_temperature,
        tau=tau,
        mi_over_me=mi_over_me,
    )
    _require_finite(target, "wall-reference face-potential target")
    return target


def _default_sheath_wall_potential(parameters: Any, dtype: Any) -> jax.Array:
    """Return the equilibrium-compatible fixed wall potential."""

    Te0 = jnp.asarray(_parameter_value(parameters, ("Te0",), 1.0), dtype=dtype)
    Ti0 = jnp.asarray(_parameter_value(parameters, ("Ti0",), 1.0), dtype=dtype)
    tau = jnp.asarray(_parameter_value(parameters, ("tau",), 1.0), dtype=dtype)
    mu = jnp.asarray(
        _parameter_value(parameters, ("mi_over_me", "mu"), 1836.0), dtype=dtype
    )
    return -equilibrium_warm_ion_floating_sheath_drop(
        Te0,
        Ti0,
        tau=tau,
        mi_over_me=mu,
    )


def _conducting_sheath_electron_velocity(
    state: FciDrbEBState,
    geometry: LocalFciGeometry3D,
    axis: int,
    side: str,
    parameters: FciDrbEBRhsParameters | None,
    wall_potential: Any,
) -> jax.Array:
    """Return the exponential electron sheath response.

    The exponent is intentionally not clipped: a negative sheath drop is an
    inverse-sheath branch and is rejected through the finite/admissibility
    checks rather than silently turned into an ordinary sheath.
    """

    if parameters is None:
        raise ValueError("conducting sheath requires model parameters")
    Te = _boundary_owner(state.Te, axis, side)
    Ti = _boundary_owner(state.Ti, axis, side)
    density = _boundary_owner(state.density, axis, side)
    phi_s = _boundary_owner(state.phi, axis, side)
    Ve_owner = _boundary_owner(state.Ve, axis, side)
    _require_positive_thermodynamics(density, Te, Ti)
    mu = jnp.asarray(
        _parameter_value(parameters, ("mi_over_me", "mu"), 1836.0), dtype=Te.dtype
    )
    if wall_potential is None:
        wall_potential = _default_sheath_wall_potential(parameters, Te.dtype)
    wall_potential = jnp.asarray(wall_potential, dtype=Te.dtype)
    b_normal = _outward_b_normal(geometry, axis, side)
    tol = jnp.asarray(1.0e-12, dtype=Te.dtype)
    grazing = jnp.abs(b_normal) <= tol
    sigma = jnp.where(b_normal >= tol, 1.0, jnp.where(b_normal <= -tol, -1.0, 0.0))
    eta = (phi_s - wall_potential) / Te
    target = sigma * jnp.sqrt(mu * Te / (2.0 * jnp.pi)) * jnp.exp(-eta)
    value = jnp.where(grazing, Ve_owner, target)
    _require_finite(value, "electron wall velocity")
    return value


@dataclass(frozen=True)
class LegacyParallelVelocityPhysicalWallModel:
    """Compatibility model for the historical primitive velocity selector."""

    parallel_velocity_wall_bc: str = "neumann"

    def __post_init__(self):
        if self.parallel_velocity_wall_bc not in ("dirichlet-zero", "neumann"):
            raise ValueError(
                "parallel_velocity_wall_bc must be 'dirichlet-zero' or 'neumann', "
                f"got {self.parallel_velocity_wall_bc!r}"
            )

    def __call__(self, state, geometry, domain, parameters):
        del parameters
        condition = (
            "no-flow" if self.parallel_velocity_wall_bc == "dirichlet-zero" else "neumann"
        )
        return _bundle_with_velocity_conditions(
            state,
            geometry,
            domain,
            ion_velocity=condition,
            electron_velocity=condition,
        )


@dataclass(frozen=True)
class NoFlowPhysicalWallModel:
    """Reflecting validation wall with zero ion and electron velocity."""

    def __call__(self, state, geometry, domain, parameters):
        del parameters
        return _bundle_with_velocity_conditions(
            state,
            geometry,
            domain,
            ion_velocity="no-flow",
            electron_velocity="no-flow",
        )


@dataclass(frozen=True)
class SimpleConductingSheathPhysicalWallModel:
    """Rung-two conducting sheath entrance bundle.

    Scalars use plasma-side Neumann traces.  The ion target applies weak
    logical Bohm outflow and supersonic pass-through; the electron target is
    the exponential sheath response using the owner (plasma-side) potential.
    A fixed ``conducting_sheath_wall_potential`` can be supplied, otherwise
    the equilibrium-compatible gauge is derived from the normalization.
    """

    conducting_sheath_wall_potential: float | None = None

    def __call__(self, state, geometry, domain, parameters):
        return _bundle_with_velocity_conditions(
            state,
            geometry,
            domain,
            ion_velocity="sheath",
            electron_velocity="sheath",
            parameters=parameters,
            wall_potential=self.conducting_sheath_wall_potential,
        )


@dataclass(frozen=True)
class SimplifiedGbsMpePhysicalWallModel:
    """Rung-three warm-ion MPE wall bundle.

    The ion and electron parallel velocities reuse the weak Bohm and sheath
    targets from the rung-two conducting sheath.  Density and potential are
    supplied as metric-aware Neumann data, while vorticity is left as a
    derived placeholder for the later polarization wiring.
    """

    conducting_sheath_wall_potential: float | None = None
    vorticity_from_polarization: bool = True

    def __call__(self, state, geometry, domain, parameters, *, topology_halos=None):
        return _bundle_with_simplified_gbs_mpe_conditions(
            state,
            geometry,
            domain,
            parameters=parameters,
            topology_halos=topology_halos,
            wall_potential=self.conducting_sheath_wall_potential,
        )


def resolve_fci_material_wall_endpoint_state(
    physical_wall_model: str,
    plasma_state: jax.Array,
    plasma_phi: jax.Array,
    endpoint_b_contra_x: jax.Array,
    endpoint_bmag: jax.Array,
    parameters: FciDrbEBRhsParameters,
    *,
    conducting_sheath_wall_potential: float | None = None,
) -> jax.Array:
    """Evaluate a physical material wall law at an actual FCI endpoint.

    ``plasma_state`` is the interpolated plasma-side primitive trace ordered
    as ``(n, Te, Ti, Vi, Ve)``.  The magnetic field is the continuous field
    retained by the tracer at that same endpoint.  Keeping this order is
    essential for branch-dependent sheath laws: interpolating a precomputed
    ``sign(B.n)`` target can reverse the endpoint-normal particle flux.

    The present computational wall is the upper logical-u surface, so the
    outward-normal orientation is the sign of contravariant ``B^u``.  Storing
    the complete endpoint field leaves the contract ready for a physical-wall
    normal in the magnetic-presheath rung.
    """

    plasma_state = jnp.asarray(plasma_state, dtype=jnp.float64)
    plasma_phi = jnp.asarray(plasma_phi, dtype=jnp.float64)
    endpoint_b_contra_x = jnp.asarray(endpoint_b_contra_x, dtype=jnp.float64)
    endpoint_bmag = jnp.asarray(endpoint_bmag, dtype=jnp.float64)
    if plasma_state.shape[-1:] != (5,):
        raise ValueError(
            "plasma_state must have final dimension 5 ordered as "
            "(density, Te, Ti, Vi, Ve)"
        )
    expected = plasma_state.shape[:-1]
    for name, value in (
        ("plasma_phi", plasma_phi),
        ("endpoint_b_contra_x", endpoint_b_contra_x),
        ("endpoint_bmag", endpoint_bmag),
    ):
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {value.shape}"
            )

    density, Te, Ti, Vi_owner, Ve_owner = (
        plasma_state[..., index] for index in range(5)
    )
    _require_positive_thermodynamics(density, Te, Ti)
    if physical_wall_model == "no-flow":
        return plasma_state.at[..., 3].set(0.0).at[..., 4].set(0.0)
    if physical_wall_model not in (
        "simple-conducting-sheath",
        "simplified-gbs-mpe",
    ):
        raise ValueError(
            "endpoint-native FCI wall evaluation requires 'no-flow', "
            "'simple-conducting-sheath', or 'simplified-gbs-mpe', got "
            f"{physical_wall_model!r}"
        )

    tau = jnp.asarray(_parameter_value(parameters, ("tau",), 1.0), dtype=Te.dtype)
    mu = jnp.asarray(
        _parameter_value(parameters, ("mi_over_me", "mu"), 1836.0),
        dtype=Te.dtype,
    )
    # Scale the zero guard by |B| rather than by a fixed field magnitude.  This
    # is only a robust sign/grazing guard for the present logical-u wall; a
    # physical incidence angle for the magnetic-presheath rung also requires
    # the endpoint metric or a stored physical unit normal.  A wall hit with
    # missing endpoint field data is invalid rather than silently falling back
    # to an interpolated grid-face branch.
    valid_geometry = (
        jnp.isfinite(endpoint_b_contra_x)
        & jnp.isfinite(endpoint_bmag)
        & (endpoint_bmag > 0.0)
    )
    normalized_normal = endpoint_b_contra_x / jnp.maximum(endpoint_bmag, 1.0e-30)
    grazing = jnp.abs(normalized_normal) <= 1.0e-12
    sigma = jnp.where(normalized_normal > 0.0, 1.0, -1.0)

    c_b = jnp.sqrt(Te + tau * Ti)
    ion_target = sigma * jnp.maximum(c_b, sigma * Vi_owner)
    wall_potential = (
        _default_sheath_wall_potential(parameters, Te.dtype)
        if conducting_sheath_wall_potential is None
        else jnp.asarray(conducting_sheath_wall_potential, dtype=Te.dtype)
    )
    eta = (plasma_phi - wall_potential) / Te
    electron_target = (
        sigma
        * jnp.sqrt(mu * Te / (2.0 * jnp.pi))
        * jnp.exp(-eta)
    )
    ion_target = jnp.where(grazing, Vi_owner, ion_target)
    electron_target = jnp.where(grazing, Ve_owner, electron_target)
    result = (
        plasma_state.at[..., 3]
        .set(ion_target)
        .at[..., 4]
        .set(electron_target)
    )
    result = jnp.where(valid_geometry[..., None], result, jnp.nan)
    return result


def physical_wall_model_from_name(
    name: str,
    *,
    legacy_parallel_velocity_wall_bc: str = "neumann",
    conducting_sheath_wall_potential: float | None = None,
) -> LocalFciDrbEBPhysicalWallModel:
    """Construct one of the four-rung stage-local wall models."""

    if name == "legacy-velocity-trace":
        return LegacyParallelVelocityPhysicalWallModel(
            legacy_parallel_velocity_wall_bc
        )
    if name == "no-flow":
        return NoFlowPhysicalWallModel()
    if name == "simple-conducting-sheath":
        return SimpleConductingSheathPhysicalWallModel(
            conducting_sheath_wall_potential=conducting_sheath_wall_potential
        )
    if name == "simplified-gbs-mpe":
        return SimplifiedGbsMpePhysicalWallModel(
            conducting_sheath_wall_potential=conducting_sheath_wall_potential
        )
    raise ValueError(
        "physical_wall_model must be one of "
        f"{PHYSICAL_WALL_MODEL_NAMES}, got {name!r}"
    )


__all__ = [
    "LegacyParallelVelocityPhysicalWallModel",
    "LocalFciDrbEBPhysicalWallModel",
    "NoFlowPhysicalWallModel",
    "PHYSICAL_WALL_MODEL_NAMES",
    "SimpleConductingSheathPhysicalWallModel",
    "SimplifiedGbsMpePhysicalWallModel",
    "equilibrium_warm_ion_floating_sheath_drop",
    "resolve_fci_material_wall_endpoint_state",
    "physical_wall_model_from_name",
    "warm_ion_floating_sheath_face_potential_target",
]
