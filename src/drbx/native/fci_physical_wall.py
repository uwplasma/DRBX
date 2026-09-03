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

from drbx.geometry import LocalDomain3D, LocalFciGeometry3D

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


def _default_sheath_wall_potential(parameters: Any, dtype: Any) -> jax.Array:
    """Return the equilibrium-compatible fixed wall potential."""

    Te0 = jnp.asarray(_parameter_value(parameters, ("Te0",), 1.0), dtype=dtype)
    Ti0 = jnp.asarray(_parameter_value(parameters, ("Ti0",), 1.0), dtype=dtype)
    tau = jnp.asarray(_parameter_value(parameters, ("tau",), 1.0), dtype=dtype)
    mu = jnp.asarray(
        _parameter_value(parameters, ("mi_over_me", "mu"), 1836.0), dtype=dtype
    )
    _require_positive_thermodynamics(1.0, Te0, Ti0)
    if not isinstance(mu, jax.core.Tracer) and (
        not np.all(np.isfinite(np.asarray(mu))) or not np.all(np.asarray(mu) > 0.0)
    ):
        raise ValueError("conducting sheath requires finite positive mass ratio")
    return -Te0 * jnp.log(
        jnp.sqrt(mu * Te0 / (2.0 * jnp.pi))
        / jnp.sqrt(Te0 + tau * Ti0)
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
    "physical_wall_model_from_name",
]
