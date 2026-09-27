"""JAX consumer for :class:`ComponentSplineBFieldEvaluator` coefficients.

The SciPy evaluator in :mod:`Bfield_evaluator` is intentionally kept as the
producer of the spline coefficients.  This module only packages those
coefficients into an immutable PyTree and evaluates the local tensor-product
interpolant with JAX.  In particular, it does not run a spline filter (which
would both be expensive and risk changing the producer's numerical values).

Only the non-extrapolating path is supported.  Queries are expected to lie in
the source ``R``/``Z`` box; toroidal coordinates are wrapped in the same way
as ``ComponentSplineBFieldEvaluator``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .Bfield_evaluator import ComponentSplineBFieldEvaluator


def _as_x64(value: Any) -> jax.Array:
    """Convert a producer value without changing its requested precision."""

    return jnp.asarray(value, dtype=jnp.float64)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxComponentSplineBFieldEvaluator:
    """Immutable, JAX-compatible view of a component spline evaluator.

    Construct instances with :meth:`from_evaluator`; the constructor is
    public mainly to make PyTree unflattening and serialization straightforward.
    The dynamic PyTree leaves are the three coordinate axes and the three
    prefiltered component coefficient arrays.  Grid metadata is static
    auxiliary data, so a state can be passed to ``jax.jit`` and replicated via
    ``NamedSharding(mesh, PartitionSpec())``.
    """

    R: Any
    phi: Any
    Z: Any
    coefficients: tuple[Any, Any, Any]
    currents: Any
    nfp: int
    period: float
    dR: float
    dphi: float
    dZ: float
    pad: int
    method: str = "cubic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", _as_x64(self.R))
        object.__setattr__(self, "phi", _as_x64(self.phi))
        object.__setattr__(self, "Z", _as_x64(self.Z))
        object.__setattr__(
            self,
            "coefficients",
            tuple(_as_x64(component) for component in self.coefficients),
        )
        object.__setattr__(self, "currents", _as_x64(self.currents).reshape(-1))
        if len(self.coefficients) != 3:
            raise ValueError("coefficients must contain three field components")
        if self.method not in {"linear", "cubic"}:
            raise ValueError("method must be 'linear' or 'cubic'")
        if self.pad != (1 if self.method == "linear" else 4):
            raise ValueError("pad is inconsistent with interpolation method")
        if self.R.ndim != 1 or self.phi.ndim != 1 or self.Z.ndim != 1:
            raise ValueError("R, phi, and Z must be one-dimensional")
        expected = (
            self.phi.size,
            self.Z.size + 2 * self.pad,
            self.R.size + 2 * self.pad,
        )
        for component in self.coefficients:
            if component.shape != expected:
                raise ValueError(
                    "coefficient shape does not match padded source axes: "
                    f"expected {expected}, got {component.shape}"
                )

    def tree_flatten(self):
        children = (self.R, self.phi, self.Z) + tuple(self.coefficients) + (self.currents,)
        aux_data = (
            int(self.nfp),
            float(self.period),
            float(self.dR),
            float(self.dphi),
            float(self.dZ),
            int(self.pad),
            self.method,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        nfp, period, dR, dphi, dZ, pad, method = aux_data
        # Keep reconstruction valid for abstract placeholders used by
        # ``jit.lower``; validation already occurred at producer conversion.
        instance = object.__new__(cls)
        for name, value in zip(("R", "phi", "Z"), children[:3]):
            object.__setattr__(instance, name, value)
        object.__setattr__(instance, "coefficients", tuple(children[3:6]))
        object.__setattr__(instance, "currents", children[6])
        for name, value in zip(
            ("nfp", "period", "dR", "dphi", "dZ", "pad", "method"),
            (nfp, period, dR, dphi, dZ, pad, method),
        ):
            object.__setattr__(instance, name, value)
        return instance

    @classmethod
    def from_evaluator(
        cls, evaluator: ComponentSplineBFieldEvaluator
    ) -> "JaxComponentSplineBFieldEvaluator":
        """Build a JAX view from a non-extrapolating SciPy evaluator.

        The producer's private ``_coefficients`` arrays are copied into JAX
        array leaves without recomputing or numerically modifying them.  The
        explicit type check catches accidental use with another evaluator
        whose coordinate/coefficient conventions are not this module's
        contract.
        """

        if not isinstance(evaluator, ComponentSplineBFieldEvaluator):
            raise TypeError(
                "evaluator must be a ComponentSplineBFieldEvaluator instance"
            )
        if evaluator.extrapolate:
            raise ValueError(
                "JAX evaluator supports only the non-extrapolating producer path"
            )
        coefficients = evaluator._coefficients
        if coefficients is None:  # pragma: no cover - guarded by extrapolate
            raise ValueError("evaluator has no prefiltered coefficient arrays")
        return cls(
            R=evaluator.R,
            phi=evaluator.phi,
            Z=evaluator.Z,
            coefficients=coefficients,
            currents=evaluator.currents,
            nfp=evaluator.nfp,
            period=evaluator.period,
            dR=evaluator._dR,
            dphi=evaluator._dphi,
            dZ=evaluator._dZ,
            pad=evaluator._pad,
            method=evaluator.method,
        )

    # A more discoverable spelling for callers that use the source class name.
    from_component_spline = from_evaluator

    @property
    def extrapolate(self) -> bool:
        """This consumer intentionally implements the non-extrapolating path."""

        return False

    def _interpolate(self, coordinates: jax.Array) -> jax.Array:
        """Interpolate flattened ``(phi, Z, R)`` coordinates pointwise."""

        order = 1 if self.method == "linear" else 3
        if order == 1:
            offsets = jnp.arange(2, dtype=jnp.int32)
        else:
            offsets = jnp.arange(-1, 3, dtype=jnp.int32)

        base = jnp.floor(coordinates).astype(jnp.int32)
        fraction = coordinates - base
        if order == 1:
            weights = jnp.stack((1.0 - fraction, fraction), axis=-1)
        else:
            t = fraction
            weights = jnp.stack(
                (
                    (1.0 - t) ** 3 / 6.0,
                    (3.0 * t**3 - 6.0 * t**2 + 4.0) / 6.0,
                    (-3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0) / 6.0,
                    t**3 / 6.0,
                ),
                axis=-1,
            )

        # SciPy's ``mode="grid-wrap"`` wraps each sampled array axis.  The
        # R/Z coefficient arrays already contain the producer's reflected
        # ghost cells; those cells make all in-domain cubic stencils local.
        phi_indices = jnp.mod(base[:, 0, None] + offsets, self.coefficients[0].shape[0])
        z_indices = jnp.mod(base[:, 1, None] + offsets, self.coefficients[0].shape[1])
        r_indices = jnp.mod(base[:, 2, None] + offsets, self.coefficients[0].shape[2])
        wphi = weights[:, 0, :]
        wz = weights[:, 1, :]
        wr = weights[:, 2, :]

        def one_component(component: jax.Array) -> jax.Array:
            samples = component[
                phi_indices[:, :, None, None],
                z_indices[:, None, :, None],
                r_indices[:, None, None, :],
            ]
            return jnp.einsum("nabc,na,nb,nc->n", samples, wphi, wz, wr)

        return jnp.stack(tuple(one_component(c) for c in self.coefficients), axis=-1)

    def evaluate_cylindrical(self, points_rphiz: Any) -> jax.Array:
        """Evaluate ``(B_R, B_phi, B_Z)`` at ``(..., 3)`` ``(R, phi, Z)`` points."""

        points = _as_x64(points_rphiz)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points_rphiz must have shape (..., 3)")
        leading_shape = points.shape[:-1]
        flat = points.reshape((-1, 3))
        wrapped_phi = self.phi[0] + jnp.mod(flat[:, 1] - self.phi[0], self.period)
        coordinates = jnp.stack(
            (
                (wrapped_phi - self.phi[0]) / self.dphi,
                (flat[:, 2] - self.Z[0]) / self.dZ + self.pad,
                (flat[:, 0] - self.R[0]) / self.dR + self.pad,
            ),
            axis=-1,
        )
        return self._interpolate(coordinates).reshape(leading_shape + (3,))

    def evaluate_cartesian(self, points_xyz: Any) -> jax.Array:
        """Evaluate ``(B_X, B_Y, B_Z)`` at ``(..., 3)`` Cartesian points."""

        points = _as_x64(points_xyz)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points_xyz must have shape (..., 3)")
        leading_shape = points.shape[:-1]
        flat = points.reshape((-1, 3))
        radius = jnp.hypot(flat[:, 0], flat[:, 1])
        phi = jnp.arctan2(flat[:, 1], flat[:, 0])
        cylindrical = jnp.stack((radius, phi, flat[:, 2]), axis=-1)
        field = self.evaluate_cylindrical(cylindrical).reshape((-1, 3))
        cosine = jnp.cos(phi)
        sine = jnp.sin(phi)
        result = jnp.stack(
            (
                field[:, 0] * cosine - field[:, 1] * sine,
                field[:, 0] * sine + field[:, 1] * cosine,
                field[:, 2],
            ),
            axis=-1,
        )
        return result.reshape(leading_shape + (3,))

    def __call__(self, points_xyz: Any) -> jax.Array:
        return self.evaluate_cartesian(points_xyz)


def jax_bfield_evaluator_from_component_spline(
    evaluator: ComponentSplineBFieldEvaluator,
) -> JaxComponentSplineBFieldEvaluator:
    """Return a JAX evaluator backed by an existing spline producer."""

    return JaxComponentSplineBFieldEvaluator.from_evaluator(evaluator)


# Short aliases make the producer/consumer boundary convenient without
# introducing imports into ``drbx.geometry.__init__``.
JaxBFieldEvaluator = JaxComponentSplineBFieldEvaluator
component_spline_to_jax = jax_bfield_evaluator_from_component_spline


__all__ = [
    "JaxComponentSplineBFieldEvaluator",
    "JaxBFieldEvaluator",
    "jax_bfield_evaluator_from_component_spline",
    "component_spline_to_jax",
]
