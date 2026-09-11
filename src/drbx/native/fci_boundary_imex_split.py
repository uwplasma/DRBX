"""Boundary-aware explicit/implicit RHS partitioning (experimental API)."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import jax.numpy as jnp

from .fci_drb_EB_rhs import (FciDrbEBState, LocalFciDrbEBRhs, RHS_TERM_FIELD_NAMES,
                              RHS_TERM_NAMES, build_local_fci_drb_eb_operator_boundary_bundle)
from .fci_boundaries import build_local_boundary_face_trace_from_halo
from ..geometry import build_local_conservative_stencil_from_field


@dataclass(frozen=True)
class BoundaryImexSplit:
    """The physical, explicit, and complementary implicit RHS fields."""

    physical: FciDrbEBState
    explicit: FciDrbEBState
    implicit: FciDrbEBState


_EXPLICIT_TERM_NAMES = frozenset(
    ("poisson_bracket", "perpendicular_diffusion", "curvature", "source")
)


@dataclass(frozen=True)
class _ReferenceModel(LocalFciDrbEBRhs):
    """Reference operator using plasma-side support for PB and diffusion."""

    reference_face_bc: object = None
    def _poisson_bracket_over_B(self, f_gradient, g_gradient, f_stencil, g_stencil, **kwargs):
        g_halo = kwargs.get("g_field_halo")
        if g_halo is None:
            raise RuntimeError("reference PB requires the captured plasma support halo")
        g_trace = kwargs.get("g_boundary_trace")
        if g_trace is None:
            raise RuntimeError("reference PB requires the g boundary trace mask")
        template = replace(
            self.reference_face_bc.density,
            mask_x=g_trace.mask_x, mask_y=g_trace.mask_y, mask_z=g_trace.mask_z,
        )
        support_bc = self._prepare_poisson_bracket_support_face_bc(template)
        kwargs["g_boundary_trace"] = build_local_boundary_face_trace_from_halo(
            g_halo, self.geometry, self.domain, support_bc
        )
        return super()._poisson_bracket_over_B(
            f_gradient, g_gradient, f_stencil, g_stencil, **kwargs
        )

    def _field_perp_diffusion(self, field_halo, face_bc, coefficient):
        if float(coefficient) == 0.0:
            return super()._field_perp_diffusion(field_halo, face_bc, coefficient)
        homogeneous = replace(
            face_bc,
            value_x=jnp.zeros_like(face_bc.value_x),
            value_y=jnp.zeros_like(face_bc.value_y),
            value_z=jnp.zeros_like(face_bc.value_z),
        )
        owned = self._owner_field(field_halo[self.domain.layout.owned_slices_cell])
        support_halo = self._prepare_scalar_halo(owned, homogeneous)
        return super()._field_perp_diffusion(support_halo, homogeneous, coefficient)

    def _curvature_rhs_contributions(self, **kwargs):
        state_halo = kwargs["state_halo"]
        context = kwargs["context"]
        support_fields = {}
        support_bcs = {}
        for name in ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity"):
            template = getattr(self.reference_face_bc, name)
            support_bcs[name] = self._prepare_poisson_bracket_support_face_bc(template)
            owner = self._owner_field(state_halo.__getattribute__(name)[self.domain.layout.owned_slices_cell])
            support_fields[name] = self._prepare_scalar_halo(owner, support_bcs[name])
        support_state_halo = state_halo.replace(**support_fields)
        support_face = replace(self.reference_face_bc, **support_bcs)
        support_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
            support_state_halo, self.geometry, self.domain, support_face,
            tau=self.parameters.tau,
        )
        kwargs = dict(kwargs)
        kwargs["state_halo"] = support_state_halo
        kwargs["density_conservative_stencil"] = build_local_conservative_stencil_from_field(
            support_fields["density"], self.geometry, context
        )
        kwargs["Te_conservative_stencil"] = build_local_conservative_stencil_from_field(
            support_fields["Te"], self.geometry, context
        )
        kwargs["Ti_conservative_stencil"] = build_local_conservative_stencil_from_field(
            support_fields["Ti"], self.geometry, context
        )
        kwargs["vorticity_conservative_stencil"] = build_local_conservative_stencil_from_field(
            support_fields["vorticity"], self.geometry, context
        )
        kwargs["operator_boundary"] = support_boundary
        return super()._curvature_rhs_contributions(**kwargs)


def _reference_model(model, face_bc):
    values = {field.name: getattr(model, field.name) for field in fields(LocalFciDrbEBRhs)}
    return _ReferenceModel(**values, reference_face_bc=face_bc)


def _zero_state(state: FciDrbEBState) -> FciDrbEBState:
    zero = jnp.zeros_like(state.density)
    return FciDrbEBState(
        density=zero, phi=zero, Te=zero, Ti=zero,
        Vi=zero, Ve=zero, vorticity=zero,
    )


def _add_field(state: FciDrbEBState, name: str, value) -> FciDrbEBState:
    return state.replace(**{name: getattr(state, name) + value})


def evaluate_boundary_imex_split(
    model,
    state: FciDrbEBState,
    *,
    source_owned: FciDrbEBState | None = None,
    polarization_multiplier=None,
) -> BoundaryImexSplit:
    """Evaluate a complete physical RHS and its boundary-safe IMEX split.

    The full physical model is evaluated with short-leg omission disabled.
    The explicit partition is assembled from the returned production term
    fields; every other returned term is placed in the complementary implicit
    field.  Thus ``physical == explicit + implicit`` field-by-field, including
    source terms. ``polarization_multiplier`` is forwarded to the production
    RHS when supplied, preserving the caller's staged polarization coupling.
    """
    physical_model = replace(
        model, parallel_short_leg_treatment="explicit",
        parallel_short_leg_selection="cfl",
    )
    if not isinstance(model, LocalFciDrbEBRhs):
        # Lightweight callers can still exercise the algebraic API.
        reference_model = physical_model
    else:
        reference_face_bc = physical_model._face_bcs(state)
        if physical_model.physical_wall_model_name == "simplified-gbs-mpe":
            reference_face_bc = replace(
                reference_face_bc,
                vorticity=physical_model._derived_vorticity_face_bc_from_polarization(
                    state, state.phi, reference_face_bc,
                    polarization_multiplier=polarization_multiplier,
                ),
            )
        reference_model = _reference_model(physical_model, reference_face_bc)
    evaluate_kwargs = {
        "source_owned": source_owned,
        "phi_owned": state.phi,
        "return_rhs_term_fields": True,
    }
    if polarization_multiplier is not None:
        evaluate_kwargs["polarization_multiplier"] = polarization_multiplier
    result = physical_model.evaluate_stage(state, **evaluate_kwargs)
    physical, term_fields = result[:2]
    if reference_model is physical_model:
        reference_terms = term_fields
    else:
        reference_result = reference_model.evaluate_stage(state, **evaluate_kwargs)
        reference_terms = reference_result[1]
    explicit = _zero_state(state)
    for field_index, field_name in enumerate(RHS_TERM_FIELD_NAMES):
        for term_index, term_name in enumerate(RHS_TERM_NAMES[field_index]):
            if term_name in _EXPLICIT_TERM_NAMES:
                explicit = _add_field(explicit, field_name, reference_terms[field_index, term_index])
    implicit = physical.replace(
        density=physical.density - explicit.density,
        Te=physical.Te - explicit.Te,
        Ti=physical.Ti - explicit.Ti,
        Vi=physical.Vi - explicit.Vi,
        Ve=physical.Ve - explicit.Ve,
        vorticity=physical.vorticity - explicit.vorticity,
        phi=jnp.zeros_like(physical.phi),
    )
    return BoundaryImexSplit(physical=physical, explicit=explicit, implicit=implicit)


__all__ = ["BoundaryImexSplit", "evaluate_boundary_imex_split"]
