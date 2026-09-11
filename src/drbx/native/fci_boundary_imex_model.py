"""Model adapter for the coupled material/polarization boundary stage.

This module only supplies the residual and field-scaled diagnostics.  Linear
preconditioning and Newton policy remain in :mod:`fci_boundary_imex`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .fci_boundary_imex import CoupledStageConfig, CoupledStageInfo, solve_damped_newton
from .fci_boundary_imex_split import evaluate_boundary_imex_split
from .fci_drb_EB_rhs import FciDrbEBState
from .fci_gmres import _spmd_sum
from .fci_boundaries import build_local_boundary_face_trace_from_halo
from .fci_physical_wall import _default_sheath_wall_potential
from ..geometry import FCI_DEP_PHYSICAL_BOUNDARY


_FIELDS = ("density", "Te", "Ti", "Vi", "Ve", "vorticity")
_STATE_FIELDS = _FIELDS + ("phi",)


@dataclass
class CoupledBoundaryStageContext:
    model: Any
    base: FciDrbEBState
    dt: Any
    source: FciDrbEBState | None = None
    residual_kernel: Any | None = None

    def __post_init__(self):
        face_bc = self.model._face_bcs(self.base)
        solver = self.model._polarization_solver(face_bc.phi)
        self.active_owned, self.mass_weights = solver._operator_mass_weights()
        self.active_owned = jnp.asarray(self.active_owned, dtype=bool)
        self.mass_weights = jnp.asarray(self.mass_weights, dtype=jnp.float64)
        self.weights_sum = _spmd_sum(jnp.sum(jnp.where(self.active_owned, self.mass_weights, 0.0)), self.model.domain)
        self.field_scales = tuple(
            jnp.maximum(1.0, jnp.sqrt(_spmd_sum(jnp.sum(jnp.where(self.active_owned, self.mass_weights, 0.0) * getattr(self.base, name) ** 2), self.model.domain) / jnp.maximum(self.weights_sum, 1.0e-30)))
            for name in _FIELDS
        )

    @property
    def augmented(self) -> bool:
        return getattr(self.model, "physical_wall_model_name", None) == "simplified-gbs-mpe"

    def pack(self, state: FciDrbEBState, multiplier: Any = 0.0) -> jnp.ndarray:
        values = [jnp.ravel(jnp.asarray(getattr(state, name), dtype=jnp.float64)) for name in _STATE_FIELDS]
        return jnp.concatenate(values + [jnp.asarray([multiplier], dtype=jnp.float64)])

    def unpack(self, vector: jnp.ndarray) -> tuple[FciDrbEBState, jnp.ndarray]:
        vector = jnp.asarray(vector, dtype=jnp.float64)
        size = int(np.prod(self.base.density.shape))
        values = [vector[i * size : (i + 1) * size].reshape(self.base.density.shape) for i in range(7)]
        return self.base.replace(**dict(zip(_STATE_FIELDS, values))), vector[7 * size]

    def _active(self) -> jnp.ndarray:
        return self.active_owned

    def _gauge(self, state: FciDrbEBState) -> jnp.ndarray:
        if not self.augmented:
            return jnp.asarray(0.0, dtype=jnp.float64)
        face_bc = self.model._face_bcs(state)
        weights, affine, target = self.model._simplified_gbs_mpe_phi_gauge_data(state, face_bc)
        weights = weights / jnp.maximum(_spmd_sum(jnp.sum(weights), self.model.domain), 1.0e-30)
        return _spmd_sum(jnp.sum(weights * state.phi), self.model.domain) + affine - target

    def admissibility(self, vector: jnp.ndarray) -> dict[str, Any]:
        """Return finite/positive and wall-closure admissibility diagnostics."""
        try:
            state, _multiplier = self.unpack(vector)
            if not bool(np.asarray(jnp.isfinite(_multiplier))):
                return {"admissible": False, "reason": "nonfinite_multiplier"}
            active = self.active_owned
            for name in _STATE_FIELDS:
                values = jnp.asarray(getattr(state, name))
                if not bool(np.asarray(jnp.all(jnp.isfinite(values)))):
                    return {"admissible": False, "reason": f"nonfinite_{name}"}
                if name in ("density", "Te", "Ti") and not bool(np.asarray(jnp.all(jnp.where(active, values > 0.0, True)))):
                    return {"admissible": False, "reason": f"nonpositive_{name}"}
            face_bc = self.model._face_bcs(state)
            # Check regular physical traces against the retarding sheath
            # admission condition.  Grazing faces are deliberately exempt.
            sheath_model = getattr(self.model, "physical_wall_model_name", None) in ("simple-conducting-sheath", "simplified-gbs-mpe")
            if sheath_model:
                wall = (self.model.conducting_sheath_wall_potential
                        if getattr(self.model, "conducting_sheath_wall_potential", None) is not None
                        else _default_sheath_wall_potential(self.model.parameters, jnp.float64))
                halo_fields = {name: self.model._prepare_scalar_halo(getattr(state, name), getattr(face_bc, name))
                               for name in ("density", "Te", "Ti", "phi")}
                for axis, letter in enumerate("xyz"):
                    bc = face_bc.phi
                    trace = build_local_boundary_face_trace_from_halo(halo_fields["phi"], self.model.geometry, self.model.domain, bc)
                    mask = getattr(bc, f"mask_{letter}")
                    value = getattr(trace, f"value_{letter}")
                    bface = getattr(self.model.geometry.face_bfield, "axes")[axis].B_contra_owned[..., axis]
                    bmag = getattr(self.model.geometry.face_bfield, "axes")[axis].Bmag_owned
                    if not bool(np.asarray(jnp.all(jnp.where(mask, jnp.isfinite(value) & jnp.isfinite(bface) & jnp.isfinite(bmag) & (bmag > 0.0), True)))):
                        return {"admissible": False, "reason": f"invalid_{letter}_wall_geometry"}
                    nongrazing = mask & (jnp.abs(bface) > 1.0e-12)
                    for name in ("density", "Te", "Ti"):
                        therm_trace = build_local_boundary_face_trace_from_halo(
                            halo_fields[name], self.model.geometry, self.model.domain, getattr(face_bc, name)
                        )
                        therm_value = getattr(therm_trace, f"value_{letter}")
                        if not bool(np.asarray(jnp.all(jnp.where(mask, jnp.isfinite(therm_value) & (therm_value > 0.0), True)))):
                            return {"admissible": False, "reason": f"invalid_{name}_{letter}_trace"}
                    if not bool(np.asarray(jnp.all(jnp.where(nongrazing, value - wall >= -64.0 * jnp.finfo(value.dtype).eps * jnp.maximum(1.0, jnp.maximum(jnp.abs(value), jnp.abs(wall))), True)))):
                        return {"admissible": False, "reason": f"negative_sheath_drop_{letter}"}
            if getattr(self.model, "parallel_operator_scheme", None) == "fci":
                state_halo = self.model._prepare_state_halo(state, face_bc)
                owned_slice = self.model.domain.layout.owned_slices_cell
                context = self.model._stencil_builder_context()
                fields = {name: getattr(state_halo, name)[owned_slice]
                          for name in ("density", "Te", "Ti", "phi")}
                stencils = {name: self.model._fci_plasma_side_stencil(
                    fields[name], getattr(face_bc, name), context
                ) for name in fields}
                for direction in ("backward", "forward"):
                    metadata = getattr(self.model.geometry.maps, direction)
                    endpoint_kind = metadata.endpoint_kind
                    endpoint_b = metadata.endpoint_b_contra_x
                    endpoint_bmag = metadata.endpoint_bmag
                    fine_active_value = getattr(self.model.geometry, "active_cell_mask", None)
                    if fine_active_value is None:
                        fine_active_value = self.model.geometry.active_cell_mask_owned
                    fine_active = jnp.asarray(fine_active_value, dtype=bool)
                    endpoint_active = (endpoint_kind == FCI_DEP_PHYSICAL_BOUNDARY) & fine_active
                    nongrazing = endpoint_active & (jnp.abs(endpoint_b) / jnp.maximum(endpoint_bmag, 1.0e-30) > 1.0e-12)
                    endpoint_name = "minus" if direction == "backward" else "plus"
                    endpoint_values = {name: getattr(stencils[name], endpoint_name) for name in fields}
                    finite = jnp.isfinite(endpoint_b) & jnp.isfinite(endpoint_bmag) & (endpoint_bmag > 0.0)
                    for name in ("density", "Te", "Ti", "phi"):
                        finite = finite & jnp.isfinite(endpoint_values[name])
                    positive = (endpoint_values["density"] > 0.0) & (endpoint_values["Te"] > 0.0) & (endpoint_values["Ti"] > 0.0)
                    if not bool(np.asarray(jnp.all(jnp.where(endpoint_active, finite, True)))):
                        return {"admissible": False, "reason": f"invalid_{direction}_wall_endpoint"}
                    if not bool(np.asarray(jnp.all(jnp.where(endpoint_active, positive, True)))):
                        return {"admissible": False, "reason": f"nonpositive_{direction}_wall_endpoint"}
                    if not sheath_model:
                        continue
                    drop = endpoint_values["phi"] - wall
                    slack = 64.0 * jnp.finfo(drop.dtype).eps * jnp.maximum(1.0, jnp.maximum(jnp.abs(endpoint_values["phi"]), jnp.abs(wall)))
                    if not bool(np.asarray(jnp.all(jnp.where(nongrazing, drop >= -slack, True)))):
                        return {"admissible": False, "reason": f"negative_{direction}_sheath_drop"}
            for name in _STATE_FIELDS:
                bc = getattr(face_bc, name)
                for axis in "xyz":
                    mask = getattr(bc, f"mask_{axis}")
                    value = getattr(bc, f"value_{axis}")
                    if not bool(np.asarray(jnp.all(jnp.where(mask, jnp.isfinite(value), True)))):
                        return {"admissible": False, "reason": f"nonfinite_{name}_boundary"}
            return {"admissible": True, "reason": "ok"}
        except (ValueError, FloatingPointError, TypeError) as exc:
            return {"admissible": False, "reason": "invalid_wall_state", "error": repr(exc)}

    def admissible(self, vector: jnp.ndarray) -> bool:
        return bool(self.admissibility(vector)["admissible"])

    def residual_components(self, vector: jnp.ndarray) -> dict[str, jnp.ndarray]:
        state, multiplier = self.unpack(vector)
        split = evaluate_boundary_imex_split(
            self.model,
            state,
            source_owned=self.source,
            polarization_multiplier=multiplier if self.augmented else None,
        )
        active = self._active()
        material = jnp.stack(tuple(
            jnp.where(active, getattr(state, name) - getattr(self.base, name) - self.dt * getattr(split.implicit, name),
                       getattr(state, name) - getattr(self.base, name))
            for name in _FIELDS
        ))
        pol = self.model.polarization_residual(state, phi_owned=state.phi)
        pol = jnp.where(active, pol + (multiplier * active if self.augmented else 0.0), state.phi - self.base.phi)
        gauge = self._gauge(state) if self.augmented else multiplier
        return {"material": material, "polarization": pol, "gauge": gauge}

    def residual(self, vector: jnp.ndarray) -> jnp.ndarray:
        if self.residual_kernel is not None:
            return self.residual_kernel(vector)
        parts = self.residual_components(vector)
        return jnp.concatenate((jnp.ravel(parts["material"]), jnp.ravel(parts["polarization"]), jnp.ravel(parts["gauge"])))

    def _norms_from_parts(self, parts):
        weights = jnp.where(self.active_owned, self.mass_weights, 0.0)
        global_weight = jnp.maximum(self.weights_sum, 1.0e-30)
        material = jnp.sqrt(_spmd_sum(jnp.sum(weights[None, ...] * parts["material"] ** 2, axis=tuple(range(1, parts["material"].ndim))), self.model.domain) / global_weight) / jnp.asarray(self.field_scales)
        pol = jnp.sqrt(_spmd_sum(jnp.sum(weights * parts["polarization"] ** 2), self.model.domain))
        gauge = jnp.abs(parts["gauge"])
        return {"material": material, "polarization": pol, "gauge": gauge}

    def norms(self, vector: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return self._norms_from_parts(self.residual_components(vector))

    def polarization_diagnostics(self, vector: jnp.ndarray, residual=None) -> dict[str, Any]:
        state, multiplier = self.unpack(vector)
        parts = self.residual_components(vector) if residual is None else residual
        pol_l2 = jnp.sqrt(_spmd_sum(jnp.sum(jnp.where(self.active_owned, self.mass_weights, 0.0) * parts["polarization"] ** 2), self.model.domain))
        balance = self.model.polarization_balance_terms(state, phi_owned=state.phi)
        raw_rhs = balance[1] + balance[2]
        raw_l2 = jnp.sqrt(_spmd_sum(jnp.sum(jnp.where(self.active_owned, self.mass_weights, 0.0) * raw_rhs ** 2), self.model.domain))
        rel = pol_l2 / jnp.maximum(raw_l2, 1.0e-30)
        raw_compatibility_defect = _spmd_sum(
            jnp.sum(jnp.where(self.active_owned, self.mass_weights, 0.0)
                    * (multiplier - parts["polarization"])),
            self.model.domain,
        ) if self.augmented else jnp.asarray(0.0, dtype=jnp.float64)
        solver_cfg = getattr(self.model, "gmres_config", None)
        target = max(getattr(solver_cfg, "atol", 1.0e-10), getattr(solver_cfg, "tol", 1.0e-8) * float(np.asarray(raw_l2)))
        accepted_target = max(getattr(solver_cfg, "acceptance_atol", 1.0e-10), getattr(solver_cfg, "acceptance_tol", 1.0e-8) * float(np.asarray(raw_l2)))
        return {"polarization_l2": pol_l2, "raw_rhs_l2": raw_l2, "polarization_relative": rel,
                "raw_compatibility_defect": raw_compatibility_defect,
                "target_met": pol_l2 <= target, "accepted": pol_l2 <= accepted_target,
                "gauge": jnp.abs(parts["gauge"]), "multiplier": multiplier}

    def norm(self, vector: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(jnp.maximum(self.inner_product(vector, vector), 0.0))

    def inner_product(self, left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
        w = jnp.ravel(jnp.where(self.active_owned, self.mass_weights, 0.0))
        material_weights = jnp.concatenate(
            tuple(w / (self.weights_sum * scale * scale) for scale in self.field_scales)
        )
        weights = jnp.concatenate((material_weights, w, jnp.ones((1,))))
        return _spmd_sum(jnp.vdot(left * weights, right), self.model.domain)

    def convergence(self, vector: jnp.ndarray, residual: jnp.ndarray, initial_norm: float) -> bool:
        size = int(np.prod(self.base.density.shape))
        parts = {"material": residual[: 6 * size].reshape((6,) + self.base.density.shape),
                 "polarization": residual[6 * size: 7 * size].reshape(self.base.density.shape),
                 "gauge": residual[7 * size]}
        norms = self._norms_from_parts(parts)
        cfg = getattr(self, "config", CoupledStageConfig())
        solver_cfg = getattr(self.model, "gmres_config", None)
        pol_rtol = getattr(solver_cfg, "acceptance_tol", cfg.material_rtol)
        pol_atol = getattr(solver_cfg, "acceptance_atol", cfg.material_atol)
        initial_parts = {"material": initial_norm[: 6 * size].reshape((6,) + self.base.density.shape),
                         "polarization": initial_norm[6 * size: 7 * size].reshape(self.base.density.shape),
                         "gauge": initial_norm[7 * size]}
        initial_scales = self._norms_from_parts(initial_parts)
        material_ok = jnp.all(norms["material"] <= jnp.maximum(cfg.material_atol, cfg.material_rtol * jnp.maximum(initial_scales["material"], 1.0e-30)))
        pd = self.polarization_diagnostics(vector, residual=parts)
        pol_ok = pd["polarization_l2"] <= max(pol_atol, pol_rtol * float(np.asarray(pd["raw_rhs_l2"])))
        return bool(np.asarray(material_ok)) and bool(np.asarray(pol_ok)) and bool(np.asarray(norms["gauge"] <= cfg.gauge_tol))


def solve_coupled_boundary_stage(model, base, *, solve_dt, source_owned=None, config=None, preconditioner_factory=None, residual_kernel=None, progress=None):
    """Solve one coupled stage, returning ``(state, multiplier, info)``."""
    cfg = CoupledStageConfig() if config is None else config
    # The external kernel is supplied to the Newton engine with dynamic
    # stage arguments; context residual evaluation remains the reference path
    # used by diagnostics and convergence checks.
    context = CoupledBoundaryStageContext(model, base, solve_dt, source_owned)
    context.config = cfg
    initial_multiplier = model.recover_polarization_multiplier(base) if context.augmented else 0.0
    initial = context.pack(base, initial_multiplier)
    def dynamic_kernel(vector, dynamic_base, dynamic_dt, dynamic_source):
        with jax.disable_jit(False):
            return residual_kernel(vector, dynamic_base, dynamic_dt, dynamic_source)
    if preconditioner_factory is None:
        from .fci_drb_EB_rhs import LocalFciDrbEBRhs
        if isinstance(model, LocalFciDrbEBRhs):
            from .fci_boundary_imex_preconditioner import build_coupled_boundary_preconditioner
            preconditioner_factory = lambda vector: build_coupled_boundary_preconditioner(context, vector)
    admission_reason = ["unknown"]
    def admissible_with_reason(vector):
        result = context.admissibility(vector)
        admission_reason[0] = str(result.get("reason", "unknown"))
        return bool(result["admissible"])
    result, info = solve_damped_newton(
        context.residual,
        initial,
        config=cfg,
        norm=context.norm,
        inner_product=context.inner_product,
        admissible=admissible_with_reason,
        preconditioner_factory=preconditioner_factory,
        convergence=context.convergence,
        residual_kernel=dynamic_kernel if residual_kernel is not None else None,
        residual_kernel_args=(base, solve_dt, source_owned), progress=progress,
    )
    if not info.converged:
        if info.reason == "inadmissible_initial_state":
            info = CoupledStageInfo(**{**info.__dict__, "reason":
                                       f"inadmissible_initial_state:{admission_reason[0]}"})
        info = CoupledStageInfo(**{**info.__dict__, "multiplier": float(np.asarray(initial_multiplier))})
        return base, jnp.asarray(initial_multiplier, dtype=jnp.float64), info
    state, multiplier = context.unpack(result)
    info = CoupledStageInfo(**{**info.__dict__, "multiplier": float(np.asarray(multiplier))})
    return state, multiplier, info


def evaluate_coupled_stage_admissibility(model, state: FciDrbEBState) -> dict[str, Any]:
    """Evaluate final-state admissibility using the same adapter contract."""
    context = CoupledBoundaryStageContext(model, state, 0.0)
    return context.admissibility(context.pack(state, 0.0))


__all__ = ["CoupledBoundaryStageContext", "evaluate_coupled_stage_admissibility", "solve_coupled_boundary_stage"]
