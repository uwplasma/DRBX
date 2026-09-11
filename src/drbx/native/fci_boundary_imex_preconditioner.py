"""Frozen material/polarization preconditioner for boundary IMEX stages."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class BoundaryPreconditionerContext:
    model: Any
    base: Any
    dt: Any
    active_owned: jnp.ndarray
    mass_weights: jnp.ndarray
    pack: Callable[..., Any]
    unpack: Callable[[Any], Any]
    wall_data: Any = None
    material_jacobian: jnp.ndarray | None = None
    phi_preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    coarse_correction: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    augmented: bool = False
    gauge_weights: jnp.ndarray | None = None
    polarization_vcycle: Callable[[jnp.ndarray], jnp.ndarray] | None = None


def _field(obj: Any, name: str) -> Any:
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _material_matrix(context: Any, data: Any = None) -> jnp.ndarray:
    matrix = getattr(context, "material_jacobian", None)
    data = getattr(context, "wall_data", None) if data is None else data
    if matrix is None and isinstance(data, dict):
        matrix = data.get("material_jacobian")
        if matrix is None:
            back = data.get("backward_incoming_matrix")
            forward = data.get("forward_incoming_matrix")
            dxm = data.get("backward_dx", data.get("backward_distance"))
            dxp = data.get("forward_dx", data.get("forward_distance"))
            if back is not None and forward is not None and dxm is not None and dxp is not None:
                matrix = -jnp.asarray(back) / jnp.maximum(jnp.asarray(dxm)[..., None, None], 1.0e-30)
                matrix = matrix + jnp.asarray(forward) / jnp.maximum(jnp.asarray(dxp)[..., None, None], 1.0e-30)
    if matrix is None:
        raise ValueError("frozen wall data must provide incoming matrices and distances")
    matrix = jnp.asarray(matrix)
    if matrix.shape[-2:] != (5, 5):
        raise ValueError(f"material Jacobian must end in (5,5), got {matrix.shape}")
    model = getattr(context, "model", None)
    if model is not None and hasattr(model, "_restrict_fine_field"):
        matrix = jnp.stack([jnp.stack([model._restrict_fine_field(matrix[..., i, j]) for j in range(5)], axis=-1) for i in range(5)], axis=-2)
    if model is not None and hasattr(model, "_owner_field"):
        matrix = jnp.where(jnp.asarray(context.active_owned)[..., None, None], matrix, 0.0)
    return matrix


def _capture_wall_data(context: Any, vector: Any) -> dict[str, Any]:
    """Freeze production FCI wall data from the state supplied to the factory."""
    model = context.model
    state, _ = context.unpack(vector)
    model = replace(model, parallel_short_leg_treatment="explicit", parallel_short_leg_selection="cfl")
    face = model._face_bcs(state)
    halo = model._prepare_state_halo(state, face)
    from .fci_drb_EB_rhs import build_local_fci_drb_eb_operator_boundary_bundle
    boundary = build_local_fci_drb_eb_operator_boundary_bundle(
        halo, model.geometry, model.domain, face, tau=model.parameters.tau
    )
    parallel = model._parallel_operator_boundary(state_halo=halo, operator_boundary=boundary)
    data = model._fci_parallel_characteristic_wall_data(
        state_halo=halo,
        face_bc=face,
        parallel_boundary=parallel,
        context=model._stencil_builder_context(),
    )
    result = dict(data.get("wall_data", data))
    stencils = data.get("primitive_stencils")
    if stencils:
        result["backward_dx"] = jnp.abs(stencils[0].dx_min)
        result["forward_dx"] = jnp.abs(stencils[0].dx_plus)
    return result


def _pack(context: Any, state: Any, multiplier: Any) -> Any:
    try:
        return context.pack(state, multiplier)
    except TypeError:
        return context.pack((state, multiplier))


def build_coupled_boundary_preconditioner(context: Any, vector: Any = None) -> Callable[[Any], Any]:
    """Build a reusable frozen block preconditioner.

    The coupled adapter contract is ``unpack(v) -> (state, multiplier)`` and
    ``pack(state, multiplier)``. Five material fields are solved pointwise;
    vorticity and the multiplier retain identity rows.
    """
    if vector is None and getattr(context, "material_jacobian", None) is None:
        raise ValueError("a representative vector is required to freeze production wall data")
    frozen_data = getattr(context, "wall_data", None)
    if getattr(context, "material_jacobian", None) is None and frozen_data is None:
        frozen_data = _capture_wall_data(context, vector)
    matrix = _material_matrix(context, frozen_data)
    dt = jnp.asarray(context.dt, dtype=matrix.dtype)
    active = jnp.asarray(context.active_owned, dtype=bool)
    block = jnp.eye(5, dtype=matrix.dtype) - dt * matrix
    phi_preconditioner = getattr(context, "phi_preconditioner", None)
    coarse = getattr(context, "coarse_correction", None)
    polarization_vcycle = getattr(context, "polarization_vcycle", None)
    if phi_preconditioner is None and getattr(context, "model", None) is not None:
        model = context.model
        state, _ = context.unpack(vector)
        face = model._face_bcs(state)
        solver = model._polarization_solver(face.phi)
        from .fci_operators import _homogeneous_local_face_bc, build_solvax_perp_laplacian_preconditioner
        cfg = replace(solver.config, preconditioner="jacobi", regularization_epsilon=(0.0 if getattr(context, "augmented", False) else solver.config.regularization_epsilon))
        raw = build_solvax_perp_laplacian_preconditioner(model.geometry, model.domain, solver.face_projectors, _homogeneous_local_face_bc(face.phi), cfg, control_volume_geometry=model.control_volume_geometry)
        phi_preconditioner = raw
        if raw is None:
            phi_preconditioner = lambda value: value
        if coarse is None and solver.coarse_data is not None:
            def coarse(value):
                return solver.coarse_data.coarse(value, active, mass, model.domain)
    augmented = bool(getattr(context, "augmented", False))
    mass = jnp.asarray(context.mass_weights)
    gauge_weights = getattr(context, "gauge_weights", None)
    if augmented and gauge_weights is None and getattr(context, "model", None) is not None:
        state0, _ = context.unpack(vector)
        face0 = context.model._face_bcs(state0)
        gauge_weights = context.model._simplified_gbs_mpe_phi_gauge_data(state0, face0)[0]
    gauge_weights = mass if gauge_weights is None else jnp.asarray(gauge_weights)

    def apply(vector):
        state, multiplier = context.unpack(vector)
        names = ("density", "Te", "Ti", "Vi", "Ve")
        rhs = jnp.stack(tuple(jnp.asarray(_field(state, name)) for name in names), axis=-1)
        correction = jnp.linalg.solve(block, rhs[..., None])[..., 0]
        correction = jnp.where(active[..., None], correction, rhs)
        state_out = state
        for index, name in enumerate(names):
            state_out = state_out.replace(**{name: correction[..., index]})
        state_out = state_out.replace(vorticity=_field(state, "vorticity"))
        phi_rhs = jnp.asarray(_field(state, "phi"))
        lambda_out = multiplier
        if augmented:
            denom = jnp.sum(jnp.where(active, mass, 0.0))
            mean = jnp.sum(jnp.where(active, mass * phi_rhs, 0.0)) / jnp.maximum(denom, 1.0e-30)
            lambda_out = mean
            phi_rhs = jnp.where(active, phi_rhs - mean, 0.0)
        phi_out = (polarization_vcycle(phi_rhs) if polarization_vcycle is not None else
                   (phi_preconditioner(phi_rhs) if phi_preconditioner is not None else phi_rhs))
        if coarse is not None and polarization_vcycle is None:
            phi_out = phi_out + coarse(phi_rhs)
        phi_out = jnp.where(active, phi_out, jnp.asarray(_field(state, "phi")))
        state_out = state_out.replace(phi=phi_out)
        if augmented:
            denom = jnp.sum(jnp.where(active, gauge_weights, 0.0))
            residual_mean = jnp.sum(jnp.where(active, gauge_weights * phi_out, 0.0)) / jnp.maximum(denom, 1.0e-30)
            phi_out = phi_out + jnp.where(active, (multiplier - residual_mean), 0.0)
            state_out = state_out.replace(phi=phi_out)
        else:
            lambda_out = multiplier
        return _pack(context, state_out, lambda_out)

    return apply


def build_multiplicative_polarization_vcycle(
    operator: Callable[[jnp.ndarray], jnp.ndarray],
    line_u: Callable[[jnp.ndarray], jnp.ndarray],
    coarse_correction: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    *,
    active_owned: jnp.ndarray | None = None,
    projector: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    applications: int = 1,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build one fixed-cost pre/coarse/post polarization V-cycle.

    The cycle applies exactly one line-U presmooth, one homogeneous operator
    residual, one coarse correction (when supplied), one residual recompute,
    and one line-U postsmooth.  It is an opt-in phi-only operator; callers
    retain responsibility for the augmented multiplier/gauge lane.
    """
    if applications not in (1, 2):
        raise ValueError("applications must be 1 or 2")
    active = None if active_owned is None else jnp.asarray(active_owned, dtype=bool)

    def project(value):
        value = jnp.asarray(value)
        if projector is not None:
            value = projector(value)
        return value if active is None else jnp.where(active, value, 0.0)

    def apply(rhs):
        rhs = project(jnp.asarray(rhs))
        correction = project(line_u(rhs))
        residual = project(rhs - operator(correction))
        if coarse_correction is not None:
            correction = project(correction + coarse_correction(residual))
        residual = project(rhs - operator(correction))
        correction = project(correction + line_u(residual))
        for _ in range(applications - 1):
            residual = project(rhs - operator(correction))
            correction = project(correction + line_u(residual))
            residual = project(rhs - operator(correction))
            if coarse_correction is not None:
                correction = project(correction + coarse_correction(residual))
            residual = project(rhs - operator(correction))
            correction = project(correction + line_u(residual))
        return project(correction)

    return apply


def build_augmented_polarization_inverse(
    phi_inverse: Callable[[jnp.ndarray], jnp.ndarray],
    operator_weights: jnp.ndarray,
    gauge_weights: jnp.ndarray,
    *,
    active_owned: jnp.ndarray | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Adapt a phi-only quotient inverse to packed ``(phi, lambda)`` lanes."""
    opw = jnp.asarray(operator_weights)
    gw = jnp.asarray(gauge_weights)
    active = jnp.ones_like(opw, dtype=bool) if active_owned is None else jnp.asarray(active_owned, bool)
    op_sum = jnp.maximum(jnp.sum(jnp.where(active, opw, 0.0)), 1.0e-30)
    gauge_sum = jnp.maximum(jnp.sum(jnp.where(active, gw, 0.0)), 1.0e-30)
    size = int(opw.size)

    def apply(value):
        value = jnp.asarray(value)
        phi_rhs = value[6 * size:7 * size].reshape(opw.shape)
        lambda_rhs = value[7 * size]
        multiplier = jnp.sum(jnp.where(active, opw * phi_rhs, 0.0)) / op_sum
        quotient_rhs = jnp.where(active, phi_rhs - multiplier, 0.0)
        phi = jnp.asarray(phi_inverse(quotient_rhs))
        phi = jnp.where(active, phi, 0.0)
        gauge_mean = jnp.sum(jnp.where(active, gw * phi, 0.0)) / gauge_sum
        phi = jnp.where(active, phi + (lambda_rhs - gauge_mean), 0.0)
        out = jnp.zeros_like(value)
        out = out.at[6 * size:7 * size].set(phi.ravel())
        out = out.at[7 * size].set(multiplier)
        return out

    return apply


def build_fixed_cost_lower_preconditioner(
    context: Any,
    vector: Any,
    *,
    material_inverse: Callable[[Any], Any],
    polarization_inverse: Callable[[Any], Any],
    complete_jacobian_action: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """Build an opt-in fixed-cost lower block preconditioner.

    The supplied inverses are deliberately single-application operators.  For
    a packed residual ``r``, this applies ``dm=M^-1 r_m``, evaluates the
    authoritative coupled action ``C dm`` from ``complete_jacobian_action``,
    then applies ``da=A^-1(r_a-C dm)``.  No nested Krylov solve or physical
    state update is performed here; callers may use coarse/Jacobi operators as
    the two fixed-cost inverses.
    """
    state, _ = context.unpack(vector)
    size = int(jnp.size(_field(state, "density")))

    def apply(residual):
        residual = jnp.asarray(residual)
        material_rhs = residual.at[6 * size:].set(0.0)
        dm = jnp.asarray(material_inverse(material_rhs))
        dm = dm.at[6 * size:].set(0.0)
        coupled_dm = jnp.asarray(complete_jacobian_action(dm))
        algebraic_rhs = residual.at[:6 * size].set(0.0) - coupled_dm
        algebraic_rhs = algebraic_rhs.at[:6 * size].set(0.0)
        da = jnp.asarray(polarization_inverse(algebraic_rhs))
        da = da.at[:6 * size].set(0.0)
        return dm + da

    return apply


def build_fixed_cost_material_defect_sweeps(
    context: Any,
    vector: Any,
    *,
    material_inverse: Callable[[Any], Any],
    material_action: Callable[[Any], Any],
    sweeps: int = 2,
    damping: float = 0.5,
) -> Callable[[Any], Any]:
    """Build a fixed-cost material correction with FCI-aware residual sweeps.

    ``material_action`` is the authoritative *complete* material Jacobian
    action, including field-line/RLP transport and the wall response.  The
    supplied ``material_inverse`` is used as the local or line smoother. Each
    call performs one initial inverse followed by exactly ``sweeps`` residual
    corrections; there is no convergence test or nested Krylov iteration.

    The packed six-lane material block is the five primitive fields plus the
    vorticity identity lane used by the existing lower-block contract.
    Potential, multiplier, and inactive/RLP-alias entries are projected out.
    A pointwise characteristic inverse can therefore be upgraded to a
    field-line Schwarz or line inverse without changing the outer contract.
    """
    if int(sweeps) < 0:
        raise ValueError("sweeps must be nonnegative")
    if not (0.0 < float(damping) <= 1.0):
        raise ValueError("damping must lie in (0, 1]")
    state, _ = context.unpack(vector)
    size = int(jnp.size(_field(state, "density")))
    active = jnp.asarray(context.active_owned, dtype=bool).ravel()
    material_mask = jnp.tile(active, 6)

    def project(value):
        value = jnp.asarray(value)
        projected = value.at[:6 * size].set(
            jnp.where(material_mask, value[:6 * size], 0.0)
        )
        return projected.at[6 * size:].set(0.0)

    def apply(rhs):
        rhs = project(rhs)
        correction = project(material_inverse(rhs))
        for _ in range(int(sweeps)):
            defect = rhs - project(material_action(correction))
            correction = project(correction + float(damping) * material_inverse(defect))
        return correction

    return apply


# Temporary compatibility name for work scripts written during the initial
# audit.  The implementation is deliberately named ``defect_sweeps`` because
# it is a fixed polynomial smoother, not a true line solve.
build_fixed_cost_material_line_sweeps = build_fixed_cost_material_defect_sweeps


def build_phi_current_schur_action(
    perpendicular_action: Callable[[Any], Any],
    phi_gradient: Callable[[Any], Any],
    current_divergence: Callable[[Any], Any],
    density: Any,
    bmag: Any,
    dt: Any,
    electron_mass_ratio: Any,
    active_owned: Any,
    projector: Callable[[Any], Any] | None = None,
) -> Callable[[Any], Any]:
    """Build the frozen leading ``phi/current`` Schur action.

    With stage residuals written as ``R=U-dt*F``, the electron equation has
    ``R_Ve,phi = -dt*mu*G`` because its electrostatic force is ``+mu*G phi``.
    Since ``j=n(Vi-Ve)``, eliminating that response gives
    ``delta j=-n*dt*mu*G phi``.  The vorticity residual contains
    ``-dt*(B**2/n)*D(j)``, and the polarization row has ``C_pol,omega=+I``.
    Thus ``S=A-C M^{-1} B`` contains the *subtracted* term::

        -dt**2 * mu * (B**2/n) * D(n * G(phi)).

    For the authoritative pair ``G=-D*``, ``D(nG)`` is negative
    semidefinite, so this subtraction adds positive stiffness.

    ``G`` and ``D`` must be the authoritative weighted-adjoint pair used by
    the physical current/potential discretization.  This builder is a frozen,
    fixed-cost action only; it does not solve or alter the physical residual.
    ``projector`` may impose the owner/quotient projection after both pieces.
    """
    n = jnp.asarray(density)
    b = jnp.asarray(bmag)
    dt = jnp.asarray(dt)
    mu = jnp.asarray(electron_mass_ratio)
    active = jnp.asarray(active_owned, dtype=bool)

    def apply(phi):
        phi = jnp.asarray(phi)
        phi = jnp.where(active, phi, 0.0)
        leading = jnp.asarray(perpendicular_action(phi))
        response = n * jnp.asarray(phi_gradient(phi))
        coupling = -(dt * dt * mu * b * b /
                    jnp.maximum(n, jnp.asarray(1.0e-30, dtype=n.dtype))
                    * jnp.asarray(current_divergence(response)))
        result = jnp.where(active, leading + coupling, 0.0)
        if projector is not None:
            result = projector(result)
        return jnp.where(active, result, 0.0)

    return apply


def estimate_hutchinson_owner_diagonal(action: Callable[[Any], Any],
                                       active_owned: Any, *,
                                       probe_count: int = 8, seed: int = 0,
                                       positive_floor: float = 1.0e-12):
    """Estimate an action's owner diagonal with deterministic Rademacher probes.

    The estimator is fixed-cost and deliberately host-deterministic.  The
    returned diagonal is masked to owners and floored in magnitude so it can
    safely define a stationary smoother even when the action is nonsymmetric.
    """
    if int(probe_count) < 1:
        raise ValueError("probe_count must be positive")
    active = np.asarray(active_owned, dtype=bool).ravel()
    rng = np.random.default_rng(int(seed))
    estimate = np.zeros(active.size, dtype=float)
    for _ in range(int(probe_count)):
        probe = rng.choice(np.array([-1.0, 1.0]), size=active.size)
        probe[~active] = 0.0
        response = np.asarray(action(probe.reshape(np.asarray(active_owned).shape)), dtype=float).ravel()
        estimate += probe * response
    estimate /= float(probe_count)
    estimate[~active] = 0.0
    floor = float(positive_floor)
    if floor <= 0.0:
        raise ValueError("positive_floor must be positive")
    safe = np.where(active, np.maximum(np.abs(estimate), floor), 1.0)
    diagnostics = {"probe_count": int(probe_count), "seed": int(seed),
                   "positive_floor": floor,
                   "floored_count": int(np.count_nonzero(active & (np.abs(estimate) < floor))),
                   "active_count": int(np.count_nonzero(active)),
                   "min_abs_raw": float(np.min(np.abs(estimate[active]))) if np.any(active) else 0.0,
                   "max_abs_raw": float(np.max(np.abs(estimate[active]))) if np.any(active) else 0.0}
    return estimate, safe, diagnostics


def build_fixed_cost_multiplicative_schur_smoother(
    diagonal: Any, s0_action: Callable[[Any], Any],
    phi_inverse: Callable[[Any], Any], active_owned: Any, *,
    projector: Callable[[Any], Any] | None = None,
    damping: float = 1.0, positive_floor: float = 1.0e-12,
) -> Callable[[Any], Any]:
    """Build fixed D-S0-P_A-S0-D multiplicative Schur smoothing.

    ``diagonal`` is the Hutchinson-estimated owner diagonal (or a supplied
    diagonal).  Each application performs exactly two S0 actions, one phi
    inverse, and two diagonal corrections; it never checks convergence.
    Gauge treatment remains the responsibility of the augmented wrapper.
    """
    if not 0.0 < float(damping) <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    active = jnp.asarray(active_owned, dtype=bool).ravel()
    diag = jnp.asarray(diagonal).ravel()
    if diag.size != active.size:
        raise ValueError("diagonal and active_owned sizes differ")
    invdiag = jnp.where(active, 1.0 / jnp.maximum(jnp.abs(diag), positive_floor), 0.0)

    def project(value):
        value = jnp.asarray(value).ravel()
        value = jnp.where(active, value, 0.0)
        return value if projector is None else jnp.asarray(projector(value)).ravel()

    def apply(rhs):
        rhs = project(rhs)
        correction = float(damping) * invdiag * rhs
        defect = project(rhs - s0_action(correction))
        correction = correction + float(damping) * jnp.asarray(phi_inverse(defect)).ravel()
        defect = project(rhs - s0_action(correction))
        correction = correction + float(damping) * invdiag * defect
        return project(correction)
    return apply


def build_fixed_cost_primitive_vorticity_material_inverse(
    context: Any, vector: Any, *, primitive_inverse: Callable[[Any], Any],
    vorticity_inverse: Callable[[Any], Any],
    complete_jacobian_action: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """Build one fixed primitive-to-vorticity material solve ``M^{-1}``.

    The result applies the primitive inverse, one complete action, then the
    vorticity inverse.  It is reusable for both triangular and symmetric
    block preconditioners.
    """
    state, _ = context.unpack(vector)
    size = int(jnp.size(_field(state, "density")))
    active = jnp.asarray(context.active_owned, dtype=bool).ravel()
    qmask = jnp.tile(active, 5)

    def project_q(value):
        out = jnp.asarray(value).at[:5 * size].set(
            jnp.where(qmask, jnp.asarray(value)[:5 * size], 0.0))
        return out.at[5 * size:].set(0.0)

    def project_omega(value):
        value = jnp.asarray(value)
        return jnp.zeros_like(value).at[5 * size:6 * size].set(
            jnp.where(active, value[5 * size:6 * size], 0.0))

    def apply(residual):
        residual = jnp.asarray(residual)
        dq = project_q(primitive_inverse(project_q(residual)))
        jdq = jnp.asarray(complete_jacobian_action(dq))
        domega = project_omega(vorticity_inverse(project_omega(residual - jdq)))
        return dq + domega

    return apply


def build_fixed_cost_three_block_lower_preconditioner(
    context: Any,
    vector: Any,
    *,
    primitive_inverse: Callable[[Any], Any],
    vorticity_inverse: Callable[[Any], Any],
    algebraic_inverse: Callable[[Any], Any],
    complete_jacobian_action: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """Build the fixed three-block lower inverse ``q -> omega -> (phi,lambda)``.

    The packed ordering is five primitive lanes ``q=(n,Te,Ti,Vi,Ve)``, the
    vorticity lane, and the augmented algebraic lanes ``(phi,lambda)``.  The
    supplied inverses are single fixed-cost applications.  This routine
    performs exactly two complete Jacobian actions and never iterates.
    """
    state, _ = context.unpack(vector)
    size = int(jnp.size(_field(state, "density")))
    def project_algebraic(value): return jnp.asarray(value).at[:6 * size].set(0.0)
    material_inverse = build_fixed_cost_primitive_vorticity_material_inverse(
        context, vector, primitive_inverse=primitive_inverse,
        vorticity_inverse=vorticity_inverse,
        complete_jacobian_action=complete_jacobian_action)

    def apply(residual):
        residual = jnp.asarray(residual)
        dm = material_inverse(residual)
        jdm = jnp.asarray(complete_jacobian_action(dm))
        ra = project_algebraic(residual - jdm)
        da = project_algebraic(algebraic_inverse(ra))
        return dm + da

    return apply


def build_fixed_cost_symmetric_mam_preconditioner(
    context: Any, vector: Any, *, primitive_inverse: Callable[[Any], Any],
    vorticity_inverse: Callable[[Any], Any],
    algebraic_inverse: Callable[[Any], Any],
    complete_jacobian_action: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """Build fixed symmetric material-algebraic-material block application.

    The material solve is reused before and after the algebraic correction;
    exactly four complete Jacobian actions occur per application.
    """
    state, _ = context.unpack(vector)
    size = int(jnp.size(_field(state, "density")))
    material_inverse = build_fixed_cost_primitive_vorticity_material_inverse(
        context, vector, primitive_inverse=primitive_inverse,
        vorticity_inverse=vorticity_inverse,
        complete_jacobian_action=complete_jacobian_action)
    def project_algebraic(value): return jnp.asarray(value).at[:6 * size].set(0.0)
    def apply(residual):
        residual = jnp.asarray(residual)
        dm = material_inverse(residual)
        jdm = jnp.asarray(complete_jacobian_action(dm))
        da = project_algebraic(algebraic_inverse(project_algebraic(residual - jdm)))
        y = dm + da
        jy = jnp.asarray(complete_jacobian_action(y))
        dm_post = material_inverse(residual - jy)
        return y + dm_post
    return apply


__all__ = ["BoundaryPreconditionerContext", "build_coupled_boundary_preconditioner",
           "build_fixed_cost_lower_preconditioner", "build_multiplicative_polarization_vcycle",
           "build_fixed_cost_material_defect_sweeps", "build_fixed_cost_material_line_sweeps",
           "build_fixed_cost_three_block_lower_preconditioner",
           "build_fixed_cost_primitive_vorticity_material_inverse",
           "build_fixed_cost_symmetric_mam_preconditioner",
           "estimate_hutchinson_owner_diagonal",
           "build_fixed_cost_multiplicative_schur_smoother",
           "build_phi_current_schur_action",
           "build_augmented_polarization_inverse"]
