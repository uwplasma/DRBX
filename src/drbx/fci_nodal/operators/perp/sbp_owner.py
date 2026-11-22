"""Owner-local P plans and the periodic slab halo exchange slab actions on the Q records mesh."""
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..nodal.nodal_layout import NodalLayout
from ..nodal.sbp_laplacian_plan import (
    LaplacianMetric, LaplacianPlan, LaplacianStructure, ReplicatedLaplacianStructure,
    ShapedLaplacianStructure, _check_layout, core_frame_A, face_coefficients,
    fourier_half_interp, radial_block, radial_face_interp, staggered_pair,
)
from ..nodal.sbp_laplacian_assembly import iter_inplane_blocks
from ..nodal.nodal_plan import NodalMetric, NodalPlan, build_nodal_plan
from .plane_preconditioner import (
    CoreSchurPreconditioner, apply_core_schur_preconditioner, build_core_schur_preconditioner,
)
from .sbp_bracket import sbp_bracket_ext, velocity_flux_ext
from .sbp_curvature import curvature_derivative_ext
from .sbp_laplacian_apply import (
    F_HALO, PLAN_HALO, PLANE_FIELDS, laplacian_action_ext, laplacian_form_ext, window_laplacian_plan,
)
from .sbp_laplacian_solve import _pcg
from .owner_plane_layout import exchange_plane_halo


AXIS = "records"
PHI_HALO = 2
NODAL_HALO = 3
NODAL_PLANE_FIELDS = ("jac", "h", "B", "K", "Hp", "core_Ginv")


@dataclass(frozen=True)
class SlabLayout:
    n_eta: int
    parts: int

    def __post_init__(self):
        if not isinstance(self.n_eta, int) or not isinstance(self.parts, int) or self.parts < 1:
            raise ValueError("The plane and owner counts must be integers with at least one owner")
        if min(self.counts) < F_HALO:
            raise ValueError("Owner SBP requires at least four planes per owner")

    @property
    def edges(self):
        return tuple((owner * self.n_eta + self.parts - 1) // self.parts for owner in range(self.parts + 1))

    @property
    def counts(self):
        return tuple(b - a for a, b in zip(self.edges[:-1], self.edges[1:]))

    @property
    def capacity(self):
        return max(self.counts)

    def plane_ids(self, owner: int, halo: int = 0):
        if not 0 <= owner < self.parts or not 0 <= halo <= F_HALO:
            raise ValueError("Invalid owner or halo width")
        return np.arange(self.edges[owner] - halo, self.edges[owner + 1] + halo) % self.n_eta


class OwnerSBPSlice(NamedTuple):
    nodal: NodalMetric
    laplacian: LaplacianMetric
    aee_h: np.ndarray
    tau: float
    tau_w: float
    ge_diag: np.ndarray
    rings_per_block: int


class OwnerPlan(NamedTuple):
    nodal: NodalPlan
    laplacian: LaplacianPlan
    factors: CoreSchurPreconditioner


def _laplacian_window(layout, metric, aee_h, tau, tau_w):
    """Use the gathered plan's plane-local construction with its supplied build products."""
    import json

    core, ring = _check_layout(layout)
    E, P, Nc, m, N = layout.n_eta, layout.P, core.n_nodes, ring.m, ring.N
    A = np.array(metric.A_log, dtype=np.float64)
    J = np.abs(np.asarray(metric.J_log, dtype=np.float64))
    aee_h = np.asarray(aee_h, dtype=np.float64)
    if A.shape != (E, P, 3, 3) or J.shape != (E, P) or aee_h.shape != (E, P):
        raise ValueError("Expected canonical owned-plus-halo Laplacian metric shapes")
    if not (np.isfinite(A).all() and np.isfinite(J).all() and (J > 0).all() and np.isfinite(aee_h).all()):
        raise ValueError("Invalid owner Laplacian metric")
    Jb = J.copy()
    A[:, :Nc], Jb[:, :Nc] = core_frame_A(A[:, :Nc], J[:, :Nc], core.u, core.theta)
    rb = radial_block(m, "sbp_cubic_trace")
    sp_ = staggered_pair(m)
    du, deta = layout.du, layout.deta
    Iu = radial_face_interp(m)
    Ith = fourier_half_interp(N, layout.delta)
    auu, att, aee = face_coefficients(A, Iu, Ith, None, Nc, m, N, evaluated={"ee": aee_h})
    axy = A[:, :Nc, :2, :2]
    a_core = np.linalg.eigvalsh(0.5 * (axy + np.swapaxes(axy, -1, -2)))[..., -1]
    kappa = a_core * (core.p / core.R_c) ** 2
    alpha = beta_th = beta_eta = None
    if metric.ginv_u is not None:
        gu = np.asarray(metric.ginv_u, dtype=np.float64)
        if gu.shape != (E, P, 3):
            raise ValueError("Invalid owner ginv_u shape")
        tw = rb.tR
        Aw = np.einsum("r,erjc->ejc", tw, A[:, Nc:, 0, :].reshape(E, m, N, 3))
        gw = np.einsum("r,erjc->ejc", tw, gu[:, Nc:].reshape(E, m, N, 3))
        alpha = Aw[..., 0] / np.sqrt(gw[..., 0])
        beta_th = Aw[..., 1] - Aw[..., 0] * gw[..., 1] / gw[..., 0]
        beta_eta = Aw[..., 2] - Aw[..., 0] * gw[..., 2] / gw[..., 0]
    structure_type = LaplacianStructure if layout.shape is None else ShapedLaplacianStructure
    phase = {} if layout.shape is None else {"delta": float(layout.delta)}
    if layout.shape is not None and layout.shape.version == 2:
        structure_type = ReplicatedLaplacianStructure
        phase["shape_json"] = json.dumps(layout.shape.to_dict(), sort_keys=True)
    structure = structure_type(n=layout.n, n_eta=E, P=P, Nc=Nc, m=m, N=N, i0=ring.i0, deta=float(deta), du=float(du),
                               R_c=float(core.R_c), p=int(core.p), **phase)
    return LaplacianPlan(
        Hp=layout.wxy[None, :] * Jb, wxy=np.asarray(layout.wxy, dtype=np.float64), A=A, auu_f=auu, att_h=att, aee_h=aee,
        Du=rb.D_unit / du, Dp=sp_.Dp / du, wf=sp_.wf * du, Iu=Iu, t_in=rb.tL.copy(), t_out=rb.tR.copy(),
        core_D1=np.asarray(core.D1), core_D2=np.asarray(core.D2), core_IP=np.eye(Nc) - core.Pi,
        Rx=np.asarray(core.Rx), cos_g=np.cos(core.theta_g), sin_g=np.sin(core.theta_g), kappa=kappa,
        tau=np.asarray(tau, dtype=np.float64), tau_w=np.asarray(tau_w, dtype=np.float64),
        wall_alpha=alpha, wall_beta_th=beta_th, wall_beta_eta=beta_eta, structure=structure)


def _pad_axis(a, count, axis=0):
    if a is None or count == 0:
        return a
    widths = [(0, 0)] * a.ndim
    widths[axis] = (0, count)
    return np.pad(np.asarray(a), widths, mode="edge")


def build_owner_plan(layout: NodalLayout, slabs: SlabLayout, owner: int, data: OwnerSBPSlice,
                     *, c_kappa=1.0, factor_dtype="float64") -> OwnerPlan:
    """Construct one plan and its factors using its owner payload."""
    if layout.n_eta != slabs.n_eta:
        raise ValueError("Nodal layout and owner slabs disagree")
    size = len(slabs.plane_ids(owner))
    nodal = build_nodal_plan(replace(layout, n_eta=size), data.nodal)
    lap = _laplacian_window(replace(layout, n_eta=size + 2 * PLAN_HALO), data.laplacian,
                            data.aee_h, data.tau, data.tau_w)
    local = window_laplacian_plan(lap, slice(PLAN_HALO, PLAN_HALO + size))
    st = local.structure
    local = replace(local, structure=replace(st, n_eta=size))
    ge = np.asarray(data.ge_diag)
    if ge.shape != (size, layout.P) or not np.isfinite(ge).all():
        raise ValueError("Expected a finite owned Ge_diag")
    shared = dict(A=local.A, auu=local.auu_f, att=local.att_h, aee=local.aee_h, Ge_diag=ge,
                  c_core=1.0, c_wall=1.0, s_in=1.0, s_wall=1.0)
    blocks = iter_inplane_blocks(local, c_kappa=c_kappa, group_planes=1, shared=shared)
    factors = build_core_schur_preconditioner(
        blocks, n_planes=size, n_core=st.Nc, n_rings=st.m, ring_size=st.N,
        rings_per_block=data.rings_per_block, factor_dtype=factor_dtype,
    )
    padding = slabs.capacity - size
    nodal = replace(nodal, structure=replace(nodal.structure, n_eta=slabs.n_eta),
                    **{name: _pad_axis(getattr(nodal, name), padding) for name in NODAL_PLANE_FIELDS})
    lap = replace(lap, structure=replace(lap.structure, n_eta=slabs.n_eta),
                  **{name: _pad_axis(getattr(lap, name), padding) for name in PLANE_FIELDS})
    factors = CoreSchurPreconditioner(
        _pad_axis(factors.lo, padding, 1), _pad_axis(factors.dinv, padding, 1),
        _pad_axis(factors.z, padding), _pad_axis(factors.sinv, padding),
        replace(factors.meta, E=slabs.capacity),
    )
    return OwnerPlan(nodal, lap, factors)


def _owned_count(slabs):
    return jnp.asarray(slabs.counts)[lax.axis_index(AXIS)]


def _mask_owned(field, slabs):
    if len(set(slabs.counts)) == 1:
        return field
    mask = jnp.arange(slabs.capacity) < _owned_count(slabs)
    return jnp.where(mask.reshape((slabs.capacity,) + (1,) * (field.ndim - 1)), field, 0)


def exchange_owner_halo(field, width: int, slabs: SlabLayout):
    """Periodic periodic slab halo exchange exchange on the records mesh, including unequal Q slabs."""
    if field.shape[0] != slabs.capacity or width not in (PHI_HALO, NODAL_HALO, F_HALO):
        raise ValueError("Expected a padded owner field and the required perpendicular stencil halo width")
    if len(set(slabs.counts)) == 1:
        return exchange_plane_halo(field, width, AXIS, slabs.parts)
    count = _owned_count(slabs)
    tail = lax.dynamic_slice_in_dim(field, count - width, width)
    lower = lax.ppermute(tail, AXIS, [(s, (s + 1) % slabs.parts) for s in range(slabs.parts)])
    upper = lax.ppermute(field[:width], AXIS, [(s, (s - 1) % slabs.parts) for s in range(slabs.parts)])
    return lax.dynamic_update_slice_in_dim(jnp.concatenate((lower, field, upper)), upper, width + count, 0)


def velocity_flux_owner(plan: NodalPlan, phi, rho_star, slabs: SlabLayout):
    return _mask_owned(velocity_flux_ext(plan, exchange_owner_halo(phi, PHI_HALO, slabs), rho_star), slabs)


def sbp_bracket_owner(plan: NodalPlan, phi, state, bcd, rho_star, slabs: SlabLayout, c_kappa=1.0):
    flux = velocity_flux_owner(plan, phi, rho_star, slabs)
    scalar = state.ndim == 2
    channels = state[..., None] if scalar else state
    ext = exchange_owner_halo(jnp.concatenate((channels, flux), axis=-1), NODAL_HALO, slabs)
    g, F = ext[..., :channels.shape[-1]], ext[..., channels.shape[-1]:]
    out = sbp_bracket_ext(plan, F, g[..., 0] if scalar else g, bcd, c_kappa)
    return _mask_owned(out, slabs)


def curvature_derivative_owner(plan: NodalPlan, flux, fields, slabs: SlabLayout):
    scalar = fields.ndim == 2
    channels = fields[..., None] if scalar else fields
    ext = exchange_owner_halo(jnp.concatenate((channels, flux), axis=-1), NODAL_HALO, slabs)
    q, F = ext[..., :channels.shape[-1]], ext[..., channels.shape[-1]:]
    return _mask_owned(curvature_derivative_ext(plan, F, q[..., 0] if scalar else q), slabs)


def laplacian_owner(plan: LaplacianPlan, f, slabs: SlabLayout, bcd=None, kinds="dirichlet", c_kappa=1.0,
                    *, neumann_mode=None, positive=False):
    field = exchange_owner_halo(f, F_HALO, slabs)
    wall = jax.tree.map(lambda a: exchange_owner_halo(a, PLAN_HALO, slabs), bcd)
    kernel = laplacian_form_ext if positive else laplacian_action_ext
    return _mask_owned(kernel(plan, field, wall, kinds, c_kappa, neumann_mode=neumann_mode), slabs)


def precondition_owner(factors: CoreSchurPreconditioner, rhs, slabs: SlabLayout):
    value = apply_core_schur_preconditioner(factors, rhs.reshape(-1))
    return _mask_owned(value.reshape(rhs.shape), slabs)


def _owner_sum(local, slabs):
    if slabs.parts == 1:
        return local
    values = lax.all_gather(local, AXIS)  # One scalar per owner.
    return lax.fori_loop(1, slabs.parts, lambda i, s: s + values[i], values[0])


def _dot(a, b, slabs):
    return _owner_sum(jnp.sum(_mask_owned(a * b, slabs)), slabs)


def _solve(matvec, rhs, precondition, norm, rtol, slabs):
    def solve(mv, b):
        return _pcg(mv, b, precondition, norm, jnp.zeros_like(b), rtol, 1000, zero_rhs_exact=True,
                    dot=lambda a, x: _dot(a, x, slabs))

    value, info = lax.custom_linear_solve(matvec, rhs, solve=solve, transpose_solve=solve, symmetric=True, has_aux=True)
    return value, lax.stop_gradient(info)


def strong_psi_owner(omega, H, datum, sigma, factors: CoreSchurPreconditioner, slabs: SlabLayout,
                     *, rho_star, rtol, positive, project, lift):
    """Strong-wall solve with the caller's existing local K, projector and datum lift."""
    lifted = lift(datum)
    forcing = omega if sigma is None else omega - sigma
    rhs = project(-H * forcing / rho_star**2 - positive(lifted))

    def matvec(x):
        px = project(x)
        return project(positive(px)) + (x - px)

    def prec(r):
        pr = project(r)
        return project(precondition_owner(factors, pr, slabs)) + (r - pr)

    value, info = _solve(matvec, rhs, prec, lambda r: jnp.sqrt(_dot(r, r, slabs)), rtol, slabs)
    return _mask_owned(lifted + project(value), slabs), info


def ampere_owner(lp: LaplacianPlan, factors: CoreSchurPreconditioner, slabs: SlabLayout, n, vi, w,
                 *, beta, mu, rtol, c_kappa=1.0, source=None):
    """Shifted Ampère and current reconstruction, with the Dirichlet Ampère operator."""
    H = lp.Hp[PLAN_HALO:PLAN_HALO + slabs.capacity] * lp.structure.deta
    # Padded entries do not enter the dot products or halo send windows.
    rhs = H * beta * n * (vi - w)
    if source is not None:
        rhs = rhs + H * beta * source
    rhs = _mask_owned(rhs, slabs)

    def positive(a):
        return laplacian_owner(lp, a, slabs, c_kappa=c_kappa, positive=True)

    def matvec(a):
        return positive(a) + _mask_owned(H * beta * mu * n * a, slabs)

    def norm(r):
        return jnp.sqrt(_owner_sum(jnp.sum(_mask_owned(r * r / H, slabs)), slabs))

    A, info = _solve(matvec, rhs, lambda r: precondition_owner(factors, r, slabs), norm, rtol, slabs)
    MA = positive(A)
    j = MA / (H * beta)
    if source is not None:
        j = j - source
    ve = vi - j / n
    true_residual = norm(MA + H * beta * mu * n * A - rhs) / jnp.where(norm(rhs) > 0, norm(rhs), 1.)
    info = dict(info, true_relative_residual=lax.stop_gradient(true_residual))
    return A, _mask_owned(j, slabs), _mask_owned(ve, slabs), info
