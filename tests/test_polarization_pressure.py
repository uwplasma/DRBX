"""Hot-ion Boussinesq polarization: omega = L_perp(phi + tau p_i), p_i = n Ti.

The default ``polarization_variable="phi_plus_tau_pi"`` replaces the legacy
``"phi_plus_tau_ti"`` form only in the polarization relation.  These tests pin
(i) the legacy selector, (ii) the new selector against a directly built
``A(n Ti)``, and (iii) the derived ``p_i`` physical-face payload.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_boundaries import (  # noqa: E402
    BC_DIRICHLET,
    BC_NEUMANN,
    LocalBoundaryFaceBC3D,
)
from drbx.native.fci_drb_EB_rhs import FciDrbEBRhsParameters  # noqa: E402
from drbx.native.fci_halo import build_local_boundary_face_trace_from_halo  # noqa: E402
from drbx.native.fci_operators import LocalPerpLaplacianInverseSolver  # noqa: E402
from drbx.native.fci_sharding import assemble_local_fci_geometry  # noqa: E402
from fci_drb_eb_test_helpers import (  # noqa: E402
    _build_rhs,
    _context_and_sharded_inputs,
)
from jax.experimental.shard_map import shard_map  # noqa: E402
from test_mms_shifted_torus_EB_sharded import _radial_dirichlet_bc  # noqa: E402


def test_polarization_variable_selector_is_validated_and_defaults_to_pi() -> None:
    assert FciDrbEBRhsParameters().polarization_variable == "phi_plus_tau_pi"
    legacy = FciDrbEBRhsParameters(polarization_variable="phi_plus_tau_ti")
    leaves, aux = jax.tree_util.tree_flatten(legacy)
    assert jax.tree_util.tree_unflatten(aux, leaves).polarization_variable == (
        "phi_plus_tau_ti"
    )
    with pytest.raises(ValueError, match="polarization_variable"):
        FciDrbEBRhsParameters(polarization_variable="phi_plus_tau_te")


def _with_variable(model, variable: str):
    return replace(
        model, parameters=replace(model.parameters, polarization_variable=variable)
    )


def _neumann(bc: LocalBoundaryFaceBC3D, value: float) -> LocalBoundaryFaceBC3D:
    return replace(
        bc,
        kind_x=jnp.where(bc.mask_x, BC_NEUMANN, bc.kind_x),
        value_x=jnp.where(bc.mask_x, value, 0.0),
    )


def _maxdiff(a, b, mask=None):
    diff = jnp.abs(jnp.asarray(a) - jnp.asarray(b))
    if mask is not None:
        diff = jnp.where(mask, diff, 0.0)
    return jnp.max(diff)


def test_polarization_pressure_terms_phi_rhs_and_face_data(monkeypatch) -> None:
    context, mesh, local, partition, fields, cell_fields = _context_and_sharded_inputs()
    captured: list = []

    def fake_solve_full_grid(self, rhs_owned, **kwargs):
        del kwargs
        captured.append(rhs_owned)
        return jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)

    monkeypatch.setattr(
        LocalPerpLaplacianInverseSolver, "solve_full_grid", fake_solve_full_grid
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells):
        geometry = assemble_local_fci_geometry(local, cells)
        base = _build_rhs(context, local, geometry)
        # Compare the physical actions only: no algebraic solver regularization.
        base = replace(
            base, gmres_config=replace(base.gmres_config, regularization_epsilon=0.0)
        )
        tau = jnp.asarray(base.parameters.tau, dtype=jnp.float64)
        legacy = _with_variable(base, "phi_plus_tau_ti")
        hot = _with_variable(base, "phi_plus_tau_pi")
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        flat = state.replace(density=jnp.ones_like(density))
        face_bc = base._face_bcs(state)
        solver = base._polarization_solver(
            face_bc.phi,
            config=replace(base.gmres_config, regularization_epsilon=0.0),
        )

        def action(values, bc):
            return base._positive_polarization_action(solver, values, bc)

        a_phi = action(phi, face_bc.phi)
        a_ti = action(Ti, face_bc.Ti)
        # Face data are n=Ti=1 Dirichlet, so p_i's payload is the same Dirichlet 1.
        a_pi = action(density * Ti, face_bc.Ti)
        out = []

        # (i) legacy selector: tau*A(Ti; Ti-face), unchanged terms/residual.
        terms = legacy.polarization_balance_terms(state, phi_owned=phi)
        out += [
            _maxdiff(terms[0], a_phi),
            _maxdiff(terms[1], -tau * a_ti),
            _maxdiff(terms[2], -vorticity),
            _maxdiff(
                legacy.polarization_residual(state, phi_owned=phi),
                terms[0] - terms[1] - terms[2],
            ),
        ]
        # (ii-a) hot-ion with uniform n=1 equals legacy exactly.
        terms_flat = hot.polarization_balance_terms(flat, phi_owned=phi)
        terms_flat_legacy = legacy.polarization_balance_terms(flat, phi_owned=phi)
        out += [_maxdiff(terms_flat, terms_flat_legacy)]
        # (ii-b) hot-ion with nonuniform n: -tau*A(n Ti) built directly, and a
        # genuine difference from the legacy -tau*A(Ti).
        terms_pi = hot.polarization_balance_terms(state, phi_owned=phi)
        out += [
            _maxdiff(terms_pi[0], a_phi),
            _maxdiff(terms_pi[1], -tau * a_pi),
            _maxdiff(terms_pi[2], -vorticity),
            _maxdiff(terms_pi[1], terms[1]),
            _maxdiff(
                hot.polarization_residual(state, phi_owned=phi),
                terms_pi[0] - terms_pi[1] - terms_pi[2],
            ),
        ]
        # phi right-hand side handed to the solve: tau*(-A(q)) - omega.
        captured.clear()
        hot.reconstruct_phi(state)
        legacy.reconstruct_phi(state)
        out += [
            _maxdiff(captured[0], tau * (-a_pi) - vorticity),
            _maxdiff(captured[1], tau * (-a_ti) - vorticity),
            _maxdiff(captured[0], captured[1]),
        ]
        # recovered omega (post-reconstruction image) uses the same q.
        q_hot, bc_hot = hot._polarization_pressure(state, face_bc)
        omega_hot = hot._vorticity_from_polarization(phi, q_hot, face_bc.phi, bc_hot)
        out += [_maxdiff(omega_hot, -(terms_pi[0] - terms_pi[1]))]
        q_leg, bc_leg = legacy._polarization_pressure(state, face_bc)
        out += [
            _maxdiff(q_leg, Ti),
            jnp.asarray(bc_leg is face_bc.Ti, dtype=jnp.float64),
        ]
        return jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in out])

    mapped = shard_map(
        kernel,
        mesh=mesh,
        in_specs=(*((partition,) * 7), partition),
        out_specs=P(),
        check_rep=False,
    )
    out = np.asarray(jax.jit(mapped)(*fields, cell_fields))
    (
        leg_phi, leg_ti, leg_om, leg_res,
        flat_vs_legacy,
        pi_phi, pi_term, pi_om, pi_vs_legacy, pi_res,
        rhs_pi, rhs_ti, rhs_diff,
        omega_hot, q_leg_diff, bc_leg_is,
    ) = out
    for name, value in (
        ("legacy phi", leg_phi), ("legacy tau*A(Ti)", leg_ti),
        ("legacy omega", leg_om), ("legacy residual", leg_res),
        ("uniform n equals legacy", flat_vs_legacy),
        ("pi phi", pi_phi), ("pi -tau*A(n Ti)", pi_term), ("pi omega", pi_om),
        ("pi residual", pi_res), ("phi rhs pi", rhs_pi), ("phi rhs legacy", rhs_ti),
        ("omega_pol", omega_hot), ("legacy q", q_leg_diff),
    ):
        assert value <= 1.0e-12, name
    assert bc_leg_is == 1.0
    assert pi_vs_legacy > 1.0e-8
    assert rhs_diff > 1.0e-8


def test_polarization_pressure_face_data_product_rule() -> None:
    context, mesh, local, partition, fields, cell_fields = _context_and_sharded_inputs()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells):
        geometry = assemble_local_fci_geometry(local, cells)
        domain = local.domain
        model = _with_variable(_build_rhs(context, local, geometry), "phi_plus_tau_pi")
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        base_bc = model._face_bcs(state)
        out = []

        # Dirichlet x Dirichlet -> Dirichlet with value n_b * Ti_b.
        n_bc = _radial_dirichlet_bc(geometry, domain, 2.0, 3.0)
        t_bc = _radial_dirichlet_bc(geometry, domain, 5.0, 7.0)
        _, bc = model._polarization_pressure(
            state, replace(base_bc, density=n_bc, Ti=t_bc)
        )
        mask = n_bc.mask_x
        out += [
            jnp.all(jnp.where(mask, bc.kind_x == BC_DIRICHLET, True)),
            jnp.all(bc.mask_x == mask),
            _maxdiff(bc.value_x[0], jnp.where(mask[0], 10.0, 0.0)),
            _maxdiff(bc.value_x[-1], jnp.where(mask[-1], 21.0, 0.0)),
            jnp.all(jnp.isfinite(bc.value_x)),
        ]

        # Neumann x Neumann -> Neumann with Ti_f g_n + n_f g_Ti (face traces).
        gn_bc = _neumann(n_bc, 0.5)
        gt_bc = _neumann(t_bc, -0.25)
        n_owned = model._owner_field(density)
        t_owned = model._owner_field(Ti)
        n_face = build_local_boundary_face_trace_from_halo(
            model._prepare_scalar_halo(n_owned, gn_bc), geometry, domain, gn_bc
        ).value_x
        t_face = build_local_boundary_face_trace_from_halo(
            model._prepare_scalar_halo(t_owned, gt_bc), geometry, domain, gt_bc
        ).value_x
        pressure, bc = model._polarization_pressure(
            state, replace(base_bc, density=gn_bc, Ti=gt_bc)
        )
        expected = t_face * 0.5 + n_face * (-0.25)
        out += [
            jnp.all(jnp.where(mask, bc.kind_x == BC_NEUMANN, True)),
            _maxdiff(bc.value_x, expected, mask),
            jnp.max(jnp.abs(jnp.where(mask, expected, 0.0))) > 0.0,
            _maxdiff(pressure, density * Ti),
        ]

        # A mixed Dirichlet/Neumann pair has no product closure: loud NaN.
        _, bc = model._polarization_pressure(
            state, replace(base_bc, density=n_bc, Ti=gt_bc)
        )
        out += [jnp.all(jnp.where(mask, jnp.isnan(bc.value_x), True))]
        return jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in out])

    mapped = shard_map(
        kernel,
        mesh=mesh,
        in_specs=(*((partition,) * 7), partition),
        out_specs=P(),
        check_rep=False,
    )
    out = np.asarray(jax.jit(mapped)(*fields, cell_fields))
    assert out[0] == 1.0 and out[1] == 1.0
    assert out[2] <= 1.0e-14 and out[3] <= 1.0e-14 and out[4] == 1.0
    assert out[5] == 1.0 and out[6] <= 1.0e-13 and out[7] == 1.0
    assert out[8] <= 1.0e-14
    assert out[9] == 1.0


def test_every_wall_model_gives_density_and_ti_matching_kinds_and_masks() -> None:
    """Why the product closure needs only Dirichlet/Dirichlet or Neumann/Neumann."""

    import numpy as _np

    from drbx.native.fci_physical_wall import (
        PHYSICAL_WALL_MODEL_NAMES,
        physical_wall_model_from_name,
    )

    context, mesh, local, partition, fields, cell_fields = _context_and_sharded_inputs()
    params = replace(
        context.parameters, parallel_characteristic_wall_law="physical-boundary-state"
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells):
        geometry = assemble_local_fci_geometry(local, cells)
        state = FciDrbEBState(
            jnp.ones_like(density) + 0.1 * density,
            phi, Te, jnp.ones_like(Ti) + 0.1 * Ti, Vi, Ve, vorticity,
        )
        out = []
        for name in PHYSICAL_WALL_MODEL_NAMES:
            model = physical_wall_model_from_name(name)
            if name == "simplified-gbs-mpe":
                halos = {
                    field: jnp.zeros(local.domain.layout.cell_halo_shape)
                    + 1.0
                    for field in ("density", "phi", "Vi", "Te", "Ti")
                }
                bundle = model(
                    state, geometry, local.domain, params, topology_halos=halos
                )
            else:
                bundle = model(state, geometry, local.domain, params)
            for axis in ("x", "y", "z"):
                kn = getattr(bundle.density, f"kind_{axis}")
                kt = getattr(bundle.Ti, f"kind_{axis}")
                mn = getattr(bundle.density, f"mask_{axis}")
                mt = getattr(bundle.Ti, f"mask_{axis}")
                out += [jnp.all(kn == kt), jnp.all(mn == mt)]
                active = mn
                both_dn = (kn == BC_DIRICHLET) | (kn == BC_NEUMANN)
                out += [jnp.all(jnp.where(active, both_dn, True))]
        return jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in out])

    mapped = shard_map(
        kernel,
        mesh=mesh,
        in_specs=(*((partition,) * 7), partition),
        out_specs=P(),
        check_rep=False,
    )
    out = _np.asarray(jax.jit(mapped)(*fields, cell_fields))
    assert out.size == 9 * len(PHYSICAL_WALL_MODEL_NAMES)
    assert _np.all(out == 1.0)
