"""Narrow conservation, antisymmetry, and eta-sharding bracket contracts."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from axis_regular_operator_support import polar_fixture  # noqa: E402
from drbx.geometry import (  # noqa: E402
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
)
from drbx.geometry.fci_control_volumes import (  # noqa: E402
    build_polar_angular_agglomeration_geometry,
)
from drbx.native.fci_angular_agglomeration import (  # noqa: E402
    lower_polar_angular_agglomeration_geometry,
)
from drbx.native.fci_boundaries import (  # noqa: E402
    BC_NEUMANN,
    LocalBoundaryFaceBC3D,
)
from drbx.native.fci_halo import LocalHaloClosure3D  # noqa: E402
from drbx.native.fci_model import inject_owned_field_to_halo  # noqa: E402
from drbx.native.fci_operators import (  # noqa: E402
    aggregate_local_control_volume_average,
    local_control_volume_projected_fine_cell_volume,
    local_poisson_bracket_compatible_flux_op,
    reconstruct_local_control_volume_poisson_field,
)
from poisson_compatible_flux_diagnostics import _identity_geometry  # noqa: E402
from test_fci_operators_domain_decomp import (  # noqa: E402
    RHO_MIN,
    _build_domain,
    _build_ghost_filler,
    _build_local_geometry,
    _build_radial_neumann_bc,
    _prepare_scalar_field_halo,
    make_mesh_for_shard_counts,
    put_scalar_field_on_mesh,
)


def _closed_cartesian_fields(geometry):
    x = geometry.grid.x.centers_halo[:, None, None]
    y = geometry.grid.y.centers_halo[None, :, None]
    z = geometry.grid.z.centers_halo[None, None, :]
    x, y, z = jnp.broadcast_arrays(x, y, z)
    s = (x - RHO_MIN) / (1.0 - RHO_MIN)
    f = (
        jnp.sin(jnp.pi * s) * (jnp.cos(y) + 0.17 * jnp.sin(z))
        + 0.14 * jnp.sin(2.0 * jnp.pi * s) * jnp.sin(2.0 * y)
    )
    g = 1.0 + jnp.sin(2.0 * jnp.pi * s) * (
        0.23 * jnp.sin(y) + 0.11 * jnp.cos(z)
    ) + 0.12 * jnp.sin(3.0 * jnp.pi * s) * jnp.cos(2.0 * y)
    h = 0.4 + jnp.sin(3.0 * jnp.pi * s) * (
        0.31 * jnp.cos(2.0 * y) + 0.09 * jnp.sin(z)
    ) + 0.08 * jnp.sin(2.0 * jnp.pi * s) * jnp.sin(y)
    return f, g, h


def test_centered_closed_identity_metric_has_only_the_narrow_integral_contract():
    """Closed constant-tensor bracket is integral-free, but not Arakawa-cyclic."""

    shape = (12, 24, 8)
    geometry = _identity_geometry(shape)
    domain = _build_domain(shape, 1)
    context = StencilBuilderContext(layout=domain.layout, domain=domain)
    f, g, h = _closed_cartesian_fields(geometry)
    stencils = tuple(
        build_local_conservative_stencil_from_field(q, geometry, context)
        for q in (f, g, h, jnp.ones_like(f))
    )

    def bracket(i, j):
        return local_poisson_bracket_compatible_flux_op(
            stencils[i], stencils[j], geometry, domain=domain,
            characteristic_scheme="centered",
        )

    bfg, bgf = bracket(0, 1), bracket(1, 0)
    bgh, bhf, bfh = bracket(1, 2), bracket(2, 0), bracket(0, 2)
    np.testing.assert_allclose(bfg + bgf, 0.0, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(bracket(0, 3), 0.0, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(bracket(3, 0), 0.0, rtol=0.0, atol=2.0e-13)

    owned = geometry.layout.owned_slices_cell
    fo, go, ho = (np.asarray(q[owned]) for q in (f, g, h))
    weights = np.asarray(
        geometry.cell_metric.J_owned
        * geometry.spacing.dx_owned
        * geometry.spacing.dy_owned
        * geometry.spacing.dz_owned
    )
    integral = lambda q: float(np.sum(weights * np.asarray(q)))
    assert abs(integral(bfg)) < 3.0e-13

    # This scheme promises pointwise argument antisymmetry, not the full
    # three-slot cyclic identity of an Arakawa Jacobian.  Keep a generic
    # counterexample so future documentation cannot silently overclaim it.
    cyclic = np.asarray(
        (integral(ho * bfg), integral(fo * bgh), integral(go * bhf))
    )
    weak_skew_defect = integral(ho * bfg + go * bfh)
    assert float(np.ptp(cyclic)) > 1.0e-2
    assert abs(weak_skew_defect) > 1.0e-2


def _rlp_case():
    shape = (8, 16, 4)
    geometry, domain, context, coordinates, exchange, scalar, *_ = polar_fixture(
        shape=shape, halo_width=2
    )
    u_faces = np.linspace(0.0, 1.0, shape[0] + 1)
    theta_faces = np.linspace(0.0, 2.0 * np.pi, shape[1] + 1)
    eta_faces = np.linspace(0.0, 2.0 * np.pi, shape[2] + 1)

    def varying_jacobian(points):
        points = np.asarray(points)
        r, theta, eta = points[..., 0], points[..., 1], points[..., 2]
        return np.maximum(
            r * (1.0 + 0.19 * r * np.cos(theta)) * (1.0 + 0.07 * np.cos(eta)),
            1.0e-14,
        )

    host = build_polar_angular_agglomeration_geometry(
        u_faces, theta_faces, eta_faces, varying_jacobian,
        quadrature_order=3, angular_group_size=(16, 4, 2, 1, 1, 1, 1, 1),
    )
    cv = lower_polar_angular_agglomeration_geometry(host, geometry)
    cells = cv.cells
    r, theta, eta = coordinates
    owned = geometry.layout.owned_slices_cell
    raw_f = (r * r * jnp.cos(2.0 * theta) + 0.03 * jnp.sin(eta))[owned]
    raw_g = (1.2 + r * r * jnp.sin(2.0 * theta) + 0.04 * jnp.cos(eta))[owned]
    owner_f = aggregate_local_control_volume_average(raw_f, cells, domain)
    owner_g = aggregate_local_control_volume_average(raw_g, cells, domain)
    fine_f = reconstruct_local_control_volume_poisson_field(owner_f, cv, domain)
    fine_g = reconstruct_local_control_volume_poisson_field(owner_g, cv, domain)
    fine_one = reconstruct_local_control_volume_poisson_field(
        jnp.where(cells.is_active_owner, 1.0, 0.0), cv, domain
    )

    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    face_bc = replace(
        face_bc,
        kind_x=face_bc.kind_x.at[-1].set(BC_NEUMANN),
        mask_x=face_bc.mask_x.at[-1].set(True),
    )
    closure = LocalHaloClosure3D(
        physical_ghost_filler=_build_ghost_filler(2),
        halo_exchange=exchange,
        topology_filler=scalar,
    )

    def stencil(fine):
        halo = closure(
            inject_owned_field_to_halo(fine, domain.layout), domain, face_bc
        )
        # Face gradients differentiate tangentially on coordinate faces and
        # therefore need polar-periodic corner closure, not just face halos.
        halo = scalar(halo, domain)
        return build_local_conservative_stencil_from_field(halo, geometry, context)

    fs, gs, ones = stencil(fine_f), stencil(fine_g), stencil(fine_one)
    raw_cell_volume = local_control_volume_projected_fine_cell_volume(geometry, cv)

    def raw_bracket(left, right, cell_volume):
        return local_poisson_bracket_compatible_flux_op(
            left, right, geometry, domain=domain,
            axis_regular_axes=(True, False, False),
            characteristic_scheme="centered", cell_volume=cell_volume,
        )

    return geometry, domain, cv, fs, gs, ones, raw_cell_volume, raw_bracket


def test_rlp_raw_volume_normalization_pairs_with_owner_volume_restriction():
    geometry, domain, cv, fs, gs, ones, raw_volume, raw_bracket = _rlp_case()
    cells = cv.cells
    active = np.asarray(cells.is_active_owner, dtype=bool)
    owner_weights = np.asarray(cells.aggregate_volume)

    raw = raw_bracket(fs, gs, raw_volume)
    reverse = raw_bracket(gs, fs, raw_volume)
    owner = np.asarray(aggregate_local_control_volume_average(raw, cells, domain))
    owner_reverse = np.asarray(
        aggregate_local_control_volume_average(reverse, cells, domain)
    )
    np.testing.assert_allclose(owner[active] + owner_reverse[active], 0.0, atol=3e-12, rtol=0.0)

    constant_right = np.asarray(
        aggregate_local_control_volume_average(
            raw_bracket(fs, ones, raw_volume), cells, domain
        )
    )
    constant_left = np.asarray(
        aggregate_local_control_volume_average(
            raw_bracket(ones, fs, raw_volume), cells, domain
        )
    )
    np.testing.assert_allclose(constant_right[active], 0.0, atol=3e-12, rtol=0.0)
    np.testing.assert_allclose(constant_left[active], 0.0, atol=3e-12, rtol=0.0)

    midpoint = raw_bracket(fs, gs, None)
    logical_volume = np.asarray(
        geometry.spacing.dx_owned
        * geometry.spacing.dy_owned
        * geometry.spacing.dz_owned
    )
    incidence_integral = float(np.sum(
        logical_volume * np.asarray(geometry.cell_metric.J_owned) * np.asarray(midpoint)
    ))
    owner_integral = float(np.sum(owner_weights[active] * owner[active]))
    np.testing.assert_allclose(owner_integral, incidence_integral, atol=2e-11, rtol=2e-12)

    # The old midpoint-J normalization is not paired with these quadrature
    # raw volumes and therefore need not reproduce the incidence integral.
    owner_midpoint = np.asarray(
        aggregate_local_control_volume_average(midpoint, cells, domain)
    )
    old_integral = float(np.sum(owner_weights[active] * owner_midpoint[active]))
    assert abs(old_integral - incidence_integral) > 1.0e-7


def _eta_fields(shape):
    nx, ny, nz = shape
    rf = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    tf = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    zf = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    r = (rf[:-1] + rf[1:])[:, None, None] / 2.0
    t = (tf[:-1] + tf[1:])[None, :, None] / 2.0
    z = (zf[:-1] + zf[1:])[None, None, :] / 2.0
    s = (r - RHO_MIN) / (1.0 - RHO_MIN)
    f = jnp.sin(jnp.pi * s) * (jnp.cos(t) + 0.19 * jnp.sin(z))
    g = 1.4 + 0.23 * jnp.cos(2.0 * t - z) + 0.11 * jnp.sin(2.0 * jnp.pi * s) * jnp.cos(z)
    return tuple(jnp.broadcast_to(q, shape) for q in (f, g))


def _eta_evaluate(shape, shard_counts):
    halo_width = 2
    owned_shape = tuple(n // p for n, p in zip(shape, shard_counts, strict=True))
    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)
    f, g = _eta_fields(shape)
    one = jnp.ones(shape, dtype=jnp.float64)
    with make_mesh_for_shard_counts(shard_counts) as mesh:
        placed = tuple(put_scalar_field_on_mesh(q, mesh) for q in (f, g, one))

        def kernel(f_owned, g_owned, one_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                owned_shape, halo_width, global_shape=shape, shard_index=shard_index
            )
            bc = _build_radial_neumann_bc(geometry, domain)
            halos = tuple(
                _prepare_scalar_field_halo(
                    q, geometry, domain, ghost_filler=ghost_filler, face_bc=bc
                )[0]
                for q in (f_owned, g_owned, one_owned)
            )
            context = StencilBuilderContext(layout=domain.layout, domain=domain)
            stencils = tuple(
                build_local_conservative_stencil_from_field(q, geometry, context)
                for q in halos
            )

            def bracket(i, j):
                return local_poisson_bracket_compatible_flux_op(
                    stencils[i], stencils[j], geometry, domain=domain,
                    characteristic_scheme="centered",
                )

            return jnp.stack(
                (bracket(0, 1), bracket(1, 0), bracket(0, 2), bracket(2, 0)),
                axis=0,
            )

        result = jax.jit(jax.shard_map(
            kernel, mesh=mesh,
            in_specs=(P("x", "y", "z"),) * 3,
            out_specs=P(None, "x", "y", "z"), check_vma=False,
        ))(*placed)
    return np.asarray(result)


def test_centered_bracket_is_eta_shard_invariant():
    if os.environ.get("DRBX_PB_ETA_CONTRACT_CHILD") != "1":
        environment = os.environ.copy()
        flags = environment.get("XLA_FLAGS", "")
        environment["XLA_FLAGS"] = (
            f"{flags} --xla_force_host_platform_device_count=2"
        ).strip()
        environment["DRBX_PB_ETA_CONTRACT_CHILD"] = "1"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__)), "-q", "-k", "eta_shard_invariant"],
            cwd=ROOT, env=environment, text=True, capture_output=True, timeout=180,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return

    shape = (6, 12, 8)
    one = _eta_evaluate(shape, (1, 1, 1))
    two = _eta_evaluate(shape, (1, 1, 2))
    np.testing.assert_allclose(two, one, atol=2.0e-12, rtol=2.0e-12)
    np.testing.assert_allclose(two[0] + two[1], 0.0, atol=2.0e-13, rtol=0.0)
    np.testing.assert_allclose(two[2], 0.0, atol=2.0e-13, rtol=0.0)
    np.testing.assert_allclose(two[3], 0.0, atol=2.0e-13, rtol=0.0)
