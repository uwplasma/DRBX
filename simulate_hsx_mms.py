#!/usr/bin/env python3
"""Short production-path HSX/RLP MMS audit.

The continuous metric is fitted once on the 64-grid.  ``build_hsx_fci_geometry``
samples that same context at 32, 48, and 64, while all RLP and EB operators are
the production implementations imported from ``simulate_hsx_blob``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import simulate_hsx_blob as blob  # noqa: E402

RESOLUTIONS = (32, 48, 64)
PERIODIC_AXES = (False, True, True)
AXIS_REGULAR_AXES = (True, False, False)
FIELDS = tuple(blob.FciDrbEBState.__dataclass_fields__)
EVOLVED = ("density", "Te", "Ti", "Vi", "Ve", "vorticity")
REGIONS = (
    "ordinary_bulk",
    "rlp_rings",
    "rlp_transition_rings",
    "physical_wall",
    "short_leg_topology_transition",
    "double_hit",
)


def _fourth_order_fd_weights(coordinates, center, derivative_order):
    """Return 5-point finite-difference weights on arbitrary coordinates."""

    x = np.asarray(coordinates, dtype=np.float64)
    if x.shape != (5,):
        raise ValueError("fourth-order stencil requires exactly five coordinates")
    if derivative_order not in (1, 2):
        raise ValueError("only first and second derivatives are supported")
    offsets = x - float(center)
    vandermonde = np.vstack([offsets ** power for power in range(5)])
    rhs = np.zeros(5, dtype=np.float64)
    rhs[derivative_order] = float(math.factorial(derivative_order))
    return np.linalg.solve(vandermonde, rhs)


def _fourth_order_structured_derivatives(values, coordinates, periods=None):
    """Differentiate a structured scalar tensor with nonuniform 5-point stencils.

    ``coordinates`` contains one coordinate vector per tensor axis.  Periodic
    axes use wrapped indices with unwrapped stencil coordinates; nonperiodic
    radial axes use one-sided five-point stencils at both ends.  The returned
    gradient and Hessian are fourth-order accurate for smooth data (up to the
    accuracy represented by the supplied coordinates).
    """

    scalar = np.asarray(values, dtype=np.float64)
    if scalar.ndim != 3:
        raise ValueError("structured derivative input must be three-dimensional")
    if periods is None:
        periods = (None, None, None)
    if len(coordinates) != 3 or len(periods) != 3:
        raise ValueError("three coordinate vectors and periods are required")

    def along_axis(data, axis, derivative_order):
        moved = np.moveaxis(np.asarray(data, dtype=np.float64), axis, 0)
        count = moved.shape[0]
        coord = np.asarray(coordinates[axis], dtype=np.float64)
        if coord.shape != (count,):
            raise ValueError("coordinate length does not match tensor shape")
        period = periods[axis]
        result = np.empty_like(moved)
        for index in range(count):
            if period is None:
                if index < 2:
                    indices = np.arange(5)
                elif index >= count - 2:
                    indices = np.arange(count - 5, count)
                else:
                    indices = np.arange(index - 2, index + 3)
                stencil_coordinates = coord[indices]
            else:
                offsets = np.arange(-2, 3)
                indices = (index + offsets) % count
                # Choose the periodic image nearest the target coordinate.
                # The Gauss coordinate vectors are strictly ordered within a
                # period, so this gives an ordered, unwrapped stencil even at
                # either periodic boundary.
                stencil_coordinates = coord[indices] + period * np.round(
                    (coord[index] - coord[indices]) / period
                )
                stencil_coordinates[2] = coord[index]
            weights = _fourth_order_fd_weights(
                stencil_coordinates, coord[index], derivative_order
            )
            result[index] = np.tensordot(
                weights, moved[indices], axes=(0, 0)
            )
        return np.moveaxis(result, 0, axis)

    gradient = np.stack(
        tuple(along_axis(scalar, axis, 1) for axis in range(3)), axis=-1
    )
    hessian = np.empty(scalar.shape + (3, 3), dtype=np.float64)
    for first_axis in range(3):
        for second_axis in range(3):
            hessian[..., first_axis, second_axis] = along_axis(
                along_axis(scalar, second_axis, 1), first_axis, 1
            )
    return gradient, hessian

# Keep the continuum-reference coefficients and the production parameter
# object in one auditable, serializable contract.  The cluster campaign keeps
# parallel diffusion/viscosity and electron collisions disabled while turning
# on the same perpendicular diffusion coefficient for every evolved field.
PHYSICAL_PARAMETERS = {
    "tau": 1.0,
    "mi_over_me": 1836.0,
    "rho_star": 1.0,
    "density_D_perp": 1.0e-5,
    "electron_temperature_D_perp": 1.0e-5,
    "ion_temperature_D_perp": 1.0e-5,
    "Vi_D_perp": 1.0e-5,
    "Ve_D_perp": 1.0e-5,
    "vorticity_D_perp": 1.0e-5,
    "density_D_parallel": 0.0,
    "electron_temperature_chi_parallel": 0.0,
    "ion_temperature_chi_parallel": 0.0,
    "Vi_parallel_viscosity": 0.0,
    "Ve_parallel_viscosity": 0.0,
    "vorticity_D_parallel": 0.0,
    "Ve_nu": 0.0,
}

# Keep the elliptic solve policy identical to the current production driver
# defaults and to the attached 64^3 reference run. These values affect the
# reconstructed potential used by every explicit IMEX stage, so they are part
# of the numerical method rather than merely launch-time performance knobs.
PRODUCTION_GMRES = {
    "target_tolerance": 1.0e-8,
    "acceptance_tolerance": 5.0e-5,
    "max_iterations": 500,
    "restart": 100,
    "preconditioner": "line-u",
    "residual_correction_steps": 1,
}


def _production_configuration(shard_counts, device_count):
    """Return the complete auditable production contract used by the MMS."""

    return {
        "flux_framework": "production-split",
        "parallel_operator_scheme": "fci",
        "parallel_velocity_layout": "cell-centered",
        "parallel_flux_pairing": "support-core",
        "parallel_boundary_pairing": "characteristic-sat",
        "parallel_characteristic_wall_law": "energy-absorbing",
        "characteristic_sat_affine_current_lift": "enabled",
        "parallel_current_phi_pair": "enabled",
        "parallel_inflow_closure": "central",
        "parallel_short_leg_treatment": "local-backward-euler",
        "parallel_short_leg_selection": "all-physical-walls",
        "parallel_short_leg_cfl_limit": 2.5,
        "parallel_short_leg_implicit_terms": [
            "selected-characteristic-material-action",
            "selected-mu-tau-grad-parallel-Ti",
        ],
        "parallel_short_leg_explicit_energy_pair": (
            "mu-grad-parallel-phi<->weighted-adjoint-current-divergence"
        ),
        "fci_parallel_leg_scheme": "centered",
        "time_integrator": "imex-ssp222",
        "poisson_bracket_scheme": "material-scalar-third-order-upwind",
        "parallel_material_scheme": "production-path",
        "curvature_scheme": "conservative",
        "curvature_operator": "production-characteristic-owner-face",
        "curvature_rlp_face_scheme": "projected-fine",
        "curvature_wall_flux_closure": (
            "bc-characteristic-operator-trace-canonical-face-state"
        ),
        "curvature_equations": ["density", "Te", "Ti", "vorticity"],
        "curvature_evolution_component": "full",
        "vorticity_current_inflow_trace": "operator",
        "angular_rlp": "automatic-radius-dependent-projected-fine",
        "neumann_ghost_scheme": "physical",
        "physical_wall_model": "legacy-velocity-trace",
        "parallel_velocity_wall_bc": "neumann",
        "fci_trace_substeps": 4,
        "halo_width": 2,
        "fit_sample_shape": [64, 64, 64],
        "toroidal_modes": 10,
        "metric_reference_resolution": [64, 64, 64],
        "metric_radial_degree": 17,
        "metric_poloidal_modes": 15,
        "metric_toroidal_modes": 3,
        "eta_projection_iterations": 0,
        "axis_core_radius": 0.03,
        "makegrid_currents": list(blob.DEFAULT_HSX_QHS_MAKEGRID_CURRENTS),
        "gmres_target_tolerance": PRODUCTION_GMRES["target_tolerance"],
        "gmres_acceptance_tolerance": PRODUCTION_GMRES[
            "acceptance_tolerance"
        ],
        "gmres_max_iterations": PRODUCTION_GMRES["max_iterations"],
        "gmres_restart": PRODUCTION_GMRES["restart"],
        "gmres_preconditioner": PRODUCTION_GMRES["preconditioner"],
        "gmres_residual_correction_steps": PRODUCTION_GMRES[
            "residual_correction_steps"
        ],
        "evolved_initial_phi": "analytic-manufactured",
        "frozen_phi_audit": "exact-and-reconstructed",
        "shard_counts": list(shard_counts),
        "device_count": int(device_count),
    }


def _production_selector_args():
    """Return the canonical Stage-7 production selector contract."""

    return SimpleNamespace(
        flux_framework="production-split",
        topology="toroidal",
        parallel_operator_scheme="fci",
        parallel_flux_pairing="support-core",
        parallel_boundary_pairing="characteristic-sat",
        parallel_characteristic_wall_law="energy-absorbing",
        physical_wall_model="legacy-velocity-trace",
        parallel_velocity_wall_bc="neumann",
        parallel_short_leg_treatment="local-backward-euler",
        parallel_short_leg_selection="all-physical-walls",
        parallel_short_leg_cfl_limit=2.5,
        time_integrator="imex-ssp222",
        poisson_bracket_scheme="material-scalar-third-order-upwind",
        rhs_replay_history=None,
    )


def analytic_self_test() -> dict[str, float]:
    """Run the independent continuum-reference identity test."""

    from hsx_mms_continuum_reference import analytic_identity_metric_self_test

    return analytic_identity_metric_self_test()


def _validate_shard_configuration(
    shard_counts,
    resolutions,
    *,
    available_devices: int | None = None,
    check_device_count: bool = True,
) -> tuple[tuple[int, int, int], int]:
    """Validate the production eta-only mesh before building any geometry."""

    try:
        counts = tuple(int(value) for value in shard_counts)
    except (TypeError, ValueError) as exc:
        raise ValueError("shard_counts must contain three integers") from exc
    if len(counts) != 3 or any(value < 1 for value in counts):
        raise ValueError(
            "shard_counts must contain three positive integers; "
            f"got {counts!r}"
        )
    if counts[:2] != (1, 1):
        raise ValueError(
            "production RLP sharding is eta-only; shard_counts must be "
            f"(1, 1, SZ), got {counts!r}"
        )
    device_count = int(np.prod(counts))
    if available_devices is None:
        available_devices = len(blob.jax.devices())
    available_devices = int(available_devices)
    if check_device_count and device_count > available_devices:
        raise ValueError(
            f"shard_counts={counts!r} requires {device_count} JAX devices, "
            f"but only {available_devices} are available"
        )
    try:
        resolution_values = tuple(int(value) for value in resolutions)
    except (TypeError, ValueError) as exc:
        raise ValueError("resolutions must contain integers") from exc
    for resolution in resolution_values:
        if resolution % counts[2]:
            raise ValueError(
                f"resolution {resolution} is not divisible by eta shard count "
                f"SZ={counts[2]}"
            )
    return counts, device_count


def _step_schedule(start_time, final_time, requested_timestep):
    """Return a stable step count and exact endpoint-aligned timestep.

    Decimal campaign ratios such as ``1e-4 / 1e-6`` can land one ulp above an
    integer in binary floating point.  Snap only ratios within roundoff of an
    integer; genuinely nonintegral durations still use the conservative ceil
    needed to avoid exceeding the requested timestep.
    """

    start = float(start_time)
    final = float(final_time)
    requested = float(requested_timestep)
    if requested <= 0.0:
        raise ValueError("requested timestep must be positive")
    duration = final - start
    if duration < 0.0:
        raise ValueError("final time must not precede start time")
    if duration == 0.0:
        return 0, 0.0
    ratio = duration / requested
    nearest = int(round(ratio))
    roundoff = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(ratio))
    if nearest >= 1 and abs(ratio - nearest) <= roundoff:
        num_steps = nearest
    else:
        num_steps = max(1, int(math.ceil(ratio)))
    return num_steps, duration / float(num_steps)


def _validate_reusable_history(
    history,
    history_path,
    *,
    expected_configuration,
    expected_initial_state,
    start_time,
    timestep,
    num_steps,
    save_every,
):
    """Reject stale histories that would change the MMS initial-value problem."""

    if "run_metadata_json" not in history:
        raise ValueError(f"{history_path} is missing run_metadata_json")
    try:
        metadata = json.loads(str(np.asarray(
            history["run_metadata_json"]
        ).reshape(-1)[0]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{history_path} has invalid run_metadata_json"
        ) from exc
    mismatches = {
        name: (metadata.get(name), expected)
        for name, expected in expected_configuration.items()
        if metadata.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            f"{history_path} production configuration mismatch: {mismatches}"
        )

    saved_times = np.asarray(history["times"], dtype=np.float64)
    saved_steps = [0] + [
        step for step in range(1, int(num_steps) + 1)
        if step % int(save_every) == 0 or step == int(num_steps)
    ]
    expected_times = float(start_time) + float(timestep) * np.asarray(
        saved_steps, dtype=np.float64
    )
    if saved_times.shape != expected_times.shape or not np.allclose(
        saved_times, expected_times, rtol=0.0, atol=1.0e-15
    ):
        raise ValueError(
            f"{history_path} saved times do not match the requested schedule"
        )

    for field in FIELDS:
        if field not in history:
            raise ValueError(f"{history_path} is missing field {field!r}")
        observed = np.asarray(history[field][0], dtype=np.float64)
        expected = np.asarray(
            getattr(expected_initial_state, field), dtype=np.float64
        )
        if observed.shape != expected.shape or not np.array_equal(
            observed, expected
        ):
            maximum = (
                float(np.max(np.abs(observed - expected)))
                if observed.shape == expected.shape else math.inf
            )
            raise ValueError(
                f"{history_path} initial {field} is not the analytical "
                f"manufactured field (max difference {maximum:.6e})"
            )


def _owner_project(state, host):
    return blob._aggregate_initial_owner_state(state, host)


class _QuadratureProjector:
    """Resolution-local high-order projector with cached metric preparation."""

    def __init__(self, reference, geometry, host, *, chunk_cells=4096):
        nodes, weights = np.polynomial.legendre.leggauss(2)
        uf, tf, ef = (np.asarray(geometry.grid.x.faces), np.asarray(geometry.grid.y.faces), np.asarray(geometry.grid.z.faces))
        # The finite-difference period is the coordinate-domain extent, not
        # necessarily the evaluator's one-field-period harmonic.  In a full
        # torus eta can contain several field periods when nfp>1.
        self.periodic_domain_lengths = (
            None,
            float(tf[-1] - tf[0]),
            float(ef[-1] - ef[0]),
        )
        lo = np.stack(np.meshgrid(uf[:-1], tf[:-1], ef[:-1], indexing="ij"), axis=-1)
        hi = np.stack(np.meshgrid(uf[1:], tf[1:], ef[1:], indexing="ij"), axis=-1)
        center, half = 0.5*(lo+hi), 0.5*(hi-lo)
        off = np.stack(np.meshgrid(nodes,nodes,nodes,indexing="ij"), axis=-1).reshape(-1,3)
        points = center[...,None,:] + half[...,None,:] * off[None,None,None,:,:]
        self.shape = geometry.shape
        self.nq = off.shape[0]
        self.reference = reference
        self.host = host
        flat = points.reshape((-1, 3))
        qweight = np.einsum("i,j,k->ijk", weights, weights, weights).reshape(-1)
        self.chunks = []
        prepared_chunks = []
        cells = int(np.prod(self.shape))
        for first in range(0, cells, int(chunk_cells)):
            last = min(first + int(chunk_cells), cells)
            a, b = first*self.nq, last*self.nq
            chunk_points = flat[a:b]
            prepared = reference.prepare(chunk_points)
            jac = np.asarray(prepared.J, dtype=np.float64)
            weighted = jac.reshape((last-first, self.nq)) * qweight[None, :]
            prepared_chunks.append((first, last, chunk_points, prepared, weighted))

        # The polarization relation is evaluated from the independent cached
        # metric tensor/divergence in ``PreparedGeometry``.  For the enabled
        # production mode, assemble those scalar omega values back onto the
        # 2n-by-2n-by-2n Gauss-point tensor, differentiate there, and slice the
        # results into the same prepared chunks used at every stage.
        if getattr(reference, "enable_generalized_potential", False):
            omega_flat = np.concatenate([
                np.asarray(chunk[3].mms_omega, dtype=np.float64)
                for chunk in prepared_chunks
            ])
            nx, ny, nz = (int(value) for value in self.shape)
            omega_cells = omega_flat.reshape((nx, ny, nz, 2, 2, 2))
            omega_structured = omega_cells.transpose(
                0, 3, 1, 4, 2, 5
            ).reshape((2 * nx, 2 * ny, 2 * nz))
            structured_points = points.reshape(
                (nx, ny, nz, 2, 2, 2, 3)
            ).transpose(0, 3, 1, 4, 2, 5, 6).reshape(
                (2 * nx, 2 * ny, 2 * nz, 3)
            )
            coordinates = (
                structured_points[:, 0, 0, 0],
                structured_points[0, :, 0, 1],
                structured_points[0, 0, :, 2],
            )
            gradient_structured, hessian_structured = (
                _fourth_order_structured_derivatives(
                    omega_structured,
                    coordinates,
                    periods=self.periodic_domain_lengths,
                )
            )
            def cell_order(structured, tail=()):
                shaped = structured.reshape(
                    (nx, 2, ny, 2, nz, 2) + tuple(tail)
                )
                return shaped.transpose(
                    (0, 2, 4, 1, 3, 5) + tuple(range(6, 6 + len(tail)))
                ).reshape((cells * self.nq,) + tuple(tail))

            gradient_flat = cell_order(gradient_structured, (3,))
            hessian_flat = cell_order(hessian_structured, (3, 3))
            for first, last, chunk_points, prepared, weighted in prepared_chunks:
                prepared = replace(
                    prepared,
                    mms_omega=omega_flat[first * self.nq:last * self.nq],
                    mms_omega_gradient=gradient_flat[
                        first * self.nq:last * self.nq
                    ],
                    mms_omega_hessian=hessian_flat[
                        first * self.nq:last * self.nq
                    ],
                )
                self.chunks.append(
                    (first, last, chunk_points, prepared, weighted)
                )
        else:
            self.chunks = prepared_chunks

    def evaluate(self, time):
        cells = int(np.prod(self.shape))
        outputs = {"values": {n: np.empty(cells) for n in FIELDS},
                   "time_derivatives": {n: np.empty(cells) for n in FIELDS},
                   "continuum": {n: np.empty(cells) for n in EVOLVED}}
        for first, last, points, prepared, weighted in self.chunks:
            data = self.reference.evaluate(points, time, prepared=prepared)
            rhs = self.reference.continuum_rhs(points, time, prepared=prepared)
            denom = np.sum(weighted, axis=1)
            for n in FIELDS:
                outputs["values"][n][first:last] = np.sum(weighted * np.asarray(data.values[n])[:,None].reshape((last-first,self.nq)), axis=1) / np.maximum(denom, 1e-30)
                outputs["time_derivatives"][n][first:last] = np.sum(weighted * np.asarray(data.time_derivatives[n])[:,None].reshape((last-first,self.nq)), axis=1) / np.maximum(denom, 1e-30)
            for n in EVOLVED:
                outputs["continuum"][n][first:last] = np.sum(weighted * np.asarray(rhs[n])[:,None].reshape((last-first,self.nq)), axis=1) / np.maximum(denom, 1e-30)
        def state(mapping, default_zero=False):
            return blob.FciDrbEBState(**{n: mapping[n].reshape(self.shape) if n in mapping else np.zeros(self.shape) for n in FIELDS})
        point_state = state(outputs["values"])
        point_qdot = state(outputs["time_derivatives"])
        continuum = blob.FciDrbEBState(**{n: outputs["continuum"][n].reshape(self.shape) if n in outputs["continuum"] else np.zeros(self.shape) for n in FIELDS})
        source = blob.FciDrbEBState(**{n: (np.asarray(getattr(point_qdot,n)) - np.asarray(getattr(continuum,n))) if n in EVOLVED else np.zeros(self.shape) for n in FIELDS})
        return point_state, point_qdot, source, continuum

    def evaluate_continuum_terms(self, time):
        """Project the independent continuum term split to raw cells."""

        cells = int(np.prod(self.shape))
        outputs = None
        for first, last, points, prepared, weighted in self.chunks:
            terms = self.reference.continuum_terms(
                points, time, prepared=prepared
            )
            if outputs is None:
                outputs = {
                    field: {
                        name: np.empty(cells, dtype=np.float64)
                        for name in field_terms
                    }
                    for field, field_terms in terms.items()
                }
            denominator = np.sum(weighted, axis=1)
            for field, field_terms in terms.items():
                for name, values in field_terms.items():
                    shaped = np.asarray(values).reshape((last - first, self.nq))
                    outputs[field][name][first:last] = np.sum(
                        weighted * shaped, axis=1
                    ) / np.maximum(denominator, 1.0e-30)
        if outputs is None:
            raise RuntimeError("continuum term projection has no quadrature cells")
        return {
            field: {
                name: values.reshape(self.shape)
                for name, values in field_terms.items()
            }
            for field, field_terms in outputs.items()
        }


def _reference_state(reference, geometry, host, time, projector=None):
    """High-order Gauss projection, using a cached per-resolution projector."""
    if projector is not None:
        return projector.evaluate(time)
    nodes, weights = np.polynomial.legendre.leggauss(2)
    uf, tf, ef = (np.asarray(geometry.grid.x.faces), np.asarray(geometry.grid.y.faces), np.asarray(geometry.grid.z.faces))
    lo = np.stack(np.meshgrid(uf[:-1], tf[:-1], ef[:-1], indexing="ij"), axis=-1)
    hi = np.stack(np.meshgrid(uf[1:], tf[1:], ef[1:], indexing="ij"), axis=-1)
    center, half = 0.5*(lo+hi), 0.5*(hi-lo)
    off = np.stack(np.meshgrid(nodes,nodes,nodes,indexing="ij"), axis=-1).reshape(-1,3)
    points = center[...,None,:] + half[...,None,:] * off[None,None,None,:,:]
    flat = points.reshape(-1,3)
    qweight = np.einsum("i,j,k->ijk", weights, weights, weights).reshape(-1)
    jac_parts = []
    for start in range(0, flat.shape[0], 100_000):
        stop = min(start + 100_000, flat.shape[0])
        jac_parts.append(np.asarray(reference.metric_evaluator.evaluate(flat[start:stop], reject_nonpositive_J=False).signed_J))
    jac = np.concatenate(jac_parts).reshape(points.shape[:-1])
    weighted = jac * qweight[None,None,None,:]
    denom = np.sum(weighted, axis=-1)
    def project(values):
        values = np.asarray(values).reshape(points.shape[:-1])
        return np.sum(weighted * values, axis=-1) / np.maximum(denom, 1e-30)
    # Keep the independent reference evaluation bounded for the 64³ case.
    chunks = []
    continuum_chunks = []
    for start in range(0, flat.shape[0], 100_000):
        stop = min(start + 100_000, flat.shape[0])
        chunks.append(reference.evaluate(flat[start:stop], time))
        prepared = reference.prepare(flat[start:stop])
        continuum_chunks.append(reference.continuum_rhs(
            flat[start:stop], time, prepared=prepared))
    def join(name, attr):
        return np.concatenate([getattr(chunk, attr)[name] for chunk in chunks])
    def join_cont(name):
        return np.concatenate([chunk[name] for chunk in continuum_chunks])
    def state_from(mapping):
        return blob.FciDrbEBState(**{n: project(mapping[n]) for n in FIELDS})
    values = {n: join(n, "values") for n in FIELDS}
    derivatives = {n: join(n, "time_derivatives") for n in FIELDS}
    state = state_from(values)
    qdot = state_from(derivatives)
    continuum_flat = {n: join_cont(n) for n in EVOLVED}
    source = blob.FciDrbEBState(**{
        n: project(derivatives[n] - continuum_flat[n]) if n in EVOLVED
        else np.zeros_like(np.asarray(state.phi)) for n in FIELDS})
    continuum = blob.FciDrbEBState(**{
        n: project(continuum_flat[n]) if n in EVOLVED
        else np.zeros_like(np.asarray(state.phi)) for n in FIELDS})
    return state, qdot, source, continuum


def _runtime(geometry, host, args):
    """Build host diagnostics and the eta-sharded production run payload."""

    from drbx.native import (
        assemble_single_device_local_fci_geometry,
        build_local_fci_geometries,
        make_shard_mesh,
    )
    from drbx.native.fci_angular_agglomeration import (
        assemble_local_polar_angular_agglomeration_geometry,
        build_sharded_polar_angular_agglomeration_payload,
        empty_angular_agglomeration_boundary_bc,
    )
    shard_counts, device_count = _validate_shard_configuration(
        getattr(args, "shard_counts", (1, 1, 1)),
        (geometry.shape[2],),
    )
    sharded_local = build_local_fci_geometries(
        geometry, shard_counts, halo_width=2,
        periodic_axes=PERIODIC_AXES,
        axis_regular_axes=AXIS_REGULAR_AXES,
    )
    sharded_descriptor, sharded_control_fields = build_sharded_polar_angular_agglomeration_payload(
        host,
        sharded_local.domain,
        compile_compact_transition_faces=False,
    )
    params = blob.FciDrbEBRhsParameters(
        tau=PHYSICAL_PARAMETERS["tau"],
        mi_over_me=PHYSICAL_PARAMETERS["mi_over_me"],
        rho_star=PHYSICAL_PARAMETERS["rho_star"],
        phi_inversion_iterations=PRODUCTION_GMRES["max_iterations"],
        phi_inversion_regularization=0.0,
        density_D_perp=PHYSICAL_PARAMETERS["density_D_perp"],
        density_D_parallel=PHYSICAL_PARAMETERS["density_D_parallel"],
        electron_temperature_chi_parallel=PHYSICAL_PARAMETERS[
            "electron_temperature_chi_parallel"
        ],
        electron_temperature_D_perp=PHYSICAL_PARAMETERS[
            "electron_temperature_D_perp"
        ],
        ion_temperature_chi_parallel=PHYSICAL_PARAMETERS[
            "ion_temperature_chi_parallel"
        ],
        ion_temperature_D_perp=PHYSICAL_PARAMETERS["ion_temperature_D_perp"],
        Ve_nu=PHYSICAL_PARAMETERS["Ve_nu"],
        Ve_D_perp=PHYSICAL_PARAMETERS["Ve_D_perp"],
        Ve_parallel_viscosity=PHYSICAL_PARAMETERS["Ve_parallel_viscosity"],
        Vi_D_perp=PHYSICAL_PARAMETERS["Vi_D_perp"],
        Vi_parallel_viscosity=PHYSICAL_PARAMETERS["Vi_parallel_viscosity"],
        vorticity_D_perp=PHYSICAL_PARAMETERS["vorticity_D_perp"],
        vorticity_D_parallel=PHYSICAL_PARAMETERS["vorticity_D_parallel"],
        parallel_characteristic_wall_law="energy-absorbing")
    sharded_boundary_bc = empty_angular_agglomeration_boundary_bc(
        max_rows=int(getattr(sharded_descriptor, "compact_face_count", 0))
    )
    local_geometry = None
    host_descriptor = None
    host_control_fields = None
    host_boundary_bc = None
    cv = None
    model = None
    frozen_execution = "eta-sharded"
    if shard_counts == (1, 1, 1):
        # Preserve the inexpensive local path for development and the real-HSX
        # wiring smoke.  Multi-device campaigns must not construct this second
        # full-torus geometry/model solely for frozen diagnostics.
        host_local = sharded_local
        host_descriptor = sharded_descriptor
        host_control_fields = sharded_control_fields
        local_geometry = assemble_single_device_local_fci_geometry(host_local)
        host_domain = replace(
            host_local.domain, mesh_axis_names=(None, None, None)
        )
        host_boundary_bc = empty_angular_agglomeration_boundary_bc(
            max_rows=int(getattr(host_descriptor, "compact_face_count", 0))
        )
        cv = assemble_local_polar_angular_agglomeration_geometry(
            host_descriptor, host_control_fields, local_geometry
        )
        model = blob.build_local_eb_model(
            local_geometry, host_domain, params,
            gmres_target_tolerance=PRODUCTION_GMRES["target_tolerance"],
            gmres_acceptance_tolerance=PRODUCTION_GMRES[
                "acceptance_tolerance"
            ],
            gmres_max_iterations=PRODUCTION_GMRES["max_iterations"],
            gmres_restart=PRODUCTION_GMRES["restart"],
            gmres_preconditioner=PRODUCTION_GMRES["preconditioner"],
            gmres_residual_correction_steps=PRODUCTION_GMRES[
                "residual_correction_steps"
            ],
            poisson_bracket_scheme="material-scalar-third-order-upwind",
            parallel_operator_scheme="fci",
            parallel_material_scheme="production-path",
            control_volume_geometry=cv,
            control_volume_boundary_bc=host_boundary_bc,
        )
        expected_contract = {
            "parallel_operator_scheme": "fci",
            "parallel_material_scheme": "production-path",
            "parallel_flux_pairing": "support-core",
            "parallel_boundary_pairing": "characteristic-sat",
            "parallel_short_leg_treatment": "local-backward-euler",
            "parallel_short_leg_selection": "all-physical-walls",
            "poisson_bracket_scheme": "material-scalar-third-order-upwind",
        }
        mismatches = {
            name: (getattr(model, name), expected)
            for name, expected in expected_contract.items()
            if getattr(model, name) != expected
        }
        if model.parameters.parallel_characteristic_wall_law != "energy-absorbing":
            mismatches["parallel_characteristic_wall_law"] = (
                model.parameters.parallel_characteristic_wall_law,
                "energy-absorbing",
            )
        if model.control_volume_geometry is None:
            mismatches["control_volume_geometry"] = (
                None, "projected-fine RLP"
            )
        if model.neumann_normal_scheme != "physical":
            mismatches["neumann_ghost_scheme"] = (
                model.neumann_normal_scheme, "physical"
            )
        if model.physical_wall_model_name != "legacy-velocity-trace":
            mismatches["physical_wall_model"] = (
                model.physical_wall_model_name, "legacy-velocity-trace"
            )
        for name, expected in (
            ("tol", PRODUCTION_GMRES["target_tolerance"]),
            ("acceptance_tol", PRODUCTION_GMRES["acceptance_tolerance"]),
            ("maxiter", PRODUCTION_GMRES["max_iterations"]),
            ("restart", PRODUCTION_GMRES["restart"]),
            ("preconditioner", PRODUCTION_GMRES["preconditioner"]),
            (
                "residual_correction_steps",
                PRODUCTION_GMRES["residual_correction_steps"],
            ),
        ):
            actual = getattr(model.gmres_config, name)
            if actual != expected:
                mismatches[f"gmres_{name}"] = (actual, expected)
        if mismatches:
            raise RuntimeError(
                f"Stage-7 production runtime contract mismatch: {mismatches}"
            )
        frozen_execution = "host-single-device"
    return SimpleNamespace(
        sharded_geometry=sharded_local,
        local_geometry=local_geometry,
        control_volume_geometry=cv,
        control_volume_descriptor=sharded_descriptor,
        control_volume_fields=sharded_control_fields,
        control_volume_boundary_bc=sharded_boundary_bc,
        host_control_volume_descriptor=host_descriptor,
        host_control_volume_fields=host_control_fields,
        host_control_volume_boundary_bc=host_boundary_bc,
        control_volume_assembler=assemble_local_polar_angular_agglomeration_geometry,
        parameters=params,
        model=model,
        mesh=make_shard_mesh(shard_counts),
        shard_counts=shard_counts,
        device_count=device_count,
        frozen_execution=frozen_execution,
        evolved_execution="eta-sharded",
    )


def _weighted_norm(value, host):
    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    volume = np.asarray(host.aggregate_chart_volume)
    value = np.asarray(value)
    return float(np.sqrt(np.sum(volume[active] * value[active]**2) / np.sum(volume[active])))


def _owner_project_array(value, host):
    """Volume-project one raw-cell scalar onto canonical RLP owners."""

    raw = np.asarray(value, dtype=np.float64)
    raw_volume = np.asarray(host.raw_volume, dtype=np.float64)
    aggregate_volume = np.asarray(host.aggregate_chart_volume, dtype=np.float64)
    owner_mask = np.asarray(host.topology.is_active_owner, dtype=bool)
    aggregate_ids = np.asarray(host.topology.aggregate_id, dtype=np.int64).ravel()
    owner_flat_ids = np.flatnonzero(owner_mask.ravel())
    owner_sums = np.zeros(raw.size, dtype=np.float64)
    np.add.at(owner_sums, aggregate_ids, (raw * raw_volume).ravel())
    projected = np.zeros(raw.size, dtype=np.float64)
    projected[owner_flat_ids] = owner_sums[owner_flat_ids] / np.maximum(
        aggregate_volume.ravel()[owner_flat_ids], np.finfo(float).tiny
    )
    return projected.reshape(raw.shape)


def _continuum_term_ledger(raw_terms, host):
    """Align independent continuum terms with the production ledger slots."""

    shape = np.asarray(host.topology.is_active_owner).shape
    ledger = np.zeros(
        (
            len(blob.RHS_TERM_FIELD_NAMES),
            max(len(names) for names in blob.RHS_TERM_NAMES),
        ) + shape,
        dtype=np.float64,
    )
    for field_index, field in enumerate(blob.RHS_TERM_FIELD_NAMES):
        for name, value in raw_terms[field].items():
            try:
                slot = blob.RHS_TERM_NAMES[field_index].index(name)
            except ValueError as exc:
                raise ValueError(
                    f"continuum term {field}.{name} has no production ledger slot"
                ) from exc
            ledger[field_index, slot] = _owner_project_array(value, host)
    return ledger


def _masked_weighted_norm(value, host, mask):
    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    selected = active & np.asarray(mask, dtype=bool)
    volume = np.asarray(host.aggregate_chart_volume, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    denominator = float(np.sum(volume[selected]))
    if denominator <= 0.0:
        return float("nan")
    return float(np.sqrt(np.sum(volume[selected] * value[selected] ** 2) / denominator))


def _region_masks(geometry, host, selected_wall):
    """Return disjoint owner masks for the Stage-7 localization audit."""

    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    aggregate_ids = np.asarray(
        host.topology.aggregate_id, dtype=np.int64
    ).ravel()

    def owner_any(raw_mask):
        owner_mask = np.zeros(active.size, dtype=bool)
        np.logical_or.at(
            owner_mask,
            aggregate_ids,
            np.asarray(raw_mask, dtype=bool).ravel(),
        )
        return owner_mask.reshape(active.shape) & active

    forward_raw = np.asarray(geometry.maps.forward_boundary, dtype=bool)
    backward_raw = np.asarray(geometry.maps.backward_boundary, dtype=bool)
    wall_raw = forward_raw | backward_raw
    topology_change_raw = np.zeros_like(wall_raw)
    for axis in (1, 2):
        for shift in (-1, 1):
            topology_change_raw |= forward_raw != np.roll(
                forward_raw, shift, axis=axis
            )
            topology_change_raw |= backward_raw != np.roll(
                backward_raw, shift, axis=axis
            )
    selected_wall_raw = np.asarray(selected_wall, dtype=bool)
    wall = owner_any(wall_raw)
    double_hit = owner_any(forward_raw & backward_raw)
    short_transition = owner_any(
        selected_wall_raw & topology_change_raw & ~(forward_raw & backward_raw)
    )

    groups = np.asarray(host.angular_group_size, dtype=np.int64)
    rlp_radial = groups > 1
    rlp_transition_radial = np.zeros_like(rlp_radial)
    if groups.size > 1:
        changes = groups[1:] != groups[:-1]
        rlp_transition_radial[:-1] |= changes
        rlp_transition_radial[1:] |= changes
    rlp = np.broadcast_to(rlp_radial[:, None, None], active.shape)
    rlp_transition = np.broadcast_to(
        rlp_transition_radial[:, None, None], active.shape
    )

    assigned = double_hit.copy()
    masks = {"double_hit": active & double_hit}
    masks["short_leg_topology_transition"] = active & short_transition & ~assigned
    assigned |= short_transition
    masks["physical_wall"] = active & wall & ~assigned
    assigned |= wall
    masks["rlp_transition_rings"] = active & rlp_transition & ~assigned
    assigned |= rlp_transition
    masks["rlp_rings"] = active & rlp & ~assigned
    assigned |= rlp
    masks["ordinary_bulk"] = active & ~assigned
    return {name: masks[name] for name in REGIONS}


def _partitioned_state_norms(state, host, masks):
    return {
        region: {
            field: _masked_weighted_norm(getattr(state, field), host, mask)
            for field in EVOLVED
        }
        for region, mask in masks.items()
    }


def _materialize_owner_mask(mask, host):
    owner_index = np.asarray(host.topology.owner_index, dtype=np.int64)
    return np.asarray(mask, dtype=bool)[
        owner_index[..., 0], owner_index[..., 1], owner_index[..., 2]
    ]


def _short_leg_mode_history(history_path, reference_at, host, masks):
    """Measure localized angular high modes and jumps through an MMS history."""

    with np.load(history_path, allow_pickle=False) as history:
        times = np.asarray(history["times"], dtype=np.float64)
        numerical = {
            field: np.asarray(history[field], dtype=np.float64)
            for field in EVOLVED
        }
    localized = _materialize_owner_mask(
        masks["short_leg_topology_transition"], host
    )
    if not np.any(localized):
        localized = _materialize_owner_mask(masks["physical_wall"], host)
    high_start = max(1, int(np.ceil(numerical[EVOLVED[0]].shape[2] / 3.0)))
    high_fraction = np.full((times.size, len(EVOLVED)), np.nan)
    high_rms = np.full_like(high_fraction, np.nan)
    maximum_jump = np.full_like(high_fraction, np.nan)
    for time_index, time_value in enumerate(times):
        exact_owner = reference_at(float(time_value))[0]
        exact = blob._materialize_owner_state(exact_owner, host)
        for field_index, field in enumerate(EVOLVED):
            error = numerical[field][time_index] - np.asarray(
                getattr(exact, field), dtype=np.float64
            )
            localized_error = np.where(localized, error, 0.0)
            # Orthonormal normalization keeps the localized energy comparable
            # across different angular resolutions.
            spectrum = np.fft.rfft(
                localized_error, axis=1, norm="ortho"
            )
            total = float(np.sum(np.abs(spectrum[:, 1:, :]) ** 2))
            high = float(np.sum(np.abs(spectrum[:, high_start:, :]) ** 2))
            if total > 0.0:
                high_fraction[time_index, field_index] = high / total
            elif high == 0.0:
                # The manufactured initial frame is exact, so both spectral
                # energies are identically zero.  Define its high-mode
                # fraction as zero instead of leaving a 0/0 NaN that would
                # make an otherwise finite diagnostic trajectory unusable.
                high_fraction[time_index, field_index] = 0.0
            count = int(np.count_nonzero(localized))
            if count:
                high_rms[time_index, field_index] = np.sqrt(high / count)
            pair = localized | np.roll(localized, 1, axis=1)
            if np.any(pair):
                jumps = error - np.roll(error, 1, axis=1)
                maximum_jump[time_index, field_index] = np.max(
                    np.abs(jumps[pair])
                )
    late_growth = np.full(len(EVOLVED), np.nan)
    late_growth_factor = np.full(len(EVOLVED), np.nan)
    late_growth_r_squared = np.full(len(EVOLVED), np.nan)
    valid_counts = np.zeros(len(EVOLVED), dtype=np.int64)
    start = max(0, times.size // 2)
    for field_index in range(len(EVOLVED)):
        values = high_rms[start:, field_index]
        valid = np.isfinite(values) & (values > 0.0)
        valid_counts[field_index] = int(np.count_nonzero(valid))
        if np.count_nonzero(valid) >= 2:
            fit_times = times[start:][valid]
            fit_values = np.log(values[valid])
            coefficients = np.polyfit(fit_times, fit_values, 1)
            late_growth[field_index] = coefficients[0]
            fitted = np.polyval(coefficients, fit_times)
            residual_sum = float(np.sum((fit_values - fitted) ** 2))
            total_sum = float(np.sum((fit_values - np.mean(fit_values)) ** 2))
            late_growth_r_squared[field_index] = (
                1.0 if total_sum <= 1.0e-30
                else max(0.0, 1.0 - residual_sum / total_sum)
            )
            late_growth_factor[field_index] = float(np.exp(
                coefficients[0] * (fit_times[-1] - fit_times[0])
            ))
    classification = []
    for field_index in range(len(EVOLVED)):
        if valid_counts[field_index] < 3:
            classification.append("insufficient-time-samples")
        elif (
            late_growth[field_index] > 0.0
            and late_growth_factor[field_index] >= 1.25
            and late_growth_r_squared[field_index] >= 0.5
        ):
            classification.append("positive-growth")
        else:
            classification.append("bounded-or-decaying-closure-layer")
    return {
        "times": times,
        "high_mode_start": high_start,
        "high_mode_fraction": high_fraction,
        "high_mode_rms": high_rms,
        "maximum_poloidal_jump": maximum_jump,
        "late_log_growth_rate": late_growth,
        "late_growth_factor": late_growth_factor,
        "late_growth_r_squared": late_growth_r_squared,
        "classification": classification,
    }


def _implicit_material_state(material, state):
    """Lift the five-field implicit material residual into EB state form."""

    zero = blob.jnp.zeros_like(state.density)
    return blob.FciDrbEBState(
        density=material[..., 0],
        phi=zero,
        Te=material[..., 1],
        Ti=material[..., 2],
        Vi=material[..., 3],
        Ve=material[..., 4],
        vorticity=zero,
    )


def _audit_one(geometry, cell_positions, nfp, args):
    host, _ = blob.build_metric_aware_polar_angular_agglomeration_geometry(geometry, args.metric_context.metric_evaluator)
    runtime = _runtime(geometry, host, args)
    model = runtime.model
    projector = _QuadratureProjector(args.reference, geometry, host)
    point_state, point_qdot, point_source, point_continuum = _reference_state(args.reference, geometry, host, args.time, projector)
    raw_continuum_terms = projector.evaluate_continuum_terms(args.time)
    if getattr(args.reference, "enable_generalized_potential", False):
        omega_max = float(np.max(np.abs(np.asarray(point_state.vorticity))))
        if not np.isfinite(omega_max) or omega_max <= 1.0e-14:
            raise RuntimeError(
                "generalized-potential MMS produced identically zero exact omega"
            )
        for lane in ("poisson_bracket", "parallel_advection", "perpendicular_diffusion"):
            lane_max = float(np.max(np.abs(
                np.asarray(raw_continuum_terms["vorticity"][lane])
            )))
            if not np.isfinite(lane_max) or lane_max <= 1.0e-14:
                raise RuntimeError(
                    f"generalized-potential MMS vorticity lane {lane!r} is zero"
                )
    state = _owner_project(point_state, host)
    qdot = _owner_project(point_qdot, host)
    source = _owner_project(point_source, host)
    continuum = _owner_project(point_continuum, host)
    if runtime.frozen_execution == "eta-sharded":
        frozen = blob.run_full_eb(
            state,
            global_geometry=geometry,
            cell_positions=cell_positions,
            nfp=int(nfp),
            sharded_geometry=runtime.sharded_geometry,
            mesh=runtime.mesh,
            parameters=runtime.parameters,
            metric_cache_path=None,
            gmres_target_tolerance=PRODUCTION_GMRES["target_tolerance"],
            gmres_acceptance_tolerance=PRODUCTION_GMRES[
                "acceptance_tolerance"
            ],
            gmres_max_iterations=PRODUCTION_GMRES["max_iterations"],
            gmres_restart=PRODUCTION_GMRES["restart"],
            gmres_preconditioner=PRODUCTION_GMRES["preconditioner"],
            gmres_residual_correction_steps=PRODUCTION_GMRES[
                "residual_correction_steps"
            ],
            time_integrator="imex-ssp222",
            advance_execution=str(args.advance_execution),
            num_steps=0,
            timestep=float(args.dt),
            start_time=float(args.time),
            output_path=args.output,
            save_every=1,
            phase_timing=False,
            reconstruct_initial_phi=False,
            neumann_ghost_scheme="physical",
            parallel_velocity_wall_bc="neumann",
            parallel_operator_scheme="fci",
            poisson_bracket_scheme="material-scalar-third-order-upwind",
            parallel_material_scheme="production-path",
            control_volume_descriptor=runtime.control_volume_descriptor,
            control_volume_fields_host=runtime.control_volume_fields,
            control_volume_boundary_bc=runtime.control_volume_boundary_bc,
            control_volume_assembler=runtime.control_volume_assembler,
            owner_host_geometry=host,
            history_dtype="float64",
            frozen_diagnostic=blob.FrozenEbDiagnosticRequest(
                source_state=source,
                implicit_solve_dt=(
                    blob.IMEX_SSP222_GAMMA * float(args.dt)
                ),
                implicit_selection_dt=float(args.dt),
                execution=str(args.advance_execution),
            ),
        )
        implicit_material = (
            frozen.exact_implicit_complete_residual_owner
        )
        selected_wall = frozen.exact_selected_wall
        spatial = frozen.exact_explicit
        ledger = frozen.exact_rhs_term_fields
        sourced = frozen.sourced_explicit
        reconstructed = frozen.reconstructed_phi
        reconstructed_spatial = frozen.reconstructed_explicit
        reconstructed_implicit_material = (
            frozen.reconstructed_implicit_complete_residual_owner
        )
        phi_diagnostics = np.asarray(frozen.phi_solver_diagnostics)
        phi_failed = bool(phi_diagnostics[2])
        phi_converged = bool(phi_diagnostics[3])
        # The general production hook exposes all three ledgers.  The MMS
        # needs only the exact-phi ledger after source pairing is established.
        del frozen
    else:
        short_leg_step = blob.jax.jit(
            lambda q: model.apply_short_leg_implicit_material_step(
                q,
                solve_dt=blob.IMEX_SSP222_GAMMA * float(args.dt),
                selection_dt=float(args.dt),
                phi_owned=q.phi,
                return_increment=True,
            )
        )
        _, _, implicit_info = short_leg_step(state)
        implicit_info = blob.jax.block_until_ready(implicit_info)
        implicit_material = implicit_info[
            "selected_complete_residual_owner"
        ]
        selected_wall = implicit_info["selected_wall"]
        # Compile the full frozen production graph as one unit. Evaluating
        # this operator primitive-by-primitive is unrepresentative and slow.
        frozen_stage = blob.jax.jit(
            lambda q, s: model.evaluate_stage(
                q,
                source_owned=s,
                phi_owned=q.phi,
                short_leg_selection_dt=float(args.dt),
                return_rhs_term_fields=True,
            )
        )
        spatial, ledger = frozen_stage(state, state.zeros_like())
        spatial, ledger = blob.jax.block_until_ready((spatial, ledger))
        sourced, _ = frozen_stage(state, source)
        sourced = blob.jax.block_until_ready(sourced)
        reconstruct_phi = blob.jax.jit(
            lambda q: model.reconstruct_phi(q, return_diagnostics=True)
        )
        reconstructed, info = reconstruct_phi(state)
        reconstructed, info = blob.jax.block_until_ready(
            (reconstructed, info)
        )
        reconstructed_state = state.replace(phi=reconstructed)
        reconstructed_spatial, _ = frozen_stage(
            reconstructed_state, state.zeros_like()
        )
        reconstructed_spatial = blob.jax.block_until_ready(
            reconstructed_spatial
        )
        _, _, reconstructed_implicit_info = short_leg_step(
            reconstructed_state
        )
        reconstructed_implicit_info = blob.jax.block_until_ready(
            reconstructed_implicit_info
        )
        reconstructed_implicit_material = reconstructed_implicit_info[
            "selected_complete_residual_owner"
        ]
        phi_converged = bool(np.asarray(info.converged))
        phi_failed = bool(np.asarray(info.failed))

    implicit_state = _implicit_material_state(implicit_material, state)
    # ``evaluate_stage`` is the explicit ARK partition and deliberately omits
    # every selected physical-wall material leg. Add the exact frozen
    # implicit residual back here so spatial truncation measures F+G, while
    # evolved runs still use the production split unchanged.
    spatial = spatial.axpy(implicit_state, scale=1.0)
    ledger = ledger.at[:5, 1].add(
        blob.jnp.moveaxis(implicit_material, -1, 0)
    )
    exact_phi_residual = {
        n: _weighted_norm(getattr(spatial, n) - getattr(continuum, n), host)
        for n in EVOLVED}
    # Independent source: this is the continuum manufactured time derivative,
    # not a value obtained by calling the discrete RHS.  The source-aware
    # result is retained to verify owner-space source addition.
    sourced = sourced.axpy(implicit_state, scale=1.0)
    forced_residual = {
        n: _weighted_norm(getattr(sourced, n) - getattr(qdot, n), host)
        for n in EVOLVED}
    source_increment = {
        n: _weighted_norm(getattr(sourced, n) - getattr(spatial, n) - getattr(source, n), host)
        for n in EVOLVED}
    phi_compare = _weighted_norm(reconstructed - state.phi, host)
    reconstructed_implicit_state = _implicit_material_state(
        reconstructed_implicit_material, state
    )
    reconstructed_spatial = reconstructed_spatial.axpy(
        reconstructed_implicit_state, scale=1.0
    )
    reconstructed_phi_residual = {
        field: _weighted_norm(
            getattr(reconstructed_spatial, field) - getattr(continuum, field),
            host,
        )
        for field in EVOLVED
    }
    phi_reconstruction_rhs_difference = {
        field: _weighted_norm(
            getattr(reconstructed_spatial, field) - getattr(spatial, field),
            host,
        )
        for field in EVOLVED
    }
    masks = _region_masks(geometry, host, selected_wall)
    spatial_error = spatial.replace(**{
        field: getattr(spatial, field) - getattr(continuum, field)
        for field in EVOLVED
    })
    forced_error = sourced.replace(**{
        field: getattr(sourced, field) - getattr(qdot, field)
        for field in EVOLVED
    })
    partitioned_spatial = _partitioned_state_norms(spatial_error, host, masks)
    partitioned_forced = _partitioned_state_norms(forced_error, host, masks)
    ledger_array = np.asarray(ledger)
    continuum_term_ledger = _continuum_term_ledger(
        raw_continuum_terms, host
    )
    term_error_array = ledger_array - continuum_term_ledger
    rhs_term_error_norms = np.asarray([
        [
            _weighted_norm(term_error_array[field_index, slot], host)
            for slot in range(term_error_array.shape[1])
        ]
        for field_index in range(term_error_array.shape[0])
    ])
    partitioned_terms = {
        region: np.asarray([
            [
                _masked_weighted_norm(ledger_array[field_index, slot], host, mask)
                for slot in range(ledger_array.shape[1])
            ]
            for field_index in range(ledger_array.shape[0])
        ])
        for region, mask in masks.items()
    }
    partitioned_term_errors = {
        region: np.asarray([
            [
                _masked_weighted_norm(
                    term_error_array[field_index, slot], host, mask
                )
                for slot in range(term_error_array.shape[1])
            ]
            for field_index in range(term_error_array.shape[0])
        ])
        for region, mask in masks.items()
    }
    materialized = blob._materialize_owner_state(state, host)
    raw = point_state
    fine_volume = np.asarray(host.raw_volume)
    representation = {
        n: float(np.sqrt(np.sum(fine_volume * (np.asarray(getattr(materialized, n)) - np.asarray(getattr(raw, n)))**2) / np.sum(fine_volume)))
        for n in EVOLVED}
    ledger_stats = (float(np.max(np.abs(ledger_array))),
                    float(np.sqrt(np.mean(ledger_array**2))))
    return dict(exact_phi_residual=exact_phi_residual, forced_residual=forced_residual,
                source_increment=source_increment,
                phi_reconstruction_difference=phi_compare,
                reconstructed_phi_residual=reconstructed_phi_residual,
                phi_reconstruction_rhs_difference=(
                    phi_reconstruction_rhs_difference
                ),
                phi_converged=phi_converged,
                phi_failed=phi_failed,
                term_ledger_stats=ledger_stats, representation_error=representation,
                region_cell_counts={name: int(np.count_nonzero(mask))
                                    for name, mask in masks.items()},
                partitioned_exact_phi_residual=partitioned_spatial,
                partitioned_forced_residual=partitioned_forced,
                partitioned_rhs_term_norms=partitioned_terms,
                rhs_term_error_norms=rhs_term_error_norms,
                partitioned_rhs_term_error_norms=partitioned_term_errors,
                _region_masks=masks,
                _model=model, _state=state, _host=host, _geometry=geometry,
                _projector=projector, _runtime=runtime)


def run(args):
    selector_args = _production_selector_args()
    blob._validate_flux_framework(selector_args)
    blob._configure_runtime_selectors(selector_args)
    resolutions = tuple(int(value) for value in args.resolutions)
    shard_counts, device_count = _validate_shard_configuration(
        getattr(args, "shard_counts", (1, 1, 1)),
        resolutions,
        check_device_count=not bool(
            getattr(args, "self_test", False) or getattr(args, "wiring_only", False)
        ),
    )
    campaign_num_steps, campaign_timestep = _step_schedule(
        args.time, args.final_time, args.dt
    )
    if args.self_test:
        print(f"[mms-self-test] {analytic_self_test()}")
    if args.wiring_only:
        print(
            "[mms-wiring] production-split/fci/support-core/"
            "characteristic-sat/energy-absorbing/all-physical-walls/"
            "local-backward-euler/imex-ssp222/"
            "material-scalar-third-order-upwind"
        )
        return
    # The reference is kept in a small independent module so no production
    # RHS object can accidentally leak into the manufactured forcing.
    if not (ROOT / "hsx_mms_continuum_reference.py").is_file():
        raise RuntimeError("the bundled HSX continuum MMS reference is missing")
    from hsx_mms_continuum_reference import ContinuumMmsReference

    finest = blob.build_hsx_fci_geometry(
        makegrid_path=args.makegrid, vessel_path=args.vessel,
        makegrid_currents=blob.DEFAULT_HSX_QHS_MAKEGRID_CURRENTS,
        resolution=(64,64,64), fit_sample_shape=(64,64,64), radial_degree=3,
        vertical_degree=3, toroidal_modes=10, metric_spline_degree=1,
        mmpde_iterations=0, axis_core_radius=0.03,
        reference_magnetic_field=None,
        topology="toroidal", metric_mesh_shape=(64,64,64),
        metric_radial_degree=17, metric_poloidal_modes=15,
        # This first object supplies only the fixed continuous representation.
        # Resolution-local production geometries below always trace real maps.
        metric_toroidal_modes=3, construct_fci_maps=False, fci_trace_substeps=4,
        metric_cache_dir=args.metric_cache_dir,
        rebuild_metric_cache=bool(args.rebuild_metric_cache),
        return_metric_evaluator=True)
    finest_geometry, _, nfp, _, evaluator = finest
    bfield = blob.bfield_evaluator_from_makegrid(
        args.makegrid, currents=blob.DEFAULT_HSX_QHS_MAKEGRID_CURRENTS,
        method="cubic")
    center_points = np.stack(
        np.meshgrid(
            np.asarray(finest_geometry.grid.x.centers),
            np.asarray(finest_geometry.grid.y.centers),
            np.asarray(finest_geometry.grid.z.centers),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, 3))
    B0 = float(np.median(np.asarray(
        evaluator.evaluate_magnetic_field(center_points, bfield).magnitude,
        dtype=np.float64,
    )))
    if not np.isfinite(B0) or B0 <= 0.0:
        raise ValueError(f"derived 64-grid reference B0 is invalid: {B0!r}")
    args.metric_context = blob.HSXMetricContext(evaluator, bfield, int(nfp))
    args.reference = ContinuumMmsReference(
        evaluator, bfield, B0,
        tau=PHYSICAL_PARAMETERS["tau"],
        mi_over_me=PHYSICAL_PARAMETERS["mi_over_me"],
        rho_star=PHYSICAL_PARAMETERS["rho_star"],
        Ve_nu=PHYSICAL_PARAMETERS["Ve_nu"],
        perp_diffusion=PHYSICAL_PARAMETERS["density_D_perp"],
        enable_generalized_potential=True,
    )
    rows = []
    for n in resolutions:
        if n % int(nfp):
            raise ValueError(f"resolution {n} must be divisible by HSX nfp={nfp}")
        built = blob.build_hsx_fci_geometry(
            makegrid_path=args.makegrid, vessel_path=args.vessel,
            makegrid_currents=blob.DEFAULT_HSX_QHS_MAKEGRID_CURRENTS,
            resolution=(n,n,n), fit_sample_shape=(64,64,64), radial_degree=3,
            vertical_degree=3, toroidal_modes=10, metric_spline_degree=1,
            mmpde_iterations=0, axis_core_radius=0.03,
            reference_magnetic_field=B0,
            topology="toroidal", metric_mesh_shape=(64,64,64),
            metric_radial_degree=17, metric_poloidal_modes=15, metric_toroidal_modes=3,
            metric_context=args.metric_context, construct_fci_maps=True,
            fci_trace_substeps=4,
            metric_cache_dir=None)
        geometry, cell_positions = built[0], built[1]
        result = _audit_one(geometry, cell_positions, nfp, args)
        # Preserve the compact execution label before private runtime objects
        # are deliberately dropped between resolutions.
        result["frozen_execution"] = result["_runtime"].frozen_execution
        if args.final_time > args.time:
            reference_cache = {}
            host = result["_host"]
            projector = result["_projector"]
            def reference_at(stage_time):
                key = float(stage_time)
                if key not in reference_cache:
                    point = _reference_state(
                        args.reference, geometry, host, key, projector
                    )
                    reference_cache[key] = (
                        _owner_project(point[0], host),
                        _owner_project(point[1], host),
                        _owner_project(point[2], host),
                        _owner_project(point[3], host),
                    )
                return reference_cache[key]

            def stage_source(stage_time):
                return reference_at(stage_time)[2]
            num_steps = campaign_num_steps
            timestep = campaign_timestep
            runtime = result["_runtime"]
            history_path = args.output.with_name(
                f"{args.output.stem}.N{n}.history.npz"
            )
            if args.reuse_history:
                if not history_path.is_file():
                    raise FileNotFoundError(
                        f"--reuse-history requested but {history_path} is missing"
                    )
                with np.load(history_path, allow_pickle=False) as history:
                    exact_initial = blob._materialize_owner_state(
                        result["_state"], host
                    )
                    _validate_reusable_history(
                        history,
                        history_path,
                        expected_configuration=_production_configuration(
                            shard_counts, device_count
                        ),
                        expected_initial_state=exact_initial,
                        start_time=args.time,
                        timestep=timestep,
                        num_steps=num_steps,
                        save_every=max(1, int(args.save_every)),
                    )
                    final_materialized = blob.FciDrbEBState(**{
                        field: np.asarray(history[field][-1], dtype=np.float64)
                        for field in FIELDS
                    })
                advanced = _owner_project(final_materialized, host)
                print(f"[mms] reusing completed production history {history_path}")
            else:
                advanced = blob.run_full_eb(
                    result["_state"],
                    global_geometry=geometry,
                    cell_positions=cell_positions,
                    nfp=int(nfp),
                    sharded_geometry=runtime.sharded_geometry,
                    mesh=runtime.mesh,
                    parameters=runtime.parameters,
                    metric_cache_path=None,
                    gmres_target_tolerance=PRODUCTION_GMRES[
                        "target_tolerance"
                    ],
                    gmres_acceptance_tolerance=PRODUCTION_GMRES[
                        "acceptance_tolerance"
                    ],
                    gmres_max_iterations=PRODUCTION_GMRES[
                        "max_iterations"
                    ],
                    gmres_restart=PRODUCTION_GMRES["restart"],
                    gmres_preconditioner=PRODUCTION_GMRES[
                        "preconditioner"
                    ],
                    gmres_residual_correction_steps=PRODUCTION_GMRES[
                        "residual_correction_steps"
                    ],
                    time_integrator="imex-ssp222",
                    advance_execution=str(args.advance_execution),
                    num_steps=num_steps,
                    timestep=timestep,
                    start_time=float(args.time),
                    output_path=history_path,
                    save_every=max(1, int(args.save_every)),
                    phase_timing=False,
                    # A classical MMS trajectory starts from the exact
                    # manufactured state. The separate frozen reconstruction
                    # lane measures omega-to-phi spatial error without
                    # contaminating the evolved error at t=0.
                    reconstruct_initial_phi=False,
                    neumann_ghost_scheme="physical",
                    parallel_velocity_wall_bc="neumann",
                    parallel_operator_scheme="fci",
                    poisson_bracket_scheme="material-scalar-third-order-upwind",
                    parallel_material_scheme="production-path",
                    control_volume_descriptor=runtime.control_volume_descriptor,
                    control_volume_fields_host=runtime.control_volume_fields,
                    control_volume_boundary_bc=runtime.control_volume_boundary_bc,
                    control_volume_assembler=runtime.control_volume_assembler,
                    owner_host_geometry=host,
                    source_evaluator=stage_source,
                    history_dtype="float64",
                    run_metadata={
                        **_production_configuration(
                            shard_counts, device_count
                        ),
                        "diagnostic": "hsx-rlp-stage7-mms",
                        "metric_reference_resolution": [64, 64, 64],
                        "reference_magnetic_field": B0,
                        "fci_trace_substeps": 4,
                        "forcing_partition": "explicit-imex-stage-source",
                        "history_dtype": "float64",
                        "shard_counts": list(shard_counts),
                        "device_count": int(device_count),
                        "frozen_execution": runtime.frozen_execution,
                        "evolved_execution": "eta-sharded",
                        "metric_cache_dir": str(args.metric_cache_dir),
                        "rebuild_metric_cache": bool(args.rebuild_metric_cache),
                    },
                )
            target, _, _, _ = _reference_state(args.reference, geometry, host, args.final_time, projector)
            target = _owner_project(target, host)
            result["integration_error_by_field"] = {
                field: _weighted_norm(
                    getattr(advanced, field) - getattr(target, field), host
                )
                for field in EVOLVED
            }
            result["integration_error"] = float(np.mean(list(
                result["integration_error_by_field"].values()
            )))
            result["history_path"] = str(history_path)
            short_leg_history = _short_leg_mode_history(
                history_path,
                reference_at,
                host,
                result["_region_masks"],
            )
            short_leg_path = history_path.with_name(
                f"{history_path.stem}.short_leg_modes.npz"
            )
            np.savez_compressed(
                short_leg_path,
                times=short_leg_history["times"],
                high_mode_start=np.asarray(short_leg_history["high_mode_start"]),
                high_mode_fraction=short_leg_history["high_mode_fraction"],
                high_mode_rms=short_leg_history["high_mode_rms"],
                maximum_poloidal_jump=short_leg_history["maximum_poloidal_jump"],
                late_log_growth_rate=short_leg_history["late_log_growth_rate"],
                late_growth_factor=short_leg_history["late_growth_factor"],
                late_growth_r_squared=short_leg_history[
                    "late_growth_r_squared"
                ],
                classification_json=np.asarray(json.dumps(
                    short_leg_history["classification"]
                )),
                field_names_json=np.asarray(json.dumps(EVOLVED)),
            )
            result["short_leg_diagnostics_path"] = str(short_leg_path)
            result["short_leg_late_growth_rate"] = short_leg_history[
                "late_log_growth_rate"
            ]
            result["short_leg_late_growth_factor"] = short_leg_history[
                "late_growth_factor"
            ]
            result["short_leg_late_growth_r_squared"] = short_leg_history[
                "late_growth_r_squared"
            ]
            result["short_leg_classification"] = short_leg_history[
                "classification"
            ]
            print(
                f"N={n} source-aware IMEX-SSP222 final-time "
                f"error={result['integration_error']:.6e}"
            )
        # Keep only compact numerical products between resolutions. The local
        # model, geometries, and quadrature cache can otherwise retain several
        # full resolution copies during a 32/48/64 campaign.
        rows.append({
            key: value for key, value in result.items()
            if not key.startswith("_")
        })
        print(f"N={n} exact-phi RMS={np.mean(list(result['exact_phi_residual'].values())):.6e} "
              f"forced RMS={np.mean(list(result['forced_residual'].values())):.6e} "
              f"source-pairing={max(result['source_increment'].values()):.3e} "
              f"phi-converged={result['phi_converged']} representation={max(result['representation_error'].values()):.3e}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        exact = np.asarray([[r['exact_phi_residual'][n] for n in EVOLVED] for r in rows])
        forced = np.asarray([[r['forced_residual'][n] for n in EVOLVED] for r in rows])
        sourced = np.asarray([[r['source_increment'][n] for n in EVOLVED] for r in rows])
        repr_err = np.asarray([[r['representation_error'][n] for n in EVOLVED] for r in rows])
        reconstructed_phi_residual = np.asarray([
            [r["reconstructed_phi_residual"][field] for field in EVOLVED]
            for r in rows
        ])
        phi_reconstruction_rhs_difference = np.asarray([
            [
                r["phi_reconstruction_rhs_difference"][field]
                for field in EVOLVED
            ]
            for r in rows
        ])
        partitioned_exact = np.asarray([
            [
                [r["partitioned_exact_phi_residual"][region][field]
                 for field in EVOLVED]
                for region in REGIONS
            ]
            for r in rows
        ])
        partitioned_forced = np.asarray([
            [
                [r["partitioned_forced_residual"][region][field]
                 for field in EVOLVED]
                for region in REGIONS
            ]
            for r in rows
        ])
        partitioned_terms = np.asarray([
            [r["partitioned_rhs_term_norms"][region] for region in REGIONS]
            for r in rows
        ])
        rhs_term_errors = np.asarray([
            r["rhs_term_error_norms"] for r in rows
        ])
        partitioned_term_errors = np.asarray([
            [
                r["partitioned_rhs_term_error_norms"][region]
                for region in REGIONS
            ]
            for r in rows
        ])
        region_cell_counts = np.asarray([
            [r["region_cell_counts"][region] for region in REGIONS]
            for r in rows
        ], dtype=np.int64)
        integration_by_field = np.asarray([
            [r.get("integration_error_by_field", {}).get(field, np.nan)
             for field in EVOLVED]
            for r in rows
        ])
        # Store compact, resolution-independent ledger statistics; raw fields
        # have resolution-dependent spatial extents and are intentionally not
        # serialized here.
        ledger_stats = np.asarray([r['term_ledger_stats'] for r in rows])
        def order(values):
            ratios = np.asarray(resolutions[1:], dtype=np.float64) / np.asarray(resolutions[:-1], dtype=np.float64)
            denominator_shape = (ratios.size,) + (1,) * (values.ndim - 1)
            return np.log(np.maximum(values[:-1], 1e-300) / np.maximum(values[1:], 1e-300)) / np.log(ratios).reshape(denominator_shape)
        np.savez(args.output, resolutions=np.asarray(resolutions),
                 metric_reference_resolution=np.asarray((64,64,64)),
                 reference_magnetic_field=np.asarray(B0),
                 nfp=np.asarray(int(nfp), dtype=np.int32),
                 shard_counts=np.asarray(shard_counts, dtype=np.int32),
                 device_count=np.asarray(device_count, dtype=np.int32),
                 frozen_execution=np.asarray(rows[0]["frozen_execution"]),
                 evolved_execution=np.asarray("eta-sharded"),
                 metric_cache_dir=np.asarray(str(args.metric_cache_dir)),
                 rebuild_metric_cache=np.asarray(
                     bool(args.rebuild_metric_cache), dtype=bool
                 ),
                 fci_trace_substeps=np.asarray(4, dtype=np.int32),
                 generalized_potential_enabled=np.asarray(True, dtype=bool),
                 reference_derivative_method=np.asarray(
                     "structured-nonuniform-five-point-finite-difference"
                 ),
                 reference_derivative_order=np.asarray(4, dtype=np.int32),
                 reference_periodic_axes_json=np.asarray(
                     json.dumps({"theta": "geometry-domain", "eta": "geometry-domain"})
                 ),
                 exact_phi_residual=exact, exact_phi_observed_order=order(exact),
                 forced_residual=forced, forced_observed_order=order(forced),
                 source_increment=sourced, source_observed_order=order(sourced),
                 representation_error=repr_err, representation_observed_order=order(repr_err),
                 phi_reconstruction_difference=np.asarray([r['phi_reconstruction_difference'] for r in rows]),
                 reconstructed_phi_residual=reconstructed_phi_residual,
                 reconstructed_phi_residual_observed_order=order(
                     reconstructed_phi_residual
                 ),
                 phi_reconstruction_rhs_difference=(
                     phi_reconstruction_rhs_difference
                 ),
                 phi_reconstruction_rhs_difference_observed_order=order(
                     phi_reconstruction_rhs_difference
                 ),
                 phi_converged=np.asarray([
                     r["phi_converged"] for r in rows
                 ], dtype=bool),
                 phi_failed=np.asarray([
                     r["phi_failed"] for r in rows
                 ], dtype=bool),
                 rhs_term_ledger_stats=ledger_stats,
                 integration_error=np.asarray([r.get("integration_error", np.nan) for r in rows]),
                 integration_error_by_field=integration_by_field,
                 continuum_total_error=np.asarray([
                     r.get("integration_error", np.nan) for r in rows
                 ]),
                 continuum_total_error_by_field=integration_by_field,
                 integration_error_definition=np.asarray(
                     "owner-volume-weighted-total-error-versus-continuum;"
                     "includes-spatial-reconstruction-and-time-integration-error"
                 ),
                 partitioned_exact_phi_residual=partitioned_exact,
                 partitioned_forced_residual=partitioned_forced,
                 partitioned_rhs_term_norms=partitioned_terms,
                 rhs_term_error_norms=rhs_term_errors,
                 rhs_term_error_observed_order=order(rhs_term_errors),
                 partitioned_rhs_term_error_norms=partitioned_term_errors,
                 partitioned_rhs_term_error_observed_order=order(
                     partitioned_term_errors
                 ),
                 region_cell_counts=region_cell_counts,
                 physical_parameters_json=np.asarray(json.dumps(
                     PHYSICAL_PARAMETERS, sort_keys=True
                 )),
                 field_names_json=np.asarray(json.dumps(EVOLVED)),
                 region_names_json=np.asarray(json.dumps(REGIONS)),
                 rhs_term_names_json=np.asarray(json.dumps(blob.RHS_TERM_NAMES)),
                 production_configuration_json=np.asarray(json.dumps({
                     **_production_configuration(shard_counts, device_count),
                     "shard_counts": list(shard_counts),
                     "device_count": int(device_count),
                     "frozen_execution": rows[0]["frozen_execution"],
                     "evolved_execution": "eta-sharded",
                     "metric_cache_dir": str(args.metric_cache_dir),
                     "rebuild_metric_cache": bool(args.rebuild_metric_cache),
                     "generalized_potential": "static-axis-regular-psi",
                     "reference_derivative_method": (
                         "structured-nonuniform-five-point-finite-difference"
                     ),
                     "reference_derivative_order": 4,
                 }, sort_keys=True)),
                 command_json=np.asarray(json.dumps([
                     str(Path(sys.executable).resolve()),
                     str(Path(__file__).resolve()),
                     *sys.argv[1:],
                 ])),
                 start_time=np.asarray(float(args.time)),
                 final_time=np.asarray(float(args.final_time)),
                 requested_timestep=np.asarray(float(args.dt)),
                 actual_timestep=np.asarray(campaign_timestep),
                 num_steps=np.asarray(
                     campaign_num_steps,
                     dtype=np.int64,
                 ),
                 advance_execution=np.asarray(str(args.advance_execution)),
                 history_paths_json=np.asarray(json.dumps([
                     r.get("history_path") for r in rows
                 ])),
                 short_leg_diagnostics_paths_json=np.asarray(json.dumps([
                     r.get("short_leg_diagnostics_path") for r in rows
                 ])),
                 short_leg_late_growth_rate=np.asarray([
                     r.get("short_leg_late_growth_rate",
                           np.full(len(EVOLVED), np.nan))
                     for r in rows
                 ]),
                 short_leg_late_growth_factor=np.asarray([
                     r.get("short_leg_late_growth_factor",
                           np.full(len(EVOLVED), np.nan))
                     for r in rows
                 ]),
                 short_leg_late_growth_r_squared=np.asarray([
                     r.get("short_leg_late_growth_r_squared",
                           np.full(len(EVOLVED), np.nan))
                     for r in rows
                 ]),
                 short_leg_classification_json=np.asarray(json.dumps([
                     r.get("short_leg_classification") for r in rows
                 ])))


def main(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--makegrid", type=Path, default=blob.DEFAULT_MAKEGRID)
    p.add_argument("--vessel", type=Path, default=blob.DEFAULT_VESSEL)
    p.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=blob.DEFAULT_METRIC_CACHE_DIR,
        help=(
            "Campaign-local cache for the fixed 64-grid metric evaluator; "
            "resolution-local geometries reuse the in-memory evaluator."
        ),
    )
    p.add_argument(
        "--rebuild-metric-cache",
        action="store_true",
        help="Rebuild the fixed 64-grid metric cache before the MMS campaign.",
    )
    p.add_argument("--time", type=float, default=0.0)
    p.add_argument("--final-time", type=float, default=0.01)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument(
        "--advance-execution",
        choices=("compiled", "eager"),
        default="compiled",
        help="Outer production advance mode; compiled matches production.",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save every N accepted IMEX steps in each per-resolution history.",
    )
    p.add_argument(
        "--reuse-history",
        action="store_true",
        help="Post-process an already completed matching production history.",
    )
    p.add_argument("--resolutions", default="32,48,64",
                   help="comma-separated even toroidal resolutions (default: 32,48,64)")
    p.add_argument(
        "--shard-counts",
        nargs=3,
        type=int,
        metavar=("SU", "SV", "SETA"),
        default=(1, 1, 1),
        help=(
            "Eta-only production decomposition (default: 1 1 1). "
            "The remote four-GPU setup uses --shard-counts 1 1 4."
        ),
    )
    p.add_argument("--output", type=Path, default=ROOT / "hsx_mms_residuals.npz")
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run the independent analytic continuum-reference self-test.",
    )
    p.add_argument(
        "--wiring-only",
        action="store_true",
        help="Validate and print the canonical production selector contract without geometry.",
    )
    args = p.parse_args(argv)
    if not 0.0 <= args.time < 0.15 or not 0.0 < args.final_time < 0.15:
        p.error("--time and --final-time must lie in [0,0.15)")
    if args.final_time < args.time:
        p.error("--final-time must be greater than or equal to --time")
    if args.dt <= 0.0:
        p.error("--dt must be positive")
    if args.save_every < 1:
        p.error("--save-every must be positive")
    try:
        args.resolutions = tuple(int(v) for v in args.resolutions.split(",") if v.strip())
    except ValueError:
        p.error("--resolutions must be comma-separated integers")
    if not args.resolutions or any(v < 4 or v % 2 for v in args.resolutions):
        p.error("--resolutions entries must be positive even integers")
    try:
        args.shard_counts, _ = _validate_shard_configuration(
            args.shard_counts,
            args.resolutions,
            check_device_count=not bool(args.self_test or args.wiring_only),
        )
    except ValueError as error:
        p.error(str(error))
    run(args)


if __name__ == "__main__":
    main()
