"""Tiny forced-CPU-device contract for the frozen EB diagnostic hook."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import simulate_hsx_blob as blob  # noqa: E402


class _FakeModel:
    """Traceable local model exposing the production diagnostic interfaces."""

    @staticmethod
    def evaluate_stage(
        state,
        source_owned=None,
        *,
        phi_owned=None,
        return_rhs_term_fields=False,
        **_kwargs,
    ):
        assert return_rhs_term_fields
        if source_owned is None:
            source_owned = state.zeros_like()
        phi = state.phi if phi_owned is None else phi_owned

        def rhs(name):
            if name == "phi":
                return jnp.zeros_like(phi)
            return (
                2.0 * getattr(state, name)
                + 0.1 * phi
                + getattr(source_owned, name)
            )

        result = blob.FciDrbEBState(**{
            name: rhs(name) for name in state.field_names()
        })
        evolved = tuple(getattr(state, name) for name in blob.RHS_TERM_FIELD_NAMES)
        terms = jnp.zeros(
            (len(evolved), max(map(len, blob.RHS_TERM_NAMES))) + phi.shape,
            dtype=jnp.float64,
        )
        for field_index, (name, value) in enumerate(
            zip(blob.RHS_TERM_FIELD_NAMES, evolved, strict=True)
        ):
            terms = terms.at[field_index, 0].set(2.0 * value + 0.1 * phi)
            source_slot = len(blob.RHS_TERM_NAMES[field_index]) - 1
            terms = terms.at[field_index, source_slot].set(
                getattr(source_owned, name)
            )
        return result, terms

    @staticmethod
    def apply_short_leg_implicit_material_step(state, **_kwargs):
        material = 0.01 * jnp.stack(
            (state.density, state.Te, state.Ti, state.Vi, state.Ve),
            axis=-1,
        )
        return state, state.zeros_like(), {
            "selected_complete_residual_owner": material,
            "selected_wall": jnp.ones(state.density.shape, dtype=bool),
        }

    @staticmethod
    def reconstruct_phi(state, *, return_diagnostics=False):
        assert return_diagnostics
        info = SimpleNamespace(
            num_steps=jnp.asarray(3, dtype=jnp.int32),
            final_residual_rel_l2=jnp.asarray(1.0e-12, dtype=jnp.float64),
            failed=jnp.asarray(False),
            converged=jnp.asarray(True),
        )
        return state.phi + 0.25, info


def run_case() -> dict[str, object]:
    shape = (2, 3, 4)
    shard_counts = (1, 1, 2)
    layout = SimpleNamespace(halo_width=2)
    domain = SimpleNamespace(
        layout=layout,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )
    sharded_geometry = SimpleNamespace(
        shard_counts=shard_counts,
        global_shape=shape,
        domain=domain,
        cell_fields=np.zeros(shape + (1,), dtype=np.float64),
        map_fields=np.zeros(shape + (len(blob.FCI_MAP_FIELDS),), dtype=np.float64),
        maps_valid=True,
    )
    mesh = blob.make_shard_mesh(shard_counts)
    host_bundle = SimpleNamespace(domain=domain)
    local_geometry = SimpleNamespace(layout=layout)

    blob.build_local_fci_geometries = lambda *_args, **_kwargs: host_bundle
    blob.assemble_single_device_local_fci_geometry = lambda _value: local_geometry
    blob.replace = lambda value, **changes: SimpleNamespace(
        **vars(value), **changes
    )

    def curvature_faces(_geometry, _domain):
        nx, ny, nz = shape
        return SimpleNamespace(axes=(
            jnp.ones((nx + 1, ny, nz), dtype=jnp.float64),
            jnp.ones((nx, ny + 1, nz), dtype=jnp.float64),
            jnp.ones((nx, ny, nz + 1), dtype=jnp.float64),
        ))

    blob.build_local_curvature_face_coefficients = curvature_faces
    blob.LocalCurvatureFaceCoefficients3D = lambda **kwargs: SimpleNamespace(
        **kwargs, axes=(kwargs["x"], kwargs["y"], kwargs["z"])
    )
    blob.assemble_local_fci_geometry = (
        lambda *_args, **_kwargs: local_geometry
    )
    blob.build_local_eb_model = lambda *_args, **_kwargs: _FakeModel()

    base = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 100.0
    state = blob.FciDrbEBState(
        density=1.0 + base,
        phi=0.2 + base,
        Te=1.1 + base,
        Ti=1.2 + base,
        Vi=0.1 + base,
        Ve=-0.1 + base,
        vorticity=0.05 + base,
    )
    source = state.map_fields(lambda value: np.full_like(value, 0.03))
    request = blob.FrozenEbDiagnosticRequest(
        source_state=source,
        implicit_solve_dt=1.0e-4,
        implicit_selection_dt=2.0e-4,
        execution="compiled",
    )
    result = blob.run_full_eb(
        state,
        global_geometry=SimpleNamespace(shape=shape),
        cell_positions=np.zeros(shape + (3,), dtype=np.float64),
        nfp=1,
        sharded_geometry=sharded_geometry,
        mesh=mesh,
        parameters=SimpleNamespace(
            parallel_characteristic_wall_law="energy-absorbing"
        ),
        metric_cache_path=None,
        gmres_target_tolerance=1.0e-8,
        gmres_acceptance_tolerance=1.0e-5,
        gmres_max_iterations=4,
        gmres_restart=4,
        gmres_preconditioner="none",
        time_integrator="imex-ssp222",
        advance_execution="compiled",
        num_steps=0,
        timestep=2.0e-4,
        start_time=0.0,
        output_path=Path("unused.npz"),
        save_every=1,
        phase_timing=False,
        reconstruct_initial_phi=False,
        parallel_operator_scheme="fci",
        parallel_material_scheme="production-path",
        frozen_diagnostic=request,
        history_dtype="float64",
    )
    source_pairing = max(
        float(np.max(np.abs(
            np.asarray(getattr(result.sourced_explicit, name))
            - np.asarray(getattr(result.exact_explicit, name))
            - np.asarray(getattr(source, name))
        )))
        for name in blob.RHS_TERM_FIELD_NAMES
    )
    reconstructed_shift = float(np.max(np.abs(
        np.asarray(result.reconstructed_phi) - np.asarray(state.phi) - 0.25
    )))
    return {
        "device_count": len(jax.devices()),
        "source_pairing": source_pairing,
        "reconstructed_shift": reconstructed_shift,
        "exact_term_shape": list(result.exact_rhs_term_fields.shape),
        "sourced_term_shape": list(result.sourced_rhs_term_fields.shape),
        "reconstructed_term_shape": list(
            result.reconstructed_rhs_term_fields.shape
        ),
        "implicit_shape": list(
            result.exact_implicit_complete_residual_owner.shape
        ),
        "selected_all": bool(np.all(np.asarray(result.exact_selected_wall))),
        "reconstructed_selected_all": bool(np.all(
            np.asarray(result.reconstructed_selected_wall)
        )),
        "phi_diagnostics": np.asarray(
            result.phi_solver_diagnostics
        ).tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(run_case(), sort_keys=True))
