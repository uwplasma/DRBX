#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.recycling_1d import (
    _apply_collision_closure,
    _apply_neutral_parallel_diffusion,
    _apply_upstream_density_feedback,
    _build_recycling_runtime_model,
    _div_par_fvv_open,
    _div_par_mod_open,
    _electron_density,
    _grad_par_open,
    _load_simple_sheath_settings,
    _override_species_fields,
    _prepare_open_field_states,
    _reaction_sources,
    _target_recycling_sources,
)
from jax_drb.native.reference_dump import load_local_reference_snapshot, load_local_reference_snapshot_cache
from jax_drb.native.runner import _apply_species_velocity_overrides, _integrated_2d_snapshot_cache_path
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.reference import run_reference_case
from jax_drb.runtime.run_config import RunConfiguration


STATE_FIELDS = ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe")
SOURCE_FIELDS = ("SNd+", "SPd+", "SNVd+", "SNd", "SPd", "SNVd")
DIAGNOSTIC_FIELDS = ("Vd+", "Vd")
SCALAR_FIELDS = ("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_SNAPSHOT_DIR = _REPO_ROOT / "references" / "baselines" / "reference_snapshots"


def production_target_band_cells(mesh, *, x_indices: tuple[int, ...] = (14, 15), z_index: int = 0) -> tuple[tuple[int, int, int], ...]:
    return tuple((int(x_index), int(mesh.ystart), int(z_index)) for x_index in x_indices)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose integrated 2D production target-band ion pressure and momentum rhs terms."
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--case", default="integrated_2d_production_one_step", choices=("integrated_2d_production_one_step",))
    parser.add_argument(
        "--use-committed-baselines",
        action="store_true",
        help="Use the committed final diagnostic snapshot instead of rerunning the reference code.",
    )
    args = parser.parse_args()

    input_path = args.reference_root / "tests" / "integrated" / "2D-production" / "data" / "BOUT.inp"
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)

    initial_snapshot = load_local_reference_snapshot_cache(
        _integrated_2d_snapshot_cache_path("integrated_2d_production_rhs"),
        field_names=STATE_FIELDS,
        optional_field_names=SOURCE_FIELDS + DIAGNOSTIC_FIELDS,
        scalar_names=SCALAR_FIELDS,
    )

    if args.use_committed_baselines:
        final_cache_path = _REFERENCE_SNAPSHOT_DIR / f"{args.case}_final_snapshot.npz"
        if not final_cache_path.exists():
            raise FileNotFoundError(f"Missing committed final diagnostic snapshot: {final_cache_path}")
        reference_final = load_local_reference_snapshot_cache(
            final_cache_path,
            field_names=STATE_FIELDS + ("ddt(Nd+)", "ddt(Pd+)", "ddt(NVd+)"),
            optional_field_names=SOURCE_FIELDS + DIAGNOSTIC_FIELDS,
            scalar_names=(),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="jaxdrb-prod-ion-terms-") as workdir:
            execution = run_reference_case(
                args.case,
                reference_root=args.reference_root,
                workdir=workdir,
                keep_workdir=True,
            )
            dump_path = Path(execution.summary.artifacts["BOUT.dmp.0.nc"])
            reference_final = load_local_reference_snapshot(
                dump_path,
                field_names=STATE_FIELDS + ("ddt(Nd+)", "ddt(Pd+)", "ddt(NVd+)"),
                optional_field_names=SOURCE_FIELDS + DIAGNOSTIC_FIELDS,
                scalar_names=(),
                time_index=1,
            )

    field_overrides = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in reference_final.fields.items()
        if name in STATE_FIELDS
    }
    field_overrides = _apply_species_velocity_overrides(
        config,
        field_overrides=field_overrides,
        velocity_field_overrides={
            "d+": np.asarray(reference_final.optional_fields["Vd+"], dtype=np.float64),
            "d": np.asarray(reference_final.optional_fields["Vd"], dtype=np.float64),
        },
    )

    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=reference_final.mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=field_overrides,
        density_source_overrides={
            "d+": np.asarray(initial_snapshot.optional_fields["SNd+"], dtype=np.float64),
            "d": np.asarray(initial_snapshot.optional_fields["SNd"], dtype=np.float64),
        },
        pressure_source_overrides={
            "d+": np.asarray(initial_snapshot.optional_fields["SPd+"], dtype=np.float64),
            "d": np.asarray(initial_snapshot.optional_fields["SPd"], dtype=np.float64),
        },
        momentum_source_overrides={
            "d+": np.asarray(initial_snapshot.optional_fields["SNVd+"], dtype=np.float64),
            "d": np.asarray(initial_snapshot.optional_fields["SNVd"], dtype=np.float64),
        },
        preserve_dump_target_state=True,
        preserve_dump_ion_target_state_only=True,
    )
    species = _override_species_fields(runtime_model.species_templates, fields=field_overrides, mesh=reference_final.mesh)
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    neutrals = tuple(sp for sp in species.values() if sp.charge == 0.0)
    electron_density = _electron_density(ions)

    density_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    momentum_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}

    reaction_terms = _reaction_sources(
        config,
        species=species,
        electron_density=electron_density,
        dataset_scalars=dataset_scalars,
    )
    for name, value in reaction_terms.density_source.items():
        density_source[name] = density_source[name] + value
    for name, value in reaction_terms.energy_source.items():
        energy_source[name] = energy_source[name] + value
    for name, value in reaction_terms.momentum_source.items():
        momentum_source[name] = momentum_source[name] + value

    prepared, ion_boundary, electron_boundary = _prepare_open_field_states(
        species,
        config=config,
        mesh=reference_final.mesh,
        metrics=reference_final.metrics,
        dataset_scalars=dataset_scalars,
        apply_sheath_boundaries=True,
        preserve_dump_target_state=True,
        preserve_dump_ion_target_state_only=True,
    )
    for name, value in ion_boundary.energy_source.items():
        energy_source[name] = energy_source[name] + value

    collision_terms = _apply_collision_closure(
        config,
        species,
        prepared,
        mesh=reference_final.mesh,
        metrics=reference_final.metrics,
        dataset_scalars=dataset_scalars,
    )
    for name, value in collision_terms.energy_source.items():
        energy_source[name] = energy_source[name] + value
    for name, value in collision_terms.momentum_source.items():
        momentum_source[name] = momentum_source[name] + value

    neutral_diffusion_terms = _apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=reference_final.mesh,
        metrics=reference_final.metrics,
        dataset_scalars=dataset_scalars,
    )
    for name, value in neutral_diffusion_terms.density_source.items():
        density_source[name] = density_source[name] + value
    for name, value in neutral_diffusion_terms.energy_source.items():
        energy_source[name] = energy_source[name] + value
    for name, value in neutral_diffusion_terms.momentum_source.items():
        momentum_source[name] = momentum_source[name] + value

    energy_source["e"] = energy_source["e"] + electron_boundary.energy_source
    simple_sheath_settings = _load_simple_sheath_settings(config, mesh=reference_final.mesh, dataset_scalars=dataset_scalars)
    recycling_terms = _target_recycling_sources(
        ions=ions,
        prepared=prepared,
        neutrals=neutrals,
        ion_velocity=ion_boundary.velocity,
        mesh=reference_final.mesh,
        metrics=reference_final.metrics,
        gamma_i=0.0 if simple_sheath_settings is None else simple_sheath_settings.gamma_i,
    )
    for name, value in recycling_terms.density_source.items():
        density_source[name] = density_source[name] + value
    for name, value in recycling_terms.energy_source.items():
        energy_source[name] = energy_source[name] + value

    feedback_terms = _apply_upstream_density_feedback(
        species,
        prepared,
        controllers=runtime_model.controllers,
        mesh=reference_final.mesh,
        feedback_integrals={name: 0.0 for name in runtime_model.feedback_names},
    )
    for name, value in feedback_terms.density_source.items():
        density_source[name] = density_source[name] + value
    for name, value in feedback_terms.energy_source.items():
        energy_source[name] = energy_source[name] + value

    density_source["d+"] = np.asarray(reference_final.optional_fields["SNd+"], dtype=np.float64)
    energy_source["d+"] = np.asarray(energy_source["d+"], dtype=np.float64)
    momentum_source["d+"] = np.asarray(reference_final.optional_fields["SNVd+"], dtype=np.float64)
    explicit_pressure_source = np.asarray(reference_final.optional_fields["SPd+"], dtype=np.float64)

    ion = species["d+"]
    ion_state = prepared["d+"]
    ion_velocity = ion_boundary.velocity["d+"]
    fastest_wave = np.sqrt(np.maximum(ion_state.temperature, 0.0) / ion.atomic_mass)

    density_divergence = -_div_par_mod_open(
        ion_state.density,
        ion_velocity,
        fastest_wave,
        mesh=reference_final.mesh,
        metrics=reference_final.metrics,
    )
    pressure_divergence = -(5.0 / 3.0) * _div_par_mod_open(
        ion_state.pressure,
        ion_velocity,
        fastest_wave,
        mesh=reference_final.mesh,
        metrics=reference_final.metrics,
    )
    pressure_advection = (2.0 / 3.0) * ion_velocity * _grad_par_open(
        ion_state.pressure,
        mesh=reference_final.mesh,
        metrics=reference_final.metrics,
    )
    pressure_energy = (2.0 / 3.0) * energy_source["d+"]
    momentum_advection = -ion.atomic_mass * _div_par_fvv_open(
        np.maximum(ion_state.density, ion.density_floor),
        ion_velocity,
        fastest_wave,
        mesh=reference_final.mesh,
        metrics=reference_final.metrics,
        fix_flux=False,
    )
    momentum_gradp = -_grad_par_open(
        ion_state.pressure,
        mesh=reference_final.mesh,
        metrics=reference_final.metrics,
    )
    momentum_error = ion_state.momentum_error

    for cell in production_target_band_cells(reference_final.mesh):
        i, j, k = cell
        density_total = density_source["d+"][i, j, k] + density_divergence[i, j, k]
        pressure_total = (
            explicit_pressure_source[i, j, k]
            + pressure_divergence[i, j, k]
            + pressure_advection[i, j, k]
            + pressure_energy[i, j, k]
        )
        momentum_total = (
            momentum_advection[i, j, k]
            + momentum_gradp[i, j, k]
            + momentum_source["d+"][i, j, k]
            + momentum_error[i, j, k]
        )
        print(
            f"cell={cell} "
            f"Nd+: source={density_source['d+'][i, j, k]:.12e} div={density_divergence[i, j, k]:.12e} "
            f"total={density_total:.12e} ref={reference_final.fields['ddt(Nd+)'][i, j, k]:.12e}"
        )
        print(
            f"cell={cell} "
            f"Pd+: explicit={explicit_pressure_source[i, j, k]:.12e} div={pressure_divergence[i, j, k]:.12e} "
            f"adv={pressure_advection[i, j, k]:.12e} energy={pressure_energy[i, j, k]:.12e} "
            f"total={pressure_total:.12e} ref={reference_final.fields['ddt(Pd+)'][i, j, k]:.12e}"
        )
        print(
            f"cell={cell} "
            f"NVd+: adv={momentum_advection[i, j, k]:.12e} gradp={momentum_gradp[i, j, k]:.12e} "
            f"source={momentum_source['d+'][i, j, k]:.12e} err={momentum_error[i, j, k]:.12e} "
            f"total={momentum_total:.12e} ref={reference_final.fields['ddt(NVd+)'][i, j, k]:.12e}"
        )


if __name__ == "__main__":
    main()
