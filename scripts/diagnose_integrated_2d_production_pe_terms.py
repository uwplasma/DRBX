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
    _assemble_electron_pressure_rhs_terms,
    _build_recycling_runtime_model,
    _electron_density,
    _electron_zero_current_velocity,
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_SNAPSHOT_DIR = _REPO_ROOT / "references" / "baselines" / "reference_snapshots"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose integrated 2D production target-band electron-pressure rhs terms."
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--case", default="integrated_2d_production_one_step", choices=("integrated_2d_production_one_step",))
    parser.add_argument(
        "--use-committed-baselines",
        action="store_true",
        help="Use the committed final diagnostic snapshot instead of rerunning the reference code.",
    )
    args = parser.parse_args()

    config = load_bout_input(args.reference_root / "tests" / "integrated" / "2D-production" / "data" / "BOUT.inp")
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)

    initial_snapshot = load_local_reference_snapshot_cache(
        _integrated_2d_snapshot_cache_path("integrated_2d_production_rhs"),
        field_names=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"),
        optional_field_names=("SNd+", "SPd+", "SNVd+", "SNd", "SPd", "SNVd", "Vd+", "Vd"),
        scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
    )

    if args.use_committed_baselines:
        final_cache_path = _REFERENCE_SNAPSHOT_DIR / f"{args.case}_final_snapshot.npz"
        if not final_cache_path.exists():
            raise FileNotFoundError(f"Missing committed final diagnostic snapshot: {final_cache_path}")
        reference_final = load_local_reference_snapshot_cache(
            final_cache_path,
            field_names=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "Ve", "ddt(Pe)"),
            optional_field_names=("Vd+", "Vd", "SNd+", "SPd+", "SNVd+", "SNd", "SPd", "SNVd"),
            scalar_names=(),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="jaxdrb-prod-pe-terms-") as workdir:
            execution = run_reference_case(
                args.case,
                reference_root=args.reference_root,
                workdir=workdir,
                keep_workdir=True,
            )
            dump_path = Path(execution.summary.artifacts["BOUT.dmp.0.nc"])
            reference_final = load_local_reference_snapshot(
                dump_path,
                field_names=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "Ve", "ddt(Pe)"),
                optional_field_names=("Vd+", "Vd", "SNd+", "SPd+", "SNVd+", "SNd", "SPd", "SNVd"),
                scalar_names=(),
                time_index=1,
            )

    field_overrides = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in reference_final.fields.items()
        if name in {"Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"}
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
    electron = species["e"]
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
        neutrals=tuple(sp for sp in species.values() if sp.charge == 0.0),
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

    electron_velocity = _electron_zero_current_velocity(
        ions,
        prepared=prepared,
        ion_velocity=ion_boundary.velocity,
        electron_density=prepared["e"].density,
    )
    electron_fastest_wave = np.sqrt(np.maximum(prepared["e"].temperature, 0.0) / electron.atomic_mass)
    terms = _assemble_electron_pressure_rhs_terms(
        explicit_pressure_source=np.asarray(runtime_model.explicit_pressure_sources["e"], dtype=np.float64),
        electron_pressure=electron_boundary.pressure,
        electron_velocity=electron_velocity,
        electron_fastest_wave=electron_fastest_wave,
        electron_energy_source=energy_source["e"],
        mesh=reference_final.mesh,
        metrics=reference_final.metrics,
    )

    for i in (14, 15):
        j = reference_final.mesh.ystart
        k = 0
        print(
            f"cell={(i, j, k)} "
            f"explicit={terms.explicit_pressure_source[i, j, k]:.12e} "
            f"div={terms.parallel_divergence[i, j, k]:.12e} "
            f"adv={terms.parallel_advection[i, j, k]:.12e} "
            f"energy={terms.energy_source[i, j, k]:.12e} "
            f"total={terms.total[i, j, k]:.12e} "
            f"ref={reference_final.fields['ddt(Pe)'][i, j, k]:.12e}"
        )

    print("source evolution against cached integrated_2d_production_rhs")
    for name in ("SNd+", "SPd+", "SNVd+", "SNd", "SPd", "SNVd"):
        print(
            name,
            *( 
                f"{cell}:{float(reference_final.optional_fields[name][cell] - initial_snapshot.optional_fields[name][cell]):.12e}"
                for cell in ((14, reference_final.mesh.ystart, 0), (15, reference_final.mesh.ystart, 0))
            ),
        )


if __name__ == "__main__":
    main()
