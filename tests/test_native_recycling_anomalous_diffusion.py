from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.recycling_anomalous_diffusion import apply_anomalous_diffusion
from jax_drb.native.recycling_setup import initialize_species
from jax_drb.native.reference_dump import (
    load_local_reference_snapshot_cache,
    synthesize_local_reference_snapshot_from_active_history,
)
from jax_drb.native.mesh import StructuredMesh
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.reference.paths import default_reference_root


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_ROOT = default_reference_root()
_REFERENCE_BASE = _REFERENCE_ROOT if _REFERENCE_ROOT is not None else Path("/nonexistent-reference-root")


def test_apply_anomalous_diffusion_uses_nonorthogonal_tokamak_metrics_on_evolved_state() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_REFERENCE_BASE / "examples/tokamak-2D/recycling-dthe/BOUT.inp")
    config = apply_bout_overrides(
        config,
        (
            "timestep=0.1",
            f"mesh:file={_REFERENCE_BASE / 'examples/tokamak-2D/recycling-dthe/tokamak.nc'}",
            "he+:diagnose=false",
            "input:error_on_unused_options=false",
        ),
    )
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    snapshot = load_local_reference_snapshot_cache(
        _REPO_ROOT / "references/baselines/reference_snapshots/tokamak_recycling_dthe_rhs_snapshot.npz",
        field_names=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Nt+", "Pt+", "NVt+", "Nt", "Pt", "NVt", "Nhe+", "Phe+", "NVhe+", "Nhe", "Phe", "NVhe", "Pe"),
        scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
    )
    evolved = synthesize_local_reference_snapshot_from_active_history(
        initial_snapshot=snapshot,
        array_history_path=_REPO_ROOT / "references/baselines/reference_arrays/tokamak_recycling_dthe_one_step.npz",
        optional_history_path=_REPO_ROOT / "references/baselines/reference_snapshots/tokamak_recycling_dthe_one_step_optional_history.npz",
        timestep=0.1,
        state_field_names=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Nt+", "Pt+", "NVt+", "Nt", "Pt", "NVt", "Nhe+", "Phe+", "NVhe+", "Nhe", "Phe", "NVhe", "Pe"),
        optional_field_names=(),
    )
    species = initialize_species(
        config,
        mesh=evolved.mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=evolved.fields,
    )
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(str(_REFERENCE_BASE / "examples/tokamak-2D/recycling-dthe/tokamak.nc")) as mesh_dataset:
        g23 = np.asarray(mesh_dataset.variables["g23"][:], dtype=np.float64)[..., None]
        g_23 = np.asarray(mesh_dataset.variables["g_23"][:], dtype=np.float64)[..., None]
    nonorthogonal_metrics = StructuredMetrics(
        dx=evolved.metrics.dx,
        dy=evolved.metrics.dy,
        dz=evolved.metrics.dz,
        J=evolved.metrics.J,
        g11=evolved.metrics.g11,
        g22=evolved.metrics.g22,
        g33=evolved.metrics.g33,
        g_22=evolved.metrics.g_22,
        g23=g23,
        Bxy=evolved.metrics.Bxy,
        g_23=g_23,
    )
    orthogonal_metrics = StructuredMetrics(
        dx=evolved.metrics.dx,
        dy=evolved.metrics.dy,
        dz=evolved.metrics.dz,
        J=evolved.metrics.J,
        g11=evolved.metrics.g11,
        g22=evolved.metrics.g22,
        g33=evolved.metrics.g33,
        g_22=evolved.metrics.g_22,
        g23=np.zeros_like(g23),
        Bxy=evolved.metrics.Bxy,
        g_23=np.zeros_like(g_23),
    )

    nonorthogonal_terms = apply_anomalous_diffusion(
        config,
        species=species,
        mesh=evolved.mesh,
        metrics=nonorthogonal_metrics,
        dataset_scalars=dataset_scalars,
    )
    orthogonal_terms = apply_anomalous_diffusion(
        config,
        species=species,
        mesh=evolved.mesh,
        metrics=orthogonal_metrics,
        dataset_scalars=dataset_scalars,
    )

    d_momentum_delta = np.asarray(
        nonorthogonal_terms.momentum_source["d+"] - orthogonal_terms.momentum_source["d+"],
        dtype=np.float64,
    )
    d_energy_delta = np.asarray(
        nonorthogonal_terms.energy_source["d+"] - orthogonal_terms.energy_source["d+"],
        dtype=np.float64,
    )

    assert np.isfinite(d_momentum_delta).all()
    assert np.isfinite(d_energy_delta).all()
    assert float(np.nanmax(np.abs(d_momentum_delta))) > 1.0e-10
    assert float(np.nanmax(np.abs(d_energy_delta))) > 1.0e-10


def test_apply_anomalous_diffusion_adds_momentum_source_for_anomalous_nu() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_REFERENCE_BASE / "tests/integrated/2D-production/data/BOUT.inp")
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    fields = {
        "Nd+": jnp.ones((4, 5, 1), dtype=jnp.float64),
        "Pd+": jnp.ones((4, 5, 1), dtype=jnp.float64),
        "NVd+": jnp.linspace(-1.0, 1.0, 20, dtype=jnp.float64).reshape(4, 5, 1),
        "Nd": jnp.zeros((4, 5, 1), dtype=jnp.float64),
        "Pd": jnp.zeros((4, 5, 1), dtype=jnp.float64),
        "NVd": jnp.zeros((4, 5, 1), dtype=jnp.float64),
        "Pe": jnp.ones((4, 5, 1), dtype=jnp.float64),
    }
    species = initialize_species(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=fields,
    )
    terms = apply_anomalous_diffusion(
        config,
        species=species,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
    )
    assert np.max(np.abs(terms.momentum_source["d+"])) > 0.0
    assert np.allclose(terms.density_source["d+"], 0.0)
    assert np.allclose(terms.energy_source["d+"], 0.0)
