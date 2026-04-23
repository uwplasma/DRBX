from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import (
    OpenFieldSpecies,
    _initialize_species,
    _prepare_open_field_states,
    compute_recycling_1d_rhs,
)
from jax_drb.native.recycling_targets import (
    electron_zero_current_velocity,
    grad_par_electron_force_balance_open,
    target_recycling_sources,
)
from jax_drb.native.reference_dump import load_local_reference_snapshot
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars


_INPUT_1D = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")


def test_target_recycling_sources_use_prepared_ion_state() -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    neutrals = tuple(sp for sp in species.values() if sp.charge == 0.0 and sp.name != "e")
    ion_velocity = {ion.name: prepared[ion.name].velocity for ion in ions}

    baseline = target_recycling_sources(
        ions=ions,
        prepared=prepared,
        neutrals=neutrals,
        ion_velocity=ion_velocity,
        mesh=mesh,
        metrics=metrics,
        gamma_i=2.5,
    )

    distorted_ions = tuple(
        OpenFieldSpecies(
            **{
                **ion.__dict__,
                "density": ion.density * 3.0,
                "pressure": ion.pressure * 5.0,
            }
        )
        for ion in ions
    )
    distorted = target_recycling_sources(
        ions=distorted_ions,
        prepared=prepared,
        neutrals=neutrals,
        ion_velocity=ion_velocity,
        mesh=mesh,
        metrics=metrics,
        gamma_i=2.5,
    )

    for neutral in ("d",):
        np.testing.assert_allclose(
            distorted.density_source[neutral],
            baseline.density_source[neutral],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            distorted.energy_source[neutral],
            baseline.energy_source[neutral],
            rtol=0.0,
            atol=0.0,
        )


def test_recycling_rhs_passes_configured_sheath_gamma_i_to_target_recycling(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_bout_input(Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-recycling/data/BOUT.inp"))
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    snapshot = load_local_reference_snapshot(
        Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-recycling/data/BOUT.dmp.0.nc"),
        field_names=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"),
        scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
    )

    captured: list[tuple[float, bool]] = []
    original = target_recycling_sources.__globals__["compute_target_recycling_sources"]

    def wrapper(*args, **kwargs):
        captured.append(
            (
                float(kwargs["gamma_i"]),
                kwargs["lower_geometry"] is not None or kwargs["upper_geometry"] is not None,
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setitem(target_recycling_sources.__globals__, "compute_target_recycling_sources", wrapper)

    compute_recycling_1d_rhs(
        config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=dataset_scalars,
        field_overrides=snapshot.fields,
        preserve_dump_target_state=True,
    )

    assert captured
    assert all(value == pytest.approx(2.5) for value, _ in captured)
    assert all(has_cached_geometry for _, has_cached_geometry in captured)


def test_electron_zero_current_velocity_uses_prepared_ion_density() -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    ion_velocity = {ion.name: prepared[ion.name].velocity for ion in ions}
    electron_density = prepared["e"].density

    baseline = electron_zero_current_velocity(
        ions,
        prepared=prepared,
        ion_velocity=ion_velocity,
        electron_density=electron_density,
    )

    distorted_ions = tuple(
        OpenFieldSpecies(
            **{
                **ion.__dict__,
                "density": ion.density * 4.0,
            }
        )
        for ion in ions
    )
    distorted = electron_zero_current_velocity(
        distorted_ions,
        prepared=prepared,
        ion_velocity=ion_velocity,
        electron_density=electron_density,
    )

    np.testing.assert_allclose(distorted, baseline, rtol=0.0, atol=0.0)


def test_electron_force_balance_gradient_matches_bout_dy_over_sqrt_g22_stencil() -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)

    field = np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    for j in range(mesh.local_ny):
        field[:, j, :] = float(j)

    gradient = grad_par_electron_force_balance_open(
        field,
        mesh=mesh,
        metrics=metrics,
    )

    dy = np.asarray(metrics.dy, dtype=np.float64)
    g_22 = np.asarray(metrics.g_22, dtype=np.float64)
    expected = np.zeros_like(field, dtype=np.float64)
    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                expected[i, j, k] = 0.5 * (field[i, j + 1, k] - field[i, j - 1, k]) / (
                    dy[i, j, k] * np.sqrt(g_22[i, j, k])
                )

    active = (slice(mesh.xstart, mesh.xend + 1), slice(mesh.ystart, mesh.yend + 1), slice(None))
    np.testing.assert_allclose(gradient[active], expected[active], rtol=1.0e-12, atol=1.0e-12)
