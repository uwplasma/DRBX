from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import StructuredMesh, build_structured_mesh
from jax_drb.native.metrics import StructuredMetrics, build_structured_metrics
from jax_drb.native.recycling_1d import (
    OpenFieldSpecies,
    _build_recycling_runtime_model,
    _build_recycling_state_fields,
    _initialize_species,
    _prepare_open_field_states,
    compute_recycling_1d_rhs,
)
from jax_drb.native.recycling_fixed_residual import (
    RecyclingFixedState,
    build_fixed_array_rhs,
    fixed_state_from_fields,
    fixed_state_to_full_fields,
)
from jax_drb.native.recycling_layout import build_recycling_packed_state_layout
from jax_drb.native.recycling_setup import build_species_field_overrider
from jax_drb.native.recycling_targets import (
    electron_zero_current_velocity,
    fixed_layout_target_recycling_field_rhs,
    grad_par_electron_force_balance_open,
    target_recycling_sources,
)
from jax_drb.native.reference_dump import load_local_reference_snapshot
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.reference.paths import default_reference_root


_REFERENCE_ROOT = default_reference_root()
_REFERENCE_BASE = _REFERENCE_ROOT if _REFERENCE_ROOT is not None else Path("/nonexistent-reference-root")
_INPUT_1D = _REFERENCE_BASE / "tests/integrated/1D-recycling/data/BOUT.inp"
_INPUT_1D_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "reference-root"
    / "tests"
    / "integrated"
    / "1D-recycling"
    / "data"
    / "BOUT.inp"
)


def _input_1d_path() -> Path:
    if _INPUT_1D.exists():
        return _INPUT_1D
    if _INPUT_1D_FIXTURE.exists():
        return _INPUT_1D_FIXTURE
    raise AssertionError("1D recycling fixture deck is missing")


def _build_runtime_target_case():
    config = load_bout_input(_input_1d_path())
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    fields = _build_recycling_state_fields(runtime_model)
    override_species = build_species_field_overrider(
        runtime_model.species_templates,
        mesh=mesh,
    )
    species = override_species(fields)
    prepared, ion_boundary, _electron_boundary = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    neutrals = tuple(
        sp for sp in species.values() if sp.charge == 0.0 and sp.name != "e"
    )
    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=runtime_model.field_names,
        feedback_names=runtime_model.feedback_names,
        mesh=mesh,
    )
    state = fixed_state_from_fields(
        fields,
        feedback_integrals={name: 0.0 for name in runtime_model.feedback_names},
        layout=layout,
    )
    return (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        fields,
        species,
        prepared,
        ion_boundary,
        ions,
        neutrals,
        layout,
        state,
    )


def test_target_recycling_sources_use_prepared_ion_state() -> None:
    config = load_bout_input(_input_1d_path())
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
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_REFERENCE_BASE / "tests/integrated/2D-recycling/data/BOUT.inp")
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    snapshot = load_local_reference_snapshot(
        _REFERENCE_BASE / "tests/integrated/2D-recycling/data/BOUT.dmp.0.nc",
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
    config = load_bout_input(_input_1d_path())
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
    config = load_bout_input(_input_1d_path())
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


def test_electron_force_balance_gradient_is_jax_jvp_transformable() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    config = load_bout_input(_input_1d_path())
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    field = np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    for j in range(mesh.local_ny):
        field[:, j, :] = 1.0 + 0.2 * float(j)
    weights = jnp.linspace(0.1, 1.3, field.size, dtype=jnp.float64).reshape(field.shape)

    def qoi(scale):
        gradient = grad_par_electron_force_balance_open(
            jnp.asarray(field, dtype=jnp.float64) * scale,
            mesh=mesh,
            metrics=metrics,
        )
        return jnp.sum(gradient * weights)

    _, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-5
    finite_difference = (qoi(1.0 + eps) - qoi(1.0 - eps)) / (2.0 * eps)

    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=2.0e-6, atol=2.0e-8)


def test_target_recycling_sources_are_jax_jvp_transformable() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh = StructuredMesh(
        nx=1,
        ny=1,
        nz=1,
        mxg=0,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=0,
        jyseps1_2=0,
        jyseps2_2=0,
        ny_inner=1,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=np.array([0.0], dtype=np.float64),
        y=np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )
    ones = np.ones((1, 3, 1), dtype=np.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g22=ones,
        g33=ones,
        g_22=ones,
        g23=np.zeros_like(ones),
        Bxy=ones,
    )

    def qoi(scale):
        density = jnp.asarray([[[0.5], [2.0], [0.75]]], dtype=jnp.float64) * scale
        velocity = jnp.asarray([[[-0.2], [1.5], [0.3]]], dtype=jnp.float64)
        temperature = jnp.asarray([[[0.4], [1.0], [0.8]]], dtype=jnp.float64)
        ion = SimpleNamespace(
            name="d+",
            density=density,
            pressure=density * temperature,
            target_recycle=True,
            recycle_as="d",
            target_recycle_multiplier=0.8,
            target_recycle_energy=3.0,
            target_fast_recycle_fraction=0.0,
            target_fast_recycle_energy_factor=1.0,
        )
        neutral = SimpleNamespace(name="d", density=jnp.ones_like(density))
        prepared = {
            "d+": SimpleNamespace(
                density=density,
                pressure=density * temperature,
                temperature=temperature,
                velocity=velocity,
                momentum=density * velocity,
            )
        }
        terms = target_recycling_sources(
            ions=(ion,),
            prepared=prepared,
            neutrals=(neutral,),
            ion_velocity={"d+": velocity},
            mesh=mesh,
            metrics=metrics,
            gamma_i=2.5,
        )
        return jnp.sum(terms.density_source["d"]) + 0.25 * jnp.sum(terms.energy_source["d"])

    value, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    step = 1.0e-5
    finite_difference = (qoi(jnp.array(1.0 + step)) - qoi(jnp.array(1.0 - step))) / (2.0 * step)

    assert np.isfinite(float(value))
    assert np.isfinite(float(tangent))
    assert abs(float(tangent)) > 0.0
    np.testing.assert_allclose(float(tangent), float(finite_difference), rtol=1.0e-7, atol=1.0e-9)


def test_fixed_layout_target_recycling_field_rhs_matches_full_source_active_slice() -> None:
    (
        _config,
        mesh,
        metrics,
        _scalars,
        _runtime_model,
        _fields,
        _species,
        prepared,
        ion_boundary,
        ions,
        neutrals,
        layout,
        _state,
    ) = _build_runtime_target_case()

    full_terms = target_recycling_sources(
        ions=ions,
        prepared=prepared,
        neutrals=neutrals,
        ion_velocity=ion_boundary.velocity,
        mesh=mesh,
        metrics=metrics,
        gamma_i=2.5,
    )
    active_rhs = fixed_layout_target_recycling_field_rhs(
        ions=ions,
        prepared=prepared,
        neutrals=neutrals,
        ion_velocity=ion_boundary.velocity,
        layout=layout,
        mesh=mesh,
        metrics=metrics,
        gamma_i=2.5,
    )

    active_slices = layout.active_slices
    layout_fields = set(layout.field_names)
    for neutral in neutrals:
        if neutral.density_name in layout_fields:
            np.testing.assert_allclose(
                np.asarray(active_rhs[neutral.density_name]),
                np.asarray(full_terms.density_source[neutral.name][active_slices]),
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        if neutral.has_pressure and neutral.pressure_name in layout_fields:
            np.testing.assert_allclose(
                np.asarray(active_rhs[neutral.pressure_name]),
                (2.0 / 3.0)
                * np.asarray(full_terms.energy_source[neutral.name][active_slices]),
                rtol=1.0e-12,
                atol=1.0e-12,
            )


def test_fixed_layout_target_recycling_promotes_to_fixed_array_rhs() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        _fields,
        _species,
        _prepared,
        _ion_boundary,
        _ions,
        _neutrals,
        layout,
        state,
    ) = _build_runtime_target_case()
    override_species = build_species_field_overrider(
        runtime_model.species_templates,
        mesh=mesh,
    )
    ion_momentum_index = layout.field_names.index("NVd+")

    def field_rhs(active_fields: dict[str, object], _feedback: object) -> dict[str, object]:
        candidate = RecyclingFixedState(
            field_values=tuple(active_fields[name] for name in layout.field_names),
            feedback_values=jnp.asarray([], dtype=jnp.float64),
        )
        full_fields = fixed_state_to_full_fields(candidate, layout=layout)
        species = override_species(full_fields)
        prepared, ion_boundary, _electron_boundary = _prepare_open_field_states(
            species,
            config=config,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=scalars,
        )
        ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
        neutrals = tuple(
            sp for sp in species.values() if sp.charge == 0.0 and sp.name != "e"
        )
        return fixed_layout_target_recycling_field_rhs(
            ions=ions,
            prepared=prepared,
            neutrals=neutrals,
            ion_velocity=ion_boundary.velocity,
            layout=layout,
            mesh=mesh,
            metrics=metrics,
            gamma_i=2.5,
        )

    fixed_rhs = build_fixed_array_rhs(field_rhs, layout=layout)

    def qoi(scale):
        scaled_fields = list(state.field_values)
        scaled_fields[ion_momentum_index] = (
            jnp.asarray(scaled_fields[ion_momentum_index], dtype=jnp.float64)
            * scale
        )
        rhs_state = fixed_rhs(
            RecyclingFixedState(
                field_values=tuple(scaled_fields),
                feedback_values=jnp.asarray(state.feedback_values, dtype=jnp.float64),
            )
        )
        rhs_fields = {
            name: value
            for name, value in zip(layout.field_names, rhs_state.field_values, strict=True)
        }
        return jnp.sum(rhs_fields["Nd"]) + 0.1 * jnp.sum(rhs_fields["Pd"])

    value, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    step = 1.0e-5
    finite_difference = (
        qoi(jnp.array(1.0 + step)) - qoi(jnp.array(1.0 - step))
    ) / (2.0 * step)

    assert np.isfinite(float(value))
    assert np.isfinite(float(tangent))
    np.testing.assert_allclose(
        float(tangent),
        float(finite_difference),
        rtol=1.0e-5,
        atol=1.0e-8,
    )
