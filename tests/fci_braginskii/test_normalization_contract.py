"""Normalization contract of the FCI Braginskii HSX backend.

Stated normalization (``docs/fci_braginskii_hsx_backend.md`` and the
``--rho-star`` help text of ``simulate_hsx_blob.py``):

* lengths in metres, L_ref = 1 m; |B| in tesla, B_ref = 1 T;
* time in L_ref / c_s with c_s = sqrt(T_e / m_i);
* rho* = rho_s / L_ref with rho_s = sqrt(T_e m_i) / (e B_ref);
* E x B and curvature drift terms scale linearly with rho*;
* the polarization relation scales with rho*^2 (omega = rho*^2 * (...));
* parallel transport (sound, current, electron force) does not depend on rho*;
* tau = T_i / T_e and mi/me are separate inputs.

The fast tests below check the parser, documentation and unit-conversion
side of the contract, plus the production ``polarization_balance_terms``
method on a stub with replaced stencil operators.  The ``slow`` test runs the
real driver on the canonical HSX geometry (``DRBX_TEST_GEOMETRY_BUNDLE``) and
evaluates the production ``evaluate_stage`` term fields at two rho* values
with the potential held fixed.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_braginskii.native import fci_drb_EB_rhs
from drbx.fci_braginskii.native.fci_drb_EB_rhs import (
    FciDrbEBRhsParameters,
    FciDrbEBState,
    LocalFciDrbEBRhs,
    RHS_TERM_FIELD_NAMES,
    RHS_TERM_NAMES,
)

jax.config.update("jax_enable_x64", True)

REPOSITORY = Path(__file__).resolve().parents[2]
DRIVER_PATH = REPOSITORY / "simulate_hsx_blob.py"
DOCS_PATH = REPOSITORY / "docs" / "fci_braginskii_hsx_backend.md"

# CODATA 2018 exact / recommended values.
ELEMENTARY_CHARGE = 1.602176634e-19  # C
PROTON_MASS = 1.67262192369e-27  # kg
ELECTRON_MASS = 9.1093837015e-31  # kg


def _driver_module():
    spec = importlib.util.spec_from_file_location(
        "simulate_hsx_blob_normalization_contract", DRIVER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sound_speed(te_ev: float, ion_mass: float = PROTON_MASS) -> float:
    return math.sqrt(te_ev * ELEMENTARY_CHARGE / ion_mass)


def _rho_star(te_ev: float, b_tesla: float = 1.0, l_ref: float = 1.0) -> float:
    rho_s = math.sqrt(te_ev * ELEMENTARY_CHARGE * PROTON_MASS) / (
        ELEMENTARY_CHARGE * b_tesla
    )
    return rho_s / l_ref


def _normalization_section() -> str:
    text = DOCS_PATH.read_text()
    match = re.search(r"## Normalization\n(.*?)\n## ", text, re.S)
    assert match is not None, "docs lost the Normalization section"
    return " ".join(match.group(1).split())


# --------------------------------------------------------------------------
# (c) driver default and documented rho* agree
# --------------------------------------------------------------------------


def test_driver_default_rho_star_matches_documented_value():
    args = _driver_module()._build_parser().parse_args(())
    section = _normalization_section()
    documented = re.search(r"default rho\* is (\S+) \(", section)
    assert documented is not None
    assert args.rho_star == float(documented.group(1)) == 5.0e-4
    # The documented time base is t = 0.15 in 200 steps.
    assert args.final_time == 0.15
    assert args.num_steps == 200
    assert math.isclose(args.final_time / args.num_steps, 7.5e-4)
    # tau = Ti/Te and the hydrogen mass ratio are the other dimensionless inputs.
    assert args.tau == 1.0
    assert args.mi_over_me == 1836.0
    assert math.isclose(args.mi_over_me, PROTON_MASS / ELECTRON_MASS, rel_tol=1.0e-3)


def test_rho_star_help_states_the_metre_reference_and_scalings():
    parser = _driver_module()._build_parser()
    action = next(a for a in parser._actions if a.dest == "rho_star")
    help_text = " ".join(action.help.split())
    assert "L_ref = 1 m" in help_text
    assert "by rho*, polarization by rho*^2" in help_text
    assert "Te ~ 24 eV and B = 1 T" in help_text


# --------------------------------------------------------------------------
# (d) physical-unit conversions stated for the reference values
# --------------------------------------------------------------------------


def test_documented_rho_star_follows_from_the_reference_values():
    # Default: hydrogen, Te ~ 24 eV, B = 1 T, L_ref = 1 m.
    assert _rho_star(24.0) == pytest.approx(5.0e-4, rel=0.01)
    section = _normalization_section()
    at_20 = re.search(r"e\.g\. (\S+) at 20 eV", section)
    assert at_20 is not None
    assert _rho_star(20.0) == pytest.approx(float(at_20.group(1)), rel=0.01)
    # rho* scales as sqrt(Te)/B.
    assert _rho_star(96.0, 2.0) == pytest.approx(_rho_star(24.0, 1.0), rel=1e-12)


def test_documented_sound_speed_and_time_unit_follow_from_the_formula():
    section = _normalization_section()
    cs_doc = float(re.search(r"c_s ~ (\S+) m/s", section).group(1))
    unit_us = float(re.search(r"one time unit is about (\d+) microseconds", section).group(1))
    t_doc, t_us = re.search(r"t = (\S+) is about (\d+) microseconds", section).groups()
    c_s = _sound_speed(24.0)
    assert c_s == pytest.approx(cs_doc, rel=0.01)
    time_unit_us = 1.0e6 * 1.0 / c_s  # L_ref / c_s with L_ref = 1 m
    assert time_unit_us == pytest.approx(float(unit_us), abs=0.5)
    assert float(t_doc) * time_unit_us == pytest.approx(float(t_us), abs=0.5)
    # Internal consistency of the two definitions: rho_s = c_s / Omega_i.
    omega_i = ELEMENTARY_CHARGE * 1.0 / PROTON_MASS
    assert c_s / omega_i == pytest.approx(_rho_star(24.0), rel=1e-12)


def test_run_metadata_records_rho_star_for_unit_reconstruction():
    source = DRIVER_PATH.read_text()
    assert '"rho_star": float(args.rho_star)' in source


# --------------------------------------------------------------------------
# (a) polarization ~ rho*^2: production polarization_balance_terms on a stub
# --------------------------------------------------------------------------


SHAPE = (3, 4, 5)


def _polarization_stub(monkeypatch, rho_star: float):
    rng = np.random.default_rng(7)
    phi = jnp.asarray(rng.standard_normal(SHAPE))
    ti = jnp.asarray(1.0 + 0.1 * rng.standard_normal(SHAPE))
    vorticity = jnp.asarray(rng.standard_normal(SHAPE))
    state = FciDrbEBState(
        density=jnp.ones(SHAPE),
        phi=phi,
        Te=jnp.ones(SHAPE),
        Ti=ti,
        Vi=jnp.zeros(SHAPE),
        Ve=jnp.zeros(SHAPE),
        vorticity=vorticity,
    )
    # A deterministic stand-in for the conservative perpendicular Laplacian;
    # the identity under test does not depend on the particular operator.
    monkeypatch.setattr(
        fci_drb_EB_rhs, "build_local_conservative_stencil_from_field",
        lambda field, geometry, context: field,
    )
    monkeypatch.setattr(
        fci_drb_EB_rhs, "local_perp_laplacian_conservative_op",
        lambda stencil, *a, **k: jnp.roll(stencil, 1, 0) + jnp.roll(stencil, -1, 1) - 2.0 * stencil,
    )
    monkeypatch.setattr(
        fci_drb_EB_rhs, "_mask_inactive_owned", lambda values, geometry: values
    )
    stub = SimpleNamespace(
        parameters=FciDrbEBRhsParameters(tau=1.3, rho_star=rho_star),
        geometry=SimpleNamespace(regular_face_geometry=None),
        domain=None,
        face_projectors=None,
        axis_regular_axes=(False, False, False),
        neumann_normal_scheme="physical",
        _face_bcs=lambda s: SimpleNamespace(phi=None, Ti=None),
        _prepare_state_halo=lambda s, bc: s,
        _prepare_phi_halo=lambda p, bc: p,
        _stencil_builder_context=lambda: None,
        _restrict_fine_field=lambda v: v,
    )
    return LocalFciDrbEBRhs.polarization_balance_terms(stub, state)


def test_polarization_balance_scales_vorticity_by_inverse_rho_star_squared(monkeypatch):
    r1, r2 = 5.0e-4, 2.0e-3
    t1 = np.asarray(_polarization_stub(monkeypatch, r1))
    t2 = np.asarray(_polarization_stub(monkeypatch, r2))
    # -Lperp phi and tau Lperp Ti do not depend on rho*.
    np.testing.assert_array_equal(t1[0], t2[0])
    np.testing.assert_array_equal(t1[1], t2[1])
    # -omega / rho*^2: equivalently omega = rho*^2 (Lperp phi + tau Lperp Ti).
    np.testing.assert_allclose(t2[2] * r2**2, t1[2] * r1**2, rtol=1e-14, atol=0)
    assert np.max(np.abs(t1[2])) > 0.0


# --------------------------------------------------------------------------
# (a), (b) E x B / curvature ~ rho*, parallel terms independent of rho*:
# production evaluate_stage on the real HSX geometry
# --------------------------------------------------------------------------

LINEAR_IN_RHO_STAR = {"poisson_bracket", "curvature"}


class _Captured(Exception):
    pass


@pytest.mark.slow
def test_evaluate_stage_term_scaling_on_hsx_geometry(monkeypatch, tmp_path):
    geometry = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
    if not geometry or not (Path(geometry) / "manifest.json").is_file():
        pytest.skip("set DRBX_TEST_GEOMETRY_BUNDLE to an HSX FCI geometry bundle")
    driver = _driver_module()
    r1, r2 = 5.0e-4, 2.0e-3
    captured: dict[str, np.ndarray] = {}
    original = LocalFciDrbEBRhs.reconstruct_phi

    def hooked(self, state_owned, *, return_diagnostics=False):
        phi, info = original(self, state_owned, return_diagnostics=True)
        arrays = []
        for rho_star in (r1, r2):
            model = dataclasses.replace(
                self, parameters=dataclasses.replace(self.parameters, rho_star=rho_star)
            )
            _, terms = model.evaluate_stage(
                state_owned, phi_owned=phi, return_rhs_term_fields=True
            )
            arrays.append(terms)
            arrays.append(model.polarization_balance_terms(state_owned, phi_owned=phi))

        def store(t1, p1, t2, p2):
            captured.update(t1=np.asarray(t1), p1=np.asarray(p1),
                            t2=np.asarray(t2), p2=np.asarray(p2))
            raise _Captured()

        jax.debug.callback(store, *arrays)
        return (phi, info) if return_diagnostics else phi

    monkeypatch.setattr(LocalFciDrbEBRhs, "reconstruct_phi", hooked)
    with pytest.raises(Exception):
        driver.main([
            "--geometry", geometry,
            "--num-steps", "1",
            "--final-time", "7.5e-4",
            "--filament-cache-dir", str(tmp_path / "filament"),
            "--advance-execution", "eager",
            "--no-phase-timing",
            "--output", str(tmp_path / "history.npz"),
        ])
    assert captured, "hook did not run"
    t1, t2 = captured["t1"], captured["t2"]
    ratio = r2 / r1
    checked_linear = checked_independent = 0
    for f, field in enumerate(RHS_TERM_FIELD_NAMES):
        for s, name in enumerate(RHS_TERM_NAMES[f]):
            a, b = t1[f, s], t2[f, s]
            scale = max(float(np.max(np.abs(a))), 1e-300)
            if name in LINEAR_IN_RHO_STAR:
                np.testing.assert_allclose(b, ratio * a, rtol=0, atol=1e-12 * ratio * scale,
                                           err_msg=f"{field}/{name}")
                checked_linear += float(np.max(np.abs(a))) > 0.0
            else:
                np.testing.assert_allclose(b, a, rtol=0, atol=1e-12 * scale,
                                           err_msg=f"{field}/{name}")
                checked_independent += float(np.max(np.abs(a))) > 0.0
    assert checked_linear >= 4
    assert checked_independent >= 4
    p1, p2 = captured["p1"], captured["p2"]
    np.testing.assert_array_equal(p1[:2], p2[:2])
    np.testing.assert_allclose(p2[2] * r2**2, p1[2] * r1**2, rtol=1e-13, atol=0)
