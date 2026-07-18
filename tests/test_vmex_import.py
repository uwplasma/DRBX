from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from drbx.geometry import (
    evaluate_vmex_surface_field,
    load_vmex_wout,
    trace_vmex_field_lines,
    traced_rotational_transform,
    vmex_boundary_rz,
    vmex_half_mesh_s,
    vmex_runtime_available,
    vmex_surface_rz,
    vmex_wout_summary,
)


def _find_local_wout() -> Path | None:
    """A Landreman-Paul QA wout from the external checkouts, when present."""

    essos_root = Path(os.environ.get("DRBX_ESSOS_ROOT", Path.home() / "local" / "ESSOS_test")).expanduser()
    vmex_root = Path(os.environ.get("DRBX_VMEX_ROOT", Path.home() / "local" / "vmec_jax")).expanduser()
    candidates = (
        essos_root / "examples" / "input_files" / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc",
        vmex_root / "examples" / "data" / "wout_LandremanPaul2021_QA_lowres.nc",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


_WOUT_PATH = _find_local_wout()
pytestmark = pytest.mark.skipif(
    not vmex_runtime_available() or _WOUT_PATH is None,
    reason="VMEX runtime or a local Landreman-Paul QA wout file is not available",
)


@pytest.fixture(scope="module")
def wout():
    return load_vmex_wout(_WOUT_PATH)


def test_adapter_loads_wout_and_summarizes(wout) -> None:
    summary = vmex_wout_summary(wout)
    assert summary["ns"] > 1
    assert summary["nfp"] >= 1
    assert np.isfinite(summary["b0"]) and summary["b0"] != 0.0
    assert np.isfinite(summary["aspect"]) and summary["aspect"] > 1.0
    assert np.isfinite(summary["iota_axis"]) and np.isfinite(summary["iota_edge"])
    s_half = vmex_half_mesh_s(wout)
    assert s_half.shape == (summary["ns"] - 1,)
    assert np.all((s_half > 0.0) & (s_half < 1.0))


def test_surface_field_is_finite_and_toroidal(wout) -> None:
    ns = int(wout.ns)
    theta = np.linspace(0.0, 2.0 * np.pi, 17)[:, None]
    phi = np.linspace(0.0, 2.0 * np.pi, 9)[None, :]
    field = evaluate_vmex_surface_field(wout, s_index=ns // 2, theta=theta, phi=phi)
    assert 0.0 < field["s"] < 1.0
    for name in ("b_sup_theta", "b_sup_phi", "mod_b"):
        assert field[name].shape == (17, 9)
        assert np.all(np.isfinite(field[name]))
    assert np.all(field["mod_b"] > 0.0)
    # B^phi never vanishes on a nested-surface equilibrium (single sign).
    signs = np.sign(field["b_sup_phi"])
    assert np.all(signs == signs.flat[0])


def test_surface_rz_and_boundary(wout) -> None:
    boundary_r, boundary_z = vmex_boundary_rz(wout, phi=0.0, n_theta=64)
    assert np.all(np.isfinite(boundary_r)) and np.all(np.isfinite(boundary_z))
    assert np.all(boundary_r > 0.0)
    np.testing.assert_allclose(boundary_r[0], boundary_r[-1], rtol=1e-12)
    np.testing.assert_allclose(boundary_z[0], boundary_z[-1], atol=1e-12)
    axis_r, axis_z = vmex_surface_rz(wout, s=0.0, theta=np.array(0.0), phi=np.array(0.0))
    assert boundary_r.min() < float(axis_r) < boundary_r.max()
    assert np.isfinite(float(axis_z))


def test_traced_iota_matches_wout_profile(wout) -> None:
    ns = int(wout.ns)
    s_index = ns // 2
    s_value = float(vmex_half_mesh_s(wout)[s_index - 1])
    phi_nodes, theta_lines = trace_vmex_field_lines(
        wout, s_index=s_index, theta0=np.array([0.0, 1.5]), n_transits=30, steps_per_transit=64
    )
    assert np.all(np.isfinite(theta_lines))
    iota_traced = traced_rotational_transform(phi_nodes, theta_lines)
    iota_wout = float(np.interp(s_value, np.linspace(0.0, 1.0, ns), np.asarray(wout.iotaf)))
    assert np.all(np.abs(iota_traced - iota_wout) < 1.0e-2 * abs(iota_wout)), (
        f"traced iota {iota_traced} vs wout iotaf {iota_wout} at s={s_value}"
    )
