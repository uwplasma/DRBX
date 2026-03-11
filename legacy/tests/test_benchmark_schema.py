from __future__ import annotations

import numpy as np

from jaxdrb.benchmarking import (
    BenchmarkBundle,
    BenchmarkNormalization,
    load_bundle_npz,
    save_bundle_npz,
)


def test_benchmark_bundle_roundtrip(tmp_path):
    norm = BenchmarkNormalization(Nnorm=1.0e19, Tnorm_eV=50.0, Bnorm_T=1.0, m_i_amu=2.0, Z_i=1.0)
    bundle = BenchmarkBundle(
        code="jax_drb",
        geometry="tokamak_open_field",
        normalization=norm,
        times_norm=np.array([0.0, 0.1, 0.2]),
        times_si=np.array([0.0, 1.0e-7, 2.0e-7]),
        axes={"x": np.arange(4), "y": np.arange(5)},
        diagnostics={"rms_n_fluct": np.array([0.01, 0.02, 0.03])},
        snapshots={"n_fluct_last": np.ones((4, 5))},
        metadata={"case": "unit"},
    )

    out = save_bundle_npz(bundle, tmp_path / "bundle.npz")
    loaded = load_bundle_npz(out)

    assert loaded.code == "jax_drb"
    assert loaded.geometry == "tokamak_open_field"
    assert np.allclose(loaded.times_norm, bundle.times_norm)
    assert np.allclose(loaded.times_si, bundle.times_si)
    assert np.allclose(loaded.axes["x"], bundle.axes["x"])
    assert np.allclose(loaded.diagnostics["rms_n_fluct"], bundle.diagnostics["rms_n_fluct"])
    assert np.allclose(loaded.snapshots["n_fluct_last"], bundle.snapshots["n_fluct_last"])
    assert loaded.metadata["case"] == "unit"

    # Derived normalization values should be finite and positive.
    assert loaded.normalization.cs0_m_s > 0.0
    assert loaded.normalization.omega_ci_s > 0.0
    assert loaded.normalization.rho_s0_m > 0.0
