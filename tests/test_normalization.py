import numpy as np

from jaxdrb.normalization import apply_normalization


def test_normalization_scales_geometry_and_transport():
    cfg = {
        "normalization": {
            "enabled": True,
            "mode": "physics",
            "Te0_eV": 10.0,
            "Ti0_eV": 20.0,
            "n0": 1e19,
            "B0": 2.0,
            "m_i_amu": 2.0,
            "Z_i": 1.0,
            "length_unit": "rho_s",
        },
        "geometry_physical": {
            "Lx": 0.1,
            "Ly": 0.2,
            "Lz": 0.3,
            "R0": 2.0,
            "B0": 1.5,
        },
        "physics_physical": {
            "omega_n": 5.0,
        },
        "transport_physical": {
            "Dn": 0.5,
        },
    }

    converted, info = apply_normalization(cfg)
    assert info is not None

    expected_Lx = 0.1 / info.length
    expected_Ly = 0.2 / info.length
    expected_Lz = 0.3 / info.length
    expected_R0 = 2.0 / info.length
    expected_B0 = 1.5 / info.B0

    geom = converted["geometry"]
    assert np.isclose(geom["Lx"], expected_Lx, rtol=1e-6)
    assert np.isclose(geom["Ly"], expected_Ly, rtol=1e-6)
    assert np.isclose(geom["Lz"], expected_Lz, rtol=1e-6)
    assert np.isclose(geom["R0"], expected_R0, rtol=1e-6)
    assert np.isclose(geom["B0"], expected_B0, rtol=1e-6)

    physics = converted["physics"]
    expected_omega_n = 5.0 * info.length
    assert np.isclose(physics["omega_n"], expected_omega_n, rtol=1e-6)

    transport = converted["transport"]
    expected_Dn = 0.5 * (info.time / (info.length**2))
    assert np.isclose(transport["Dn"], expected_Dn, rtol=1e-6)

    # tau_i should be set when not supplied
    assert np.isclose(physics["tau_i"], info.Ti0_eV / info.Te0_eV, rtol=1e-6)
    # me_hat and atomic mass should follow normalization defaults
    assert np.isclose(physics["average_atomic_mass"], 2.0, rtol=1e-12)
    expected_me_hat = 9.1093837015e-31 / (2.0 * 1.67262192369e-27)
    assert np.isclose(physics["me_hat"], expected_me_hat, rtol=1e-6)

    # poisson_scale defaults to (rho_s / Lref)^2 when normalization is enabled
    numerics = converted.get("numerics", {})
    expected_poisson_scale = (info.rho_s / info.length) ** 2
    assert np.isclose(numerics["poisson_scale"], expected_poisson_scale, rtol=1e-6)


def test_normalization_manual_mode():
    cfg = {
        "normalization": {
            "enabled": True,
            "mode": "manual",
            "length": 2.0,
            "time": 0.5,
            "density": 4.0,
            "temperature": 10.0,
            "potential": 20.0,
            "velocity": 3.0,
            "B0": 5.0,
            "n0": 4.0,
        },
        "geometry_physical": {"Lx": 2.0},
        "physics_physical": {"omega_n": 2.0},
        "transport_physical": {"Dn": 1.0},
    }

    converted, info = apply_normalization(cfg)
    assert info is not None

    assert np.isclose(converted["geometry"]["Lx"], 1.0, rtol=1e-6)
    assert np.isclose(converted["physics"]["omega_n"], 4.0, rtol=1e-6)
    expected_Dn = 1.0 * (info.time / (info.length**2))
    assert np.isclose(converted["transport"]["Dn"], expected_Dn, rtol=1e-6)


def test_normalization_bc_phi_timescale():
    cfg = {
        "normalization": {
            "enabled": True,
            "mode": "manual",
            "length": 2.0,
            "time": 0.5,
            "density": 1.0,
            "temperature": 1.0,
            "potential": 1.0,
            "velocity": 1.0,
            "B0": 1.0,
            "n0": 1.0,
        },
        "bc_physical": {"phi_boundary_timescale": 2.0},
    }

    converted, info = apply_normalization(cfg)
    assert info is not None
    bc = converted.get("bc", {})
    expected = info.time / 2.0
    assert np.isclose(bc["bc_enforce_nu_phi"], expected, rtol=1e-6)


def test_normalization_fills_mass_params_without_physical_sections():
    cfg = {
        "normalization": {
            "enabled": True,
            "mode": "physics",
            "Te0_eV": 30.0,
            "Ti0_eV": 30.0,
            "n0": 2e19,
            "B0": 1.2,
            "m_i_amu": 3.0,
            "Z_i": 1.0,
        },
        "physics": {"boussinesq": True},
    }

    converted, info = apply_normalization(cfg)
    assert info is not None
    physics = converted["physics"]
    assert np.isclose(physics["average_atomic_mass"], 3.0, rtol=1e-12)
    expected_me_hat = 9.1093837015e-31 / (3.0 * 1.67262192369e-27)
    assert np.isclose(physics["me_hat"], expected_me_hat, rtol=1e-6)
