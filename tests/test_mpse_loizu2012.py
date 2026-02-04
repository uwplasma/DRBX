from __future__ import annotations

import numpy as np

from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium, State, rhs_nonlinear
from jaxdrb.models.params import DRBParams


def test_loizu2012_full_mpse_bc_produces_no_nans() -> None:
    nl = 65
    geom = OpenSlabGeometry.make(nl=nl, length=float(2 * np.pi), shat=0.0, curvature0=0.0)
    params = DRBParams(
        omega_n=0.2,
        omega_Te=0.8,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        sheath_bc_on=True,
        sheath_bc_model=1,  # Loizu 2012 full set
        sheath_bc_nu_factor=1.0,
        sheath_cos2=1.0,
    )

    y = State.zeros(nl)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    dy = rhs_nonlinear(0.0, y, params, geom, kx=0.0, ky=0.3, eq=eq)

    for arr in [dy.n, dy.omega, dy.vpar_e, dy.vpar_i, dy.Te]:
        a = np.asarray(arr)
        assert np.all(np.isfinite(a.real))
        assert np.all(np.isfinite(a.imag))


def test_loizu2012_bc_terms_vanish_for_matching_state() -> None:
    nl = 65
    geom = OpenSlabGeometry.make(nl=nl, length=float(2 * np.pi), shat=0.0, curvature0=0.0)
    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        sheath_bc_on=True,
        sheath_bc_model=1,
        sheath_bc_nu_factor=1.0,
        sheath_cos2=1.0,
    )
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)

    # Construct a state that matches the linearized velocity BCs at the boundaries.
    mask = np.asarray(geom.sheath_mask)
    sign = np.asarray(geom.sheath_sign)
    Te = np.zeros(nl, dtype=np.complex128)
    phi = np.zeros(nl, dtype=np.complex128)
    vpar_i = (sign * 0.5 * Te.real).astype(np.complex128)
    vpar_e = (sign * (0.5 * Te.real - phi.real)).astype(np.complex128)
    omega = np.zeros(nl, dtype=np.complex128)

    y = State(
        n=np.zeros(nl, dtype=np.complex128),
        omega=omega,
        vpar_e=vpar_e,
        vpar_i=vpar_i,
        Te=Te,
    )

    # With zero fields and matching BC values, the MPSE terms should produce no forcing.
    dy = rhs_nonlinear(0.0, y, params, geom, kx=0.0, ky=0.3, eq=eq)
    assert np.max(np.abs(np.asarray(dy.vpar_i) * mask)) < 1e-12
    assert np.max(np.abs(np.asarray(dy.vpar_e) * mask)) < 1e-12
