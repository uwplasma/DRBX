from __future__ import annotations

import numpy as np

from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium, phi_from_omega
from jaxdrb.models.em_drb import State as EMState
from jaxdrb.models.em_drb import rhs_nonlinear as em_rhs
from jaxdrb.models.hot_ion_drb import State as HotState
from jaxdrb.models.hot_ion_drb import rhs_nonlinear as hot_rhs
from jaxdrb.models.params import DRBParams
from jaxdrb.models.sheath import (
    apply_loizu2012_mpse_full_linear_bc,
    sheath_energy_losses,
)


def test_em_loizu2012_fullset_current_closure_matches_dpsi_update() -> None:
    """Quantitative gate: EM sheath closure maps (dv_i - dv_e) -> dpsi via Ampère closure."""

    nl = 33
    kx = 0.0
    ky = 0.35
    geom = OpenSlabGeometry.make(nl=nl, length=6.0, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)

    params_on = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.05,
        beta=0.1,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        Dpsi=0.0,
        sheath_bc_on=True,
        sheath_bc_model=1,
        sheath_bc_nu_factor=1.0,
        sheath_end_damp_on=False,
        sheath_loss_on=False,
        sheath_heat_on=False,
        kperp2_min=1e-6,
    )
    params_off = DRBParams(**{**params_on.__dict__, "sheath_bc_on": False})

    # Deterministic nontrivial state.
    x = np.linspace(0.0, 2.0 * np.pi, nl)
    y = EMState(
        n=(0.1 * np.sin(2.0 * x) + 0.05j * np.cos(x)).astype(np.complex128),
        omega=(0.07 * np.cos(3.0 * x) + 0.03j * np.sin(2.0 * x)).astype(np.complex128),
        psi=(0.06 * np.sin(x) + 0.02j * np.cos(2.0 * x)).astype(np.complex128),
        vpar_i=(0.04 * np.cos(2.0 * x) + 0.01j * np.sin(3.0 * x)).astype(np.complex128),
        Te=(0.03 * np.sin(3.0 * x) + 0.02j * np.cos(x)).astype(np.complex128),
    )

    r_off = em_rhs(0.0, y, params_off, geom, kx=kx, ky=ky, eq=eq)
    r_on = em_rhs(0.0, y, params_on, geom, kx=kx, ky=ky, eq=eq)

    k2 = geom.kperp2(kx, ky)
    phi = phi_from_omega(
        y.omega,
        k2,
        kperp2_min=params_on.kperp2_min,
        boussinesq=params_on.boussinesq,
        n0=eq.n0,
        n0_min=params_on.n0_min,
    )
    jpar = k2 * y.psi
    vpar_e = y.vpar_i - jpar

    dn_bc, domega_bc, dvpar_e_bc, dvpar_i_bc, dTe_bc = apply_loizu2012_mpse_full_linear_bc(
        params=params_on,
        geom=geom,
        eq=eq,
        kperp2=k2,
        phi=phi,
        n=y.n,
        omega=y.omega,
        vpar_e=vpar_e,
        vpar_i=y.vpar_i,
        Te=y.Te,
        dpar=geom.dpar,
        d2par=lambda f: geom.dpar(geom.dpar(f)),
    )

    k2_safe = np.maximum(np.asarray(k2), params_on.kperp2_min)
    expected_dpsi_delta = (np.asarray(dvpar_i_bc) - np.asarray(dvpar_e_bc)) / k2_safe

    np.testing.assert_allclose(
        np.asarray(r_on.n - r_off.n), np.asarray(dn_bc), atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(r_on.omega - r_off.omega), np.asarray(domega_bc), atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(r_on.vpar_i - r_off.vpar_i), np.asarray(dvpar_i_bc), atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(r_on.Te - r_off.Te), np.asarray(dTe_bc), atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(r_on.psi - r_off.psi), expected_dpsi_delta, atol=1e-12, rtol=0.0
    )


def test_hot_ion_sheath_heat_and_see_toggles_are_quantitatively_consistent() -> None:
    """Quantitative gate: heat/SEE toggles add the exact sheath energy closure terms."""

    nl = 33
    kx = 0.0
    ky = 0.4
    geom = OpenSlabGeometry.make(nl=nl, length=6.0, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    x = np.linspace(0.0, 2.0 * np.pi, nl)
    y = HotState(
        n=(0.02 * np.sin(x)).astype(np.complex128),
        omega=(0.03 * np.cos(2.0 * x)).astype(np.complex128),
        vpar_e=(0.01 * np.sin(2.0 * x)).astype(np.complex128),
        vpar_i=(0.01 * np.cos(3.0 * x)).astype(np.complex128),
        Te=(0.2 + 0.03 * np.sin(2.0 * x)).astype(np.complex128),
        Ti=(0.15 + 0.02 * np.cos(3.0 * x)).astype(np.complex128),
    )

    common = dict(
        omega_n=0.0,
        omega_Te=0.0,
        omega_Ti=0.0,
        eta=0.0,
        me_hat=0.05,
        tau_i=1.0,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        DTi=0.0,
        chi_par_Te=0.0,
        chi_par_Ti=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        sheath_bc_on=True,
        sheath_bc_model=1,
        sheath_bc_nu_factor=1.0,
        sheath_end_damp_on=False,
        sheath_loss_on=False,
        sheath_gamma_auto=True,
        sheath_lambda=3.28,
        sheath_gamma_i=3.5,
    )

    p_base = DRBParams(**{**common, "sheath_heat_on": False, "sheath_see_on": False})
    p_heat_no_see = DRBParams(**{**common, "sheath_heat_on": True, "sheath_see_on": False})
    p_heat_see = DRBParams(
        **{
            **common,
            "sheath_heat_on": True,
            "sheath_see_on": True,
            "sheath_see_yield": 0.2,
        }
    )

    r_base = hot_rhs(0.0, y, p_base, geom, kx=kx, ky=ky, eq=eq)
    r_heat0 = hot_rhs(0.0, y, p_heat_no_see, geom, kx=kx, ky=ky, eq=eq)
    r_heat1 = hot_rhs(0.0, y, p_heat_see, geom, kx=kx, ky=ky, eq=eq)

    dTe_sh0, dTi_sh0 = sheath_energy_losses(params=p_heat_no_see, geom=geom, Te=y.Te, Ti=y.Ti)
    dTe_sh1, dTi_sh1 = sheath_energy_losses(params=p_heat_see, geom=geom, Te=y.Te, Ti=y.Ti)

    np.testing.assert_allclose(
        np.asarray(r_heat0.Te - r_base.Te), np.asarray(dTe_sh0), atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(r_heat1.Te - r_base.Te), np.asarray(dTe_sh1), atol=1e-12, rtol=0.0
    )
    assert dTi_sh0 is not None and dTi_sh1 is not None
    np.testing.assert_allclose(
        np.asarray(r_heat0.Ti - r_base.Ti), np.asarray(dTi_sh0), atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(r_heat1.Ti - r_base.Ti), np.asarray(dTi_sh1), atol=1e-12, rtol=0.0
    )

    # SEE lowers Lambda_eff and therefore lowers the electron heat-transmission coefficient gamma_e.
    mask = np.asarray(geom.sheath_mask).astype(bool)
    assert np.all(np.abs(np.asarray(dTe_sh1)[mask]) < np.abs(np.asarray(dTe_sh0)[mask]))
