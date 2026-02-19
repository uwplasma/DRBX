from __future__ import annotations

import numpy as np

from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium
from jaxdrb.models.params import DRBParams
from jaxdrb.models.sheath import apply_loizu_mpse_boundary_conditions


def test_bohm_mpse_bc_terms_vanish_for_matching_state_linearized() -> None:
    nl = 33
    geom = OpenSlabGeometry.make(nl=nl, length=6.0, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = DRBParams(sheath_bc_on=True, sheath_bc_linearized=True, sheath_bc_nu_factor=1.0)

    mask = np.asarray(geom.sheath_mask)
    sign = np.asarray(geom.sheath_sign)

    Te = np.zeros(nl, dtype=np.complex128)
    phi = np.zeros(nl, dtype=np.complex128)
    vpar_i = (sign * 0.5 * Te.real).astype(np.complex128)
    vpar_e = (sign * (0.5 * Te.real - phi.real)).astype(np.complex128)

    dvpar_e, dvpar_i = apply_loizu_mpse_boundary_conditions(
        params=params, geom=geom, eq=eq, phi=phi, vpar_e=vpar_e, vpar_i=vpar_i, Te=Te
    )

    assert np.max(np.abs(np.asarray(dvpar_i) * mask)) < 1e-12
    assert np.max(np.abs(np.asarray(dvpar_e) * mask)) < 1e-12


def test_bohm_mpse_bc_has_expected_sign_for_positive_Te() -> None:
    nl = 33
    geom = OpenSlabGeometry.make(nl=nl, length=6.0, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = DRBParams(sheath_bc_on=True, sheath_bc_linearized=True, sheath_bc_nu_factor=1.0)

    sign = np.asarray(geom.sheath_sign)
    Te = np.zeros(nl, dtype=np.complex128)
    Te[0] = 1.0 + 0j
    Te[-1] = 1.0 + 0j
    phi = np.zeros(nl, dtype=np.complex128)
    vpar_i = np.zeros(nl, dtype=np.complex128)
    vpar_e = np.zeros(nl, dtype=np.complex128)

    dvpar_e, dvpar_i = apply_loizu_mpse_boundary_conditions(
        params=params, geom=geom, eq=eq, phi=phi, vpar_e=vpar_e, vpar_i=vpar_i, Te=Te
    )

    # Target δv_i = ± 0.5 δTe, and the RHS is -nu (v - v_target), so dv_i at the ends should
    # have the same sign as v_target.
    assert np.sign(np.real(dvpar_i[0])) == np.sign(sign[0])
    assert np.sign(np.real(dvpar_i[-1])) == np.sign(sign[-1])
