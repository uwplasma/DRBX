from __future__ import annotations

import numpy as np

from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium
from jaxdrb.models.params import DRBParams
from jaxdrb.models.sheath import apply_loizu2012_mpse_full_linear_bc


def test_loizu2012_full_set_vanishing_forcing_for_constructed_state() -> None:
    """Construct a state that satisfies the enforced Loizu2012 targets.

    This test exercises:
      - Eq (21) potential-gradient -> omega target via polarization
      - Eq (22) density-gradient -> n boundary target
      - Eq (23) Te Neumann target
      - Eq (24) vorticity relation -> v_{adjacent} targets
    """

    nl = 65
    geom = OpenSlabGeometry.make(nl=nl, length=float(2 * np.pi), shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = DRBParams(
        sheath_bc_on=True,
        sheath_bc_model=1,
        sheath_bc_nu_factor=1.0,
        sheath_cos2=1.0,
        kperp2_min=1e-6,
    )

    kperp2 = np.ones((nl,), dtype=float)

    # Make a simple interior profile; keep most fields zero for clarity.
    Te = np.zeros((nl,), dtype=np.complex128)
    n = np.zeros((nl,), dtype=np.complex128)
    omega = np.zeros((nl,), dtype=np.complex128)
    phi = np.zeros((nl,), dtype=np.complex128)
    vpar_e = np.zeros((nl,), dtype=np.complex128)
    vpar_i = np.zeros((nl,), dtype=np.complex128)

    # Left boundary targets:
    # Choose an omega0 target, then phi0 = -omega0/k2. Enforce Eq (21) by choosing phi1 accordingly.
    omega0 = 0.2 + 0.0j
    phi0 = -omega0 / kperp2[0]

    # With Te=0 -> v0_target=0. Choose v2=v3=0 and solve Eq (24) for v1_target.
    dl = float(geom.dl)
    v2_target = -omega0 / (params.sheath_cos2 * np.sqrt(eq.Te0[0]) + 1e-12)
    v1_target = (-(dl**2) * v2_target) / 5.0

    vpar_i[0] = 0.0
    vpar_i[1] = v1_target
    vpar_i[2] = 0.0
    vpar_i[3] = 0.0

    phi[0] = phi0
    phi[1] = phi0 - np.sqrt(eq.Te0[0]) * (vpar_i[1] - vpar_i[0])

    # Eq (22) density gradient at boundary.
    n[1] = 0.0
    n[0] = n[1] + (1.0 / np.sqrt(eq.Te0[0])) * (vpar_i[1] - vpar_i[0])

    # Te Neumann.
    Te[0] = Te[1]

    # v_e linearized response at boundary: v_e = sign*(0.5 Te - phi).
    # Left sign is -1 -> v_e = -0.5 Te + phi.
    vpar_e[0] = +phi[0]

    omega[0] = omega0

    # Mirror the same construction at the right boundary.
    omegaN = -0.15 + 0.0j
    phiN = -omegaN / kperp2[-1]
    v2_target_R = -omegaN / (params.sheath_cos2 * np.sqrt(eq.Te0[-1]) + 1e-12)
    vNm1_target = (-(dl**2) * v2_target_R) / 5.0

    vpar_i[-1] = 0.0
    vpar_i[-2] = vNm1_target
    vpar_i[-3] = 0.0
    vpar_i[-4] = 0.0

    phi[-1] = phiN
    phi[-2] = phiN - np.sqrt(eq.Te0[-1]) * (vpar_i[-2] - vpar_i[-1])

    n[-2] = 0.0
    n[-1] = n[-2] + (1.0 / np.sqrt(eq.Te0[-1])) * (vpar_i[-2] - vpar_i[-1])
    Te[-1] = Te[-2]
    # Right sign is +1 -> v_e = -phi (since Te=0 here).
    vpar_e[-1] = -phi[-1]
    omega[-1] = omegaN

    z = np.zeros((nl,), dtype=np.complex128)

    dn, domega, dvpar_e, dvpar_i, dTe = apply_loizu2012_mpse_full_linear_bc(
        params=params,
        geom=geom,
        eq=eq,
        kperp2=kperp2,
        phi=phi,
        n=n,
        omega=omega,
        vpar_e=vpar_e,
        vpar_i=vpar_i,
        Te=Te,
        dpar=lambda f: z,
        d2par=lambda f: z,
    )

    mask = np.asarray(geom.sheath_mask).astype(bool)
    assert np.max(np.abs(np.asarray(domega)[mask])) < 1e-10
    assert np.max(np.abs(np.asarray(dn)[mask])) < 1e-10
    assert np.max(np.abs(np.asarray(dTe)[mask])) < 1e-10
    assert np.max(np.abs(np.asarray(dvpar_i)[mask])) < 1e-10
    assert np.max(np.abs(np.asarray(dvpar_e)[mask])) < 1e-10

    # Adjacent-point enforcement for Eq (24) should also vanish.
    assert abs(np.asarray(dvpar_i)[1]) < 1e-10
    assert abs(np.asarray(dvpar_i)[-2]) < 1e-10
