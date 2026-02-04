from __future__ import annotations

import numpy as np

from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium
from jaxdrb.models.params import DRBParams
from jaxdrb.models.sheath import apply_loizu2012_mpse_full_linear_bc_hot_ion, sheath_bc_rate


def test_loizu2012_full_set_hot_ion_adds_Ti_neumann_terms() -> None:
    nl = 33
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
    z = np.zeros((nl,), dtype=np.complex128)
    Ti = (np.linspace(0.0, 1.0, nl) + 0.0j).astype(np.complex128)

    dn, domega, dvpar_e, dvpar_i, dTe, dTi = apply_loizu2012_mpse_full_linear_bc_hot_ion(
        params=params,
        geom=geom,
        eq=eq,
        kperp2=kperp2,
        phi=z,
        n=z,
        omega=z,
        vpar_e=z,
        vpar_i=z,
        Te=z,
        Ti=Ti,
        dpar=lambda f: z,
        d2par=lambda f: z,
    )

    # Basic sanity: finite outputs.
    for arr in [dn, domega, dvpar_e, dvpar_i, dTe, dTi]:
        a = np.asarray(arr)
        assert np.all(np.isfinite(a.real))
        assert np.all(np.isfinite(a.imag))

    bc = sheath_bc_rate(params, geom)
    assert bc is not None
    nu, mask = bc
    nu = float(np.asarray(nu))
    mask = np.asarray(mask, dtype=float)

    Ti_target = Ti.copy()
    Ti_target[0] = Ti[1]
    Ti_target[-1] = Ti[-2]
    expected = -nu * mask * (Ti - Ti_target)

    np.testing.assert_allclose(np.asarray(dTi), expected, rtol=0.0, atol=1e-12)
    assert np.allclose(np.asarray(dTi)[1:-1], 0.0)


def test_loizu2012_full_set_hot_ion_Ti_neumann_vanishes_when_satisfied() -> None:
    nl = 33
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
    z = np.zeros((nl,), dtype=np.complex128)
    Ti = (np.sin(np.linspace(0.0, 2 * np.pi, nl)) + 0.0j).astype(np.complex128)
    Ti[0] = Ti[1]
    Ti[-1] = Ti[-2]

    *_rest, dTi = apply_loizu2012_mpse_full_linear_bc_hot_ion(
        params=params,
        geom=geom,
        eq=eq,
        kperp2=kperp2,
        phi=z,
        n=z,
        omega=z,
        vpar_e=z,
        vpar_i=z,
        Te=z,
        Ti=Ti,
        dpar=lambda f: z,
        d2par=lambda f: z,
    )

    mask = np.asarray(geom.sheath_mask).astype(bool)
    assert np.max(np.abs(np.asarray(dTi)[mask])) < 1e-12
