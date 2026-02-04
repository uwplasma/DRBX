from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.models.params import DRBParams
from jaxdrb.models.sheath import sheath_energy_losses, sheath_gamma_e, sheath_lambda_effective


def test_sheath_lambda_effective_decreases_with_see() -> None:
    params0 = DRBParams(sheath_lambda=3.28, sheath_see_on=False, sheath_see_yield=0.2)
    params1 = DRBParams(sheath_lambda=3.28, sheath_see_on=True, sheath_see_yield=0.2)

    lam0 = float(sheath_lambda_effective(params0))
    lam1 = float(sheath_lambda_effective(params1))
    assert lam1 < lam0
    assert np.isclose(lam1, 3.28 + np.log(0.8), rtol=0.0, atol=1e-12)


def test_sheath_gamma_e_auto_uses_lambda_effective() -> None:
    params = DRBParams(
        sheath_lambda=3.0, sheath_gamma_auto=True, sheath_see_on=True, sheath_see_yield=0.1
    )
    ge = float(sheath_gamma_e(params))
    assert np.isclose(ge, 2.0 + 3.0 + np.log(0.9), rtol=0.0, atol=1e-12)


def test_sheath_gamma_e_manual_override() -> None:
    params = DRBParams(sheath_gamma_auto=False, sheath_gamma_e=9.0)
    assert float(sheath_gamma_e(params)) == 9.0


def test_sheath_energy_losses_match_masked_gamma_times_nu() -> None:
    nl = 33
    geom = OpenSlabGeometry.make(nl=nl, length=6.0)
    Te = jnp.ones((nl,), dtype=jnp.float64)
    Ti = 2.0 * jnp.ones((nl,), dtype=jnp.float64)

    params = DRBParams(
        sheath_bc_on=True,
        sheath_bc_nu_factor=1.0,
        sheath_heat_on=True,
        sheath_gamma_auto=False,
        sheath_gamma_e=5.0,
        sheath_gamma_i=3.5,
    )

    dTe, dTi = sheath_energy_losses(params=params, geom=geom, Te=Te, Ti=Ti)

    # nu = 2/Lpar with Lpar = 6.0 for the chosen geometry.
    nu = 2.0 / 6.0
    mask = np.zeros((nl,))
    mask[0] = 1.0
    mask[-1] = 1.0
    expected_dTe = (-nu * 5.0) * mask * np.ones((nl,))
    expected_dTi = (-nu * 3.5) * mask * 2.0 * np.ones((nl,))

    assert np.allclose(np.asarray(dTe), expected_dTe, rtol=0.0, atol=1e-12)
    assert dTi is not None
    assert np.allclose(np.asarray(dTi), expected_dTi, rtol=0.0, atol=1e-12)
