from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.terms.fields import _electron_pressure


def test_electron_pressure_model_default_is_nte_even_with_source_flag() -> None:
    params = DRBSystemParams().update(
        electron_pressure_model="nTe",
        source_Te_is_pressure=True,
    )
    n = jnp.asarray([[2.0, 3.0]])
    Te = jnp.asarray([[5.0, 7.0]])
    out = _electron_pressure(params, n, Te)
    np.testing.assert_allclose(np.asarray(out), np.asarray(n * Te))


def test_electron_pressure_model_te_uses_temperature_channel() -> None:
    params = DRBSystemParams().update(electron_pressure_model="Te")
    n = jnp.asarray([[2.0, 3.0]])
    Te = jnp.asarray([[5.0, 7.0]])
    out = _electron_pressure(params, n, Te)
    np.testing.assert_allclose(np.asarray(out), np.asarray(Te))
