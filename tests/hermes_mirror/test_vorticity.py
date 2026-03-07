from __future__ import annotations

import tomllib
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.context import build_context
from jaxdrb.driver import build_system_from_config
from jaxdrb.hermes_mirror import full_omega_exb_advection, pi_hat

_CFG = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "open_field_line"
    / "input_tokamak_bxcv_alignment_strict_early.toml"
)
_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_vorticity_global_t1.npz"
)


def _load_cfg() -> dict:
    return tomllib.loads(_CFG.read_text(encoding="utf-8"))


def test_full_omega_exb_advection_is_finite_on_dump_backed_snapshot() -> None:
    cfg = _load_cfg()
    built = build_system_from_config(cfg)

    with np.load(_FIXTURE, allow_pickle=False) as data:
        y = DRBSystemState(
            n=jnp.asarray(data["Ne"], dtype=jnp.float64),
            omega=jnp.asarray(data["Vort"], dtype=jnp.float64),
            vpar_e=jnp.zeros_like(jnp.asarray(data["Ne"], dtype=jnp.float64)),
            vpar_i=jnp.zeros_like(jnp.asarray(data["Nd+"], dtype=jnp.float64)),
            Te=jnp.asarray(data["Te"], dtype=jnp.float64),
            Ti=jnp.asarray(data["Td+"], dtype=jnp.float64),
            psi=None,
            N=None,
        )
        phi = jnp.asarray(data["phi"], dtype=jnp.float64)

    ctx = build_context(built.system.params, built.system.geom, y)
    ctx = eqx.tree_at(lambda c: c.phi, ctx, phi)
    out = full_omega_exb_advection(ctx, y, phi=ctx.phi, scale=ctx.nonlinear_scale)

    assert np.isfinite(np.asarray(out)).all()


def test_pi_hat_is_zero_when_diamagnetic_polarisation_is_disabled() -> None:
    cfg = _load_cfg()
    cfg.setdefault("physics", {})["diamagnetic_polarisation_on"] = False
    built = build_system_from_config(cfg)
    shape = (4, 4, 6)
    out = pi_hat(
        built.system.params,
        n_phys=jnp.ones(shape, dtype=jnp.float64),
        Te_phys=2.0 * jnp.ones(shape, dtype=jnp.float64),
        Ti=3.0 * jnp.ones(shape, dtype=jnp.float64),
    )
    np.testing.assert_allclose(np.asarray(out), 0.0, rtol=1e-12, atol=1e-12)
