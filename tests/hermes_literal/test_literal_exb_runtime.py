from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.bc import BC2D
from jaxdrb.hermes_literal.exb import div_n_bxgrad_f_b_xppm

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_local_rank0_t1.npz"
_TERM_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_term_local_rank0_t1.npz"
)


def _runtime_fixture_inputs() -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    with (
        np.load(_FIXTURE, allow_pickle=False) as data,
        np.load(_TERM_FIXTURE, allow_pickle=False) as terms,
    ):
        ps = int(np.asarray(data["pstart"]))
        pe = int(np.asarray(data["pend"]))
        xs = int(np.asarray(data["xstart"]))
        xe = int(np.asarray(data["xend"]))
        sl = (slice(ps, pe + 1), slice(xs, xe + 1), slice(None))
        bc = BC2D(kind_x=2, kind_y=0, x_value=0.0, y_value=0.0, x_grad=0.0, y_grad=0.0)
        kwargs: dict[str, object] = {
            "jacobian": jnp.asarray(data["J"][sl[0], sl[1]], dtype=jnp.float64),
            "dx": jnp.asarray(data["dx"][sl[0], sl[1]], dtype=jnp.float64),
            "dy": jnp.asarray(data["dy"][sl[0], sl[1]], dtype=jnp.float64),
            "dz": jnp.asarray(data["dz"][sl[0], sl[1]], dtype=jnp.float64),
            "g11": jnp.asarray(data["g11"][sl[0], sl[1]], dtype=jnp.float64),
            "g23": jnp.asarray(data["g23"][sl[0], sl[1]], dtype=jnp.float64),
            "bxy": jnp.asarray(data["Bxy"][sl[0], sl[1]], dtype=jnp.float64),
            "z_shift": jnp.asarray(data["zShift"][sl[0], sl[1]], dtype=jnp.float64),
            "zlength": float(np.asarray(data["zlength"])),
            "bc_phi": bc,
            "bc_adv": bc,
            "bndry_flux": True,
            "poloidal": True,
            "periodic_parallel": False,
            "periodic_binormal": True,
            "lower_boundary_open": bool(np.asarray(data["lower_boundary_open"])),
            "upper_boundary_open": bool(np.asarray(data["upper_boundary_open"])),
            "poisson_invert_set": True,
        }
        ne = np.asarray(data["Ne"][sl], dtype=np.float64)
        pe_field = np.asarray(data["Pe"][sl], dtype=np.float64)
        phi = np.asarray(data["phi"][sl], dtype=np.float64)
        kwargs["phi"] = jnp.asarray(phi, dtype=jnp.float64)
        ne_ref = np.asarray(terms["term_Ne_exb"][sl], dtype=np.float64)
        pe_ref = np.asarray(terms["term_Pe_exb"][sl], dtype=np.float64)
    return kwargs, np.stack([ne, pe_field]), np.stack([ne_ref, pe_ref])


def test_exb_runtime_wrapper_matches_dump_backed_interior_terms() -> None:
    kwargs, fields, refs = _runtime_fixture_inputs()
    phi = kwargs.pop("phi")
    names = ("Ne", "Pe")
    thresholds = {"Ne": 3.0e-4, "Pe": 3.0e-4}

    for idx, name in enumerate(names):
        term = -div_n_bxgrad_f_b_xppm(jnp.asarray(fields[idx]), phi, **kwargs)
        term_np = np.asarray(term)
        ref_np = refs[idx]
        rms = float(np.sqrt(np.mean((term_np - ref_np) ** 2)))
        corr = float(np.corrcoef(term_np.ravel(), ref_np.ravel())[0, 1])
        assert rms < thresholds[name], (name, rms)
        assert corr > 0.98, (name, corr)


def test_exb_runtime_wrapper_is_differentiable() -> None:
    kwargs, fields, _ = _runtime_fixture_inputs()
    phi = kwargs.pop("phi")

    grad = jax.grad(lambda arr: jnp.sum(div_n_bxgrad_f_b_xppm(arr, phi, **kwargs)))(
        jnp.asarray(fields[0], dtype=jnp.float64)
    )
    assert np.isfinite(np.asarray(grad)).all()
