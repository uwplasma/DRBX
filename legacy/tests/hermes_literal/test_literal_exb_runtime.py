from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdrb.bc import BC2D
from jaxdrb.hermes_literal.exb import div_n_bxgrad_f_b_xppm

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_local_rank0_t1.npz"
_TERM_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_term_local_rank0_t1.npz"
)
_UPPER_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_local_rank5_t1.npz"
)
_GLOBAL_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_global_t1.npz"
)


def _runtime_fixture_inputs(
    fixture_path: Path,
    term_fixture_path: Path | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    with np.load(fixture_path, allow_pickle=False) as data:
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
        if term_fixture_path is None:
            ne_ref = np.asarray(data["term_Ne_exb"][sl], dtype=np.float64)
            pe_ref = np.asarray(data["term_Pe_exb"][sl], dtype=np.float64)
        else:
            with np.load(term_fixture_path, allow_pickle=False) as terms:
                ne_ref = np.asarray(terms["term_Ne_exb"][sl], dtype=np.float64)
                pe_ref = np.asarray(terms["term_Pe_exb"][sl], dtype=np.float64)
    return kwargs, np.stack([ne, pe_field]), np.stack([ne_ref, pe_ref])


@pytest.mark.parametrize(
    ("fixture_path", "term_fixture_path"),
    [
        (_FIXTURE, _TERM_FIXTURE),
        (_UPPER_FIXTURE, None),
    ],
)
def test_exb_runtime_wrapper_matches_dump_backed_interior_terms(
    fixture_path: Path,
    term_fixture_path: Path | None,
) -> None:
    kwargs, fields, refs = _runtime_fixture_inputs(fixture_path, term_fixture_path)
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
    kwargs, fields, _ = _runtime_fixture_inputs(_FIXTURE, _TERM_FIXTURE)
    phi = kwargs.pop("phi")

    grad = jax.grad(lambda arr: jnp.sum(div_n_bxgrad_f_b_xppm(arr, phi, **kwargs)))(
        jnp.asarray(fields[0], dtype=jnp.float64)
    )
    assert np.isfinite(np.asarray(grad)).all()


def test_exb_runtime_parallel_subdomains_improve_stitched_global_fixture() -> None:
    with np.load(_GLOBAL_FIXTURE, allow_pickle=False) as data:
        bc = BC2D(kind_x=2, kind_y=0, x_value=0.0, y_value=0.0, x_grad=0.0, y_grad=0.0)
        common: dict[str, object] = {
            "jacobian": jnp.asarray(data["J"], dtype=jnp.float64),
            "dx": jnp.asarray(data["dx"], dtype=jnp.float64),
            "dy": jnp.asarray(data["dy"], dtype=jnp.float64),
            "dz": jnp.asarray(data["dz"], dtype=jnp.float64),
            "g11": jnp.asarray(data["g11"], dtype=jnp.float64),
            "g23": jnp.asarray(data["g23"], dtype=jnp.float64),
            "bxy": jnp.asarray(data["Bxy"], dtype=jnp.float64),
            "z_shift": jnp.asarray(data["zShift"], dtype=jnp.float64),
            "zlength": float(np.asarray(data["zlength"])),
            "bc_phi": bc,
            "bc_adv": bc,
            "bndry_flux": True,
            "poloidal": True,
            "periodic_parallel": False,
            "periodic_binormal": True,
            "lower_boundary_open": True,
            "upper_boundary_open": True,
            "poisson_invert_set": True,
            "parallel_edge_block": int(np.asarray(data["mysub"])),
        }
        block = int(np.asarray(data["mysub"]))
        for field, ref_key in (("Ne", "term_Ne_exb"), ("Pe", "term_Pe_exb")):
            full = -div_n_bxgrad_f_b_xppm(
                jnp.asarray(data[field], dtype=jnp.float64),
                jnp.asarray(data["phi"], dtype=jnp.float64),
                **common,
            )
            blocked = -div_n_bxgrad_f_b_xppm(
                jnp.asarray(data[field], dtype=jnp.float64),
                jnp.asarray(data["phi"], dtype=jnp.float64),
                parallel_subdomain_size=block,
                **common,
            )
            ref = np.asarray(data[ref_key], dtype=np.float64)
            full_rel = float(
                np.sqrt(np.mean((np.asarray(full) - ref) ** 2)) / np.sqrt(np.mean(ref**2))
            )
            blocked_rel = float(
                np.sqrt(np.mean((np.asarray(blocked) - ref) ** 2)) / np.sqrt(np.mean(ref**2))
            )
            assert blocked_rel < full_rel, (field, full_rel, blocked_rel)
            assert blocked_rel < 0.08, (field, blocked_rel)
