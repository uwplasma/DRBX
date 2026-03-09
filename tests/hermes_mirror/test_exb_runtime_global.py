from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.legacy_hermes import div_n_bxgrad_f_b_xppm

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_global_t1.npz"


def _fixture_inputs() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(_FIXTURE, allow_pickle=False) as data:
        bc = BC2D(kind_x=2, kind_y=0, x_value=0.0, y_value=0.0, x_grad=0.0, y_grad=0.0)
        kwargs: dict[str, object] = {
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
        }
        arrays = {name: np.asarray(data[name], dtype=np.float64) for name in data.files}
    return kwargs, arrays


def test_runtime_edge_block_wrapper_improves_global_dump_backed_terms() -> None:
    kwargs, data = _fixture_inputs()
    edge_block = int(np.asarray(data["mysub"]))

    for field, ref_key in (("Ne", "term_Ne_exb"), ("Pe", "term_Pe_exb")):
        full = -div_n_bxgrad_f_b_xppm(
            jnp.asarray(data[field], dtype=jnp.float64),
            jnp.asarray(data["phi"], dtype=jnp.float64),
            **kwargs,
        )
        edge = -div_n_bxgrad_f_b_xppm(
            jnp.asarray(data[field], dtype=jnp.float64),
            jnp.asarray(data["phi"], dtype=jnp.float64),
            parallel_edge_block=edge_block,
            **kwargs,
        )
        ref = data[ref_key]

        full_rms = float(np.sqrt(np.mean((np.asarray(full) - ref) ** 2)))
        edge_rms = float(np.sqrt(np.mean((np.asarray(edge) - ref) ** 2)))
        edge_corr = float(np.corrcoef(np.asarray(edge).ravel(), ref.ravel())[0, 1])

        assert edge_rms < full_rms, (field, full_rms, edge_rms)
        assert edge_rms < 6.0e-4, (field, edge_rms)
        assert edge_corr > 0.98, (field, edge_corr)
