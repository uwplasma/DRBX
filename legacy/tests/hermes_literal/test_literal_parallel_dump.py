from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxdrb.hermes_literal import div_par_centered, div_par_mod

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_parallel_local_rank0_t1.npz"
)


def _physical_and_ghost(arr: np.ndarray, pstart: int, pend: int, xstart: int, xend: int) -> tuple:
    phys = arr[pstart : pend + 1, xstart : xend + 1, :]
    glow = arr[pstart - 1, xstart : xend + 1, :]
    ghigh = arr[pend + 1, xstart : xend + 1, :]
    return phys, glow, ghigh


def _physical_metric(
    arr: np.ndarray, pstart: int, pend: int, xstart: int, xend: int, nz: int
) -> np.ndarray:
    base = arr[pstart : pend + 1, xstart : xend + 1]
    return np.broadcast_to(base[..., None], (base.shape[0], base.shape[1], nz))


def test_parallel_dump_backed_density_pressure_and_jpar_terms() -> None:
    with np.load(_FIXTURE, allow_pickle=False) as data:
        pstart = int(np.asarray(data["pstart"]))
        pend = int(np.asarray(data["pend"]))
        xstart = int(np.asarray(data["xstart"]))
        xend = int(np.asarray(data["xend"]))

        J = np.asarray(data["J"], dtype=np.float64)
        dy = np.asarray(data["dy"], dtype=np.float64)
        g22 = np.asarray(data["g_22"], dtype=np.float64)
        ne = np.asarray(data["Ne"], dtype=np.float64)
        pe = np.asarray(data["Pe"], dtype=np.float64)
        te = np.asarray(data["Te"], dtype=np.float64)
        nd = np.asarray(data["Nd+"], dtype=np.float64)
        td = np.asarray(data["Td+"], dtype=np.float64)
        ve = np.asarray(data["Ve"], dtype=np.float64)
        nve = np.asarray(data["NVe"], dtype=np.float64)
        nvd = np.asarray(data["NVd+"], dtype=np.float64)

        term_ne_par = np.asarray(data["term_Ne_par"], dtype=np.float64)
        term_pe_par = np.asarray(data["term_Pe_par"], dtype=np.float64)
        term_vort_jpar = np.asarray(data["term_Vort_jpar"], dtype=np.float64)

    me_hat = 0.0005446623093681916
    mi_hat = 2.0

    ne_phys, ne_glow, ne_ghigh = _physical_and_ghost(ne, pstart, pend, xstart, xend)
    pe_phys, pe_glow, pe_ghigh = _physical_and_ghost(pe, pstart, pend, xstart, xend)
    te_phys, _, _ = _physical_and_ghost(te, pstart, pend, xstart, xend)
    nd_phys, _, _ = _physical_and_ghost(nd, pstart, pend, xstart, xend)
    td_phys, _, _ = _physical_and_ghost(td, pstart, pend, xstart, xend)
    ve_phys, ve_glow, ve_ghigh = _physical_and_ghost(ve, pstart, pend, xstart, xend)
    nve_phys, nve_glow, nve_ghigh = _physical_and_ghost(nve, pstart, pend, xstart, xend)
    nvd_phys, nvd_glow, nvd_ghigh = _physical_and_ghost(nvd, pstart, pend, xstart, xend)
    ne_term_phys, _, _ = _physical_and_ghost(term_ne_par, pstart, pend, xstart, xend)
    pe_term_phys, _, _ = _physical_and_ghost(term_pe_par, pstart, pend, xstart, xend)
    vort_term_phys, _, _ = _physical_and_ghost(term_vort_jpar, pstart, pend, xstart, xend)
    J_phys = _physical_metric(J, pstart, pend, xstart, xend, ne_phys.shape[-1])
    dy_phys = _physical_metric(dy, pstart, pend, xstart, xend, ne_phys.shape[-1])
    g22_phys = _physical_metric(g22, pstart, pend, xstart, xend, ne_phys.shape[-1])

    total_pressure = pe_phys + nd_phys * td_phys
    total_density = me_hat * ne_phys + mi_hat * nd_phys
    sound_speed = np.sqrt(total_pressure / np.maximum(total_density, 1e-10))
    fastest_wave = np.maximum(np.sqrt(te_phys / me_hat), np.sqrt(td_phys / mi_hat))
    fastest_wave = np.maximum(fastest_wave, sound_speed)

    ne_literal = -div_par_mod(
        jnp.asarray(ne_phys),
        jnp.asarray(ve_phys),
        jnp.asarray(fastest_wave),
        dz=1.0,
        dy=jnp.asarray(dy_phys),
        limiter="mc",
        J=jnp.asarray(J_phys),
        gpar=jnp.asarray(g22_phys),
        ghost_low_f=jnp.asarray(ne_glow),
        ghost_high_f=jnp.asarray(ne_ghigh),
        ghost_low_v=jnp.asarray(ve_glow),
        ghost_high_v=jnp.asarray(ve_ghigh),
    )
    pe_literal = -(5.0 / 3.0) * div_par_mod(
        jnp.asarray(pe_phys),
        jnp.asarray(ve_phys),
        jnp.asarray(fastest_wave),
        dz=1.0,
        dy=jnp.asarray(dy_phys),
        limiter="mc",
        J=jnp.asarray(J_phys),
        gpar=jnp.asarray(g22_phys),
        ghost_low_f=jnp.asarray(pe_glow),
        ghost_high_f=jnp.asarray(pe_ghigh),
        ghost_low_v=jnp.asarray(ve_glow),
        ghost_high_v=jnp.asarray(ve_ghigh),
    )

    jpar = nvd_phys / mi_hat - nve_phys / me_hat
    jpar_glow = nvd_glow / mi_hat - nve_glow / me_hat
    jpar_ghigh = nvd_ghigh / mi_hat - nve_ghigh / me_hat
    vort_literal = div_par_centered(
        jnp.asarray(jpar),
        dz=1.0,
        dy=jnp.asarray(dy_phys),
        J=jnp.asarray(J_phys),
        gpar=jnp.asarray(g22_phys),
        ghost_low=jnp.asarray(jpar_glow),
        ghost_high=jnp.asarray(jpar_ghigh),
    )

    ne_diff = np.asarray(ne_literal) - ne_term_phys
    pe_diff = np.asarray(pe_literal) - pe_term_phys
    vort_diff = np.asarray(vort_literal) - vort_term_phys

    ne_rms = float(np.sqrt(np.mean(ne_diff * ne_diff)))
    pe_rms = float(np.sqrt(np.mean(pe_diff * pe_diff)))
    vort_rms = float(np.sqrt(np.mean(vort_diff * vort_diff)))

    assert ne_rms < 6e-4
    assert pe_rms < 8e-4
    assert vort_rms < 5e-4
