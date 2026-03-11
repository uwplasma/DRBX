from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxdrb.legacy_hermes.parallel import div_par_centered, div_par_mod
from jaxdrb.legacy_hermes.sheath import build_parallel_sheath_state

_FIXTURE_RANK0 = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_parallel_local_rank0_t1.npz"
)
_FIXTURE_RANK5 = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_parallel_local_rank5_t1.npz"
)

_ME_HAT = 0.0005446623093681916
_MI_HAT = 2.0


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


def _load_parallel_fixture(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        pstart = int(np.asarray(data["pstart"]))
        pend = int(np.asarray(data["pend"]))
        xstart = int(np.asarray(data["xstart"]))
        xend = int(np.asarray(data["xend"]))

        ne, ne_glow, ne_ghigh = _physical_and_ghost(
            np.asarray(data["Ne"], dtype=np.float64), pstart, pend, xstart, xend
        )
        te, te_glow, te_ghigh = _physical_and_ghost(
            np.asarray(data["Te"], dtype=np.float64), pstart, pend, xstart, xend
        )
        pe, pe_glow, pe_ghigh = _physical_and_ghost(
            np.asarray(data["Pe"], dtype=np.float64), pstart, pend, xstart, xend
        )
        nd, nd_glow, nd_ghigh = _physical_and_ghost(
            np.asarray(data["Nd+"], dtype=np.float64), pstart, pend, xstart, xend
        )
        td, td_glow, td_ghigh = _physical_and_ghost(
            np.asarray(data["Td+"], dtype=np.float64), pstart, pend, xstart, xend
        )
        phi, phi_glow, phi_ghigh = _physical_and_ghost(
            np.asarray(data["phi"], dtype=np.float64), pstart, pend, xstart, xend
        )
        ve, ve_glow, ve_ghigh = _physical_and_ghost(
            np.asarray(data["Ve"], dtype=np.float64), pstart, pend, xstart, xend
        )
        nve, nve_glow, nve_ghigh = _physical_and_ghost(
            np.asarray(data["NVe"], dtype=np.float64), pstart, pend, xstart, xend
        )
        nvd, nvd_glow, nvd_ghigh = _physical_and_ghost(
            np.asarray(data["NVd+"], dtype=np.float64), pstart, pend, xstart, xend
        )

        J = _physical_metric(
            np.asarray(data["J"], dtype=np.float64), pstart, pend, xstart, xend, ne.shape[-1]
        )
        dy = _physical_metric(
            np.asarray(data["dy"], dtype=np.float64), pstart, pend, xstart, xend, ne.shape[-1]
        )
        g22 = _physical_metric(
            np.asarray(data["g_22"], dtype=np.float64), pstart, pend, xstart, xend, ne.shape[-1]
        )

        return {
            "lower_open": bool(np.asarray(data["lower_boundary_open"])),
            "upper_open": bool(np.asarray(data["upper_boundary_open"])),
            "Ne": ne,
            "Ne_glow": ne_glow,
            "Ne_ghigh": ne_ghigh,
            "Te": te,
            "Te_glow": te_glow,
            "Te_ghigh": te_ghigh,
            "Pe": pe,
            "Pe_glow": pe_glow,
            "Pe_ghigh": pe_ghigh,
            "Nd": nd,
            "Nd_glow": nd_glow,
            "Nd_ghigh": nd_ghigh,
            "Td": td,
            "Td_glow": td_glow,
            "Td_ghigh": td_ghigh,
            "phi": phi,
            "phi_glow": phi_glow,
            "phi_ghigh": phi_ghigh,
            "Ve": ve,
            "Ve_glow": ve_glow,
            "Ve_ghigh": ve_ghigh,
            "NVe": nve,
            "NVe_glow": nve_glow,
            "NVe_ghigh": nve_ghigh,
            "NVd": nvd,
            "NVd_glow": nvd_glow,
            "NVd_ghigh": nvd_ghigh,
            "J": J,
            "dy": dy,
            "g22": g22,
            "term_Ne_par": _physical_and_ghost(
                np.asarray(data["term_Ne_par"], dtype=np.float64), pstart, pend, xstart, xend
            )[0],
            "term_Pe_par": _physical_and_ghost(
                np.asarray(data["term_Pe_par"], dtype=np.float64), pstart, pend, xstart, xend
            )[0],
            "term_Vort_jpar": _physical_and_ghost(
                np.asarray(data["term_Vort_jpar"], dtype=np.float64), pstart, pend, xstart, xend
            )[0],
        }


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _build_state(fixture: dict[str, object]):
    nd = np.asarray(fixture["Nd"], dtype=np.float64)
    nvd = np.asarray(fixture["NVd"], dtype=np.float64)
    vi = nvd / np.maximum(_MI_HAT * nd, 1.0e-12)
    return build_parallel_sheath_state(
        n_e=jnp.asarray(fixture["Ne"]),
        Te=jnp.asarray(fixture["Te"]),
        pe=jnp.asarray(fixture["Pe"]),
        phi=jnp.asarray(fixture["phi"]),
        v_e=jnp.asarray(fixture["Ve"]),
        n_i=jnp.asarray(nd),
        Ti=jnp.asarray(fixture["Td"]),
        pi=jnp.asarray(nd * np.asarray(fixture["Td"], dtype=np.float64)),
        v_i=jnp.asarray(vi),
        me_hat=_ME_HAT,
        ion_mass=_MI_HAT,
        nve=jnp.asarray(fixture["NVe"]),
        nvi=jnp.asarray(fixture["NVd"]),
    )


def test_parallel_sheath_state_matches_dump_backed_open_end_guards() -> None:
    rank0 = _load_parallel_fixture(_FIXTURE_RANK0)
    rank5 = _load_parallel_fixture(_FIXTURE_RANK5)
    low = _build_state(rank0)
    high = _build_state(rank5)

    assert rank0["lower_open"] is True
    assert rank5["upper_open"] is True

    assert _rms(low.n_ghost_low, rank0["Ne_glow"]) < 9.0e-4
    assert _rms(low.Te_ghost_low, rank0["Te_glow"]) < 4.0e-4
    assert _rms(low.pe_ghost_low, rank0["Pe_glow"]) < 2.0e-6
    assert _rms(low.phi_ghost_low, rank0["phi_glow"]) < 1.0e-5
    assert _rms(low.ve_ghost_low, rank0["Ve_glow"]) < 7.0e-3
    assert _rms(low.nve_ghost_low, rank0["NVe_glow"]) < 4.0e-6
    assert _rms(low.nvi_ghost_low, rank0["NVd_glow"]) < 2.0e-3
    assert _rms(low.j_ghost_low, rank0["NVd_glow"] / _MI_HAT - rank0["NVe_glow"] / _ME_HAT) < 5.0e-3

    assert _rms(high.n_ghost_high, rank5["Ne_ghigh"]) < 3.0e-3
    assert _rms(high.Te_ghost_high, rank5["Te_ghigh"]) < 1.0e-3
    assert _rms(high.pe_ghost_high, rank5["Pe_ghigh"]) < 3.0e-6
    assert _rms(high.phi_ghost_high, rank5["phi_ghigh"]) < 1.0e-5
    assert _rms(high.ve_ghost_high, rank5["Ve_ghigh"]) < 3.0e-2
    assert _rms(high.nve_ghost_high, rank5["NVe_ghigh"]) < 2.0e-5
    assert _rms(high.nvi_ghost_high, rank5["NVd_ghigh"]) < 5.0e-3
    assert (
        _rms(high.j_ghost_high, rank5["NVd_ghigh"] / _MI_HAT - rank5["NVe_ghigh"] / _ME_HAT)
        < 2.0e-2
    )


def test_parallel_mirror_operator_matches_hermes_with_reconstructed_open_sheath_guards() -> None:
    for fixture_path in (_FIXTURE_RANK0, _FIXTURE_RANK5):
        fixture = _load_parallel_fixture(fixture_path)
        state = _build_state(fixture)

        total_pressure = np.asarray(fixture["Pe"], dtype=np.float64) + np.asarray(
            fixture["Nd"], dtype=np.float64
        ) * np.asarray(fixture["Td"], dtype=np.float64)
        total_density = _ME_HAT * np.asarray(
            fixture["Ne"], dtype=np.float64
        ) + _MI_HAT * np.asarray(fixture["Nd"], dtype=np.float64)
        sound_speed = np.sqrt(total_pressure / np.maximum(total_density, 1.0e-10))
        fastest_wave = np.maximum(
            np.sqrt(np.asarray(fixture["Te"], dtype=np.float64) / _ME_HAT),
            np.sqrt(np.asarray(fixture["Td"], dtype=np.float64) / _MI_HAT),
        )
        fastest_wave = np.maximum(fastest_wave, sound_speed)

        ne_glow = np.asarray(fixture["Ne_glow"], dtype=np.float64)
        ne_ghigh = np.asarray(fixture["Ne_ghigh"], dtype=np.float64)
        pe_glow = np.asarray(fixture["Pe_glow"], dtype=np.float64)
        pe_ghigh = np.asarray(fixture["Pe_ghigh"], dtype=np.float64)
        ve_glow = np.asarray(fixture["Ve_glow"], dtype=np.float64)
        ve_ghigh = np.asarray(fixture["Ve_ghigh"], dtype=np.float64)
        j_glow = (
            np.asarray(fixture["NVd_glow"], dtype=np.float64) / _MI_HAT
            - np.asarray(fixture["NVe_glow"], dtype=np.float64) / _ME_HAT
        )
        j_ghigh = (
            np.asarray(fixture["NVd_ghigh"], dtype=np.float64) / _MI_HAT
            - np.asarray(fixture["NVe_ghigh"], dtype=np.float64) / _ME_HAT
        )

        if fixture["lower_open"]:
            ne_glow = np.asarray(state.n_ghost_low)
            pe_glow = np.asarray(state.pe_ghost_low)
            ve_glow = np.asarray(state.ve_ghost_low)
            j_glow = np.asarray(state.j_ghost_low)
        if fixture["upper_open"]:
            ne_ghigh = np.asarray(state.n_ghost_high)
            pe_ghigh = np.asarray(state.pe_ghost_high)
            ve_ghigh = np.asarray(state.ve_ghost_high)
            j_ghigh = np.asarray(state.j_ghost_high)

        ne_mirror = -div_par_mod(
            jnp.asarray(fixture["Ne"]),
            jnp.asarray(fixture["Ve"]),
            jnp.asarray(fastest_wave),
            dz=1.0,
            dy=jnp.asarray(fixture["dy"]),
            limiter="mc",
            J=jnp.asarray(fixture["J"]),
            gpar=jnp.asarray(fixture["g22"]),
            ghost_low_f=jnp.asarray(ne_glow),
            ghost_high_f=jnp.asarray(ne_ghigh),
            ghost_low_v=jnp.asarray(ve_glow),
            ghost_high_v=jnp.asarray(ve_ghigh),
        )
        pe_mirror = -(5.0 / 3.0) * div_par_mod(
            jnp.asarray(fixture["Pe"]),
            jnp.asarray(fixture["Ve"]),
            jnp.asarray(fastest_wave),
            dz=1.0,
            dy=jnp.asarray(fixture["dy"]),
            limiter="mc",
            J=jnp.asarray(fixture["J"]),
            gpar=jnp.asarray(fixture["g22"]),
            ghost_low_f=jnp.asarray(pe_glow),
            ghost_high_f=jnp.asarray(pe_ghigh),
            ghost_low_v=jnp.asarray(ve_glow),
            ghost_high_v=jnp.asarray(ve_ghigh),
        )
        jpar = (
            np.asarray(fixture["NVd"], dtype=np.float64) / _MI_HAT
            - np.asarray(fixture["NVe"], dtype=np.float64) / _ME_HAT
        )
        vort_mirror = div_par_centered(
            jnp.asarray(jpar),
            dz=1.0,
            dy=jnp.asarray(fixture["dy"]),
            J=jnp.asarray(fixture["J"]),
            gpar=jnp.asarray(fixture["g22"]),
            ghost_low=jnp.asarray(j_glow),
            ghost_high=jnp.asarray(j_ghigh),
        )

        assert _rms(ne_mirror, fixture["term_Ne_par"]) < 4.0e-4
        assert _rms(pe_mirror, fixture["term_Pe_par"]) < 3.0e-4
        assert _rms(vort_mirror, fixture["term_Vort_jpar"]) < 2.0e-5
