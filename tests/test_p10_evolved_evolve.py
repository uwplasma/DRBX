"""P10 evolved MMS, chunk C4 (RK4 evolution driver): the RK4 order, exactness with the discrete source, bitwise resume, the stop rule,
the output schema and the compile budget, on the n = 16 family-A synthetic bundle (``p10_evolved_mms.synthetic``)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from p10_evolved_mms import evolve as EV                          # noqa: E402
from p10_evolved_mms import fields as F                           # noqa: E402
from p10_evolved_mms import model as M                            # noqa: E402
from p10_evolved_mms import synthetic as syn                      # noqa: E402

# O(1) spatial terms (rho* = 0.7, a_phi = 1, D ~ 1e-2) with a slowly varying exact solution (time_scale = 0.05). The semi-discrete
# hyperbolic operator on the n = 16 synthetic bundle has imaginary rates up to ~5.7e3, and the exact semi-discrete solution is only
# reproduced up to the RK4 truncation error: measured local error of one step from the exact vorticity, 1.5e-11 / 4.5e-13 / 1.4e-14
# at dt = 4e-4 / 2e-4 / 1e-4 (dt^5, then round-off ~7e-16 from dt = 5e-5). dt = 5e-5 (|z| ~ 0.3) puts the 20-step error at round-off.
P_SLOW = F.default_params(rho_star=0.7, time_scale=0.05, a_phi=1.0, a_omega=1.0 / 0.7 ** 2, D=np.array([0.01, 0.012, 0.014, 0.008]))
DT, NSTEPS, CHUNK = 5e-5, 20, 4
T = NSTEPS * DT
FRACS = (0.5, 1.0)                                                # snapshot steps 10, 20; segments 4 4 2 2 4 4 (scan scan 1 1 scan scan)
SEGMENTS = [4, 8, 10, 12, 16, 20]
MODES = ("diffusion", "hyperbolic", "coupled")


@pytest.fixture(scope="module")
def bundle():
    return syn.synthetic_bundle()


def read_run(out: Path):
    rec = json.loads((out / "run.json").read_text())
    snaps = {}
    for f in sorted((out / "snapshots").glob("snap_*.npz")):
        with np.load(f) as z:
            snaps[f.stem] = {k: np.asarray(z[k]) for k in z.files}
    with np.load(out / "checkpoint.npz") as z:
        ckpt = {k: np.asarray(z[k]) for k in z.files}
    return rec, snaps, ckpt


@pytest.fixture(scope="module")
def runs(bundle, tmp_path_factory):
    """One discrete-source run per mode (pattern NNN-D): ``mode -> (out_dir, traces)``; ``traces`` is the number of step programs
    traced (= compiled) by that run."""
    out = {}
    for mode in MODES:
        d = tmp_path_factory.mktemp(f"run_{mode}")
        before = dict(EV.TRACES)
        rec = EV.run(bundle, mode, "discrete", P_SLOW, "NNN-D", dt=DT, T=T, out_dir=d, chunk=CHUNK, snapshot_fracs=FRACS)
        out[mode] = (d, {k: EV.TRACES[k] - before.get(k, 0) for k in ("scan", "one")}, rec)
    return out


# ----------------------------------------------------------------------------------------------------- 1. RK4 order
LAM, AMP = -1.5, 0.1


def toy_rhs(state, t, carry):
    """``dq/dt = lam q + amp cos t`` (every component)."""
    q = state.array()
    rhs = LAM * q + AMP * jnp.cos(t)
    return M.NodalState.from_array(rhs), carry, M.make_stage_info(t, q, rhs)


def toy_exact(q0, t):
    a = -LAM * AMP / (1.0 + LAM ** 2)                                      # particular solution a cos t + b sin t
    b = AMP / (1.0 + LAM ** 2)
    return np.exp(LAM * t) * (q0 - a) + a * np.cos(t) + b * np.sin(t)


def test_rk4_order_four_with_a_toy_rhs_through_run(bundle, tmp_path):
    """The driver advances with the classical RK4: for ``dq/dt = lam q + amp cos t`` the end-point error drops by 16 per halving
    of ``dt`` (``T = 1``, ``dt = T/8, T/16, T/32``; chunks of 3 steps so scans and single steps both run)."""
    q0 = np.asarray(M.exact_state(bundle, P_SLOW, 0.0))
    errs = []
    for k, n in enumerate((8, 16, 32)):
        d = tmp_path / f"toy{k}"
        rec = EV.run(bundle, "diffusion", "discrete", P_SLOW, "NNN-D", dt=1.0 / n, T=1.0, out_dir=d, chunk=3, snapshot_fracs=(1.0,),
                     rhs_fn=toy_rhs, diagnostics=False)
        assert rec["status"] == "complete" and rec["nsteps"] == n
        with np.load(d / "checkpoint.npz") as z:
            assert int(z["step"]) == n and float(z["t"]) == 1.0
            errs.append(float(np.abs(z["state"] - toy_exact(q0, 1.0)).max()))
    ratios = [errs[0] / errs[1], errs[1] / errs[2]]
    print("RK4 toy errors", errs, "ratios", ratios)
    assert all(14.0 < r < 18.0 for r in ratios), ratios
    assert errs[2] < 1e-6


def test_dt_must_divide_T(bundle, tmp_path):
    with pytest.raises(ValueError, match="does not divide"):
        EV.run(bundle, "diffusion", "discrete", P_SLOW, "NNN-D", dt=0.3, T=1.0, out_dir=tmp_path / "x", rhs_fn=toy_rhs, diagnostics=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        EV.run(bundle, "diffusion", "discrete", P_SLOW, "NNN-D", dt=0.5, T=1.0, out_dir=tmp_path / "y", rhs_fn=toy_rhs,
               diagnostics=False)                                                   # 2 steps: snapshot steps 0 / 1 / 1 / 2 / 2


# ----------------------------------------------------------------------------------------------------- 2. exactness
@pytest.mark.slow                                                                   # compiles both step programs of 3 modes (~30 s cold)
@pytest.mark.parametrize("mode", MODES)
def test_discrete_source_run_stays_on_the_exact_solution(runs, mode):
    """With the discrete source ``S_h`` the exact nodal ``q(t)`` solves the semi-discrete system, so the numerical state stays on
    it up to the RK4 truncation error (dt^5 of the fast modes, see ``DT``) and round-off: prescribed modes ``<= 1e-12`` (measured
    4e-16 diffusion, 6e-15 hyperbolic after 20 steps). Coupled: the ``psi`` solve is only exact to the CG tolerance ``rtol = 1e-11``
    (conditioned against ``Omega``); the defect of the RHS is ``~ few rtol max|R_h|`` (``max|R_h| ~ 470``), accumulating linearly in
    time: ``3 rtol max|R_h| T / max|q| ~ 1e-11``, measured 1.5e-11; the bound ``1e-10`` is ~7x the measurement."""
    out, _traces, rec = runs[mode]
    assert rec["status"] == "complete"
    _, snaps, _ = read_run(out)
    assert set(snaps) == {"snap_00", "snap_01", "snap_02"}
    np.testing.assert_array_equal(snaps["snap_00"]["q"], snaps["snap_00"]["q_exact"])        # q(0) is the exact state
    errs = [float(np.abs(s["q"] - s["q_exact"]).max() / np.abs(s["q_exact"]).max()) for s in snaps.values()]
    print(mode, "relative errors at the snapshots", errs)
    bound = 1e-10 if mode == "coupled" else 1e-12
    assert max(errs) <= bound, errs
    # the state really moved (a frozen state would trivially stay exact)
    assert float(np.abs(snaps["snap_02"]["q_exact"] - snaps["snap_00"]["q_exact"]).max()) > 1e-5


# ----------------------------------------------------------------------------------------------------- 3. resume
class _Interrupt(Exception):
    pass


def _compare(rec_a, snaps_a, ckpt_a, rec_b, snaps_b, ckpt_b):
    assert snaps_a.keys() == snaps_b.keys()
    for name in snaps_a:
        assert snaps_a[name].keys() == snaps_b[name].keys()
        for key in snaps_a[name]:
            np.testing.assert_array_equal(snaps_a[name][key], snaps_b[name][key], err_msg=f"{name}/{key}")
    for key in ckpt_a:
        np.testing.assert_array_equal(ckpt_a[key], ckpt_b[key], err_msg=key)
    strip = lambda chunks: [{k: v for k, v in c.items() if k != "wall_s"} for c in chunks]
    assert strip(rec_a["chunks"]) == strip(rec_b["chunks"])
    for key in ("status", "stop_reason", "stop_step", "dt", "T", "nsteps", "chunk", "identity", "params", "git_commit"):
        assert rec_a[key] == rec_b[key], key


@pytest.mark.slow                                                                   # uses the ``runs`` fixture (compiles, ~30 s cold)
def test_resume_is_bitwise_identical_to_the_uninterrupted_run(bundle, runs, tmp_path):
    """Coupled (the carry ``psi`` matters): interrupt after the first segment (``on_chunk`` raises after the checkpoint), resume;
    then simulate a kill between ``run.json`` and ``checkpoint.npz`` of the third segment (stale checkpoint) and resume again:
    the final state, the carry, all snapshots and the chunk records equal the uninterrupted run bitwise."""
    ref_dir, _, _ = runs["coupled"]
    ref = read_run(ref_dir)
    kw = dict(dt=DT, T=T, chunk=CHUNK, snapshot_fracs=FRACS)

    # (a) interrupted after the first segment
    d = tmp_path / "a"

    def stop_after_first(record):
        raise _Interrupt

    with pytest.raises(_Interrupt):
        EV.run(bundle, "coupled", "discrete", P_SLOW, "NNN-D", out_dir=d, on_chunk=stop_after_first, **kw)
    rec, snaps, ckpt = read_run(d)
    assert rec["status"] == "running" and int(ckpt["step"]) == 4 and set(snaps) == {"snap_00"}
    EV.run(bundle, "coupled", "discrete", P_SLOW, "NNN-D", out_dir=d, **kw)
    got = read_run(d)
    assert got[0]["status"] == "complete" and got[0]["resumes"][0]["step"] == 4
    _compare(*ref, *got)

    # (b) kill between run.json and the checkpoint: the checkpoint of segment 2 is stale while run.json lists segment 3
    d = tmp_path / "b"
    copies = []

    def keep_checkpoints(record):
        copies.append((record["step_end"], d / f"ck_{record['step_end']}.npz"))
        shutil.copy(d / "checkpoint.npz", copies[-1][1])
        if record["step_end"] == 10:
            raise _Interrupt

    with pytest.raises(_Interrupt):
        EV.run(bundle, "coupled", "discrete", P_SLOW, "NNN-D", out_dir=d, on_chunk=keep_checkpoints, **kw)
    stale = dict(copies)[8]
    shutil.copy(stale, d / "checkpoint.npz")                                        # run.json is at step 10, the checkpoint at 8
    assert json.loads((d / "run.json").read_text())["chunks"][-1]["step_end"] == 10
    EV.run(bundle, "coupled", "discrete", P_SLOW, "NNN-D", out_dir=d, **kw)
    got = read_run(d)
    _compare(*ref, *got)

    # (c) a complete run is returned as is; mismatches and resume=False are refused
    again = EV.run(bundle, "coupled", "discrete", P_SLOW, "NNN-D", out_dir=d, **kw)
    assert again["status"] == "complete" and len(again["chunks"]) == len(SEGMENTS)
    with pytest.raises(ValueError, match="cannot resume"):
        EV.run(bundle, "coupled", "discrete", P_SLOW, "NNN-D", out_dir=d, **{**kw, "dt": DT * 1.0000001, "T": T * 1.0000001})
    with pytest.raises(ValueError, match="cannot resume"):
        EV.run(bundle, "coupled", "discrete", P_SLOW._replace(a_phi=0.9), "NNN-D", out_dir=d, **kw)
    with pytest.raises(FileExistsError):
        EV.run(bundle, "coupled", "discrete", P_SLOW, "NNN-D", out_dir=d, resume=False, **kw)


# ----------------------------------------------------------------------------------------------------- 4. stop rule
K_FAIL = 6                                                                          # mid-segment of the segment of steps 5..8


def _toy_stop(kind, slope):
    """Toy right-hand sides: ``n`` falls linearly (``kind="negative"``, uniform slope ``slope``) or a NaN enters ``Ti`` from the stage
    time ``(K_FAIL - 0.25) dt`` on (``kind="nan"``: first in stage 2 of step ``K_FAIL``)."""
    def rhs_fn(state, t, carry):
        q = state.array()
        rhs = jnp.zeros_like(q)
        if kind == "negative":
            rhs = rhs.at[..., 0].set(-slope)
        else:
            rhs = rhs.at[..., 2].set(jnp.where(t > (K_FAIL - 0.25) * DT, jnp.nan, 0.0))
        return M.NodalState.from_array(rhs), carry, M.make_stage_info(t, q, rhs)

    return rhs_fn


@pytest.mark.parametrize("kind,reason", [("negative", "min n <= 0"), ("nan", "non-finite")])
def test_stop_rule_stops_after_the_chunk_with_the_first_failing_step(bundle, tmp_path, kind, reason):
    """``n`` driven below zero (slope chosen so that ``n_k = min n0 (1 - k / (K - 1/2))`` first is <= 0 at step ``K = 6``; the
    stage-4 state of step ``k`` is ``n_k``) or a NaN injected into stage 2 of step 6, in the segment of steps 5..8: the run stops
    after that segment, ``stop_step = 6``, the later segments never run, and the violating segment gets no snapshot."""
    d = tmp_path / kind
    q0 = np.asarray(M.exact_state(bundle, P_SLOW, 0.0))
    slope = q0[..., 0].min() / ((K_FAIL - 0.5) * DT)
    rhs_fn = _toy_stop(kind, slope)
    kw = dict(dt=DT, T=T, out_dir=d, chunk=CHUNK, rhs_fn=rhs_fn, diagnostics=False)
    rec = EV.run(bundle, "diffusion", "discrete", P_SLOW, "NNN-D", **kw)
    assert rec["status"] == "stopped" and reason in rec["stop_reason"], rec["stop_reason"]
    assert rec["stop_step"] == K_FAIL
    assert [c["step_end"] for c in rec["chunks"]] == [4, 8]
    last = rec["chunks"][-1]
    if kind == "negative":
        assert last["all_finite"] and last["min_n"] <= 0.0 and rec["chunks"][0]["min_n"] > 0.0
    else:
        assert not last["all_finite"] and rec["chunks"][0]["all_finite"]
    _, snaps, ckpt = read_run(d)
    assert int(ckpt["step"]) == 8 and sorted(snaps) == ["snap_00", "snap_01"]        # steps 4 (ok) but not 8 (violating segment)
    # a stopped run is not continued
    again = EV.run(bundle, "diffusion", "discrete", P_SLOW, "NNN-D", **kw)
    assert again["status"] == "stopped" and again["stop_step"] == K_FAIL and len(again["chunks"]) == 2


def test_stop_rule_on_a_non_converged_potential_solve(bundle, tmp_path):
    """An exhausted CG budget (``phi_maxit = 1``, tolerance 1e-11) in the first stage of the first step stops the run after the first
    segment with ``stop_step = 1``."""
    d = tmp_path / "cg"
    rec = EV.run(bundle, "coupled", "discrete", P_SLOW, "NNN-D", dt=DT, T=T, out_dir=d, chunk=CHUNK, snapshot_fracs=FRACS,
                 opts_override={"phi_maxit": 1}, diagnostics=False)
    assert rec["status"] == "stopped" and rec["stop_step"] == 1 and "CG" in rec["stop_reason"]
    assert [c["step_end"] for c in rec["chunks"]] == [4] and rec["chunks"][0]["all_converged"] is False
    assert rec["chunks"][0]["max_cg_iterations"] == 1


# ----------------------------------------------------------------------------------------------------- 5. schema
RUN_KEYS = {"schema", "identity", "mode", "source", "pattern", "params", "dt", "T", "nsteps", "chunk", "git_commit", "status",
            "stop_reason", "stop_step", "chunks", "timings"}
CHUNK_KEYS = {"step_end", "t_end", "max_cg_iterations", "max_cg_relative_residual", "all_converged", "min_n", "min_Te", "min_Ti",
              "all_finite", "wall_s"}
COUPLED_KEYS = {"psi", "phi", "psi_exact", "phi_exact", "psi_ctrl", "phi_ctrl"}


@pytest.mark.slow                                                                   # uses the ``runs`` fixture (compiles, ~30 s cold)
@pytest.mark.parametrize("mode", MODES)
def test_output_schema(bundle, runs, mode):
    out, _, _ = runs[mode]
    rec, snaps, ckpt = read_run(out)
    E, P = bundle.E, bundle.P
    assert RUN_KEYS <= set(rec) and rec["schema"] == "drbx.p10-run-v1"
    assert rec["status"] == "complete" and rec["stop_reason"] is None and rec["stop_step"] is None
    assert (rec["mode"], rec["source"], rec["pattern"]) == (mode, "discrete", "NNN-D")
    assert set(rec["params"]) == {"rho_star", "tau", "D", "a_phi", "time_scale", "w1", "a_omega"} and len(rec["params"]["D"]) == 4
    assert rec["dt"] == DT and rec["T"] == T and rec["nsteps"] == NSTEPS and rec["chunk"] == CHUNK
    ident = rec["identity"]
    assert ident["bundle_sha256"] == bundle.identity["bundle_sha256"] and ident["arm"] == "raw" and ident["n"] == 16
    assert [c["step_end"] for c in rec["chunks"]] == SEGMENTS
    for c in rec["chunks"]:
        assert set(c) == CHUNK_KEYS and c["all_converged"] and c["all_finite"] and c["wall_s"] > 0.0
        assert c["min_n"] > 0 and c["min_Te"] > 0 and c["min_Ti"] > 0
        assert c["t_end"] == c["step_end"] * DT
        assert mode == "coupled" or c["max_cg_iterations"] == 0                      # no solve in the prescribed modes
        assert c["max_cg_relative_residual"] <= 1e-11
    assert rec["chunks"][-1]["t_end"] == pytest.approx(T, rel=1e-12)
    assert {"total_wall_s", "snapshot_wall_s"} <= set(rec["timings"])
    # snapshots: 00 (t = 0), 01 (50 %), 02 (100 %)
    assert sorted(snaps) == ["snap_00", "snap_01", "snap_02"]
    for name, step in (("snap_00", 0), ("snap_01", 10), ("snap_02", 20)):
        s = snaps[name]
        assert int(s["step"]) == step and float(s["t"]) == step * DT
        base = {"t", "step", "q", "q_exact", "tau"}
        assert set(s) >= base and (set(s) >= COUPLED_KEYS) == (mode == "coupled")
        for key in ("q", "q_exact", "tau"):
            assert s[key].shape == (E, P, 4) and s[key].dtype == np.float64
        if mode == "coupled":
            for key in COUPLED_KEYS:
                assert s[key].shape == (E, P)
            # the exact-state potential of a solve with sigma_h is the exact psi (cold start, CG tolerance); the elliptic control
            # solve with the continuum sigma differs from it by the Laplacian truncation
            assert np.abs(s["psi"] - s["psi_exact"]).max() <= 1e-8 * np.abs(s["psi_exact"]).max()
            assert np.abs(s["psi_ctrl"] - s["psi_exact"]).max() > 1e-6 * np.abs(s["psi_exact"]).max()
            np.testing.assert_allclose(s["phi_exact"], s["psi_exact"] - rec["params"]["tau"] * s["q_exact"][..., 0] * s["q_exact"][..., 2],
                                       rtol=1e-13, atol=1e-13)
        assert np.abs(s["tau"]).max() > 1e-6                                         # the truncation is visible
    # checkpoint
    assert set(ckpt) == {"state", "carry", "t", "step", "digest"} and int(ckpt["step"]) == NSTEPS
    assert ckpt["state"].shape == (E, P, 4) and ckpt["carry"].shape == (E, P)
    np.testing.assert_array_equal(ckpt["state"], snaps["snap_02"]["q"])
    assert str(ckpt["digest"]) == EV._digest(ckpt["state"])
    # geometry
    with np.load(out / "geometry.npz") as g:
        assert set(g.files) == {"H", "region_names", "region_masks", "n", "n_eta"}
        from drbx.native.fci_perpendicular_sbp_norms import h_weights, ring_region_masks

        np.testing.assert_array_equal(g["H"], np.asarray(h_weights(bundle.ctx.plan)))
        masks = ring_region_masks(bundle.layout, bundle.E)
        assert list(g["region_names"]) == list(masks) and g["region_masks"].shape == (len(masks), E, P)
        assert g["region_masks"].dtype == bool and int(g["n"]) == 16 and int(g["n_eta"]) == E
        assert g["region_masks"][list(g["region_names"]).index("all")].all()
    # no temporary files left behind
    assert not list(out.rglob("*.tmp"))


# ----------------------------------------------------------------------------------------------------- 6. compile budget
@pytest.mark.slow                                                                   # uses the ``runs`` fixture (compiles, ~30 s cold)
@pytest.mark.parametrize("mode", MODES)
def test_at_most_two_step_programs_are_compiled_per_run(runs, mode):
    """The run of 20 steps (segments 4 4 2 2 4 4) uses the ``chunk``-step scan program and the 1-step program, each traced once."""
    _out, traces, _ = runs[mode]
    assert traces["scan"] <= 1 and traces["one"] <= 1
    cfg = M.stage_config(mode, "discrete", "NNN-D")
    scan_prog, one_prog = EV._programs(M.stage_rhs_unbound(cfg), CHUNK)
    assert scan_prog._cache_size() == 1 and one_prog._cache_size() == 1


def test_toy_run_compiles_each_program_once(bundle, tmp_path):
    """A fresh right-hand side: exactly one trace of each program over the whole run, however many segments and snapshots."""
    before = dict(EV.TRACES)
    rec = EV.run(bundle, "diffusion", "discrete", P_SLOW, "NNN-D", dt=DT, T=T, out_dir=tmp_path / "t", chunk=CHUNK,
                 rhs_fn=lambda s, t, c: toy_rhs(s, t, c), diagnostics=False)
    assert rec["status"] == "complete"
    assert EV.TRACES["scan"] - before.get("scan", 0) == 1 and EV.TRACES["one"] - before.get("one", 0) == 0   # 5 snapshot-aligned scans
    assert [c["step_end"] for c in rec["chunks"]] == [4, 8, 12, 16, 20]


# ----------------------------------------------------------------------------------------------------- command line
def test_cli_round_trip_with_a_saved_synthetic_bundle(bundle, tmp_path, capsys):
    """``python -m p10_evolved_mms.evolve --bundle DIR ...`` on a saved (synthetic) bundle: loads it, runs 10 steps of the diffusion
    mode with the continuum source (default configuration parameters except the overrides), exit code 0, complete run record."""
    from p10_evolved_mms import bundle as bundle_mod

    bdir = tmp_path / "bundle"
    bundle_mod.save_bundle(bundle, bdir)
    out = tmp_path / "run"
    code = EV.main(["--bundle", str(bdir), "--mode", "diffusion", "--source", "continuum", "--pattern", "DDDD", "--dt", "5e-5",
                    "--T", "5e-4", "--out", str(out), "--chunk", "4", "--rho-star", "0.7", "--time-scale", "0.05", "--a-phi", "1.0"])
    assert code == 0
    rec, snaps, ckpt = read_run(out)
    assert rec["status"] == "complete" and rec["nsteps"] == 10 and rec["pattern"] == "DDDD" and rec["source"] == "continuum"
    assert rec["params"]["rho_star"] == 0.7 and rec["params"]["time_scale"] == 0.05 and rec["params"]["D"][0] == 1.0e-5
    assert rec["identity"]["bundle_sha256"] == bundle.identity["bundle_sha256"]
    assert sorted(snaps) == [f"snap_{i:02d}" for i in range(6)] and int(ckpt["step"]) == 10
    assert '"status": "complete"' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        EV.main(["--bundle", str(bdir), "--mode", "diffusion", "--source", "continuum", "--pattern", "DDDD", "--dt", "5e-5",
                 "--T", "5e-4", "--out", str(out), "--chunk", "4", "--no-resume"])
