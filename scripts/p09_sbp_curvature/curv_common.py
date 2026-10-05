"""Shared helpers of the P09 M6 curvature campaign (paths, thread defaults, M3 harness imports, field-set evaluation)."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("NPROC", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
M3 = SCRIPTS / "p09_sbp_bracket"
WORKSPACE = HERE.parents[2]
for _p in (str(SCRIPTS), str(M3), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                                      # noqa: E402

CONFIG = json.loads((HERE / "configuration.json").read_text())
TAU = float(CONFIG["tau"])
PSI = CONFIG["psi"]
METHOD = CONFIG["absolute_method"]
GRIDS = tuple(CONFIG["resolutions"])
VARIANTS = CONFIG["variants"]
OUT = WORKSPACE / CONFIG["output"]
ARM_ROOT = {arm: WORKSPACE / rel for arm, rel in CONFIG["arms"].items()}
SLOTS = ("n", "Te", "Ti", "omega", "phi")
EQUATIONS = SLOTS[:4]

import references as M3R                                                                 # noqa: E402  (M3 harness, import only)

log, peak_rss_gib, layout_points, load_layout_metric = M3R.log, M3R.peak_rss_gib, M3R.layout_points, M3R.load_layout_metric


def arm_dir(arm: str, n: int) -> Path:
    return OUT / arm / f"N{n}"


def load_plan(arm: str, n: int):
    from drbx.stencils.nodal_plan import build_nodal_plan
    layout, metric, meta = load_layout_metric(n, ARM_ROOT[arm])
    return layout, metric, meta, build_nodal_plan(layout, metric)


def eta_filter_option(meta: dict):
    f = meta.get("eta_filter")
    return None if f is None else {k: v for k, v in f.items() if k != "arm_sha256"}


class CaseSet:
    """One catalogue case: five fields ``(n, Te, Ti, omega, phi)`` with values and logical gradients at points."""

    def __init__(self, catalogue, name, evaluator, gate_eqs, constant=False, group="gate"):
        self.catalogue, self.name, self.evaluator = catalogue, name, evaluator
        self.gate_eqs, self.constant, self.group = tuple(gate_eqs), constant, group

    @property
    def label(self) -> str:
        return f"{self.catalogue}/{self.name}"

    def evaluate(self, points, chunk=16384):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        vs, gs = [], []
        for s in range(0, len(pts), chunk):
            v, g = self.evaluator(pts[s:s + chunk])
            vs.append(np.asarray(v, dtype=np.float64)); gs.append(np.asarray(g, dtype=np.float64))
        return np.concatenate(vs, axis=1), np.concatenate(gs, axis=1)


def lean_reference():
    """The frozen MMS reference object without its geometry: ``_fields_raw`` is purely analytic (it reads only ``tau``,
    ``polarization_variable``, ``enable_generalized_potential``, the generalized-potential terms and ``eta_period``), so
    ``ContinuumMmsReference`` built with the p07 ``reference`` options and no evaluators returns the frozen reference's fields
    bit for bit (checked against the frozen provider at N32) at ~0.2 GB instead of ~1.2 GB."""
    sys.path.insert(0, str(SCRIPTS.parent))
    from hsx_mms_continuum_reference import ContinuumMmsReference

    return ContinuumMmsReference(None, None, 1.0, tau=1.0, mi_over_me=1836.0, rho_star=1.0, Ve_nu=0.0, perp_diffusion=1e-5,
                                 enable_generalized_potential=True)


def p06_cases(ref=None, which=None):
    """The four P06 campaign states. ``corrected_frozen_mms`` needs the frozen reference ``ref``."""
    from p06_structured_global import numerics as num

    period = float(ref.eta_period) if ref is not None else float(CONFIG["eta_period"])
    t = float(CONFIG["time"])

    class _R:
        eta_period = period

    out = []
    for name in num.FIELD_NAMES:
        if which is not None and name not in which:
            continue
        if name == "corrected_frozen_mms":
            if ref is None:
                continue
            ev = (lambda p, nm=name: num._evaluate_fields(nm, ref, p, t))
        else:
            ev = (lambda p, nm=name: num._evaluate_fields(nm, _R, p, t))
        out.append(CaseSet("p06", name, ev, EQUATIONS))
    return out


def p06n_cases():
    import p06n_field_derived_global.fields as p06f

    cat = json.loads((SCRIPTS / "p06n_field_derived_global" / "p06n_catalogue.json").read_text())
    period = float(CONFIG["eta_period"])
    out = []
    for case, spec in cat["cases"].items():
        names = [spec[s].split(":")[0] for s in SLOTS]
        const = case.startswith("control_constant")

        def ev(q, names=tuple(names)):
            vs, gs = [], []
            for nm in names:
                v, g, _H = p06f.evaluate(None, q, nm, period)
                vs.append(np.asarray(v, dtype=np.float64).reshape(len(q))); gs.append(np.asarray(g, dtype=np.float64).reshape(len(q), 3))
            return np.stack(vs), np.stack(gs)

        out.append(CaseSet("p06n", case, ev, EQUATIONS[:3], constant=const, group="control" if const else "gate"))
    return out


def all_cases(ref=None):
    return p06_cases(ref) + p06n_cases()


def expected_labels():
    cat = json.loads((SCRIPTS / "p06n_field_derived_global" / "p06n_catalogue.json").read_text())
    return [f"p06/{n}" for n in CONFIG["catalogues"]["p06"]["cases"]] + [f"p06n/{c}" for c in cat["cases"]]


def chunked_apply(fn, plan, q, wall, n_chunks: int = 4, halo: int = 3):
    """Apply a jitted ``fn(plan_chunk, F_ext, q_ext, wall_chunk, B_ext) -> (Ec, P, 4)`` of ``sbp_curvature_ext`` plane chunk by plane chunk.

    Every operation is local in the eta planes up to the halo, so the chunks (each with ``halo`` periodic neighbour planes of
    ``F`` and ``q``) reproduce the full operator; only the temporaries shrink by ``n_chunks``.
    """
    import dataclasses
    import jax.numpy as jnp

    from drbx.native.fci_perpendicular_sbp_curvature import curvature_flux

    E = plan.jac.shape[0]
    assert E % n_chunks == 0
    F = np.asarray(curvature_flux(plan))
    Bfull = np.asarray(plan.B)
    q = np.asarray(q)
    step = E // n_chunks
    out = []
    for c in range(n_chunks):
        sl = slice(c * step, (c + 1) * step)
        idx = np.arange(c * step - halo, (c + 1) * step + halo) % E
        sub = dataclasses.replace(plan, jac=plan.jac[sl], h=plan.h[sl], B=plan.B[sl], K=plan.K[sl], Hp=plan.Hp[sl],
                                  core_Ginv=None if plan.core_Ginv is None else plan.core_Ginv[sl])
        out.append(np.asarray(fn(sub, jnp.asarray(F[idx]), jnp.asarray(q[idx]), jnp.asarray(np.asarray(wall)[sl]),
                          jnp.asarray(Bfull[idx]))))
    return np.concatenate(out, axis=0)


def case_file(name: str) -> str:
    return name.replace("/", "__")
