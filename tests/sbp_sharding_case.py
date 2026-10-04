"""Multi-device case of the eta-sharded nodal SBP bracket (run in a subprocess with forced host devices).

``tests/test_fci_perpendicular_sbp_sharding.py`` runs this module as a script with
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` (set before JAX is imported) and reads the JSON on the last
stdout line. Modes: ``agree`` (sharded against single-device ``sbp_bracket`` for 1/2/4 shards) and ``short`` (a plan
with fewer than 3 planes per shard is rejected).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax                                                                                       # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                                                          # noqa: E402

from drbx.geometry.nodal_layout import build_nodal_layout                                        # noqa: E402
from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData                           # noqa: E402
from drbx.native.fci_perpendicular_sbp_bracket import sbp_bracket                                # noqa: E402
from drbx.native.fci_perpendicular_sbp_sharding import (                                         # noqa: E402
    make_plane_mesh, shard_nodal_plan, sharded_sbp_bracket)
from drbx.stencils.nodal_plan import build_nodal_plan, nodal_metric_from_callable                # noqa: E402

N = 32
LEVELS = [(8, 16, 16), (16, 32, 32)]
RHO = 0.05


def metric(points):
    u, th, et = points.T
    h = np.stack([0.1 * np.sin(th), 0.2 * np.cos(et), 1.0 + 0.1 * u], -1)
    return h, u * (1.0 + 0.1 * np.cos(th) * np.cos(et)), 1.0 + 0 * u, np.zeros((len(u), 3))


def build(n_eta):
    lay = build_nodal_layout(N, LEVELS, n_eta=n_eta, inner="wall")
    return lay, build_nodal_plan(lay, nodal_metric_from_callable(lay, metric))


def run_agree(n_eta=16, shard_counts=(1, 2, 4)) -> dict:
    lay, plan = build(n_eta)
    rng = np.random.default_rng(11)
    phi = rng.standard_normal((n_eta, lay.P))
    g = rng.standard_normal((n_eta, lay.P, 2))
    bcd = SatBoundaryData(tuple(jnp.asarray(rng.standard_normal((n_eta, w[3], 2))) for w in plan.structure.walls), None)
    ref = np.asarray(jax.jit(sbp_bracket)(plan, phi, g, bcd, RHO))
    out = {}
    for s in shard_counts:
        mesh = make_plane_mesh(s)
        sharded = shard_nodal_plan(plan, s, mesh)
        got = np.asarray(jax.jit(lambda sp, ph, gg, b, r: sharded_sbp_bracket(sp, ph, gg, b, r, mesh))(
            sharded, phi, g, bcd, RHO))
        out[f"S{s}"] = float(np.abs(got - ref).max() / np.abs(ref).max())
        out[f"S{s}_bitwise"] = bool(np.array_equal(got, ref))
    out["ref_max"] = float(np.abs(ref).max())
    return out


def run_short() -> dict:
    _, plan = build(8)
    errors = {}
    for s in (4, 8):
        try:
            shard_nodal_plan(plan, s)
            errors[f"S{s}"] = None
        except ValueError as exc:
            errors[f"S{s}"] = str(exc)
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("agree", "short"))
    args = parser.parse_args(argv)
    result = run_agree() if args.mode == "agree" else run_short()
    result["devices"] = len(jax.devices())
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
