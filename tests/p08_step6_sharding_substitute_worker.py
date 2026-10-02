"""TEST-ONLY substitute worker of the P08 step-6 sharding stage (``sharding.worker_command`` is monkeypatched to this file).

Same command line and result contract as ``p08_step6_global/sharding.py --worker``; it runs the SAME ``sharding.run_checks``
(single-device references, per-shard-count lowering, gates' raw metrics, peak RSS) but on substitute data instead of the
remote re-freeze artifact:

* RHS part: the bounded N32 owner closure whose targets lie on every eta plane (``perpendicular_sharding_real_case``;
  pinned operator options, ``compact_c3`` evaluator) with the P06N variants ``main_phi_dirichlet`` and ``dirichlet_rich``;
  the metric is taken over the closure's target owners (``mask``), the other owners of the closure plan are not defined.
* phi part: the local N32 P07 Dirichlet export (``work/p08-step5-export-.../N32``) with a manufactured smooth solution and
  Dirichlet data (``perpendicular_phi_sharding_real_case.manufactured``), rtol ``1e-10``.

The export folder may be overridden with ``P08_STEP6_SHARDING_SUBSTITUTE_EXPORT``. N must be 32.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = _REPO_ROOT.parent
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax                                                                                       # noqa: E402
jax.config.update("jax_enable_x64", True)

import os                                                                                        # noqa: E402

EXPORT = Path(os.environ.get("P08_STEP6_SHARDING_SUBSTITUTE_EXPORT",
                             WORKSPACE / "work/p08-step5-export-2ff50718-20261001T165909Z-ebcbd2"))
RHS_VARIANTS = ("main_phi_dirichlet", "dirichlet_rich")
PHI_RTOL = 1.0e-10


def rhs_setup(n: int, halo: int):
    from drbx.native.fci_perpendicular_p06_operator import bc_columns
    from drbx.native.fci_perpendicular_rhs import FIELDS, PHI
    from perpendicular_sharding_real_case import OPTIONS, closure_owners
    from perpendicular_structured.reconstruction import load_context
    from p_shared import step3_gates

    raw = load_context(n, str(WORKSPACE)).ro
    owners = closure_owners(raw, 2)
    setup = step3_gates.build_setup(n, ("p06n",), owners=owners, **OPTIONS)
    closure = setup.closure
    adapter = closure.adapters["p06n"]
    values_all = np.asarray(adapter.owner_values, dtype=np.float64)
    cases = []
    for variant in RHS_VARIANTS:
        rec = adapter.reconstructions[variant]
        columns = np.asarray(rec.columns)
        values = values_all[:, columns]
        cases.append(SimpleNamespace(
            name=variant, state={f: values[:, i] for i, f in enumerate(FIELDS)}, phi=values[:, 4],
            bc=bc_columns(closure.bc["p06n"], columns), kinds=dict(zip((*FIELDS, PHI), rec.field_kinds))))
    mask = np.zeros(len(values_all), dtype=bool)
    mask[owners] = True
    return SimpleNamespace(plan=closure.plan, raw_to_owner=raw, n=n, halo=halo, mask=mask, cases=cases), len(owners)


def phi_setup(n: int):
    from drbx.native.fci_perpendicular_p07_sparse import boundary_source, load_p07_sparse
    from drbx.native.fci_perpendicular_phi_solver import phi_solver_from_operator
    from perpendicular_phi_sharding_real_case import manufactured

    gdir = EXPORT / f"N{n}"
    op = load_p07_sparse(gdir / "p07_dirichlet.npz")
    with np.load(gdir / "owner_map.npz") as z:
        raw = np.asarray(z["raw_to_owner"])
    x_true, bc = manufactured(op, raw, n)
    rhs = op.matrix @ x_true + boundary_source(op, bc)[:, 0]
    solver = phi_solver_from_operator(op, raw, n, rtol=PHI_RTOL)
    case = SimpleNamespace(name="manufactured", rhs=rhs, bc=bc, rhs_source="manufactured")
    return SimpleNamespace(solver=solver, raw_to_owner=raw, n=n, volume=np.asarray(op.owner_volume), cases=[case])


def main(argv=None) -> int:
    from drbx.native.fci_perpendicular_rhs import PerpendicularParams
    from p08_step6_global import sharding

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.worker.read_text())
    n, scfg = int(spec["n"]), spec["cfg"]["sharding"]
    if n != 32:
        raise ValueError("the substitute worker only has N32 data")
    started = time.time()
    rhs, n_targets = rhs_setup(n, int(scfg["plan_halo"]))
    phi = phi_setup(n)
    params = PerpendicularParams(rho_star=0.05, tau=1.0, diffusion={"density": 1e-2, "Te": 2e-2, "Ti": 3e-2,
                                                                    "vorticity": 4e-2})
    result = sharding.run_checks(n=n, shard_counts=spec["shard_counts"], rhs=rhs, phi=phi, params=params,
                                 phi_rtol=PHI_RTOL)
    result.update(identity=spec["identity"], source="substitute", owners=int(len(rhs.mask)), targets=n_targets,
                  wall_seconds=time.time() - started)
    result["peak_rss_gib"]["final"] = sharding.peak_rss_gib()
    args.result.write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
