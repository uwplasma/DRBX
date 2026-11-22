#!/usr/bin/env python3
"""Record and compare deterministic reduced FCI cases.

    python scripts/fci_gate.py record OUT [--skip-pytest] [--cases a,b]
    python scripts/fci_gate.py compare BASE NEW

``record`` runs from the checkout on ``PYTHONPATH`` and writes pytest counts,
source line counts and case times to ``summary.json``. Each case writes its
arrays to a separate ``.npz``. ``compare`` checks the saved arrays for bitwise
equality and compares pytest counts when recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LINE_COUNT_PACKAGES = ("geometry", "native", "fci_nodal")


# ---------------------------------------------------------------- reduced cases
def _tests_import():
    for path in (str(REPO), str(REPO / "tests")):
        if path not in sys.path:
            sys.path.insert(0, path)


def case_rotating_ellipse_mms() -> dict:
    """Rotating-ellipse FCI grad_parallel (direct and traced maps) on the MMS field."""
    _tests_import()
    import test_rotating_ellipse_fci as t
    from drbx.geometry import build_rotating_ellipse_geometry
    from drbx.native.fci_operators import grad_parallel_op_direct, grad_parallel_op_fci

    geometry = build_rotating_ellipse_geometry((16, 16, 16), **t._GEOMETRY_KWARGS)
    traced = build_rotating_ellipse_geometry(
        (16, 16, 16), construct_fci_maps=True, map_substeps=8, **t._GEOMETRY_KWARGS
    )
    return {
        "direct": grad_parallel_op_direct(t._direct_coordinate_stencil(geometry), geometry),
        "fci": grad_parallel_op_fci(t._field_line_stencil(traced), traced),
    }


def case_shifted_torus_2field() -> dict:
    _tests_import()
    import test_mms_shifted_torus_2_field as t

    geometry = t.build_shifted_torus_2field_geometry((12, 12, 12))
    state, times, density, v_parallel = t.simulate_mms_2field_shifted_torus(
        geometry, final_time=0.004, timestep=0.001
    )
    return {"state": state, "times": times, "density": density, "v_parallel": v_parallel}


def case_shifted_torus_eb() -> dict:
    _tests_import()
    import test_mms_shifted_torus_EB as t

    context = t.build_shifted_torus_eb_mms_context((6, 8, 6))
    state, final_time = t.simulate_mms_shifted_torus_eb(context, final_time=0.002, num_steps=2)
    return {"state": state, "final_time": final_time}


def _four_field_blob(geometry) -> dict:
    _tests_import()
    import test_rotating_ellipse_filament as t
    from drbx.native import Fci4FieldBlobParameters

    parameters = Fci4FieldBlobParameters(
        rho_star=1.0, phi_inversion_tol=5.0e-5, phi_inversion_maxiter=100, phi_inversion_restart=200
    )
    state, *history = t.simulate_shifted_torus_4field_blob(
        geometry,
        t._build_blob_initial_state(geometry),
        t._build_free_decay_boundary_conditions(geometry, 0.0),
        parameters=parameters,
        final_time=3 * 2.0e-3,
        timestep=2.0e-3,
    )
    return {"state": state, "history": tuple(history)}


def case_shifted_torus_4field() -> dict:
    _tests_import()
    import test_mms_shifted_torus_4_field as t

    return _four_field_blob(t.build_shifted_torus_4field_geometry((10, 10, 10)))


def case_shifted_torus_4field_mms() -> dict:
    """One RK4 step of the 4-field MMS driver (8^3), incl. its end-of-step phi solve."""
    _tests_import()
    import test_mms_shifted_torus_4_field as t

    geometry = t.build_shifted_torus_4field_geometry((8, 8, 8))
    state, *history = t.simulate_mms_shifted_torus_4field(geometry, timestep=1.0e-3, final_time=1.0e-3)
    return {"state": state, "history": tuple(history)}


def case_rotating_ellipse_4field() -> dict:
    _tests_import()
    import test_rotating_ellipse_filament as t

    return _four_field_blob(t._rotating_ellipse_geometry())


def case_island_divertor() -> dict:
    from drbx.geometry import build_island_divertor_geometry
    from drbx.native.stellarator_turbulence import run_stellarator_turbulence

    geometry = build_island_divertor_geometry((12, 16, 8), open_field_line_masks=True, mask_max_transits=20)
    run = run_stellarator_turbulence(geometry, steps=3, dt=2.0e-3, seed=2, sheath_sink=True)
    return dict(run._asdict())


def case_fci_differentiable() -> dict:
    from drbx.native.fci_differentiable_case import build_context, differentiability_report

    report = differentiability_report(
        build_context((10, 10, 6), sigma=0.6), amp0=0.1, n_steps=2, dt=1.0e-3, fd_step=1.0e-5
    )
    return {k: v for k, v in report.items() if not isinstance(v, str)}


CASES = {name[5:]: fn for name, fn in globals().items() if name.startswith("case_")}


def _flatten(tree) -> dict[str, np.ndarray]:
    import jax

    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    return {jax.tree_util.keystr(path) or "value": np.asarray(leaf) for path, leaf in leaves}


def _run_case(name: str, out: Path) -> None:
    from drbx.runtime import configure_jax_runtime

    configure_jax_runtime(precision="float64")
    start = time.perf_counter()
    arrays = _flatten(CASES[name]())
    np.savez(out / f"{name}.npz", **arrays)
    (out / f"{name}.time").write_text(f"{time.perf_counter() - start:.3f}\n")


# ---------------------------------------------------------------- pytest + lines
def _record_pytest(summary: dict) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not slow", "-n", "2"],
        cwd=REPO, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-500:]
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed)", tail)}
    summary["pytest"] = {"returncode": proc.returncode, "counts": counts, "tail": tail}


def _line_counts() -> dict[str, int]:
    src = REPO / "src" / "drbx"
    return {
        package: sum(len(p.read_text().splitlines()) for p in (src / package).rglob("*.py"))
        for package in LINE_COUNT_PACKAGES
    }


def record(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    import jax

    summary = {"commit": head, "jax": jax.__version__, "numpy": np.__version__, "lines": _line_counts(), "cases": {}}
    names = args.cases.split(",") if args.cases else list(CASES)
    for name in names:
        print(f"[gate] case {name}", flush=True)
        code = subprocess.call([sys.executable, __file__, "_case", name, str(out)])
        summary["cases"][name] = (
            float((out / f"{name}.time").read_text()) if code == 0 else f"failed ({code})"
        )
    if not args.skip_pytest:
        print("[gate] pytest -m 'not slow' -n 2", flush=True)
        _record_pytest(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0


# ---------------------------------------------------------------- compare
def _diff(a: np.ndarray, b: np.ndarray) -> str:
    if a.shape != b.shape or a.dtype != b.dtype:
        return f"shape/dtype {a.shape}{a.dtype} vs {b.shape}{b.dtype}"
    if not np.issubdtype(a.dtype, np.number):
        return "non-numeric difference"
    a64, b64 = a.astype(np.complex128 if np.iscomplexobj(a) else np.float64), b.astype(
        np.complex128 if np.iscomplexobj(b) else np.float64
    )
    delta = np.abs(a64 - b64)
    scale = np.maximum(np.abs(a64), np.abs(b64))
    rel = np.where(scale > 0, delta / np.where(scale > 0, scale, 1.0), 0.0)
    return f"max abs {np.nanmax(delta):.3e}  max rel {np.nanmax(rel):.3e}"


def compare(args: argparse.Namespace) -> int:
    base, new = Path(args.base), Path(args.new)
    sa, sb = (json.loads((p / "summary.json").read_text()) for p in (base, new))
    ok = True
    files = sorted({p.name for p in base.glob("*.npz")} | {p.name for p in new.glob("*.npz")})
    for name in files:
        if not (base / name).exists() or not (new / name).exists():
            print(f"MISSING {name}")
            ok = False
            continue
        with np.load(base / name) as da, np.load(new / name) as db:
            for key in sorted(set(da.files) | set(db.files)):
                if key not in da.files or key not in db.files:
                    print(f"MISSING {name}:{key}")
                    ok = False
                elif np.array_equal(da[key], db[key], equal_nan=da[key].dtype.kind in "fc"):
                    print(f"bitwise   {name}:{key}")
                else:
                    print(f"DIFFERENT {name}:{key}  {_diff(da[key], db[key])}")
                    ok = False
    pa, pb = (s.get("pytest", {}).get("counts") for s in (sa, sb))
    if pa is not None or pb is not None:
        ok &= pa == pb
        print(f"pytest counts: {pa} vs {pb}")
    line_deltas = {}
    for package in sorted(sa["lines"].keys() | sb["lines"].keys()):
        before, after = (summary["lines"].get(package, 0) for summary in (sa, sb))
        line_deltas[package] = f"{before} -> {after} ({after - before:+d})"
    print("lines:", line_deltas)
    print("case wall s:", {k: (sa["cases"].get(k), sb["cases"].get(k)) for k in sa["cases"]})
    print("GATE PASS (bitwise)" if ok else "GATE FAIL")
    return 0 if ok else 1


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "_case":
        _run_case(sys.argv[2], Path(sys.argv[3]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("out")
    rec.add_argument("--skip-pytest", action="store_true")
    rec.add_argument("--cases", help=f"comma-separated subset of {','.join(CASES)}")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("base")
    cmp_.add_argument("new")
    args = parser.parse_args()
    return record(args) if args.command == "record" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
