#!/usr/bin/env python3
"""FCI unification gate (issue #41): record one checkout, compare two records.

    python scripts/fci_gate.py record OUT [--skip-pytest] [--skip-hsx] [--cases a,b]
    python scripts/fci_gate.py compare BASE NEW

``record`` runs from the checkout that is on ``PYTHONPATH`` (or installed) and
writes into ``OUT``:

* ``summary.json``: pytest ``-m "not slow" -n 2`` counts, source line counts,
  HSX GMRES log values and per-case wall times;
* ``<case>.npz``: every array a case produced (reduced FCI models, HSX state).

HSX cases need ``DRBX_TEST_GEOMETRY_BUNDLE`` (an HSX FCI geometry directory)
and are skipped with a message when it is unset.  ``compare`` exits non-zero
unless every array is bitwise equal (``np.array_equal``), the GMRES log values
match and the pytest counts are unchanged; otherwise it prints max abs/rel
differences per array.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LINE_COUNT_PACKAGES = ("geometry", "native", "fci_braginskii")
HSX_FINAL_TIME = 0.00375
HSX_STEPS = 5


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


# ---------------------------------------------------------------- HSX via drbx run
def _drbx(deck: Path, log: Path) -> float:
    start = time.perf_counter()
    with log.open("w") as handle:
        code = subprocess.call(
            [sys.executable, "-c", "import sys; from drbx.cli import main; sys.exit(main())", "run", str(deck)],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if code:
        raise SystemExit(f"drbx run {deck} failed ({code}); see {log}")
    return time.perf_counter() - start


def _deck(path: Path, bundle: str, output: Path, **keys) -> Path:
    lines = ['[model]', 'backend = "fci_braginskii"', "[fci_braginskii]"]
    keys = {"geometry": bundle, "output": str(output), **keys}
    lines += [f"{k} = {json.dumps(v if not isinstance(v, Path) else str(v))}" for k, v in keys.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


def _gmres_values(log: Path) -> dict[str, list[float]]:
    """GMRES log values (compared) and per-step step/op/gmres seconds (reported)."""
    text = log.read_text()
    values = {
        key: [float(v) for v in re.findall(re.escape(key) + r"=([-+0-9.eE]+)", text)]
        for key in ("gmres-iters(avg4)", "gmres-relres(max4)")
    }
    for key in ("step", "op", "gmres"):
        values[f"{key}_s"] = [float(v) for v in re.findall(rf" {key}=([0-9.]+)s", text)]
    return values


def _npz_arrays(directory: Path) -> dict[str, np.ndarray]:
    arrays = {}
    for path in sorted(directory.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            # run metadata carries wall times and paths, so it is not compared
            arrays.update({f"{path.stem}/{k}": data[k] for k in data.files if "metadata" not in k})
    return arrays


def _record_hsx(out: Path, summary: dict) -> None:
    bundle = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
    if not bundle:
        print("[gate] HSX skipped: set DRBX_TEST_GEOMETRY_BUNDLE to an HSX FCI geometry bundle")
        summary["hsx"] = "skipped (DRBX_TEST_GEOMETRY_BUNDLE unset)"
        return
    work = Path(tempfile.mkdtemp(prefix="fci_gate_hsx_"))
    common = dict(final_time=HSX_FINAL_TIME, num_steps=HSX_STEPS, save_every=HSX_STEPS)
    runs = {
        # five steps in one go
        "hsx": _deck(work / "hsx.toml", bundle, work / "hsx" / "hsx", **common),
        # restart round trip: 2 steps, then restart from that history for 3 more
        "hsx_restart_a": _deck(
            work / "ra.toml", bundle, work / "ra" / "hsx",
            final_time=HSX_FINAL_TIME * 2 / 5, num_steps=2, save_every=2, checkpoint_every=2,
        ),
    }
    hsx = {}
    for name, deck in runs.items():
        log = out / f"{name}.log"
        hsx[name] = {"wall_s": _drbx(deck, log), **_gmres_values(log)}
    # restart from the full-precision checkpoint (the history file is float32)
    first = sorted((work / "ra").rglob("*checkpoint_step*.npz")) or sorted((work / "ra").glob("*.npz"))
    restart = _deck(
        work / "rb.toml", bundle, work / "rb" / "hsx", restart_from=first[0], final_time=HSX_FINAL_TIME, num_steps=3, save_every=3
    )
    hsx["hsx_restart_b"] = {"wall_s": _drbx(restart, out / "hsx_restart_b.log"), **_gmres_values(out / "hsx_restart_b.log")}
    for name in ("hsx", "ra", "rb"):
        np.savez(out / f"{'hsx' if name == 'hsx' else 'hsx_restart_' + name[1]}.npz", **_npz_arrays(work / name))
    straight, restarted = _npz_arrays(work / "hsx"), _npz_arrays(work / "rb")
    hsx["restart_vs_straight_max_abs"] = {
        field: float(np.max(np.abs(straight[f"hsx/{field}"][-1] - restarted[f"hsx/{field}"][-1])))
        for field in ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")
    }
    summary["hsx"] = hsx


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
    if not args.skip_hsx:
        _record_hsx(out, summary)
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
    for name in sorted(set(sa.get("hsx", {}) or {}) if isinstance(sa.get("hsx"), dict) else []):
        for key in ("gmres-iters(avg4)", "gmres-relres(max4)"):
            a, b = sa["hsx"][name].get(key), (sb.get("hsx") or {}).get(name, {}).get(key)
            same = a == b
            ok &= same
            print(f"{'same     ' if same else 'DIFFERENT'} {name} {key}: {a} vs {b}")
    pa, pb = (s.get("pytest", {}).get("counts") for s in (sa, sb))
    if pa is not None or pb is not None:
        ok &= pa == pb
        print(f"pytest counts: {pa} vs {pb}")
    print("lines:", {k: f"{sa['lines'][k]} -> {sb['lines'][k]} ({sb['lines'][k] - sa['lines'][k]:+d})" for k in sa["lines"]})
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
    rec.add_argument("--skip-hsx", action="store_true")
    rec.add_argument("--cases", help=f"comma-separated subset of {','.join(CASES)}")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("base")
    cmp_.add_argument("new")
    args = parser.parse_args()
    return record(args) if args.command == "record" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
