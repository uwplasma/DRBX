from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _set_key(lines: list[str], section: str | None, key: str, value: str) -> list[str]:
    out = list(lines)
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")
    sec_re = re.compile(r"^\s*\[(.+)\]\s*$")
    in_sec = section is None
    sec_start = 0
    sec_end = len(out)
    if section is not None:
        found = False
        for i, line in enumerate(out):
            m = sec_re.match(line)
            if m:
                name = m.group(1).strip()
                if name == section:
                    found = True
                    in_sec = True
                    sec_start = i + 1
                    continue
                if found:
                    sec_end = i
                    break
        if not found:
            out.extend(["", f"[{section}]"])
            sec_start = len(out)
            sec_end = len(out)
            in_sec = True

    if in_sec:
        for i in range(sec_start, sec_end):
            if key_re.match(out[i]):
                out[i] = f"{key} = {value}"
                return out
        out.insert(sec_end, f"{key} = {value}")
    return out


def _prepare(base_dir: Path, out_dir: Path, nout: int, timestep: float) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_dir / "tokamak.nc", out_dir / "tokamak.nc")
    src_inp = base_dir / "BOUT.inp"
    if not src_inp.exists():
        raise FileNotFoundError(f"Missing {src_inp}")
    lines = src_inp.read_text(encoding="utf-8").splitlines()

    lines = _set_key(lines, None, "nout", str(int(nout)))
    lines = _set_key(lines, None, "timestep", f"{float(timestep):.12g}")

    for sec in ("e", "d+", "vorticity", "polarisation_drift"):
        lines = _set_key(lines, sec, "diagnose", "true")
    lines = _set_key(lines, "vorticity", "diagnose_terms", "true")

    for sec in ("braginskii_collisions", "braginskii_friction", "braginskii_heat_exchange"):
        lines = _set_key(lines, sec, "diagnose", "true")

    inp_path = out_dir / "BOUT.inp"
    inp_text = "\n".join(lines) + "\n"
    inp_path.write_text(inp_text, encoding="utf-8")

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "BOUT.inp").write_text(inp_text, encoding="utf-8")
    return inp_path


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Prepare a dense-output Hermes run directory with diagnostic channels enabled "
            "for early mismatch tracing."
        )
    )
    p.add_argument(
        "--base-run-dir", required=True, help="Existing Hermes run dir with BOUT.inp/tokamak.nc."
    )
    p.add_argument("--out-run-dir", required=True, help="Output run dir.")
    p.add_argument(
        "--hermes-bin",
        default="hermes-3",
        help="Hermes executable (name or path) used in the run hint.",
    )
    p.add_argument("--nout", type=int, default=100, help="Number of output steps.")
    p.add_argument(
        "--timestep", type=float, default=0.01, help="Output timestep (normalized time)."
    )
    args = p.parse_args()

    inp = _prepare(
        Path(args.base_run_dir).resolve(),
        Path(args.out_run_dir).resolve(),
        int(args.nout),
        float(args.timestep),
    )
    print(f"Prepared {inp}")
    print("Run Hermes with:")
    print(f"  cd {inp.parent} && {args.hermes_bin} -d data")


if __name__ == "__main__":
    main()
