from __future__ import annotations

import argparse
import json
from pathlib import Path

from jaxdrb.benchmarking import compare_bundle_diagnostics, load_bundle_npz


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare compact/full benchmark bundles.")
    p.add_argument("--reference", required=True, help="Reference benchmark bundle (.npz).")
    p.add_argument("--candidate", required=True, help="Candidate benchmark bundle (.npz).")
    p.add_argument(
        "--keys",
        default="rms_n_fluct,rms_Te_fluct,rms_omega_fluct,rms_phi_fluct,psd_n_f,psd_n_ky",
        help="Comma-separated diagnostic keys to compare.",
    )
    p.add_argument("--out-json", default="", help="Optional JSON summary path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    keys = tuple(k.strip() for k in args.keys.split(",") if k.strip())
    reference = load_bundle_npz(args.reference)
    candidate = load_bundle_npz(args.candidate)
    out = compare_bundle_diagnostics(reference, candidate, keys=keys)
    summary = {
        "reference": Path(args.reference).name,
        "candidate": Path(args.candidate).name,
        "keys": list(out.per_key_rel_l2),
        "per_key_rel_l2": out.per_key_rel_l2,
        "mean_rel_l2": out.mean_rel_l2,
        "max_rel_l2": out.max_rel_l2,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
