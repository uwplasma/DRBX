from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

from jax_drb.validation import create_tcv_x21_selected_field_parity_package

PUBLIC_BENCHMARK_FILES = {
    "TCV_forward_field.nc": "https://raw.githubusercontent.com/SPCData/TCV-X21/main/1.experimental_data/TCV_forward_field.nc",
    "TCV_ortho.nc": "https://raw.githubusercontent.com/SPCData/TCV-X21/main/tests/sample_data/TCV_ortho.nc",
    "snaps00000.nc": "https://raw.githubusercontent.com/SPCData/TCV-X21/main/tests/sample_data/snaps00000.nc",
    "vgrid.nc": "https://raw.githubusercontent.com/SPCData/TCV-X21/main/tests/sample_data/vgrid.nc",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the reduced selected-field parity package for the TCV-X21 3D lane. "
            "If no explicit inputs are given, the demo prefers a public benchmark-data root "
            "and falls back to the synthetic scaffold preview pair."
        )
    )
    parser.add_argument("--reference-workdir", type=Path, default=None)
    parser.add_argument("--candidate-workdir", type=Path, default=None)
    parser.add_argument("--benchmark-data-root", type=Path, default=None)
    parser.add_argument("--download-public-benchmark-data", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "docs" / "data" / "tokamak_tcv_x21_selected_field_artifacts",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_data_root = args.benchmark_data_root
    if args.download_public_benchmark_data:
        benchmark_data_root = benchmark_data_root or Path("/tmp/tcv_x21_public_benchmark")
        _download_public_benchmark_data(benchmark_data_root)
    artifacts = create_tcv_x21_selected_field_parity_package(
        reference_workdir=args.reference_workdir,
        candidate_workdir=args.candidate_workdir,
        benchmark_data_root=benchmark_data_root,
        output_root=args.output_root,
    )
    if args.quiet:
        return
    print("\n== TCV-X21 Selected-Field Parity ==")
    print(f"  - reference_workdir: {args.reference_workdir if args.reference_workdir is not None else '<none>'}")
    print(f"  - candidate_workdir: {args.candidate_workdir if args.candidate_workdir is not None else '<derived or synthetic>'}")
    print(f"  - benchmark_data_root: {benchmark_data_root if benchmark_data_root is not None else '<none>'}")
    print(f"  - parity_json: {artifacts.parity_json_path}")
    print(f"  - parity_arrays: {artifacts.parity_arrays_npz_path}")
    print(f"  - parity_plot: {artifacts.parity_plot_png_path}")
    print(f"  - observable_report: {artifacts.observable_report_json_path}")
    print(
        "  - benchmark_data_report: "
        f"{artifacts.benchmark_data_report_json_path if artifacts.benchmark_data_report_json_path is not None else '<none>'}"
    )


def _download_public_benchmark_data(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, url in PUBLIC_BENCHMARK_FILES.items():
        target = root / name
        if target.exists():
            continue
        urllib.request.urlretrieve(url, target)


if __name__ == "__main__":
    main()
