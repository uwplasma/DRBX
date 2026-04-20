#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from jax_drb.native.reference_dump import load_local_reference_snapshot
from jax_drb.native.runner import run_curated_case
from jax_drb.parity.reference import (
    _prepare_workdir,
    _run_reference_binary,
    discover_reference_binary,
    make_default_overrides,
    merge_overrides,
    resolve_reference_case,
)


STATE_FIELDS = ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe")


def strip_anomalous_diffusion_from_boutinp_text(text: str, *, species_names: tuple[str, ...] = ("d+", "e")) -> str:
    lines = text.splitlines()
    current_section: str | None = None
    targets = set(species_names)
    output: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            output.append(line)
            continue
        if current_section in targets and stripped.startswith("type") and "=" in line:
            prefix, raw_value = line.split("=", 1)
            parts = [part.strip() for part in raw_value.split(",")]
            filtered = [part for part in parts if part and part != "anomalous_diffusion"]
            output.append(f"{prefix}= {', '.join(filtered)}")
            continue
        output.append(line)
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def _run_reference_variant(
    *,
    case_name: str,
    reference_root: Path,
    strip_anomalous_diffusion: bool,
) -> tuple[Path, Path]:
    case, input_path = resolve_reference_case(case_name, reference_root=reference_root)
    binary = discover_reference_binary(reference_root=reference_root)
    with tempfile.TemporaryDirectory(prefix="jaxdrb-prod-anomdiag-") as workdir:
        staged = _prepare_workdir(case, input_path, workdir=workdir)
        staged_input = staged / "BOUT.inp"
        if strip_anomalous_diffusion:
            staged_input.write_text(
                strip_anomalous_diffusion_from_boutinp_text(staged_input.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
        stdout_path = staged / "run.stdout"
        overrides = merge_overrides(make_default_overrides(case.parity_mode), case.extra_overrides)
        _run_reference_binary(
            binary=binary,
            workdir=staged,
            overrides=overrides,
            stdout_path=stdout_path,
            process_count=case.process_count,
        )
        dump_path = staged / "BOUT.dmp.0.nc"
        kept = Path(tempfile.mkdtemp(prefix="jaxdrb-prod-anomdiag-keep-"))
        target_dump = kept / dump_path.name
        target_stdout = kept / stdout_path.name
        target_dump.write_bytes(dump_path.read_bytes())
        target_stdout.write_text(stdout_path.read_text(encoding="utf-8"), encoding="utf-8")
        return target_dump, target_stdout


def _load_final_state(path: Path):
    return load_local_reference_snapshot(
        path,
        field_names=STATE_FIELDS,
        optional_field_names=("Sd_target_recycle", "Ed_target_recycle"),
        scalar_names=(),
        time_index=1,
    )


def _target_band_cells(mesh, *, x_indices: tuple[int, ...] = (14, 15), z_index: int = 0) -> tuple[tuple[int, int, int], ...]:
    return tuple((int(x_index), int(mesh.ystart), int(z_index)) for x_index in x_indices)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether missing anomalous_diffusion explains the integrated 2D production "
            "one-step target-band residuals by comparing current native output against Hermes "
            "with and without anomalous_diffusion enabled for d+ and e."
        )
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument(
        "--case",
        default="integrated_2d_production_one_step",
        choices=("integrated_2d_production_one_step",),
    )
    args = parser.parse_args()

    reference_dump, reference_stdout = _run_reference_variant(
        case_name=args.case,
        reference_root=args.reference_root,
        strip_anomalous_diffusion=False,
    )
    no_anom_dump, no_anom_stdout = _run_reference_variant(
        case_name=args.case,
        reference_root=args.reference_root,
        strip_anomalous_diffusion=True,
    )
    reference_final = _load_final_state(reference_dump)
    no_anom_final = _load_final_state(no_anom_dump)
    native = run_curated_case(args.case, reference_root=args.reference_root)

    print(f"case={args.case}")
    print(f"reference_dump={reference_dump}")
    print(f"reference_stdout={reference_stdout}")
    print(f"no_anom_dump={no_anom_dump}")
    print(f"no_anom_stdout={no_anom_stdout}")

    for cell in _target_band_cells(reference_final.mesh):
        i, j, k = cell
        print(f"cell={cell}")
        for field in ("Pe", "Pd+", "NVd+", "Nd+", "Sd_target_recycle", "Ed_target_recycle"):
            if field in reference_final.fields:
                reference_value = np.asarray(reference_final.fields[field], dtype=np.float64)[i, j, k]
                no_anom_value = np.asarray(no_anom_final.fields[field], dtype=np.float64)[i, j, k]
            else:
                reference_value = np.asarray(reference_final.optional_fields[field], dtype=np.float64)[i, j, k]
                no_anom_value = np.asarray(no_anom_final.optional_fields[field], dtype=np.float64)[i, j, k]
            native_value = np.asarray(native.variables[field], dtype=np.float64)[1, i, j, k]
            native_minus_reference = native_value - reference_value
            no_anom_minus_reference = no_anom_value - reference_value
            ratio = np.nan
            if abs(no_anom_minus_reference) > 1.0e-12:
                ratio = native_minus_reference / no_anom_minus_reference
            print(
                f"  {field}: native-ref={native_minus_reference:.12e} "
                f"no_anom-ref={no_anom_minus_reference:.12e} ratio={ratio:.12e}"
            )


if __name__ == "__main__":
    main()
