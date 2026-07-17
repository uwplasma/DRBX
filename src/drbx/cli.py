from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

from .config.boutinp import load_bout_input
from .runtime import configure_jax_runtime, resolve_runtime_precision
from .runtime.run_config import RunConfiguration


def main(argv: list[str] | None = None) -> int:
    normalized_argv = _normalize_cli_argv(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(normalized_argv)
    return args.command(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drbx",
        description="Inspect or run JAX-DRB inputs using the native model configuration structure.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=False)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect an input deck and print the resolved plan."
    )
    inspect_parser.add_argument("input_file", type=Path)
    inspect_parser.set_defaults(command=_inspect_command)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a supported native input, write result artifacts, and optionally continue from a restart bundle.",
    )
    run_parser.add_argument("input_file", type=Path)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only inspect configuration and exit successfully.",
    )
    run_parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default=None,
        help="Override runtime floating-point precision for this run.",
    )
    run_parser.add_argument(
        "--case-name",
        type=str,
        default=None,
        help="Optional case label for output metadata.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write standard run artifacts into this directory.",
    )
    run_parser.add_argument(
        "--json-out", type=Path, default=None, help="Write the portable summary JSON."
    )
    run_parser.add_argument(
        "--arrays-out", type=Path, default=None, help="Write the portable array NPZ."
    )
    run_parser.add_argument(
        "--restart-out", type=Path, default=None, help="Write the restart NPZ bundle."
    )
    run_parser.add_argument(
        "--log-out", type=Path, default=None, help="Write a verbose run log JSON."
    )
    run_parser.add_argument(
        "--restart-in",
        type=Path,
        default=None,
        help="Resume from a previously written restart NPZ bundle.",
    )
    run_parser.add_argument(
        "--resume-steps",
        type=int,
        default=None,
        help="Additional output intervals to run after loading --restart-in.",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit detailed staged terminal output for this run.",
    )
    run_parser.add_argument(
        "--quiet", action="store_true", help="Suppress the pretty terminal run summary."
    )
    run_parser.set_defaults(command=_run_command)

    parser.set_defaults(command=_default_command)
    return parser


def _default_command(args: argparse.Namespace) -> int:
    if getattr(args, "subcommand", None) is None:
        raise SystemExit(
            "Use `drbx inspect <input>` or `drbx <input> --dry-run`."
        )
    return args.command(args)


def _normalize_cli_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    known_subcommands = {
        "inspect",
        "run",
    }
    head = argv[0]
    if head in known_subcommands or head.startswith("-"):
        return argv
    return ["run", *argv]


def _inspect_command(args: argparse.Namespace) -> int:
    config = load_bout_input(args.input_file)
    configure_jax_runtime(precision=resolve_runtime_precision(config=config))
    run_config = RunConfiguration.from_config(config)

    print(f"input: {args.input_file}")
    print(f"sections: {', '.join(config.section_names())}")
    print(f"time: nout={run_config.time.nout}, timestep={run_config.time.timestep:g}")
    print(
        "mesh: "
        f"nx={run_config.mesh.nx}, ny={run_config.mesh.ny}, nz={run_config.mesh.nz}, "
        f"MXG={run_config.mesh.mxg}, MYG={run_config.mesh.myg}, "
        f"parallel_transform={run_config.mesh.parallel_transform.type}"
    )
    print(
        f"scheduled components: {', '.join(request.label for request in run_config.components)}"
    )

    if run_config.normalization is not None:
        normalization = run_config.normalization
        print(
            "normalization: "
            f"Nnorm={normalization.Nnorm:g}, "
            f"Tnorm={normalization.Tnorm:g}, "
            f"Bnorm={normalization.Bnorm:g}, "
            f"Cs0={normalization.Cs0:.8e}, "
            f"Omega_ci={normalization.Omega_ci:.8e}, "
            f"rho_s0={normalization.rho_s0:.8e}"
        )
    else:
        print("normalization: unresolved (missing one or more of Nnorm, Tnorm, Bnorm)")

    return 0


def _run_command(args: argparse.Namespace) -> int:
    if args.dry_run:
        return _inspect_command(args)
    config = load_bout_input(args.input_file)
    run_config = RunConfiguration.from_config(config)
    resolved_precision = resolve_runtime_precision(
        requested=args.precision, config=config
    )
    cache_dir = configure_jax_runtime(precision=resolved_precision)
    import jax
    from .native import run_input_case
    from .native.deck_runner import (
        NativeRestartState,
        build_portable_array_payload,
        build_restart_state,
        write_portable_array_payload,
        write_portable_summary_payload,
    )
    from .runtime import (
        build_run_log_payload,
        load_restart_bundle,
        print_run_log,
        write_restart_bundle,
        write_run_log_payload,
    )
    from .runtime.output import build_run_event, print_run_event

    command_started_at = time.perf_counter()
    output_dir = args.output_dir or _config_path(config, "output", "directory")
    case_name = (
        args.case_name
        or _config_string(config, "output", "case_name")
        or args.input_file.stem
    )
    restart_in = args.restart_in or _config_path(config, "restart", "input")
    resume_steps = (
        args.resume_steps
        if args.resume_steps is not None
        else _config_int(config, "restart", "resume_steps")
    )
    logging_quiet = _config_bool(config, "runtime:logging", "quiet", default=False)
    logging_verbose = _config_optional_bool(config, "runtime:logging", "verbose")
    logging_verbosity = _config_string(config, "runtime:logging", "verbosity")
    if logging_verbosity is None:
        logging_verbosity = "detailed" if logging_verbose else "summary"
    if args.verbose:
        logging_verbosity = "detailed"
    emit_terminal_log = not args.quiet and not logging_quiet
    write_summary = _config_bool(config, "output", "write_summary", default=True)
    write_arrays = _config_bool(config, "output", "write_arrays", default=True)
    write_restart = _config_bool(config, "output", "write_restart", default=True)
    write_log = _config_bool(config, "output", "write_log", default=True)
    if args.json_out is None:
        args.json_out = _config_path(config, "output", "summary_json")
    if args.arrays_out is None:
        args.arrays_out = _config_path(config, "output", "arrays_npz")
    if args.restart_out is None:
        args.restart_out = _config_path(config, "output", "restart_npz")
    if args.log_out is None:
        args.log_out = _config_path(config, "output", "run_log_json")
    events: list[dict[str, Any]] = []

    def record_event(stage: str, message: str, **details: Any) -> None:
        event = build_run_event(
            stage=stage,
            message=message,
            elapsed_seconds=time.perf_counter() - command_started_at,
            details=details or None,
        )
        events.append(event)
        if emit_terminal_log:
            print_run_event(event, verbosity=logging_verbosity)

    record_event(
        "configuration",
        "Loaded input configuration",
        input_file=args.input_file,
        case_name=case_name,
        capability_tier="native_exact",
        precision=resolved_precision,
        nout=run_config.time.nout,
        timestep=run_config.time.timestep,
        output_dir=output_dir if output_dir is not None else "(none)",
        verbosity=logging_verbosity,
        verbose=logging_verbosity == "detailed",
    )
    restart_state = None
    bundle = None
    if restart_in is not None:
        bundle = load_restart_bundle(restart_in)
        restart_state = NativeRestartState(
            time_offset=bundle.current_time,
            completed_steps=bundle.completed_steps,
            configured_timestep=bundle.configured_timestep,
            variables=bundle.state_variables,
        )
        record_event(
            "restart",
            "Loaded restart bundle",
            restart_in=restart_in,
            current_time=bundle.current_time,
            completed_steps=bundle.completed_steps,
            variables=",".join(sorted(bundle.state_variables)),
            requested_resume_steps=resume_steps
            if resume_steps is not None
            else "(default)",
        )

    def relay_native_event(event: Mapping[str, Any]) -> None:
        if str(event.get("stage", "")) != "progress":
            return
        details = event.get("details")
        if isinstance(details, Mapping):
            record_event(
                str(event.get("stage", "progress")),
                str(event.get("message", "Native progress update")),
                **dict(details),
            )
        else:
            record_event(
                str(event.get("stage", "progress")),
                str(event.get("message", "Native progress update")),
            )

    started_at = time.perf_counter()
    record_event(
        "run", "Launching native run", mode="run", restart=restart_state is not None
    )
    result = run_input_case(
        args.input_file,
        case_name=case_name,
        parity_mode="run",
        restart_state=restart_state,
        output_steps=resume_steps,
        verbose=False,
        event_logger=relay_native_event,
    )
    elapsed_seconds = time.perf_counter() - started_at
    record_event(
        "run",
        "Native run completed",
        elapsed_seconds=f"{elapsed_seconds:.3f}",
        stored_states=len(result.time_points),
        compare_variables=",".join(result.variables),
    )

    output_paths: dict[str, str] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.json_out is None and write_summary:
            args.json_out = output_dir / f"{case_name}_summary.json"
        if args.arrays_out is None and write_arrays:
            args.arrays_out = output_dir / f"{case_name}_arrays.npz"
        if args.restart_out is None and write_restart:
            args.restart_out = output_dir / f"{case_name}_restart.npz"
        if args.log_out is None and write_log:
            args.log_out = output_dir / f"{case_name}_run_log.json"
        record_event(
            "artifacts",
            "Resolved artifact destinations",
            summary_json=args.json_out if args.json_out is not None else "(disabled)",
            arrays_npz=args.arrays_out if args.arrays_out is not None else "(disabled)",
            restart_npz=args.restart_out
            if args.restart_out is not None
            else "(disabled)",
            run_log_json=args.log_out if args.log_out is not None else "(disabled)",
        )

    if args.json_out is not None:
        path = write_portable_summary_payload(result.payload, args.json_out)
        output_paths["summary_json"] = _sanitize_logged_path(path) or str(path)
        record_event("artifacts", "Wrote summary JSON", path=path)
    if args.arrays_out is not None:
        array_payload = build_portable_array_payload(
            case_name=str(result.payload["case_name"]),
            parity_mode=str(result.payload["parity_mode"]),
            capability_tier=str(result.payload.get("capability_tier", "native_exact")),
            compare_variables=tuple(str(name) for name in result.variables),
            component_labels=tuple(result.payload.get("component_labels", [])),
            dimensions=result.payload.get("dimensions", {}),
            time_points=tuple(float(value) for value in result.time_points),
            dataset_scalars=result.payload.get("dataset_scalars", {}),
            variables=result.variables,
            overrides=tuple(result.payload.get("overrides", [])),
            configured_nout=result.payload.get("configured_nout"),
            configured_timestep=result.payload.get("configured_timestep"),
            producer=str(result.payload.get("producer", "drbx")),
        )
        path = write_portable_array_payload(array_payload, args.arrays_out)
        output_paths["arrays_npz"] = _sanitize_logged_path(path) or str(path)
        record_event(
            "artifacts", "Wrote arrays NPZ", path=path, variables=len(result.variables)
        )

    restart_bundle = build_restart_state(result, parity_mode="run")
    if args.restart_out is not None and restart_bundle is not None:
        path = write_restart_bundle(restart_bundle, args.restart_out)
        output_paths["restart_npz"] = _sanitize_logged_path(path) or str(path)
        record_event(
            "artifacts",
            "Wrote restart bundle",
            path=path,
            completed_steps=restart_bundle.completed_steps,
            current_time=restart_bundle.current_time,
        )
    elif args.restart_out is not None and restart_bundle is None:
        output_paths["restart_npz"] = "(unsupported for this component set)"
        record_event(
            "artifacts",
            "Restart bundle unsupported for this run",
            path=args.restart_out,
        )

    if args.log_out is not None:
        output_paths["run_log_json"] = _sanitize_logged_path(args.log_out) or str(
            args.log_out
        )
    if output_paths:
        record_event("artifacts", "Planned run artifacts", **output_paths)

    log_payload = build_run_log_payload(
        input_file=_sanitize_logged_path(args.input_file) or args.input_file,
        case_name=case_name,
        parity_mode="run",
        capability_tier=str(result.payload.get("capability_tier", "native_exact")),
        component_labels=tuple(result.payload.get("component_labels", [])),
        time_points=tuple(float(value) for value in result.time_points),
        dimensions=result.payload.get("dimensions", {}),
        compare_variables=tuple(result.payload.get("compare_variables", [])),
        restart_supported=restart_bundle is not None,
        outputs=output_paths,
        variable_summaries=result.payload.get("variable_summaries", {}),
        run_configuration=_serialize_run_configuration(
            run_config,
            precision=resolved_precision,
            backend=jax.default_backend(),
            device=str(jax.devices()[0]) if jax.devices() else None,
            jax_version=getattr(jax, "__version__", None),
            cache_dir=cache_dir,
            elapsed_seconds=elapsed_seconds,
            output_directory=output_dir,
            logging_verbosity=logging_verbosity,
            logging_quiet=emit_terminal_log is False,
            restart_in=restart_in,
            resume_steps=resume_steps,
            working_directory=Path.cwd(),
        ),
        restart_info=_serialize_restart_info(
            restart_in=restart_in,
            loaded_bundle=bundle if restart_in is not None else None,
            requested_additional_steps=resume_steps,
            saved_bundle=restart_bundle,
        ),
        events=tuple(events),
    )
    if args.log_out is not None:
        record_event(
            "artifacts",
            "Writing verbose run log JSON",
            path=args.log_out,
            event_count=len(events),
        )
        log_payload["events"] = list(events)
        log_payload["event_count"] = len(events)
        log_payload["event_stages"] = [str(event.get("stage", "")) for event in events]
        path = write_run_log_payload(log_payload, args.log_out)
        output_paths["run_log_json"] = _sanitize_logged_path(path) or str(path)
        log_payload["outputs"] = output_paths
        log_payload["events"] = list(events)
        log_payload["event_count"] = len(events)
        log_payload["event_stages"] = [str(event.get("stage", "")) for event in events]

    if emit_terminal_log:
        print_run_log(log_payload, verbosity=logging_verbosity)
    return 0


def _serialize_run_configuration(
    run_config: RunConfiguration,
    *,
    precision: str,
    backend: str | None = None,
    device: str | None = None,
    cache_dir: Path | None = None,
    elapsed_seconds: float | None = None,
    output_directory: Path | None = None,
    logging_verbosity: str | None = None,
    logging_quiet: bool | None = None,
    restart_in: Path | None = None,
    resume_steps: int | None = None,
    jax_version: str | None = None,
    working_directory: Path | None = None,
) -> dict[str, object]:
    return {
        "time": {
            "nout": run_config.time.nout,
            "timestep": run_config.time.timestep,
        },
        "mesh": {
            "nx": run_config.mesh.nx,
            "ny": run_config.mesh.ny,
            "nz": run_config.mesh.nz,
            "mxg": run_config.mesh.mxg,
            "myg": run_config.mesh.myg,
            "file": run_config.mesh.file,
            "parallel_transform": run_config.mesh.parallel_transform.type,
        },
        "solver": {
            "type": run_config.solver.type,
            "mxstep": run_config.solver.mxstep,
            "rtol": run_config.solver.rtol,
            "atol": run_config.solver.atol,
            "use_precon": run_config.solver.use_precon,
            "cvode_max_order": run_config.solver.cvode_max_order,
        },
        "runtime": {
            "precision": precision,
            "backend": backend,
            "device": device,
            "jax_version": jax_version,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "process_id": os.getpid(),
            "compilation_cache_dir": _sanitize_logged_path(cache_dir),
            "elapsed_seconds": elapsed_seconds,
            "logging": {
                "verbosity": logging_verbosity,
                "verbose": logging_verbosity == "detailed",
                "quiet": logging_quiet,
            },
        },
        "output": {
            "directory": _sanitize_logged_path(output_directory),
            "working_directory": _sanitize_logged_path(working_directory),
        },
        "restart_request": {
            "restart_in": _sanitize_logged_path(restart_in),
            "resume_steps": resume_steps,
        },
        "components": [request.label for request in run_config.components],
        "root_scalars": dict(run_config.root_scalars),
        "model_scalars": dict(run_config.model_scalars),
    }


def _serialize_restart_info(
    *,
    restart_in: Path | None,
    loaded_bundle,
    requested_additional_steps: int | None,
    saved_bundle,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    if restart_in is not None and loaded_bundle is not None:
        payload["loaded_from"] = _sanitize_logged_path(restart_in)
        payload["start_time"] = loaded_bundle.current_time
        payload["input_completed_steps"] = loaded_bundle.completed_steps
        payload["loaded_state_variables"] = sorted(loaded_bundle.state_variables)
    if requested_additional_steps is not None:
        payload["requested_additional_steps"] = requested_additional_steps
    if saved_bundle is not None:
        payload["saved_completed_steps"] = saved_bundle.completed_steps
        payload["saved_current_time"] = saved_bundle.current_time
        payload["saved_state_variables"] = sorted(saved_bundle.state_variables)
    return payload


def _sanitize_logged_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    try:
        resolved = resolved.resolve()
    except FileNotFoundError:
        resolved = resolved.absolute()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        pass
    try:
        return f"~/{resolved.relative_to(Path.home()).as_posix()}"
    except ValueError:
        return resolved.as_posix()


def _config_value(config, section: str, key: str, default: Any = None) -> Any:
    if config.has_option(section, key):
        return config.parsed(section, key)
    return default


def _config_string(
    config, section: str, key: str, default: str | None = None
) -> str | None:
    value = _config_value(config, section, key, default)
    if value is None:
        return None
    return str(value)


def _config_int(
    config, section: str, key: str, default: int | None = None
) -> int | None:
    value = _config_value(config, section, key, default)
    if value is None:
        return None
    return int(value)


def _config_bool(config, section: str, key: str, default: bool = False) -> bool:
    value = _config_value(config, section, key, default)
    return bool(value)


def _config_optional_bool(config, section: str, key: str) -> bool | None:
    if not config.has_option(section, key):
        return None
    return bool(config.parsed(section, key))


def _config_path(config, section: str, key: str) -> Path | None:
    value = _config_value(config, section, key)
    if value in (None, ""):
        return None
    return Path(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
