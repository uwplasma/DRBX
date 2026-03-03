from __future__ import annotations

import argparse

import jax.numpy as jnp

from jaxdrb.core.terms.registry import available_terms
from jaxdrb.core.geometry_registry import available_geometries
from jaxdrb.driver import build_system_from_config, run_simulation
from jaxdrb.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified JAX DRB solver (config-driven).")
    parser.add_argument("config", nargs="?", type=str, help="Path to input TOML file.")
    parser.add_argument(
        "--list-terms",
        action="store_true",
        help="List available RHS terms and exit.",
    )
    parser.add_argument(
        "--list-geometries",
        action="store_true",
        help="List available geometry kinds and exit.",
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List available solver engines and exit.",
    )
    parser.add_argument(
        "--compile-cache",
        type=str,
        default="$HOME/.cache/jaxdrb/compilation",
        help="Directory for JAX persistent compilation cache (use 'off' to disable).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run a time integration defined by the config (JIT/diffrax).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save npz outputs when using --run.",
    )
    parser.add_argument(
        "--output-minimal",
        action="store_true",
        help="Save minimal diagnostics (t + RMS) to avoid large host transfers.",
    )
    parser.add_argument(
        "--no-host-transfer",
        action="store_true",
        help="Skip host transfer of diagnostics (useful for quick runs).",
    )
    args = parser.parse_args()

    if args.list_terms:
        for name in available_terms():
            print(name)
        return
    if args.list_geometries:
        for spec in available_geometries():
            req = ", ".join(spec.required) if spec.required else "-"
            if spec.required_any:
                req_any = " | ".join("(" + " or ".join(group) + ")" for group in spec.required_any)
                req = req_any if req == "-" else f"{req}, {req_any}"
            opt = ", ".join(spec.optional) if spec.optional else "-"
            aliases = ", ".join(spec.aliases) if spec.aliases else "-"
            print(f"{spec.kind}: required=[{req}] optional=[{opt}] aliases=[{aliases}]")
        return
    if args.list_engines:
        for engine in ("unified", "parity_fv"):
            print(engine)
        return

    cache_opt = str(args.compile_cache).strip()
    if cache_opt and cache_opt.lower() not in ("off", "false", "0", "none"):
        import os
        from jax.experimental import compilation_cache

        cache_dir = os.path.expandvars(os.path.expanduser(cache_opt))
        os.makedirs(cache_dir, exist_ok=True)
        compilation_cache.compilation_cache.set_cache_dir(cache_dir)
        compilation_cache.compilation_cache.initialize_cache(cache_dir)

    if args.config is None:
        parser.error("config is required unless listing metadata")

    cfg = load_config(args.config)
    if args.run:
        if args.output:
            time_cfg = cfg.data.get("time", {})
            if not isinstance(time_cfg, dict):
                time_cfg = {}
            time_cfg = dict(time_cfg)
            time_cfg["return_numpy"] = True
            cfg.data["time"] = time_cfg
        # Avoid large host transfers unless explicitly requested.
        if args.output and args.output_minimal:
            time_cfg = cfg.data.get("time", {})
            if not isinstance(time_cfg, dict):
                time_cfg = {}
            time_cfg = dict(time_cfg)
            time_cfg["diag_mode"] = "rms_only"
            time_cfg["diag_phi_every"] = 0
            cfg.data["time"] = time_cfg

        result = run_simulation(
            cfg.data,
            as_numpy=(
                False if (args.no_host_transfer or args.output_minimal) else bool(args.output)
            ),
        )
        print("Run complete.")
        if args.output:
            import numpy as np
            import jax

            state = jax.device_get(result.final_state)
            built = build_system_from_config(cfg.data)
            n_phys = built.system._phys_n(jnp.asarray(state.n))
            Te_phys = built.system._phys_Te(jnp.asarray(state.Te))
            phi = built.system._phi_from_omega(
                jnp.asarray(state.omega),
                n=n_phys,
                Ti=getattr(state, "Ti", None),
                Te=Te_phys,
            )
            diagnostics = result.diagnostics
            if args.output_minimal and diagnostics:
                keep = {"t", "rms_n", "rms_Te", "rms_omega", "rms_phi", "times"}
                diagnostics = {
                    k: np.asarray(jax.device_get(v)) for k, v in diagnostics.items() if k in keep
                }
            else:
                diagnostics = {k: np.asarray(jax.device_get(v)) for k, v in diagnostics.items()}

            np.savez(
                args.output,
                **diagnostics,
                snapshot_n=np.asarray(state.n),
                snapshot_omega=np.asarray(state.omega),
                snapshot_Te=np.asarray(state.Te),
                snapshot_vpar_e=np.asarray(state.vpar_e),
                snapshot_vpar_i=np.asarray(state.vpar_i),
                snapshot_phi=np.asarray(jax.device_get(phi)),
                snapshot_Ti=np.asarray(state.Ti) if state.Ti is not None else None,
                snapshot_psi=np.asarray(state.psi) if state.psi is not None else None,
                snapshot_N=np.asarray(state.N) if state.N is not None else None,
            )
        if args.no_host_transfer:
            return
        return

    built = build_system_from_config(cfg.data)
    dy = built.system.rhs(0.0, built.state)

    def norm(arr):
        return float(jnp.sqrt(jnp.mean(jnp.abs(arr) ** 2)))

    print("DRBSystem initialized.")
    print(
        {
            "n": norm(dy.n),
            "omega": norm(dy.omega),
            "vpar_e": norm(dy.vpar_e),
            "vpar_i": norm(dy.vpar_i),
            "Te": norm(dy.Te),
        }
    )


if __name__ == "__main__":
    main()
