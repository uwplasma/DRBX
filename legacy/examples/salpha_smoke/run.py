from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from jaxdrb.driver import build_system_from_config
from jaxdrb.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="s-alpha smoke test (RHS eval)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("input.toml")),
        help="Path to input TOML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    built = build_system_from_config(cfg.data)
    dy = built.system.rhs(0.0, built.state)

    def norm(a):
        return float(jnp.sqrt(jnp.mean(jnp.abs(a) ** 2)))

    report = {
        "n": norm(dy.n),
        "omega": norm(dy.omega),
        "vpar_e": norm(dy.vpar_e),
        "vpar_i": norm(dy.vpar_i),
        "Te": norm(dy.Te),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
