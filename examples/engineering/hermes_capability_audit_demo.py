from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import write_hermes_capability_audit


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the maintained Hermes capability audit for jax_drb.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_repo_root() / "docs" / "data" / "hermes_capability_audit.json",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_hermes_capability_audit(args.output)
    if args.quiet:
        return
    print("\n== Hermes Capability Audit ==")
    print(f"  - output: {output}")


if __name__ == "__main__":
    main()
