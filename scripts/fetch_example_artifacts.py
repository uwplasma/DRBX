from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drbx.runtime.artifacts import ensure_docs_media, ensure_hsx_geometry  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore release-backed example artifacts for a lightweight drbx checkout. "
            "Use GitHub CLI authentication or GH_TOKEN/GITHUB_TOKEN for private releases."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="drbx checkout root.",
    )
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="Do not restore docs figures, movies, and NPZ arrays.",
    )
    parser.add_argument(
        "--hsx-geometry",
        action="store_true",
        help="Also restore the canonical 32-cubed HSX geometry bundle for simulate_hsx_blob.py.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract artifacts even if sentinels exist.",
    )
    return parser.parse_args()


def _auth_hint() -> str:
    return (
        "Artifact download failed. For this private repository, authenticate with "
        "`gh auth login --hostname github.com` or set `GH_TOKEN`/`GITHUB_TOKEN` "
        "to a token with access to `uwplasma/drbx`."
    )


def main() -> None:
    args = _parse_args()
    root = args.root.expanduser().resolve()
    if args.skip_media and not args.hsx_geometry:
        raise SystemExit("Nothing to fetch: --skip-media was set.")

    try:
        if not args.skip_media:
            docs_data = ensure_docs_media(root=root, force=args.force)
            print(f"Restored docs media under {docs_data}")
        if args.hsx_geometry:
            geometry_dir = ensure_hsx_geometry(root=root, force=args.force)
            print(f"Restored HSX geometry bundle under {geometry_dir}")
    except Exception as error:
        print(_auth_hint(), file=sys.stderr)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
