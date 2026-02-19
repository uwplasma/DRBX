#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOLVER_DIR=""
BIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --solver-dir)
      SOLVER_DIR="$2"
      shift 2
      ;;
    --bin)
      BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SOLVER_DIR" ]]; then
  echo "Usage: $0 --solver-dir /path/to/external/solver --bin /path/to/solver/binary" >&2
  exit 1
fi

if [[ -z "$BIN" ]]; then
  echo "Usage: $0 --solver-dir /path/to/external/solver --bin /path/to/solver/binary" >&2
  exit 1
fi

EXAMPLE_DIR="$SOLVER_DIR/examples_min/salpha_grid"
GEN_SCRIPT="$EXAMPLE_DIR/generate_salpha_grid.py"
GRID_FILE="$EXAMPLE_DIR/salpha.nc"

if [[ ! -x "$BIN" ]]; then
  echo "solver binary not found at: $BIN" >&2
  echo "Build the solver binary first." >&2
  exit 1
fi

if [[ ! -f "$GEN_SCRIPT" ]]; then
  echo "Grid generator not found: $GEN_SCRIPT" >&2
  exit 1
fi

python "$GEN_SCRIPT" --file "$GRID_FILE" --nx 32 --ny 32 --R 2.0 --r0 0.2 --dr 0.05 --Bt 1.0 --q0 2.0 --shat 1.0 --alpha 0.0

cd "$EXAMPLE_DIR"
"$BIN" -d "$EXAMPLE_DIR"

echo "Reference run complete. Outputs in $EXAMPLE_DIR"
