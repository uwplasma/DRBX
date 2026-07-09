#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_all_analysis.sh --run-name NAME --output-path PATH --frame-stride N

Runs the EB blob analysis scripts that share the run-name/output-path
contract in sequence.
EOF
}

RUN_NAME="EB_perp_diffusion"
OUTPUT_PATH=""
FRAME_STRIDE=2
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="${2:?missing value for --run-name}"
      shift 2
      ;;
    --output-path)
      OUTPUT_PATH="${2:?missing value for --output-path}"
      shift 2
      ;;
    --frame-stride)
      FRAME_STRIDE="${2:?missing value for --frame-stride}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="${SCRIPT_DIR}/${RUN_NAME}"
fi

run_script() {
  local script_name="$1"
  shift
  echo "==> ${script_name}" >&2
  "${PYTHON_BIN}" "${SCRIPT_DIR}/${script_name}" "$@"
}

run_script analyze_ve.py --run-name "$RUN_NAME" --output-path "$OUTPUT_PATH"
run_script analyze_vi.py --run-name "$RUN_NAME" --output-path "$OUTPUT_PATH"
run_script analyze_te.py --run-name "$RUN_NAME" --output-path "$OUTPUT_PATH" --frame-stride "$FRAME_STRIDE"
run_script analyze_ti.py --run-name "$RUN_NAME" --output-path "$OUTPUT_PATH" --frame-stride "$FRAME_STRIDE"
run_script analyze_density.py --run-name "$RUN_NAME" --output-path "$OUTPUT_PATH" --frame-stride "$FRAME_STRIDE"
run_script analyze_vorticity.py --run-name "$RUN_NAME" --output-path "$OUTPUT_PATH"
run_script analyze_parallel_operators.py --run-name "$RUN_NAME" --output-path "$OUTPUT_PATH" --frame-stride "$FRAME_STRIDE"
run_script make_movie.py --run-name "$RUN_NAME" --output-path "$OUTPUT_PATH" --frame-stride "$FRAME_STRIDE"
run_script analyze_EB_density.py --run-name "$RUN_NAME" --output-path "$OUTPUT_PATH" --movie-frame-stride "$FRAME_STRIDE"
