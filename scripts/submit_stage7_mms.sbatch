#!/usr/bin/env bash

# Allocation-neutral launcher for the HSX/RLP Stage-7 MMS campaigns. The
# operator chooses the host, scheduler, account, wall time, CPUs, and GPUs
# before invoking this script; no Slurm allocation is requested here.
#
# Required production inputs may be overridden through the environment:
#   DRBX_ROOT, MAKEGRID, VESSEL, OUTPUT_ROOT, PYTHON_BIN, CAMPAIGN_KIND
# Optional cache overrides:
#   METRIC_CACHE_DIR, JAX_CACHE_DIR
#
# DRY_RUN=1 performs only shell-side command/manifest construction.  It does
# not inspect JAX devices, import the production driver, or build
# geometry.  This makes the launcher testable on a login node or workstation.

set -euo pipefail

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
DRBX_ROOT=${DRBX_ROOT:-$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)}
MAKEGRID=${MAKEGRID:-"$(dirname "$DRBX_ROOT")/mgrid_res2p5cm_180pln.nc"}
VESSEL=${VESSEL:-"$(dirname "$DRBX_ROOT")/vessel_hsx_flare.txt"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$DRBX_ROOT/work/stage7_mms"}
PYTHON_BIN=${PYTHON_BIN:-python}
CAMPAIGN_KIND=${CAMPAIGN_KIND:-frozen}
DRY_RUN=${DRY_RUN:-0}
METRIC_CACHE_DIR=${METRIC_CACHE_DIR:-"$OUTPUT_ROOT/cache/metric_64"}
JAX_CACHE_DIR=${JAX_CACHE_DIR:-"$OUTPUT_ROOT/cache/jax"}

case "$DRY_RUN" in
  0|1) ;;
  *) printf 'ERROR: DRY_RUN must be 0 or 1, got %q\n' "$DRY_RUN" >&2; exit 2 ;;
esac

case "$CAMPAIGN_KIND" in
  frozen)
    CAMPAIGN_NAME=frozen_32_48_64
    RESOLUTIONS=32,48,64
    START_TIME=1e-6
    FINAL_TIME=1e-6
    TIMESTEP=1e-6
    SAVE_EVERY=1
    ;;
  evolved)
    CAMPAIGN_NAME=evolved_32_48_64_t2e-5_dt1e-6
    RESOLUTIONS=32,48,64
    START_TIME=0
    FINAL_TIME=2e-5
    TIMESTEP=1e-6
    SAVE_EVERY=1
    ;;
  temporal_5e-7)
    CAMPAIGN_NAME=evolved_N64_t2e-5_dt5e-7
    RESOLUTIONS=64
    START_TIME=0
    FINAL_TIME=2e-5
    TIMESTEP=5e-7
    SAVE_EVERY=2
    ;;
  temporal_2p5e-7)
    CAMPAIGN_NAME=evolved_N64_t2e-5_dt2p5e-7
    RESOLUTIONS=64
    START_TIME=0
    FINAL_TIME=2e-5
    TIMESTEP=2.5e-7
    SAVE_EVERY=4
    ;;
  *)
    printf 'ERROR: unsupported CAMPAIGN_KIND=%q\n' "$CAMPAIGN_KIND" >&2
    printf 'Expected frozen, evolved, temporal_5e-7, or temporal_2p5e-7.\n' >&2
    exit 2
    ;;
esac

DRIVER="$DRBX_ROOT/simulate_hsx_mms.py"
ANALYZER="$DRBX_ROOT/scripts/analyze_hsx_mms.py"
CAMPAIGN_DIR="$OUTPUT_ROOT/$CAMPAIGN_NAME"
LOG_DIR="$CAMPAIGN_DIR/logs"
AGGREGATE="$CAMPAIGN_DIR/${CAMPAIGN_NAME}.npz"
MANIFEST="$CAMPAIGN_DIR/run_manifest.txt"
PREREQUISITE_GATE_REPORT="$CAMPAIGN_DIR/prerequisite_gate.json"

FROZEN_CAMPAIGN_DIR="$OUTPUT_ROOT/frozen_32_48_64"
FROZEN_AGGREGATE="$FROZEN_CAMPAIGN_DIR/frozen_32_48_64.npz"
FROZEN_MANIFEST="$FROZEN_CAMPAIGN_DIR/run_manifest.txt"
EVOLVED_CAMPAIGN_DIR="$OUTPUT_ROOT/evolved_32_48_64_t2e-5_dt1e-6"
EVOLVED_AGGREGATE="$EVOLVED_CAMPAIGN_DIR/evolved_32_48_64_t2e-5_dt1e-6.npz"
EVOLVED_MANIFEST="$EVOLVED_CAMPAIGN_DIR/run_manifest.txt"

mkdir -p "$LOG_DIR" "$METRIC_CACHE_DIR" "$JAX_CACHE_DIR"

# Protect completed campaign evidence before installing the EXIT trap or
# rewriting any manifest/timestamp files.  An accidental resubmission must
# leave the successful aggregate, source hashes, and exit status untouched.
if [[ "$DRY_RUN" == 0 && -e "$AGGREGATE" ]]; then
  printf 'ERROR: aggregate already exists; choose a fresh OUTPUT_ROOT: %s\n' "$AGGREGATE" >&2
  exit 6
fi

finish() {
  status=$?
  set +e
  printf '%s\n' "$status" > "$CAMPAIGN_DIR/run_exit_status.txt"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$CAMPAIGN_DIR/end_time_utc.txt"
}
trap finish EXIT

for required in \
  "$DRIVER" \
  "$DRBX_ROOT/hsx_mms_continuum_reference.py" \
  "$DRBX_ROOT/simulate_hsx_blob.py" \
  "$DRBX_ROOT/scripts/analyze_hsx_mms.py" \
  "$DRBX_ROOT/src/drbx/native/fci_drb_EB_rhs.py"; do
  if [[ ! -f "$required" ]]; then
    printf 'ERROR: required source file is missing: %s\n' "$required" >&2
    exit 3
  fi
done

if [[ ! -f "$MAKEGRID" ]]; then
  printf 'ERROR: MAKEGRID input is missing: %s\n' "$MAKEGRID" >&2
  exit 3
fi
if [[ ! -f "$VESSEL" ]]; then
  printf 'ERROR: VESSEL input is missing: %s\n' "$VESSEL" >&2
  exit 3
fi

PYTHON_ARGS=(
  "$DRIVER"
  --makegrid "$MAKEGRID"
  --vessel "$VESSEL"
  --metric-cache-dir "$METRIC_CACHE_DIR"
  --resolutions "$RESOLUTIONS"
  --shard-counts 1 1 4
  --time "$START_TIME"
  --final-time "$FINAL_TIME"
  --dt "$TIMESTEP"
  --advance-execution compiled
  --save-every "$SAVE_EVERY"
  --output "$AGGREGATE"
)
RUN_COMMAND=(
  "$PYTHON_BIN" -u
  "${PYTHON_ARGS[@]}"
)

PREREQUISITE_GATE_MODE=none
PREREQUISITE_AGGREGATES=()
PREREQUISITE_MANIFESTS=()
PREREQUISITE_GATE_COMMAND=()
case "$CAMPAIGN_KIND" in
  evolved)
    PREREQUISITE_GATE_MODE=spatial
    PREREQUISITE_AGGREGATES=("$FROZEN_AGGREGATE")
    PREREQUISITE_MANIFESTS=("$FROZEN_MANIFEST")
    PREREQUISITE_GATE_COMMAND=(
      "$PYTHON_BIN" -u "$ANALYZER"
      "$FROZEN_AGGREGATE"
      --require-spatial
      --output "$PREREQUISITE_GATE_REPORT"
    )
    ;;
  temporal_5e-7|temporal_2p5e-7)
    PREREQUISITE_GATE_MODE=evolved
    PREREQUISITE_AGGREGATES=("$FROZEN_AGGREGATE" "$EVOLVED_AGGREGATE")
    PREREQUISITE_MANIFESTS=("$FROZEN_MANIFEST" "$EVOLVED_MANIFEST")
    PREREQUISITE_GATE_COMMAND=(
      "$PYTHON_BIN" -u "$ANALYZER"
      "$FROZEN_AGGREGATE" "$EVOLVED_AGGREGATE"
      --require-evolved
      --output "$PREREQUISITE_GATE_REPORT"
    )
    ;;
esac

write_shell_command() {
  printf '%q ' "$@"
  printf '\n'
}

write_hashes() {
  # Hash relative names so manifests from different checkout roots remain
  # directly comparable.  The NUL-delimited find/read path and quoted arrays
  # preserve every whitespace character in a valid source filename.  Sort in
  # Bash rather than relying on non-portable `sort -z` (missing from the
  # default macOS toolchain used by the dry-run tests).
  local LC_ALL=C
  local -a hash_inputs
  local source_path source_key
  local source_index source_position
  hash_inputs=(
    "scripts/submit_stage7_mms.sbatch"
    "simulate_hsx_mms.py"
    "hsx_mms_continuum_reference.py"
    "simulate_hsx_blob.py"
    "scripts/analyze_hsx_mms.py"
  )
  while IFS= read -r -d '' source_path; do
    hash_inputs+=("${source_path#"$DRBX_ROOT"/}")
  done < <(find "$DRBX_ROOT/src/drbx" -type f -name '*.py' -print0)

  # Stable bytewise insertion sort; the source list is small enough that the
  # quadratic shell implementation is negligible beside source hashing.
  for ((source_index = 1; source_index < ${#hash_inputs[@]}; source_index++)); do
    source_key=${hash_inputs[$source_index]}
    source_position=$((source_index - 1))
    while ((source_position >= 0)) && [[ ${hash_inputs[$source_position]} > "$source_key" ]]; do
      hash_inputs[$((source_position + 1))]=${hash_inputs[$source_position]}
      source_position=$((source_position - 1))
    done
    hash_inputs[$((source_position + 1))]=$source_key
  done

  if command -v shasum >/dev/null 2>&1; then
    (cd "$DRBX_ROOT" && shasum -a 256 "${hash_inputs[@]}")
  elif command -v sha256sum >/dev/null 2>&1; then
    (cd "$DRBX_ROOT" && sha256sum "${hash_inputs[@]}")
  else
    printf 'ERROR: shasum or sha256sum is required for the source manifest\n' >&2
    return 4
  fi
}

file_sha256() {
  local path=$1
  if [[ ! -f "$path" ]]; then
    printf 'missing'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    printf 'ERROR: shasum or sha256sum is required for prerequisite hashing\n' >&2
    return 4
  fi
}

source_hash_block() {
  local manifest=$1
  awk '
    /^sha256_begin$/ { inside=1; next }
    /^sha256_end$/ { if (inside) { found=1; exit } }
    inside { print }
    END { if (!found) exit 1 }
  ' "$manifest"
}

validate_prerequisites() {
  local current_hashes prerequisite_hashes
  local prerequisite_index prerequisite_aggregate prerequisite_manifest
  local prerequisite_status

  current_hashes=$(source_hash_block "$MANIFEST") || {
    printf 'ERROR: current manifest has no complete deterministic source-hash block: %s\n' "$MANIFEST" >&2
    return 7
  }
  for ((prerequisite_index = 0; prerequisite_index < ${#PREREQUISITE_AGGREGATES[@]}; prerequisite_index++)); do
    prerequisite_aggregate=${PREREQUISITE_AGGREGATES[$prerequisite_index]}
    prerequisite_manifest=${PREREQUISITE_MANIFESTS[$prerequisite_index]}
    if [[ ! -f "$prerequisite_aggregate" ]]; then
      printf 'ERROR: required prerequisite aggregate is missing: %s\n' "$prerequisite_aggregate" >&2
      return 7
    fi
    if [[ ! -f "$prerequisite_manifest" ]]; then
      printf 'ERROR: required prerequisite manifest is missing: %s\n' "$prerequisite_manifest" >&2
      return 7
    fi
    prerequisite_status="$(dirname "$prerequisite_manifest")/run_exit_status.txt"
    if [[ ! -f "$prerequisite_status" ]] || [[ $(<"$prerequisite_status") != 0 ]]; then
      printf 'ERROR: prerequisite campaign does not record exit status 0: %s\n' "$prerequisite_status" >&2
      return 7
    fi
    prerequisite_hashes=$(source_hash_block "$prerequisite_manifest") || {
      printf 'ERROR: prerequisite manifest has no complete deterministic source-hash block: %s\n' "$prerequisite_manifest" >&2
      return 7
    }
    if [[ "$prerequisite_hashes" != "$current_hashes" ]]; then
      printf 'ERROR: prerequisite/current deterministic source-hash blocks differ: %s\n' "$prerequisite_manifest" >&2
      return 7
    fi
  done
}

{
  printf 'schema=hsx-rlp-stage7-mms-launch-v1\n'
  printf 'prepared_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf 'campaign_kind=%s\n' "$CAMPAIGN_KIND"
  printf 'campaign_name=%s\n' "$CAMPAIGN_NAME"
  printf 'dry_run=%s\n' "$DRY_RUN"
  printf 'drbx_root=%s\n' "$DRBX_ROOT"
  printf 'makegrid=%s\n' "$MAKEGRID"
  printf 'vessel=%s\n' "$VESSEL"
  printf 'output_root=%s\n' "$OUTPUT_ROOT"
  printf 'aggregate=%s\n' "$AGGREGATE"
  printf 'metric_cache_dir=%s\n' "$METRIC_CACHE_DIR"
  printf 'jax_cache_dir=%s\n' "$JAX_CACHE_DIR"
  printf 'python_bin=%s\n' "$PYTHON_BIN"
  printf 'resolutions=%s\n' "$RESOLUTIONS"
  printf 'shard_counts=1,1,4\n'
  printf 'start_time=%s\n' "$START_TIME"
  printf 'final_time=%s\n' "$FINAL_TIME"
  printf 'dt=%s\n' "$TIMESTEP"
  printf 'save_every=%s\n' "$SAVE_EVERY"
  printf 'prerequisite_gate_mode=%s\n' "$PREREQUISITE_GATE_MODE"
  printf 'prerequisite_gate_report=%s\n' "$PREREQUISITE_GATE_REPORT"
  printf 'prerequisite_count=%s\n' "${#PREREQUISITE_AGGREGATES[@]}"
  for ((prerequisite_index = 0; prerequisite_index < ${#PREREQUISITE_AGGREGATES[@]}; prerequisite_index++)); do
    prerequisite_aggregate=${PREREQUISITE_AGGREGATES[$prerequisite_index]}
    prerequisite_manifest=${PREREQUISITE_MANIFESTS[$prerequisite_index]}
    printf 'prerequisite_aggregate_%s=%s\n' "$prerequisite_index" "$prerequisite_aggregate"
    printf 'prerequisite_aggregate_sha256_%s=%s\n' \
      "$prerequisite_index" "$(file_sha256 "$prerequisite_aggregate")"
    printf 'prerequisite_manifest_%s=%s\n' "$prerequisite_index" "$prerequisite_manifest"
  done
  printf 'prerequisite_gate_command_shell='
  if ((${#PREREQUISITE_GATE_COMMAND[@]})); then
    write_shell_command "${PREREQUISITE_GATE_COMMAND[@]}"
  else
    printf 'none\n'
  fi
  printf 'scheduler_allocation=operator-managed\n'
  printf 'launcher_requires_visible_gpus=4\n'
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-not-applicable}"
  printf 'command_shell='
  write_shell_command "${RUN_COMMAND[@]}"
  if git -C "$DRBX_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'git_commit=%s\n' "$(git -C "$DRBX_ROOT" rev-parse HEAD)"
    printf '%s\n' 'git_status_begin'
    git -C "$DRBX_ROOT" status --short
    printf '%s\n' 'git_status_end'
  else
    printf 'git_commit=not-a-git-worktree\n'
  fi
  printf '%s\n' 'sha256_begin'
  write_hashes
  printf '%s\n' 'sha256_end'
} > "$MANIFEST"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$CAMPAIGN_DIR/start_time_utc.txt"

if [[ "$DRY_RUN" == 1 ]]; then
  printf '[stage7-mms] dry run; no GPU/Python command was launched.\n'
  printf '[stage7-mms] campaign=%s\n' "$CAMPAIGN_KIND"
  printf '[stage7-mms] manifest=%s\n' "$MANIFEST"
  printf '[stage7-mms] command='
  write_shell_command "${RUN_COMMAND[@]}"
  printf '[stage7-mms] prerequisite-gate='
  if ((${#PREREQUISITE_GATE_COMMAND[@]})); then
    write_shell_command "${PREREQUISITE_GATE_COMMAND[@]}"
  else
    printf 'none\n'
  fi
  exit 0
fi

if ((${#PREREQUISITE_GATE_COMMAND[@]})); then
  validate_prerequisites
  "${PREREQUISITE_GATE_COMMAND[@]}" \
    > "$LOG_DIR/prerequisite_gate.stdout.log" \
    2> "$LOG_DIR/prerequisite_gate.stderr.log"
fi

export DRBX_CACHE_DIR="$JAX_CACHE_DIR"
export OMP_NUM_THREADS=64
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda,cpu
export PYTHONPATH="$DRBX_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -u -c \
  'import jax; devices=jax.devices("gpu"); print(f"jax={jax.__version__} gpu_devices={devices}", flush=True); assert len(devices) == 4, devices' \
  > "$LOG_DIR/gpu_preflight.log" 2>&1

"$PYTHON_BIN" -u "$DRIVER" \
  --shard-counts 1 1 4 --self-test --wiring-only \
  > "$LOG_DIR/mms_preflight.log" 2>&1

"${RUN_COMMAND[@]}" \
  > "$LOG_DIR/stdout.log" \
  2> "$LOG_DIR/stderr.log"
