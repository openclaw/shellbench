#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: remote_run.sh ROOT TASKS_ROOT ENV_FILE RUN_LABEL HARNESS MODEL_SLUG REP EXPECTED_COUNT PUBLIC_TASKS_COMMIT RUN_DATE CONCURRENCY HARNESS_VERSION MODEL_ID MODEL_PROVIDER PROXY_MODEL_NAME RERUN_OF_CANONICAL_RUN [TASK_NAME ...]" >&2
  exit 2
}

[[ $# -ge 16 ]] || usage

ROOT="$1"
TASKS_ROOT="$2"
ENV_FILE="$3"
RUN_LABEL="$4"
HARNESS="$5"
MODEL_SLUG="$6"
REPETITION="$7"
EXPECTED_TASK_COUNT="$8"
PUBLIC_TASKS_COMMIT="$9"
RUN_DATE="${10}"
CONCURRENCY="${11}"
HARNESS_VERSION="${12}"
MODEL_ID="${13}"
MODEL_PROVIDER="${14}"
PROXY_MODEL_NAME="${15}"
RERUN_OF_CANONICAL_RUN="${16}"
shift 16
TASK_NAMES=("$@")
TASK_SUITE_PATH="combined tasks/tasks"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-/opt/shellbench-native}"
PROXY_DIR="$ROOT/proxy/$RUN_LABEL"
PROXY_CONFIG="$PROXY_DIR/config.json"
PROXY_LOG="$PROXY_DIR/proxy.log"
PROXY_PID=""
RUN_STATE_DIR="/tmp/shellbench-runs/$RUN_LABEL"
RUN_STATUS=1

# shellcheck disable=SC2329
stop_proxy() {
  [[ -n "$PROXY_PID" ]] || return
  kill -0 "$PROXY_PID" 2>/dev/null || return
  kill "$PROXY_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$PROXY_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$PROXY_PID" 2>/dev/null; then
    kill -KILL "$PROXY_PID" 2>/dev/null || true
  fi
  wait "$PROXY_PID" 2>/dev/null || true
}

# shellcheck disable=SC2329
cleanup() {
  stop_proxy
  mkdir -p "$RUN_STATE_DIR"
  printf '%s\n' "$RUN_STATUS" > "$RUN_STATE_DIR/exit_status"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_STATE_DIR/finished_at_utc"
  touch "$RUN_STATE_DIR/done"
}
trap cleanup EXIT

umask 077
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

mkdir -p \
  "$ROOT/results/jobs" \
  "$PROXY_DIR" \
  /tmp/"shellbench_meta-$RUN_LABEL" \
  "$RUN_STATE_DIR"
printf '%s\n' "$$" > "$RUN_STATE_DIR/pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_STATE_DIR/started_at_utc"
printf '%s\n' "running" > "$RUN_STATE_DIR/state"
export SHELLBENCH_PROXY_KEY="${SHELLBENCH_PROXY_KEY:-$(openssl rand -hex 32)}"

cd "$ROOT/runner"
python3 - <<PY
from pathlib import Path
from scripts.native_eval.proxy import write_proxy_config
write_proxy_config(Path(${PROXY_CONFIG@Q}))
PY

"$TOOLCHAIN_ROOT/litellm-venv/bin/litellm" \
  --config "$PROXY_CONFIG" \
  --host 0.0.0.0 \
  --port 4000 \
  >"$PROXY_LOG" 2>&1 &
PROXY_PID="$!"

for _ in $(seq 1 120); do
  if curl -fsS \
    -H "Authorization: Bearer $SHELLBENCH_PROXY_KEY" \
    http://127.0.0.1:4000/health/liveliness >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "LiteLLM proxy exited during startup" >&2
    tail -100 "$PROXY_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done

curl -fsS \
  -H "Authorization: Bearer $SHELLBENCH_PROXY_KEY" \
  http://127.0.0.1:4000/health/liveliness >/dev/null

RUN_COMMAND=(
  python3 -m scripts.native_eval.run_job
  --tasks-root "$TASKS_ROOT"
  --jobs-dir "$ROOT/results/jobs"
  --run-label "$RUN_LABEL"
  --harness "$HARNESS"
  --harness-version "$HARNESS_VERSION"
  --model-slug "$MODEL_SLUG"
  --model-id "$MODEL_ID"
  --model-provider "$MODEL_PROVIDER"
  --proxy-model-name "$PROXY_MODEL_NAME"
  --repetition "$REPETITION"
  --expected-task-count "$EXPECTED_TASK_COUNT"
  --public-tasks-commit "$PUBLIC_TASKS_COMMIT"
  --task-suite-path "$TASK_SUITE_PATH"
  --run-date "$RUN_DATE"
  --toolchain-root "$TOOLCHAIN_ROOT"
  --proxy-url "http://host.docker.internal:4000"
  --concurrency "$CONCURRENCY"
)
if [[ -n "$RERUN_OF_CANONICAL_RUN" ]]; then
  RUN_COMMAND+=(--rerun-of-canonical-run "$RERUN_OF_CANONICAL_RUN")
fi
for task_name in "${TASK_NAMES[@]}"; do
  RUN_COMMAND+=(--task "$task_name")
done

set +e
"${RUN_COMMAND[@]}"
RUN_STATUS="$?"
set -e

META_DIR="/tmp/shellbench_meta-$RUN_LABEL"
date -u +%Y-%m-%dT%H:%M:%SZ > "$META_DIR/exported_at_utc.txt"
hostname > "$META_DIR/hostname.txt"
git -C "$ROOT/runner" rev-parse HEAD > "$META_DIR/runner_commit.txt" || true
find "$ROOT/results/jobs/$RUN_LABEL" -mindepth 2 -maxdepth 2 -name result.json \
  | wc -l | tr -d ' ' > "$META_DIR/result_json_count.txt"
find "$ROOT/results/jobs/$RUN_LABEL" -mindepth 1 -maxdepth 1 -type d -print \
  > "$META_DIR/trial_dirs.txt"
cp "$ROOT/results/jobs/$RUN_LABEL/run_manifest.json" \
  "$META_DIR/run_manifest.json" 2>/dev/null || true
cp "$TOOLCHAIN_ROOT/manifest.json" \
  "$META_DIR/toolchain_manifest.json" 2>/dev/null || true
sha256sum "$ROOT/results/jobs/$RUN_LABEL/run_manifest.json" \
  > "$META_DIR/run_manifest.sha256" 2>/dev/null || true

ARCHIVE_ITEMS=(
  -C "$ROOT" "results/jobs/$RUN_LABEL"
  -C "$ROOT" "proxy/$RUN_LABEL"
)
for suffix in stdout stderr; do
  if [[ -f "$ROOT/run-logs/$RUN_LABEL.$suffix.log" ]]; then
    ARCHIVE_ITEMS+=(-C "$ROOT" "run-logs/$RUN_LABEL.$suffix.log")
  fi
done
ARCHIVE_ITEMS+=(-C /tmp "shellbench_meta-$RUN_LABEL")
sudo tar -czf "/tmp/$RUN_LABEL-final-artifacts.tar.gz" "${ARCHIVE_ITEMS[@]}"
sudo chown "$(id -u):$(id -g)" "/tmp/$RUN_LABEL-final-artifacts.tar.gz"

exit "$RUN_STATUS"
