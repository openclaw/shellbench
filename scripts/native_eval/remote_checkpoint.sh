#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: remote_checkpoint.sh ROOT RUN_LABEL ARCHIVE_NAME" >&2
  exit 2
fi

ROOT="$1"
RUN_LABEL="$2"
ARCHIVE_NAME="$3"
JOB_DIR="$ROOT/results/jobs/$RUN_LABEL"
PROXY_DIR="$ROOT/proxy/$RUN_LABEL"
SNAPSHOT_DIR="$(mktemp -d "/tmp/$RUN_LABEL-checkpoint.XXXXXX")"
ARCHIVE_PATH="/tmp/$ARCHIVE_NAME"
TEMP_ARCHIVE="$ARCHIVE_PATH.tmp"

cleanup() {
  rm -rf "$SNAPSHOT_DIR"
  rm -f "$TEMP_ARCHIVE"
}
trap cleanup EXIT

case "$ARCHIVE_NAME" in
  "$RUN_LABEL"-checkpoint-[0-9][0-9][0-9][0-9]-artifacts.tar.gz) ;;
  *)
    echo "invalid checkpoint archive name: $ARCHIVE_NAME" >&2
    exit 2
    ;;
esac

mkdir -p \
  "$SNAPSHOT_DIR/results/jobs/$RUN_LABEL" \
  "$SNAPSHOT_DIR/shellbench_meta-$RUN_LABEL"

if [[ -d "$JOB_DIR" ]]; then
  rsync -a --exclude '*.tmp' "$JOB_DIR/" \
    "$SNAPSHOT_DIR/results/jobs/$RUN_LABEL/"
fi
if [[ -d "$PROXY_DIR" ]]; then
  mkdir -p "$SNAPSHOT_DIR/proxy/$RUN_LABEL"
  rsync -a --exclude '*.tmp' "$PROXY_DIR/" \
    "$SNAPSHOT_DIR/proxy/$RUN_LABEL/"
fi
if [[ -d "$ROOT/run-logs" ]]; then
  mkdir -p "$SNAPSHOT_DIR/run-logs"
  for suffix in stdout stderr; do
    source_log="$ROOT/run-logs/$RUN_LABEL.$suffix.log"
    if [[ -f "$source_log" ]]; then
      cp "$source_log" "$SNAPSHOT_DIR/run-logs/"
    fi
  done
fi

META_DIR="$SNAPSHOT_DIR/shellbench_meta-$RUN_LABEL"
date -u +%Y-%m-%dT%H:%M:%SZ > "$META_DIR/exported_at_utc.txt"
hostname > "$META_DIR/hostname.txt"
find "$SNAPSHOT_DIR/results/jobs/$RUN_LABEL" \
  -mindepth 2 -maxdepth 2 -name result.json \
  | wc -l | tr -d ' ' > "$META_DIR/result_json_count.txt"
cp "$JOB_DIR/run_manifest.json" "$META_DIR/run_manifest.json" 2>/dev/null || true
cp /opt/shellbench-native/manifest.json \
  "$META_DIR/toolchain_manifest.json" 2>/dev/null || true

tar -czf "$TEMP_ARCHIVE" -C "$SNAPSHOT_DIR" .
tar -tzf "$TEMP_ARCHIVE" >/dev/null
mv "$TEMP_ARCHIVE" "$ARCHIVE_PATH"
printf '%s\n' "$ARCHIVE_PATH"
