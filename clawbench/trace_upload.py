"""Verified private S3 publishing for native ShellBench trace bundles."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


TRACE_AUDIT_FILES = (
    "TRACE_COVERAGE_AUDIT.md",
    "trace_gaps.csv",
)
TRACE_VALIDATION_FILES = (
    "trajectory_validation.json",
    "trajectory_validation.full.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def collect_trace_bundle(run_dir: str | Path) -> list[tuple[Path, str]]:
    """Collect immutable trace archives and their audit metadata."""

    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {root}")

    selected: dict[str, Path] = {}
    raw_dir = root / "raw"
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*-artifacts.tar.gz")):
            selected[f"raw/{path.name}"] = path
    if not any(key.startswith("raw/") for key in selected):
        raise ValueError(f"no trace artifact archives found under {raw_dir}")

    summaries_dir = root / "summaries"
    for name in TRACE_AUDIT_FILES:
        path = summaries_dir / name
        if path.is_file():
            selected[f"summaries/{name}"] = path
    if summaries_dir.is_dir():
        for path in sorted(summaries_dir.glob("*/trajectory_validation*.json")):
            if path.name in TRACE_VALIDATION_FILES:
                relative = path.relative_to(root).as_posix()
                selected[relative] = path

    run_index = root / "manifests" / "run_index.json"
    if run_index.is_file():
        selected["manifests/run_index.json"] = run_index
    return [(selected[key], key) for key in sorted(selected)]


def trace_bundle_plan(run_dir: str | Path) -> dict[str, Any]:
    files = collect_trace_bundle(run_dir)
    return {
        "source_run": Path(run_dir).name,
        "file_count": len(files),
        "archive_count": sum(key.startswith("raw/") for _path, key in files),
        "total_bytes": sum(path.stat().st_size for path, _key in files),
        "files": [
            {"relative_key": key, "size": path.stat().st_size}
            for path, key in files
        ],
    }


def _not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", {})
    error = response.get("Error", {}) if isinstance(response, dict) else {}
    return str(error.get("Code") or "") in {"404", "NoSuchKey", "NotFound"}


def _object_matches(client: Any, bucket: str, key: str, size: int, digest: str) -> bool:
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _not_found(exc):
            return False
        raise
    return (
        int(head["ContentLength"]) == size
        and head.get("Metadata", {}).get("sha256") == digest
    )


def _build_client() -> tuple[Any, Any]:
    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "S3 trace upload requires boto3; install the project with .[s3]"
        ) from exc

    session = boto3.Session()
    client = session.client(
        "s3",
        config=Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            max_pool_connections=32,
        ),
    )
    transfer = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=4,
        use_threads=True,
    )
    return client, transfer


def upload_trace_bundle(
    run_dir: str | Path,
    *,
    bucket: str,
    prefix: str | None = None,
    workers: int = 4,
    client: Any | None = None,
    transfer_config: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Upload and verify a private trace bundle using the AWS credential chain."""

    root = Path(run_dir)
    files = collect_trace_bundle(root)
    if not bucket.strip():
        raise ValueError("bucket must not be empty")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    object_prefix = (prefix or root.name).strip("/")
    if not object_prefix:
        raise ValueError("prefix must not be empty")

    if client is None:
        client, transfer_config = _build_client()
    client.head_bucket(Bucket=bucket)

    records = []
    for path, relative_key in files:
        records.append(
            {
                "path": path,
                "relative_key": relative_key,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    lock = threading.Lock()
    completed = 0

    def upload(record: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        key = f"{object_prefix}/{record['relative_key']}"
        status = "skipped"
        if not _object_matches(
            client,
            bucket,
            key,
            int(record["size"]),
            str(record["sha256"]),
        ):
            kwargs = {
                "ExtraArgs": {
                    "ServerSideEncryption": "AES256",
                    "Metadata": {"sha256": str(record["sha256"])},
                }
            }
            if transfer_config is not None:
                kwargs["Config"] = transfer_config
            client.upload_file(str(record["path"]), bucket, key, **kwargs)
            status = "uploaded"

        if not _object_matches(
            client,
            bucket,
            key,
            int(record["size"]),
            str(record["sha256"]),
        ):
            raise RuntimeError(f"uploaded object failed verification: {key}")

        with lock:
            completed += 1
            if progress is not None:
                progress(f"{completed}/{len(records)} {status}: {record['relative_key']}")
        return {
            "key": key,
            "size": record["size"],
            "sha256": record["sha256"],
            "status": status,
        }

    uploaded = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(upload, record) for record in records]
        for future in as_completed(futures):
            uploaded.append(future.result())
    uploaded.sort(key=lambda item: str(item["key"]))

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "prefix": object_prefix,
        "source_run": root.name,
        "file_count": len(uploaded),
        "archive_count": sum(
            str(item["key"]).startswith(f"{object_prefix}/raw/")
            for item in uploaded
        ),
        "total_bytes": sum(int(item["size"]) for item in uploaded),
        "files": uploaded,
    }
    manifest_path = root / "summaries" / "S3_UPLOAD_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_body = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_body).hexdigest()
    manifest_key = f"{object_prefix}/S3_UPLOAD_MANIFEST.json"
    client.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=manifest_body,
        ServerSideEncryption="AES256",
        Metadata={"sha256": manifest_digest},
        ContentType="application/json",
    )
    if not _object_matches(
        client,
        bucket,
        manifest_key,
        len(manifest_body),
        manifest_digest,
    ):
        raise RuntimeError("upload manifest failed verification")

    return {
        "bucket": bucket,
        "prefix": object_prefix,
        "file_count": len(uploaded),
        "archive_count": manifest["archive_count"],
        "total_bytes": manifest["total_bytes"],
        "uploaded_count": sum(item["status"] == "uploaded" for item in uploaded),
        "skipped_count": sum(item["status"] == "skipped" for item in uploaded),
        "manifest_path": str(manifest_path),
        "manifest_key": manifest_key,
    }
