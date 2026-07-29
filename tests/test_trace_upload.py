import json
import threading
from pathlib import Path

from click.testing import CliRunner

from clawbench.cli import cli
from clawbench.trace_upload import collect_trace_bundle, upload_trace_bundle


class MissingObjectError(Exception):
    response = {"Error": {"Code": "404"}}


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict] = {}
        self.lock = threading.Lock()

    def head_bucket(self, *, Bucket: str) -> None:
        assert Bucket

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        with self.lock:
            try:
                stored = self.objects[(Bucket, Key)]
            except KeyError as exc:
                raise MissingObjectError from exc
            return {
                "ContentLength": len(stored["body"]),
                "Metadata": stored["metadata"],
            }

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict,
        Config=None,
    ) -> None:
        del Config
        with self.lock:
            self.objects[(bucket, key)] = {
                "body": Path(filename).read_bytes(),
                "metadata": ExtraArgs["Metadata"],
                "encryption": ExtraArgs["ServerSideEncryption"],
            }

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ServerSideEncryption: str,
        Metadata: dict,
        ContentType: str,
    ) -> None:
        with self.lock:
            self.objects[(Bucket, Key)] = {
                "body": Body,
                "metadata": Metadata,
                "encryption": ServerSideEncryption,
                "content_type": ContentType,
            }


def _trace_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "runs-full-example"
    raw = root / "raw"
    raw.mkdir(parents=True)
    (raw / "run-a-checkpoint-0001-artifacts.tar.gz").write_bytes(b"checkpoint")
    (raw / "run-a-final-artifacts.tar.gz").write_bytes(b"final")
    (raw / "ignore.txt").write_text("not uploaded", encoding="utf-8")

    summaries = root / "summaries"
    (summaries / "low").mkdir(parents=True)
    (summaries / "TRACE_COVERAGE_AUDIT.md").write_text("audit\n", encoding="utf-8")
    (summaries / "trace_gaps.csv").write_text("run_label\n", encoding="utf-8")
    (summaries / "low" / "trajectory_validation.full.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (summaries / "unrelated.json").write_text("{}\n", encoding="utf-8")

    manifests = root / "manifests"
    manifests.mkdir()
    (manifests / "run_index.json").write_text('{"runs":[]}\n', encoding="utf-8")
    return root


def test_collect_trace_bundle_selects_archives_and_audit_metadata(tmp_path: Path):
    root = _trace_fixture(tmp_path)

    selected = collect_trace_bundle(root)

    assert [key for _path, key in selected] == [
        "manifests/run_index.json",
        "raw/run-a-checkpoint-0001-artifacts.tar.gz",
        "raw/run-a-final-artifacts.tar.gz",
        "summaries/TRACE_COVERAGE_AUDIT.md",
        "summaries/low/trajectory_validation.full.json",
        "summaries/trace_gaps.csv",
    ]


def test_upload_trace_bundle_verifies_and_skips_existing_objects(tmp_path: Path):
    root = _trace_fixture(tmp_path)
    client = FakeS3Client()

    first = upload_trace_bundle(
        root,
        bucket="private-bucket",
        prefix="matrix/example",
        workers=2,
        client=client,
    )
    second = upload_trace_bundle(
        root,
        bucket="private-bucket",
        prefix="matrix/example",
        workers=2,
        client=client,
    )

    assert first["file_count"] == 6
    assert first["archive_count"] == 2
    assert first["uploaded_count"] == 6
    assert first["skipped_count"] == 0
    assert second["uploaded_count"] == 0
    assert second["skipped_count"] == 6
    assert all(
        stored["encryption"] == "AES256"
        for stored in client.objects.values()
    )

    manifest_path = root / "summaries" / "S3_UPLOAD_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bucket"] == "private-bucket"
    assert manifest["prefix"] == "matrix/example"
    assert manifest["file_count"] == 6
    assert ("private-bucket", "matrix/example/S3_UPLOAD_MANIFEST.json") in client.objects


def test_trace_upload_cli_dry_run_does_not_require_aws_authentication(tmp_path: Path):
    root = _trace_fixture(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "trace-upload",
            "--run-dir",
            str(root),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["file_count"] == 6
    assert plan["archive_count"] == 2


def test_trace_upload_cli_exposes_no_static_credential_options():
    result = CliRunner().invoke(cli, ["trace-upload", "--help"])

    assert result.exit_code == 0
    assert "access-key" not in result.output.lower()
    assert "secret-key" not in result.output.lower()
