import json
from pathlib import Path

import pytest

from scripts.native_eval.audit import build_metadata_supplements, main


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_build_metadata_supplements_only_adds_missing_provenance(tmp_path: Path):
    run_index = tmp_path / "run_index.json"
    extracted = tmp_path / "extracted"
    _write_json(
        run_index,
        {
            "fleet": {
                "execution_mode": "native",
                "harbor_reference_commit": "harbor-sha",
                "judge_model_id": "gpt-5.5",
            }
        },
    )
    _write_json(
        extracted / "older" / "shellbench_meta-older" / "run_manifest.json",
        {"run_label": "older"},
    )
    _write_json(
        extracted / "newer" / "shellbench_meta-newer" / "run_manifest.json",
        {
            "run_label": "newer",
            "execution_mode": "native",
            "harbor_reference_commit": "harbor-sha",
            "judge_model_id": "gpt-5.5",
        },
    )

    report = build_metadata_supplements(run_index, extracted)

    assert report["raw_archives_mutated"] is False
    assert report["provenance"]["execution_mode"] == "native"
    assert report["runs"] == [
        {
            "run_label": "older",
            "archived_manifest": "older/shellbench_meta-older/run_manifest.json",
            "supplements": {
                "execution_mode": "native",
                "harbor_reference_commit": "harbor-sha",
                "judge_model_id": "gpt-5.5",
            },
        }
    ]

    output = tmp_path / "manifests" / "run_metadata_supplements.json"
    assert main([str(run_index), str(extracted), str(output)]) == 0
    assert json.loads(output.read_text())["runs"][0]["run_label"] == "older"


def test_build_metadata_supplements_allows_native_without_harbor_reference(
    tmp_path: Path,
):
    run_index = tmp_path / "run_index.json"
    extracted = tmp_path / "extracted"
    _write_json(
        run_index,
        {
            "fleet": {
                "execution_mode": "native",
                "harbor_reference_commit": "",
                "judge_model_id": "gpt-5.5",
            }
        },
    )
    _write_json(
        extracted / "run" / "shellbench_meta-run" / "run_manifest.json",
        {"run_label": "run"},
    )

    report = build_metadata_supplements(run_index, extracted)

    assert report["provenance"]["harbor_reference_commit"] == ""
    assert report["runs"] == [
        {
            "run_label": "run",
            "archived_manifest": "run/shellbench_meta-run/run_manifest.json",
            "supplements": {
                "execution_mode": "native",
                "judge_model_id": "gpt-5.5",
            },
        }
    ]


def test_build_metadata_supplements_requires_non_native_harbor_reference(
    tmp_path: Path,
):
    run_index = tmp_path / "run_index.json"
    extracted = tmp_path / "extracted"
    _write_json(
        run_index,
        {
            "fleet": {
                "execution_mode": "harbor",
                "harbor_reference_commit": "",
                "judge_model_id": "gpt-5.5",
            }
        },
    )

    with pytest.raises(ValueError, match="harbor_reference_commit"):
        build_metadata_supplements(run_index, extracted)


def test_build_metadata_supplements_rejects_conflicting_provenance(tmp_path: Path):
    run_index = tmp_path / "run_index.json"
    extracted = tmp_path / "extracted"
    _write_json(
        run_index,
        {
            "fleet": {
                "execution_mode": "native",
                "harbor_reference_commit": "harbor-sha",
                "judge_model_id": "gpt-5.5",
            }
        },
    )
    _write_json(
        extracted / "run" / "shellbench_meta-run" / "run_manifest.json",
        {
            "run_label": "run",
            "execution_mode": "harbor",
        },
    )

    with pytest.raises(ValueError, match="conflicting execution_mode"):
        build_metadata_supplements(run_index, extracted)


@pytest.mark.parametrize(
    ("observed", "status"),
    [
        (None, "missing"),
        (
            {
                "source_kind": "npm_tarball",
                "package_name": "openclaw",
                "package_version": "different",
                "sha256": "different",
                "artifact_filename": "openclaw-candidate.tgz",
            },
            "mismatch",
        ),
        ("expected", "match"),
    ],
)
def test_build_metadata_supplements_classifies_openclaw_candidate(
    tmp_path: Path,
    observed: dict | str | None,
    status: str,
) -> None:
    candidate = {
        "source_kind": "npm_tarball",
        "package_name": "openclaw",
        "package_version": "2026.7.29-candidate.1",
        "sha256": "candidate-sha",
        "artifact_filename": "openclaw-candidate.tgz",
    }
    run_index = tmp_path / "run_index.json"
    extracted = tmp_path / "extracted"
    _write_json(
        run_index,
        {
            "fleet": {
                "execution_mode": "native",
                "harbor_reference_commit": "harbor-sha",
                "judge_model_id": "gpt-5.5",
                "openclaw_package": candidate,
            }
        },
    )
    manifest = {
        "run_label": "candidate-run",
        "execution_mode": "native",
        "harbor_reference_commit": "harbor-sha",
        "judge_model_id": "gpt-5.5",
    }
    if observed is not None:
        manifest["openclaw_package"] = candidate if observed == "expected" else observed
    _write_json(
        extracted / "run" / "shellbench_meta-candidate-run" / "run_manifest.json",
        manifest,
    )

    report = build_metadata_supplements(run_index, extracted)

    assert report["runs"][0]["openclaw_candidate_status"] == status
    assert report["openclaw_candidate_counts"][status] == 1
    if status == "missing":
        assert "openclaw_package" not in report["runs"][0]["supplements"]
