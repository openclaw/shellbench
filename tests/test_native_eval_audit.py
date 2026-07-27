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
