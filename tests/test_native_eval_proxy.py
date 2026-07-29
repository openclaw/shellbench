import json
from pathlib import Path

from scripts.native_eval.proxy import JUDGE_PROXY_MODEL_NAME, write_proxy_config


def test_openai_proxy_models_enforce_reasoning_effort(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("SHELLBENCH_REASONING_EFFORT", "high")
    config_path = tmp_path / "proxy.json"

    write_proxy_config(config_path)

    config = json.loads(config_path.read_text())
    models = {item["model_name"]: item for item in config["model_list"]}
    assert models["gpt-5.5"]["litellm_params"]["additional_drop_params"] == [
        "temperature"
    ]
    assert models["gpt-5.6-sol"]["litellm_params"]["additional_drop_params"] == [
        "temperature"
    ]
    assert models["gpt-5.5"]["litellm_params"]["reasoning_effort"] == "high"
    assert models["gpt-5.6-sol"]["litellm_params"]["reasoning_effort"] == "high"
    assert models["gpt-5.6-luna"]["litellm_params"]["reasoning_effort"] == "high"
    assert models["gpt-5.6-terra"]["litellm_params"]["reasoning_effort"] == "high"
    assert config["shellbench_native"]["reasoning_effort"] == "high"


def test_openai_proxy_accepts_xhigh_reasoning_effort(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("SHELLBENCH_REASONING_EFFORT", "xhigh")
    config_path = tmp_path / "proxy.json"

    write_proxy_config(config_path)

    config = json.loads(config_path.read_text())
    models = {item["model_name"]: item for item in config["model_list"]}
    assert models["gpt-5.6-sol"]["litellm_params"]["reasoning_effort"] == "xhigh"
    assert config["shellbench_native"]["reasoning_effort"] == "xhigh"


def test_proxy_keeps_judge_reasoning_independent(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("SHELLBENCH_REASONING_EFFORT", "low")
    monkeypatch.setenv("SHELLBENCH_JUDGE_MODEL_ID", "gpt-5.5")
    monkeypatch.setenv("SHELLBENCH_JUDGE_REASONING_EFFORT", "high")
    config_path = tmp_path / "proxy.json"

    write_proxy_config(config_path)

    config = json.loads(config_path.read_text())
    models = {item["model_name"]: item for item in config["model_list"]}
    assert models["gpt-5.5"]["litellm_params"]["reasoning_effort"] == "low"
    assert (
        models[JUDGE_PROXY_MODEL_NAME]["litellm_params"]["reasoning_effort"]
        == "high"
    )
    assert config["shellbench_native"]["judge_model_id"] == "gpt-5.5"
    assert config["shellbench_native"]["judge_reasoning_effort"] == "high"
