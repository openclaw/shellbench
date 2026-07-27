import json
from pathlib import Path

from scripts.native_eval.proxy import write_proxy_config


def test_gpt55_proxy_drops_unsupported_temperature(tmp_path: Path):
    config_path = tmp_path / "proxy.json"

    write_proxy_config(config_path)

    config = json.loads(config_path.read_text())
    models = {item["model_name"]: item for item in config["model_list"]}
    assert models["sb-gpt55"]["litellm_params"]["additional_drop_params"] == [
        "temperature"
    ]
    assert "additional_drop_params" not in models["sb-gpt56-sol"]["litellm_params"]
