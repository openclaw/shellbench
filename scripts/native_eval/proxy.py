from __future__ import annotations

import json
from pathlib import Path

from scripts.native_eval.models import LITELLM_VERSION, MODELS


def write_proxy_config(path: Path) -> None:
    model_list = []
    for model in MODELS:
        litellm_params = {
            "model": f"{model.provider}/{model.provider_model_id}",
            "api_key": f"os.environ/{model.provider.upper()}_API_KEY",
        }
        if model.slug == "gpt55":
            litellm_params["additional_drop_params"] = ["temperature"]
        model_list.append(
            {
                "model_name": model.proxy_model_name,
                "litellm_params": litellm_params,
                "model_info": {
                    "friendly_name": model.friendly_name,
                    "provider": model.provider,
                    "provider_model_id": model.provider_model_id,
                },
            }
        )
    config = {
        "model_list": model_list,
        "general_settings": {
            "master_key": "os.environ/SHELLBENCH_PROXY_KEY",
            "disable_spend_logs": False,
        },
        "litellm_settings": {
            "drop_params": True,
            "modify_params": True,
            "route_all_chat_openai_to_responses": True,
        },
        "router_settings": {
            "routing_strategy": "simple-shuffle",
            "num_retries": 2,
            "retry_after": 1,
        },
        "shellbench_native": {
            "litellm_version": LITELLM_VERSION,
            "models": [
                {
                    "slug": model.slug,
                    "provider": model.provider,
                    "provider_model_id": model.provider_model_id,
                    "proxy_model_name": model.proxy_model_name,
                }
                for model in MODELS
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
