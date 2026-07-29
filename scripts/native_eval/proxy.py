from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.native_eval.models import LITELLM_VERSION, MODELS


JUDGE_PROXY_MODEL_NAME = "shellbench-judge"
REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}


def write_proxy_config(path: Path) -> None:
    reasoning_effort = os.environ.get("SHELLBENCH_REASONING_EFFORT", "").strip()
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            "SHELLBENCH_REASONING_EFFORT must be low, medium, high, or xhigh"
        )
    judge_model_id = os.environ.get("SHELLBENCH_JUDGE_MODEL_ID", "gpt-5.5").strip()
    judge_reasoning_effort = os.environ.get(
        "SHELLBENCH_JUDGE_REASONING_EFFORT",
        reasoning_effort,
    ).strip()
    if judge_reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            "SHELLBENCH_JUDGE_REASONING_EFFORT must be low, medium, high, or xhigh"
        )
    judge_model = next(
        (model for model in MODELS if model.provider_model_id == judge_model_id),
        None,
    )
    if judge_model is None:
        raise ValueError(f"unknown SHELLBENCH_JUDGE_MODEL_ID: {judge_model_id}")

    model_list = []
    for model in MODELS:
        litellm_params = {
            "model": f"{model.provider}/{model.provider_model_id}",
            "api_key": f"os.environ/{model.provider.upper()}_API_KEY",
        }
        if model.provider == "openai":
            litellm_params["additional_drop_params"] = ["temperature"]
            litellm_params["reasoning_effort"] = reasoning_effort
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
    judge_litellm_params = {
        "model": f"{judge_model.provider}/{judge_model.provider_model_id}",
        "api_key": f"os.environ/{judge_model.provider.upper()}_API_KEY",
    }
    if judge_model.provider == "openai":
        judge_litellm_params["additional_drop_params"] = ["temperature"]
        judge_litellm_params["reasoning_effort"] = judge_reasoning_effort
    model_list.append(
        {
            "model_name": JUDGE_PROXY_MODEL_NAME,
            "litellm_params": judge_litellm_params,
            "model_info": {
                "friendly_name": judge_model.friendly_name,
                "provider": judge_model.provider,
                "provider_model_id": judge_model.provider_model_id,
                "role": "judge",
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
            "reasoning_effort": reasoning_effort,
            "judge_model_id": judge_model_id,
            "judge_proxy_model_name": JUDGE_PROXY_MODEL_NAME,
            "judge_reasoning_effort": judge_reasoning_effort,
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
