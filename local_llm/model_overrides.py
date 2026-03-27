from __future__ import annotations

from dataclasses import replace
from typing import Any

from .settings import ChatConfig, ModelConfig, ServerConfig


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def apply_model_chat_overrides(chat: ChatConfig, model: ModelConfig) -> ChatConfig:
    overrides = model.chat_overrides
    if not overrides:
        return chat

    updates: dict[str, object] = {}
    required_int_fields = ("max_tokens", "retry_total")
    optional_int_fields = ("top_k", "seed")
    required_float_fields = (
        "temperature",
        "top_p",
        "connect_timeout_sec",
        "read_timeout_sec",
        "retry_backoff_factor",
    )
    optional_float_fields = (
        "min_p",
        "typical_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    )
    bool_fields = ("stream", "disable_thinking", "show_reasoning")

    for key in required_int_fields:
        if key in overrides:
            value = overrides.get(key)
            if value is None:
                raise ValueError(f"models.{model.alias}.chat_overrides.{key} cannot be null")
            updates[key] = int(value)
    for key in optional_int_fields:
        if key in overrides:
            value = overrides.get(key)
            updates[key] = None if value is None else int(value)
    for key in required_float_fields:
        if key in overrides:
            value = overrides.get(key)
            if value is None:
                raise ValueError(f"models.{model.alias}.chat_overrides.{key} cannot be null")
            updates[key] = float(value)
    for key in optional_float_fields:
        if key in overrides:
            value = overrides.get(key)
            updates[key] = None if value is None else float(value)
    for key in bool_fields:
        if key in overrides:
            value = overrides.get(key)
            updates[key] = bool(value)

    if "url" in overrides and overrides.get("url") is not None:
        updates["url"] = str(overrides["url"])
    if "system_prompt" in overrides and overrides.get("system_prompt") is not None:
        updates["system_prompt"] = str(overrides["system_prompt"])

    if "chat_template_kwargs" in overrides:
        raw_kwargs = overrides.get("chat_template_kwargs")
        if raw_kwargs is None:
            updates["chat_template_kwargs"] = {}
        else:
            chat_template_kwargs = _require_mapping(
                raw_kwargs,
                f"models.{model.alias}.chat_overrides.chat_template_kwargs",
            )
            updates["chat_template_kwargs"] = {
                str(k): v for k, v in chat_template_kwargs.items()
            }

    if "request_overrides" in overrides:
        raw_request_overrides = overrides.get("request_overrides")
        if raw_request_overrides is None:
            updates["request_overrides"] = {}
        else:
            request_overrides = _require_mapping(
                raw_request_overrides,
                f"models.{model.alias}.chat_overrides.request_overrides",
            )
            updates["request_overrides"] = {
                str(k): v for k, v in request_overrides.items()
            }

    if "retry_status_forcelist" in overrides:
        raw_retry_codes = overrides.get("retry_status_forcelist")
        if raw_retry_codes is None:
            updates["retry_status_forcelist"] = ()
        else:
            retry_codes = _require_list(
                raw_retry_codes,
                f"models.{model.alias}.chat_overrides.retry_status_forcelist",
            )
            updates["retry_status_forcelist"] = tuple(int(code) for code in retry_codes)

    if "stop" in overrides:
        raw_stop = overrides.get("stop")
        if raw_stop is None:
            updates["stop"] = ()
        else:
            stop_items = _require_list(raw_stop, f"models.{model.alias}.chat_overrides.stop")
            updates["stop"] = tuple(str(item) for item in stop_items)

    if not updates:
        return chat
    return replace(chat, **updates)


def apply_model_server_overrides(server: ServerConfig, model: ModelConfig) -> ServerConfig:
    overrides = model.server_overrides
    if not overrides:
        return server

    updates: dict[str, object] = {}
    int_fields = (
        "ctx_size",
        "gpu_layers",
        "parallel",
        "threads",
        "threads_http",
        "batch_size",
        "ubatch_size",
    )
    bool_fields = ("flash_attn", "cache_prompt", "metrics")

    for key in int_fields:
        if key in overrides:
            value = overrides.get(key)
            if value is None:
                raise ValueError(f"models.{model.alias}.server_overrides.{key} cannot be null")
            updates[key] = int(value)
    for key in bool_fields:
        if key in overrides:
            value = overrides.get(key)
            updates[key] = bool(value)

    if "reasoning_budget" in overrides:
        reasoning_budget = overrides.get("reasoning_budget")
        updates["reasoning_budget"] = None if reasoning_budget is None else int(reasoning_budget)

    if "extra_args" in overrides:
        extra_args = overrides.get("extra_args")
        if extra_args is None:
            updates["extra_args"] = ()
        else:
            extra_args_list = _require_list(
                extra_args, f"models.{model.alias}.server_overrides.extra_args"
            )
            updates["extra_args"] = tuple(str(x) for x in extra_args_list)

    if not updates:
        return server
    return replace(server, **updates)
