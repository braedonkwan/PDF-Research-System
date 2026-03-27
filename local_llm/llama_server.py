from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

from .settings import ModelConfig, ServerConfig


@lru_cache(maxsize=8)
def server_help_text(server_path: str) -> str:
    try:
        result = subprocess.run(
            [server_path, "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return (result.stdout + "\n" + result.stderr).lower()
    except Exception:
        return ""


def _contains_short_flag(help_text: str, short_flag: str) -> bool:
    escaped = re.escape(short_flag)
    return re.search(rf"(^|\s){escaped}(\s|,|$)", help_text, flags=re.MULTILINE) is not None


def _contains_long_flag(help_text: str, long_flag: str) -> bool:
    escaped = re.escape(long_flag)
    return (
        re.search(rf"(^|[\s,]){escaped}([\s=,]|$)", help_text, flags=re.MULTILINE)
        is not None
    )


def _is_flag_supported(help_text: str, raw_flag: str) -> bool:
    flag = raw_flag.lower().split("=", 1)[0]
    if flag.startswith("--"):
        return _contains_long_flag(help_text, flag)
    if flag.startswith("-"):
        return _contains_short_flag(help_text, flag)
    return True


def _filter_optional_args(
    help_text: str, raw_args: tuple[str, ...], source: str
) -> tuple[list[str], list[str]]:
    if not raw_args:
        return [], []
    if not help_text:
        return list(raw_args), []

    filtered: list[str] = []
    warnings: list[str] = []
    i = 0
    while i < len(raw_args):
        token = str(raw_args[i])
        if not token.startswith("-"):
            filtered.append(token)
            i += 1
            continue

        if _is_flag_supported(help_text, token):
            filtered.append(token)
            if (
                "=" not in token
                and i + 1 < len(raw_args)
                and not str(raw_args[i + 1]).startswith("-")
            ):
                filtered.append(str(raw_args[i + 1]))
                i += 2
                continue
            i += 1
            continue

        warnings.append(
            f"Skipping {source} arg '{token}': unsupported by this llama-server build."
        )
        if (
            "=" not in token
            and i + 1 < len(raw_args)
            and not str(raw_args[i + 1]).startswith("-")
        ):
            i += 2
        else:
            i += 1

    return filtered, warnings


def _flash_attn_flag_style(help_text: str) -> str:
    if "--flash-attn" not in help_text and not _contains_short_flag(help_text, "-fa"):
        return "none"
    for line in help_text.splitlines():
        if "--flash-attn" in line:
            lowered = line.lower()
            if (
                "on|off" in lowered
                or "{on,off}" in lowered
                or "<on|off>" in lowered
                or "on/off" in lowered
            ):
                return "long_with_value"
            return "long"
    if _contains_short_flag(help_text, "-fa"):
        return "short"
    return "none"


def _resolve_reasoning_budget(help_text: str, configured_budget: int) -> tuple[int | None, str | None]:
    if "--reasoning-budget" not in help_text:
        return None, "Skipping --reasoning-budget: unsupported by this llama-server build."

    if configured_budget in (-1, 0):
        return configured_budget, None

    for line in help_text.splitlines():
        if "--reasoning-budget" not in line:
            continue
        lowered = line.lower()
        if (
            ("only one of" in lowered or "currently only one of" in lowered)
            and "-1" in lowered
            and "0" in lowered
        ):
            return (
                -1,
                f"Adjusting --reasoning-budget from {configured_budget} to -1: "
                "this llama-server build only accepts -1 or 0.",
            )
        break

    return configured_budget, None


def build_server_command(server: ServerConfig, model: ModelConfig) -> tuple[list[str], list[str]]:
    help_text = server_help_text(str(server.llama_server_path))
    warnings: list[str] = []
    if not help_text:
        warnings.append(
            "Could not read llama-server --help; skipping optional compatibility flags."
        )

    cmd = [
        str(server.llama_server_path),
        "-m",
        str(model.gguf_path),
        "--host",
        server.host,
        "--port",
        str(server.port),
        "-c",
        str(server.ctx_size),
        "-ngl",
        str(server.gpu_layers),
        "--parallel",
        str(server.parallel),
        "--threads",
        str(server.threads),
        "--threads-http",
        str(server.threads_http),
        "--batch-size",
        str(server.batch_size),
        "--ubatch-size",
        str(server.ubatch_size),
    ]

    if server.flash_attn:
        flash_style = _flash_attn_flag_style(help_text)
        if flash_style == "long_with_value":
            cmd.extend(["--flash-attn", "on"])
        elif flash_style == "long":
            cmd.append("--flash-attn")
        elif flash_style == "short":
            cmd.append("-fa")
        elif help_text:
            warnings.append("Skipping flash attention flag: unsupported by this llama-server build.")

    if server.cache_prompt and "--cache-prompt" in help_text:
        cmd.append("--cache-prompt")
    elif server.cache_prompt and help_text:
        warnings.append("Skipping --cache-prompt: unsupported by this llama-server build.")

    if server.metrics and "--metrics" in help_text:
        cmd.append("--metrics")
    elif server.metrics and help_text:
        warnings.append("Skipping --metrics: unsupported by this llama-server build.")

    if server.reasoning_budget is not None and help_text:
        resolved_budget, warning = _resolve_reasoning_budget(help_text, int(server.reasoning_budget))
        if resolved_budget is not None:
            cmd.extend(["--reasoning-budget", str(resolved_budget)])
        if warning:
            warnings.append(warning)

    model_extra_args, model_extra_warnings = _filter_optional_args(
        help_text,
        model.server_extra_args,
        f"models.{model.alias}.server_extra_args",
    )
    server_extra_args, server_extra_warnings = _filter_optional_args(
        help_text,
        server.extra_args,
        "server.extra_args",
    )
    warnings.extend(model_extra_warnings)
    warnings.extend(server_extra_warnings)

    cmd.extend(model_extra_args)
    cmd.extend(server_extra_args)
    return cmd, warnings


def validate_server_paths(server: ServerConfig, model: ModelConfig) -> None:
    if not Path(server.llama_server_path).exists():
        raise FileNotFoundError(f"llama-server binary not found: {server.llama_server_path}")
    if not model.gguf_path.exists():
        raise FileNotFoundError(f"GGUF model file not found: {model.gguf_path}")
