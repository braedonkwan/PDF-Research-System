from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .defaults import (
    DEFAULT_LAST_N_ROUNDS,
    DEFAULT_DEBUG_OUTPUT,
    DEFAULT_AGENT1_NAME,
    DEFAULT_AGENT1_SYSTEM_PROMPT,
    DEFAULT_AGENT2_NAME,
    DEFAULT_AGENT2_SYSTEM_PROMPT,
    DEFAULT_AGENT_LOOP_ENABLED,
    DEFAULT_AGENT_LOOP_MAX_ROUNDS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RAG_CHILD_CHUNK_OVERLAP_WORDS,
    DEFAULT_RAG_CHILD_CHUNK_WORDS,
    DEFAULT_RAG_BATCH_SIZE,
    DEFAULT_RAG_MIN_CHILD_SCORE,
    DEFAULT_RAG_MIN_SCORE_MARGIN,
    DEFAULT_RAG_MODE,
    DEFAULT_RAG_PARENT_CHUNK_OVERLAP_WORDS,
    DEFAULT_RAG_PARENT_CHUNK_WORDS,
    DEFAULT_RAG_PARENT_CONTEXT_CHARS,
    DEFAULT_RAG_TOP_K,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_WORKING_MEMORY_BATCH_SIZE,
    DEFAULT_WORKING_MEMORY_CLEAR_OLD_SESSIONS_ON_START,
    DEFAULT_WORKING_MEMORY_CHILD_CHUNK_OVERLAP_WORDS,
    DEFAULT_WORKING_MEMORY_CHILD_CHUNK_WORDS,
    DEFAULT_WORKING_MEMORY_MIN_RERANK_MARGIN,
    DEFAULT_WORKING_MEMORY_MIN_RERANK_SCORE,
    DEFAULT_WORKING_MEMORY_MODE,
    DEFAULT_WORKING_MEMORY_OLDER_TOP_K_ROUNDS,
    DEFAULT_WORKING_MEMORY_PARENT_CHUNK_OVERLAP_WORDS,
    DEFAULT_WORKING_MEMORY_PARENT_CHUNK_WORDS,
    DEFAULT_WORKING_MEMORY_PARENT_CONTEXT_CHARS,
    DEFAULT_WORKING_MEMORY_EXCLUDE_LATEST_TURNS,
    DEFAULT_WORKING_MEMORY_RECENT_TOP_K_ROUNDS,
    DEFAULT_WORKING_MEMORY_RECENT_WINDOW_ROUNDS,
    DEFAULT_WORKING_MEMORY_ROOT,
)
from .persistent_storage import apply_persistent_cache_env_defaults

ENV_CONFIG_PATH = "LOCAL_LLM_CONFIG"
ENV_MODEL_ALIAS = "LOCAL_LLM_MODEL"
ENV_LLAMA_SERVER_PATH = "LOCAL_LLM_LLAMA_SERVER_PATH"
ENV_MODELS_ROOT = "LOCAL_LLM_MODELS_ROOT"
ENV_VOLUME_ROOT = "LOCAL_LLM_VOLUME_ROOT"
DEFAULT_CHAT_RETRY_STATUS_FORCELIST = (502, 503, 504)


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    api_name: str
    gguf_path: Path
    repo_id: str | None = None
    filename: str | None = None
    local_dir: Path | None = None
    supports_thinking_toggle: bool = False
    system_prompt: str | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    request_overrides: dict[str, Any] = field(default_factory=dict)
    chat_overrides: dict[str, Any] = field(default_factory=dict)
    server_overrides: dict[str, Any] = field(default_factory=dict)
    server_extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatConfig:
    url: str
    system_prompt: str
    max_tokens: int = 400
    temperature: float = 0.65
    top_p: float = 0.9
    top_k: int | None = None
    min_p: float | None = None
    typical_p: float | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    stream: bool = True
    disable_thinking: bool = False
    show_reasoning: bool = False
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    request_overrides: dict[str, Any] = field(default_factory=dict)
    connect_timeout_sec: float = 10.0
    read_timeout_sec: float = 180.0
    retry_total: int = 2
    retry_backoff_factor: float = 0.3
    retry_status_forcelist: tuple[int, ...] = (502, 503, 504)


@dataclass(frozen=True)
class ServerConfig:
    llama_server_path: Path
    host: str = "127.0.0.1"
    port: int = 8080
    ctx_size: int = 3072
    gpu_layers: int = 999
    parallel: int = 1
    threads: int = 6
    threads_http: int = 2
    batch_size: int = 512
    ubatch_size: int = 256
    flash_attn: bool = True
    cache_prompt: bool = True
    metrics: bool = True
    reasoning_budget: int | None = None
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class RagDefaultsConfig:
    mode: str = DEFAULT_RAG_MODE
    min_child_score: float = DEFAULT_RAG_MIN_CHILD_SCORE
    min_score_margin: float = DEFAULT_RAG_MIN_SCORE_MARGIN
    top_k: int = DEFAULT_RAG_TOP_K
    parent_context_chars: int = DEFAULT_RAG_PARENT_CONTEXT_CHARS
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str | None = DEFAULT_RERANKER_MODEL
    parent_chunk_words: int = DEFAULT_RAG_PARENT_CHUNK_WORDS
    parent_chunk_overlap_words: int = DEFAULT_RAG_PARENT_CHUNK_OVERLAP_WORDS
    child_chunk_words: int = DEFAULT_RAG_CHILD_CHUNK_WORDS
    child_chunk_overlap_words: int = DEFAULT_RAG_CHILD_CHUNK_OVERLAP_WORDS
    batch_size: int = DEFAULT_RAG_BATCH_SIZE


@dataclass(frozen=True)
class WorkingMemoryDefaultsConfig:
    mode: str = DEFAULT_WORKING_MEMORY_MODE
    root_dir: Path = Path(DEFAULT_WORKING_MEMORY_ROOT)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str | None = DEFAULT_RERANKER_MODEL
    min_rerank_score: float = DEFAULT_WORKING_MEMORY_MIN_RERANK_SCORE
    min_rerank_margin: float = DEFAULT_WORKING_MEMORY_MIN_RERANK_MARGIN
    recent_window_rounds: int = DEFAULT_WORKING_MEMORY_RECENT_WINDOW_ROUNDS
    recent_top_k_rounds: int = DEFAULT_WORKING_MEMORY_RECENT_TOP_K_ROUNDS
    older_top_k_rounds: int = DEFAULT_WORKING_MEMORY_OLDER_TOP_K_ROUNDS
    parent_context_chars: int = DEFAULT_WORKING_MEMORY_PARENT_CONTEXT_CHARS
    exclude_latest_turns: int = DEFAULT_WORKING_MEMORY_EXCLUDE_LATEST_TURNS
    parent_chunk_words: int = DEFAULT_WORKING_MEMORY_PARENT_CHUNK_WORDS
    parent_chunk_overlap_words: int = DEFAULT_WORKING_MEMORY_PARENT_CHUNK_OVERLAP_WORDS
    child_chunk_words: int = DEFAULT_WORKING_MEMORY_CHILD_CHUNK_WORDS
    child_chunk_overlap_words: int = DEFAULT_WORKING_MEMORY_CHILD_CHUNK_OVERLAP_WORDS
    batch_size: int = DEFAULT_WORKING_MEMORY_BATCH_SIZE
    clear_old_sessions_on_start: bool = DEFAULT_WORKING_MEMORY_CLEAR_OLD_SESSIONS_ON_START


@dataclass(frozen=True)
class AgentLoopDefaultsConfig:
    enabled: bool = DEFAULT_AGENT_LOOP_ENABLED
    max_rounds: int = DEFAULT_AGENT_LOOP_MAX_ROUNDS
    agent1_name: str = DEFAULT_AGENT1_NAME
    agent2_name: str = DEFAULT_AGENT2_NAME
    agent1_system_prompt: str = DEFAULT_AGENT1_SYSTEM_PROMPT
    agent2_system_prompt: str = DEFAULT_AGENT2_SYSTEM_PROMPT


@dataclass(frozen=True)
class RuntimeDefaultsConfig:
    last_n_rounds: int = DEFAULT_LAST_N_ROUNDS
    debug_output: bool = DEFAULT_DEBUG_OUTPUT
    rag: RagDefaultsConfig = field(default_factory=RagDefaultsConfig)
    working_memory: WorkingMemoryDefaultsConfig = field(
        default_factory=WorkingMemoryDefaultsConfig
    )
    agent_loop: AgentLoopDefaultsConfig = field(default_factory=AgentLoopDefaultsConfig)


@dataclass(frozen=True)
class AppConfig:
    default_model: str
    models: dict[str, ModelConfig]
    chat: ChatConfig
    server: ServerConfig
    source_path: Path = field(repr=False)
    runtime: RuntimeDefaultsConfig = field(default_factory=RuntimeDefaultsConfig)

    def get_model(self, alias: str | None = None) -> ModelConfig:
        env_alias = os.environ.get(ENV_MODEL_ALIAS) if alias is None else None
        selected_alias = alias or env_alias or self.default_model
        try:
            return self.models[selected_alias]
        except KeyError as exc:
            available = ", ".join(sorted(self.models))
            raise KeyError(
                f"Unknown model alias '{selected_alias}'. Available aliases: {available}"
            ) from exc

    def model_aliases(self) -> tuple[str, ...]:
        return tuple(sorted(self.models))

    def formatted_model_lines(self) -> list[str]:
        lines: list[str] = []
        for alias in self.model_aliases():
            model = self.models[alias]
            default_suffix = " (default)" if alias == self.default_model else ""
            lines.append(f"- {alias}{default_suffix}: {model.api_name}")
        return lines


MODEL_CHAT_OVERRIDE_KEYS = frozenset(
    {
        "url",
        "system_prompt",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "typical_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
        "stream",
        "disable_thinking",
        "show_reasoning",
        "chat_template_kwargs",
        "request_overrides",
        "connect_timeout_sec",
        "read_timeout_sec",
        "retry_total",
        "retry_backoff_factor",
        "retry_status_forcelist",
    }
)
MODEL_SERVER_OVERRIDE_KEYS = frozenset(
    {
        "ctx_size",
        "gpu_layers",
        "parallel",
        "threads",
        "threads_http",
        "batch_size",
        "ubatch_size",
        "flash_attn",
        "cache_prompt",
        "metrics",
        "reasoning_budget",
        "extra_args",
    }
)


def _validate_override_keys(
    mapping: dict[str, Any],
    valid_keys: frozenset[str],
    field_name: str,
) -> None:
    unknown_keys = sorted(str(key) for key in mapping.keys() if str(key) not in valid_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in '{field_name}': {', '.join(unknown_keys)}. "
            f"Allowed keys: {', '.join(sorted(valid_keys))}"
        )


def _resolve_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    path = Path(expanded)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _require_mapping(data: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"'{field_name}' must be an object in settings.json")
    return data


def _optional_mapping(data: Any, field_name: str) -> dict[str, Any]:
    if data is None:
        return {}
    return _require_mapping(data, field_name)


def _optional_list(data: Any, field_name: str) -> list[Any]:
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"'{field_name}' must be an array in settings.json")
    return data


def _require_field(data: dict[str, Any], key: str, field_name: str) -> Any:
    value = data.get(key)
    if value is None:
        raise ValueError(f"'{field_name}' is required in settings.json")
    return value


def _current_os_key() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    return "default"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _detect_volume_root() -> Path | None:
    env_root = os.environ.get(ENV_VOLUME_ROOT)
    if env_root:
        candidate = Path(os.path.expandvars(os.path.expanduser(env_root))).resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate
        return None

    default_workspace = Path("/workspace")
    if default_workspace.exists() and default_workspace.is_dir():
        return default_workspace
    return None


def _detect_models_root() -> Path | None:
    env_root = os.environ.get(ENV_MODELS_ROOT)
    if env_root:
        return Path(os.path.expandvars(os.path.expanduser(env_root))).resolve()
    volume_root = _detect_volume_root()
    if volume_root is None:
        return None
    return (volume_root / "models").resolve()


def _resolve_server_path(server_raw: dict[str, Any]) -> str:
    env_server_path = os.environ.get(ENV_LLAMA_SERVER_PATH)
    if env_server_path:
        return str(env_server_path)

    volume_root = _detect_volume_root()
    if volume_root is not None:
        volume_binary = (volume_root / "bin" / "llama-server").resolve()
        if volume_binary.exists():
            return str(volume_binary)

    raw_value = server_raw.get("llama_server_path")
    if raw_value is not None:
        return str(raw_value)

    paths_value = server_raw.get("llama_server_paths")
    paths = _require_mapping(paths_value, "server.llama_server_paths")
    os_key = _current_os_key()
    selected = paths.get(os_key)
    if selected is None:
        selected = paths.get("default")
    if selected is None:
        available = ", ".join(sorted(str(k) for k in paths.keys()))
        raise ValueError(
            "server.llama_server_path is missing and server.llama_server_paths "
            f"has no '{os_key}' or 'default' entry. Available keys: {available}"
        )
    return str(selected)


def _parse_model(alias: str, raw_model: Any, base_dir: Path) -> ModelConfig:
    item = _require_mapping(raw_model, f"models.{alias}")
    model_chat_kwargs = _optional_mapping(
        item.get("chat_template_kwargs"),
        f"models.{alias}.chat_template_kwargs",
    )
    model_request_overrides = _optional_mapping(
        item.get("request_overrides"),
        f"models.{alias}.request_overrides",
    )
    model_chat_overrides = _optional_mapping(
        item.get("chat_overrides"),
        f"models.{alias}.chat_overrides",
    )
    model_server_overrides = _optional_mapping(
        item.get("server_overrides"),
        f"models.{alias}.server_overrides",
    )
    _validate_override_keys(
        model_chat_overrides,
        MODEL_CHAT_OVERRIDE_KEYS,
        f"models.{alias}.chat_overrides",
    )
    _validate_override_keys(
        model_server_overrides,
        MODEL_SERVER_OVERRIDE_KEYS,
        f"models.{alias}.server_overrides",
    )
    model_server_extra_args = _optional_list(
        item.get("server_extra_args"),
        f"models.{alias}.server_extra_args",
    )

    raw_local_dir = item.get("local_dir")
    local_dir = _resolve_path(base_dir, str(raw_local_dir)) if raw_local_dir is not None else None
    gguf_path = Path(
        _resolve_path(
            base_dir,
            str(_require_field(item, "gguf_path", f"models.{alias}.gguf_path")),
        )
    )

    models_root = _detect_models_root()
    if models_root is not None:
        forced_models_root = os.environ.get(ENV_MODELS_ROOT) is not None
        volume_root = _detect_volume_root()
        should_rewrite = forced_models_root or (
            volume_root is not None and not _is_within(gguf_path, volume_root)
        )
        if should_rewrite:
            configured_local_dir_name = (
                Path(str(raw_local_dir)).name if raw_local_dir is not None else alias
            )
            target_local_dir = (models_root / configured_local_dir_name).resolve()
            filename = item.get("filename")
            target_filename = str(filename) if filename is not None else gguf_path.name
            local_dir = target_local_dir
            gguf_path = (target_local_dir / target_filename).resolve()

    return ModelConfig(
        alias=alias,
        api_name=str(_require_field(item, "api_name", f"models.{alias}.api_name")),
        gguf_path=gguf_path,
        repo_id=(str(item["repo_id"]) if item.get("repo_id") is not None else None),
        filename=(str(item["filename"]) if item.get("filename") is not None else None),
        local_dir=local_dir,
        supports_thinking_toggle=bool(item.get("supports_thinking_toggle", False)),
        system_prompt=(str(item["system_prompt"]) if item.get("system_prompt") is not None else None),
        chat_template_kwargs={str(k): v for k, v in model_chat_kwargs.items()},
        request_overrides={str(k): v for k, v in model_request_overrides.items()},
        chat_overrides={str(k): v for k, v in model_chat_overrides.items()},
        server_overrides={str(k): v for k, v in model_server_overrides.items()},
        server_extra_args=tuple(str(x) for x in model_server_extra_args),
    )


def _parse_models(models_raw: dict[str, Any], base_dir: Path) -> dict[str, ModelConfig]:
    models: dict[str, ModelConfig] = {}
    for alias, raw_model in models_raw.items():
        models[str(alias)] = _parse_model(str(alias), raw_model, base_dir)
    if not models:
        raise ValueError("'models' must contain at least one model in settings.json")
    return models


def _parse_chat(chat_raw: dict[str, Any]) -> ChatConfig:
    chat_template_kwargs = _optional_mapping(
        chat_raw.get("chat_template_kwargs"),
        "chat.chat_template_kwargs",
    )
    chat_request_overrides = _optional_mapping(
        chat_raw.get("request_overrides"),
        "chat.request_overrides",
    )
    retry_codes_raw = chat_raw.get("retry_status_forcelist")
    if retry_codes_raw is None:
        retry_codes = DEFAULT_CHAT_RETRY_STATUS_FORCELIST
    else:
        retry_codes = tuple(
            int(code)
            for code in _optional_list(retry_codes_raw, "chat.retry_status_forcelist")
        )
    stop_raw = chat_raw.get("stop")
    if stop_raw is None:
        stop = ()
    else:
        stop = tuple(str(item) for item in _optional_list(stop_raw, "chat.stop"))

    return ChatConfig(
        url=str(_require_field(chat_raw, "url", "chat.url")),
        system_prompt=str(_require_field(chat_raw, "system_prompt", "chat.system_prompt")),
        max_tokens=int(chat_raw.get("max_tokens", 400)),
        temperature=float(chat_raw.get("temperature", 0.65)),
        top_p=float(chat_raw.get("top_p", 0.9)),
        top_k=(int(chat_raw["top_k"]) if chat_raw.get("top_k") is not None else None),
        min_p=(float(chat_raw["min_p"]) if chat_raw.get("min_p") is not None else None),
        typical_p=(
            float(chat_raw["typical_p"])
            if chat_raw.get("typical_p") is not None
            else None
        ),
        repeat_penalty=(
            float(chat_raw["repeat_penalty"])
            if chat_raw.get("repeat_penalty") is not None
            else None
        ),
        presence_penalty=(
            float(chat_raw["presence_penalty"])
            if chat_raw.get("presence_penalty") is not None
            else None
        ),
        frequency_penalty=(
            float(chat_raw["frequency_penalty"])
            if chat_raw.get("frequency_penalty") is not None
            else None
        ),
        seed=(int(chat_raw["seed"]) if chat_raw.get("seed") is not None else None),
        stop=stop,
        stream=bool(chat_raw.get("stream", True)),
        disable_thinking=bool(chat_raw.get("disable_thinking", False)),
        show_reasoning=bool(chat_raw.get("show_reasoning", False)),
        chat_template_kwargs={str(k): v for k, v in chat_template_kwargs.items()},
        request_overrides={str(k): v for k, v in chat_request_overrides.items()},
        connect_timeout_sec=float(chat_raw.get("connect_timeout_sec", 10.0)),
        read_timeout_sec=float(chat_raw.get("read_timeout_sec", 180.0)),
        retry_total=int(chat_raw.get("retry_total", 2)),
        retry_backoff_factor=float(chat_raw.get("retry_backoff_factor", 0.3)),
        retry_status_forcelist=retry_codes,
    )


def _parse_server(server_raw: dict[str, Any], base_dir: Path) -> ServerConfig:
    extra_args = _optional_list(server_raw.get("extra_args"), "server.extra_args")
    reasoning_budget_raw = server_raw.get("reasoning_budget")
    reasoning_budget = int(reasoning_budget_raw) if reasoning_budget_raw is not None else None
    server_path = _resolve_server_path(server_raw)
    return ServerConfig(
        llama_server_path=Path(
            _resolve_path(
                base_dir,
                server_path,
            )
        ),
        host=str(server_raw.get("host", "127.0.0.1")),
        port=int(server_raw.get("port", 8080)),
        ctx_size=int(server_raw.get("ctx_size", 3072)),
        gpu_layers=int(server_raw.get("gpu_layers", 999)),
        parallel=int(server_raw.get("parallel", 1)),
        threads=int(server_raw.get("threads", 6)),
        threads_http=int(server_raw.get("threads_http", 2)),
        batch_size=int(server_raw.get("batch_size", 512)),
        ubatch_size=int(server_raw.get("ubatch_size", 256)),
        flash_attn=bool(server_raw.get("flash_attn", True)),
        cache_prompt=bool(server_raw.get("cache_prompt", True)),
        metrics=bool(server_raw.get("metrics", True)),
        reasoning_budget=reasoning_budget,
        extra_args=tuple(str(x) for x in extra_args),
    )


def _parse_optional_model_name(value: Any, default: str | None) -> str | None:
    if value is None:
        return default
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.lower() in {"off", "none", "null", "false", "disable", "disabled"}:
        return None
    return normalized


def _parse_runtime(runtime_raw: Any, base_dir: Path) -> RuntimeDefaultsConfig:
    runtime = _optional_mapping(runtime_raw, "runtime")
    rag_raw = _optional_mapping(runtime.get("rag"), "runtime.rag")
    working_memory_raw = _optional_mapping(
        runtime.get("working_memory"),
        "runtime.working_memory",
    )
    agent_loop_raw = _optional_mapping(runtime.get("agent_loop"), "runtime.agent_loop")

    legacy_rag_keys = [key for key in ("source_k", "parent_k", "child_k") if key in rag_raw]
    if legacy_rag_keys:
        joined = ", ".join(sorted(legacy_rag_keys))
        raise ValueError(
            "Deprecated runtime.rag key(s) are not supported: "
            f"{joined}. Use runtime.rag.top_k."
        )
    legacy_working_memory_keys = [
        key
        for key in (
            "parent_k",
            "child_k",
            "recent_turn_window",
            "recent_parent_k",
            "recent_child_k",
            "recent_parent_context_chars",
            "dedupe_all_time_against_recent",
        )
        if key in working_memory_raw
    ]
    if legacy_working_memory_keys:
        joined = ", ".join(sorted(legacy_working_memory_keys))
        raise ValueError(
            "Deprecated runtime.working_memory key(s) are not supported: "
            f"{joined}. Use recent_window_rounds/recent_top_k_rounds/older_top_k_rounds."
        )

    rag = RagDefaultsConfig(
        mode=str(rag_raw.get("mode", DEFAULT_RAG_MODE)),
        min_child_score=float(rag_raw.get("min_child_score", DEFAULT_RAG_MIN_CHILD_SCORE)),
        min_score_margin=float(rag_raw.get("min_score_margin", DEFAULT_RAG_MIN_SCORE_MARGIN)),
        top_k=max(1, int(rag_raw.get("top_k", DEFAULT_RAG_TOP_K))),
        parent_context_chars=max(
            120,
            int(rag_raw.get("parent_context_chars", DEFAULT_RAG_PARENT_CONTEXT_CHARS)),
        ),
        embedding_model=str(rag_raw.get("embedding_model", DEFAULT_EMBEDDING_MODEL)),
        reranker_model=_parse_optional_model_name(
            rag_raw.get("reranker_model"),
            DEFAULT_RERANKER_MODEL,
        ),
        parent_chunk_words=max(
            120,
            int(rag_raw.get("parent_chunk_words", DEFAULT_RAG_PARENT_CHUNK_WORDS)),
        ),
        parent_chunk_overlap_words=max(
            0,
            int(
                rag_raw.get(
                    "parent_chunk_overlap_words",
                    DEFAULT_RAG_PARENT_CHUNK_OVERLAP_WORDS,
                )
            ),
        ),
        child_chunk_words=max(
            60,
            int(rag_raw.get("child_chunk_words", DEFAULT_RAG_CHILD_CHUNK_WORDS)),
        ),
        child_chunk_overlap_words=max(
            0,
            int(
                rag_raw.get(
                    "child_chunk_overlap_words",
                    DEFAULT_RAG_CHILD_CHUNK_OVERLAP_WORDS,
                )
            ),
        ),
        batch_size=max(1, int(rag_raw.get("batch_size", DEFAULT_RAG_BATCH_SIZE))),
    )
    if rag.parent_chunk_overlap_words >= rag.parent_chunk_words:
        rag = replace(
            rag,
            parent_chunk_overlap_words=rag.parent_chunk_words - 1,
        )
    if rag.child_chunk_overlap_words >= rag.child_chunk_words:
        rag = replace(
            rag,
            child_chunk_overlap_words=rag.child_chunk_words - 1,
        )

    raw_memory_root = (
        working_memory_raw.get("root_dir")
        if working_memory_raw.get("root_dir") is not None
        else working_memory_raw.get("root")
    )
    if raw_memory_root is None:
        raw_memory_root = DEFAULT_WORKING_MEMORY_ROOT
    resolved_memory_root = Path(
        _resolve_path(base_dir, str(raw_memory_root)) or (base_dir / DEFAULT_WORKING_MEMORY_ROOT)
    )

    working_memory = WorkingMemoryDefaultsConfig(
        mode=str(working_memory_raw.get("mode", DEFAULT_WORKING_MEMORY_MODE)),
        root_dir=resolved_memory_root,
        embedding_model=str(
            working_memory_raw.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        ),
        reranker_model=_parse_optional_model_name(
            working_memory_raw.get("reranker_model"),
            DEFAULT_RERANKER_MODEL,
        ),
        min_rerank_score=float(
            working_memory_raw.get(
                "min_rerank_score",
                DEFAULT_WORKING_MEMORY_MIN_RERANK_SCORE,
            )
        ),
        min_rerank_margin=float(
            working_memory_raw.get(
                "min_rerank_margin",
                DEFAULT_WORKING_MEMORY_MIN_RERANK_MARGIN,
            )
        ),
        recent_window_rounds=max(
            1,
            int(
                working_memory_raw.get(
                    "recent_window_rounds",
                    DEFAULT_WORKING_MEMORY_RECENT_WINDOW_ROUNDS,
                )
            ),
        ),
        recent_top_k_rounds=max(
            1,
            int(
                working_memory_raw.get(
                    "recent_top_k_rounds",
                    DEFAULT_WORKING_MEMORY_RECENT_TOP_K_ROUNDS,
                )
            ),
        ),
        older_top_k_rounds=max(
            1,
            int(
                working_memory_raw.get(
                    "older_top_k_rounds",
                    DEFAULT_WORKING_MEMORY_OLDER_TOP_K_ROUNDS,
                )
            ),
        ),
        parent_context_chars=max(
            120,
            int(
                working_memory_raw.get(
                    "parent_context_chars",
                    DEFAULT_WORKING_MEMORY_PARENT_CONTEXT_CHARS,
                )
            ),
        ),
        exclude_latest_turns=max(
            0,
            int(
                working_memory_raw.get(
                    "exclude_latest_turns",
                    DEFAULT_WORKING_MEMORY_EXCLUDE_LATEST_TURNS,
                )
            ),
        ),
        parent_chunk_words=max(
            120,
            int(
                working_memory_raw.get(
                    "parent_chunk_words",
                    DEFAULT_WORKING_MEMORY_PARENT_CHUNK_WORDS,
                )
            ),
        ),
        parent_chunk_overlap_words=max(
            0,
            int(
                working_memory_raw.get(
                    "parent_chunk_overlap_words",
                    DEFAULT_WORKING_MEMORY_PARENT_CHUNK_OVERLAP_WORDS,
                )
            ),
        ),
        child_chunk_words=max(
            40,
            int(
                working_memory_raw.get(
                    "child_chunk_words",
                    DEFAULT_WORKING_MEMORY_CHILD_CHUNK_WORDS,
                )
            ),
        ),
        child_chunk_overlap_words=max(
            0,
            int(
                working_memory_raw.get(
                    "child_chunk_overlap_words",
                    DEFAULT_WORKING_MEMORY_CHILD_CHUNK_OVERLAP_WORDS,
                )
            ),
        ),
        batch_size=max(
            1,
            int(working_memory_raw.get("batch_size", DEFAULT_WORKING_MEMORY_BATCH_SIZE)),
        ),
        clear_old_sessions_on_start=bool(
            working_memory_raw.get(
                "clear_old_sessions_on_start",
                DEFAULT_WORKING_MEMORY_CLEAR_OLD_SESSIONS_ON_START,
            )
        ),
    )
    if working_memory.parent_chunk_overlap_words >= working_memory.parent_chunk_words:
        working_memory = replace(
            working_memory,
            parent_chunk_overlap_words=working_memory.parent_chunk_words - 1,
        )
    if working_memory.child_chunk_overlap_words >= working_memory.child_chunk_words:
        working_memory = replace(
            working_memory,
            child_chunk_overlap_words=working_memory.child_chunk_words - 1,
        )

    agent_loop = AgentLoopDefaultsConfig(
        enabled=bool(agent_loop_raw.get("enabled", DEFAULT_AGENT_LOOP_ENABLED)),
        max_rounds=max(0, int(agent_loop_raw.get("max_rounds", DEFAULT_AGENT_LOOP_MAX_ROUNDS))),
        agent1_name=str(agent_loop_raw.get("agent1_name", DEFAULT_AGENT1_NAME)).strip()
        or DEFAULT_AGENT1_NAME,
        agent2_name=str(agent_loop_raw.get("agent2_name", DEFAULT_AGENT2_NAME)).strip()
        or DEFAULT_AGENT2_NAME,
        agent1_system_prompt=str(
            agent_loop_raw.get("agent1_system_prompt", DEFAULT_AGENT1_SYSTEM_PROMPT)
        ).strip()
        or DEFAULT_AGENT1_SYSTEM_PROMPT,
        agent2_system_prompt=str(
            agent_loop_raw.get("agent2_system_prompt", DEFAULT_AGENT2_SYSTEM_PROMPT)
        ).strip()
        or DEFAULT_AGENT2_SYSTEM_PROMPT,
    )
    return RuntimeDefaultsConfig(
        last_n_rounds=max(1, int(runtime.get("last_n_rounds", DEFAULT_LAST_N_ROUNDS))),
        debug_output=bool(runtime.get("debug_output", DEFAULT_DEBUG_OUTPUT)),
        rag=rag,
        working_memory=working_memory,
        agent_loop=agent_loop,
    )


def _validate_default_model(default_model: str, models: dict[str, ModelConfig]) -> None:
    if not default_model:
        raise ValueError("'default_model' is required in settings.json")
    if default_model not in models:
        raise ValueError(
            f"default_model '{default_model}' not found in models. "
            f"Available aliases: {', '.join(sorted(models))}"
        )


def load_settings(config_path: str | Path | None = None) -> AppConfig:
    apply_persistent_cache_env_defaults()
    raw_config_path = config_path or os.environ.get(ENV_CONFIG_PATH) or "settings.json"
    config_file = Path(raw_config_path).resolve()
    with config_file.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    root = _require_mapping(raw, "root")
    models_raw = _require_mapping(root.get("models"), "models")
    chat_raw = _require_mapping(root.get("chat"), "chat")
    server_raw = _require_mapping(root.get("server"), "server")
    runtime_raw = root.get("runtime", {})
    base_dir = config_file.parent

    models = _parse_models(models_raw, base_dir)
    chat = _parse_chat(chat_raw)
    server = _parse_server(server_raw, base_dir)
    runtime = _parse_runtime(runtime_raw, base_dir)
    default_model = str(root.get("default_model", ""))
    _validate_default_model(default_model, models)

    return AppConfig(
        default_model=default_model,
        models=models,
        chat=chat,
        server=server,
        source_path=config_file,
        runtime=runtime,
    )
