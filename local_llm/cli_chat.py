from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .chat_runtime import (
    AgentLoopRuntimeOptions,
    ChatRuntimeOptions,
    RagRuntimeOptions,
    WorkingMemoryRuntimeOptions,
)
from .client import run_interactive_chat
from .cli_common import add_common_arguments, handle_list_models_flag, load_cli_settings
from .defaults import (
    DEFAULT_LAST_N_ROUNDS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_WORKING_MEMORY_ROOT,
)
from .settings import AppConfig, ModelConfig

LEGACY_CLI_FLAGS: dict[str, str] = {
    "--rag-source-k": "--rag-top-k",
    "--rag-parent-k": "--rag-top-k",
    "--rag-child-k": "--rag-top-k",
    "--working-memory-parent-k": "--working-memory-recent-top-k-rounds / --working-memory-older-top-k-rounds",
    "--working-memory-child-k": "--working-memory-recent-top-k-rounds / --working-memory-older-top-k-rounds",
    "--working-memory-recent-turn-window": "--working-memory-recent-window-rounds",
    "--working-memory-recent-parent-k": "--working-memory-recent-top-k-rounds",
    "--working-memory-recent-child-k": "--working-memory-recent-top-k-rounds",
    "--working-memory-recent-parent-context-chars": "--working-memory-parent-context-chars",
    "--working-memory-dedupe-all-time-against-recent": "none (removed in hard-cutover)",
    "--working-memory-no-dedupe-all-time-against-recent": "none (removed in hard-cutover)",
}


def _coalesce(value: object, fallback: object) -> object:
    return fallback if value is None else value


def _fail_on_legacy_cli_flags(parser: argparse.ArgumentParser) -> None:
    for raw_arg in sys.argv[1:]:
        for legacy_flag, replacement in LEGACY_CLI_FLAGS.items():
            if raw_arg == legacy_flag or raw_arg.startswith(f"{legacy_flag}="):
                parser.error(
                    f"Flag '{legacy_flag}' is removed in hard-cutover mode. "
                    f"Use {replacement} instead."
                )


def _runtime_defaults_from_config(config: AppConfig) -> ChatRuntimeOptions:
    runtime = config.runtime
    return ChatRuntimeOptions(
        last_n_rounds=runtime.last_n_rounds,
        rag=RagRuntimeOptions(
            mode=runtime.rag.mode,
            min_child_score=runtime.rag.min_child_score,
            min_score_margin=runtime.rag.min_score_margin,
            top_k=runtime.rag.top_k,
            parent_context_chars=runtime.rag.parent_context_chars,
        ),
        working_memory=WorkingMemoryRuntimeOptions(
            mode=runtime.working_memory.mode,
            min_rerank_score=runtime.working_memory.min_rerank_score,
            min_rerank_margin=runtime.working_memory.min_rerank_margin,
            recent_window_rounds=runtime.working_memory.recent_window_rounds,
            recent_top_k_rounds=runtime.working_memory.recent_top_k_rounds,
            older_top_k_rounds=runtime.working_memory.older_top_k_rounds,
            parent_context_chars=runtime.working_memory.parent_context_chars,
            exclude_latest_turns=runtime.working_memory.exclude_latest_turns,
        ),
        agent_loop=AgentLoopRuntimeOptions(
            enabled=runtime.agent_loop.enabled,
            max_rounds=runtime.agent_loop.max_rounds,
            agent1_name=runtime.agent_loop.agent1_name,
            agent2_name=runtime.agent_loop.agent2_name,
            agent1_system_prompt=runtime.agent_loop.agent1_system_prompt,
            agent2_system_prompt=runtime.agent_loop.agent2_system_prompt,
        ),
    )


def _resolve_runtime_options(
    args: argparse.Namespace,
    *,
    defaults: ChatRuntimeOptions,
) -> ChatRuntimeOptions:
    return ChatRuntimeOptions(
        last_n_rounds=int(_coalesce(args.last_n_rounds, defaults.last_n_rounds)),
        rag=RagRuntimeOptions(
            mode=str(_coalesce(args.rag_mode, defaults.rag.mode)),
            min_child_score=float(
                _coalesce(args.rag_min_child_score, defaults.rag.min_child_score)
            ),
            min_score_margin=float(
                _coalesce(args.rag_min_score_margin, defaults.rag.min_score_margin)
            ),
            top_k=int(_coalesce(args.rag_top_k, defaults.rag.top_k)),
            parent_context_chars=int(
                _coalesce(
                    args.rag_parent_context_chars,
                    defaults.rag.parent_context_chars,
                )
            ),
        ),
        working_memory=WorkingMemoryRuntimeOptions(
            mode=str(_coalesce(args.working_memory_mode, defaults.working_memory.mode)),
            min_rerank_score=float(
                _coalesce(
                    args.working_memory_min_rerank_score,
                    defaults.working_memory.min_rerank_score,
                )
            ),
            min_rerank_margin=float(
                _coalesce(
                    args.working_memory_min_rerank_margin,
                    defaults.working_memory.min_rerank_margin,
                )
            ),
            recent_window_rounds=int(
                _coalesce(
                    args.working_memory_recent_window_rounds,
                    defaults.working_memory.recent_window_rounds,
                )
            ),
            recent_top_k_rounds=int(
                _coalesce(
                    args.working_memory_recent_top_k_rounds,
                    defaults.working_memory.recent_top_k_rounds,
                )
            ),
            older_top_k_rounds=int(
                _coalesce(
                    args.working_memory_older_top_k_rounds,
                    defaults.working_memory.older_top_k_rounds,
                )
            ),
            parent_context_chars=int(
                _coalesce(
                    args.working_memory_parent_context_chars,
                    defaults.working_memory.parent_context_chars,
                )
            ),
            exclude_latest_turns=int(
                _coalesce(
                    args.working_memory_exclude_latest_turns,
                    defaults.working_memory.exclude_latest_turns,
                )
            ),
        ),
        agent_loop=AgentLoopRuntimeOptions(
            enabled=bool(_coalesce(args.agent_loop, defaults.agent_loop.enabled)),
            max_rounds=int(_coalesce(args.agent_loop_rounds, defaults.agent_loop.max_rounds)),
            agent1_name=str(_coalesce(args.agent1_name, defaults.agent_loop.agent1_name)),
            agent2_name=str(_coalesce(args.agent2_name, defaults.agent_loop.agent2_name)),
            agent1_system_prompt=str(
                _coalesce(
                    args.agent1_system_prompt,
                    defaults.agent_loop.agent1_system_prompt,
                )
            ),
            agent2_system_prompt=str(
                _coalesce(
                    args.agent2_system_prompt,
                    defaults.agent_loop.agent2_system_prompt,
                )
            ),
        ),
        markdown_log_path=(Path(args.md_log) if args.md_log is not None else None),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive chat client")
    add_common_arguments(parser)
    parser.add_argument(
        "--last-n-rounds",
        type=int,
        default=None,
        help=(
            "How many completed rounds to keep as explicit short-term context "
            f"(default: settings.runtime.last_n_rounds or {DEFAULT_LAST_N_ROUNDS})"
        ),
    )
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream",
        action="store_true",
        help="Force streaming responses",
    )
    stream_group.add_argument(
        "--no-stream",
        action="store_true",
        help="Force non-stream responses",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Request non-thinking mode when the selected model supports it",
    )
    thinking_group.add_argument(
        "--allow-thinking",
        action="store_true",
        help="Do not force non-thinking mode",
    )
    parser.add_argument(
        "--rag-store",
        default=None,
        help="Path to a hierarchical RAG store created by rag.py ingest",
    )
    parser.add_argument(
        "--rag-mode",
        choices=("auto", "force", "off"),
        default=None,
        help="RAG gating mode (defaults to settings.runtime.rag.mode)",
    )
    parser.add_argument(
        "--rag-min-child-score",
        type=float,
        default=None,
        help="Minimum top child similarity score for --rag-mode auto",
    )
    parser.add_argument(
        "--rag-min-score-margin",
        type=float,
        default=None,
        help="Minimum top-vs-second child score margin for --rag-mode auto",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=None,
        help="Top parent sections to return for PDF RAG",
    )
    parser.add_argument(
        "--rag-parent-context-chars",
        type=int,
        default=None,
        help="Parent text characters included as context per parent section",
    )
    parser.add_argument(
        "--rag-reranker-model",
        default=None,
        help=(
            "Cross-encoder reranker model for PDF RAG "
            "(defaults to settings.runtime.rag.reranker_model; "
            f"fallback: {DEFAULT_RERANKER_MODEL}; use 'off' to disable)"
        ),
    )
    parser.add_argument(
        "--md-log",
        default=None,
        help="Append chat turns to a Markdown transcript file",
    )
    parser.add_argument(
        "--working-memory-mode",
        choices=("auto", "force", "off"),
        default=None,
        help="Working-memory gating mode (defaults to settings.runtime.working_memory.mode)",
    )
    parser.add_argument(
        "--working-memory-root",
        default=None,
        help=(
            "Root directory for per-agent working-memory stores "
            f"(default: settings.runtime.working_memory.root or {DEFAULT_WORKING_MEMORY_ROOT})"
        ),
    )
    parser.add_argument(
        "--working-memory-agent-id",
        default=None,
        help="Agent identifier used for memory partitioning (default: selected model alias)",
    )
    parser.add_argument(
        "--working-memory-session-id",
        default=None,
        help="Session id for memory persistence (default: current UTC timestamp)",
    )
    parser.add_argument(
        "--working-memory-embedding-model",
        default=None,
        help=(
            "Sentence-transformers embedding model for memory "
            "(defaults to settings.runtime.working_memory.embedding_model; "
            f"fallback: {DEFAULT_EMBEDDING_MODEL})"
        ),
    )
    parser.add_argument(
        "--working-memory-reranker-model",
        default=None,
        help=(
            "Cross-encoder reranker model for working memory "
            "(defaults to settings.runtime.working_memory.reranker_model; "
            f"fallback: {DEFAULT_RERANKER_MODEL}; use 'off' to disable)"
        ),
    )
    parser.add_argument(
        "--working-memory-min-rerank-score",
        type=float,
        default=None,
        help="Minimum top memory rerank score required in --working-memory-mode auto",
    )
    parser.add_argument(
        "--working-memory-min-rerank-margin",
        type=float,
        default=None,
        help="Minimum top-vs-second memory rerank margin in --working-memory-mode auto",
    )
    parser.add_argument(
        "--working-memory-recent-window-rounds",
        type=int,
        default=None,
        help=(
            "Recent round window used for short-memory retrieval "
            "(defaults to settings.runtime.working_memory.recent_window_rounds)"
        ),
    )
    parser.add_argument(
        "--working-memory-recent-top-k-rounds",
        type=int,
        default=None,
        help="Top recent memory rounds to return",
    )
    parser.add_argument(
        "--working-memory-older-top-k-rounds",
        type=int,
        default=None,
        help="Top older long-term memory rounds to return",
    )
    parser.add_argument(
        "--working-memory-parent-context-chars",
        type=int,
        default=None,
        help="Parent memory context characters included per returned round",
    )
    parser.add_argument(
        "--working-memory-exclude-latest-turns",
        type=int,
        default=None,
        help=(
            "Exclude this many newest turns from both recent and older memory retrieval "
            "(defaults to settings.runtime.working_memory.exclude_latest_turns)"
        ),
    )
    loop_group = parser.add_mutually_exclusive_group()
    loop_group.add_argument(
        "--agent-loop",
        dest="agent_loop",
        action="store_true",
        default=None,
        help="Enable two-agent critique loop mode",
    )
    loop_group.add_argument(
        "--no-agent-loop",
        dest="agent_loop",
        action="store_false",
        help="Disable two-agent critique loop mode",
    )
    parser.add_argument(
        "--agent-loop-rounds",
        type=int,
        default=None,
        help="Maximum critique rounds per user question in agent-loop mode (0 = unlimited)",
    )
    parser.add_argument(
        "--agent1-name",
        default=None,
        help="Display name for Agent 1 in agent-loop mode",
    )
    parser.add_argument(
        "--agent2-name",
        default=None,
        help="Display name for Agent 2 in agent-loop mode",
    )
    parser.add_argument(
        "--agent1-system-prompt",
        default=None,
        help="System prompt override for Agent 1 in agent-loop mode",
    )
    parser.add_argument(
        "--agent2-system-prompt",
        default=None,
        help="System prompt override for Agent 2 in agent-loop mode",
    )
    return parser


def _resolve_rag_store(args: argparse.Namespace, *, config: AppConfig):
    if not args.rag_store:
        return None
    from .rag import HierarchicalRagStore

    reranker_model_name = _coalesce(
        args.rag_reranker_model,
        config.runtime.rag.reranker_model,
    )
    if reranker_model_name is None:
        reranker_model_name = "off"
    return HierarchicalRagStore.load(
        args.rag_store,
        reranker_model_name=str(reranker_model_name),
    )


def _resolve_working_memory_store(
    args: argparse.Namespace,
    *,
    config: AppConfig,
    model: ModelConfig,
    runtime_options: ChatRuntimeOptions,
):
    if runtime_options.working_memory.mode == "off":
        return None

    from .working_memory import WorkingMemoryStore

    root_dir = Path(
        _coalesce(
            args.working_memory_root,
            str(config.runtime.working_memory.root_dir),
        )
    )
    embedding_model = str(
        _coalesce(
            args.working_memory_embedding_model,
            config.runtime.working_memory.embedding_model,
        )
    )
    reranker_model_value = _coalesce(
        args.working_memory_reranker_model,
        config.runtime.working_memory.reranker_model,
    )
    if reranker_model_value is None:
        reranker_model_value = "off"
    reranker_model = str(reranker_model_value)

    return WorkingMemoryStore.create(
        root_dir,
        agent_id=str(args.working_memory_agent_id or model.alias),
        session_id=(
            str(args.working_memory_session_id)
            if args.working_memory_session_id
            else None
        ),
        embedding_model_name=embedding_model,
        reranker_model_name=reranker_model,
        parent_chunk_words=config.runtime.working_memory.parent_chunk_words,
        parent_chunk_overlap_words=config.runtime.working_memory.parent_chunk_overlap_words,
        child_chunk_words=config.runtime.working_memory.child_chunk_words,
        child_chunk_overlap_words=config.runtime.working_memory.child_chunk_overlap_words,
        batch_size=config.runtime.working_memory.batch_size,
        clear_old_sessions_on_start=(
            config.runtime.working_memory.clear_old_sessions_on_start
        ),
    )


def main() -> int:
    parser = _build_parser()
    _fail_on_legacy_cli_flags(parser)
    args = parser.parse_args()
    working_memory_store = None
    try:
        config = load_cli_settings(args.config)
        if handle_list_models_flag(config, args.list_models):
            return 0

        if args.stream:
            config = replace(config, chat=replace(config.chat, stream=True))
        elif args.no_stream:
            config = replace(config, chat=replace(config.chat, stream=False))

        if args.disable_thinking:
            config = replace(config, chat=replace(config.chat, disable_thinking=True))
        elif args.allow_thinking:
            config = replace(config, chat=replace(config.chat, disable_thinking=False))

        model = config.get_model(args.model)
        runtime_defaults = _runtime_defaults_from_config(config)

        runtime_options = _resolve_runtime_options(args, defaults=runtime_defaults)
        rag_store = _resolve_rag_store(args, config=config)
        working_memory_store = _resolve_working_memory_store(
            args,
            config=config,
            model=model,
            runtime_options=runtime_options,
        )

        return run_interactive_chat(
            config=config,
            model=model,
            rag_store=rag_store,
            working_memory_store=working_memory_store,
            runtime_options=runtime_options,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    finally:
        if working_memory_store is not None:
            try:
                working_memory_store.drop_session()
            except Exception:
                pass
