from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .defaults import (
    DEFAULT_LAST_N_ROUNDS,
    DEFAULT_DEBUG_OUTPUT,
    DEFAULT_AGENT1_NAME,
    DEFAULT_AGENT1_SYSTEM_PROMPT,
    DEFAULT_AGENT2_NAME,
    DEFAULT_AGENT2_SYSTEM_PROMPT,
    DEFAULT_AGENT_LOOP_ENABLED,
    DEFAULT_AGENT_LOOP_MAX_ROUNDS,
    DEFAULT_RAG_MIN_CHILD_SCORE,
    DEFAULT_RAG_MIN_SCORE_MARGIN,
    DEFAULT_RAG_MODE,
    DEFAULT_RAG_PARENT_CONTEXT_CHARS,
    DEFAULT_RAG_TOP_K,
    DEFAULT_WORKING_MEMORY_MIN_RERANK_MARGIN,
    DEFAULT_WORKING_MEMORY_MIN_RERANK_SCORE,
    DEFAULT_WORKING_MEMORY_MODE,
    DEFAULT_WORKING_MEMORY_PARENT_CONTEXT_CHARS,
    DEFAULT_WORKING_MEMORY_EXCLUDE_LATEST_TURNS,
    DEFAULT_WORKING_MEMORY_OLDER_TOP_K_ROUNDS,
    DEFAULT_WORKING_MEMORY_RECENT_TOP_K_ROUNDS,
    DEFAULT_WORKING_MEMORY_RECENT_WINDOW_ROUNDS,
)


@dataclass(frozen=True)
class RagRuntimeOptions:
    mode: str = DEFAULT_RAG_MODE
    min_child_score: float = DEFAULT_RAG_MIN_CHILD_SCORE
    min_score_margin: float = DEFAULT_RAG_MIN_SCORE_MARGIN
    top_k: int = DEFAULT_RAG_TOP_K
    parent_context_chars: int = DEFAULT_RAG_PARENT_CONTEXT_CHARS

    def normalized(self) -> "RagRuntimeOptions":
        return RagRuntimeOptions(
            mode=str(self.mode),
            min_child_score=max(0.0, float(self.min_child_score)),
            min_score_margin=max(0.0, float(self.min_score_margin)),
            top_k=max(1, int(self.top_k)),
            parent_context_chars=max(120, int(self.parent_context_chars)),
        )


@dataclass(frozen=True)
class WorkingMemoryRuntimeOptions:
    mode: str = DEFAULT_WORKING_MEMORY_MODE
    min_rerank_score: float = DEFAULT_WORKING_MEMORY_MIN_RERANK_SCORE
    min_rerank_margin: float = DEFAULT_WORKING_MEMORY_MIN_RERANK_MARGIN
    recent_window_rounds: int = DEFAULT_WORKING_MEMORY_RECENT_WINDOW_ROUNDS
    recent_top_k_rounds: int = DEFAULT_WORKING_MEMORY_RECENT_TOP_K_ROUNDS
    older_top_k_rounds: int = DEFAULT_WORKING_MEMORY_OLDER_TOP_K_ROUNDS
    parent_context_chars: int = DEFAULT_WORKING_MEMORY_PARENT_CONTEXT_CHARS
    exclude_latest_turns: int = DEFAULT_WORKING_MEMORY_EXCLUDE_LATEST_TURNS

    def normalized(self) -> "WorkingMemoryRuntimeOptions":
        return WorkingMemoryRuntimeOptions(
            mode=str(self.mode),
            min_rerank_score=max(0.0, float(self.min_rerank_score)),
            min_rerank_margin=max(0.0, float(self.min_rerank_margin)),
            recent_window_rounds=max(1, int(self.recent_window_rounds)),
            recent_top_k_rounds=max(1, int(self.recent_top_k_rounds)),
            older_top_k_rounds=max(1, int(self.older_top_k_rounds)),
            parent_context_chars=max(120, int(self.parent_context_chars)),
            exclude_latest_turns=max(0, int(self.exclude_latest_turns)),
        )


@dataclass(frozen=True)
class AgentLoopRuntimeOptions:
    enabled: bool = DEFAULT_AGENT_LOOP_ENABLED
    max_rounds: int = DEFAULT_AGENT_LOOP_MAX_ROUNDS
    agent1_name: str = DEFAULT_AGENT1_NAME
    agent2_name: str = DEFAULT_AGENT2_NAME
    agent1_system_prompt: str = DEFAULT_AGENT1_SYSTEM_PROMPT
    agent2_system_prompt: str = DEFAULT_AGENT2_SYSTEM_PROMPT

    def normalized(self) -> "AgentLoopRuntimeOptions":
        agent1_name = str(self.agent1_name).strip() or DEFAULT_AGENT1_NAME
        agent2_name = str(self.agent2_name).strip() or DEFAULT_AGENT2_NAME
        agent1_system_prompt = (
            str(self.agent1_system_prompt).strip()
            or DEFAULT_AGENT1_SYSTEM_PROMPT
        )
        agent2_system_prompt = (
            str(self.agent2_system_prompt).strip()
            or DEFAULT_AGENT2_SYSTEM_PROMPT
        )
        return AgentLoopRuntimeOptions(
            enabled=bool(self.enabled),
            max_rounds=max(0, int(self.max_rounds)),
            agent1_name=agent1_name,
            agent2_name=agent2_name,
            agent1_system_prompt=agent1_system_prompt,
            agent2_system_prompt=agent2_system_prompt,
        )


@dataclass(frozen=True)
class ChatRuntimeOptions:
    last_n_rounds: int = DEFAULT_LAST_N_ROUNDS
    debug_output: bool = DEFAULT_DEBUG_OUTPUT
    rag: RagRuntimeOptions = field(default_factory=RagRuntimeOptions)
    working_memory: WorkingMemoryRuntimeOptions = field(
        default_factory=WorkingMemoryRuntimeOptions
    )
    agent_loop: AgentLoopRuntimeOptions = field(default_factory=AgentLoopRuntimeOptions)
    markdown_log_path: Path | None = None

    def normalized(self) -> "ChatRuntimeOptions":
        resolved_log_path = (
            self.markdown_log_path.resolve()
            if isinstance(self.markdown_log_path, Path)
            else None
        )
        return ChatRuntimeOptions(
            last_n_rounds=max(1, int(self.last_n_rounds)),
            debug_output=bool(self.debug_output),
            rag=self.rag.normalized(),
            working_memory=self.working_memory.normalized(),
            agent_loop=self.agent_loop.normalized(),
            markdown_log_path=resolved_log_path,
        )
