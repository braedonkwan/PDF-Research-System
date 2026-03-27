from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .chat_runtime import ChatRuntimeOptions
from .context_pipeline import collect_context_for_query

if TYPE_CHECKING:
    from .rag import HierarchicalRagStore
    from .working_memory import WorkingMemoryStore


@dataclass(frozen=True)
class RetrievedContext:
    context_text: str | None
    rag_summary: str | None
    memory_summary: str | None
    status_lines: list[str]
    retrieval_query_text: str


def collect_query_context(
    query: str,
    *,
    options: ChatRuntimeOptions,
    rag_store: "HierarchicalRagStore | None",
    working_memory_store: "WorkingMemoryStore | None",
    always_context_rounds: list[dict[str, object]] | None = None,
    recent_memory_exclude_last_turns: int = 0,
) -> RetrievedContext:
    context_text, rag_summary, memory_summary, status_lines = collect_context_for_query(
        query,
        rag_store=rag_store,
        working_memory_store=working_memory_store,
        options=options,
        always_context_rounds=always_context_rounds,
        recent_memory_exclude_last_turns=recent_memory_exclude_last_turns,
    )
    return RetrievedContext(
        context_text=context_text,
        rag_summary=rag_summary,
        memory_summary=memory_summary,
        status_lines=status_lines,
        retrieval_query_text=query,
    )


def collect_context_with_last_rounds(
    query: str,
    *,
    options: ChatRuntimeOptions,
    rag_store: "HierarchicalRagStore | None",
    working_memory_store: "WorkingMemoryStore | None",
    last_n_rounds_override: list[dict[str, object]] | None = None,
    retrieval_query_text: str | None = None,
    recent_memory_exclude_last_turns: int = 0,
) -> RetrievedContext:
    last_rounds_text = last_n_rounds_override
    retrieval_query = (retrieval_query_text or query).strip() or query
    result = collect_query_context(
        retrieval_query,
        options=options,
        rag_store=rag_store,
        working_memory_store=working_memory_store,
        always_context_rounds=last_rounds_text,
        recent_memory_exclude_last_turns=recent_memory_exclude_last_turns,
    )
    status_lines = list(result.status_lines)
    status_lines.insert(0, f"[Retrieval] query='{retrieval_query}'")
    return RetrievedContext(
        context_text=result.context_text,
        rag_summary=result.rag_summary,
        memory_summary=result.memory_summary,
        status_lines=status_lines,
        retrieval_query_text=retrieval_query,
    )
