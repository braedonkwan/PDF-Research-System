from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from .chat_runtime import ChatRuntimeOptions

if TYPE_CHECKING:
    from .rag import HierarchicalRagStore, RetrievalResult
    from .working_memory import MemoryRetrievalResult, WorkingMemoryStore

DOC_INTENT_HINTS = (
    "document",
    "pdf",
    "manual",
    "policy",
    "contract",
    "section",
    "chapter",
    "page",
    "according to",
    "based on the file",
    "from the file",
)
GENERIC_NON_DOC_QUERIES = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "how are you",
    "what's up",
}

def _is_document_intent_query(query: str) -> bool:
    normalized = query.lower()
    return any(hint in normalized for hint in DOC_INTENT_HINTS)


def _should_apply_rag_context(
    query: str,
    retrieval: "RetrievalResult",
    *,
    rag_mode: str,
    min_child_score: float,
    min_score_margin: float,
) -> tuple[bool, str]:
    if rag_mode == "off":
        return False, "rag mode off"
    if rag_mode == "force":
        if retrieval.parent_hits:
            return True, "forced by rag mode"
        return False, "forced mode but no retrieved parent sections"

    normalized = query.strip().lower().strip("?!.,")
    if normalized in GENERIC_NON_DOC_QUERIES:
        return False, "generic non-document prompt"

    document_intent = _is_document_intent_query(query)
    if not retrieval.parent_hits:
        return False, "no parent hits"

    top_child_score = float(retrieval.parent_hits[0].score)
    effective_min_score = max(0.0, min_child_score - (0.05 if document_intent else 0.0))
    if top_child_score < effective_min_score:
        return (
            False,
            f"top parent score {top_child_score:.3f} below {effective_min_score:.3f}",
        )

    if len(retrieval.parent_hits) >= 2 and not document_intent:
        second_score = float(retrieval.parent_hits[1].score)
        score_margin = top_child_score - second_score
        if score_margin < min_score_margin:
            return False, f"ambiguous retrieval margin {score_margin:.3f}"

    if len(query.split()) <= 2 and not document_intent:
        return False, "short non-document query"

    return True, f"top parent score {top_child_score:.3f}"


def _should_apply_working_memory_context(
    query: str,
    retrieval: "MemoryRetrievalResult",
    *,
    memory_mode: str,
    min_rerank_score: float,
    min_rerank_margin: float,
) -> tuple[bool, str]:
    if memory_mode == "off":
        return False, "memory mode off"
    if memory_mode == "force":
        if retrieval.parent_hits:
            return True, "forced by memory mode"
        return False, "forced mode but no memory hits"

    if not retrieval.parent_hits:
        return False, "no memory hits"

    normalized = query.strip().lower().strip("?!.,")
    if normalized in GENERIC_NON_DOC_QUERIES:
        return False, "generic non-memory prompt"

    top_score = float(retrieval.parent_hits[0].score)
    if top_score < min_rerank_score:
        return False, f"top memory score {top_score:.3f} below {min_rerank_score:.3f}"

    if len(retrieval.parent_hits) >= 2:
        second_score = float(retrieval.parent_hits[1].score)
        score_margin = top_score - second_score
        if score_margin < min_rerank_margin:
            return False, f"ambiguous memory margin {score_margin:.3f}"

    if len(query.split()) <= 2:
        return False, "short query"

    return True, f"top memory score {top_score:.3f}"


def _build_tagged_context_envelope(
    *,
    last_rounds: list[dict[str, object]] | None,
    memory_payload: dict[str, object] | None,
    rag_payload: dict[str, object] | None,
) -> str:
    payload = {
        "working_memory": {
            "last_n_rounds": last_rounds or [],
            "long_term_memory": memory_payload,
        },
        "knowledge": rag_payload,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _split_memory_round_text(raw_text: str) -> tuple[str, str]:
    user_lines: list[str] = []
    assistant_lines: list[str] = []
    active_role: str | None = None
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("user:"):
            active_role = "user"
            value = stripped[5:].strip()
            if value:
                user_lines.append(value)
            continue
        if lowered.startswith("assistant:"):
            active_role = "assistant"
            value = stripped[10:].strip()
            if value:
                assistant_lines.append(value)
            continue
        if active_role == "assistant":
            assistant_lines.append(stripped)
        else:
            user_lines.append(stripped)

    user_text = " ".join(user_lines).strip() or "(empty)"
    assistant_text = " ".join(assistant_lines).strip() or "(empty)"
    return user_text, assistant_text


def _serialize_memory_rounds(
    hits: list[object],
    *,
    max_chars: int,
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    limit = max(120, int(max_chars))
    ordered_hits = sorted(
        hits,
        key=lambda hit: (
            int(getattr(hit, "turn_start", 0)),
            int(getattr(hit, "turn_end", 0)),
        ),
    )
    for hit in ordered_hits:
        raw_text = str(getattr(hit, "text", "")).strip()
        user_text, assistant_text = _split_memory_round_text(raw_text)
        payload.append(
            {
                "round_id": getattr(hit, "parent_id", ""),
                "score": float(getattr(hit, "score", 0.0)),
                "turn_start": int(getattr(hit, "turn_start", 0)),
                "turn_end": int(getattr(hit, "turn_end", 0)),
                "user_query": {
                    "speaker": "User",
                    "text": user_text[:limit],
                },
                "response": {
                    "speaker": "Assistant",
                    "text": assistant_text[:limit],
                },
            }
        )
    return payload


def _serialize_rag_sources(hits: list[object]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for hit in hits:
        payload.append(
            {
                "source_id": str(getattr(hit, "source_id", "")),
                "source_path": str(getattr(hit, "source_path", "")),
                "score": float(getattr(hit, "score", 0.0)),
                "parent_count": int(getattr(hit, "parent_count", 0)),
            }
        )
    return payload


def _serialize_rag_parents(
    hits: list[object],
    *,
    max_chars: int,
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    limit = max(120, int(max_chars))
    for hit in hits:
        raw_text = str(getattr(hit, "text", "")).strip()
        payload.append(
            {
                "parent_id": str(getattr(hit, "parent_id", "")),
                "heading": str(getattr(hit, "heading", "")),
                "page_start": int(getattr(hit, "page_start", 0)),
                "page_end": int(getattr(hit, "page_end", 0)),
                "score": float(getattr(hit, "score", 0.0)),
                "source_id": str(getattr(hit, "source_id", "")),
                "source_path": str(getattr(hit, "source_path", "")),
                "text": raw_text[:limit],
            }
        )
    return payload


def _collect_working_memory_context(
    query: str,
    *,
    store: "WorkingMemoryStore",
    mode: str,
    min_rerank_score: float,
    min_rerank_margin: float,
    recent_window_rounds: int,
    recent_top_k_rounds: int,
    older_top_k_rounds: int,
    recent_exclude_last_turns: int,
    parent_context_chars: int,
) -> tuple[dict[str, object] | None, str, str]:
    resolved_recent_exclude_last_turns = max(0, int(recent_exclude_last_turns))
    turn_ceiling = max(0, int(store.turn_count) - resolved_recent_exclude_last_turns)
    recent_floor = max(1, turn_ceiling - max(1, int(recent_window_rounds)) + 1) if turn_ceiling > 0 else 1

    recent_retrieval = store.retrieve(
        query,
        top_k_rounds=recent_top_k_rounds,
        turn_floor=recent_floor,
        turn_ceiling=turn_ceiling,
    )
    older_retrieval = store.retrieve(
        query,
        top_k_rounds=older_top_k_rounds,
        turn_floor=1,
        turn_ceiling=max(0, recent_floor - 1),
    )

    if mode == "off":
        use_recent = False
        recent_decision = "memory mode off"
        use_all_time = False
        all_time_decision = "memory mode off"
    else:
        use_recent = bool(recent_retrieval.parent_hits)
        recent_decision = (
            f"retrieved {len(recent_retrieval.parent_hits)} rounds"
            if use_recent
            else "no recent rounds retrieved"
        )
        use_all_time, all_time_decision = _should_apply_working_memory_context(
            query,
            older_retrieval,
            memory_mode=mode,
            min_rerank_score=min_rerank_score,
            min_rerank_margin=min_rerank_margin,
        )

    context_payload: dict[str, object] | None = None
    if use_recent or use_all_time:
        context_payload = {
            "recent_long_term_rounds": _serialize_memory_rounds(
                recent_retrieval.parent_hits if use_recent else [],
                max_chars=parent_context_chars,
            ),
            "older_long_term_rounds": _serialize_memory_rounds(
                older_retrieval.parent_hits if use_all_time else [],
                max_chars=parent_context_chars,
            ),
            "metadata": {
                "recent_window_rounds": int(recent_window_rounds),
                "exclude_latest_turns": resolved_recent_exclude_last_turns,
                "recent_decision": recent_decision,
                "older_decision": all_time_decision,
                "turn_count": int(store.turn_count),
            },
        }

    status_line = (
        "[Memory] recent "
        f"{'using' if use_recent else 'skipped'} "
        f"({recent_decision}); older "
        f"{'using' if use_all_time else 'skipped'} "
        f"({all_time_decision})"
    )

    summary = (
        f"- Recent Applied: {'yes' if use_recent else 'no'}\n"
        f"- Recent Decision: {recent_decision}\n"
        f"- Recent Parent hits: {len(recent_retrieval.parent_hits)}\n"
        f"- Recent Window rounds: {max(1, int(recent_window_rounds))}\n"
        f"- Recent Excluded latest turns: {resolved_recent_exclude_last_turns}\n"
        f"- Older Applied: {'yes' if use_all_time else 'no'}\n"
        f"- Older Decision: {all_time_decision}\n"
        f"- Older Parent hits: {len(older_retrieval.parent_hits)}\n"
        f"- Older Excluded latest turns: {resolved_recent_exclude_last_turns}\n"
        f"- Turns in memory: {store.turn_count}"
    )
    return context_payload, summary, status_line


def _collect_rag_context(
    query: str,
    *,
    store: "HierarchicalRagStore",
    mode: str,
    min_child_score: float,
    min_score_margin: float,
    top_k: int,
    parent_context_chars: int,
) -> tuple[dict[str, object] | None, str, str]:
    retrieval = store.retrieve(
        query,
        top_k=top_k,
    )
    use_rag, rag_decision = _should_apply_rag_context(
        query,
        retrieval,
        rag_mode=mode,
        min_child_score=min_child_score,
        min_score_margin=min_score_margin,
    )

    context_payload: dict[str, object] | None = None
    if use_rag:
        context_payload = {
            "sources": _serialize_rag_sources(retrieval.source_hits),
            "parent_sections": _serialize_rag_parents(
                retrieval.parent_hits,
                max_chars=parent_context_chars,
            ),
            "metadata": {
                "decision": rag_decision,
                "top_k": int(top_k),
                "parent_context_chars": int(parent_context_chars),
            },
        }
        status_line = (
            f"[RAG] using {len(retrieval.source_hits)} source docs, "
            f"{len(retrieval.parent_hits)} parent sections ({rag_decision})"
        )
    else:
        status_line = f"[RAG] skipped ({rag_decision})"

    source_labels: list[str] = []
    for hit in retrieval.source_hits:
        label = Path(hit.source_path).name if hit.source_path else (hit.source_id or "")
        if not label or label in source_labels:
            continue
        source_labels.append(label)
        if len(source_labels) >= 5:
            break

    summary = (
        f"- Applied: {'yes' if use_rag else 'no'}\n"
        f"- Decision: {rag_decision}\n"
        f"- Source hits: {len(retrieval.source_hits)}\n"
        f"- Parent hits: {len(retrieval.parent_hits)}\n"
        f"- Sources: {', '.join(source_labels) if source_labels else 'n/a'}"
    )
    return context_payload, summary, status_line


def _working_memory_collect_kwargs(
    options: ChatRuntimeOptions,
    *,
    recent_exclude_last_turns: int,
) -> dict[str, int | float | str | bool]:
    working = options.working_memory
    return {
        "mode": working.mode,
        "min_rerank_score": working.min_rerank_score,
        "min_rerank_margin": working.min_rerank_margin,
        "recent_window_rounds": working.recent_window_rounds,
        "recent_top_k_rounds": working.recent_top_k_rounds,
        "older_top_k_rounds": working.older_top_k_rounds,
        "recent_exclude_last_turns": recent_exclude_last_turns,
        "parent_context_chars": working.parent_context_chars,
    }


def _rag_collect_kwargs(options: ChatRuntimeOptions) -> dict[str, int | float | str]:
    rag = options.rag
    return {
        "mode": rag.mode,
        "min_child_score": rag.min_child_score,
        "min_score_margin": rag.min_score_margin,
        "top_k": rag.top_k,
        "parent_context_chars": rag.parent_context_chars,
    }


def collect_context_for_query(
    query: str,
    *,
    rag_store: "HierarchicalRagStore | None",
    working_memory_store: "WorkingMemoryStore | None",
    options: ChatRuntimeOptions,
    always_context_rounds: list[dict[str, object]] | None = None,
    recent_memory_exclude_last_turns: int | None = None,
) -> tuple[str | None, str | None, str | None, list[str]]:
    status_lines: list[str] = []
    rag_summary: str | None = None
    memory_summary: str | None = None
    memory_context_payload: dict[str, object] | None = None
    rag_context_payload: dict[str, object] | None = None

    resolved_recent_exclude_last_turns = (
        options.working_memory.exclude_latest_turns
        if recent_memory_exclude_last_turns is None
        else int(recent_memory_exclude_last_turns)
    )
    working_memory_kwargs = _working_memory_collect_kwargs(
        options,
        recent_exclude_last_turns=resolved_recent_exclude_last_turns,
    )
    rag_kwargs = _rag_collect_kwargs(options)

    if working_memory_store is not None and rag_store is not None:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="context-retrieval") as pool:
            memory_future: Future[tuple[dict[str, object] | None, str, str]] = pool.submit(
                _collect_working_memory_context,
                query,
                store=working_memory_store,
                **working_memory_kwargs,
            )
            rag_future: Future[tuple[dict[str, object] | None, str, str]] = pool.submit(
                _collect_rag_context,
                query,
                store=rag_store,
                **rag_kwargs,
            )

            memory_context_payload, memory_summary, memory_status_line = memory_future.result()
            rag_context_payload, rag_summary, rag_status_line = rag_future.result()
        status_lines.append(memory_status_line)
        status_lines.append(rag_status_line)
    else:
        if working_memory_store is not None:
            memory_context_payload, memory_summary, memory_status_line = _collect_working_memory_context(
                query,
                store=working_memory_store,
                **working_memory_kwargs,
            )
            status_lines.append(memory_status_line)

        if rag_store is not None:
            rag_context_payload, rag_summary, rag_status_line = _collect_rag_context(
                query,
                store=rag_store,
                **rag_kwargs,
            )
            status_lines.append(rag_status_line)

    context_text = _build_tagged_context_envelope(
        last_rounds=always_context_rounds,
        memory_payload=memory_context_payload,
        rag_payload=rag_context_payload,
    )
    return context_text, rag_summary, memory_summary, status_lines
